"""
Advanced reporter for the Agentic Web-App QA automation project.

This module extends the basic reporter with:

1. Screenshot-on-failure embedding (base64 data URIs -> self-contained HTML).
2. Historical trend tracking (reports/history.json) with an inline SVG chart.
3. Client-side filtering (vanilla JS: All / Passed / Failed).
4. A collapsible "Run details" metadata section.
5. A CI-friendly ``has_failures()`` helper for ``sys.exit(1)`` integration.
6. Defensive error handling so a QA run never crashes mid-report.

Design contract (kept backward compatible):

    def generate_report(
        results: List[StepResult],
        output_dir: str = "reports",
        test_name: str = "Agentic Web-App Test",
        filename_stem: Optional[str] = None,
    ) -> tuple[Path, Path]: ...

The signature above is preserved; all new behaviour is exposed through
*additional* optional keyword arguments (``screenshot_dir``, ``metadata``,
``history_size``) so existing callers in ``main.py`` continue to work
unchanged.

Constraints: Python 3.13, stdlib + Pydantic v2 only. No Jinja2, no external
JS/CSS. The produced HTML is a single portable file.
"""

from __future__ import annotations

import base64
import html
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List, Literal, Optional

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Custom exceptions
# --------------------------------------------------------------------------- #
class ReporterError(Exception):
    """Raised when report generation fails in an unrecoverable way.

    This is used to wrap low-level ``OSError`` (disk full, permission
    denied, etc.) that occurs while writing the *primary* report artifacts,
    so callers see a clear, domain-specific error instead of a raw OS
    exception. Non-critical failures (history tracking, screenshot reads)
    are logged and degraded rather than raised.
    """


# --------------------------------------------------------------------------- #
# Data contract (mirrors executor.StepResult without importing it, so this
# module stays importable/testable in isolation for the smoke test)
# --------------------------------------------------------------------------- #
class StepResult(BaseModel):
    """Result of executing a single test step.

    This mirrors the ``StepResult`` defined in ``executor.py``. It is
    re-declared here (rather than imported) only so that this module and its
    ``__main__`` smoke test can run without pulling in the executor / a live
    browser. At runtime, ``generate_report`` accepts any object exposing the
    same attributes, so the executor's own ``StepResult`` works transparently.

    Attributes:
        step_number: 1-based index of the step within the plan.
        action: Machine-readable action name (e.g. ``"click"``, ``"assert"``).
        description: Human-readable description of the step.
        status: Either ``"pass"`` or ``"fail"``.
        message: Detail message (error text on failure, confirmation on pass).
        duration_ms: Wall-clock duration of the step in milliseconds.
    """

    step_number: int
    action: str
    description: str
    status: Literal["pass", "fail"]
    message: str
    duration_ms: float


class HistoryEntry(BaseModel):
    """One summary row persisted to ``reports/history.json`` per run.

    Attributes:
        timestamp: ISO-8601 UTC timestamp of when the run was recorded.
        test_name: Human-readable name of the test suite/run.
        total: Total number of steps executed.
        passed: Number of steps that passed.
        failed: Number of steps that failed.
        pass_rate: Fraction in ``[0.0, 1.0]`` of steps that passed.
        total_duration_ms: Sum of all step durations in milliseconds.
    """

    timestamp: str
    test_name: str
    total: int
    passed: int
    failed: int
    pass_rate: float = Field(ge=0.0, le=1.0)
    total_duration_ms: float


# --------------------------------------------------------------------------- #
# Public helpers
# --------------------------------------------------------------------------- #
def has_failures(results: List[StepResult]) -> bool:
    """Return ``True`` if any step in ``results`` has status ``"fail"``.

    Intended for CI integration, e.g.::

        results = run_plan(page, steps)
        generate_report(results)
        if has_failures(results):
            sys.exit(1)

    Args:
        results: The list of step results from a run.

    Returns:
        ``True`` when at least one step failed, otherwise ``False``.
        An empty list returns ``False`` (nothing failed).
    """
    return any(getattr(r, "status", None) == "fail" for r in results)


# --------------------------------------------------------------------------- #
# Summary computation
# --------------------------------------------------------------------------- #
def _summarize(results: List[StepResult]) -> dict[str, Any]:
    """Compute aggregate statistics for a set of step results.

    Args:
        results: The list of step results from a run.

    Returns:
        A dict with keys ``total``, ``passed``, ``failed``, ``pass_rate``
        (float in ``[0, 1]``), and ``total_duration_ms``.
    """
    total = len(results)
    passed = sum(1 for r in results if getattr(r, "status", None) == "pass")
    failed = total - passed
    pass_rate = (passed / total) if total else 0.0
    total_duration_ms = float(sum(getattr(r, "duration_ms", 0.0) or 0.0 for r in results))
    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": pass_rate,
        "total_duration_ms": total_duration_ms,
    }


# --------------------------------------------------------------------------- #
# History tracking (defensive: never crash the report)
# --------------------------------------------------------------------------- #
def _load_history(history_path: Path) -> List[HistoryEntry]:
    """Load and validate the historical run summaries.

    Handles three non-fatal conditions gracefully by returning an empty list
    (and logging): the file not existing (first run), unreadable file I/O,
    and corrupt / unparseable JSON.

    Args:
        history_path: Path to ``history.json``.

    Returns:
        A list of validated :class:`HistoryEntry` objects (possibly empty).
    """
    if not history_path.exists():
        logger.info("No history file at %s (first run); starting fresh.", history_path)
        return []

    try:
        raw = history_path.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - environment dependent
        logger.warning("Could not read history file %s: %s. Starting fresh.", history_path, exc)
        return []

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            "History file %s is corrupt/unparseable: %s. Starting fresh.", history_path, exc
        )
        return []

    if not isinstance(data, list):
        logger.warning(
            "History file %s did not contain a JSON list (got %s). Starting fresh.",
            history_path,
            type(data).__name__,
        )
        return []

    entries: List[HistoryEntry] = []
    for i, item in enumerate(data):
        try:
            entries.append(HistoryEntry.model_validate(item))
        except ValidationError as exc:
            logger.warning("Skipping malformed history entry #%d in %s: %s", i, history_path, exc)
    return entries


def _append_history(
    history_path: Path,
    entry: HistoryEntry,
    existing: List[HistoryEntry],
) -> List[HistoryEntry]:
    """Append ``entry`` to the history file and return the updated list.

    Writing is best-effort: on any I/O error the in-memory list is still
    returned (so the current report can render its chart) but the failure is
    logged. History tracking must never crash report generation.

    Args:
        history_path: Path to ``history.json``.
        entry: The new summary entry for this run.
        existing: Previously loaded history entries.

    Returns:
        The combined list (``existing + [entry]``) regardless of write success.
    """
    updated = [*existing, entry]
    payload = json.dumps([e.model_dump() for e in updated], indent=2)

    try:
        _ensure_dir(history_path.parent)
        # Atomic-ish write via a temp file, then replace, to avoid corrupting
        # the history file if the process dies mid-write.
        tmp = history_path.with_suffix(history_path.suffix + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(history_path)
    except OSError as exc:
        logger.warning(
            "Could not persist history to %s: %s. Report will still render.",
            history_path,
            exc,
        )
    return updated


def _ensure_dir(path: Path) -> None:
    """Create ``path`` (and parents) if it does not already exist.

    Args:
        path: Directory to create.

    Raises:
        OSError: Propagated to the caller, which decides whether the failure
            is fatal (primary report dir) or degradable (history dir).
    """
    path.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Screenshot embedding (defensive)
# --------------------------------------------------------------------------- #
def _load_screenshot_data_uri(screenshot_dir: Optional[str], step_number: int) -> Optional[str]:
    """Return a base64 ``data:`` URI for a step's failure screenshot, if any.

    Naming convention: ``{screenshot_dir}/step_{step_number}.png``.

    Degrades gracefully: returns ``None`` (never raises) when no directory is
    configured, the file is missing, or the file cannot be read.

    Args:
        screenshot_dir: Directory that may contain per-step screenshots.
        step_number: The 1-based step number to look up.

    Returns:
        A ``data:image/png;base64,...`` string, or ``None`` if unavailable.
    """
    if not screenshot_dir:
        return None

    shot_path = Path(screenshot_dir) / f"step_{step_number}.png"
    if not shot_path.exists():
        logger.debug("No screenshot for failed step %d at %s.", step_number, shot_path)
        return None

    try:
        raw = shot_path.read_bytes()
    except OSError as exc:
        logger.warning("Could not read screenshot %s: %s. Omitting image.", shot_path, exc)
        return None

    try:
        encoded = base64.b64encode(raw).decode("ascii")
    except (ValueError, TypeError) as exc:  # pragma: no cover - defensive
        logger.warning("Could not base64-encode screenshot %s: %s. Omitting.", shot_path, exc)
        return None

    return f"data:image/png;base64,{encoded}"


# --------------------------------------------------------------------------- #
# Inline SVG chart (pure Python, no dependencies)
# --------------------------------------------------------------------------- #
def _render_sparkline_svg(history: List[HistoryEntry], last_n: int = 10) -> str:
    """Render a small inline SVG bar chart of pass_rate over the last N runs.

    The most recent ``last_n`` entries are shown left-to-right. Each bar's
    height encodes the pass rate (0-100%); a full-pass run is coloured green,
    anything with failures is coloured red. Returns a self-contained ``<svg>``
    fragment (no external references).

    Args:
        history: All known history entries (chronological order assumed).
        last_n: Maximum number of most-recent runs to display.

    Returns:
        An HTML/SVG string. If there is no history, a small placeholder
        message is returned instead.
    """
    recent = history[-last_n:] if history else []
    if not recent:
        return '<p class="muted">No historical runs yet — this is the first recorded run.</p>'

    # Geometry.
    width = 420
    height = 120
    pad_left = 34
    pad_bottom = 20
    pad_top = 10
    pad_right = 10
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    n = len(recent)
    gap = 6
    bar_w = max(4.0, (plot_w - gap * (n - 1)) / n)

    parts: List[str] = [
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="Pass rate over last {n} runs" '
        f'xmlns="http://www.w3.org/2000/svg">'
    ]

    # Y-axis gridlines at 0%, 50%, 100%.
    for frac, label in ((0.0, "0%"), (0.5, "50%"), (1.0, "100%")):
        y = pad_top + (1.0 - frac) * plot_h
        parts.append(
            f'<line x1="{pad_left}" y1="{y:.1f}" x2="{width - pad_right}" y2="{y:.1f}" '
            f'stroke="#e0e0e0" stroke-width="1" />'
        )
        parts.append(
            f'<text x="{pad_left - 6}" y="{y + 4:.1f}" font-size="9" '
            f'text-anchor="end" fill="#888">{label}</text>'
        )

    # Bars.
    for i, entry in enumerate(recent):
        rate = max(0.0, min(1.0, entry.pass_rate))
        bar_h = rate * plot_h
        x = pad_left + i * (bar_w + gap)
        y = pad_top + (plot_h - bar_h)
        colour = "#2e7d32" if entry.failed == 0 else "#c62828"
        pct = f"{rate * 100:.0f}%"
        tooltip = html.escape(
            f"{entry.timestamp} - {entry.passed}/{entry.total} passed ({pct})"
        )
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" '
            f'fill="{colour}" rx="2"><title>{tooltip}</title></rect>'
        )
        # Percentage label above each bar.
        parts.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{y - 3:.1f}" font-size="8" '
            f'text-anchor="middle" fill="#555">{pct}</text>'
        )

    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------------- #
# HTML rendering helpers
# --------------------------------------------------------------------------- #
def _e(value: Any) -> str:
    """HTML-escape an arbitrary value (stringified first).

    Args:
        value: Any value to render as safe HTML text.

    Returns:
        An HTML-escaped string.
    """
    return html.escape(str(value), quote=True)


def _render_metadata_section(metadata: Optional[dict]) -> str:
    """Render a collapsible "Run details" section from a metadata dict.

    Args:
        metadata: Arbitrary key/value run details (browser, viewport, base
            URL, git commit, etc.), or ``None``.

    Returns:
        An HTML ``<details>`` fragment, or an empty string if no metadata.
    """
    if not metadata:
        return ""

    rows = "".join(
        f"<tr><th>{_e(key)}</th><td>{_e(val)}</td></tr>"
        for key, val in metadata.items()
    )
    return (
        '<details class="run-details">'
        "<summary>Run details</summary>"
        f'<table class="meta-table"><tbody>{rows}</tbody></table>'
        "</details>"
    )


def _render_step_rows(results: List[StepResult], screenshot_dir: Optional[str]) -> str:
    """Render the ``<tr>`` rows of the step results table.

    Failed steps get a ``data-status="fail"`` attribute (used by the JS
    filter) and, when available, an embedded base64 screenshot row.

    Args:
        results: The step results to render.
        screenshot_dir: Optional directory of per-step failure screenshots.

    Returns:
        Concatenated HTML table-row markup.
    """
    rows: List[str] = []
    for r in results:
        status = getattr(r, "status", "fail")
        status_class = "pass" if status == "pass" else "fail"
        badge = "PASS" if status == "pass" else "FAIL"

        rows.append(
            f'<tr class="step-row {status_class}" data-status="{_e(status)}">'
            f"<td>{_e(getattr(r, 'step_number', ''))}</td>"
            f"<td>{_e(getattr(r, 'action', ''))}</td>"
            f"<td>{_e(getattr(r, 'description', ''))}</td>"
            f'<td><span class="badge {status_class}">{badge}</span></td>'
            f"<td>{_e(getattr(r, 'message', ''))}</td>"
            f"<td>{_e(format(getattr(r, 'duration_ms', 0.0), '.1f'))}</td>"
            "</tr>"
        )

        if status == "fail":
            step_num = getattr(r, "step_number", -1)
            data_uri = _load_screenshot_data_uri(screenshot_dir, step_num)
            if data_uri:
                rows.append(
                    f'<tr class="shot-row" data-status="fail">'
                    f'<td colspan="6">'
                    f'<details><summary>Screenshot (step '
                    f"{_e(step_num)})</summary>"
                    f'<img class="shot" src="{data_uri}" '
                    f'alt="Failure screenshot for step {_e(step_num)}" />'
                    f"</details></td></tr>"
                )
    return "".join(rows)


def _render_html(
    results: List[StepResult],
    test_name: str,
    summary: dict[str, Any],
    history: List[HistoryEntry],
    metadata: Optional[dict],
    screenshot_dir: Optional[str],
    history_size: int,
) -> str:
    """Assemble the full, self-contained HTML report as a string.

    Args:
        results: The step results.
        test_name: Human-readable name of the run.
        summary: Aggregate stats from :func:`_summarize`.
        history: History entries (including the current run).
        metadata: Optional run-details dict.
        screenshot_dir: Optional screenshot directory.
        history_size: Number of recent runs to chart.

    Returns:
        A complete HTML document as a single string.
    """
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    pass_pct = f"{summary['pass_rate'] * 100:.1f}%"
    overall_class = "pass" if summary["failed"] == 0 else "fail"

    step_rows = _render_step_rows(results, screenshot_dir)
    chart_svg = _render_sparkline_svg(history, last_n=history_size)
    meta_section = _render_metadata_section(metadata)

    # NOTE: CSS/JS braces are doubled so they survive the f-string.
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{_e(test_name)} — QA Report</title>
<style>
  :root {{ --pass: #2e7d32; --fail: #c62828; --ink: #222; --muted: #888; }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
         color: var(--ink); margin: 0; padding: 24px; background: #fafafa; }}
  h1 {{ font-size: 1.4rem; margin: 0 0 4px; }}
  .muted {{ color: var(--muted); font-size: 0.85rem; }}
  .card {{ background: #fff; border: 1px solid #eee; border-radius: 8px;
          padding: 16px; margin-bottom: 16px; box-shadow: 0 1px 2px rgba(0,0,0,.04); }}
  .summary-grid {{ display: flex; flex-wrap: wrap; gap: 16px; }}
  .stat {{ min-width: 90px; }}
  .stat .num {{ font-size: 1.6rem; font-weight: 700; }}
  .stat .lbl {{ font-size: 0.75rem; color: var(--muted); text-transform: uppercase; }}
  .stat.pass .num {{ color: var(--pass); }}
  .stat.fail .num {{ color: var(--fail); }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid #eee;
           font-size: 0.9rem; vertical-align: top; }}
  th {{ background: #f5f5f5; font-size: 0.75rem; text-transform: uppercase; color: #555; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px;
           font-size: 0.72rem; font-weight: 700; color: #fff; }}
  .badge.pass {{ background: var(--pass); }}
  .badge.fail {{ background: var(--fail); }}
  tr.fail td {{ background: #fff6f6; }}
  .shot {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; margin-top: 8px; }}
  .filter-bar {{ margin-bottom: 12px; }}
  .filter-bar button {{ border: 1px solid #ccc; background: #fff; padding: 6px 14px;
                       border-radius: 6px; cursor: pointer; font-size: 0.85rem; margin-right: 6px; }}
  .filter-bar button.active {{ background: var(--ink); color: #fff; border-color: var(--ink); }}
  .meta-table th {{ width: 180px; text-transform: none; background: #fafafa; }}
  details.run-details summary, .shot-row summary {{ cursor: pointer; font-weight: 600; }}
  .overall.pass {{ color: var(--pass); }} .overall.fail {{ color: var(--fail); }}
</style>
</head>
<body>
  <header class="card">
    <h1>{_e(test_name)}</h1>
    <div class="muted">Generated {_e(generated_at)} —
      <span class="overall {overall_class}">
        {"ALL PASSED" if summary['failed'] == 0 else "FAILURES DETECTED"}</span>
    </div>
  </header>

  <section class="card">
    <div class="summary-grid">
      <div class="stat"><div class="num">{summary['total']}</div><div class="lbl">Total</div></div>
      <div class="stat pass"><div class="num">{summary['passed']}</div><div class="lbl">Passed</div></div>
      <div class="stat fail"><div class="num">{summary['failed']}</div><div class="lbl">Failed</div></div>
      <div class="stat"><div class="num">{pass_pct}</div><div class="lbl">Pass rate</div></div>
      <div class="stat"><div class="num">{summary['total_duration_ms']:.0f}</div><div class="lbl">Total ms</div></div>
    </div>
  </section>

  <section class="card">
    <h2 style="font-size:1rem;margin-top:0;">Pass-rate trend (last {history_size} runs)</h2>
    {chart_svg}
  </section>

  {f'<section class="card">{meta_section}</section>' if meta_section else ''}

  <section class="card">
    <div class="filter-bar">
      <button type="button" class="active" data-filter="all">All</button>
      <button type="button" data-filter="pass">Passed</button>
      <button type="button" data-filter="fail">Failed</button>
    </div>
    <table id="results">
      <thead>
        <tr><th>#</th><th>Action</th><th>Description</th><th>Status</th><th>Message</th><th>ms</th></tr>
      </thead>
      <tbody>
        {step_rows}
      </tbody>
    </table>
  </section>

  <footer class="muted">Agentic Web-App QA — self-contained report. Safe to email or attach.</footer>

<script>
(function () {{
  "use strict";
  var buttons = document.querySelectorAll(".filter-bar button");
  var rows = document.querySelectorAll("#results tbody tr");

  function applyFilter(mode) {{
    rows.forEach(function (row) {{
      var status = row.getAttribute("data-status");
      var show = (mode === "all") || (status === mode);
      row.style.display = show ? "" : "none";
    }});
  }}

  buttons.forEach(function (btn) {{
    btn.addEventListener("click", function () {{
      buttons.forEach(function (b) {{ b.classList.remove("active"); }});
      btn.classList.add("active");
      applyFilter(btn.getAttribute("data-filter"));
    }});
  }});
}})();
</script>
</body>
</html>"""


# --------------------------------------------------------------------------- #
# Primary entry point
# --------------------------------------------------------------------------- #
def generate_report(
    results: List[StepResult],
    output_dir: str = "reports",
    test_name: str = "Agentic Web-App Test",
    filename_stem: Optional[str] = None,
    *,
    screenshot_dir: Optional[str] = None,
    metadata: Optional[dict] = None,
    history_size: int = 10,
) -> tuple[Path, Path]:
    """Generate a self-contained HTML report (and JSON sidecar) for a run.

    Backward compatible with the basic reporter: the first four positional
    parameters and the ``(html_path, json_path)`` return type are unchanged.
    New behaviour is opt-in via keyword-only arguments.

    Behaviour:
        * Writes ``<stem>.html`` (portable, screenshots embedded as base64)
          and ``<stem>.json`` (machine-readable results) to ``output_dir``.
        * Appends a summary row to ``<output_dir>/history.json`` and renders
          an inline SVG trend chart of the last ``history_size`` runs.
        * Embeds a failure screenshot for any failed step that has a matching
          ``{screenshot_dir}/step_{n}.png`` file (missing ones are skipped).
        * Renders ``metadata`` in a collapsible "Run details" section.

    All non-critical I/O (history, screenshots) degrades gracefully with a
    logged warning. Only failure to write the primary HTML/JSON artifacts is
    fatal, and that is surfaced as :class:`ReporterError`.

    Args:
        results: Step results from ``run_plan``.
        output_dir: Directory to write artifacts into (created if needed).
        test_name: Human-readable name shown in the report.
        filename_stem: Base filename (without extension). Defaults to a
            timestamped stem when omitted.
        screenshot_dir: Optional directory containing per-step screenshots.
        metadata: Optional run-details dict (browser, viewport, commit, ...).
        history_size: Number of recent runs to include in the trend chart.

    Returns:
        A tuple ``(html_path, json_path)`` of the written report files.

    Raises:
        ReporterError: If the output directory cannot be created or the
            primary HTML/JSON files cannot be written.
    """
    out_dir = Path(output_dir)

    # Fatal if we cannot create the primary output directory.
    try:
        _ensure_dir(out_dir)
    except OSError as exc:
        raise ReporterError(f"Could not create output directory {out_dir!r}: {exc}") from exc

    stem = filename_stem or f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    html_path = out_dir / f"{stem}.html"
    json_path = out_dir / f"{stem}.json"

    summary = _summarize(results)

    # --- History (non-fatal) ------------------------------------------------
    history_path = out_dir / "history.json"
    try:
        existing_history = _load_history(history_path)
    except Exception as exc:  # pragma: no cover - _load_history is defensive
        logger.warning("Unexpected error loading history: %s. Starting fresh.", exc)
        existing_history = []

    current_entry = HistoryEntry(
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        test_name=test_name,
        total=summary["total"],
        passed=summary["passed"],
        failed=summary["failed"],
        pass_rate=summary["pass_rate"],
        total_duration_ms=summary["total_duration_ms"],
    )
    full_history = _append_history(history_path, current_entry, existing_history)

    # --- Render + write primary artifacts (fatal on failure) ---------------
    html_doc = _render_html(
        results=results,
        test_name=test_name,
        summary=summary,
        history=full_history,
        metadata=metadata,
        screenshot_dir=screenshot_dir,
        history_size=history_size,
    )

    json_payload = {
        "test_name": test_name,
        "generated_at": current_entry.timestamp,
        "summary": summary,
        "steps": [
            r.model_dump() if isinstance(r, BaseModel)
            else {
                "step_number": getattr(r, "step_number", None),
                "action": getattr(r, "action", None),
                "description": getattr(r, "description", None),
                "status": getattr(r, "status", None),
                "message": getattr(r, "message", None),
                "duration_ms": getattr(r, "duration_ms", None),
            }
            for r in results
        ],
        "metadata": metadata or {},
    }

    try:
        html_path.write_text(html_doc, encoding="utf-8")
    except OSError as exc:
        raise ReporterError(f"Failed to write HTML report to {html_path!r}: {exc}") from exc

    try:
        json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
    except OSError as exc:
        raise ReporterError(f"Failed to write JSON report to {json_path!r}: {exc}") from exc

    logger.info("Report written: %s (%s)", html_path, "PASS" if summary["failed"] == 0 else "FAIL")
    return html_path, json_path


# --------------------------------------------------------------------------- #
# Smoke test — runs without a live browser / LLM.
# --------------------------------------------------------------------------- #
def _smoke_test() -> None:
    """Exercise the reporter end-to-end with fake data in a temp directory.

    Creates fake ``StepResult`` objects, a fake screenshot for a failed step,
    a pre-seeded (and a deliberately corrupt) history file, then generates a
    report and asserts the artifacts exist. Intended for quick sanity checks::

        python reporter.py
    """
    import tempfile

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    fake_results: List[StepResult] = [
        StepResult(step_number=1, action="navigate", description="Open home page",
                   status="pass", message="Loaded in 320ms", duration_ms=320.4),
        StepResult(step_number=2, action="click", description="Click login button",
                   status="pass", message="Element clicked", duration_ms=88.1),
        StepResult(step_number=3, action="assert", description="Dashboard is visible",
                   status="fail", message="Timeout: selector '#dash' not found", duration_ms=5010.7),
        StepResult(step_number=4, action="fill", description="Enter search term",
                   status="pass", message="Typed 'pytest'", duration_ms=42.0),
    ]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        out_dir = tmp_path / "reports"
        shot_dir = tmp_path / "screenshots"
        out_dir.mkdir(parents=True, exist_ok=True)
        shot_dir.mkdir(parents=True, exist_ok=True)

        # A minimal valid 1x1 PNG, base64-decoded, as the "screenshot" for
        # failed step 3. (No external file / browser needed.)
        png_1x1 = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42m" 
            "Nk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )
        (shot_dir / "step_3.png").write_bytes(png_1x1)

        # Pre-seed history with a couple of valid entries...
        seeded = [
            {"timestamp": "2026-07-01T10:00:00+00:00", "test_name": "Demo",
             "total": 4, "passed": 4, "failed": 0, "pass_rate": 1.0, "total_duration_ms": 500.0},
            {"timestamp": "2026-07-02T10:00:00+00:00", "test_name": "Demo",
             "total": 4, "passed": 2, "failed": 2, "pass_rate": 0.5, "total_duration_ms": 900.0},
        ]
        (out_dir / "history.json").write_text(json.dumps(seeded), encoding="utf-8")

        metadata = {
            "browser": "chromium",
            "viewport": "1280x720",
            "base_url": "https://example.test",
            "git_commit": "a1b2c3d",
        }

        html_path, json_path = generate_report(
            fake_results,
            output_dir=str(out_dir),
            test_name="Smoke Test Run",
            filename_stem="smoke",
            screenshot_dir=str(shot_dir),
            metadata=metadata,
            history_size=10,
        )

        assert html_path.exists(), "HTML report was not written"
        assert json_path.exists(), "JSON report was not written"
        html_text = html_path.read_text(encoding="utf-8")
        assert "data:image/png;base64," in html_text, "Screenshot was not embedded"
        assert "<svg" in html_text, "Trend chart SVG missing"
        assert "Run details" in html_text, "Metadata section missing"
        assert 'data-filter="fail"' in html_text, "Filter bar missing"

        # has_failures helper.
        assert has_failures(fake_results) is True
        assert has_failures([r for r in fake_results if r.status == "pass"]) is False

        # Corrupt-history resilience: overwrite and regenerate; must not crash.
        (out_dir / "history.json").write_text("{ this is not valid json ", encoding="utf-8")
        html_path2, _ = generate_report(
            fake_results, output_dir=str(out_dir), test_name="Corrupt History Run",
            filename_stem="smoke2",
        )
        assert html_path2.exists(), "Report failed to generate after corrupt history"

        print("smoke test OK")
        print(f"  html: {html_path}")
        print(f"  json: {json_path}")
        print(f"  history entries now: {len(_load_history(out_dir / 'history.json'))}")


if __name__ == "__main__":
    _smoke_test()