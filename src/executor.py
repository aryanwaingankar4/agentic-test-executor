"""
executor.py
-----------
Executes validated TestStep objects against a real browser using
Playwright's synchronous API.
"""

from __future__ import annotations
import os
import logging
import time
from pathlib import Path
from typing import Callable, List, Literal, Optional

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Page
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from pydantic import BaseModel
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from parser import TestStep


logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_MS: float = 5000


class ExecutorError(Exception):
    """Base class for all errors raised by the executor module."""


class ElementNotFoundError(ExecutorError):
    """Raised when no locator strategy can find the requested element."""


class NavigationError(ExecutorError):
    """Raised when navigating to a URL fails."""


class StepResult(BaseModel):
    step_number: int
    action: str
    description: str
    status: Literal["pass", "fail"]
    message: str
    duration_ms: float


import re

_FILLER_WORDS = re.compile(
    r"\b(field|button|box|link|the|input|dropdown|menu|icon)\b", re.IGNORECASE
)


def _prefer_actionable(locator):
    """
    Given a locator that may resolve to multiple elements, prefer the first
    element that is both visible and enabled. Falls back to ``.first`` when
    no such element exists (or when introspection fails), mirroring the
    module's defensive, never-crash philosophy.
    """
    try:
        count = locator.count()
    except Exception:
        return locator.first

    for index in range(count):
        candidate = locator.nth(index)
        try:
            if candidate.is_visible() and candidate.is_enabled():
                return candidate
        except Exception:
            continue

    return locator.first


# --------------------------------------------------------------------------- #
# Self-healing selector support (retry-aware Gemini call + fallback)
# --------------------------------------------------------------------------- #
_TRANSIENT_MARKERS = ("429", "500", "502", "503", "504", "timeout", "deadline")


def _is_transient_selector_error(exc: BaseException) -> bool:
    """Heuristic: does this exception look like a temporary server hiccup?"""
    message = str(exc).lower()
    return any(marker in message for marker in _TRANSIENT_MARKERS)


@retry(
    retry=retry_if_exception(_is_transient_selector_error),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
def _call_gemini_for_selector(prompt: str) -> str:
    """
    Perform the actual Gemini API call for selector generation, retrying on
    transient-looking failures (503, timeouts, etc.) with exponential
    backoff. Non-transient errors (e.g. missing/invalid API key) propagate
    immediately via ``reraise=True`` and are handled by the caller's
    outer safety net. Requires GEMINI_API_KEY to be set in the environment.
    """
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            http_options=types.HttpOptions(timeout=15000),
        ),
    )
    return (response.text or "").strip()


def _find_element_via_llm(page, description: str) -> Optional[str]:
    """
    Last-resort, LLM-assisted selector generation for a described element.

    When every static/heuristic locator strategy in ``find_element`` has
    already failed, this helper asks Gemini to propose a single CSS selector
    that best matches the human-readable ``description`` within the current
    page's HTML. It is intentionally a "never raises" helper, mirroring the
    defensive philosophy used everywhere else in this project (parser.py's
    broad exception handling, ``_capture_failure_screenshot`` here): a
    self-healing selector is a best-effort enhancement, and any failure
    (missing API key, network/timeout error, empty or garbage response,
    invalid selector) must simply yield ``None`` so ``find_element`` falls
    through to its remaining static strategies (and, ultimately, its
    existing ``ElementNotFoundError``).

    Transient Gemini errors (503, timeouts, etc.) are retried up to 3 times
    with exponential backoff via ``_call_gemini_for_selector`` before this
    function gives up and returns ``None``.

    Args:
        page: The active Playwright Page object to read HTML from.
        description: The original human-readable target description (e.g.
            "the login button", "username field").

    Returns:
        A cleaned CSS selector string suggested by the LLM, or ``None`` if
        anything went wrong or no usable selector could be produced.
    """
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.warning(
                "LLM-assisted selector skipped for %r: GEMINI_API_KEY is not set",
                description,
            )
            return None

        html = page.content()[:15000]

        prompt = (
            "You are a web automation expert. Given the HTML of a page and a "
            "human-readable description of an element, respond with ONLY a "
            "single CSS selector string that best matches the described "
            "element. No prose, no explanation, no markdown fences.\n\n"
            f"Description: {description}\n\n"
            f"HTML:\n{html}"
        )

        raw = _call_gemini_for_selector(prompt)
        if not raw:
            logger.warning(
                "LLM-assisted selector returned an empty response for %r", description
            )
            return None

        # Defensively strip markdown code fences and whitespace, the same way
        # parser.py's _extract_json_array does, in case Gemini ignores the
        # "only a selector" instruction.
        cleaned = raw
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned)
        cleaned = cleaned.strip().strip("`").strip()

        # Guard against multi-line prose slipping through: take the first
        # non-empty line only.
        for line in cleaned.splitlines():
            candidate = line.strip()
            if candidate:
                cleaned = candidate
                break

        if not cleaned:
            logger.warning(
                "LLM-assisted selector was empty after cleaning for %r", description
            )
            return None

        logger.debug(
            "LLM-assisted selector for %r resolved to CSS selector: %s",
            description, cleaned,
        )
        return cleaned
    except Exception as exc:  # noqa: BLE001
        # Broad catch-all is deliberate, matching this file's established
        # philosophy (see _capture_failure_screenshot): missing API key,
        # network error, timeout, malformed response, or an unusable
        # selector must all degrade to "no match" rather than crash the run
        # or mask the real reason a step ultimately failed. This still wraps
        # the retrying _call_gemini_for_selector helper, so any exhausted
        # retry or genuinely permanent/unexpected error also degrades to None.
        logger.warning(
            "LLM-assisted selector generation failed for %r: %s", description, exc
        )
        return None


def find_element(page, description: str):
    keyword = _FILLER_WORDS.sub("", description).strip()
    if not keyword:
        keyword = description

    pattern = re.compile(re.escape(keyword), re.IGNORECASE)

    # NEW: data-testid / data-test placed early — the standard hook for
    # testable web apps — ahead of generic id/name matching.
    #
    # NOTE on ordering: "llm-assisted" is deliberately placed BEFORE
    # "submit-fallback". submit-fallback matches ANY submit-type button on
    # the page unconditionally, regardless of whether the description has
    # anything to do with submitting — so if it were listed before the LLM
    # strategy, it would shadow the LLM strategy on virtually every page
    # that has a submit button, and the self-healing feature would never
    # actually run. Keeping submit-fallback last preserves it as a true
    # last-resort behind the LLM-assisted strategy.
    strategies = [
        ("data-testid", lambda: page.locator(f"[data-testid*='{keyword}' i]")),
        ("data-test", lambda: page.locator(f"[data-test*='{keyword}' i]")),
        ("placeholder", lambda: page.get_by_placeholder(pattern)),
        ("label", lambda: page.get_by_label(pattern)),
        ("role=button", lambda: page.get_by_role("button", name=pattern)),
        ("role=link", lambda: page.get_by_role("link", name=pattern)),
        ("text", lambda: page.get_by_text(pattern)),
        ("id-exact", lambda: page.locator(f"#{keyword}")),
        ("name-contains", lambda: page.locator(f"[name*='{keyword}' i]")),
        ("id-contains", lambda: page.locator(f"[id*='{keyword}' i]")),
        ("aria-label-contains", lambda: page.locator(f"[aria-label*='{keyword}' i]")),
        ("llm-assisted", lambda: page.locator(_find_element_via_llm(page, description) or ":root:not(*)")),
        ("submit-fallback", lambda: page.locator("button[type=submit], input[type=submit]")),
    ]

    for name, strategy in strategies:
        try:
            locator = strategy()
            if locator.count() > 0:
                logger.debug(
                    "find_element matched via strategy '%s' for description %r (keyword %r)",
                    name, description, keyword,
                )
                return _prefer_actionable(locator)
        except Exception:
            continue

    raise ElementNotFoundError(f"No locator strategy could find an element for: '{description}'")


def _do_goto(page: Page, step: TestStep) -> str:
    if not step.url:
        raise NavigationError("goto step is missing a 'url' value")
    try:
        page.goto(step.url, timeout=DEFAULT_TIMEOUT_MS)
    except Exception as exc:
        raise NavigationError(f"Failed to navigate to '{step.url}': {exc}") from exc
    return f"Navigated to {step.url}"


def _do_type(page: Page, step: TestStep) -> str:
    if not step.target:
        raise ExecutorError("type step is missing a 'target'")
    if step.value is None:
        raise ExecutorError("type step is missing a 'value'")
    locator = find_element(page, step.target)
    locator.fill(step.value, timeout=DEFAULT_TIMEOUT_MS)
    return f"Typed '{step.value}' into '{step.target}'"


def _do_click(page: Page, step: TestStep) -> str:
    if not step.target:
        raise ExecutorError("click step is missing a 'target'")
    locator = find_element(page, step.target)
    locator.click(timeout=DEFAULT_TIMEOUT_MS)
    return f"Clicked '{step.target}'"


def _do_verify(page: Page, step: TestStep) -> str:
    if step.value is None:
        raise ExecutorError("verify step is missing a 'value' to check for")
    content = page.content()
    if step.value not in content:
        raise ExecutorError(f"Expected text '{step.value}' not found on page")
    return f"Verified text '{step.value}' is present"


def _do_select(page: Page, step: TestStep) -> str:
    if not step.target:
        raise ExecutorError("select step is missing a 'target'")
    if step.value is None:
        raise ExecutorError("select step is missing a 'value'")
    locator = find_element(page, step.target)
    locator.select_option(label=step.value, timeout=DEFAULT_TIMEOUT_MS)
    return f"Selected '{step.value}' in '{step.target}'"


def _do_hover(page: Page, step: TestStep) -> str:
    if not step.target:
        raise ExecutorError("hover step is missing a 'target'")
    locator = find_element(page, step.target)
    locator.hover(timeout=DEFAULT_TIMEOUT_MS)
    return f"Hovered over '{step.target}'"


def _do_press(page: Page, step: TestStep) -> str:
    if step.value is None:
        raise ExecutorError("press step is missing a 'value' (the key to press)")
    if step.target:
        locator = find_element(page, step.target)
        locator.press(step.value, timeout=DEFAULT_TIMEOUT_MS)
        return f"Pressed '{step.value}' on '{step.target}'"
    page.keyboard.press(step.value)
    return f"Pressed '{step.value}' on the current focus"


def _do_wait(page: Page, step: TestStep) -> str:
    if step.value is None:
        raise ExecutorError("wait step is missing a 'value' (ms or a selector/description)")

    # Numeric value -> fixed wait in milliseconds.
    stripped = step.value.strip()
    try:
        milliseconds = float(stripped)
    except ValueError:
        milliseconds = None

    if milliseconds is not None:
        page.wait_for_timeout(milliseconds)
        return f"Waited {stripped} ms"

    # Otherwise treat the value as a selector/description to wait for.
    locator = find_element(page, stripped)
    locator.wait_for(state="visible", timeout=DEFAULT_TIMEOUT_MS)
    return f"Waited for '{stripped}' to appear"


def _do_screenshot(
    page: Page,
    step: TestStep,
    step_number: int = 1,
    screenshot_dir: Optional[str] = None,
) -> str:
    """
    Manually capture a screenshot regardless of pass/fail state.

    Reuses the reporter's ``step_{n}.png`` naming convention so the report
    can embed it. When no ``screenshot_dir`` is configured this degrades to
    a no-op success rather than failing the step, consistent with the
    project's defensive philosophy.
    """
    if not screenshot_dir:
        logger.info("screenshot step %d requested but no screenshot_dir configured", step_number)
        return "Screenshot skipped (no screenshot_dir configured)"

    directory = Path(screenshot_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"step_{step_number}.png"
    page.screenshot(path=str(path))
    return f"Captured screenshot to {path}"


_ACTION_HANDLERS: dict[str, Callable[..., str]] = {
    "goto": _do_goto,
    "type": _do_type,
    "click": _do_click,
    "verify": _do_verify,
    "select": _do_select,
    "hover": _do_hover,
    "press": _do_press,
    "wait": _do_wait,
    "screenshot": _do_screenshot,
}


# --------------------------------------------------------------------------- #
# NEW: failure screenshot capture
# --------------------------------------------------------------------------- #
def _capture_failure_screenshot(page: Page, screenshot_dir: str, step_number: int) -> None:
    """
    Best-effort screenshot capture for a failed step.

    This is intentionally a "never raises" helper, mirroring the same
    defensive philosophy used everywhere else in this project (parser.py's
    broad exception handling, executor.py's run_step safety net): a
    screenshot is a nice-to-have for the report, and a failure to capture
    one (page already closed, disk full, permissions issue) must never mask
    the *original* step failure or crash the run.

    Args:
        page: The active Playwright Page object at the moment of failure.
        screenshot_dir: Directory to save the screenshot into (created if
            it doesn't exist).
        step_number: 1-based step number, used to build the filename so the
            reporter's naming convention (`step_{n}.png`) can find it.
    """
    try:
        directory = Path(screenshot_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"step_{step_number}.png"
        page.screenshot(path=str(path))
        logger.info("Saved failure screenshot for step %d to %s", step_number, path)
    except Exception as exc:  # noqa: BLE001
        # Broad catch-all is deliberate: ANY screenshot failure (Playwright
        # error, OSError, permission error, closed page, etc.) must degrade
        # silently rather than propagate and hide the real step failure.
        logger.warning("Could not capture screenshot for step %d: %s", step_number, exc)


def run_step(
    page: Page,
    step: TestStep,
    step_number: int = 1,
    screenshot_dir: Optional[str] = None,
) -> StepResult:
    """
    Execute exactly one TestStep and return a structured StepResult.

    Args:
        page: The active Playwright Page object.
        step: A validated TestStep to execute.
        step_number: 1-based index of this step, used in the report.
        screenshot_dir: Optional directory to save a screenshot into when
            this step fails. If omitted, no screenshot is captured. It is
            also passed through to the manual ``screenshot`` action so it
            can save ``step_{n}.png`` even on a passing step.

    Returns:
        A StepResult describing the outcome (pass or fail), with timing.
    """
    description = _describe(step)
    start = time.perf_counter()

    try:
        handler = _ACTION_HANDLERS.get(step.action)
        if handler is None:
            raise ExecutorError(f"Unsupported action: '{step.action}'")

        # The manual screenshot action needs the step number and directory to
        # build its filename; all other handlers take (page, step) only.
        if step.action == "screenshot":
            message = handler(page, step, step_number=step_number, screenshot_dir=screenshot_dir)
        else:
            message = handler(page, step)
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info("Step %d PASS (%s): %s", step_number, step.action, message)
        return StepResult(
            step_number=step_number,
            action=step.action,
            description=description,
            status="pass",
            message=message,
            duration_ms=duration_ms,
        )

    except PlaywrightTimeoutError as exc:
        message = f"Timed out ({DEFAULT_TIMEOUT_MS} ms): {exc}"
    except (ElementNotFoundError, NavigationError, ExecutorError) as exc:
        message = str(exc)
    except PlaywrightError as exc:
        message = f"Playwright error: {exc}"
    except Exception as exc:  # noqa: BLE001
        message = f"Unexpected error: {type(exc).__name__}: {exc}"

    duration_ms = (time.perf_counter() - start) * 1000
    logger.error("Step %d FAIL (%s): %s", step_number, step.action, message)

    # NEW: capture a screenshot for this failure, best-effort. Placed after
    # the duration/logging so a screenshot failure never affects the
    # measured step timing or masks the original error log line above.
    if screenshot_dir:
        _capture_failure_screenshot(page, screenshot_dir, step_number)

    return StepResult(
        step_number=step_number,
        action=step.action,
        description=description,
        status="fail",
        message=message,
        duration_ms=duration_ms,
    )


def run_plan(
    page: Page,
    steps: List[TestStep],
    screenshot_dir: Optional[str] = None,
) -> List[StepResult]:
    """
    Execute a list of TestSteps in order, without stopping on failure.

    Args:
        page: The active Playwright Page object.
        steps: The ordered list of validated TestSteps to execute.
        screenshot_dir: Optional directory to save failure screenshots into.
            Passed straight through to run_step for each failed step.

    Returns:
        A list of StepResult objects, one per input step, in order.
    """
    results: List[StepResult] = []
    logger.info("Starting run_plan with %d step(s)", len(steps))

    for index, step in enumerate(steps, start=1):
        result = run_step(page, step, step_number=index, screenshot_dir=screenshot_dir)
        results.append(result)

    passed = sum(1 for r in results if r.status == "pass")
    logger.info("run_plan complete: %d/%d step(s) passed", passed, len(results))
    return results


def _describe(step: TestStep) -> str:
    parts = [step.action]
    if step.url:
        parts.append(step.url)
    if step.target:
        parts.append(f"'{step.target}'")
    if step.value is not None and step.action != "goto":
        parts.append(f"= '{step.value}'")
    return " ".join(parts)