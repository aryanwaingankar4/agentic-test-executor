"""
executor.py
-----------
Executes validated TestStep objects against a real browser using
Playwright's synchronous API.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, List, Literal, Optional

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Page
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from pydantic import BaseModel

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

_FILLER_WORDS = re.compile(r"\b(field|button|box|link|the)\b", re.IGNORECASE)


def find_element(page, description: str):
    keyword = _FILLER_WORDS.sub("", description).strip()
    if not keyword:
        keyword = description

    pattern = re.compile(re.escape(keyword), re.IGNORECASE)

    strategies = [
        lambda: page.get_by_placeholder(pattern),
        lambda: page.get_by_label(pattern),
        lambda: page.get_by_role("button", name=pattern),
        lambda: page.get_by_role("link", name=pattern),
        lambda: page.get_by_text(pattern),
        lambda: page.locator(f"#{keyword}"),
        lambda: page.locator(f"[name*='{keyword}' i]"),
        lambda: page.locator(f"[id*='{keyword}' i]"),
        lambda: page.locator(f"[aria-label*='{keyword}' i]"),
        lambda: page.locator("button[type=submit], input[type=submit]"),
    ]

    for strategy in strategies:
        try:
            locator = strategy()
            if locator.count() > 0:
                return locator.first
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


_ACTION_HANDLERS: dict[str, Callable[[Page, TestStep], str]] = {
    "goto": _do_goto,
    "type": _do_type,
    "click": _do_click,
    "verify": _do_verify,
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
            this step fails. If omitted, no screenshot is captured.

    Returns:
        A StepResult describing the outcome (pass or fail), with timing.
    """
    description = _describe(step)
    start = time.perf_counter()

    try:
        handler = _ACTION_HANDLERS.get(step.action)
        if handler is None:
            raise ExecutorError(f"Unsupported action: '{step.action}'")

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