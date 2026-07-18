"""
executor.py
-----------
Executes validated TestStep objects against a real browser using
Playwright's synchronous API.

This module is the "hands" of the agentic test framework: parser.py turns
plain-English into validated TestStep objects, and executor.py performs
those steps in a live browser and reports the outcome of each one.

Design goals (see module docstring notes and inline comments):
  * Never crash the whole run because of one bad step.
  * Locate elements robustly using several fallback strategies.
  * Fail fast (explicit timeouts) instead of hanging on defaults.
  * Report the outcome of every step, even after a failure.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, List, Literal, Optional

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Page
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from pydantic import BaseModel

# Import your existing schema. Adjust the module path if parser.py lives
# somewhere else in your package.
from parser import TestStep


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
# We configure a module-level logger rather than using print(). Logging gives
# us levels (INFO/WARNING/ERROR), timestamps, and the ability to redirect
# output to a file or a UI later without touching call sites. print() would
# force us to rip out statements everywhere when we productionise this.
logger = logging.getLogger(__name__)


# A single explicit timeout used for every locator/action. Playwright's default
# is 30s, which means a broken selector search can hang the run for half a
# minute per step. 5s is long enough for a normally-loading element but short
# enough to fail fast on a genuinely missing one.
DEFAULT_TIMEOUT_MS: float = 5000


# --------------------------------------------------------------------------- #
# Custom exception hierarchy
# --------------------------------------------------------------------------- #
# Mirrors the parser's ParserError/LLMParseError/RuleParseError pattern.
# Custom exceptions let calling code distinguish *our* well-understood failure
# modes from unexpected bugs, and let us attach clear, domain-specific messages.
class ExecutorError(Exception):
    """Base class for all errors raised by the executor module."""


class ElementNotFoundError(ExecutorError):
    """Raised when no locator strategy can find the requested element."""


class NavigationError(ExecutorError):
    """Raised when navigating to a URL fails."""


# --------------------------------------------------------------------------- #
# Result model
# --------------------------------------------------------------------------- #
class StepResult(BaseModel):
    """
    Structured outcome of executing a single TestStep.

    Using a Pydantic model (rather than a dict) keeps the result schema
    validated and self-documenting, and makes it trivial to serialise the
    final report to JSON for a UI or CI pipeline.
    """

    step_number: int
    action: str
    description: str
    status: Literal["pass", "fail"]
    message: str
    duration_ms: float


# --------------------------------------------------------------------------- #
# Smart element locating
# --------------------------------------------------------------------------- #
import re

_FILLER_WORDS = re.compile(r"\b(field|button|box|link|the)\b", re.IGNORECASE)


def find_element(page, description: str):
    """
    Try several strategies to locate an element matching a human description.
    Strips filler words (e.g. "field", "button") since real HTML rarely
    contains them literally — searching for "username field" verbatim will
    almost never match anything, even though "username" alone will.
    """
    keyword = _FILLER_WORDS.sub("", description).strip()
    if not keyword:
        keyword = description  # fallback if stripping left nothing useful

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


# --------------------------------------------------------------------------- #
# Per-action helpers
# --------------------------------------------------------------------------- #
def _do_goto(page: Page, step: TestStep) -> str:
    """
    Navigate the page to step.url.

    Args:
        page: The active Playwright Page object.
        step: The TestStep whose action is "goto".

    Returns:
        A success message describing the navigation.

    Raises:
        NavigationError: If step.url is missing or navigation fails.
    """
    if not step.url:
        raise NavigationError("goto step is missing a 'url' value")
    try:
        page.goto(step.url, timeout=DEFAULT_TIMEOUT_MS)
    except Exception as exc:  # noqa: BLE001
        # Wrap ANY navigation failure (timeout, DNS, connection refused, etc.)
        # in our own exception with context, rather than leaking Playwright's.
        raise NavigationError(f"Failed to navigate to '{step.url}': {exc}") from exc
    return f"Navigated to {step.url}"


def _do_type(page: Page, step: TestStep) -> str:
    """
    Fill the target element with step.value.

    Args:
        page: The active Playwright Page object.
        step: The TestStep whose action is "type".

    Returns:
        A success message describing what was typed.

    Raises:
        ExecutorError: If target or value is missing.
        ElementNotFoundError: If the target element cannot be located.
    """
    if not step.target:
        raise ExecutorError("type step is missing a 'target'")
    if step.value is None:
        raise ExecutorError("type step is missing a 'value'")
    locator = find_element(page, step.target)
    locator.fill(step.value, timeout=DEFAULT_TIMEOUT_MS)
    return f"Typed '{step.value}' into '{step.target}'"


def _do_click(page: Page, step: TestStep) -> str:
    """
    Click the target element.

    Args:
        page: The active Playwright Page object.
        step: The TestStep whose action is "click".

    Returns:
        A success message describing the click.

    Raises:
        ExecutorError: If target is missing.
        ElementNotFoundError: If the target element cannot be located.
    """
    if not step.target:
        raise ExecutorError("click step is missing a 'target'")
    locator = find_element(page, step.target)
    locator.click(timeout=DEFAULT_TIMEOUT_MS)
    return f"Clicked '{step.target}'"


def _do_verify(page: Page, step: TestStep) -> str:
    """
    Verify that step.value text is present on the page.

    We check the rendered page content for the expected text. Using
    page.content() (the current DOM HTML) keeps the check simple and robust;
    for stricter assertions you could swap in expect(locator).to_be_visible().

    Args:
        page: The active Playwright Page object.
        step: The TestStep whose action is "verify".

    Returns:
        A success message confirming the text was found.

    Raises:
        ExecutorError: If value is missing or the text is not present.
    """
    if step.value is None:
        raise ExecutorError("verify step is missing a 'value' to check for")
    content = page.content()
    if step.value not in content:
        raise ExecutorError(f"Expected text '{step.value}' not found on page")
    return f"Verified text '{step.value}' is present"


# Dispatch table maps each action to its handler. Cleaner than a long if/elif
# chain and, like the strategy list, makes adding a new action a one-line edit.
_ACTION_HANDLERS: dict[str, Callable[[Page, TestStep], str]] = {
    "goto": _do_goto,
    "type": _do_type,
    "click": _do_click,
    "verify": _do_verify,
}


# --------------------------------------------------------------------------- #
# Single-step execution
# --------------------------------------------------------------------------- #
def run_step(page: Page, step: TestStep, step_number: int = 1) -> StepResult:
    """
    Execute exactly one TestStep and return a structured StepResult.

    This function is the safety boundary of the executor. EVERY possible
    failure -- expected or not -- is caught here and converted into a
    status="fail" StepResult. Nothing is allowed to propagate out and crash
    the run. That guarantee is what lets run_plan continue past failures.

    Args:
        page: The active Playwright Page object.
        step: A validated TestStep to execute.
        step_number: 1-based index of this step, used in the report.

    Returns:
        A StepResult describing the outcome (pass or fail), with timing.
    """
    description = _describe(step)
    start = time.perf_counter()

    try:
        handler = _ACTION_HANDLERS.get(step.action)
        if handler is None:
            # Should be impossible given the Pydantic Literal, but we defend
            # against it anyway rather than assume upstream validation held.
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

    # --- Specific handling first, for clearer messages / logging ---------- #
    except PlaywrightTimeoutError as exc:
        message = f"Timed out ({DEFAULT_TIMEOUT_MS} ms): {exc}"
    except (ElementNotFoundError, NavigationError, ExecutorError) as exc:
        # Our own well-understood domain errors.
        message = str(exc)
    except PlaywrightError as exc:
        # Any other Playwright-level error (detached element, bad selector...).
        message = f"Playwright error: {exc}"
    # --- Broad catch-all LAST: this is the critical safety net ------------ #
    except Exception as exc:  # noqa: BLE001
        # This mirrors the exact bug you hit in parser.py: an *unanticipated*
        # exception type (there, a Gemini ServerError) had no handler and
        # crashed everything. Here, anything we didn't foresee still becomes a
        # clean failed result instead of taking down the whole test run.
        message = f"Unexpected error: {type(exc).__name__}: {exc}"

    duration_ms = (time.perf_counter() - start) * 1000
    logger.error("Step %d FAIL (%s): %s", step_number, step.action, message)
    return StepResult(
        step_number=step_number,
        action=step.action,
        description=description,
        status="fail",
        message=message,
        duration_ms=duration_ms,
    )


# --------------------------------------------------------------------------- #
# Full-plan execution
# --------------------------------------------------------------------------- #
def run_plan(page: Page, steps: List[TestStep]) -> List[StepResult]:
    """
    Execute a list of TestSteps in order, without stopping on failure.

    Unlike a typical short-circuiting script, we run EVERY step so the final
    report shows the outcome of each one. This is far more useful for
    debugging: you see not just where it first broke, but everything that
    happened after, which often reveals cascading or unrelated issues.

    Args:
        page: The active Playwright Page object.
        steps: The ordered list of validated TestSteps to execute.

    Returns:
        A list of StepResult objects, one per input step, in order.
    """
    results: List[StepResult] = []
    logger.info("Starting run_plan with %d step(s)", len(steps))

    for index, step in enumerate(steps, start=1):
        # run_step never raises, so a single bad step can't abort the loop.
        result = run_step(page, step, step_number=index)
        results.append(result)

    passed = sum(1 for r in results if r.status == "pass")
    logger.info("run_plan complete: %d/%d step(s) passed", passed, len(results))
    return results


# --------------------------------------------------------------------------- #
# Small utility
# --------------------------------------------------------------------------- #
def _describe(step: TestStep) -> str:
    """
    Build a short human-readable description of a step for the report.

    Args:
        step: The TestStep to describe.

    Returns:
        A concise one-line description, e.g. "click 'Sign in'".
    """
    parts = [step.action]
    if step.url:
        parts.append(step.url)
    if step.target:
        parts.append(f"'{step.target}'")
    if step.value is not None and step.action != "goto":
        parts.append(f"= '{step.value}'")
    return " ".join(parts)
