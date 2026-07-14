"""
parser.py
=========

Converts plain-English QA test steps into a validated list of `TestStep`
objects that a downstream Playwright module can execute.

Two parsing strategies are provided:

    1. parse_with_llm(...)   -> uses Gemini (google-genai SDK)
    2. parse_with_rules(...) -> regex-based, no network needed

The public entry point `parse_steps(...)` tries the LLM first and falls
back to the rules parser if the LLM path fails for any reason.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Literal, Optional

from pydantic import BaseModel, ValidationError, model_validator
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
# We never use print(). print() writes to stdout with no severity level, no
# timestamp, and no way to route/silence it. logging gives us levels
# (DEBUG/INFO/WARNING/ERROR), timestamps, and lets whoever imports this module
# decide where messages go. Getting a named logger (__name__) means log lines
# are tagged "parser", so in a bigger app you can tell which module spoke.
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# The model name lives in ONE place (Bug #1: model deprecation)
# ---------------------------------------------------------------------------
# Google deprecates dated model names (e.g. "gemini-2.5-flash") on a schedule.
# The "-latest" alias always points at the current stable Flash model, and by
# putting it in a single named constant, the day it *does* need changing you
# edit exactly one line instead of hunting through the file.
GEMINI_MODEL: str = "gemini-flash-latest"


# ---------------------------------------------------------------------------
# Custom exceptions (so we never write `except Exception`)
# ---------------------------------------------------------------------------
# A bare `except Exception` catches *everything* — including bugs you'd rather
# see crash loudly (typos, KeyErrors) — and hides them. Custom exception types
# let callers catch exactly the failure they know how to handle. We also build
# a small hierarchy so a caller can catch the parent `ParserError` to mean
# "any parsing failure", or a specific child for finer control.
class ParserError(Exception):
    """Base class for all parsing errors in this module."""


class LLMParseError(ParserError):
    """Raised when the LLM path fails (bad JSON, empty response, etc.)."""


class RuleParseError(ParserError):
    """Raised when the rule-based parser cannot produce any valid step."""


# ---------------------------------------------------------------------------
# The data model
# ---------------------------------------------------------------------------
class TestStep(BaseModel):
    """
    One executable test instruction.

    Using Pydantic (not a plain dict) means every step is *validated* the
    moment it's created: wrong types, unknown actions, or missing required
    fields raise a ValidationError immediately, instead of silently blowing
    up later inside Playwright.
    """

    # Literal[...] means `action` can ONLY be one of these four strings.
    # Anything else (e.g. "scroll") is rejected by Pydantic automatically —
    # this is our first line of defense before the cross-field rules below.
    action: Literal["goto", "type", "click", "verify"]

    # These are Optional because which ones are *required* depends on the
    # action. A "goto" step has no `target`; a "type" step needs both
    # `target` and `value`. We express those conditional rules below.
    target: Optional[str] = None   # e.g. a CSS selector or field label
    value: Optional[str] = None    # text to type, or text to verify
    url: Optional[str] = None       # only used by "goto"

    @model_validator(mode="after")
    def _check_cross_field_rules(self) -> "TestStep":
        """
        Enforce the per-action requirements.

        `mode="after"` runs this AFTER Pydantic has parsed and type-checked
        each field, so here `self` is a fully-built object and we can safely
        compare fields against each other. This is THE enforcement layer that
        our LLM prompt relies on (see Bug #2 explanation), so it must be
        strict and explicit.
        """
        if self.action == "goto":
            if not self.url:
                raise ValueError("action 'goto' requires a 'url'")

        elif self.action == "type":
            if not self.target or not self.value:
                raise ValueError(
                    "action 'type' requires both 'target' and 'value'"
                )

        elif self.action == "click":
            if not self.target:
                raise ValueError("action 'click' requires a 'target'")

        elif self.action == "verify":
            if not self.target and not self.value:
                raise ValueError(
                    "action 'verify' requires 'target' or 'value'"
                )

        return self


# ---------------------------------------------------------------------------
# The LLM prompt (Bug #2: schema/prompt conflict)
# ---------------------------------------------------------------------------
# WHY NOT a rigid JSON schema?
# If we passed a schema derived from TestStep where every field is Optional,
# Gemini reads that as "null is always acceptable" and will happily emit
# {"action": "goto", "url": null}. Worse, structured-output mode tends to
# *ignore* plain-English rules in the prompt because the schema "wins".
# The SDK docs even warn against duplicating a schema in the prompt.
#
# So instead: no schema. We give a crystal-clear instruction plus ONE full
# worked example (input -> output). The example teaches format far better than
# prose, and Pydantic's model_validator remains the *real* enforcement — if
# the model slips, validation catches it and we self-correct (see below).
SYSTEM_PROMPT = """\
You are a QA test-step parser. Convert the user's plain-English test steps
into a JSON array of step objects. Output ONLY the JSON array — no prose, no
explanation, no markdown code fences.

Each object has these fields:
  - "action": one of "goto", "type", "click", "verify"
  - "target": a UI element description or selector (string or null)
  - "value":  text to type or text to verify (string or null)
  - "url":    a URL (string or null)

Rules you MUST follow (do not emit null where a value is required):
  - "goto"   requires "url".
  - "type"   requires "target" AND "value".
  - "click"  requires "target".
  - "verify" requires "target" OR "value".

Worked example.

Input:
  Go to https://example.com/login
  Type admin into the username field
  Click the login button
  Verify text Welcome appears

Output:
[
  {"action": "goto", "target": null, "value": null, "url": "https://example.com/login"},
  {"action": "type", "target": "username field", "value": "admin", "url": null},
  {"action": "click", "target": "login button", "value": null, "url": null},
  {"action": "verify", "target": null, "value": "Welcome", "url": null}
]
"""


# ---------------------------------------------------------------------------
# JSON extraction helper (Bug #3: JSON parsing failures)
# ---------------------------------------------------------------------------
def _extract_json_array(raw_output: str) -> str:
    """
    Clean up an LLM response so it can be handed to json.loads().

    Handles two common Gemini quirks:
      1. Output wrapped in markdown fences: ```json ... ```
      2. Leading/trailing chatter around the array.

    Strategy: strip any code fences, then slice from the first '[' to the
    last ']'. That reliably isolates the array even if the model added a
    stray sentence. We do NOT try to json.loads here — we just return the
    cleaned string so the caller controls error handling.
    """
    if not raw_output or not raw_output.strip():
        raise LLMParseError("LLM returned an empty response")

    text = raw_output.strip()

    # Remove ```json ... ``` or ``` ... ``` fences if present.
    fence_pattern = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
    fence_match = fence_pattern.search(text)
    if fence_match:
        text = fence_match.group(1).strip()

    # Slice between the first '[' and the last ']'.
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise LLMParseError(
            f"Could not locate a JSON array in LLM output: {raw_output!r}"
        )

    return text[start : end + 1]


def _parse_json_to_steps(cleaned_json: str) -> list[TestStep]:
    """
    Turn a cleaned JSON-array string into validated TestStep objects.

    Two distinct failure modes, two distinct exceptions:
      - json.JSONDecodeError  -> the text isn't valid JSON at all.
      - ValidationError       -> it's valid JSON but breaks our rules.
    We re-raise the first as LLMParseError, but let ValidationError bubble
    up UNCHANGED, because the caller uses its message to build a self-correct
    retry prompt (see parse_with_llm).
    """
    try:
        data = json.loads(cleaned_json)
    except json.JSONDecodeError as exc:
        # Genuine parse failure — wrap it, don't swallow it.
        raise LLMParseError(f"Invalid JSON from LLM: {exc}") from exc

    if not isinstance(data, list):
        raise LLMParseError("Expected a JSON array of steps at the top level")

    # Let ValidationError propagate on purpose (used for self-correction).
    return [TestStep(**item) for item in data]


# ---------------------------------------------------------------------------
# The raw Gemini network call, with tenacity backoff for TRANSIENT errors
# ---------------------------------------------------------------------------
# tenacity retries ONLY transient network/API problems (429 rate-limits,
# 5xx server errors, timeouts). We detect these crudely-but-safely by
# inspecting the exception message. Crucially, tenacity does NOT retry
# validation failures — those aren't transient, and retrying identically
# would just waste calls. Validation self-correction is handled separately,
# one level up, where we can actually change the prompt.
_TRANSIENT_MARKERS = ("429", "500", "502", "503", "504", "timeout", "deadline")


def _is_transient(exc: BaseException) -> bool:
    """Heuristic: does this exception look like a temporary server hiccup?"""
    message = str(exc).lower()
    return any(marker in message for marker in _TRANSIENT_MARKERS)


from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

@retry(
    retry=retry_if_exception(_is_transient),   # retry only when our check says "transient"
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)

def _call_gemini(contents: str) -> str:
    """
    Make a single Gemini call and return the raw text response.

    Imports of the SDK happen INSIDE the function so that the rules-only path
    works even if google-genai isn't installed — a nice property for a tool
    that advertises an "offline fallback".
    """
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        # Not transient, so tenacity won't retry this — correct behaviour.
        raise LLMParseError("GEMINI_API_KEY environment variable is not set")

    client = genai.Client(api_key=api_key)

    logger.debug("Calling Gemini model %s", GEMINI_MODEL)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0,  # deterministic: we want parsing, not creativity
        ),
    )

    if not response.text:
        raise LLMParseError("Gemini returned no text content")

    return response.text


# ---------------------------------------------------------------------------
# Public: LLM parser with ONE validation self-correction retry
# ---------------------------------------------------------------------------
def parse_with_llm(raw_text: str) -> list[TestStep]:
    """
    Parse via Gemini. On a *validation* failure, retry ONCE with the error
    appended so the model can fix its own mistake, then give up.

    Note the two different retry mechanisms working together:
      - Inside _call_gemini, tenacity retries transient NETWORK errors.
      - Here, we retry ONCE for VALIDATION errors (a different problem needing
        a different, prompt-changing fix). Keeping them separate keeps each
        simple and prevents, say, retrying a rate-limit with a longer prompt.
    """
    if not raw_text or not raw_text.strip():
        raise LLMParseError("Cannot parse empty input")

    prompt = f"Parse these test steps:\n{raw_text}"

    # ---- Attempt 1 ----
    try:
        raw_output = _call_gemini(prompt)
        cleaned = _extract_json_array(raw_output)
        steps = _parse_json_to_steps(cleaned)
        logger.info("LLM parsed %d step(s) on first attempt", len(steps))
        return steps
    except ValidationError as first_error:
        logger.warning(
            "LLM output failed validation, retrying once with feedback: %s",
            first_error,
        )

    # ---- Attempt 2: self-correction ----
    # We feed the validation error back so the model knows exactly what to fix.
    correction_prompt = (
        f"{prompt}\n\n"
        f"Your previous answer failed validation with this error:\n"
        f"{first_error}\n"  # noqa: F821  (defined above in the except branch)
        f"Fix the issues and return ONLY the corrected JSON array."
    )
    try:
        raw_output = _call_gemini(correction_prompt)
        cleaned = _extract_json_array(raw_output)
        steps = _parse_json_to_steps(cleaned)
        logger.info("LLM parsed %d step(s) after self-correction", len(steps))
        return steps
    except ValidationError as second_error:
        # Still broken after one correction — convert to LLMParseError so the
        # top-level parse_steps() can decide to fall back to the rules parser.
        raise LLMParseError(
            f"LLM output still invalid after self-correction: {second_error}"
        ) from second_error


# ---------------------------------------------------------------------------
# Public: rule-based fallback parser (no API needed)
# ---------------------------------------------------------------------------
# Each pattern captures the pieces we need. `re.IGNORECASE` lets users write
# "Go To" or "goto". Named groups (?P<name>...) make the code readable.
_RULE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "goto",
        re.compile(
            r"^\s*(?:go\s*to|goto|navigate\s*to|open)\s+(?P<url>\S+)",
            re.IGNORECASE,
        ),
    ),
    (
        "type",
        re.compile(
            r"^\s*(?:type|enter|input)\s+(?P<value>.+?)\s+"
            r"(?:into|in)\s+(?:the\s+)?(?P<target>.+)",
            re.IGNORECASE,
        ),
    ),
    (
        "click",
        re.compile(
            r"^\s*(?:click|press|tap)\s+(?:on\s+)?(?:the\s+)?(?P<target>.+)",
            re.IGNORECASE,
        ),
    ),
    (
        "verify",
        re.compile(
            r"^\s*(?:verify|assert|check|confirm)\s+"
            r"(?:that\s+)?(?:text\s+)?(?P<value>.+?)"
            r"(?:\s+appears|\s+is\s+visible|\s+exists)?\s*$",
            re.IGNORECASE,
        ),
    ),
]


def parse_with_rules(raw_text: str) -> list[TestStep]:
    """
    Regex fallback. Processes the input line by line, tries each pattern in
    order, and validates EVERY match through TestStep before keeping it — so
    the rules parser can never emit a step the LLM parser couldn't.
    """
    if not raw_text or not raw_text.strip():
        raise RuleParseError("Cannot parse empty input")

    steps: list[TestStep] = []

    for line_number, line in enumerate(raw_text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue  # skip blank lines

        matched = False
        for action, pattern in _RULE_PATTERNS:
            match = pattern.match(stripped)
            if not match:
                continue

            fields = match.groupdict()

            # Clean captured text (trim trailing punctuation like "button.").
            cleaned_fields = {
                key: value.strip().rstrip(".") if value else None
                for key, value in fields.items()
            }

            try:
                # Same validation as the LLM path — the single source of truth.
                step = TestStep(action=action, **cleaned_fields)
            except ValidationError as exc:
                logger.warning(
                    "Line %d matched '%s' but failed validation: %s",
                    line_number, action, exc,
                )
                # Try the next pattern for this line rather than bailing.
                continue

            steps.append(step)
            matched = True
            break  # first successful pattern wins for this line

        if not matched:
            logger.warning("Line %d did not match any rule: %r",
                           line_number, stripped)

    if not steps:
        raise RuleParseError("No valid steps could be parsed from the input")

    logger.info("Rule-based parser produced %d step(s)", len(steps))
    return steps


# ---------------------------------------------------------------------------
# Public: top-level orchestrator — LLM first, rules as fallback
# ---------------------------------------------------------------------------
def parse_steps(raw_text: str) -> list[TestStep]:
    """
    Main entry point. Try the LLM; if anything in that path fails, log it and
    fall back to the rules parser. If BOTH fail, raise ParserError so the
    caller knows parsing genuinely failed (we don't return an empty list and
    pretend success).
    """
    logger.info("Parsing %d character(s) of test steps", len(raw_text or ""))

    try:
        return parse_with_llm(raw_text)
    except ParserError as llm_error:
        # ParserError covers LLMParseError AND the api-key-missing case.
        logger.warning("LLM parsing failed (%s); falling back to rules",
                       llm_error)

    try:
        return parse_with_rules(raw_text)
    except RuleParseError as rule_error:
        logger.error("Rule-based parsing also failed: %s", rule_error)
        raise ParserError(
            "Both LLM and rule-based parsing failed for the given input"
        ) from rule_error


# ---------------------------------------------------------------------------
# Quick manual check (only runs when you execute `python parser.py` directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    sample = (
        "Go to https://example.com/login\n"
        "Type admin into the username field\n"
        "Click the login button\n"
        "Verify text Welcome appears\n"
    )

    for parsed_step in parse_steps(sample):
        logger.info("Parsed: %s", parsed_step.model_dump())
