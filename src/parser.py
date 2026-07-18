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
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

GEMINI_MODEL: str = "gemini-flash-latest"


class ParserError(Exception):
    """Base class for all parsing errors in this module."""


class LLMParseError(ParserError):
    """Raised when the LLM path fails (bad JSON, empty response, API error, etc.)."""


class RuleParseError(ParserError):
    """Raised when the rule-based parser cannot produce any valid step."""


class TestStep(BaseModel):
    """One executable test instruction, validated on creation."""

    action: Literal["goto", "type", "click", "verify"]
    target: Optional[str] = None
    value: Optional[str] = None
    url: Optional[str] = None

    @model_validator(mode="after")
    def _check_cross_field_rules(self) -> "TestStep":
        if self.action == "goto":
            if not self.url:
                raise ValueError("action 'goto' requires a 'url'")
        elif self.action == "type":
            if not self.target or not self.value:
                raise ValueError("action 'type' requires both 'target' and 'value'")
        elif self.action == "click":
            if not self.target:
                raise ValueError("action 'click' requires a 'target'")
        elif self.action == "verify":
            if not self.target and not self.value:
                raise ValueError("action 'verify' requires 'target' or 'value'")
        return self


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


def _extract_json_array(raw_output: str) -> str:
    """Strip markdown fences and isolate the JSON array from LLM output."""
    if not raw_output or not raw_output.strip():
        raise LLMParseError("LLM returned an empty response")

    text = raw_output.strip()

    fence_pattern = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
    fence_match = fence_pattern.search(text)
    if fence_match:
        text = fence_match.group(1).strip()

    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise LLMParseError(f"Could not locate a JSON array in LLM output: {raw_output!r}")

    return text[start : end + 1]


def _parse_json_to_steps(cleaned_json: str) -> list[TestStep]:
    """Turn a cleaned JSON-array string into validated TestStep objects."""
    try:
        data = json.loads(cleaned_json)
    except json.JSONDecodeError as exc:
        raise LLMParseError(f"Invalid JSON from LLM: {exc}") from exc

    if not isinstance(data, list):
        raise LLMParseError("Expected a JSON array of steps at the top level")

    return [TestStep(**item) for item in data]


_TRANSIENT_MARKERS = ("429", "500", "502", "503", "504", "timeout", "deadline")


def _is_transient(exc: BaseException) -> bool:
    """Heuristic: does this exception look like a temporary server hiccup?"""
    message = str(exc).lower()
    return any(marker in message for marker in _TRANSIENT_MARKERS)


@retry(
    retry=retry_if_exception(_is_transient),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
def _call_gemini(contents: str) -> str:
    """
    Make a single Gemini call and return the raw text response.
    """
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise LLMParseError("GEMINI_API_KEY environment variable is not set")

    client = genai.Client(api_key=api_key)

    logger.debug("Calling Gemini model %s", GEMINI_MODEL)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0,
            http_options=types.HttpOptions(timeout=15000),
        ),
    )

    if not response.text:
        raise LLMParseError("Gemini returned no text content")

    return response.text


def parse_with_llm(raw_text: str) -> list[TestStep]:
    """
    Parse via Gemini. On a *validation* failure, retry ONCE with the error
    appended so the model can fix its own mistake. On ANY OTHER failure,
    convert it to LLMParseError immediately so parse_steps() can fall back.
    """
    if not raw_text or not raw_text.strip():
        raise LLMParseError("Cannot parse empty input")

    prompt = f"Parse these test steps:\n{raw_text}"

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
    except LLMParseError:
        raise
    except Exception as exc:
        logger.warning("Gemini call failed with an unexpected error: %s", exc)
        raise LLMParseError(f"Gemini call failed: {exc}") from exc

    correction_prompt = (
        f"{prompt}\n\n"
        f"Your previous answer failed validation with this error:\n"
        f"{first_error}\n"
        f"Fix the issues and return ONLY the corrected JSON array."
    )
    try:
        raw_output = _call_gemini(correction_prompt)
        cleaned = _extract_json_array(raw_output)
        steps = _parse_json_to_steps(cleaned)
        logger.info("LLM parsed %d step(s) after self-correction", len(steps))
        return steps
    except ValidationError as second_error:
        raise LLMParseError(
            f"LLM output still invalid after self-correction: {second_error}"
        ) from second_error
    except LLMParseError:
        raise
    except Exception as exc:
        logger.warning("Gemini self-correction call failed: %s", exc)
        raise LLMParseError(f"Gemini self-correction call failed: {exc}") from exc


# --------------------------------------------------------------------------- #
# NEW: quote-stripping helper for the rule-based fallback parser
# --------------------------------------------------------------------------- #
_QUOTED_TEXT_RE = re.compile(r'["\']([^"\']+)["\']')


def _strip_quotes(text: Optional[str]) -> Optional[str]:
    """
    Extract the quoted portion of a string if present, otherwise return it
    trimmed as-is. Handles two real issues seen with raw regex captures:
      1. `type "tomsmith" into...` capturing the value WITH its quote marks.
      2. `verify the page shows the text "X"` capturing the whole trailing
         phrase instead of just the quoted part — searching for a quoted
         substring anywhere in the captured text fixes both at once.
    """
    if not text:
        return text
    match = _QUOTED_TEXT_RE.search(text)
    return match.group(1) if match else text.strip()


_RULE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "goto",
        re.compile(r"^\s*(?:go\s*to|goto|navigate\s*to|open)\s+(?P<url>\S+)", re.IGNORECASE),
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
        re.compile(r"^\s*(?:click|press|tap)\s+(?:on\s+)?(?:the\s+)?(?P<target>.+)", re.IGNORECASE),
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
    """Regex fallback — validated through the same TestStep model."""
    if not raw_text or not raw_text.strip():
        raise RuleParseError("Cannot parse empty input")

    steps: list[TestStep] = []

    for line_number, line in enumerate(raw_text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue

        matched = False
        for action, pattern in _RULE_PATTERNS:
            match = pattern.match(stripped)
            if not match:
                continue

            fields = match.groupdict()
            # CHANGED: the "value" field now goes through _strip_quotes so
            # captured text like '"tomsmith"' becomes 'tomsmith', and a
            # captured phrase like 'the page shows the text "X"' becomes
            # just 'X'. target/url are cleaned as before.
            cleaned_fields = {
                key: (
                    _strip_quotes(value.rstrip("."))
                    if key == "value"
                    else value.strip().rstrip(".") if value else None
                )
                for key, value in fields.items()
            }

            try:
                step = TestStep(action=action, **cleaned_fields)
            except ValidationError as exc:
                logger.warning(
                    "Line %d matched '%s' but failed validation: %s",
                    line_number, action, exc,
                )
                continue

            steps.append(step)
            matched = True
            break

        if not matched:
            logger.warning("Line %d did not match any rule: %r", line_number, stripped)

    if not steps:
        raise RuleParseError("No valid steps could be parsed from the input")

    logger.info("Rule-based parser produced %d step(s)", len(steps))
    return steps


def parse_steps(raw_text: str) -> list[TestStep]:
    """Main entry point. Try the LLM; fall back to rules on any ParserError."""
    logger.info("Parsing %d character(s) of test steps", len(raw_text or ""))

    try:
        return parse_with_llm(raw_text)
    except ParserError as llm_error:
        logger.warning("LLM parsing failed (%s); falling back to rules", llm_error)

    try:
        return parse_with_rules(raw_text)
    except RuleParseError as rule_error:
        logger.error("Rule-based parsing also failed: %s", rule_error)
        raise ParserError(
            "Both LLM and rule-based parsing failed for the given input"
        ) from rule_error


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