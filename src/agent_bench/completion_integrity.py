"""Conservative deterministic completion statements from model response evidence."""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from typing import Literal

from agent_bench.events import NormalizedEvent, RawEvent

SELF_REPORT_METHOD = "model_self_report_phrase_v1"
TECHNICAL_PRIMARY_TERMINATIONS = frozenset({"success", "no_changes"})
SelfReportCategory = Literal[
    "time_limit", "token_limit", "context_limit", "step_or_turn_limit", "general_incomplete",
]


@dataclass(frozen=True)
class SelfReportMatch:
    category: SelfReportCategory
    source_event_id: str
    response_index: int
    rule_id: str
    matched_phrase: str


def primary_eligibility(
    *, termination_class: object, functional_expected: bool,
    functional_validation_status: object, hard_gate_pass: object,
    self_reported_incomplete: object,
) -> tuple[bool, tuple[str, ...]]:
    """One report-wide population rule; no free-text termination inference."""
    reasons: list[str] = []
    if termination_class not in TECHNICAL_PRIMARY_TERMINATIONS:
        reasons.append("objective_technical_incomplete")
    if functional_expected:
        if functional_validation_status != "pass":
            reasons.append("functional_failure")
        if hard_gate_pass is not True:
            reasons.append("hard_gate_failure")
    if self_reported_incomplete is True:
        reasons.append("self_reported_incomplete")
    return not reasons, tuple(reasons)


_RULES: tuple[tuple[SelfReportCategory, str, re.Pattern[str]], ...] = (
    ("time_limit", "time_limit_explicit_v1", re.compile(r"\b(?:i (?:ran|have run) out of time|i(?:'m| am) out of time|i(?:'ve| have)? ?reached the time limit|i hit the time limit|the time limit was reached)\b", re.I)),
    ("token_limit", "token_limit_explicit_v1", re.compile(r"\b(?:i (?:ran|have run) out of tokens|i(?:'m| am) out of tokens|i(?:'ve| have)? ?reached the token limit|i hit the token limit|my token budget is exhausted|the token budget is exhausted)\b", re.I)),
    ("context_limit", "context_limit_explicit_v1", re.compile(r"\b(?:i ran out of context|i(?:'ve| have)? ?reached the context limit|i hit the context limit|the context window is full|context limit was reached)\b", re.I)),
    ("step_or_turn_limit", "step_turn_limit_explicit_v1", re.compile(r"\b(?:i(?:'ve| have)? ?reached the (?:step|turn) limit|i hit the (?:step|turn) limit|maximum steps reached|turn limit reached|iteration limit reached)\b", re.I)),
    ("general_incomplete", "general_incomplete_explicit_v1", re.compile(r"\bi (?:could not|couldn't|was unable to) (?:finish|complete)\b", re.I)),
)


def detect_model_self_reports(
    events: tuple[NormalizedEvent, ...], raw_events: tuple[RawEvent, ...],
) -> tuple[SelfReportMatch, ...]:
    """Inspect only captured model/assistant response text, never prompts or tools."""
    responses = list(_raw_response_texts(raw_events))
    # Some controlled/test captures expose final assistant text directly in a
    # normalized llm response.  Do not inspect any other normalized payload.
    responses.extend(
        (event.event_id, text)
        for event in events
        if event.event_kind == "llm_response"
        for text in (_response_text_field(event.payload),)
        if text
    )
    matches: list[SelfReportMatch] = []
    seen: set[tuple[str, str, str]] = set()
    for index, (event_id, text) in enumerate(responses, start=1):
        # Quoted/code spans are commonly copied source or documentation.  It
        # is safer to miss an ambiguous statement than classify it as a limit.
        candidate = re.sub(r"(?:`[^`]*`|'[^']*'|\"[^\"]*\")", " ", text)
        candidate = " ".join(candidate.split())
        for category, rule_id, pattern in _RULES:
            found = pattern.search(candidate)
            if found is None:
                continue
            phrase = found.group(0)
            key = (event_id, rule_id, phrase.casefold())
            if key not in seen:
                seen.add(key)
                matches.append(SelfReportMatch(category, event_id, index, rule_id, phrase))
    return tuple(matches)


def _raw_response_texts(raw_events: tuple[RawEvent, ...]):
    for event in raw_events:
        if event.source == "proxy" and event.event_type == "llm_response":
            text = _proxy_visible_text(event.payload)
        elif event.event_type == "hermes_session_message":
            native = event.payload.get("native_event")
            text = _assistant_content(native)
        elif event.event_type == "pi_event":
            native = event.payload.get("native_event")
            text = _pi_assistant_content(native)
        else:
            text = None
        if text:
            yield event.raw_event_id, text


def _response_text_field(payload: object) -> str | None:
    if not isinstance(payload, dict):
        return None
    for key in ("visible_content", "response_text", "assistant_text"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return None


def _proxy_visible_text(payload: object) -> str | None:
    if not isinstance(payload, dict) or not isinstance(payload.get("body_base64"), str):
        return None
    try:
        body = base64.b64decode(payload["body_base64"], validate=True)
    except (ValueError, TypeError):
        return None
    parts: list[str] = []
    for line in body.splitlines() if body.lstrip().startswith(b"data:") else (body,):
        if line.startswith(b"data:"):
            line = line[5:].strip()
        if not line or line == b"[DONE]":
            continue
        try:
            parsed = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(parsed, dict):
            continue
        for choice in parsed.get("choices", []):
            if not isinstance(choice, dict):
                continue
            for message in (choice.get("message"), choice.get("delta")):
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    parts.append(message["content"])
    return "".join(parts) or None


def _assistant_content(native: object) -> str | None:
    if not isinstance(native, dict) or native.get("role") != "assistant":
        return None
    content = native.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(item["text"] for item in content if isinstance(item, dict) and item.get("type") in {"text", "output_text"} and isinstance(item.get("text"), str)) or None
    return None


def _pi_assistant_content(native: object) -> str | None:
    if not isinstance(native, dict) or native.get("type") != "message_end":
        return None
    return _assistant_content(native.get("message"))
