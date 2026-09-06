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
VisibleResponseCaptureStatus = Literal["complete", "partial", "unknown"]


@dataclass(frozen=True)
class VisibleModelResponse:
    """One sealed, user-visible assistant response and its exact provenance."""

    text: str
    source_event_id: str
    response_index: int
    finish_reason: str | None
    capture_status: VisibleResponseCaptureStatus
    extraction_method: str


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
    responses = extract_visible_model_responses(events, raw_events)
    matches: list[SelfReportMatch] = []
    seen: set[tuple[str, str, str]] = set()
    for response in responses:
        # Quoted/code spans are commonly copied source or documentation.  It
        # is safer to miss an ambiguous statement than classify it as a limit.
        candidate = re.sub(r"(?:`[^`]*`|'[^']*'|\"[^\"]*\")", " ", response.text)
        candidate = " ".join(candidate.split())
        for category, rule_id, pattern in _RULES:
            found = pattern.search(candidate)
            if found is None:
                continue
            phrase = found.group(0)
            key = (response.source_event_id, rule_id, phrase.casefold())
            if key not in seen:
                seen.add(key)
                matches.append(SelfReportMatch(
                    category, response.source_event_id, response.response_index,
                    rule_id, phrase,
                ))
    return tuple(matches)


def extract_visible_model_responses(
    events: tuple[NormalizedEvent, ...], raw_events: tuple[RawEvent, ...],
) -> tuple[VisibleModelResponse, ...]:
    """Extract only sealed, user-visible assistant text from supported sources.

    This is deliberately the single parser for completion self-reports and
    report presentation.  It never inspects prompts, tool data, or reasoning.
    Normalized visible fields are a fallback for captures that expose text only
    in the normalized representation; they are not duplicated when their raw
    source already yielded visible text.
    """
    extracted: list[tuple[int, int, str, str, str | None, VisibleResponseCaptureStatus, str]] = []
    for event in raw_events:
        if event.source == "proxy" and event.event_type == "llm_response":
            text = _proxy_visible_text(event.payload)
            finish_reason = _finish_reason(event.payload)
            status = _capture_status(finish_reason)
            method = "proxy_response_body_visible_content_v1"
        elif event.event_type == "hermes_session_message":
            native = event.payload.get("native_event")
            text = _assistant_content(native)
            finish_reason = _finish_reason(native)
            status = _capture_status(finish_reason)
            method = "hermes_session_assistant_content_v1"
        elif event.event_type == "pi_event":
            native = event.payload.get("native_event")
            text = _pi_assistant_content(native)
            message = native.get("message") if isinstance(native, dict) else None
            finish_reason = _finish_reason(message)
            status = _capture_status(finish_reason)
            method = "pi_message_end_assistant_content_v1"
        else:
            continue
        if text:
            extracted.append((
                event.sequence, 0, event.raw_event_id, text, finish_reason, status,
                method,
            ))

    raw_ids = {source_event_id for _position, _origin, source_event_id, *_rest in extracted}
    for event in events:
        if event.event_kind != "llm_response" or any(
            reference.raw_event_id in raw_ids for reference in event.raw_event_refs
        ):
            continue
        text = _response_text_field(event.payload)
        if not text:
            continue
        finish_reason = _finish_reason(event.payload)
        explicit_status = event.payload.get("capture_status") if isinstance(event.payload, dict) else None
        status = (
            explicit_status if explicit_status in {"complete", "partial", "unknown"}
            else _capture_status(finish_reason)
        )
        raw_position = min(
            (reference.raw_sequence for reference in event.raw_event_refs),
            default=len(raw_events) + event.sequence,
        )
        extracted.append((
            raw_position, 1, event.event_id, text, finish_reason, status,
            "normalized_llm_response_visible_field_v1",
        ))
    return tuple(
        VisibleModelResponse(
            text=text, source_event_id=source_event_id, response_index=index,
            finish_reason=finish_reason, capture_status=status,
            extraction_method=method,
        )
        for index, (_position, _origin, source_event_id, text, finish_reason, status, method)
        in enumerate(sorted(extracted), start=1)
    )


def select_final_visible_model_response(
    events: tuple[NormalizedEvent, ...], raw_events: tuple[RawEvent, ...],
    *, termination_class: object,
) -> tuple[VisibleModelResponse | None, VisibleModelResponse | None]:
    """Return (ordinary final response, last output before abnormal end)."""
    responses = extract_visible_model_responses(events, raw_events)
    if termination_class in TECHNICAL_PRIMARY_TERMINATIONS:
        final = next(
            (response for response in reversed(responses)
             if response.capture_status == "complete"
             and response.finish_reason != "tool_calls"),
            None,
        )
        return final, None
    return None, (responses[-1] if responses else None)


def _finish_reason(payload: object) -> str | None:
    if not isinstance(payload, dict):
        return None
    for key in ("finish_reason", "stopReason"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return None


def _capture_status(finish_reason: str | None) -> VisibleResponseCaptureStatus:
    if finish_reason in {"length", "max_tokens", "max_output_tokens", "aborted", "cancelled", "error"}:
        return "partial"
    if finish_reason in {"stop", "tool_calls", "end_turn", "completed", "complete", "success"}:
        return "complete"
    # An emitted message or a completed HTTP exchange without a native finish
    # state is useful captured text, but it does not prove ordinary completion.
    return "unknown"


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
