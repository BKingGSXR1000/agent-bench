"""Versioned capture capabilities and deterministic history validation."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import Field

from agent_bench.models import Identifier, PersistedModel

CaptureMethod = Literal[
    "api_exact",
    "proxy_exact",
    "harness_exact",
    "reconstructed",
    "unavailable",
]


class CaptureCapabilities(PersistedModel):
    """Declare which observations a run can measure and by which boundary."""

    capability_id: Identifier
    backend_id: Identifier | None = None
    harness_id: str | None = None
    raw_request_payload: CaptureMethod
    raw_response_payload: CaptureMethod
    request_generation_parameters: CaptureMethod
    input_token_usage: CaptureMethod
    output_token_usage: CaptureMethod
    reasoning_content: CaptureMethod
    reasoning_token_count: CaptureMethod
    context_token_count: CaptureMethod
    finish_reason: CaptureMethod
    tool_calls: CaptureMethod
    tool_results: CaptureMethod
    compaction_events: CaptureMethod
    session_identity: CaptureMethod = "unavailable"
    serialized_prompt_history_validation: CaptureMethod
    empty_historical_think_block_detection: CaptureMethod
    notes: tuple[str, ...] = ()

    def supports_complete_llm_exchange_capture(self) -> bool:
        """Return whether request and response bodies are both captured exactly."""
        return (
            self.raw_request_payload != "unavailable"
            and self.raw_response_payload != "unavailable"
        )


def fixed_proxy_capture_capabilities() -> CaptureCapabilities:
    """Capabilities implemented at the M5 transparent proxy boundary."""
    return CaptureCapabilities(
        capability_id="llamacpp-proxy-m5-v1",
        backend_id="llamacpp-qwen38-agent-bench-v1",
        raw_request_payload="proxy_exact",
        raw_response_payload="proxy_exact",
        request_generation_parameters="proxy_exact",
        input_token_usage="api_exact",
        output_token_usage="api_exact",
        reasoning_content="proxy_exact",
        reasoning_token_count="api_exact",
        context_token_count="api_exact",
        finish_reason="proxy_exact",
        tool_calls="proxy_exact",
        tool_results="proxy_exact",
        compaction_events="unavailable",
        session_identity="unavailable",
        serialized_prompt_history_validation="unavailable",
        empty_historical_think_block_detection="unavailable",
        notes=(
            "Token counts are exact only when llama.cpp emits compatible usage fields.",
            "Reasoning token counts are exact only when llama.cpp emits a compatible usage detail field.",
            "The proxy captures message history but not llama.cpp's rendered Jinja prompt.",
            "Real-harness empty-history validation remains pending M6-M8 rendered-prompt evidence.",
        ),
    )


def fake_harness_capture_capabilities() -> CaptureCapabilities:
    """Capabilities of the deterministic FakeHarness fixture."""
    return CaptureCapabilities(
        capability_id="fake-harness-m3-v1",
        harness_id="fake-harness",
        raw_request_payload="harness_exact",
        raw_response_payload="harness_exact",
        request_generation_parameters="harness_exact",
        input_token_usage="harness_exact",
        output_token_usage="harness_exact",
        reasoning_content="harness_exact",
        reasoning_token_count="harness_exact",
        context_token_count="harness_exact",
        finish_reason="harness_exact",
        tool_calls="harness_exact",
        tool_results="harness_exact",
        compaction_events="harness_exact",
        session_identity="harness_exact",
        serialized_prompt_history_validation="unavailable",
        empty_historical_think_block_detection="unavailable",
        notes=("Synthetic fixture observations are exact only within FakeHarness.",),
    )


class EmptyThinkOccurrence(PersistedModel):
    """One deterministic empty closed thinking block in captured history."""

    location: str = Field(min_length=1)
    matched_text: str


class EmptyThinkValidation(PersistedModel):
    """Metric-ready result from inspecting supplied serialized history evidence."""

    validator_id: Literal["empty-historical-think-block-v1"] = (
        "empty-historical-think-block-v1"
    )
    evidence_kind: Literal["serialized_prompt", "request_messages"]
    empty_think_blocks_in_history: int = Field(ge=0)
    occurrences: tuple[EmptyThinkOccurrence, ...] = ()


_EMPTY_THINK = re.compile(r"<think>[\t\r\n ]*</think>")


def detect_empty_historical_think_blocks(
    evidence: str | list[object] | tuple[object, ...],
) -> EmptyThinkValidation:
    """Count closed empty think blocks without interpreting reasoning semantics.

    An open generation prefix such as ``<think>`` cannot match this validator.
    Message evidence is historical by definition because it is already present in
    an inbound request. Rendered prompt text can be supplied later by a backend
    capture source without changing the algorithm.
    """
    occurrences: list[EmptyThinkOccurrence] = []
    if isinstance(evidence, str):
        _scan_text(evidence, "serialized_prompt", occurrences)
        kind: Literal["serialized_prompt", "request_messages"] = "serialized_prompt"
    else:
        kind = "request_messages"
        for index, message in enumerate(evidence):
            _scan_message_value(message, f"messages[{index}]", occurrences)
    return EmptyThinkValidation(
        evidence_kind=kind,
        empty_think_blocks_in_history=len(occurrences),
        occurrences=tuple(occurrences),
    )


def _scan_message_value(
    value: object,
    location: str,
    occurrences: list[EmptyThinkOccurrence],
) -> None:
    if isinstance(value, str):
        _scan_text(value, location, occurrences)
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            if key in {"content", "reasoning_content", "text"}:
                _scan_message_value(value[key], f"{location}.{key}", occurrences)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _scan_message_value(item, f"{location}[{index}]", occurrences)


def _scan_text(
    text: str,
    location: str,
    occurrences: list[EmptyThinkOccurrence],
) -> None:
    for match in _EMPTY_THINK.finditer(text):
        occurrences.append(
            EmptyThinkOccurrence(location=location, matched_text=match.group(0))
        )
