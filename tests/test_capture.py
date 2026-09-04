from __future__ import annotations

import json

from pydantic import ValidationError

from agent_bench.capture import (
    CaptureCapabilities,
    detect_empty_historical_think_blocks,
    fixed_proxy_capture_capabilities,
)


def test_capture_capabilities_are_versioned_frozen_and_honest() -> None:
    capabilities = fixed_proxy_capture_capabilities()

    assert capabilities.schema_version == "1.0.0"
    assert capabilities.raw_request_payload == "proxy_exact"
    assert capabilities.context_token_count == "api_exact"
    assert capabilities.reasoning_token_count == "api_exact"
    assert capabilities.compaction_events == "unavailable"
    assert capabilities.serialized_prompt_history_validation == "unavailable"
    assert capabilities.empty_historical_think_block_detection == "unavailable"
    assert capabilities.definition_digest
    assert json.loads(capabilities.model_dump_json())["capability_id"] == "llamacpp-proxy-m5-v1"
    try:
        capabilities.raw_request_payload = "unavailable"  # type: ignore[misc]
    except ValidationError:
        pass
    else:
        raise AssertionError("CaptureCapabilities must be immutable")


def test_empty_history_think_detector_counts_only_closed_empty_blocks() -> None:
    validation = detect_empty_historical_think_blocks(
        [
            {"role": "assistant", "content": "<think>\n\n</think>\nanswer"},
            {"role": "assistant", "content": "<think>real reasoning</think>"},
            {"role": "assistant", "content": "<think>"},
            {"role": "tool", "content": [{"type": "text", "text": "ok"}]},
        ]
    )

    assert validation.empty_think_blocks_in_history == 1
    assert validation.occurrences[0].location == "messages[0].content"


def test_empty_history_think_detector_accepts_rendered_prompt_fixture() -> None:
    clean = detect_empty_historical_think_blocks(
        "<|im_start|>assistant\n<think>careful</think>\nanswer"
    )
    broken = detect_empty_historical_think_blocks(
        "<|im_start|>assistant\n<think>  \n </think>\nanswer"
    )

    assert clean.empty_think_blocks_in_history == 0
    assert broken.empty_think_blocks_in_history == 1
    assert clean.evidence_kind == "serialized_prompt"


def test_capability_values_are_restricted() -> None:
    data = fixed_proxy_capture_capabilities().model_dump(
        exclude={"definition_digest"}
    )
    data["raw_request_payload"] = "estimated"
    try:
        CaptureCapabilities.model_validate(data)
    except ValidationError:
        pass
    else:
        raise AssertionError("unsupported capture method was accepted")
