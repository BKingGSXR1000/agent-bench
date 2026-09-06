"""Focused deterministic completion-integrity unit coverage."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from agent_bench.completion_integrity import detect_model_self_reports, primary_eligibility
from agent_bench.events import NormalizedEvent, RawEvent, RawEventReference


def _response(text: str) -> NormalizedEvent:
    raw = RawEvent.create(raw_event_id="raw-1", run_id="run-1", sequence=1,
        timestamp_utc=datetime.now(timezone.utc), source="harness",
        event_type="captured", payload={})
    return NormalizedEvent.create(event_id="response-1", run_id="run-1",
        event_kind="llm_response", sequence=1, timestamp_utc=datetime.now(timezone.utc),
        raw_event_refs=(RawEventReference(raw_event_id=raw.raw_event_id, raw_sequence=1,
        raw_record_digest=raw.record_digest),), confidence="direct",
        payload={"assistant_text": text})


@pytest.mark.parametrize(("text", "category"), [
    ("I ran out of time before I could finish.", "time_limit"),
    ("The time limit was reached.", "time_limit"),
    ("I reached the token limit.", "token_limit"),
    ("The token budget is exhausted.", "token_limit"),
    ("I hit the context limit.", "context_limit"),
    ("Context limit was reached.", "context_limit"),
    ("I reached the step limit.", "step_or_turn_limit"),
    ("I couldn't finish the requested changes.", "general_incomplete"),
])
def test_explicit_model_self_reports_are_conservatively_categorized(text: str, category: str) -> None:
    matches = detect_model_self_reports((_response(text),), ())
    assert [item.category for item in matches] == [category]
    assert matches[0].source_event_id == "response-1"
    assert matches[0].rule_id.endswith("_v1")


@pytest.mark.parametrize("text", [
    "I fixed the token limit error.", "The code handles a context limit.",
    "Tests ensure we don't run out of time.", 'The file says "I ran out of time".',
])
def test_non_completion_language_and_quoted_source_are_not_self_reports(text: str) -> None:
    assert detect_model_self_reports((_response(text),), ()) == ()


def test_detector_only_reads_response_events() -> None:
    event = _response("all done")
    tool = event.model_copy(update={"event_id": "tool-1", "event_kind": "shell_command",
                                    "payload": {"assistant_text": "I ran out of time"}})
    assert detect_model_self_reports((tool,), ()) == ()


def test_primary_eligibility_keeps_technical_termination_separate() -> None:
    assert primary_eligibility(termination_class="success", functional_expected=True,
        functional_validation_status="pass", hard_gate_pass=True,
        self_reported_incomplete=False) == (True, ())
    eligible, reasons = primary_eligibility(termination_class="success", functional_expected=True,
        functional_validation_status="fail", hard_gate_pass=False,
        self_reported_incomplete=True)
    assert not eligible
    assert reasons == ("functional_failure", "hard_gate_failure", "self_reported_incomplete")
    assert primary_eligibility(termination_class="output_truncation", functional_expected=False,
        functional_validation_status="not_applicable", hard_gate_pass=None,
        self_reported_incomplete=False)[1] == ("objective_technical_incomplete",)
