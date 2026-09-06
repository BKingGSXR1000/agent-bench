"""Deterministic extraction of model reasoning/thinking blocks.

This module deliberately treats captured reasoning text as a separate evidence
stream from ordinary assistant output.  It never estimates tokens from text
length.  A caller may supply an exact tokenizer separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from agent_bench.events import NormalizedEvent, RawEvent


_NORMALIZED_NATIVE_TIMING = "exact_native_timing_from_normalized_evidence"
_RECOVERED_RAW_NATIVE_TIMING = "exact_native_timing_recovered_from_historical_raw_evidence"


@dataclass(frozen=True)
class ReasoningBlock:
    """One de-duplicated captured model reasoning/thinking block."""

    event_id: str
    sequence: int
    harness: str | None
    turn_id: str | None
    block_index: int
    text: str
    start_ms: int | float | None
    end_ms: int | float | None
    timing_provenance: str
    source_event_ids: tuple[str, ...]
    source_artifact_paths: tuple[str, ...] = ("raw/events.jsonl",)
    following_action_type: str | None = None
    precedes_tool: bool = False
    precedes_edit: bool = False
    ended_without_action: bool = False

    @property
    def characters(self) -> int:
        return len(self.text)

    @property
    def duration_seconds(self) -> float | None:
        if not isinstance(self.start_ms, (int, float)) or isinstance(self.start_ms, bool):
            return None
        if not isinstance(self.end_ms, (int, float)) or isinstance(self.end_ms, bool):
            return None
        duration = (self.end_ms - self.start_ms) / 1000
        return duration if duration >= 0 else None


def extract_reasoning_blocks(
    events: Iterable[NormalizedEvent], *, raw_events: Iterable[RawEvent] = (),
) -> tuple[ReasoningBlock, ...]:
    """Extract each normalized reasoning event exactly once.

    Harness normalizers collapse their native duplicate representations before
    this stage.  A final identity/text guard handles duplicated normalized
    events without conflating distinct equal-text turns.
    """
    ordered = sorted(events, key=lambda event: event.sequence)
    raw_by_id = {event.raw_event_id: event for event in raw_events}
    seen: set[tuple[str, str]] = set()
    drafts: list[ReasoningBlock] = []
    per_turn: dict[str, int] = {}
    for event in ordered:
        if event.event_kind != "reasoning":
            continue
        text = event.payload.get("text")
        if not isinstance(text, str) or not text:
            continue
        turn = _turn_identity(event)
        native = event.payload.get("native_part_id") or event.payload.get("message_id") or event.payload.get("response_id") or event.event_id
        identity = (str(native), text)
        if identity in seen:
            continue
        seen.add(identity)
        index = per_turn.get(turn or event.event_id, 0) + 1
        per_turn[turn or event.event_id] = index
        start_value = _number(event.payload.get("native_reasoning_start_ms"))
        end_value = _number(event.payload.get("native_reasoning_end_ms"))
        timing_provenance = "unavailable"
        source_event_ids = (event.event_id,)
        source_artifact_paths = ("normalized/events.jsonl",)
        if _valid_timing(start_value, end_value):
            timing_provenance = _NORMALIZED_NATIVE_TIMING
        else:
            recovered = _recover_historical_opencode_timing(event, raw_by_id)
            if recovered is not None:
                start_value, end_value, raw_event_id = recovered
                timing_provenance = _RECOVERED_RAW_NATIVE_TIMING
                source_event_ids = (event.event_id, raw_event_id)
                source_artifact_paths = ("raw/events.jsonl",)
        drafts.append(
            ReasoningBlock(
                event_id=event.event_id,
                sequence=event.sequence,
                harness=_string(event.payload.get("harness")),
                turn_id=turn,
                block_index=index,
                text=text,
                start_ms=start_value,
                end_ms=end_value,
                timing_provenance=timing_provenance,
                source_event_ids=source_event_ids,
                source_artifact_paths=source_artifact_paths,
            )
        )

    result: list[ReasoningBlock] = []
    for block in drafts:
        # The model's tool-call intent is a reasoning boundary even when a
        # timeout prevents the harness from executing that proposal.  Actual
        # executions remain separately represented by tool_call_start.
        following = next((event for event in ordered if event.sequence > block.sequence and event.event_kind in {"tool_call_start", "model_tool_call_observed"}), None)
        category = following.payload.get("category") if following is not None else None
        action = str(category) if isinstance(category, str) else None
        result.append(
            ReasoningBlock(
                **{**block.__dict__, "following_action_type": action, "precedes_tool": following is not None,
                   "precedes_edit": action in {"edit", "write"}, "ended_without_action": following is None}
            )
        )
    return tuple(result)


def _turn_identity(event: NormalizedEvent) -> str | None:
    for key in ("turn_id", "message_id", "response_id", "native_part_id"):
        value = event.payload.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _number(value: object) -> int | float | None:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _valid_timing(start: int | float | None, end: int | float | None) -> bool:
    return start is not None and end is not None and end >= start


def _recover_historical_opencode_timing(
    event: NormalizedEvent, raw_by_id: dict[str, RawEvent],
) -> tuple[int | float, int | float, str] | None:
    """Recover exact timing from a referenced legacy OpenCode raw record.

    Historical normalized streams predate the timing fields now emitted by the
    OpenCode normalizer.  This intentionally does not re-normalize or alter
    those streams: it accepts only the raw record already integrity-linked to
    the normalized event and requires all identity-bearing fields to agree.
    """
    native_part_id = _string(event.payload.get("native_part_id"))
    text = event.payload.get("text")
    turn_id = _turn_identity(event)
    if native_part_id is None or not isinstance(text, str):
        return None

    candidates: list[tuple[int | float, int | float, str]] = []
    for reference in event.raw_event_refs:
        raw = raw_by_id.get(reference.raw_event_id)
        if raw is None or raw.event_type != "opencode_event":
            continue
        native = raw.payload.get("native_event")
        if not isinstance(native, dict) or native.get("type") != "reasoning":
            continue
        part = native.get("part")
        if not isinstance(part, dict):
            continue
        if part.get("id") != native_part_id or part.get("text") != text:
            continue
        if turn_id is not None and part.get("messageID") != turn_id:
            continue
        time = part.get("time")
        if not isinstance(time, dict):
            continue
        start, end = _number(time.get("start")), _number(time.get("end"))
        if _valid_timing(start, end):
            candidates.append((start, end, raw.raw_event_id))

    # A missing or ambiguous match is unavailable rather than inferred.
    return candidates[0] if len(candidates) == 1 else None
