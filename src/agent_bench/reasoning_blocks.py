"""Deterministic extraction of model reasoning/thinking blocks.

This module deliberately treats captured reasoning text as a separate evidence
stream from ordinary assistant output.  It never estimates tokens from text
length.  A caller may supply an exact tokenizer separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from agent_bench.events import NormalizedEvent


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


def extract_reasoning_blocks(events: Iterable[NormalizedEvent]) -> tuple[ReasoningBlock, ...]:
    """Extract each normalized reasoning event exactly once.

    Harness normalizers collapse their native duplicate representations before
    this stage.  A final identity/text guard handles duplicated normalized
    events without conflating distinct equal-text turns.
    """
    ordered = sorted(events, key=lambda event: event.sequence)
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
        start = event.payload.get("native_reasoning_start_ms")
        end = event.payload.get("native_reasoning_end_ms")
        start_value = start if isinstance(start, (int, float)) and not isinstance(start, bool) else None
        end_value = end if isinstance(end, (int, float)) and not isinstance(end, bool) else None
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
                timing_provenance=("exact_native" if start_value is not None and end_value is not None else "unavailable"),
                source_event_ids=(event.event_id,),
            )
        )

    result: list[ReasoningBlock] = []
    for block in drafts:
        following = next((event for event in ordered if event.sequence > block.sequence and event.event_kind == "tool_call_start"), None)
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
