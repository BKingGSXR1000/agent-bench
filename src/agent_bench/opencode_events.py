"""Deterministic normalization of OpenCode 1.18.25 JSON run events."""

from __future__ import annotations

import hashlib
import shlex
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from agent_bench.events import DerivedEvent, RawEvent, normalize_raw_events
from agent_bench.models import JsonMapping, canonical_sha256

OPENCODE_NORMALIZER_NAME = "agent-bench-opencode"
OPENCODE_NORMALIZER_VERSION = "1.0.1"

_COMMON_TYPES = frozenset(
    {
        "run_start",
        "run_end",
        "llm_request",
        "llm_response",
        "reasoning",
        "tool_call_start",
        "tool_call_end",
        "file_read",
        "file_search",
        "file_edit",
        "file_write",
        "shell_command",
        "test_execution",
        "compaction_start",
        "compaction_end",
        "output_truncation",
        "context_overflow",
        "harness_error",
        "backend_error",
        "timeout",
        "process_termination",
    }
)
_TOOL_CATEGORIES = {
    "read": "read",
    "grep": "search",
    "glob": "search",
    "list": "search",
    "edit": "edit",
    "apply_patch": "edit",
    "patch": "edit",
    "write": "write",
    "bash": "shell",
    "shell": "shell",
}
OPENCODE_NORMALIZER_CONFIGURATION_DIGEST = canonical_sha256(
    {
        "normalizer": OPENCODE_NORMALIZER_NAME,
        "version": OPENCODE_NORMALIZER_VERSION,
        "common_types": sorted(_COMMON_TYPES),
        "tool_categories": _TOOL_CATEGORIES,
        "native_event_type": "opencode_event",
        "test_classifier": "opencode-shell-test-v1",
        "native_clock": "unix_epoch_milliseconds_relative_to_run_start_utc",
    }
)


class _OpenCodeTransformer:
    def __init__(self) -> None:
        self._task_start_epoch_ns: int | None = None
        self._workspace: str | None = None

    def __call__(self, raw: RawEvent) -> tuple[DerivedEvent, ...]:
        if raw.event_type == "run_start":
            self._task_start_epoch_ns = _datetime_epoch_ns(raw.timestamp_utc)
            isolated = raw.payload.get("isolated_paths")
            if isinstance(isolated, dict):
                workspace = isolated.get("workspace")
                if isinstance(workspace, str):
                    self._workspace = workspace
        if raw.event_type in _COMMON_TYPES:
            return (DerivedEvent(event_kind=raw.event_type, payload=raw.payload),)  # type: ignore[arg-type]
        if raw.event_type != "opencode_event":
            return ()
        native = raw.payload.get("native_event")
        if not isinstance(native, dict):
            return ()
        native_type = native.get("type")
        if native_type == "reasoning":
            return self._reasoning(native)
        if native_type == "tool_use":
            return self._tool(native)
        if native_type == "error":
            error = native.get("error")
            payload: JsonMapping = {
                "error_type": "opencode_native_error",
                "native_error": error,
                "session_id": native.get("sessionID"),
            }
            events = [DerivedEvent("harness_error", payload, "parsed")]
            if isinstance(error, dict) and error.get("name") == "MessageOutputLengthError":
                events.append(
                    DerivedEvent(
                        "output_truncation",
                        {"source": "opencode_native_error", "native_error": error},
                        "parsed",
                    )
                )
            return tuple(events)
        if native_type == "step_finish":
            part = native.get("part")
            if isinstance(part, dict) and part.get("reason") in {
                "length",
                "max_tokens",
                "max_output_tokens",
            }:
                return (
                    DerivedEvent(
                        "output_truncation",
                        {
                            "source": "opencode_step_finish",
                            "finish_reason": part.get("reason"),
                            "session_id": native.get("sessionID"),
                        },
                        "parsed",
                    ),
                )
        return ()

    def _reasoning(self, native: dict[str, object]) -> tuple[DerivedEvent, ...]:
        part = native.get("part")
        if not isinstance(part, dict):
            return ()
        text = part.get("text")
        if not isinstance(text, str):
            return ()
        payload: JsonMapping = {
            "text": text,
            "turn_id": part.get("messageID"),
            "session_id": native.get("sessionID"),
            "native_part_id": part.get("id"),
        }
        time_value = part.get("time")
        start = time_value.get("start") if isinstance(time_value, dict) else None
        return (self._at("reasoning", payload, start),)

    def _tool(self, native: dict[str, object]) -> tuple[DerivedEvent, ...]:
        part = native.get("part")
        if not isinstance(part, dict):
            return ()
        state = part.get("state")
        if not isinstance(state, dict) or state.get("status") not in {"completed", "error"}:
            return ()
        call_id = part.get("callID")
        tool_name = part.get("tool")
        if not isinstance(call_id, str) or not call_id or not isinstance(tool_name, str):
            return ()
        arguments = state.get("input")
        if not isinstance(arguments, dict):
            arguments = {}
        category = _TOOL_CATEGORIES.get(tool_name, "other")
        command = _string_field(arguments, "command")
        if category == "shell" and command is not None and is_test_command(command):
            category = "test"
        path = _worktree_relative_path(_tool_path(arguments), self._workspace)
        start_payload: JsonMapping = {
            "tool_call_id": call_id,
            "tool_name": tool_name,
            "category": category,
            "arguments": arguments,
            "turn_id": part.get("messageID"),
            "session_id": native.get("sessionID"),
        }
        if path is not None:
            start_payload["path"] = path
        if command is not None:
            start_payload["command"] = command
            start_payload["working_directory"] = self._workspace
            start_payload["environment"] = {}
            start_payload["uses_shell"] = True
        time_value = state.get("time")
        start_ms = time_value.get("start") if isinstance(time_value, dict) else None
        end_ms = time_value.get("end") if isinstance(time_value, dict) else None
        events = [self._at("tool_call_start", start_payload, start_ms)]
        operation_kind = {
            "read": "file_read",
            "search": "file_search",
            "edit": "file_edit",
            "write": "file_write",
            "shell": "shell_command",
            "test": "test_execution",
        }.get(category)
        if operation_kind is not None:
            operation_payload: JsonMapping = {
                "tool_call_id": call_id,
                "tool_name": tool_name,
            }
            if path is not None:
                operation_payload["path"] = path
            if command is not None:
                operation_payload["command"] = command
                operation_payload["working_directory"] = self._workspace
            events.append(self._at(operation_kind, operation_payload, start_ms))  # type: ignore[arg-type]
        outcome = "success" if state.get("status") == "completed" else "failure"
        end_payload: JsonMapping = {
            "tool_call_id": call_id,
            "outcome": outcome,
        }
        output = state.get("output")
        if isinstance(output, str):
            end_payload["output_sha256"] = hashlib.sha256(output.encode("utf-8")).hexdigest()
        error = state.get("error")
        if isinstance(error, str):
            end_payload["error"] = error
        events.append(self._at("tool_call_end", end_payload, end_ms))
        return tuple(events)

    def _at(
        self,
        kind: str,
        payload: JsonMapping,
        native_ms: object,
    ) -> DerivedEvent:
        if (
            isinstance(native_ms, (int, float))
            and not isinstance(native_ms, bool)
            and self._task_start_epoch_ns is not None
        ):
            epoch_ns = int(native_ms * 1_000_000)
            elapsed_ns = epoch_ns - self._task_start_epoch_ns
            if elapsed_ns >= 0:
                return DerivedEvent(
                    event_kind=kind,  # type: ignore[arg-type]
                    payload=payload,
                    confidence="parsed",
                    timestamp_utc=datetime.fromtimestamp(
                        epoch_ns / 1_000_000_000, tz=timezone.utc
                    ),
                    elapsed_ns=elapsed_ns,
                    clock_source="harness_wall_clock",
                )
        return DerivedEvent(
            event_kind=kind,  # type: ignore[arg-type]
            payload=payload,
            confidence="parsed",
        )


def normalize_opencode_events(raw_path: Path, normalized_path: Path) -> None:
    """Normalize common proxy/runner records and OpenCode's official JSON stream."""
    normalize_raw_events(
        raw_path,
        normalized_path,
        transformer=_OpenCodeTransformer(),
        normalizer_name=OPENCODE_NORMALIZER_NAME,
        normalizer_version=OPENCODE_NORMALIZER_VERSION,
        normalizer_configuration_digest=OPENCODE_NORMALIZER_CONFIGURATION_DIGEST,
    )


def is_test_command(command: str) -> bool:
    """Classify an observed shell command with fixed lexical rules."""
    try:
        words = shlex.split(command)
    except ValueError:
        words = command.split()
    lowered = [word.lower() for word in words]
    if not lowered:
        return False
    joined = " ".join(lowered)
    return (
        lowered[0] in {"pytest", "tox", "jest", "vitest"}
        or joined.startswith("python -m pytest")
        or joined.startswith("python3 -m pytest")
        or joined.startswith("npm test")
        or joined.startswith("npm run test")
        or joined.startswith("pnpm test")
        or joined.startswith("yarn test")
        or joined.startswith("bun test")
        or joined.startswith("cargo test")
        or joined.startswith("go test")
        or joined.startswith("dotnet test")
    )


def _datetime_epoch_ns(value: datetime) -> int:
    utc = value.astimezone(timezone.utc)
    seconds = int(utc.timestamp())
    return seconds * 1_000_000_000 + utc.microsecond * 1_000


def _tool_path(arguments: dict[str, object]) -> str | None:
    for key in ("filePath", "filepath", "file_path", "path"):
        value = arguments.get(key)
        if isinstance(value, str):
            return value
    return None


def _worktree_relative_path(value: str | None, workspace: str | None) -> str | None:
    """Make native absolute paths relative only when they are inside the worktree."""
    if value is None or workspace is None:
        return value
    path = PurePosixPath(value.replace("\\", "/"))
    root = PurePosixPath(workspace.replace("\\", "/"))
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _string_field(arguments: dict[str, object], key: str) -> str | None:
    value = arguments.get(key)
    return value if isinstance(value, str) else None
