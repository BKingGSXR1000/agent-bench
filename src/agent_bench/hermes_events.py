"""Deterministic normalization of Hermes 0.21.0 persisted session evidence."""

from __future__ import annotations

import json
import shlex
from pathlib import Path, PurePosixPath

from agent_bench.events import DerivedEvent, RawEvent, normalize_raw_events
from agent_bench.models import JsonMapping, canonical_sha256

HERMES_NORMALIZER_NAME = "agent-bench-hermes"
HERMES_NORMALIZER_VERSION = "1.0.3"
_COMMON_TYPES = frozenset({"run_start", "run_end", "llm_request", "llm_response", "reasoning", "tool_call_start", "tool_call_end", "file_read", "file_search", "file_edit", "file_write", "shell_command", "test_execution", "compaction_start", "compaction_end", "output_truncation", "context_overflow", "harness_error", "backend_error", "timeout", "process_termination"})
_TOOL_CATEGORIES = {"read_file": "read", "read": "read", "grep": "search", "search": "search", "search_files": "search", "find": "search", "glob": "search", "list_files": "search", "write_file": "write", "write": "write", "edit_file": "edit", "edit": "edit", "patch": "edit", "apply_patch": "edit", "terminal": "shell", "terminal_tool": "shell", "bash": "shell", "shell": "shell", "execute_command": "shell"}
HERMES_NORMALIZER_CONFIGURATION_DIGEST = canonical_sha256({"normalizer": HERMES_NORMALIZER_NAME, "version": HERMES_NORMALIZER_VERSION, "common_types": sorted(_COMMON_TYPES), "tool_categories": _TOOL_CATEGORIES, "native_event_type": "hermes_session_message", "test_classifier": "hermes-shell-test-v1"})


class _HermesTransformer:
    def __init__(self) -> None:
        self._workspace: str | None = None
        self._started: set[str] = set()

    def __call__(self, raw: RawEvent) -> tuple[DerivedEvent, ...]:
        if raw.event_type == "run_start":
            paths = raw.payload.get("isolated_paths")
            if isinstance(paths, dict) and isinstance(paths.get("workspace"), str): self._workspace = paths["workspace"]
        if raw.event_type in _COMMON_TYPES:
            return (DerivedEvent(event_kind=raw.event_type, payload=raw.payload),)  # type: ignore[arg-type]
        if raw.event_type == "hermes_session_compaction":
            return (DerivedEvent("compaction_end", {"source": "hermes_sqlite_compacted_message", "native_event": raw.payload.get("native_event")}, "parsed"),)
        if raw.event_type != "hermes_session_message": return ()
        native = raw.payload.get("native_event")
        if not isinstance(native, dict): return ()
        return self._message(native)

    def _message(self, native: dict[str, object]) -> tuple[DerivedEvent, ...]:
        events: list[DerivedEvent] = []
        role = native.get("role")
        if role == "assistant":
            # Hermes persists the same thought in both fields on some builds.
            # reasoning_content is authoritative; reasoning is a fallback only.
            reasoning = native.get("reasoning_content")
            reasoning_field = "reasoning_content"
            if not isinstance(reasoning, str) or not reasoning:
                reasoning = native.get("reasoning")
                reasoning_field = "reasoning"
            if isinstance(reasoning, str) and reasoning:
                events.append(DerivedEvent("reasoning", {"text": reasoning, "message_id": native.get("id"), "source_field": reasoning_field, "timing_provenance": "unavailable"}, "parsed"))
            for call in _tool_calls(native.get("tool_calls")):
                call_id, name, arguments = _call_parts(call)
                if not call_id or not name: continue
                events.extend(
                    self._tool_start(
                        call_id,
                        name,
                        arguments,
                        native.get("timestamp"),
                    )
                )
            finish = native.get("finish_reason")
            if finish in {"length", "max_tokens", "max_output_tokens"}:
                events.append(DerivedEvent("output_truncation", {"source": "hermes_session", "finish_reason": finish}, "parsed"))
        if role == "tool":
            call_id = native.get("tool_call_id")
            if isinstance(call_id, str) and call_id:
                payload: JsonMapping = {
                    "tool_call_id": call_id,
                    "outcome": "failure" if native.get("effect_disposition") in {"error", "failure"} else "success",
                    "result_sha256": canonical_sha256(native.get("content")),
                    # Hermes 0.21.0 exports these SQLite messages after the
                    # one-shot process exits.  This is a recorded result
                    # timestamp, not an execution-end timestamp.
                    "timing_semantics": "tool_result_recorded_then_exported",
                }
                timestamp = native.get("timestamp")
                if isinstance(timestamp, (int, float)) and not isinstance(timestamp, bool):
                    payload["native_message_timestamp_seconds"] = timestamp
                events.append(DerivedEvent("tool_call_end", payload, "parsed"))
        return tuple(events)

    def _tool_start(
        self,
        call_id: str,
        name: str,
        arguments: dict[str, object],
        native_timestamp: object,
    ) -> tuple[DerivedEvent, ...]:
        if call_id in self._started: return ()
        self._started.add(call_id)
        category = _TOOL_CATEGORIES.get(name, "other")
        command = _string_field(arguments, "command") or _string_field(arguments, "cmd")
        if category == "shell" and command and is_test_command(command): category = "test"
        path = _relative_path(_tool_path(arguments), self._workspace)
        payload: JsonMapping = {
            "tool_call_id": call_id,
            "tool_name": name,
            "category": category,
            "arguments": arguments,
            # The assistant message contains a requested tool call, but the
            # persisted Hermes SQLite export is emitted only after process
            # completion.  Its capture timestamp must not be treated as a
            # harness execution-start time.
            "timing_semantics": "tool_call_recorded_then_exported",
        }
        if isinstance(native_timestamp, (int, float)) and not isinstance(native_timestamp, bool):
            payload["native_message_timestamp_seconds"] = native_timestamp
        if path is not None: payload["path"] = path
        if command is not None: payload.update({"command": command, "working_directory": self._workspace, "environment": {}, "uses_shell": True})
        events = [DerivedEvent("tool_call_start", payload, "parsed")]
        operation = {"read": "file_read", "search": "file_search", "edit": "file_edit", "write": "file_write", "shell": "shell_command", "test": "test_execution"}.get(category)
        if operation:
            detail: JsonMapping = {"tool_call_id": call_id, "tool_name": name}
            if path is not None: detail["path"] = path
            if command is not None: detail.update({"command": command, "working_directory": self._workspace})
            events.append(DerivedEvent(operation, detail, "parsed"))  # type: ignore[arg-type]
        return tuple(events)


def normalize_hermes_events(raw_path: Path, normalized_path: Path) -> None:
    normalize_raw_events(raw_path, normalized_path, transformer=_HermesTransformer(), normalizer_name=HERMES_NORMALIZER_NAME, normalizer_version=HERMES_NORMALIZER_VERSION, normalizer_configuration_digest=HERMES_NORMALIZER_CONFIGURATION_DIGEST)


def is_test_command(command: str) -> bool:
    try: words = shlex.split(command)
    except ValueError: words = command.split()
    lowered = [word.lower() for word in words]
    if not lowered: return False
    joined = " ".join(lowered)
    return lowered[0] in {"pytest", "tox", "jest", "vitest"} or any(joined.startswith(prefix) for prefix in ("python -m pytest", "python3 -m pytest", "npm test", "npm run test", "pnpm test", "yarn test", "bun test", "cargo test", "go test", "dotnet test"))


def _tool_calls(value: object) -> list[object]:
    if isinstance(value, str):
        try: value = json.loads(value)
        except json.JSONDecodeError: return []
    return value if isinstance(value, list) else []


def _call_parts(value: object) -> tuple[str | None, str | None, dict[str, object]]:
    if not isinstance(value, dict): return None, None, {}
    call_id = value.get("id") or value.get("tool_call_id")
    name = value.get("name") or value.get("tool_name")
    function = value.get("function")
    if isinstance(function, dict):
        name = name or function.get("name")
        arguments = function.get("arguments")
    else: arguments = value.get("arguments") or value.get("args")
    if isinstance(arguments, str):
        try: arguments = json.loads(arguments)
        except json.JSONDecodeError: arguments = {}
    return (call_id if isinstance(call_id, str) else None, name if isinstance(name, str) else None, arguments if isinstance(arguments, dict) else {})


def _tool_path(arguments: dict[str, object]) -> str | None:
    for key in ("path", "file_path", "filePath", "filepath"):
        if isinstance(arguments.get(key), str): return arguments[key]  # type: ignore[return-value]
    return None


def _relative_path(value: str | None, workspace: str | None) -> str | None:
    if value is None or workspace is None: return value
    path, root = PurePosixPath(value.replace("\\", "/")), PurePosixPath(workspace.replace("\\", "/"))
    if not path.is_absolute(): return path.as_posix()
    try: return path.relative_to(root).as_posix()
    except ValueError: return path.as_posix()


def _string_field(arguments: dict[str, object], key: str) -> str | None:
    return arguments[key] if isinstance(arguments.get(key), str) else None  # type: ignore[return-value]
