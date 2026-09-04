"""Deterministic normalization of Pi 0.84.4 native JSON session events."""

from __future__ import annotations

import hashlib
import shlex
from pathlib import Path, PurePosixPath

from agent_bench.events import DerivedEvent, RawEvent, normalize_raw_events
from agent_bench.models import JsonMapping, canonical_sha256

PI_NORMALIZER_NAME = "agent-bench-pi"
PI_NORMALIZER_VERSION = "1.0.1"
_COMMON_TYPES = frozenset({"run_start", "run_end", "llm_request", "llm_response", "reasoning", "tool_call_start", "tool_call_end", "file_read", "file_search", "file_edit", "file_write", "shell_command", "test_execution", "compaction_start", "compaction_end", "output_truncation", "context_overflow", "harness_error", "backend_error", "timeout", "process_termination"})
_TOOL_CATEGORIES = {"read": "read", "grep": "search", "find": "search", "ls": "search", "edit": "edit", "write": "write", "bash": "shell", "powershell": "shell"}
PI_NORMALIZER_CONFIGURATION_DIGEST = canonical_sha256({"normalizer": PI_NORMALIZER_NAME, "version": PI_NORMALIZER_VERSION, "common_types": sorted(_COMMON_TYPES), "tool_categories": _TOOL_CATEGORIES, "native_event_type": "pi_event", "test_classifier": "pi-shell-test-v1"})


class _PiTransformer:
    def __init__(self) -> None:
        self._workspace: str | None = None

    def __call__(self, raw: RawEvent) -> tuple[DerivedEvent, ...]:
        if raw.event_type == "run_start":
            isolated = raw.payload.get("isolated_paths")
            if isinstance(isolated, dict) and isinstance(isolated.get("workspace"), str): self._workspace = isolated["workspace"]
        if raw.event_type in _COMMON_TYPES:
            return (DerivedEvent(event_kind=raw.event_type, payload=raw.payload),)  # type: ignore[arg-type]
        if raw.event_type != "pi_event": return ()
        native = raw.payload.get("native_event")
        if not isinstance(native, dict): return ()
        kind = native.get("type")
        if kind == "message_end": return self._message_end(native)
        if kind == "tool_execution_start": return self._tool_start(native)
        if kind == "tool_execution_end": return self._tool_end(native)
        if kind in {"compaction_start", "compaction_end"}:
            return (DerivedEvent(event_kind=kind, payload={"native_event": native}, confidence="parsed"),)
        return ()

    def _message_end(self, native: dict[str, object]) -> tuple[DerivedEvent, ...]:
        message = native.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant": return ()
        events: list[DerivedEvent] = []
        content = message.get("content")
        if isinstance(content, list):
            for index, block in enumerate(content):
                if isinstance(block, dict) and block.get("type") == "thinking" and isinstance(block.get("thinking"), str) and block["thinking"]:
                    events.append(DerivedEvent("reasoning", {"text": block["thinking"], "content_index": index, "response_id": message.get("responseId"), "thinking_signature": block.get("thinkingSignature")}, "parsed"))
        stop_reason = message.get("stopReason")
        if stop_reason in {"length", "max_tokens", "max_output_tokens"}:
            events.append(DerivedEvent("output_truncation", {"source": "pi_message_end", "finish_reason": stop_reason}, "parsed"))
        if stop_reason in {"error", "aborted"}:
            events.append(DerivedEvent("harness_error", {"error_type": "pi_assistant_stop", "stop_reason": stop_reason, "message": message.get("errorMessage")}, "parsed"))
        return tuple(events)

    def _tool_start(self, native: dict[str, object]) -> tuple[DerivedEvent, ...]:
        call_id, name = native.get("toolCallId"), native.get("toolName")
        if not isinstance(call_id, str) or not isinstance(name, str): return ()
        arguments = native.get("args") if isinstance(native.get("args"), dict) else {}
        category = _TOOL_CATEGORIES.get(name, "other")
        command = _string_field(arguments, "command")
        if category == "shell" and command and is_test_command(command): category = "test"
        path = _relative_path(_tool_path(arguments), self._workspace)
        payload: JsonMapping = {
            "tool_call_id": call_id,
            "tool_name": name,
            "category": category,
            "arguments": arguments,
            # Pi exposes a live event named tool_execution_start, but does not
            # expose a native timestamp.  Agent Bench records when it observes
            # the line; that is not an exact execution-clock measurement.
            "timing_semantics": "tool_event_observed",
        }
        if path is not None: payload["path"] = path
        if command is not None: payload.update({"command": command, "working_directory": self._workspace, "environment": {}, "uses_shell": True})
        events = [DerivedEvent("tool_call_start", payload, "parsed")]
        operation = {"read": "file_read", "search": "file_search", "edit": "file_edit", "write": "file_write", "shell": "shell_command", "test": "test_execution"}.get(category)
        if operation:
            operation_payload: JsonMapping = {"tool_call_id": call_id, "tool_name": name}
            if path is not None: operation_payload["path"] = path
            if command is not None: operation_payload.update({"command": command, "working_directory": self._workspace})
            events.append(DerivedEvent(operation, operation_payload, "parsed"))  # type: ignore[arg-type]
        return tuple(events)

    def _tool_end(self, native: dict[str, object]) -> tuple[DerivedEvent, ...]:
        call_id = native.get("toolCallId")
        if not isinstance(call_id, str): return ()
        result = native.get("result")
        payload: JsonMapping = {
            "tool_call_id": call_id,
            "outcome": "failure" if native.get("isError") is True else "success",
            "timing_semantics": "tool_event_observed",
        }
        if result is not None:
            encoded = canonical_sha256(result)
            payload["result_sha256"] = encoded
        return (DerivedEvent("tool_call_end", payload, "parsed"),)


def normalize_pi_events(raw_path: Path, normalized_path: Path) -> None:
    normalize_raw_events(raw_path, normalized_path, transformer=_PiTransformer(), normalizer_name=PI_NORMALIZER_NAME, normalizer_version=PI_NORMALIZER_VERSION, normalizer_configuration_digest=PI_NORMALIZER_CONFIGURATION_DIGEST)


def is_test_command(command: str) -> bool:
    try: words = shlex.split(command)
    except ValueError: words = command.split()
    lowered = [word.lower() for word in words]
    if not lowered: return False
    joined = " ".join(lowered)
    return lowered[0] in {"pytest", "tox", "jest", "vitest"} or any(joined.startswith(prefix) for prefix in ("python -m pytest", "python3 -m pytest", "npm test", "npm run test", "pnpm test", "yarn test", "bun test", "cargo test", "go test", "dotnet test"))


def _tool_path(arguments: dict[str, object]) -> str | None:
    for key in ("path", "filePath", "filepath", "file_path"):
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
