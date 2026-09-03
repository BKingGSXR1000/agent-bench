"""Deterministic, test-only harness scenarios with no LLM dependency."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from agent_bench.harness import (
    HarnessExecutionResult,
    HarnessRunContext,
)

FakeScenario = Literal[
    "metrics",
    "success",
    "no_change",
    "failed_tool",
    "timeout",
    "crash",
    "output_truncation",
    "reasoning_without_action",
]
FAKE_SCENARIOS: tuple[FakeScenario, ...] = (
    "metrics",
    "success",
    "no_change",
    "failed_tool",
    "timeout",
    "crash",
    "output_truncation",
    "reasoning_without_action",
)


class FakeHarnessCrash(RuntimeError):
    """The intentional exception used by the deterministic crash scenario."""


class FakeHarness:
    """Exercise the adapter/runner boundary without invoking a real harness."""

    adapter_id = "fake-harness"
    adapter_version = "1.0.0"

    def __init__(self, scenario: FakeScenario) -> None:
        if scenario not in FAKE_SCENARIOS:
            raise ValueError(f"unsupported FakeHarness scenario: {scenario}")
        self.scenario = scenario

    def run(self, context: HarnessRunContext) -> HarnessExecutionResult:
        self._record_isolation(context)
        if self.scenario == "metrics":
            return self._metrics(context)
        if self.scenario == "success":
            return self._success(context)
        if self.scenario == "no_change":
            return self._no_change(context)
        if self.scenario == "failed_tool":
            return self._failed_tool(context)
        if self.scenario == "timeout":
            return self._timeout(context)
        if self.scenario == "crash":
            return self._crash(context)
        if self.scenario == "output_truncation":
            return self._output_truncation(context)
        return self._reasoning_without_action(context)

    def _record_isolation(self, context: HarnessRunContext) -> None:
        environment = context.paths.environment()
        if not all(Path(path).is_dir() for path in environment.values()):
            raise RuntimeError("FakeHarness received incomplete isolated paths")
        state = {
            "schema_version": "1.0.0",
            "scenario": self.scenario,
            "prompt_sha256": hashlib.sha256(
                context.prompt_content.encode("utf-8")
            ).hexdigest(),
        }
        (context.paths.harness_state / "session.json").write_text(
            json.dumps(state, separators=(",", ":"), sort_keys=True) + "\n",
            encoding="utf-8",
        )
        context.events.emit(
            source="harness",
            event_type="harness_environment",
            payload={
                "environment": environment,
                "all_paths_existed": True,
                "fresh_session": True,
            },
        )

    def _success(self, context: HarnessRunContext) -> HarnessExecutionResult:
        self._reason(context, "Inspect the baseline before making deterministic changes.")
        self._read_tracked(context, call_id="read-001")

        self._tool_start(
            context,
            "edit-001",
            "edit",
            "edit",
            arguments={"path": "tracked.txt"},
            path="tracked.txt",
        )
        (context.paths.workspace / "tracked.txt").write_text(
            "updated by FakeHarness\n", encoding="utf-8"
        )
        context.events.emit(
            source="harness",
            event_type="file_edit",
            payload={"tool_call_id": "edit-001", "path": "tracked.txt"},
        )
        self._tool_end(context, "edit-001", "success")

        self._tool_start(
            context,
            "write-001",
            "write",
            "write",
            arguments={"path": "fake-created.txt"},
            path="fake-created.txt",
        )
        (context.paths.workspace / "fake-created.txt").write_text(
            "created by FakeHarness\n", encoding="utf-8"
        )
        context.events.emit(
            source="harness",
            event_type="file_write",
            payload={"tool_call_id": "write-001", "path": "fake-created.txt"},
        )
        self._tool_end(context, "write-001", "success")
        return HarnessExecutionResult(completed_normally=True)

    def _no_change(self, context: HarnessRunContext) -> HarnessExecutionResult:
        self._reason(context, "Inspect the project and finish without changes.")
        self._read_tracked(context, call_id="read-001")
        return HarnessExecutionResult(completed_normally=True)

    def _failed_tool(self, context: HarnessRunContext) -> HarnessExecutionResult:
        self._reason(context, "Try one invalid operation, then recover with a read.")
        self._tool_start(context, "edit-failed-001", "edit", "edit")
        self._tool_end(
            context,
            "edit-failed-001",
            "failure",
            error="deterministic fake tool failure",
        )
        self._read_tracked(context, call_id="read-recovery-001")
        return HarnessExecutionResult(completed_normally=True)

    def _timeout(self, context: HarnessRunContext) -> HarnessExecutionResult:
        self._reason(context, "Wait for the runner's deterministic cancellation signal.")
        context.cancellation.wait()
        return HarnessExecutionResult(completed_normally=False)

    def _crash(self, context: HarnessRunContext) -> HarnessExecutionResult:
        self._reason(context, "Raise the deterministic crash after recording evidence.")
        raise FakeHarnessCrash("deterministic FakeHarness crash")

    def _output_truncation(
        self, context: HarnessRunContext
    ) -> HarnessExecutionResult:
        context.events.emit(
            source="harness",
            event_type="llm_response",
            payload={
                "response_id": "fake-response-001",
                "outcome": "truncated",
                "finish_reason": "length",
                "token_counts": "unavailable_not_applicable",
            },
        )
        context.events.emit(
            source="harness",
            event_type="output_truncation",
            payload={
                "response_id": "fake-response-001",
                "finish_reason": "length",
            },
        )
        return HarnessExecutionResult(
            completed_normally=True,
            output_truncated=True,
        )

    def _reasoning_without_action(
        self, context: HarnessRunContext
    ) -> HarnessExecutionResult:
        self._reason(context, "Produce a reasoning-only turn with no tool action.")
        context.events.emit(
            source="harness",
            event_type="llm_response",
            payload={
                "response_id": "fake-response-001",
                "turn_id": "turn-001",
                "outcome": "success",
                "finish_reason": "stop",
                "token_counts": "unavailable_not_applicable",
                "visible_answer_present": False,
            },
        )
        return HarnessExecutionResult(completed_normally=True)

    def _metrics(self, context: HarnessRunContext) -> HarnessExecutionResult:
        """Emit rich, directly sourced synthetic evidence for M4 tests."""
        self._llm_exchange(context, 1, 100, 20, 8, 12, "turn-001")
        self._reason(context, "Read the baseline twice.", turn_id="turn-001")
        self._read_tracked(context, call_id="read-001", turn_id="turn-001")
        self._read_tracked(context, call_id="read-002", turn_id="turn-001")

        self._llm_exchange(context, 2, 180, 30, 10, 20, "turn-002")
        self._tool_start(
            context,
            "edit-001",
            "edit",
            "edit",
            arguments={"path": "tracked.txt", "content": "metrics update\n"},
            path="tracked.txt",
            turn_id="turn-002",
        )
        (context.paths.workspace / "tracked.txt").write_text(
            "metrics update\n", encoding="utf-8"
        )
        (context.paths.workspace / "delete-me.txt").unlink()
        generated_test = context.paths.workspace / "tests/test_generated.py"
        generated_test.parent.mkdir()
        generated_test.write_text("def test_generated():\n    assert True\n", encoding="utf-8")
        (context.paths.workspace / "generated.bin").write_bytes(b"\x00fake-binary\n")
        context.events.emit(
            source="harness",
            event_type="file_edit",
            payload={"tool_call_id": "edit-001", "path": "tracked.txt"},
        )
        self._tool_end(context, "edit-001", "success")
        self._read_tracked(context, call_id="read-003", turn_id="turn-002")

        context.events.emit(
            source="harness",
            event_type="compaction_start",
            payload={
                "compaction_id": "compact-001",
                "before_context_tokens": 180,
                "after_context_tokens": 80,
                "configured_max_context_tokens": 1000,
                "token_source": "api_exact",
            },
        )
        context.events.emit(
            source="harness",
            event_type="compaction_end",
            payload={"compaction_id": "compact-001", "outcome": "success"},
        )
        self._llm_exchange(context, 3, 90, 15, 5, 10, "turn-003")
        for call_id in ("search-001", "search-002"):
            self._tool_start(
                context,
                call_id,
                "search",
                "search",
                arguments={"query": "tracked"},
                turn_id="turn-003",
            )
            context.events.emit(
                source="harness",
                event_type="file_search",
                payload={"tool_call_id": call_id, "query": "tracked"},
            )
            self._tool_end(context, call_id, "success")
        for call_id in ("shell-001", "shell-002"):
            self._tool_start(
                context,
                call_id,
                "shell",
                "shell",
                arguments={"command": ["pwd"]},
                command=["pwd"],
                working_directory=".",
                turn_id="turn-003",
            )
            context.events.emit(
                source="harness",
                event_type="shell_command",
                payload={"tool_call_id": call_id, "command": ["pwd"]},
            )
            self._tool_end(context, call_id, "success")
        self._tool_start(
            context,
            "test-001",
            "pytest",
            "test",
            arguments={"command": ["pytest", "-q"]},
            command=["pytest", "-q"],
            working_directory=".",
            uses_shell=True,
            turn_id="turn-003",
        )
        context.events.emit(
            source="harness",
            event_type="test_execution",
            payload={"tool_call_id": "test-001", "command": ["pytest", "-q"]},
        )
        self._tool_end(context, "test-001", "success")

        self._llm_exchange(context, 4, 120, 5, 5, 0, "turn-004", visible=False)
        self._reason(context, "A final reasoning-only turn.", turn_id="turn-004")
        return HarnessExecutionResult(completed_normally=True)

    @staticmethod
    def _reason(
        context: HarnessRunContext,
        text: str,
        *,
        turn_id: str = "turn-001",
    ) -> None:
        context.events.emit(
            source="harness",
            event_type="reasoning",
            payload={"turn_id": turn_id, "text": text},
        )

    @staticmethod
    def _read_tracked(
        context: HarnessRunContext,
        *,
        call_id: str,
        turn_id: str = "turn-001",
    ) -> None:
        FakeHarness._tool_start(
            context,
            call_id,
            "read",
            "read",
            arguments={"path": "tracked.txt"},
            path="tracked.txt",
            turn_id=turn_id,
        )
        content = (context.paths.workspace / "tracked.txt").read_bytes()
        context.events.emit(
            source="harness",
            event_type="file_read",
            payload={
                "tool_call_id": call_id,
                "path": "tracked.txt",
                "content_sha256": hashlib.sha256(content).hexdigest(),
            },
        )
        FakeHarness._tool_end(context, call_id, "success")

    @staticmethod
    def _tool_start(
        context: HarnessRunContext,
        call_id: str,
        tool_name: str,
        category: str,
        **details: object,
    ) -> None:
        context.events.emit(
            source="harness",
            event_type="tool_call_start",
            payload={  # type: ignore[arg-type]
                "tool_call_id": call_id,
                "tool_name": tool_name,
                "category": category,
                **details,
            },
        )

    @staticmethod
    def _tool_end(
        context: HarnessRunContext,
        call_id: str,
        outcome: str,
        *,
        error: str | None = None,
    ) -> None:
        payload = {"tool_call_id": call_id, "outcome": outcome}
        if error is not None:
            payload["error"] = error
        context.events.emit(
            source="harness",
            event_type="tool_call_end",
            payload=payload,
        )

    @staticmethod
    def _llm_exchange(
        context: HarnessRunContext,
        index: int,
        context_tokens: int,
        output_tokens: int,
        reasoning_tokens: int,
        visible_tokens: int,
        turn_id: str,
        *,
        visible: bool = True,
    ) -> None:
        request_id = f"request-{index:03d}"
        context.events.emit(
            source="proxy",
            event_type="llm_request",
            payload={
                "request_id": request_id,
                "request_index": index,
                "turn_id": turn_id,
                "context_tokens": context_tokens,
                "configured_max_context_tokens": 1000,
                "token_source": "api_exact",
            },
        )
        context.events.emit(
            source="proxy",
            event_type="llm_response",
            payload={
                "request_id": request_id,
                "response_id": f"response-{index:03d}",
                "turn_id": turn_id,
                "outcome": "success",
                "output_tokens": output_tokens,
                "reasoning_tokens": reasoning_tokens,
                "visible_answer_tokens": visible_tokens,
                "reasoning_is_subset_of_output": True,
                "visible_answer_present": visible,
                "token_source": "api_exact",
            },
        )
