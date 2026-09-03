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
    "success",
    "no_change",
    "failed_tool",
    "timeout",
    "crash",
    "output_truncation",
    "reasoning_without_action",
]
FAKE_SCENARIOS: tuple[FakeScenario, ...] = (
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

        self._tool_start(context, "edit-001", "edit", "edit")
        (context.paths.workspace / "tracked.txt").write_text(
            "updated by FakeHarness\n", encoding="utf-8"
        )
        context.events.emit(
            source="harness",
            event_type="file_edit",
            payload={"tool_call_id": "edit-001", "path": "tracked.txt"},
        )
        self._tool_end(context, "edit-001", "success")

        self._tool_start(context, "write-001", "write", "write")
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
            },
        )
        return HarnessExecutionResult(completed_normally=True)

    @staticmethod
    def _reason(context: HarnessRunContext, text: str) -> None:
        context.events.emit(
            source="harness",
            event_type="reasoning",
            payload={"turn_id": "turn-001", "text": text},
        )

    @staticmethod
    def _read_tracked(context: HarnessRunContext, *, call_id: str) -> None:
        FakeHarness._tool_start(context, call_id, "read", "read")
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
    ) -> None:
        context.events.emit(
            source="harness",
            event_type="tool_call_start",
            payload={
                "tool_call_id": call_id,
                "tool_name": tool_name,
                "category": category,
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
