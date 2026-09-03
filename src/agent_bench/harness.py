"""Minimal harness-adapter boundary used by the M3 runner."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from agent_bench.events import EventSource, RawEvent
from agent_bench.models import JsonMapping, RunDefinition, RunLimits


class EventSink(Protocol):
    """The only event-writing capability exposed to an adapter."""

    def emit(
        self,
        *,
        source: EventSource,
        event_type: str,
        payload: JsonMapping | None = None,
        timed: bool = True,
    ) -> RawEvent: ...


@dataclass(frozen=True)
class HarnessRunPaths:
    """Fresh paths allocated for exactly one harness invocation."""

    workspace: Path
    home: Path
    xdg_config_home: Path
    xdg_cache_home: Path
    xdg_data_home: Path
    harness_state: Path

    def environment(self) -> dict[str, str]:
        """Return the isolated path variables a process adapter would receive."""
        return {
            "HOME": str(self.home),
            "XDG_CONFIG_HOME": str(self.xdg_config_home),
            "XDG_CACHE_HOME": str(self.xdg_cache_home),
            "XDG_DATA_HOME": str(self.xdg_data_home),
            "AGENT_BENCH_HARNESS_STATE": str(self.harness_state),
        }


@dataclass(frozen=True)
class HarnessRunContext:
    """Harness-independent inputs for one adapter invocation."""

    run_definition: RunDefinition
    paths: HarnessRunPaths
    prompt_content: str
    events: EventSink
    limits: RunLimits
    cancellation: threading.Event


@dataclass(frozen=True)
class HarnessExecutionResult:
    """Minimal direct adapter outcome; formal classification belongs to M4."""

    completed_normally: bool
    output_truncated: bool = False


class HarnessAdapter(Protocol):
    """Small boundary that future process-backed adapters can implement."""

    @property
    def adapter_id(self) -> str: ...

    @property
    def adapter_version(self) -> str: ...

    def run(self, context: HarnessRunContext) -> HarnessExecutionResult: ...
