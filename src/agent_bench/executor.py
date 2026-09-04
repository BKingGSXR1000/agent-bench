"""Sequential, resumable experiment orchestration (M9B)."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from agent_bench.context_analysis import derive_context_analysis
from agent_bench.context_storage import (
    store_context_analysis_artifact,
    verify_context_analysis_artifact,
)
from agent_bench.matrix import expand_experiment
from agent_bench.metrics_storage import verify_metrics_artifact
from agent_bench.models import ExperimentDefinition, RunDefinition, canonical_sha256
from agent_bench.opencode_run import execute_controlled_opencode_run
from agent_bench.pi_run import execute_controlled_pi_run
from agent_bench.hermes_run import execute_controlled_hermes_run
from agent_bench.preservation import verify_artifact
from agent_bench.result_store import ResultStoreError, publish_result_ref, verify_published_result
from agent_bench.subject import FrozenSubject, materialize_baseline, verify_materialized_baseline

RunState = Literal[
    "pending", "preflight", "running", "preserving", "analyzing", "completed",
    "failed", "interrupted", "invalid",
]
TERMINAL = {"completed", "failed", "invalid"}


class ExecutorError(RuntimeError):
    """Raised when persisted executor state cannot be used safely."""


class RunProgress(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str
    execution_index: int
    state: RunState = "pending"
    detail: str | None = None


class ExperimentState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1.0.0"] = "1.0.0"
    experiment_id: str
    definition_digest: str
    expansion_digest: str
    ordering: dict[str, object]
    runs: list[RunProgress]
    interrupted: bool = False
    updated_at: str


Dispatch = Callable[[RunDefinition, Path], bool]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def state_path(root: Path) -> Path:
    return root / "experiment-state.json"


def create_state(experiment: ExperimentDefinition) -> ExperimentState:
    runs = expand_experiment(experiment)
    expansion = canonical_sha256([run.model_dump(mode="json") for run in runs])
    return ExperimentState(
        experiment_id=experiment.experiment_id,
        definition_digest=experiment.definition_digest,
        expansion_digest=expansion,
        ordering=experiment.ordering.model_dump(mode="json"),
        runs=[RunProgress(run_id=run.run_id, execution_index=index) for index, run in enumerate(runs, 1)],
        updated_at=_now(),
    )


def write_state(root: Path, state: ExperimentState) -> None:
    """Atomically replace the small mutable progress record."""
    root.mkdir(parents=True, exist_ok=True)
    target = state_path(root)
    payload = state.model_copy(update={"updated_at": _now()}).model_dump(mode="json")
    descriptor, temporary = tempfile.mkstemp(prefix=".state-", dir=root)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _append_executor_event(root: Path, progress: RunProgress) -> None:
    """Append an operational progress event; state remains authoritative."""
    root.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": _now(), "run_id": progress.run_id, "execution_index": progress.execution_index, "state": progress.state, "detail": progress.detail}
    with (root / "executor-events.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_or_create(root: Path, experiment: ExperimentDefinition, resume: bool) -> ExperimentState:
    target = state_path(root)
    if not target.exists():
        if resume:
            raise ExecutorError("cannot resume: experiment state does not exist")
        state = create_state(experiment)
        write_state(root, state)
        return state
    state = ExperimentState.model_validate_json(target.read_bytes())
    expected = create_state(experiment)
    if state.definition_digest != expected.definition_digest or state.expansion_digest != expected.expansion_digest:
        raise ExecutorError("experiment definition or expansion differs from persisted state")
    if not resume:
        raise ExecutorError("experiment state already exists; use --resume")
    return state


def completed_artifact(root: Path, run_id: str) -> bool:
    artifact = root / "artifacts" / run_id
    try:
        manifest = verify_artifact(artifact)
        verify_metrics_artifact(root / "analysis" / run_id / "metrics-v1")
        verify_context_analysis_artifact(root / "analysis" / run_id / "context-analysis-v2")
        verify_published_result(root, manifest)
    except Exception:
        return False
    return (artifact / "normalized/events.jsonl").is_file()


@dataclass
class ExperimentExecutor:
    experiment: ExperimentDefinition
    output_root: Path
    dispatch: Dispatch

    def plan(self) -> tuple[RunDefinition, ...]:
        return expand_experiment(self.experiment)

    def run(self, *, resume: bool = False, limit: int | None = None, selected: set[str] | None = None) -> ExperimentState:
        if limit is not None and limit < 1:
            raise ExecutorError("limit must be at least one")
        state = load_or_create(self.output_root, self.experiment, resume)
        planned = {run.run_id: run for run in self.plan()}
        count = 0
        for progress in state.runs:
            if selected is not None and progress.run_id not in selected:
                continue
            if progress.state == "completed":
                if completed_artifact(self.output_root, progress.run_id):
                    continue
                progress.state, progress.detail = "invalid", "completed-run integrity verification failed"
                write_state(self.output_root, state)
                _append_executor_event(self.output_root, progress)
                continue
            if progress.state in TERMINAL and progress.state != "completed":
                continue
            if limit is not None and count >= limit:
                break
            progress.state, progress.detail = "preflight", None
            write_state(self.output_root, state)
            _append_executor_event(self.output_root, progress)
            progress.state = "running"
            write_state(self.output_root, state)
            _append_executor_event(self.output_root, progress)
            try:
                passed = self.dispatch(planned[progress.run_id], self.output_root)
                progress.state = "completed" if passed else "failed"
                progress.detail = None if passed else "run-local failure"
            except (KeyboardInterrupt, SystemExit):
                progress.state, progress.detail = "interrupted", "executor interrupted"
                state.interrupted = True
                write_state(self.output_root, state)
                _append_executor_event(self.output_root, progress)
                raise
            except Exception as exc:
                progress.state = "failed"
                progress.detail = f"{type(exc).__name__}: {exc}"
            write_state(self.output_root, state)
            _append_executor_event(self.output_root, progress)
            count += 1
        return state


def controlled_dispatch(experiment: ExperimentDefinition, subject: FrozenSubject) -> Dispatch:
    """Create M9B dispatch from the existing controlled harness lifecycles."""
    if experiment.identity_version != "2.0.0" or experiment.portable_baseline != subject.identity:
        raise ExecutorError("experiment portable_baseline does not match the frozen subject")
    prompts = {prompt.prompt_id: prompt.content for prompt in experiment.prompts}

    def dispatch(run: RunDefinition, output_root: Path) -> bool:
        baseline = output_root / "runtime" / "baselines" / run.run_id
        materialize_baseline(subject, baseline)
        preserved = False
        try:
            verify_materialized_baseline(baseline, subject.identity)
            local_run = run.model_copy(update={"baseline_repository": baseline, "baseline_revision": subject.identity.baseline_commit})
            if run.harness_id == "opencode":
                result = execute_controlled_opencode_run(run_definition=local_run, prompt_content=prompts[run.prompt_id], output_root=output_root)
            elif run.harness_id == "pi":
                result = execute_controlled_pi_run(run_definition=local_run, prompt_content=prompts[run.prompt_id], output_root=output_root)
            elif run.harness_id == "hermes":
                result = execute_controlled_hermes_run(run_definition=local_run, prompt_content=prompts[run.prompt_id], output_root=output_root)
            else:  # pragma: no cover - pydantic restricts harness IDs
                raise ExecutorError(f"unsupported harness {run.harness_id}")
            if result.failed_run is not None:
                return False
            assert result.run is not None and result.metrics is not None
            verify_artifact(result.run.artifact_path, repository=baseline)
            verify_metrics_artifact(result.metrics.root)
            context_path = getattr(result, "context_analysis_path", None)
            if context_path is None:
                context_path = store_context_analysis_artifact(
                    source_artifact=result.run.artifact_path,
                    output_root=output_root / "analysis",
                    analysis=derive_context_analysis(result.run.artifact_path),
                )
            verify_context_analysis_artifact(context_path)
            publish_result_ref(output_root, baseline, result.run.artifact_manifest)
            verify_published_result(output_root, result.run.artifact_manifest)
            preserved = True
            return True
        except Exception as exc:
            _preserve_result_store_failure(output_root, run.run_id, baseline, exc)
            raise
        finally:
            # Retain the only source clone on any preservation failure.
            if preserved and baseline.exists():
                shutil.rmtree(baseline)

    return dispatch


def _preserve_result_store_failure(output_root: Path, run_id: str, baseline: Path, error: Exception) -> None:
    """Leave the source clone intact and write non-overwriting recovery evidence."""
    root = output_root / "preservation-failures"
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{run_id}-result-store-failure.json"
    if target.exists():
        return
    payload = {"schema_version": "1.0.0", "run_id": run_id, "timestamp": _now(), "source_repository": str(baseline), "error_type": type(error).__name__, "error": str(error)}
    descriptor, temporary = tempfile.mkstemp(prefix=".failure-", dir=root)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
    except FileExistsError:
        pass
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def status(state: ExperimentState) -> dict[str, object]:
    names = ("pending", "preflight", "running", "preserving", "analyzing", "completed", "failed", "interrupted", "invalid")
    counts = {name: sum(run.state == name for run in state.runs) for name in names}
    return {"experiment_id": state.experiment_id, "total": len(state.runs), "counts": counts, "interrupted": state.interrupted, "expansion_digest": state.expansion_digest}
