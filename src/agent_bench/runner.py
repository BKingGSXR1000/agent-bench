"""Single-run lifecycle coordination for harness adapters."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agent_bench.capture import (
    CaptureCapabilities,
    fake_harness_capture_capabilities,
)
from agent_bench.events import (
    RawEvent,
    RawEventWriter,
    normalize_raw_events,
)
from agent_bench.git import (
    GIT_OBJECT_ID_PATTERN,
    DetachedWorktree,
    create_detached_worktree,
    git_bytes,
    ref_exists,
    remove_worktree,
    resolve_baseline,
    result_ref,
)
from agent_bench.harness import (
    HarnessAdapter,
    HarnessExecutionResult,
    HarnessRunContext,
    HarnessRunPaths,
    RunTaskService,
)
from agent_bench.models import Identifier, RunDefinition, Sha256, canonical_sha256
from agent_bench.preservation import (
    ArtifactManifest,
    PreservationError,
    preserve_worktree,
    verify_artifact,
)

RAW_EVENTS_PATH = "raw/events.jsonl"
NORMALIZED_EVENTS_PATH = "normalized/events.jsonl"
RUN_MANIFEST_PATH = "run/manifest.json"
HARNESS_STATE_PATH = "run/harness-state/session.json"
PROMPT_PATH = "run/prompt.txt"

ExecutionOutcome = Literal[
    "success",
    "no_changes",
    "timeout",
    "harness_crash",
    "output_truncation",
]


class RunLifecycleError(RuntimeError):
    """An infrastructure failure with retained recovery locations."""

    def __init__(
        self,
        message: str,
        *,
        worktree_path: Path | None = None,
        isolation_root: Path | None = None,
        artifact_path: Path | None = None,
    ) -> None:
        super().__init__(message)
        self.worktree_path = worktree_path
        self.isolation_root = isolation_root
        self.artifact_path = artifact_path


class IsolationPathsRecord(BaseModel):
    """Observed execution-host paths allocated for one run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0.0"] = "1.0.0"
    path_kind: Literal["execution_host_absolute"] = "execution_host_absolute"
    workspace: Path
    isolation_root: Path
    home: Path
    xdg_config_home: Path
    xdg_cache_home: Path
    xdg_data_home: Path
    xdg_state_home: Path
    harness_state: Path
    raw_events_during_execution: Path
    normalized_events_during_execution: Path

    @model_validator(mode="after")
    def require_absolute_paths(self) -> IsolationPathsRecord:
        for field_name in (
            "workspace",
            "isolation_root",
            "home",
            "xdg_config_home",
            "xdg_cache_home",
            "xdg_data_home",
            "xdg_state_home",
            "harness_state",
            "raw_events_during_execution",
            "normalized_events_during_execution",
        ):
            if not getattr(self, field_name).is_absolute():
                raise ValueError(f"{field_name} must be an absolute path")
        return self


class RunManifest(BaseModel):
    """M3 execution record linked to sealed event and artifact evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0.0"] = "1.0.0"
    run_manifest_id: str = Field(min_length=1)
    run_id: Identifier
    experiment_id: Identifier
    run_definition_digest: Sha256
    harness_id: str = Field(min_length=1)
    profile_id: Identifier
    prompt_id: Identifier
    prompt_sha256: Sha256
    prompt_path: Literal["run/prompt.txt"] = PROMPT_PATH
    adapter_id: str = Field(min_length=1)
    adapter_version: str = Field(min_length=1)
    adapter_scenario: str | None = None
    lifecycle_state: Literal["execution_complete"] = "execution_complete"
    baseline_repository: Path
    baseline_commit: str = Field(pattern=GIT_OBJECT_ID_PATTERN)
    isolation: IsolationPathsRecord
    raw_events_path: Literal["raw/events.jsonl"] = RAW_EVENTS_PATH
    normalized_events_path: Literal["normalized/events.jsonl"] = (
        NORMALIZED_EVENTS_PATH
    )
    artifact_manifest_path: Literal["manifest.json"] = "manifest.json"
    preservation_status_source: Literal["artifact_manifest"] = "artifact_manifest"
    task_start_timestamp_utc: datetime
    task_end_timestamp_utc: datetime
    task_start_monotonic_ns: int = Field(ge=0)
    task_elapsed_ns: int = Field(ge=0)
    observed_execution_outcome: ExecutionOutcome
    terminal_raw_event_ids: tuple[str, ...] = Field(min_length=1)
    harness_evidence_paths: tuple[str, ...] = ()
    proxy_endpoint: str | None = None
    run_seed: int | None = None
    capture_capabilities: CaptureCapabilities | None = None
    record_digest: Sha256

    @field_validator("task_start_timestamp_utc", "task_end_timestamp_utc")
    @classmethod
    def require_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("run manifest timestamps must include a UTC offset")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def validate_timing_and_digest(self) -> RunManifest:
        if self.task_end_timestamp_utc < self.task_start_timestamp_utc:
            raise ValueError("task end timestamp precedes task start")
        expected = canonical_sha256(
            self.model_dump(
                mode="json",
                exclude={"record_digest"},
                exclude_computed_fields=True,
            )
        )
        if self.record_digest != expected:
            raise ValueError("record_digest does not match run manifest content")
        return self

    @classmethod
    def create(cls, **values: object) -> RunManifest:
        """Construct a manifest with a digest over canonical JSON content."""
        content = {"schema_version": "1.0.0", **values}
        draft = cls.model_construct(**content, record_digest="0" * 64)
        canonical_content = draft.model_dump(
            mode="json",
            exclude={"record_digest"},
            exclude_computed_fields=True,
        )
        return cls.model_validate(
            {
                **canonical_content,
                "record_digest": canonical_sha256(canonical_content),
            }
        )


@dataclass(frozen=True)
class RunExecutionResult:
    """Paths and identities returned after a completely preserved run."""

    run_manifest: RunManifest
    artifact_manifest: ArtifactManifest
    artifact_path: Path
    raw_event_path: Path
    normalized_event_path: Path
    former_worktree_path: Path
    former_isolation_root: Path


@dataclass(frozen=True)
class _RuntimePaths:
    root: Path
    home: Path
    xdg_config: Path
    xdg_cache: Path
    xdg_data: Path
    xdg_state: Path
    harness_state: Path
    raw_events: Path
    normalized_events: Path
    run_manifest: Path
    prompt: Path


@dataclass
class _AdapterThreadResult:
    result: HarnessExecutionResult | None = None
    error: BaseException | None = None


def execute_run(
    *,
    run_definition: RunDefinition,
    prompt_content: str,
    adapter: HarnessAdapter,
    artifacts_root: Path,
    worktrees_root: Path,
    isolation_root: Path,
    adapter_scenario: str | None = None,
    proxy_endpoint: str | None = None,
    run_seed: int | None = None,
    task_service: RunTaskService | None = None,
) -> RunExecutionResult:
    """Execute and preserve exactly one adapter-backed benchmark task."""
    prompt_bytes = prompt_content.encode("utf-8")
    if hashlib.sha256(prompt_bytes).hexdigest() != run_definition.prompt_sha256:
        raise RunLifecycleError("prompt content does not match RunDefinition SHA256")

    baseline = resolve_baseline(
        run_definition.baseline_repository,
        run_definition.baseline_revision,
    )
    artifacts = artifacts_root.expanduser().resolve()
    worktrees = worktrees_root.expanduser().resolve()
    isolation_storage = isolation_root.expanduser().resolve()
    for label, path in (
        ("artifacts_root", artifacts),
        ("worktrees_root", worktrees),
        ("isolation_root", isolation_storage),
    ):
        if _is_relative_to(path, baseline.repository):
            raise RunLifecycleError(
                f"{label} must be outside the baseline repository"
            )
    final_artifact = artifacts / run_definition.run_id
    if final_artifact.exists():
        raise RunLifecycleError(f"artifact destination already exists: {final_artifact}")
    if ref_exists(baseline.repository, result_ref(run_definition.run_id)):
        raise RunLifecycleError(
            f"result ref already exists: {result_ref(run_definition.run_id)}"
        )

    runtime = _create_runtime_paths(isolation_storage, run_definition.run_id)
    runtime.prompt.parent.mkdir(parents=True, exist_ok=True)
    runtime.prompt.write_bytes(prompt_bytes)
    worktree: DetachedWorktree | None = None
    try:
        worktree = create_detached_worktree(
            baseline,
            worktrees,
            label=run_definition.run_id,
        )
        paths = HarnessRunPaths(
            workspace=worktree.path,
            home=runtime.home,
            xdg_config_home=runtime.xdg_config,
            xdg_cache_home=runtime.xdg_cache,
            xdg_data_home=runtime.xdg_data,
            xdg_state_home=runtime.xdg_state,
            harness_state=runtime.harness_state,
        )
        run_manifest, adapter_result, service_evidence = _execute_and_record(
            run_definition=run_definition,
            prompt_content=prompt_content,
            adapter=adapter,
            adapter_scenario=adapter_scenario,
            worktree=worktree,
            paths=paths,
            runtime=runtime,
            proxy_endpoint=proxy_endpoint,
            run_seed=run_seed,
            task_service=task_service,
        )
        _write_model(runtime.run_manifest, run_manifest)
        artifact_manifest = preserve_worktree(
            worktree=worktree,
            run_id=run_definition.run_id,
            experiment_id=run_definition.experiment_id,
            artifacts_root=artifacts,
            supplemental_files={
                RAW_EVENTS_PATH: runtime.raw_events,
                NORMALIZED_EVENTS_PATH: runtime.normalized_events,
                RUN_MANIFEST_PATH: runtime.run_manifest,
                PROMPT_PATH: runtime.prompt,
                **_evidence_mapping(runtime, adapter_result, service_evidence),
            },
        )
        verify_artifact(final_artifact, repository=baseline.repository)
        remove_worktree(worktree)
        shutil.rmtree(runtime.root)
    except Exception as exc:
        if isinstance(exc, RunLifecycleError):
            if exc.worktree_path is None and worktree is not None:
                exc.worktree_path = worktree.path
            if exc.isolation_root is None:
                exc.isolation_root = runtime.root
            raise
        if isinstance(exc, PreservationError):
            message = str(exc)
            incomplete = exc.incomplete_artifact_path
        else:
            message = f"run lifecycle failed: {exc}"
            incomplete = None
        raise RunLifecycleError(
            message,
            worktree_path=worktree.path if worktree is not None else None,
            isolation_root=runtime.root,
            artifact_path=incomplete,
        ) from exc

    return RunExecutionResult(
        run_manifest=run_manifest,
        artifact_manifest=artifact_manifest,
        artifact_path=final_artifact,
        raw_event_path=final_artifact / RAW_EVENTS_PATH,
        normalized_event_path=final_artifact / NORMALIZED_EVENTS_PATH,
        former_worktree_path=worktree.path,
        former_isolation_root=runtime.root,
    )


def _execute_and_record(
    *,
    run_definition: RunDefinition,
    prompt_content: str,
    adapter: HarnessAdapter,
    adapter_scenario: str | None,
    worktree: DetachedWorktree,
    paths: HarnessRunPaths,
    runtime: _RuntimePaths,
    proxy_endpoint: str | None,
    run_seed: int | None,
    task_service: RunTaskService | None,
) -> tuple[RunManifest, HarnessExecutionResult | None, tuple[tuple[str, Path], ...]]:
    cancellation = threading.Event()
    terminal_events: list[RawEvent] = []
    adapter_result: HarnessExecutionResult | None = None
    service_evidence: tuple[tuple[str, Path], ...] = ()
    service_started = False
    service_stopped = False
    writer = RawEventWriter(
        runtime.raw_events,
        run_definition.run_id,
    )
    try:
        if task_service is not None:
            task_service.start(writer)
            service_started = True
        task_start_ns = time.monotonic_ns()
        writer.reset_task_start(task_start_ns)
        start_event = writer.emit(
            source="runner",
            event_type="run_start",
            payload={
                "run_definition_digest": run_definition.definition_digest,
                "baseline_commit": worktree.baseline_commit,
                "prompt_sha256": run_definition.prompt_sha256,
                "wall_timeout_seconds": run_definition.limits.wall_timeout_seconds,
                "isolated_paths": {
                    "workspace": str(paths.workspace),
                    **paths.environment(),
                },
            },
        )
        context = HarnessRunContext(
            run_definition=run_definition,
            paths=paths,
            prompt_content=prompt_content,
            events=writer,
            limits=run_definition.limits,
            cancellation=cancellation,
            proxy_endpoint=proxy_endpoint,
            run_seed=run_seed,
        )
        adapter_result, adapter_error, timed_out = _invoke_adapter(
            adapter,
            context,
        )
        if timed_out:
            terminal_events.append(
                writer.emit(
                    source="runner",
                    event_type="timeout",
                    payload={
                        "limit_seconds": run_definition.limits.wall_timeout_seconds,
                        "trigger": "runner_deadline",
                    },
                )
            )
            outcome: ExecutionOutcome = "timeout"
            process_status = "cancelled_after_timeout"
        elif adapter_error is not None:
            terminal_events.append(
                writer.emit(
                    source="runner",
                    event_type="harness_error",
                    payload={
                        "error_type": type(adapter_error).__name__,
                        "message": str(adapter_error),
                    },
                )
            )
            outcome = "harness_crash"
            process_status = "crashed"
        else:
            if adapter_result is None:
                raise RunLifecycleError("adapter returned no result")
            if not adapter_result.completed_normally:
                terminal_events.append(
                    writer.emit(
                        source="runner",
                        event_type="harness_error",
                        payload={
                            "error_type": "abnormal_adapter_completion",
                            "message": "adapter did not report normal completion",
                        },
                    )
                )
                outcome = "harness_crash"
                process_status = "abnormal_completion"
            elif adapter_result.output_truncated:
                outcome = "output_truncation"
                process_status = "completed"
            else:
                process_status = "completed"
                status = git_bytes(
                    worktree.path,
                    "status",
                    "--porcelain=v1",
                    "--no-renames",
                    "--untracked-files=all",
                    "--ignored=matching",
                )
                has_changes = bool(status)
                writer.emit(
                    source="git",
                    event_type="workspace_status",
                    payload={
                        "has_changes": has_changes,
                        "method": "git-status-porcelain-v1-with-ignored-no-renames",
                    },
                )
                outcome = "success" if has_changes else "no_changes"

        terminal_events.append(
            writer.emit(
                source="runner",
                event_type="process_termination",
                payload={"status": process_status, "adapter_id": adapter.adapter_id},
            )
        )
        end_event = writer.emit(
            source="runner",
            event_type="run_end",
            payload={
                "observed_execution_outcome": outcome,
                "formal_classification": "deferred_to_m4",
            },
        )
        terminal_events.append(end_event)
        if task_service is not None:
            service_stopped = True
            service_evidence = task_service.stop(writer)
    finally:
        if task_service is not None and service_started and not service_stopped:
            service_stopped = True
            try:
                service_evidence = task_service.stop(writer)
            except Exception:
                writer.seal()
                raise
        writer.seal()

    adapter_normalizer = getattr(adapter, "normalize_events", None)
    if callable(adapter_normalizer):
        adapter_normalizer(runtime.raw_events, runtime.normalized_events)
    else:
        normalize_raw_events(runtime.raw_events, runtime.normalized_events)
    isolation_record = IsolationPathsRecord(
        workspace=paths.workspace,
        isolation_root=runtime.root,
        home=paths.home,
        xdg_config_home=paths.xdg_config_home,
        xdg_cache_home=paths.xdg_cache_home,
        xdg_data_home=paths.xdg_data_home,
        xdg_state_home=paths.xdg_state_home,
        harness_state=paths.harness_state,
        raw_events_during_execution=runtime.raw_events,
        normalized_events_during_execution=runtime.normalized_events,
    )
    manifest = RunManifest.create(
        run_manifest_id=f"{run_definition.run_id}-manifest",
        run_id=run_definition.run_id,
        experiment_id=run_definition.experiment_id,
        run_definition_digest=run_definition.definition_digest,
        harness_id=run_definition.harness_id,
        profile_id=run_definition.profile_id,
        prompt_id=run_definition.prompt_id,
        prompt_sha256=run_definition.prompt_sha256,
        adapter_id=adapter.adapter_id,
        adapter_version=adapter.adapter_version,
        adapter_scenario=adapter_scenario,
        baseline_repository=worktree.repository,
        baseline_commit=worktree.baseline_commit,
        isolation=isolation_record,
        task_start_timestamp_utc=start_event.timestamp_utc,
        task_end_timestamp_utc=end_event.timestamp_utc,
        task_start_monotonic_ns=task_start_ns,
        task_elapsed_ns=end_event.elapsed_ns,
        observed_execution_outcome=outcome,
        terminal_raw_event_ids=tuple(
            event.raw_event_id for event in terminal_events
        ),
        harness_evidence_paths=tuple(
            sorted(_evidence_mapping(runtime, adapter_result, service_evidence))
        ),
        proxy_endpoint=proxy_endpoint,
        run_seed=run_seed,
        capture_capabilities=(
            fake_harness_capture_capabilities()
            if adapter.adapter_id == "fake-harness"
            else getattr(adapter, "capture_capabilities", None)
        ),
    )
    return manifest, adapter_result, service_evidence


def _invoke_adapter(
    adapter: HarnessAdapter,
    context: HarnessRunContext,
) -> tuple[HarnessExecutionResult | None, BaseException | None, bool]:
    holder = _AdapterThreadResult()

    def invoke() -> None:
        try:
            holder.result = adapter.run(context)
        except BaseException as exc:
            holder.error = exc

    thread = threading.Thread(
        target=invoke,
        name=f"agent-bench-{context.run_definition.run_id}",
        daemon=True,
    )
    thread.start()
    thread.join(context.limits.wall_timeout_seconds)
    if not thread.is_alive():
        return holder.result, holder.error, False

    context.cancellation.set()
    thread.join(timeout=1.0)
    if thread.is_alive():
        raise RunLifecycleError(
            "adapter did not stop after the wall-clock timeout; temporary state retained"
        )
    return holder.result, holder.error, True


def _create_runtime_paths(root: Path, run_id: str) -> _RuntimePaths:
    root.mkdir(parents=True, exist_ok=True)
    run_root = Path(tempfile.mkdtemp(prefix=f"agent-bench-{run_id}-", dir=root))
    home = run_root / "home"
    xdg_config = run_root / "xdg-config"
    xdg_cache = run_root / "xdg-cache"
    xdg_data = run_root / "xdg-data"
    xdg_state = run_root / "xdg-state"
    harness_state = run_root / "harness-state"
    for path in (home, xdg_config, xdg_cache, xdg_data, xdg_state, harness_state):
        path.mkdir()
    return _RuntimePaths(
        root=run_root,
        home=home,
        xdg_config=xdg_config,
        xdg_cache=xdg_cache,
        xdg_data=xdg_data,
        xdg_state=xdg_state,
        harness_state=harness_state,
        raw_events=run_root / RAW_EVENTS_PATH,
        normalized_events=run_root / NORMALIZED_EVENTS_PATH,
        run_manifest=run_root / RUN_MANIFEST_PATH,
        prompt=run_root / PROMPT_PATH,
    )


def _evidence_mapping(
    runtime: _RuntimePaths,
    adapter_result: HarnessExecutionResult | None,
    service_evidence: tuple[tuple[str, Path], ...],
) -> dict[str, Path]:
    """Collect generic run-local state plus explicitly named native evidence."""
    result: dict[str, Path] = {}
    for source in sorted(runtime.harness_state.rglob("*")):
        if source.is_file() and not source.is_symlink():
            relative = source.relative_to(runtime.harness_state).as_posix()
            result[f"run/harness-state/{relative}"] = source
    declared = (
        adapter_result.evidence_files if adapter_result is not None else ()
    ) + service_evidence
    for relative, source in declared:
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
            raise RunLifecycleError(f"unsafe harness evidence path: {relative}")
        normalized = candidate.as_posix()
        existing = result.get(normalized)
        if existing is not None and existing.resolve() != source.resolve():
            raise RunLifecycleError(f"duplicate harness evidence path: {normalized}")
        if not source.is_file() or source.is_symlink():
            raise RunLifecycleError(f"harness evidence is not a regular file: {source}")
        result[normalized] = source
    return result


def _write_model(path: Path, model: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        json.dumps(
            model.model_dump(mode="json", exclude_computed_fields=True),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    with path.open("xb") as output:
        output.write(content)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True
