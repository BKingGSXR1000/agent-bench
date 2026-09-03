from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import pytest

import agent_bench.runner as runner_module
from agent_bench.events import load_normalized_events, load_raw_events
from agent_bench.fake_harness import FakeHarness
from agent_bench.git import DetachedWorktree, remove_worktree
from agent_bench.models import RunLimits
from agent_bench.preservation import PreservationError, restore_artifact, verify_artifact
from agent_bench.runner import RunLifecycleError, RunManifest, execute_run
from conftest import GitRepositoryFixture, RunFixture


def _run(
    git_repository: GitRepositoryFixture,
    run_fixture: RunFixture,
    scenario: str,
    *,
    timeout_seconds: float | None = None,
):
    definition = run_fixture.run_definition
    if timeout_seconds is not None:
        definition = definition.model_copy(
            update={"limits": RunLimits(wall_timeout_seconds=timeout_seconds)}
        )
    return execute_run(
        run_definition=definition,
        prompt_content=run_fixture.prompt_content,
        adapter=FakeHarness(scenario),  # type: ignore[arg-type]
        adapter_scenario=scenario,
        artifacts_root=git_repository.artifacts_root,
        worktrees_root=git_repository.worktrees_root,
        isolation_root=git_repository.path.parent / "isolation",
    )


def test_success_run_is_isolated_normalized_preserved_and_restorable(
    git_repository: GitRepositoryFixture,
    run_fixture: RunFixture,
    tmp_path: Path,
) -> None:
    baseline_status = git_repository.git("status", "--porcelain")

    result = _run(git_repository, run_fixture, "success")

    assert result.run_manifest.observed_execution_outcome == "success"
    assert result.run_manifest.baseline_commit == git_repository.baseline_commit
    assert result.artifact_manifest.baseline_commit == git_repository.baseline_commit
    assert not result.former_worktree_path.exists()
    assert not result.former_isolation_root.exists()
    assert git_repository.git("status", "--porcelain") == baseline_status == ""
    assert verify_artifact(result.artifact_path) == result.artifact_manifest

    raw = load_raw_events(result.raw_event_path)
    normalized = load_normalized_events(result.normalized_event_path)
    assert [event.sequence for event in raw] == list(range(1, len(raw) + 1))
    assert normalized[0].event_kind == "run_start"
    assert normalized[-1].event_kind == "run_end"
    assert {event.event_kind for event in normalized} >= {
        "reasoning",
        "file_read",
        "file_edit",
        "file_write",
        "tool_call_start",
        "tool_call_end",
        "process_termination",
    }
    environment_event = next(
        event for event in raw if event.event_type == "harness_environment"
    )
    assert environment_event.payload["all_paths_existed"] is True
    assert environment_event.payload["fresh_session"] is True

    restored = tmp_path / "restored-success"
    restore_artifact(result.artifact_path, restored)
    assert (restored / "tracked.txt").read_text(encoding="utf-8") == (
        "updated by FakeHarness\n"
    )
    assert (restored / "fake-created.txt").read_text(encoding="utf-8") == (
        "created by FakeHarness\n"
    )
    assert (result.artifact_path / "run/harness-state/session.json").is_file()


def test_run_manifest_links_definition_events_isolation_and_artifact(
    git_repository: GitRepositoryFixture,
    run_fixture: RunFixture,
) -> None:
    result = _run(git_repository, run_fixture, "success")
    manifest = RunManifest.model_validate_json(
        (result.artifact_path / "run/manifest.json").read_bytes()
    )

    assert manifest.run_definition_digest == run_fixture.run_definition.definition_digest
    assert manifest.raw_events_path == "raw/events.jsonl"
    assert manifest.normalized_events_path == "normalized/events.jsonl"
    assert manifest.artifact_manifest_path == "manifest.json"
    assert manifest.preservation_status_source == "artifact_manifest"
    assert manifest.adapter_id == "fake-harness"
    assert manifest.adapter_scenario == "success"
    assert manifest.task_elapsed_ns >= 0
    assert manifest.isolation.workspace == result.former_worktree_path
    assert manifest.isolation.home.is_absolute()


def test_no_change_run_is_still_preserved(
    git_repository: GitRepositoryFixture,
    run_fixture: RunFixture,
) -> None:
    result = _run(git_repository, run_fixture, "no_change")

    assert result.run_manifest.observed_execution_outcome == "no_changes"
    assert result.artifact_manifest.result_commit
    assert result.artifact_path.is_dir()
    assert verify_artifact(result.artifact_path).preservation_status == "sealed"


def test_failed_tool_scenario_records_failure_then_recovery(
    git_repository: GitRepositoryFixture,
    run_fixture: RunFixture,
) -> None:
    result = _run(git_repository, run_fixture, "failed_tool")
    events = load_normalized_events(result.normalized_event_path)
    tool_ends = [event for event in events if event.event_kind == "tool_call_end"]

    assert result.run_manifest.observed_execution_outcome == "no_changes"
    assert tool_ends[0].payload["outcome"] == "failure"
    assert tool_ends[-1].payload["outcome"] == "success"


def test_crash_run_records_error_and_is_preserved(
    git_repository: GitRepositoryFixture,
    run_fixture: RunFixture,
) -> None:
    result = _run(git_repository, run_fixture, "crash")
    events = load_normalized_events(result.normalized_event_path)

    assert result.run_manifest.observed_execution_outcome == "harness_crash"
    error = next(event for event in events if event.event_kind == "harness_error")
    assert error.payload["error_type"] == "FakeHarnessCrash"
    assert events[-1].payload["observed_execution_outcome"] == "harness_crash"
    assert verify_artifact(result.artifact_path)


def test_timeout_is_short_runner_enforced_and_preserved(
    git_repository: GitRepositoryFixture,
    run_fixture: RunFixture,
) -> None:
    started = time.monotonic()
    result = _run(
        git_repository,
        run_fixture,
        "timeout",
        timeout_seconds=0.01,
    )
    duration = time.monotonic() - started
    events = load_normalized_events(result.normalized_event_path)

    assert duration < 1
    assert result.run_manifest.observed_execution_outcome == "timeout"
    timeout = next(event for event in events if event.event_kind == "timeout")
    assert timeout.payload["limit_seconds"] == 0.01
    assert verify_artifact(result.artifact_path)


def test_output_truncation_run_is_preserved_with_direct_signal(
    git_repository: GitRepositoryFixture,
    run_fixture: RunFixture,
) -> None:
    result = _run(git_repository, run_fixture, "output_truncation")
    events = load_normalized_events(result.normalized_event_path)

    assert result.run_manifest.observed_execution_outcome == "output_truncation"
    response = next(event for event in events if event.event_kind == "llm_response")
    assert response.payload["outcome"] == "truncated"
    assert any(event.event_kind == "output_truncation" for event in events)
    assert verify_artifact(result.artifact_path)


def test_reasoning_without_action_has_no_tool_event(
    git_repository: GitRepositoryFixture,
    run_fixture: RunFixture,
) -> None:
    result = _run(git_repository, run_fixture, "reasoning_without_action")
    events = load_normalized_events(result.normalized_event_path)
    kinds = [event.event_kind for event in events]

    assert result.run_manifest.observed_execution_outcome == "no_changes"
    assert "reasoning" in kinds
    assert "llm_response" in kinds
    assert "tool_call_start" not in kinds
    assert "tool_call_end" not in kinds


def test_prompt_identity_mismatch_fails_before_allocating_run_state(
    git_repository: GitRepositoryFixture,
    run_fixture: RunFixture,
) -> None:
    with pytest.raises(RunLifecycleError, match="prompt content"):
        execute_run(
            run_definition=run_fixture.run_definition,
            prompt_content="different prompt",
            adapter=FakeHarness("success"),
            adapter_scenario="success",
            artifacts_root=git_repository.artifacts_root,
            worktrees_root=git_repository.worktrees_root,
            isolation_root=git_repository.path.parent / "isolation",
        )

    assert not git_repository.artifacts_root.exists()
    assert not git_repository.worktrees_root.exists()


def test_preservation_failure_retains_worktree_and_isolated_state(
    git_repository: GitRepositoryFixture,
    run_fixture: RunFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_preservation(**kwargs: object) -> None:
        raise PreservationError("injected runner preservation failure")

    monkeypatch.setattr(runner_module, "preserve_worktree", fail_preservation)

    with pytest.raises(
        RunLifecycleError, match="injected runner preservation failure"
    ) as caught:
        _run(git_repository, run_fixture, "success")

    error = caught.value
    assert error.worktree_path is not None and error.worktree_path.is_dir()
    assert error.isolation_root is not None and error.isolation_root.is_dir()
    assert (error.isolation_root / "raw/events.jsonl").is_file()
    assert (error.isolation_root / "normalized/events.jsonl").is_file()
    assert (error.isolation_root / "run/manifest.json").is_file()

    remove_worktree(
        DetachedWorktree(
            repository=git_repository.path,
            path=error.worktree_path,
            baseline_commit=git_repository.baseline_commit,
        )
    )
    shutil.rmtree(error.isolation_root)


def test_persisted_run_manifest_rejects_tampering(
    git_repository: GitRepositoryFixture,
    run_fixture: RunFixture,
) -> None:
    result = _run(git_repository, run_fixture, "no_change")
    path = result.artifact_path / "run/manifest.json"
    content = json.loads(path.read_text(encoding="utf-8"))
    content["observed_execution_outcome"] = "success"

    with pytest.raises(ValueError, match="record_digest"):
        RunManifest.model_validate(content)
