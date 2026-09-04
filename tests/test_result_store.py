"""Durability tests for experiment-local Git result stores."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from agent_bench.result_store import ResultStoreError, publish_result_ref, results_store_path, verify_published_result


def test_result_store_owns_result_objects_after_source_clone_is_removed(preserved_run: object, tmp_path: Path) -> None:
    run = preserved_run
    output = tmp_path / "execution"
    store = publish_result_ref(output, run.baseline.repository, run.manifest)  # type: ignore[attr-defined]
    assert store == results_store_path(output)
    assert (store / "objects" / "info" / "alternates").exists() is False
    shutil.rmtree(run.baseline.repository)  # type: ignore[attr-defined]
    verify_published_result(output, run.manifest)  # type: ignore[attr-defined]


def test_result_store_ref_conflict_never_overwrites(preserved_run: object, tmp_path: Path) -> None:
    run = preserved_run
    output = tmp_path / "execution"
    publish_result_ref(output, run.baseline.repository, run.manifest)  # type: ignore[attr-defined]
    run.baseline.repository.joinpath(".git").exists()  # type: ignore[attr-defined]
    import subprocess
    subprocess.run(["git", "-C", str(run.baseline.repository), "update-ref", run.manifest.result_ref, run.manifest.baseline_commit], check=True)  # type: ignore[attr-defined]
    conflicting = run.manifest.model_copy(update={"result_commit": run.manifest.baseline_commit})  # type: ignore[attr-defined]
    with pytest.raises(ResultStoreError, match="ref conflict"):
        publish_result_ref(output, run.baseline.repository, conflicting)  # type: ignore[attr-defined]
    verify_published_result(output, run.manifest)  # type: ignore[attr-defined]


def test_same_semantic_run_id_is_independent_per_output_root(preserved_run: object, tmp_path: Path) -> None:
    run = preserved_run
    first = tmp_path / "first"
    second = tmp_path / "second"
    publish_result_ref(first, run.baseline.repository, run.manifest)  # type: ignore[attr-defined]
    publish_result_ref(second, run.baseline.repository, run.manifest)  # type: ignore[attr-defined]
    assert results_store_path(first) != results_store_path(second)
    verify_published_result(first, run.manifest)  # type: ignore[attr-defined]
    verify_published_result(second, run.manifest)  # type: ignore[attr-defined]


def test_missing_persistent_ref_is_not_a_completed_artifact(preserved_run: object, tmp_path: Path, monkeypatch: object) -> None:
    from agent_bench.executor import completed_artifact
    import agent_bench.executor as executor

    run = preserved_run
    root = tmp_path / "execution"
    # Isolate the Git-store prerequisite: all artifact/analysis checks succeed,
    # but no durable ref has ever been published.
    monkeypatch.setattr(executor, "verify_artifact", lambda _path: run.manifest)  # type: ignore[attr-defined]
    monkeypatch.setattr(executor, "verify_metrics_artifact", lambda _path: object())  # type: ignore[attr-defined]
    monkeypatch.setattr(executor, "verify_context_analysis_artifact", lambda _path: object())  # type: ignore[attr-defined]
    assert not completed_artifact(root, run.manifest.run_id)  # type: ignore[attr-defined]
