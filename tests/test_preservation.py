from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

import pytest

import agent_bench.preservation as preservation_module
from agent_bench.git import DetachedWorktree, ref_exists, remove_worktree
from agent_bench.preservation import (
    CHECKSUMS_PATH,
    GIT_DIFF_PATH,
    MANIFEST_PATH,
    SOURCE_SNAPSHOT_PATH,
    PreservationError,
    VerificationError,
    create_source_snapshot,
    preserve_isolated_operation,
    restore_artifact,
    verify_artifact,
)
from conftest import (
    GitRepositoryFixture,
    full_result_operation,
)


def test_complete_result_is_preserved_and_worktree_is_cleaned(
    git_repository: GitRepositoryFixture,
    preserved_run: preservation_module.PreservedRun,
) -> None:
    artifact = preserved_run.artifact_path
    manifest = preserved_run.manifest

    assert not preserved_run.former_worktree_path.exists()
    assert artifact == git_repository.artifacts_root / "preserved-run"
    assert (artifact / MANIFEST_PATH).is_file()
    assert (artifact / SOURCE_SNAPSHOT_PATH).is_file()
    assert (artifact / GIT_DIFF_PATH).is_file()
    assert (artifact / CHECKSUMS_PATH).is_file()
    assert (artifact / "build").is_dir()
    assert manifest.preservation_status == "sealed"
    assert manifest.result_ref == "refs/agent-bench/results/preserved-run"
    assert manifest.baseline_commit == git_repository.baseline_commit
    assert manifest.preserved_file_count == 9
    assert manifest.excluded_file_count == 3
    assert manifest.build_artifacts == ()
    assert manifest.build_command is None
    assert manifest.launch_command is None

    assert (git_repository.path / "tracked.txt").read_text(encoding="utf-8") == (
        "baseline\n"
    )
    assert (git_repository.path / "delete-me.txt").is_file()
    assert git_repository.git("status", "--porcelain") == ""
    assert str(preserved_run.former_worktree_path) not in git_repository.git(
        "worktree", "list", "--porcelain"
    )


def test_snapshot_contains_tracked_untracked_ignored_and_conservative_directories(
    preserved_run: preservation_module.PreservedRun,
) -> None:
    snapshot = preserved_run.artifact_path / SOURCE_SNAPSHOT_PATH
    with tarfile.open(snapshot, mode="r:") as archive:
        names = set(archive.getnames())
        tracked = archive.extractfile("tracked.txt")
        ignored = archive.extractfile("ignored/generated.bin")

        assert tracked is not None and tracked.read() == b"modified\n"
        assert ignored is not None and ignored.read() == b"ignored but required\n"

    assert "delete-me.txt" not in names
    assert "new.txt" in names
    assert "ignored/generated.bin" in names
    assert "node_modules/pkg/index.js" in names
    assert "dist/app.js" in names
    assert "build/output.bin" in names
    assert "vendor/library.txt" in names
    assert "generated/asset.txt" in names
    assert ".git" not in names
    assert "__pycache__/module.pyc" not in names
    assert ".pytest_cache/state" not in names


def test_git_metadata_and_result_ref_are_preserved(
    git_repository: GitRepositoryFixture,
    preserved_run: preservation_module.PreservedRun,
) -> None:
    artifact = preserved_run.artifact_path
    manifest = preserved_run.manifest
    status = (artifact / "git/status.txt").read_text(encoding="utf-8")
    untracked = (artifact / "git/untracked.txt").read_text(encoding="ascii")
    ignored = (artifact / "git/ignored.txt").read_text(encoding="ascii")
    diff = (artifact / GIT_DIFF_PATH).read_text(encoding="utf-8")

    assert "tracked.txt" in status
    assert "delete-me.txt" in status
    assert "new.txt" in untracked
    assert "node_modules/pkg/index.js" in untracked
    assert "ignored/generated.bin" in ignored
    assert "tracked.txt" in diff
    assert "delete-me.txt" in diff
    assert ref_exists(git_repository.path, manifest.result_ref)
    assert git_repository.git("rev-parse", manifest.result_ref).strip() == (
        manifest.result_commit
    )
    assert git_repository.git("show", f"{manifest.result_commit}:tracked.txt") == (
        "modified\n"
    )
    result_tree = git_repository.git(
        "ls-tree", "-r", "--name-only", manifest.result_commit
    ).splitlines()
    assert "delete-me.txt" not in result_tree
    assert "new.txt" not in result_tree


def test_checksums_and_manifest_verify(
    git_repository: GitRepositoryFixture,
    preserved_run: preservation_module.PreservedRun,
) -> None:
    manifest = verify_artifact(
        preserved_run.artifact_path,
        repository=git_repository.path,
    )
    checksum_text = (
        preserved_run.artifact_path / CHECKSUMS_PATH
    ).read_text(encoding="utf-8")

    assert manifest == preserved_run.manifest
    assert f"  {MANIFEST_PATH}\n" in checksum_text
    assert f"  {SOURCE_SNAPSHOT_PATH}\n" in checksum_text
    assert f"  {GIT_DIFF_PATH}\n" in checksum_text
    assert hashlib.sha256(
        (preserved_run.artifact_path / SOURCE_SNAPSHOT_PATH).read_bytes()
    ).hexdigest() == manifest.source_snapshot_sha256


def test_restoration_after_cleanup_recreates_complete_preserved_state(
    tmp_path: Path,
    preserved_run: preservation_module.PreservedRun,
) -> None:
    assert not preserved_run.former_worktree_path.exists()
    destination = tmp_path / "restored"

    restore_artifact(preserved_run.artifact_path, destination)

    assert (destination / "tracked.txt").read_text(encoding="utf-8") == "modified\n"
    assert not (destination / "delete-me.txt").exists()
    assert (destination / "new.txt").read_text(encoding="utf-8") == "untracked\n"
    assert (destination / "ignored/generated.bin").read_bytes() == (
        b"ignored but required\n"
    )
    assert (destination / "node_modules/pkg/index.js").is_file()
    assert (destination / "dist/app.js").is_file()
    assert (destination / "build/output.bin").is_file()
    assert (destination / "vendor/library.txt").is_file()
    assert (destination / "generated/asset.txt").is_file()
    assert not (destination / ".git").exists()
    assert not (destination / "__pycache__").exists()
    assert not (destination / ".pytest_cache").exists()


def test_restoration_rejects_a_file_destination(
    tmp_path: Path,
    preserved_run: preservation_module.PreservedRun,
) -> None:
    destination = tmp_path / "not-a-directory"
    destination.write_text("occupied\n", encoding="utf-8")

    with pytest.raises(VerificationError, match="is not a directory"):
        restore_artifact(preserved_run.artifact_path, destination)


def test_snapshot_tar_is_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "file.txt").write_text("same bytes\n", encoding="utf-8")
    (source / "empty").mkdir()
    first = tmp_path / "first.tar"
    second = tmp_path / "second.tar"

    first_stats = create_source_snapshot(source, first)
    (source / "file.txt").touch()
    second_stats = create_source_snapshot(source, second)

    assert first.read_bytes() == second.read_bytes()
    assert first_stats == second_stats


def test_duplicate_run_artifact_or_ref_is_rejected(
    git_repository: GitRepositoryFixture,
    preserved_run: preservation_module.PreservedRun,
    tmp_path: Path,
) -> None:
    with pytest.raises(PreservationError, match="artifact destination already exists"):
        preserve_isolated_operation(
            repository=git_repository.path,
            baseline_ref="HEAD",
            run_id="preserved-run",
            experiment_id="test-experiment",
            artifacts_root=git_repository.artifacts_root,
            worktrees_root=git_repository.worktrees_root,
            operation=lambda _: None,
        )

    with pytest.raises(PreservationError, match="result ref already exists"):
        preserve_isolated_operation(
            repository=git_repository.path,
            baseline_ref="HEAD",
            run_id="preserved-run",
            experiment_id="test-experiment",
            artifacts_root=tmp_path / "other-artifacts",
            worktrees_root=tmp_path / "other-worktrees",
            operation=lambda _: None,
        )


def test_verification_detects_tampering(
    preserved_run: preservation_module.PreservedRun,
) -> None:
    diff = preserved_run.artifact_path / GIT_DIFF_PATH
    diff.write_bytes(diff.read_bytes() + b"tampered\n")

    with pytest.raises(VerificationError, match="checksum mismatch"):
        verify_artifact(preserved_run.artifact_path)


def test_verification_rejects_unlisted_artifact_files(
    preserved_run: preservation_module.PreservedRun,
) -> None:
    (preserved_run.artifact_path / "injected.txt").write_text(
        "not checksummed\n", encoding="utf-8"
    )

    with pytest.raises(VerificationError, match="unlisted files: injected.txt"):
        verify_artifact(preserved_run.artifact_path)


def test_verification_rejects_missing_required_metadata_even_if_checksum_removed(
    preserved_run: preservation_module.PreservedRun,
) -> None:
    artifact = preserved_run.artifact_path
    (artifact / "git/status.txt").unlink()
    checksum_path = artifact / CHECKSUMS_PATH
    checksum_path.write_text(
        "".join(
            line
            for line in checksum_path.read_text(encoding="utf-8").splitlines(True)
            if not line.endswith("  git/status.txt\n")
        ),
        encoding="utf-8",
    )

    with pytest.raises(VerificationError, match="missing required checksums"):
        verify_artifact(artifact)


def test_manifest_is_versioned_and_records_exclusions(
    preserved_run: preservation_module.PreservedRun,
) -> None:
    manifest_data = json.loads(
        (preserved_run.artifact_path / MANIFEST_PATH).read_text(encoding="utf-8")
    )
    exclusions = (preserved_run.artifact_path / "source/excluded.txt").read_text(
        encoding="ascii"
    )

    assert manifest_data["schema_version"] == "1.0.0"
    assert manifest_data["exclusion_policy"]["policy_id"] == "m2-default-v1"
    assert '".git"' in exclusions
    assert '"__pycache__"' in exclusions
    assert '".pytest_cache"' in exclusions


def test_preservation_failure_retains_worktree_and_incomplete_artifacts(
    git_repository: GitRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_snapshot(*args: object, **kwargs: object) -> None:
        raise PreservationError("injected snapshot failure")

    monkeypatch.setattr(preservation_module, "create_source_snapshot", fail_snapshot)

    with pytest.raises(PreservationError, match="injected snapshot failure") as caught:
        preserve_isolated_operation(
            repository=git_repository.path,
            baseline_ref="HEAD",
            run_id="failed-run",
            experiment_id="test-experiment",
            artifacts_root=git_repository.artifacts_root,
            worktrees_root=git_repository.worktrees_root,
            operation=lambda worktree: (worktree / "tracked.txt").write_text(
                "recover me\n", encoding="utf-8"
            ),
        )

    error = caught.value
    assert error.worktree_path is not None and error.worktree_path.is_dir()
    assert error.incomplete_artifact_path is not None
    assert error.incomplete_artifact_path.is_dir()
    assert (error.worktree_path / "tracked.txt").read_text(encoding="utf-8") == (
        "recover me\n"
    )
    assert not ref_exists(
        git_repository.path,
        "refs/agent-bench/results/failed-run",
    )

    remove_worktree(
        DetachedWorktree(
            repository=git_repository.path,
            path=error.worktree_path,
            baseline_commit=git_repository.baseline_commit,
        )
    )


def test_verification_failure_retains_worktree_incomplete_artifact_and_ref(
    git_repository: GitRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_verification(*args: object, **kwargs: object) -> None:
        raise VerificationError("injected verification failure")

    monkeypatch.setattr(preservation_module, "_verify_artifact", fail_verification)

    with pytest.raises(VerificationError, match="injected verification failure") as caught:
        preserve_isolated_operation(
            repository=git_repository.path,
            baseline_ref="HEAD",
            run_id="verification-failed-run",
            experiment_id="test-experiment",
            artifacts_root=git_repository.artifacts_root,
            worktrees_root=git_repository.worktrees_root,
            operation=full_result_operation,
        )

    error = caught.value
    assert error.worktree_path is not None and error.worktree_path.is_dir()
    assert error.incomplete_artifact_path is not None
    assert error.incomplete_artifact_path.is_dir()
    failed_manifest = json.loads(
        (error.incomplete_artifact_path / MANIFEST_PATH).read_text(encoding="utf-8")
    )
    assert failed_manifest["preservation_status"] == "failed"
    assert ref_exists(
        git_repository.path,
        "refs/agent-bench/results/verification-failed-run",
    )

    remove_worktree(
        DetachedWorktree(
            repository=git_repository.path,
            path=error.worktree_path,
            baseline_commit=git_repository.baseline_commit,
        )
    )


def test_artifact_and_worktree_roots_cannot_modify_baseline_repository(
    git_repository: GitRepositoryFixture,
) -> None:
    with pytest.raises(PreservationError, match="outside the baseline repository"):
        preserve_isolated_operation(
            repository=git_repository.path,
            baseline_ref="HEAD",
            run_id="unsafe-root",
            experiment_id="test-experiment",
            artifacts_root=git_repository.path / "artifacts",
            worktrees_root=git_repository.worktrees_root,
            operation=full_result_operation,
        )
