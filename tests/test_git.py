from __future__ import annotations

from pathlib import Path

import pytest

from agent_bench.git import (
    GitOperationError,
    create_detached_worktree,
    create_result_commit_and_ref,
    ref_exists,
    remove_worktree,
    resolve_baseline,
)
from conftest import GitRepositoryFixture


def test_resolves_valid_baseline_without_changing_working_tree(
    git_repository: GitRepositoryFixture,
) -> None:
    before = git_repository.git("status", "--porcelain")

    baseline = resolve_baseline(git_repository.path, "HEAD")

    assert baseline.repository == git_repository.path.resolve()
    assert baseline.commit == git_repository.baseline_commit
    assert git_repository.git("status", "--porcelain") == before == ""


def test_rejects_missing_or_non_git_baseline(tmp_path: Path) -> None:
    with pytest.raises(GitOperationError, match="does not exist"):
        resolve_baseline(tmp_path / "missing", "HEAD")

    ordinary_directory = tmp_path / "ordinary"
    ordinary_directory.mkdir()
    with pytest.raises(GitOperationError, match="not a git repository"):
        resolve_baseline(ordinary_directory, "HEAD")


def test_rejects_invalid_baseline_ref(
    git_repository: GitRepositoryFixture,
) -> None:
    with pytest.raises(GitOperationError, match="unknown revision|Needed a single revision"):
        resolve_baseline(git_repository.path, "does-not-exist")


def test_rejects_ref_that_is_not_a_commit(
    git_repository: GitRepositoryFixture,
) -> None:
    blob = git_repository.git("hash-object", "-w", "tracked.txt").strip()
    git_repository.git("update-ref", "refs/test/blob", blob)

    with pytest.raises(GitOperationError):
        resolve_baseline(git_repository.path, "refs/test/blob")


def test_detached_worktree_starts_at_exact_baseline_and_is_explicitly_removed(
    git_repository: GitRepositoryFixture,
) -> None:
    baseline = resolve_baseline(git_repository.path, "HEAD")
    worktree = create_detached_worktree(
        baseline,
        git_repository.worktrees_root,
        label="isolation-test",
    )

    assert worktree.path.is_dir()
    assert git_repository.git(
        "-C", str(worktree.path), "rev-parse", "HEAD"
    ).strip() == baseline.commit
    assert git_repository.git(
        "-C", str(worktree.path), "symbolic-ref", "-q", "HEAD", check=False
    ) == ""

    (worktree.path / "tracked.txt").write_text("isolated\n", encoding="utf-8")
    assert (git_repository.path / "tracked.txt").read_text(encoding="utf-8") == (
        "baseline\n"
    )
    assert git_repository.git("status", "--porcelain") == ""

    remove_worktree(worktree)

    assert not worktree.path.exists()
    assert str(worktree.path) not in git_repository.git("worktree", "list", "--porcelain")


def test_result_commit_and_ref_capture_tracked_state_only(
    git_repository: GitRepositoryFixture,
) -> None:
    baseline = resolve_baseline(git_repository.path, "HEAD")
    worktree = create_detached_worktree(
        baseline,
        git_repository.worktrees_root,
        label="result-test",
    )
    (worktree.path / "tracked.txt").write_text("result\n", encoding="utf-8")
    (worktree.path / "delete-me.txt").unlink()
    (worktree.path / "untracked.txt").write_text("snapshot only\n", encoding="utf-8")

    result_commit, reference = create_result_commit_and_ref(worktree, "git-result")

    assert reference == "refs/agent-bench/results/git-result"
    assert ref_exists(git_repository.path, reference)
    assert git_repository.git("rev-parse", reference).strip() == result_commit
    assert git_repository.git("show", f"{result_commit}:tracked.txt") == "result\n"
    tree = git_repository.git("ls-tree", "-r", "--name-only", result_commit)
    assert "delete-me.txt" not in tree.splitlines()
    assert "untracked.txt" not in tree.splitlines()
    assert (worktree.path / "untracked.txt").is_file()

    with pytest.raises(GitOperationError, match="already exists"):
        create_result_commit_and_ref(worktree, "git-result")

    remove_worktree(worktree)
