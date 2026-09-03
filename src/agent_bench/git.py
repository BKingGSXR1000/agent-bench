"""Deterministic Git baseline isolation and result-ref plumbing."""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

GIT_OBJECT_ID_PATTERN = r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$"
RESULT_REF_PREFIX = "refs/agent-bench/results/"
_RUN_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")


class GitOperationError(RuntimeError):
    """Raised when a required Git operation fails."""


class BaselineIdentity(BaseModel):
    """Resolved immutable Git baseline identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0.0"] = "1.0.0"
    repository: Path
    requested_ref: str = Field(min_length=1)
    commit: str = Field(pattern=GIT_OBJECT_ID_PATTERN)


@dataclass(frozen=True)
class DetachedWorktree:
    """A detached worktree that is removed only through explicit cleanup."""

    repository: Path
    path: Path
    baseline_commit: str


def resolve_baseline(repository: Path, reference: str) -> BaselineIdentity:
    """Resolve a Git reference to an immutable commit without changing the tree."""
    repository_path = repository.expanduser().resolve()
    if not repository_path.is_dir():
        raise GitOperationError(f"baseline repository does not exist: {repository_path}")
    if not reference:
        raise GitOperationError("baseline reference must not be empty")

    _git(repository_path, "rev-parse", "--git-dir")
    top_level = _run_git(
        repository_path,
        "rev-parse",
        "--show-toplevel",
        check=False,
    )
    if top_level.returncode == 0:
        repository_path = Path(_decode_stdout(top_level).strip()).resolve()

    commit = _git(
        repository_path,
        "rev-parse",
        "--verify",
        "--end-of-options",
        f"{reference}^{{commit}}",
    ).strip()
    object_type = _git(repository_path, "cat-file", "-t", commit).strip()
    if object_type != "commit":
        raise GitOperationError(
            f"baseline reference {reference!r} did not resolve to a commit"
        )
    return BaselineIdentity(
        repository=repository_path,
        requested_ref=reference,
        commit=commit,
    )


def create_detached_worktree(
    baseline: BaselineIdentity,
    storage_root: Path,
    *,
    label: str = "run",
) -> DetachedWorktree:
    """Create a unique detached worktree at the exact baseline commit."""
    root = storage_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    prefix = f"agent-bench-{_safe_label(label)}-"
    reserved = Path(tempfile.mkdtemp(prefix=prefix, dir=root))
    reserved.rmdir()

    try:
        _git(
            baseline.repository,
            "worktree",
            "add",
            "--detach",
            str(reserved),
            baseline.commit,
        )
        actual_commit = _git(reserved, "rev-parse", "HEAD").strip()
        branch = _git(reserved, "symbolic-ref", "-q", "HEAD", check=False).strip()
        if actual_commit != baseline.commit or branch:
            raise GitOperationError(
                "created worktree is not detached at the exact baseline commit"
            )
    except Exception:
        # A partially created worktree is evidence for manual recovery.
        raise

    return DetachedWorktree(
        repository=baseline.repository,
        path=reserved,
        baseline_commit=baseline.commit,
    )


def remove_worktree(worktree: DetachedWorktree) -> None:
    """Remove a worktree explicitly after preservation has verified."""
    _git(
        worktree.repository,
        "worktree",
        "remove",
        "--force",
        str(worktree.path),
    )
    if worktree.path.exists():
        raise GitOperationError(f"Git did not remove worktree: {worktree.path}")


def result_ref(run_id: str) -> str:
    """Return the required immutable result-ref name for a run ID."""
    if not _RUN_ID_PATTERN.fullmatch(run_id):
        raise GitOperationError(f"invalid run ID for result ref: {run_id!r}")
    return f"{RESULT_REF_PREFIX}{run_id}"


def ref_exists(repository: Path, reference: str) -> bool:
    """Return whether an exact Git ref already exists."""
    result = _run_git(
        repository,
        "show-ref",
        "--verify",
        "--quiet",
        reference,
        check=False,
    )
    if result.returncode not in {0, 1}:
        raise _command_error(result)
    return result.returncode == 0


def create_result_commit_and_ref(
    worktree: DetachedWorktree,
    run_id: str,
) -> tuple[str, str]:
    """Commit tracked result state and atomically create its immutable ref."""
    reference = result_ref(run_id)
    if ref_exists(worktree.repository, reference):
        raise GitOperationError(f"result ref already exists: {reference}")

    descriptor, index_name = tempfile.mkstemp(
        prefix=".agent-bench-index-",
        dir=worktree.path.parent,
    )
    os.close(descriptor)
    index_path = Path(index_name)
    index_path.unlink()
    index_environment = {"GIT_INDEX_FILE": str(index_path)}
    try:
        _git(
            worktree.path,
            "read-tree",
            worktree.baseline_commit,
            extra_environment=index_environment,
        )
        _git(
            worktree.path,
            "add",
            "-u",
            "--",
            ".",
            extra_environment=index_environment,
        )
        tree = _git(
            worktree.path,
            "write-tree",
            extra_environment=index_environment,
        ).strip()
        baseline_timestamp = _git(
            worktree.path,
            "show",
            "-s",
            "--format=%ct",
            worktree.baseline_commit,
        ).strip()
        commit_environment = {
            **index_environment,
            "GIT_AUTHOR_NAME": "Agent Bench",
            "GIT_AUTHOR_EMAIL": "agent-bench@invalid",
            "GIT_COMMITTER_NAME": "Agent Bench",
            "GIT_COMMITTER_EMAIL": "agent-bench@invalid",
            "GIT_AUTHOR_DATE": f"{baseline_timestamp} +0000",
            "GIT_COMMITTER_DATE": f"{baseline_timestamp} +0000",
        }
        result_commit = _git(
            worktree.path,
            "commit-tree",
            tree,
            "-p",
            worktree.baseline_commit,
            input_bytes=f"Agent Bench result {run_id}\n".encode("utf-8"),
            extra_environment=commit_environment,
        ).strip()
    finally:
        index_path.unlink(missing_ok=True)

    _git(
        worktree.repository,
        "update-ref",
        reference,
        result_commit,
        "",
    )
    resolved = _git(
        worktree.repository,
        "rev-parse",
        "--verify",
        "--end-of-options",
        f"{reference}^{{commit}}",
    ).strip()
    if resolved != result_commit:
        raise GitOperationError(
            f"result ref verification failed: {reference} resolved to {resolved}"
        )
    return result_commit, reference


def git_text(
    repository: Path,
    *arguments: str,
    check: bool = True,
) -> str:
    """Run a deterministic read-only Git command and return decoded output."""
    return _git(repository, *arguments, check=check)


def git_bytes(repository: Path, *arguments: str) -> bytes:
    """Run a deterministic read-only Git command and return exact stdout bytes."""
    return _run_git(repository, *arguments).stdout


def _git(
    repository: Path,
    *arguments: str,
    check: bool = True,
    input_bytes: bytes | None = None,
    extra_environment: dict[str, str] | None = None,
) -> str:
    result = _run_git(
        repository,
        *arguments,
        check=check,
        input_bytes=input_bytes,
        extra_environment=extra_environment,
    )
    return _decode_stdout(result)


def _run_git(
    repository: Path,
    *arguments: str,
    check: bool = True,
    input_bytes: bytes | None = None,
    extra_environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[bytes]:
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
    }
    if extra_environment:
        environment.update(extra_environment)
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=environment,
    )
    if check and result.returncode != 0:
        raise _command_error(result)
    return result


def _decode_stdout(result: subprocess.CompletedProcess[bytes]) -> str:
    return result.stdout.decode("utf-8", errors="surrogateescape")


def _command_error(result: subprocess.CompletedProcess[bytes]) -> GitOperationError:
    stderr = result.stderr.decode("utf-8", errors="replace").strip()
    command = " ".join(str(part) for part in result.args)
    detail = stderr or f"exit status {result.returncode}"
    return GitOperationError(f"Git command failed ({command}): {detail}")


def _safe_label(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-._")[:40] or "run"
