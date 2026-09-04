"""Durable, experiment-scoped Git result-object storage."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from agent_bench.git import GitOperationError, result_ref
from agent_bench.preservation import ArtifactManifest


class ResultStoreError(RuntimeError):
    """A persistent result ref could not be safely published or verified."""


def results_store_path(output_root: Path) -> Path:
    """Return the bare object store owned by one experiment execution."""
    return output_root / "git" / "results.git"


def ensure_results_store(output_root: Path) -> Path:
    """Create (once) and validate the experiment-local bare object store."""
    store = results_store_path(output_root)
    if not store.exists():
        store.parent.mkdir(parents=True, exist_ok=True)
        _run(("init", "--bare", str(store)))
    if _run(("--git-dir", str(store), "rev-parse", "--is-bare-repository")).stdout.strip() != b"true":
        raise ResultStoreError(f"result store is not a bare repository: {store}")
    alternates = store / "objects" / "info" / "alternates"
    if alternates.exists() and alternates.read_text(encoding="utf-8").strip():
        raise ResultStoreError(f"result store must not use Git alternates: {alternates}")
    return store


def publish_result_ref(output_root: Path, source_repository: Path, manifest: ArtifactManifest) -> Path:
    """Transfer a sealed run's reachable commit graph into the durable store.

    Existing refs are immutable: an identical publication is verified, whereas a
    different commit is a deterministic preservation failure.
    """
    store = ensure_results_store(output_root)
    reference = result_ref(manifest.run_id)
    expected_commit = manifest.result_commit
    source_commit = _run(("-C", str(source_repository), "rev-parse", f"{reference}^{{commit}}"))
    if source_commit.stdout.strip().decode() != expected_commit:
        raise ResultStoreError("run-local result ref does not match the sealed artifact manifest")
    current = _run(("--git-dir", str(store), "for-each-ref", "--format=%(objectname)", reference))
    current_value = current.stdout.strip().decode()
    if current_value:
        if current_value != expected_commit:
            raise ResultStoreError(f"ref conflict: refusing to overwrite {reference}")
    else:
        # fetch copies reachable objects; the destination receives no alternates.
        _run(("--git-dir", str(store), "fetch", "--no-tags", "--no-recurse-submodules", str(source_repository), f"{reference}:{reference}"))
    verify_published_result(output_root, manifest)
    return store


def verify_published_result(output_root: Path, manifest: ArtifactManifest) -> None:
    """Verify that the immutable sealed result is self-contained in this store."""
    store = ensure_results_store(output_root)
    reference = result_ref(manifest.run_id)
    result_commit = manifest.result_commit
    baseline_commit = manifest.baseline_commit
    resolved = _run(("--git-dir", str(store), "rev-parse", f"{reference}^{{commit}}"), check=False)
    if resolved.returncode != 0 or resolved.stdout.strip().decode() != result_commit:
        raise ResultStoreError(f"persistent result ref is missing or differs: {reference}")
    _run(("--git-dir", str(store), "cat-file", "-e", f"{result_commit}^{{tree}}"))
    _run(("--git-dir", str(store), "cat-file", "-e", f"{baseline_commit}^{{tree}}"))
    ancestry = _run(("--git-dir", str(store), "merge-base", "--is-ancestor", baseline_commit, result_commit), check=False)
    if ancestry.returncode != 0:
        raise ResultStoreError("persistent result commit does not retain baseline ancestry")
    alternates = store / "objects" / "info" / "alternates"
    if alternates.exists() and alternates.read_text(encoding="utf-8").strip():
        raise ResultStoreError("persistent result store relies on Git alternates")
    _run(("--git-dir", str(store), "fsck", "--full", "--no-dangling"))


def _run(arguments: tuple[str, ...], *, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    environment = {
        "PATH": os.environ.get("PATH", ""), "LANG": "C", "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
    }
    result = subprocess.run(["git", *arguments], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, env=environment)
    if check and result.returncode != 0:
        raise _error(result)
    return result


def _error(result: subprocess.CompletedProcess[bytes]) -> ResultStoreError:
    detail = result.stderr.decode("utf-8", errors="replace").strip() or f"exit status {result.returncode}"
    return ResultStoreError(f"Git result-store command failed ({' '.join(result.args)}): {detail}")
