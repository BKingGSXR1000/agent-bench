"""Frozen benchmark-subject verification and fresh baseline materialization."""

from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml

from agent_bench.models import PortableBaselineIdentity


class SubjectError(RuntimeError):
    """Raised when a checked-in subject cannot be reconstructed exactly."""


@dataclass(frozen=True)
class FrozenSubject:
    root: Path
    identity: PortableBaselineIdentity
    source_directory: Path
    bundle: Path


def load_frozen_subject(root: Path) -> FrozenSubject:
    """Read the portable subject definition and verify its tracked bundle."""
    subject_root = root.expanduser().resolve()
    definition_path = subject_root / "subject.yaml"
    try:
        raw = yaml.safe_load(definition_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SubjectError(f"cannot read subject definition {definition_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise SubjectError("subject.yaml must be a mapping")
    try:
        identity = PortableBaselineIdentity.model_validate(
            {
                "subject_id": raw["subject_id"],
                "subject_version": raw["subject_version"],
                "baseline_commit": raw["baseline_commit"],
                "baseline_tree": raw["baseline_tree"],
                "baseline_bundle_sha256": raw["baseline_bundle_sha256"],
            }
        )
        source = subject_root / str(raw["baseline_repository"])
        bundle = subject_root / str(raw["baseline_bundle"])
    except (KeyError, ValueError) as exc:
        raise SubjectError(f"invalid subject identity: {exc}") from exc
    if not source.is_dir():
        raise SubjectError(f"tracked baseline source is missing: {source}")
    _verify_sha256(bundle, identity.baseline_bundle_sha256)
    return FrozenSubject(subject_root, identity, source, bundle)


def materialize_baseline(subject: FrozenSubject, destination: Path) -> Path:
    """Clone the tracked bundle to a new clean repository and verify its tree.

    The destination must not already exist: no previously mutable checkout is
    ever reused as a baseline for a benchmark run.
    """
    target = destination.expanduser().resolve()
    if target.exists():
        raise SubjectError(f"baseline destination already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    _git("clone", str(subject.bundle), str(target))
    _git("-C", str(target), "checkout", "--detach", subject.identity.baseline_commit)
    commit = _git_output("-C", str(target), "rev-parse", "HEAD")
    tree = _git_output("-C", str(target), "rev-parse", "HEAD^{tree}")
    status = _git_output("-C", str(target), "status", "--porcelain")
    if commit != subject.identity.baseline_commit or tree != subject.identity.baseline_tree:
        raise SubjectError("materialized baseline commit/tree differs from subject identity")
    if status:
        raise SubjectError("newly materialized baseline is not clean")
    if _source_fingerprint(subject.source_directory) != _source_fingerprint(target):
        raise SubjectError("tracked baseline source differs from the verified bundle")
    return target


def verify_materialized_baseline(path: Path, identity: PortableBaselineIdentity) -> None:
    """Verify an executor-local checkout before a harness lifecycle starts."""
    root = path.expanduser().resolve()
    if not (root / ".git").exists():
        raise SubjectError(f"materialized baseline is not a Git checkout: {root}")
    if _git_output("-C", str(root), "rev-parse", "HEAD") != identity.baseline_commit:
        raise SubjectError("materialized baseline commit differs from portable identity")
    if _git_output("-C", str(root), "rev-parse", "HEAD^{tree}") != identity.baseline_tree:
        raise SubjectError("materialized baseline tree differs from portable identity")
    if _git_output("-C", str(root), "status", "--porcelain"):
        raise SubjectError("materialized baseline has unexpected local changes")


def _verify_sha256(path: Path, expected: str) -> None:
    if not path.is_file():
        raise SubjectError(f"baseline bundle is missing: {path}")
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != expected:
        raise SubjectError(f"baseline bundle SHA256 mismatch: expected {expected}, observed {observed}")


def _source_fingerprint(root: Path) -> str:
    """Hash ordinary source contents while deliberately ignoring clone metadata."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] == ".git":
            continue
        if path.is_file():
            digest.update(b"F\0" + relative.as_posix().encode("utf-8") + b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        elif path.is_symlink():
            digest.update(b"L\0" + relative.as_posix().encode("utf-8") + b"\0")
            digest.update(os.readlink(path).encode("utf-8"))
    return digest.hexdigest()


def _git(*arguments: str) -> None:
    result = subprocess.run(
        ["git", *arguments], env=_git_environment(), stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, check=False,
    )
    if result.returncode:
        raise SubjectError(result.stderr.strip() or f"git {' '.join(arguments)} failed")


def _git_output(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], env=_git_environment(), stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, check=False,
    )
    if result.returncode:
        raise SubjectError(result.stderr.strip() or f"git {' '.join(arguments)} failed")
    return result.stdout.strip()


def _git_environment() -> dict[str, str]:
    return {
        "PATH": os.environ.get("PATH", ""), "LANG": "C", "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
    }
