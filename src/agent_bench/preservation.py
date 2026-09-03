"""Immutable preservation and restoration for isolated run worktrees."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import tarfile
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agent_bench.git import (
    GIT_OBJECT_ID_PATTERN,
    BaselineIdentity,
    DetachedWorktree,
    GitOperationError,
    create_detached_worktree,
    create_result_commit_and_ref,
    git_bytes,
    git_text,
    ref_exists,
    remove_worktree,
    resolve_baseline,
    result_ref,
)
from agent_bench.models import Identifier, Sha256

SOURCE_SNAPSHOT_PATH = "source/source.tar"
EXCLUDED_PATHS_PATH = "source/excluded.txt"
CHECKSUMS_PATH = "checksums.sha256"
MANIFEST_PATH = "manifest.json"
GIT_DIFF_PATH = "git/diff.patch"
GIT_TRACKED_NUMSTAT_PATH = "git/tracked-numstat.json"
GIT_UNTRACKED_NUMSTAT_PATH = "git/untracked-numstat.json"


class PreservationError(RuntimeError):
    """A preservation failure with retained recovery locations."""

    def __init__(
        self,
        message: str,
        *,
        worktree_path: Path | None = None,
        incomplete_artifact_path: Path | None = None,
    ) -> None:
        super().__init__(message)
        self.worktree_path = worktree_path
        self.incomplete_artifact_path = incomplete_artifact_path


class VerificationError(PreservationError):
    """Raised when preserved artifacts fail deterministic verification."""


class ExclusionPolicyRecord(BaseModel):
    """Persisted identity of the narrow default snapshot exclusion policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0.0"] = "1.0.0"
    policy_id: Literal["m2-default-v1"] = "m2-default-v1"
    excluded_directory_names: tuple[str, ...]
    excluded_file_names: tuple[str, ...]
    excluded_file_suffixes: tuple[str, ...]


DEFAULT_EXCLUSION_POLICY = ExclusionPolicyRecord(
    excluded_directory_names=(
        ".agent-bench-tmp",
        ".bench",
        ".git",
        ".pytest_cache",
        "__pycache__",
    ),
    excluded_file_names=(".git",),
    excluded_file_suffixes=(".pyc",),
)


class ArtifactManifest(BaseModel):
    """Versioned immutable index of one successfully preserved M2 result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0.0"] = "1.0.0"
    artifact_manifest_id: str = Field(min_length=1, max_length=256)
    run_id: Identifier
    experiment_id: Identifier
    baseline_repository: Path
    baseline_commit: str = Field(pattern=GIT_OBJECT_ID_PATTERN)
    result_commit: str = Field(pattern=GIT_OBJECT_ID_PATTERN)
    result_ref: str = Field(min_length=1)
    source_snapshot_path: str
    source_snapshot_format: Literal["tar-pax-v1"] = "tar-pax-v1"
    source_snapshot_sha256: Sha256
    git_diff_path: str
    git_diff_sha256: Sha256
    checksums_path: str
    creation_timestamp_utc: datetime
    preserved_file_count: int = Field(ge=0)
    preserved_byte_count: int = Field(ge=0)
    excluded_file_count: int = Field(ge=0)
    exclusion_policy: ExclusionPolicyRecord
    preservation_status: Literal["verifying", "sealed", "failed"]
    build_artifacts: tuple[str, ...] = ()
    build_command: tuple[str, ...] | None = None
    launch_command: tuple[str, ...] | None = None
    manifest_checksum_strategy: Literal["checksums-file-includes-manifest"] = (
        "checksums-file-includes-manifest"
    )

    @field_validator("creation_timestamp_utc")
    @classmethod
    def require_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("creation_timestamp_utc must include a UTC offset")
        return value.astimezone(timezone.utc)

    @field_validator(
        "source_snapshot_path",
        "git_diff_path",
        "checksums_path",
    )
    @classmethod
    def require_artifact_relative_path(cls, value: str) -> str:
        _validate_relative_artifact_path(value)
        return value

    @model_validator(mode="after")
    def validate_fixed_paths_and_result_ref(self) -> ArtifactManifest:
        expected_paths = {
            "source_snapshot_path": SOURCE_SNAPSHOT_PATH,
            "git_diff_path": GIT_DIFF_PATH,
            "checksums_path": CHECKSUMS_PATH,
        }
        for field_name, expected_path in expected_paths.items():
            if getattr(self, field_name) != expected_path:
                raise ValueError(f"{field_name} must be {expected_path}")
        expected = result_ref(self.run_id)
        if self.result_ref != expected:
            raise ValueError(f"result_ref must be {expected}")
        return self


class GitNumstatEntry(BaseModel):
    """Git-native line accounting for one preserved non-tracked path."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0.0"] = "1.0.0"
    path: str = Field(min_length=1)
    lines_added: int | None = Field(default=None, ge=0)
    lines_deleted: int | None = Field(default=None, ge=0)
    binary: bool
    availability: Literal["available", "unavailable"]
    unavailable_reason: str | None = None

    @model_validator(mode="after")
    def validate_availability(self) -> GitNumstatEntry:
        _validate_relative_artifact_path(self.path)
        if self.availability == "available":
            if self.unavailable_reason is not None:
                raise ValueError("available numstat cannot have an unavailable reason")
            if not self.binary and (self.lines_added is None or self.lines_deleted is None):
                raise ValueError("available text numstat requires line counts")
            if self.binary and (self.lines_added is not None or self.lines_deleted is not None):
                raise ValueError("binary numstat cannot have text line counts")
        elif (
            self.unavailable_reason is None
            or self.lines_added is not None
            or self.lines_deleted is not None
        ):
            raise ValueError("unavailable numstat requires null counts and a reason")
        return self


class GitNumstatRecord(BaseModel):
    """Versioned preservation-time Git-native line-count evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0.0"] = "1.0.0"
    algorithm: Literal[
        "git-diff-numstat-no-renames-v1",
        "git-diff-no-index-numstat-v1",
    ]
    git_version: str = Field(min_length=1)
    entries: tuple[GitNumstatEntry, ...]


@dataclass(frozen=True)
class SnapshotStats:
    """Deterministic source-snapshot inventory statistics."""

    preserved_file_count: int
    preserved_byte_count: int
    excluded_file_count: int
    excluded_paths: tuple[str, ...]


@dataclass(frozen=True)
class PreservedRun:
    """Completed preservation result; its temporary worktree is gone."""

    baseline: BaselineIdentity
    artifact_path: Path
    manifest: ArtifactManifest
    former_worktree_path: Path


@dataclass(frozen=True)
class _SnapshotEntry:
    absolute_path: Path
    relative_path: PurePosixPath
    kind: Literal["file", "directory", "symlink"]


def preserve_isolated_operation(
    *,
    repository: Path,
    baseline_ref: str,
    run_id: str,
    experiment_id: str,
    artifacts_root: Path,
    worktrees_root: Path,
    operation: Callable[[Path], None],
) -> PreservedRun:
    """Run an injected filesystem operation and preserve it before cleanup."""
    baseline = resolve_baseline(repository, baseline_ref)
    final_artifact_path = _validate_new_run_destination(
        baseline=baseline,
        run_id=run_id,
        artifacts_root=artifacts_root,
        worktrees_root=worktrees_root,
    )
    worktree = create_detached_worktree(
        baseline,
        worktrees_root,
        label=run_id,
    )
    try:
        operation(worktree.path)
        manifest = preserve_worktree(
            worktree=worktree,
            run_id=run_id,
            experiment_id=experiment_id,
            artifacts_root=artifacts_root,
        )
        artifact_path = final_artifact_path
        verify_artifact(artifact_path, repository=baseline.repository)
        remove_worktree(worktree)
    except Exception as exc:
        if isinstance(exc, PreservationError):
            if exc.worktree_path is None:
                exc.worktree_path = worktree.path
            raise
        raise PreservationError(
            f"preservation failed; worktree retained at {worktree.path}: {exc}",
            worktree_path=worktree.path,
        ) from exc

    return PreservedRun(
        baseline=baseline,
        artifact_path=artifact_path,
        manifest=manifest,
        former_worktree_path=worktree.path,
    )


def preserve_worktree(
    *,
    worktree: DetachedWorktree,
    run_id: str,
    experiment_id: str,
    artifacts_root: Path,
    exclusion_policy: ExclusionPolicyRecord = DEFAULT_EXCLUSION_POLICY,
    supplemental_files: Mapping[str, Path] | None = None,
) -> ArtifactManifest:
    """Preserve a prepared worktree without removing it."""
    final_path = _validate_new_run_destination(
        baseline=BaselineIdentity(
            repository=worktree.repository,
            requested_ref=worktree.baseline_commit,
            commit=worktree.baseline_commit,
        ),
        run_id=run_id,
        artifacts_root=artifacts_root,
        worktrees_root=worktree.path.parent,
    )
    artifacts_root_path = final_path.parent
    artifacts_root_path.mkdir(parents=True, exist_ok=True)
    staging_path = Path(
        tempfile.mkdtemp(
            prefix=f".{run_id}.incomplete-",
            dir=artifacts_root_path,
        )
    )
    try:
        source_directory = staging_path / "source"
        git_directory = staging_path / "git"
        (staging_path / "build").mkdir()
        source_directory.mkdir()
        git_directory.mkdir()
        _copy_supplemental_files(staging_path, supplemental_files or {})

        status = git_bytes(
            worktree.path,
            "status",
            "--porcelain=v1",
            "--no-renames",
            "--untracked-files=all",
            "--ignored=matching",
        )
        diff = git_bytes(
            worktree.path,
            "diff",
            "--binary",
            "--full-index",
            "--no-ext-diff",
            "--no-renames",
            worktree.baseline_commit,
            "--",
        )
        tracked_numstat = git_bytes(
            worktree.path,
            "diff",
            "--numstat",
            "--no-renames",
            "-z",
            worktree.baseline_commit,
            "--",
        )
        untracked = git_bytes(
            worktree.path,
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
        )
        ignored = git_bytes(
            worktree.path,
            "ls-files",
            "--others",
            "--ignored",
            "--exclude-standard",
            "-z",
        )

        _write_new(git_directory / "status.txt", status)
        _write_new(git_directory / "diff.patch", diff)
        git_version = _git_version()
        _write_new(
            staging_path / GIT_TRACKED_NUMSTAT_PATH,
            _model_json_bytes(_parse_tracked_numstat(tracked_numstat, git_version)),
        )
        _write_new(git_directory / "untracked.txt", _inventory_text(untracked))
        _write_new(git_directory / "ignored.txt", _inventory_text(ignored))
        _write_new(
            staging_path / GIT_UNTRACKED_NUMSTAT_PATH,
            _model_json_bytes(
                _capture_untracked_numstat(
                    worktree.path,
                    untracked + ignored,
                    exclusion_policy,
                    git_version,
                )
            ),
        )
        _write_new(
            git_directory / "baseline.txt",
            f"{worktree.baseline_commit}\n".encode("ascii"),
        )

        snapshot_path = staging_path / SOURCE_SNAPSHOT_PATH
        stats = create_source_snapshot(
            worktree.path,
            snapshot_path,
            exclusion_policy=exclusion_policy,
        )
        _write_new(
            staging_path / EXCLUDED_PATHS_PATH,
            _quoted_lines(stats.excluded_paths),
        )

        result_commit, reference = create_result_commit_and_ref(worktree, run_id)
        _write_new(
            git_directory / "result.txt",
            f"{result_commit}\n".encode("ascii"),
        )

        manifest = ArtifactManifest(
            artifact_manifest_id=f"{run_id}-manifest",
            run_id=run_id,
            experiment_id=experiment_id,
            baseline_repository=worktree.repository,
            baseline_commit=worktree.baseline_commit,
            result_commit=result_commit,
            result_ref=reference,
            source_snapshot_path=SOURCE_SNAPSHOT_PATH,
            source_snapshot_sha256=_sha256_file(snapshot_path),
            git_diff_path=GIT_DIFF_PATH,
            git_diff_sha256=_sha256_file(staging_path / GIT_DIFF_PATH),
            checksums_path=CHECKSUMS_PATH,
            creation_timestamp_utc=datetime.now(timezone.utc),
            preserved_file_count=stats.preserved_file_count,
            preserved_byte_count=stats.preserved_byte_count,
            excluded_file_count=stats.excluded_file_count,
            exclusion_policy=exclusion_policy,
            preservation_status="verifying",
        )
        _write_manifest(staging_path, manifest)
        _write_checksums(staging_path)
        _verify_artifact(staging_path, repository=worktree.repository, allow_verifying=True)

        manifest = manifest.model_copy(update={"preservation_status": "sealed"})
        _write_manifest(staging_path, manifest)
        _write_checksums(staging_path)
        _verify_artifact(staging_path, repository=worktree.repository)

        staging_path.rename(final_path)
        return manifest
    except Exception as exc:
        _mark_staging_failed(staging_path)
        if isinstance(exc, PreservationError):
            exc.incomplete_artifact_path = staging_path
            raise
        raise PreservationError(
            f"could not preserve worktree; incomplete artifacts retained at "
            f"{staging_path}: {exc}",
            worktree_path=worktree.path,
            incomplete_artifact_path=staging_path,
        ) from exc


def create_source_snapshot(
    source: Path,
    destination: Path,
    *,
    exclusion_policy: ExclusionPolicyRecord = DEFAULT_EXCLUSION_POLICY,
) -> SnapshotStats:
    """Create a deterministic uncompressed PAX tar of the resulting source tree."""
    source_path = source.resolve()
    entries: list[_SnapshotEntry] = []
    excluded_paths: list[str] = []
    excluded_count = _scan_source_tree(
        source_path,
        source_path,
        entries,
        excluded_paths,
        exclusion_policy,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise PreservationError(f"snapshot destination already exists: {destination}")

    preserved_count = 0
    preserved_bytes = 0
    with tarfile.open(destination, mode="x", format=tarfile.PAX_FORMAT) as archive:
        for entry in entries:
            info = _tar_info(entry)
            if entry.kind == "file":
                with entry.absolute_path.open("rb") as source_file:
                    archive.addfile(info, source_file)
                preserved_count += 1
                preserved_bytes += info.size
            else:
                archive.addfile(info)
                if entry.kind == "symlink":
                    preserved_count += 1

    return SnapshotStats(
        preserved_file_count=preserved_count,
        preserved_byte_count=preserved_bytes,
        excluded_file_count=excluded_count,
        excluded_paths=tuple(sorted(excluded_paths)),
    )


def verify_artifact(
    artifact_path: Path,
    *,
    repository: Path | None = None,
) -> ArtifactManifest:
    """Verify checksums, manifest coherence, archive contents, and result ref."""
    root = _artifact_root(artifact_path)
    return _verify_artifact(root, repository=repository)


def restore_artifact(artifact_path: Path, destination: Path) -> ArtifactManifest:
    """Verify and restore a source snapshot into a separate empty directory."""
    root = _artifact_root(artifact_path)
    manifest = verify_artifact(root)
    destination_path = destination.expanduser().resolve()
    if destination_path == root or _is_relative_to(destination_path, root):
        raise VerificationError("restore destination must be outside the artifact")
    if destination_path.exists():
        if not destination_path.is_dir():
            raise VerificationError(
                f"restore destination is not a directory: {destination_path}"
            )
        if any(destination_path.iterdir()):
            raise VerificationError(
                f"restore destination is not empty: {destination_path}"
            )
    destination_path.mkdir(parents=True, exist_ok=True)

    snapshot_path = root / manifest.source_snapshot_path
    with tarfile.open(snapshot_path, mode="r:") as archive:
        _validate_archive(archive, manifest)
        archive.extractall(destination_path, filter="data")
    return manifest


def _verify_artifact(
    root: Path,
    *,
    repository: Path | None,
    allow_verifying: bool = False,
) -> ArtifactManifest:
    manifest_path = root / MANIFEST_PATH
    checksums_path = root / CHECKSUMS_PATH
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise VerificationError(f"missing artifact manifest: {manifest_path}")
    if not checksums_path.is_file() or checksums_path.is_symlink():
        raise VerificationError(f"missing checksum file: {checksums_path}")
    try:
        manifest = ArtifactManifest.model_validate_json(manifest_path.read_bytes())
    except Exception as exc:
        raise VerificationError(f"invalid artifact manifest: {exc}") from exc
    allowed_statuses = {"sealed", "verifying"} if allow_verifying else {"sealed"}
    if manifest.preservation_status not in allowed_statuses:
        raise VerificationError(
            f"artifact preservation status is {manifest.preservation_status!r}"
        )

    checksums = _read_checksums(checksums_path)
    required_checksum_paths = {
        MANIFEST_PATH,
        manifest.source_snapshot_path,
        EXCLUDED_PATHS_PATH,
        "git/baseline.txt",
        "git/result.txt",
        "git/status.txt",
        manifest.git_diff_path,
        GIT_TRACKED_NUMSTAT_PATH,
        "git/untracked.txt",
        "git/ignored.txt",
        GIT_UNTRACKED_NUMSTAT_PATH,
    }
    missing_checksums = required_checksum_paths - checksums.keys()
    if missing_checksums:
        raise VerificationError(
            "missing required checksums: " + ", ".join(sorted(missing_checksums))
        )
    artifact_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != CHECKSUMS_PATH
    }
    if artifact_files != checksums.keys():
        unlisted = artifact_files - checksums.keys()
        missing = checksums.keys() - artifact_files
        details: list[str] = []
        if unlisted:
            details.append("unlisted files: " + ", ".join(sorted(unlisted)))
        if missing:
            details.append("listed files missing: " + ", ".join(sorted(missing)))
        raise VerificationError("checksum inventory mismatch (" + "; ".join(details) + ")")
    for relative_path, expected in checksums.items():
        artifact_file = root / relative_path
        if not artifact_file.is_file() or artifact_file.is_symlink():
            raise VerificationError(f"checksummed artifact is missing: {relative_path}")
        actual = _sha256_file(artifact_file)
        if actual != expected:
            raise VerificationError(
                f"checksum mismatch for {relative_path}: expected {expected}, got {actual}"
            )

    build_directory = root / "build"
    if not build_directory.is_dir() or build_directory.is_symlink():
        raise VerificationError(f"missing build artifact directory: {build_directory}")

    source_path = root / manifest.source_snapshot_path
    diff_path = root / manifest.git_diff_path
    if _sha256_file(source_path) != manifest.source_snapshot_sha256:
        raise VerificationError("source snapshot hash disagrees with manifest")
    if _sha256_file(diff_path) != manifest.git_diff_sha256:
        raise VerificationError("Git diff hash disagrees with manifest")
    if (root / "git/baseline.txt").read_text(encoding="ascii").strip() != (
        manifest.baseline_commit
    ):
        raise VerificationError("baseline.txt disagrees with manifest")
    if (root / "git/result.txt").read_text(encoding="ascii").strip() != (
        manifest.result_commit
    ):
        raise VerificationError("result.txt disagrees with manifest")

    with tarfile.open(source_path, mode="r:") as archive:
        _validate_archive(archive, manifest)

    repository_path = repository
    if repository_path is None and manifest.baseline_repository.is_dir():
        repository_path = manifest.baseline_repository
    if repository_path is not None:
        try:
            resolved = git_text(
                repository_path,
                "rev-parse",
                "--verify",
                "--end-of-options",
                f"{manifest.result_ref}^{{commit}}",
            ).strip()
        except GitOperationError as exc:
            raise VerificationError(
                f"result ref is unavailable: {manifest.result_ref}: {exc}"
            ) from exc
        if resolved != manifest.result_commit:
            raise VerificationError(
                f"result ref {manifest.result_ref} resolves to {resolved}, "
                f"not {manifest.result_commit}"
            )
    return manifest


def _validate_archive(archive: tarfile.TarFile, manifest: ArtifactManifest) -> None:
    names: set[str] = set()
    preserved_count = 0
    preserved_bytes = 0
    for member in archive.getmembers():
        _validate_relative_artifact_path(member.name)
        if member.name in names:
            raise VerificationError(f"duplicate path in source snapshot: {member.name}")
        names.add(member.name)
        path = PurePosixPath(member.name)
        if _is_excluded(path, member.isdir(), manifest.exclusion_policy):
            raise VerificationError(f"excluded path present in snapshot: {member.name}")
        if member.isfile():
            preserved_count += 1
            preserved_bytes += member.size
        elif member.issym():
            _validate_link_target(path, member.linkname)
            preserved_count += 1
        elif not member.isdir():
            raise VerificationError(
                f"unsupported member type in source snapshot: {member.name}"
            )
    if preserved_count != manifest.preserved_file_count:
        raise VerificationError(
            "source snapshot file count disagrees with manifest: "
            f"{preserved_count} != {manifest.preserved_file_count}"
        )
    if preserved_bytes != manifest.preserved_byte_count:
        raise VerificationError(
            "source snapshot byte count disagrees with manifest: "
            f"{preserved_bytes} != {manifest.preserved_byte_count}"
        )


def _scan_source_tree(
    root: Path,
    directory: Path,
    entries: list[_SnapshotEntry],
    excluded_paths: list[str],
    policy: ExclusionPolicyRecord,
) -> int:
    excluded_count = 0
    with os.scandir(directory) as iterator:
        children = sorted(iterator, key=lambda entry: os.fsencode(entry.name))
    for child in children:
        child_path = Path(child.path)
        relative = PurePosixPath(child_path.relative_to(root).as_posix())
        is_directory = child.is_dir(follow_symlinks=False)
        if _is_excluded(relative, is_directory, policy):
            excluded_paths.append(relative.as_posix())
            excluded_count += _count_file_entries(child_path)
            continue
        if child.is_symlink():
            _validate_link_target(relative, os.readlink(child_path))
            entries.append(_SnapshotEntry(child_path, relative, "symlink"))
        elif is_directory:
            entries.append(_SnapshotEntry(child_path, relative, "directory"))
            excluded_count += _scan_source_tree(
                root,
                child_path,
                entries,
                excluded_paths,
                policy,
            )
        elif child.is_file(follow_symlinks=False):
            entries.append(_SnapshotEntry(child_path, relative, "file"))
        else:
            raise PreservationError(
                f"unsupported special file in source tree: {child_path}"
            )
    return excluded_count


def _tar_info(entry: _SnapshotEntry) -> tarfile.TarInfo:
    metadata = entry.absolute_path.lstat()
    info = tarfile.TarInfo(entry.relative_path.as_posix())
    info.mode = stat.S_IMODE(metadata.st_mode)
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.pax_headers = {}
    if entry.kind == "file":
        info.type = tarfile.REGTYPE
        info.size = metadata.st_size
    elif entry.kind == "directory":
        info.type = tarfile.DIRTYPE
        info.size = 0
    else:
        info.type = tarfile.SYMTYPE
        info.size = 0
        info.linkname = os.readlink(entry.absolute_path)
    return info


def _count_file_entries(path: Path) -> int:
    if path.is_symlink() or not path.is_dir():
        return 1
    count = 0
    with os.scandir(path) as iterator:
        children = list(iterator)
    for child in children:
        count += _count_file_entries(Path(child.path))
    return count


def _is_excluded(
    path: PurePosixPath,
    is_directory: bool,
    policy: ExclusionPolicyRecord,
) -> bool:
    if any(part in policy.excluded_directory_names for part in path.parts):
        return True
    if not is_directory and path.name in policy.excluded_file_names:
        return True
    return not is_directory and any(
        path.name.endswith(suffix) for suffix in policy.excluded_file_suffixes
    )


def _validate_link_target(path: PurePosixPath, target: str) -> None:
    target_path = PurePosixPath(target)
    if target_path.is_absolute():
        raise PreservationError(f"absolute symlink is unsafe to restore: {path}")
    depth = len(path.parent.parts)
    for part in target_path.parts:
        if part == "..":
            depth -= 1
            if depth < 0:
                raise PreservationError(f"symlink escapes source tree: {path}")
        elif part not in {"", "."}:
            depth += 1


def _write_new(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        output.write(content)


def _copy_supplemental_files(
    staging_path: Path,
    supplemental_files: Mapping[str, Path],
) -> None:
    reserved_roots = {
        MANIFEST_PATH,
        CHECKSUMS_PATH,
        "source",
        "git",
        "build",
    }
    for relative_name, source in sorted(supplemental_files.items()):
        _validate_relative_artifact_path(relative_name)
        relative_path = PurePosixPath(relative_name)
        if relative_path.parts[0] in reserved_roots:
            raise PreservationError(
                f"supplemental artifact uses reserved path: {relative_name}"
            )
        configured_source = source.expanduser().absolute()
        if configured_source.is_symlink() or not configured_source.is_file():
            raise PreservationError(
                f"supplemental artifact is not a regular file: {configured_source}"
            )
        source_path = configured_source.resolve()
        destination = staging_path.joinpath(*relative_path.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source_path.open("rb") as input_stream, destination.open("xb") as output:
            for chunk in iter(lambda: input_stream.read(1024 * 1024), b""):
                output.write(chunk)


def _write_manifest(root: Path, manifest: ArtifactManifest) -> None:
    content = (
        json.dumps(
            manifest.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    temporary = root / ".manifest.json.tmp"
    temporary.write_bytes(content)
    os.replace(temporary, root / MANIFEST_PATH)


def _write_checksums(root: Path) -> None:
    paths = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.name != CHECKSUMS_PATH
        and not path.name.startswith(".manifest.json.tmp")
    )
    content = "".join(f"{_sha256_file(root / path)}  {path}\n" for path in paths)
    temporary = root / ".checksums.sha256.tmp"
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, root / CHECKSUMS_PATH)


def _read_checksums(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        digest, separator, relative_path = line.partition("  ")
        if (
            not separator
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise VerificationError(f"invalid checksum line {line_number}")
        _validate_relative_artifact_path(relative_path)
        if relative_path in checksums:
            raise VerificationError(f"duplicate checksum path: {relative_path}")
        checksums[relative_path] = digest
    return checksums


def _inventory_text(raw_paths: bytes) -> bytes:
    paths = [
        path.decode("utf-8", errors="surrogateescape")
        for path in raw_paths.split(b"\0")
        if path
    ]
    return _quoted_lines(tuple(sorted(paths, key=os.fsencode)))


def _capture_untracked_numstat(
    worktree: Path,
    raw_paths: bytes,
    policy: ExclusionPolicyRecord,
    git_version: str,
) -> GitNumstatRecord:
    paths = sorted({item for item in raw_paths.split(b"\0") if item})
    entries: list[GitNumstatEntry] = []
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
    }
    for raw_path in paths:
        path = raw_path.decode("utf-8", errors="surrogateescape")
        relative = PurePosixPath(path)
        absolute = worktree.joinpath(*relative.parts)
        if _is_excluded(relative, absolute.is_dir(), policy) or absolute.is_dir():
            continue
        completed = subprocess.run(
            ["git", "diff", "--no-index", "--numstat", "--", os.devnull, path],
            cwd=worktree,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
        )
        if completed.returncode not in {0, 1}:
            entries.append(
                GitNumstatEntry(
                    path=path,
                    binary=False,
                    availability="unavailable",
                    unavailable_reason="git-diff-no-index-failed",
                )
            )
            continue
        lines = completed.stdout.splitlines()
        if len(lines) != 1:
            entries.append(
                GitNumstatEntry(
                    path=path,
                    binary=False,
                    availability="unavailable",
                    unavailable_reason="unexpected-git-numstat-record-count",
                )
            )
            continue
        fields = lines[0].split(b"\t", 2)
        if len(fields) != 3:
            entries.append(
                GitNumstatEntry(
                    path=path,
                    binary=False,
                    availability="unavailable",
                    unavailable_reason="malformed-git-numstat",
                )
            )
            continue
        binary = fields[0] == b"-" or fields[1] == b"-"
        entries.append(
            GitNumstatEntry(
                path=path,
                lines_added=None if binary else int(fields[0]),
                lines_deleted=None if binary else int(fields[1]),
                binary=binary,
                availability="available",
            )
        )
    return GitNumstatRecord(
        algorithm="git-diff-no-index-numstat-v1",
        git_version=git_version,
        entries=tuple(entries),
    )


def _parse_tracked_numstat(raw: bytes, git_version: str) -> GitNumstatRecord:
    entries: list[GitNumstatEntry] = []
    for record in (item for item in raw.split(b"\0") if item):
        fields = record.split(b"\t", 2)
        if len(fields) != 3:
            raise PreservationError("Git produced malformed tracked numstat")
        path = fields[2].decode("utf-8", errors="surrogateescape")
        binary = fields[0] == b"-" or fields[1] == b"-"
        entries.append(
            GitNumstatEntry(
                path=path,
                lines_added=None if binary else int(fields[0]),
                lines_deleted=None if binary else int(fields[1]),
                binary=binary,
                availability="available",
            )
        )
    return GitNumstatRecord(
        algorithm="git-diff-numstat-no-renames-v1",
        git_version=git_version,
        entries=tuple(entries),
    )


def _git_version() -> str:
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
    }
    return subprocess.run(
        ["git", "--version"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        text=True,
    ).stdout.strip()


def _model_json_bytes(value: BaseModel) -> bytes:
    return (
        json.dumps(
            value.model_dump(mode="json"),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _quoted_lines(paths: tuple[str, ...]) -> bytes:
    return "".join(
        json.dumps(path, ensure_ascii=True) + "\n" for path in paths
    ).encode("ascii")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    return resolved.parent if resolved.name == MANIFEST_PATH else resolved


def _validate_new_run_destination(
    *,
    baseline: BaselineIdentity,
    run_id: str,
    artifacts_root: Path,
    worktrees_root: Path,
) -> Path:
    reference = result_ref(run_id)
    artifacts = artifacts_root.expanduser().resolve()
    worktrees = worktrees_root.expanduser().resolve()
    if _is_relative_to(artifacts, baseline.repository):
        raise PreservationError("artifacts_root must be outside the baseline repository")
    if _is_relative_to(worktrees, baseline.repository):
        raise PreservationError("worktrees_root must be outside the baseline repository")
    final_path = artifacts / run_id
    if final_path.exists():
        raise PreservationError(f"artifact destination already exists: {final_path}")
    if artifacts.exists() and any(artifacts.glob(f".{run_id}.incomplete-*")):
        raise PreservationError(f"incomplete artifact already exists for run {run_id}")
    if ref_exists(baseline.repository, reference):
        raise PreservationError(f"result ref already exists: {reference}")
    return final_path


def _mark_staging_failed(staging_path: Path) -> None:
    manifest_path = staging_path / MANIFEST_PATH
    if not manifest_path.is_file():
        return
    try:
        manifest = ArtifactManifest.model_validate_json(manifest_path.read_bytes())
        failed = manifest.model_copy(update={"preservation_status": "failed"})
        _write_manifest(staging_path, failed)
        _write_checksums(staging_path)
    except Exception:
        # Never conceal the original preservation error with recovery bookkeeping.
        return


def _validate_relative_artifact_path(value: str) -> None:
    path = PurePosixPath(value)
    if not value or value == "." or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe artifact-relative path: {value!r}")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True
