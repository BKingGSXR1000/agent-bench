"""Immutable, separately checksummed storage for M4 metrics artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent_bench.metric_models import RunMetrics
from agent_bench.models import Identifier, Sha256, canonical_sha256
from agent_bench.preservation import (
    GIT_TRACKED_NUMSTAT_PATH,
    GIT_UNTRACKED_NUMSTAT_PATH,
    verify_artifact,
)

METRICS_PATH = "metrics.json"
METRICS_MANIFEST_PATH = "manifest.json"
METRICS_CHECKSUMS_PATH = "checksums.sha256"


class MetricsStorageError(RuntimeError):
    """Raised when immutable metric storage or verification fails."""


class MetricsArtifactManifest(BaseModel):
    """Versioned integrity link from analysis output to its sealed run input."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0.0"] = "1.0.0"
    metrics_artifact_id: str = Field(min_length=1)
    run_id: Identifier
    source_artifact_manifest_id: str = Field(min_length=1)
    source_artifact_manifest_sha256: Sha256
    source_run_manifest_sha256: Sha256
    metrics_id: str = Field(min_length=1)
    metrics_record_digest: Sha256
    metrics_path: Literal["metrics.json"] = METRICS_PATH
    metrics_sha256: Sha256
    checksums_path: Literal["checksums.sha256"] = METRICS_CHECKSUMS_PATH
    storage_status: Literal["sealed"] = "sealed"
    record_digest: Sha256

    @model_validator(mode="after")
    def validate_digest(self) -> MetricsArtifactManifest:
        expected = canonical_sha256(
            self.model_dump(mode="json", exclude={"record_digest"})
        )
        if self.record_digest != expected:
            raise ValueError("record_digest does not match metrics manifest")
        return self

    @classmethod
    def create(cls, **values: object) -> MetricsArtifactManifest:
        content = {"schema_version": "1.0.0", **values}
        draft = cls.model_construct(**content, record_digest="0" * 64)
        canonical = draft.model_dump(mode="json", exclude={"record_digest"})
        return cls.model_validate(
            {**canonical, "record_digest": canonical_sha256(canonical)}
        )


@dataclass(frozen=True)
class StoredMetrics:
    root: Path
    manifest: MetricsArtifactManifest
    metrics: RunMetrics


def store_metrics_artifact(
    *,
    source_artifact: Path,
    output_root: Path,
    metrics: RunMetrics,
) -> StoredMetrics:
    """Create one new immutable analysis artifact without touching its source."""
    source_root = source_artifact.expanduser().resolve()
    source_manifest = verify_artifact(source_root)
    if source_manifest.run_id != metrics.run_id:
        raise MetricsStorageError("metrics run ID does not match source artifact")
    expected_inputs = {
        "artifact manifest": (
            _sha256_file(source_root / "manifest.json"),
            metrics.input_identity.artifact_manifest_sha256,
        ),
        "run manifest": (
            _sha256_file(source_root / "run/manifest.json"),
            metrics.input_identity.run_manifest_sha256,
        ),
        "raw events": (
            _sha256_file(source_root / "raw/events.jsonl"),
            metrics.input_identity.raw_events_sha256,
        ),
        "normalized events": (
            _sha256_file(source_root / "normalized/events.jsonl"),
            metrics.input_identity.normalized_events_sha256,
        ),
        "source snapshot": (
            source_manifest.source_snapshot_sha256,
            metrics.input_identity.source_snapshot_sha256,
        ),
        "Git diff": (
            source_manifest.git_diff_sha256,
            metrics.input_identity.git_diff_sha256,
        ),
        "tracked Git numstat": (
            _sha256_file(source_root / GIT_TRACKED_NUMSTAT_PATH),
            metrics.input_identity.git_tracked_numstat_sha256,
        ),
        "non-tracked Git numstat": (
            _sha256_file(source_root / GIT_UNTRACKED_NUMSTAT_PATH),
            metrics.input_identity.git_untracked_numstat_sha256,
        ),
    }
    mismatches = [name for name, pair in expected_inputs.items() if pair[0] != pair[1]]
    if mismatches:
        raise MetricsStorageError(
            "metrics input identity does not match source artifact: "
            + ", ".join(mismatches)
        )
    output = output_root.expanduser().resolve()
    final = output / metrics.run_id / "metrics-v1"
    if _is_relative_to(final, source_root):
        raise MetricsStorageError("metrics artifact must be outside the sealed source artifact")
    if final.exists():
        raise MetricsStorageError(f"metrics artifact already exists: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".metrics-v1.incomplete-", dir=final.parent))
    try:
        metrics_bytes = metrics.canonical_json_bytes()
        (staging / METRICS_PATH).write_bytes(metrics_bytes)
        manifest = MetricsArtifactManifest.create(
            metrics_artifact_id=f"{metrics.run_id}-metrics-artifact-v1",
            run_id=metrics.run_id,
            source_artifact_manifest_id=source_manifest.artifact_manifest_id,
            source_artifact_manifest_sha256=_sha256_file(source_root / "manifest.json"),
            source_run_manifest_sha256=metrics.input_identity.run_manifest_sha256,
            metrics_id=metrics.metrics_id,
            metrics_record_digest=metrics.record_digest,
            metrics_sha256=hashlib.sha256(metrics_bytes).hexdigest(),
        )
        (staging / METRICS_MANIFEST_PATH).write_bytes(_json_bytes(manifest))
        _write_checksums(staging)
        _verify_metrics_root(staging)
        staging.rename(final)
    except Exception as exc:
        if isinstance(exc, MetricsStorageError):
            raise
        raise MetricsStorageError(
            f"could not seal metrics artifact; incomplete data retained at {staging}: {exc}"
        ) from exc
    return StoredMetrics(final, manifest, metrics)


def verify_metrics_artifact(path: Path) -> StoredMetrics:
    """Verify checksums, schema validation, and internal identity links."""
    root = _metrics_root(path)
    return _verify_metrics_root(root)


def _verify_metrics_root(root: Path) -> StoredMetrics:
    try:
        manifest = MetricsArtifactManifest.model_validate_json(
            (root / METRICS_MANIFEST_PATH).read_bytes()
        )
        metrics = RunMetrics.model_validate_json((root / METRICS_PATH).read_bytes())
        checksums = _read_checksums(root / METRICS_CHECKSUMS_PATH)
    except Exception as exc:
        raise MetricsStorageError(f"invalid metrics artifact: {exc}") from exc
    if set(checksums) != {METRICS_MANIFEST_PATH, METRICS_PATH}:
        raise MetricsStorageError("metrics checksum inventory is not exact")
    for relative, expected in checksums.items():
        if _sha256_file(root / relative) != expected:
            raise MetricsStorageError(f"checksum mismatch for {relative}")
    metrics_sha = _sha256_file(root / METRICS_PATH)
    if manifest.metrics_sha256 != metrics_sha:
        raise MetricsStorageError("metrics SHA256 disagrees with manifest")
    if manifest.run_id != metrics.run_id or manifest.metrics_id != metrics.metrics_id:
        raise MetricsStorageError("metrics identity disagrees with manifest")
    if manifest.metrics_record_digest != metrics.record_digest:
        raise MetricsStorageError("metrics record digest disagrees with manifest")
    if manifest.source_run_manifest_sha256 != metrics.input_identity.run_manifest_sha256:
        raise MetricsStorageError("source run identity link disagrees with metrics")
    return StoredMetrics(root, manifest, metrics)


def _json_bytes(value: BaseModel) -> bytes:
    return (
        json.dumps(
            value.model_dump(mode="json"),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _write_checksums(root: Path) -> None:
    content = "".join(
        f"{_sha256_file(root / name)}  {name}\n"
        for name in (METRICS_MANIFEST_PATH, METRICS_PATH)
    )
    temporary = root / ".checksums.sha256.tmp"
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, root / METRICS_CHECKSUMS_PATH)


def _read_checksums(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if not separator or len(digest) != 64 or relative in result:
            raise MetricsStorageError("invalid metrics checksum file")
        result[relative] = digest
    return result


def _metrics_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    return resolved.parent if resolved.name in {METRICS_PATH, METRICS_MANIFEST_PATH} else resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True
