"""Immutable storage for timing-provenance-v1 analyses."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

from agent_bench.metrics_storage import _read_checksums
from agent_bench.models import canonical_sha256
from agent_bench.preservation import verify_artifact
from agent_bench.timing_provenance import TimingProvenanceAnalysis, derive_hermes_timing_provenance

ANALYSIS_PATH = "timing-provenance.json"
MANIFEST_PATH = "manifest.json"
CHECKSUMS_PATH = "checksums.sha256"


class TimingProvenanceStorageError(RuntimeError):
    """Raised when timing-provenance storage is invalid."""


def store_timing_provenance_artifact(*, source_artifact: Path, output_root: Path, analysis: TimingProvenanceAnalysis) -> Path:
    source = source_artifact.resolve()
    source_manifest = verify_artifact(source)
    if source_manifest.run_id != analysis.run_id:
        raise TimingProvenanceStorageError("analysis run ID does not match source artifact")
    final = output_root.resolve() / analysis.run_id / "timing-provenance-v1"
    if final.exists():
        raise TimingProvenanceStorageError(f"timing provenance artifact already exists: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".timing-provenance-v1.incomplete-", dir=final.parent))
    try:
        (staging / ANALYSIS_PATH).write_bytes(analysis.canonical_json_bytes())
        manifest = {
            "schema_version": "1.0.0",
            "analysis_artifact_id": f"{analysis.run_id}-timing-provenance-artifact-v1",
            "run_id": analysis.run_id,
            "source_artifact_manifest_sha256": analysis.source_artifact_manifest_sha256,
            "analysis_id": analysis.analysis_id,
            "analysis_record_digest": analysis.record_digest,
            "analysis_sha256": _sha(staging / ANALYSIS_PATH),
            "storage_status": "sealed",
        }
        manifest["record_digest"] = canonical_sha256(manifest)
        (staging / MANIFEST_PATH).write_bytes(_json_bytes(manifest))
        (staging / CHECKSUMS_PATH).write_text(
            f"{_sha(staging / MANIFEST_PATH)}  {MANIFEST_PATH}\n"
            f"{_sha(staging / ANALYSIS_PATH)}  {ANALYSIS_PATH}\n",
            encoding="utf-8",
            newline="\n",
        )
        verify_timing_provenance_artifact(staging)
        staging.rename(final)
    except Exception:
        raise
    return final


def calculate_and_store_hermes_timing_provenance(*, source_artifact: Path, output_root: Path) -> Path:
    return store_timing_provenance_artifact(source_artifact=source_artifact, output_root=output_root, analysis=derive_hermes_timing_provenance(source_artifact))


def verify_timing_provenance_artifact(path: Path) -> TimingProvenanceAnalysis:
    root = path.resolve()
    try:
        analysis = TimingProvenanceAnalysis.model_validate_json((root / ANALYSIS_PATH).read_bytes())
        manifest = json.loads((root / MANIFEST_PATH).read_text(encoding="utf-8"))
        checksums = _read_checksums(root / CHECKSUMS_PATH)
    except Exception as exc:
        raise TimingProvenanceStorageError(f"invalid timing provenance artifact: {exc}") from exc
    if set(checksums) != {MANIFEST_PATH, ANALYSIS_PATH}:
        raise TimingProvenanceStorageError("timing provenance checksum inventory is not exact")
    if any(_sha(root / name) != expected for name, expected in checksums.items()):
        raise TimingProvenanceStorageError("timing provenance checksum mismatch")
    expected = {key: value for key, value in manifest.items() if key != "record_digest"}
    if manifest.get("record_digest") != canonical_sha256(expected):
        raise TimingProvenanceStorageError("timing provenance manifest digest mismatch")
    if manifest.get("analysis_record_digest") != analysis.record_digest or manifest.get("analysis_sha256") != _sha(root / ANALYSIS_PATH):
        raise TimingProvenanceStorageError("timing provenance identity mismatch")
    return analysis


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_bytes(value: dict[str, object]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
