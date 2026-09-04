"""Immutable storage for the versioned context-analysis-v2 layer."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

from agent_bench.context_analysis import ContextAnalysis, ContextAnalysisError, derive_context_analysis
from agent_bench.metrics_storage import _json_bytes, _read_checksums
from agent_bench.preservation import verify_artifact

ANALYSIS_PATH = "context-analysis.json"
MANIFEST_PATH = "manifest.json"
CHECKSUMS_PATH = "checksums.sha256"


class ContextAnalysisStorageError(RuntimeError): pass


def store_context_analysis_artifact(*, source_artifact: Path, output_root: Path, analysis: ContextAnalysis) -> Path:
    source = source_artifact.resolve(); source_manifest = verify_artifact(source)
    if source_manifest.run_id != analysis.run_id: raise ContextAnalysisStorageError("analysis run ID does not match source artifact")
    final = output_root.resolve() / analysis.run_id / "context-analysis-v2"
    if final.exists(): raise ContextAnalysisStorageError(f"context analysis artifact already exists: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".context-analysis-v2.incomplete-", dir=final.parent))
    try:
        (staging / ANALYSIS_PATH).write_bytes(analysis.canonical_json_bytes())
        manifest = {
            "schema_version": "2.0.0", "analysis_artifact_id": f"{analysis.run_id}-context-analysis-artifact-v2",
            "run_id": analysis.run_id, "source_artifact_manifest_sha256": analysis.source_artifact_manifest_sha256,
            "analysis_id": analysis.analysis_id, "analysis_record_digest": analysis.record_digest,
            "analysis_sha256": _sha(staging / ANALYSIS_PATH), "storage_status": "sealed",
        }
        from agent_bench.models import canonical_sha256
        manifest["record_digest"] = canonical_sha256(manifest)
        (staging / MANIFEST_PATH).write_bytes(_json_bytes_dict(manifest))
        (staging / CHECKSUMS_PATH).write_text(f"{_sha(staging / MANIFEST_PATH)}  {MANIFEST_PATH}\n{_sha(staging / ANALYSIS_PATH)}  {ANALYSIS_PATH}\n", encoding="utf-8", newline="\n")
        verify_context_analysis_artifact(staging)
        staging.rename(final)
    except Exception:
        raise
    return final


def calculate_and_store_context_analysis(*, source_artifact: Path, output_root: Path) -> Path:
    return store_context_analysis_artifact(source_artifact=source_artifact, output_root=output_root, analysis=derive_context_analysis(source_artifact))


def verify_context_analysis_artifact(path: Path) -> ContextAnalysis:
    root = path.resolve()
    try:
        analysis = ContextAnalysis.model_validate_json((root / ANALYSIS_PATH).read_bytes())
        import json
        manifest = json.loads((root / MANIFEST_PATH).read_text(encoding="utf-8"))
        checksums = _read_checksums(root / CHECKSUMS_PATH)
    except Exception as exc: raise ContextAnalysisStorageError(f"invalid context analysis artifact: {exc}") from exc
    if set(checksums) != {MANIFEST_PATH, ANALYSIS_PATH}: raise ContextAnalysisStorageError("context-analysis checksum inventory is not exact")
    if any(_sha(root / name) != expected for name, expected in checksums.items()): raise ContextAnalysisStorageError("context-analysis checksum mismatch")
    expected = {key: value for key, value in manifest.items() if key != "record_digest"}
    from agent_bench.models import canonical_sha256
    if manifest.get("record_digest") != canonical_sha256(expected): raise ContextAnalysisStorageError("context-analysis manifest digest mismatch")
    if manifest.get("analysis_record_digest") != analysis.record_digest or manifest.get("analysis_sha256") != _sha(root / ANALYSIS_PATH): raise ContextAnalysisStorageError("context-analysis identity mismatch")
    return analysis


def _sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def _json_bytes_dict(value: dict[str, object]) -> bytes:
    import json
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
