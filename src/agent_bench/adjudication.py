"""Append-only local human functional adjudications.

This is intentionally separate from both sealed run evidence and automated
validator artifacts.  A later revocation never deletes an earlier decision.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from agent_bench.models import Identifier, Sha256, canonical_sha256
from agent_bench.preservation import verify_artifact

ADJUDICATION_DIRECTORY = "adjudications"
METHOD = "local_human_visual_verification_v1"


class AdjudicationError(RuntimeError):
    """Manual adjudication evidence cannot safely be read or written."""


class ManualAdjudication(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0.0"] = "1.0.0"
    run_id: Identifier
    decision: Literal["pass", "revoked"]
    method: Literal["local_human_visual_verification_v1"] = METHOD
    timestamp_utc: str
    source_artifact_manifest_sha256: Sha256
    source_snapshot_sha256: Sha256
    revision: int = Field(ge=1)
    record_digest: Sha256

    @classmethod
    def create(cls, **values: object) -> "ManualAdjudication":
        content = {"schema_version": "1.0.0", "method": METHOD, **values}
        draft = cls.model_construct(**content, record_digest="0" * 64)
        body = draft.model_dump(mode="json", exclude={"record_digest"}, exclude_computed_fields=True)
        return cls.model_validate({**body, "record_digest": canonical_sha256(body)})


def adjudication_root(experiment_output: Path, run_id: str) -> Path:
    return experiment_output.expanduser().resolve() / ADJUDICATION_DIRECTORY / run_id


def load_adjudications(experiment_output: Path, run_id: str) -> tuple[ManualAdjudication, ...]:
    root = adjudication_root(experiment_output, run_id)
    if not root.exists():
        return ()
    values: list[ManualAdjudication] = []
    for path in sorted(root.glob("revision-*.json")):
        try:
            item = ManualAdjudication.model_validate_json(path.read_bytes())
        except Exception as exc:
            raise AdjudicationError(f"invalid manual adjudication {path}: {exc}") from exc
        expected = canonical_sha256(item.model_dump(mode="json", exclude={"record_digest"}, exclude_computed_fields=True))
        if item.record_digest != expected or item.run_id != run_id:
            raise AdjudicationError(f"manual adjudication integrity failure: {path}")
        values.append(item)
    if any(item.revision != index for index, item in enumerate(values, 1)):
        raise AdjudicationError("manual adjudication revisions must be contiguous and append-only")
    return tuple(values)


def active_adjudication(experiment_output: Path, run_id: str) -> ManualAdjudication | None:
    values = load_adjudications(experiment_output, run_id)
    return values[-1] if values and values[-1].decision == "pass" else None


def append_adjudication(*, experiment_output: Path, artifact_root: Path, run_id: str, decision: Literal["pass", "revoked"]) -> ManualAdjudication:
    artifact = verify_artifact(artifact_root)
    if artifact.run_id != run_id:
        raise AdjudicationError("manual adjudication run ID does not match preserved artifact")
    root = adjudication_root(experiment_output, run_id)
    root.mkdir(parents=True, exist_ok=True)
    prior = load_adjudications(experiment_output, run_id)
    item = ManualAdjudication.create(
        run_id=run_id, decision=decision, timestamp_utc=datetime.now(timezone.utc).isoformat(),
        source_artifact_manifest_sha256=hashlib.sha256((artifact_root / "manifest.json").read_bytes()).hexdigest(),
        source_snapshot_sha256=artifact.source_snapshot_sha256, revision=len(prior) + 1,
    )
    destination = root / f"revision-{item.revision:03d}.json"
    try:
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise AdjudicationError("manual adjudication revision already exists") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(item.model_dump(mode="json", exclude_computed_fields=True), handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
    return item
