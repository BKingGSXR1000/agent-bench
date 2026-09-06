"""Immutable M13 functional-validation-v1 artifacts derived from sealed runs."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent_bench.functional import FunctionalValidationResult, evaluate_workspace, load_functional_scenario
from agent_bench.metrics_storage import _read_checksums
from agent_bench.models import FunctionalScenarioAssociation, Identifier, PortableBaselineIdentity, Sha256, canonical_sha256
from agent_bench.preservation import restore_artifact, verify_artifact

RESULT_PATH = "functional-validation.json"
MANIFEST_PATH = "manifest.json"
CHECKSUMS_PATH = "checksums.sha256"


class FunctionalValidationStorageError(RuntimeError):
    """Sealed-result validation or functional-artifact storage failed."""


class FunctionalValidationArtifact(BaseModel):
    """One source-linked, immutable functional acceptance record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0.0"] = "1.0.0"
    functional_validation_id: str = Field(min_length=1)
    run_id: Identifier
    experiment_id: Identifier
    source_artifact_manifest_sha256: Sha256
    source_snapshot_sha256: Sha256
    source_run_manifest_sha256: Sha256
    prompt_id: Identifier
    prompt_sha256: Sha256
    scenario: FunctionalScenarioAssociation
    frozen_baseline: PortableBaselineIdentity
    validation_status: Literal["pass", "fail", "error", "unavailable"]
    acceptance_score_numerator: int | None
    acceptance_score_denominator: int | None
    functional_result: FunctionalValidationResult
    record_digest: Sha256

    @model_validator(mode="after")
    def validate_record(self) -> "FunctionalValidationArtifact":
        if self.validation_status in {"error", "unavailable"}:
            if self.acceptance_score_numerator is not None or self.acceptance_score_denominator is not None:
                raise ValueError("validator error/unavailable artifacts must not claim an acceptance score")
        elif (self.acceptance_score_numerator, self.acceptance_score_denominator) != (
            self.functional_result.score_numerator, self.functional_result.score_denominator,
        ):
            raise ValueError("acceptance score disagrees with functional result")
        if self.frozen_baseline != self.functional_result.baseline_identity:
            raise ValueError("frozen baseline disagrees with functional result")
        expected = canonical_sha256(self.model_dump(mode="json", exclude={"record_digest"}, exclude_computed_fields=True))
        if self.record_digest != expected:
            raise ValueError("record_digest does not match functional validation artifact")
        return self

    @classmethod
    def create(cls, **values: object) -> "FunctionalValidationArtifact":
        content = {"schema_version": "1.0.0", **values}
        draft = cls.model_construct(**content, record_digest="0" * 64)
        canonical = draft.model_dump(mode="json", exclude={"record_digest"}, exclude_computed_fields=True)
        return cls.model_validate({**canonical, "record_digest": canonical_sha256(canonical)})


class FunctionalValidationManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0.0"] = "1.0.0"
    functional_validation_artifact_id: str = Field(min_length=1)
    run_id: Identifier
    source_artifact_manifest_sha256: Sha256
    source_snapshot_sha256: Sha256
    functional_validation_id: str = Field(min_length=1)
    functional_validation_record_digest: Sha256
    functional_validation_sha256: Sha256
    storage_status: Literal["sealed"] = "sealed"
    record_digest: Sha256

    @model_validator(mode="after")
    def validate_record(self) -> "FunctionalValidationManifest":
        expected = canonical_sha256(self.model_dump(mode="json", exclude={"record_digest"}, exclude_computed_fields=True))
        if self.record_digest != expected:
            raise ValueError("record_digest does not match functional validation manifest")
        return self

    @classmethod
    def create(cls, **values: object) -> "FunctionalValidationManifest":
        content = {"schema_version": "1.0.0", **values}
        draft = cls.model_construct(**content, record_digest="0" * 64)
        canonical = draft.model_dump(mode="json", exclude={"record_digest"}, exclude_computed_fields=True)
        return cls.model_validate({**canonical, "record_digest": canonical_sha256(canonical)})


@dataclass(frozen=True)
class StoredFunctionalValidation:
    root: Path
    manifest: FunctionalValidationManifest
    result: FunctionalValidationArtifact


def validate_and_store_functional_artifact(
    *, source_artifact: Path, output_root: Path, run_id: str, experiment_id: str,
    association: FunctionalScenarioAssociation,
) -> StoredFunctionalValidation:
    """Restore a verified snapshot, validate it, then seal a new analysis artifact."""
    source = source_artifact.expanduser().resolve()
    source_manifest = verify_artifact(source)
    if source_manifest.run_id != run_id:
        raise FunctionalValidationStorageError("functional validation run ID does not match source artifact")
    scenario = load_functional_scenario(association.scenario_definition)
    if scenario.scenario_id != association.scenario_id:
        raise FunctionalValidationStorageError("functional scenario ID does not match pinned association")
    if _sha(association.scenario_definition) != association.scenario_definition_sha256 or _sha(scenario.validator) != association.validator_sha256:
        raise FunctionalValidationStorageError("functional scenario or validator digest mismatch")
    output = output_root.expanduser().resolve()
    final = output / run_id / "functional-validation-v1"
    if final.exists():
        raise FunctionalValidationStorageError(f"functional validation artifact already exists: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".functional-restore-", dir=final.parent) as temporary:
        restored = Path(temporary) / "source"
        try:
            restored_manifest = restore_artifact(source, restored)
            if restored_manifest.source_snapshot_sha256 != source_manifest.source_snapshot_sha256:
                raise FunctionalValidationStorageError("restored source snapshot identity differs from sealed artifact")
            raw_result = evaluate_workspace(scenario, restored, run_id)
        except FunctionalValidationStorageError:
            raise
        except Exception as exc:
            raise FunctionalValidationStorageError(f"could not restore or run functional validator: {exc}") from exc
    status = _status(raw_result)
    numerator = raw_result.score_numerator if status in {"pass", "fail"} else None
    denominator = raw_result.score_denominator if status in {"pass", "fail"} else None
    try:
        run_manifest = json.loads((source / "run/manifest.json").read_text(encoding="utf-8"))
        prompt_id = run_manifest["prompt_id"]
        prompt_sha256 = run_manifest["prompt_sha256"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise FunctionalValidationStorageError(f"sealed source run manifest is invalid: {exc}") from exc
    if run_manifest.get("run_id") != run_id or run_manifest.get("experiment_id") != experiment_id:
        raise FunctionalValidationStorageError("sealed source run manifest does not match functional validation identity")
    result = FunctionalValidationArtifact.create(
        functional_validation_id=f"{run_id}-functional-validation-v1", run_id=run_id,
        experiment_id=experiment_id, source_artifact_manifest_sha256=_sha(source / "manifest.json"),
        source_snapshot_sha256=source_manifest.source_snapshot_sha256,
        source_run_manifest_sha256=_sha(source / "run/manifest.json"), scenario=association,
        prompt_id=prompt_id, prompt_sha256=prompt_sha256,
        frozen_baseline=raw_result.baseline_identity,
        validation_status=status, acceptance_score_numerator=numerator,
        acceptance_score_denominator=denominator, functional_result=raw_result,
    )
    return store_functional_validation_artifact(source_artifact=source, output_root=output, result=result)


def store_functional_validation_artifact(*, source_artifact: Path, output_root: Path, result: FunctionalValidationArtifact) -> StoredFunctionalValidation:
    source = source_artifact.expanduser().resolve(); source_manifest = verify_artifact(source)
    if source_manifest.run_id != result.run_id:
        raise FunctionalValidationStorageError("functional result run ID does not match source artifact")
    if result.source_artifact_manifest_sha256 != _sha(source / "manifest.json") or result.source_snapshot_sha256 != source_manifest.source_snapshot_sha256:
        raise FunctionalValidationStorageError("functional result source identity does not match sealed artifact")
    final = output_root.expanduser().resolve() / result.run_id / "functional-validation-v1"
    if final.exists():
        raise FunctionalValidationStorageError(f"functional validation artifact already exists: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".functional-validation-v1.incomplete-", dir=final.parent))
    try:
        result_bytes = _json_bytes(result)
        (staging / RESULT_PATH).write_bytes(result_bytes)
        manifest = FunctionalValidationManifest.create(
            functional_validation_artifact_id=f"{result.run_id}-functional-validation-artifact-v1", run_id=result.run_id,
            source_artifact_manifest_sha256=result.source_artifact_manifest_sha256,
            source_snapshot_sha256=result.source_snapshot_sha256,
            functional_validation_id=result.functional_validation_id,
            functional_validation_record_digest=result.record_digest,
            functional_validation_sha256=hashlib.sha256(result_bytes).hexdigest(),
        )
        (staging / MANIFEST_PATH).write_bytes(_json_bytes(manifest))
        _write_checksums(staging)
        _verify_root(staging)
        staging.rename(final)
    except Exception as exc:
        if isinstance(exc, FunctionalValidationStorageError):
            raise
        raise FunctionalValidationStorageError(f"could not seal functional validation artifact; incomplete data retained at {staging}: {exc}") from exc
    return StoredFunctionalValidation(final, manifest, result)


def verify_functional_validation_artifact(path: Path) -> StoredFunctionalValidation:
    return _verify_root(path.expanduser().resolve())


def _verify_root(root: Path) -> StoredFunctionalValidation:
    try:
        result = FunctionalValidationArtifact.model_validate_json((root / RESULT_PATH).read_bytes())
        manifest = FunctionalValidationManifest.model_validate_json((root / MANIFEST_PATH).read_bytes())
        checksums = _read_checksums(root / CHECKSUMS_PATH)
    except Exception as exc:
        raise FunctionalValidationStorageError(f"invalid functional validation artifact: {exc}") from exc
    if set(checksums) != {RESULT_PATH, MANIFEST_PATH}:
        raise FunctionalValidationStorageError("functional validation checksum inventory is not exact")
    if any(_sha(root / name) != expected for name, expected in checksums.items()):
        raise FunctionalValidationStorageError("functional validation checksum mismatch")
    if manifest.functional_validation_sha256 != _sha(root / RESULT_PATH):
        raise FunctionalValidationStorageError("functional validation SHA256 disagrees with manifest")
    if manifest.run_id != result.run_id or manifest.functional_validation_id != result.functional_validation_id:
        raise FunctionalValidationStorageError("functional validation identity disagrees with manifest")
    if manifest.functional_validation_record_digest != result.record_digest:
        raise FunctionalValidationStorageError("functional validation record digest disagrees with manifest")
    return StoredFunctionalValidation(root, manifest, result)


def _status(result: FunctionalValidationResult) -> Literal["pass", "fail", "error", "unavailable"]:
    if result.error_tests:
        return "error"
    if result.unavailable_tests:
        return "unavailable"
    return "pass" if result.hard_gate_pass else "fail"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_bytes(value: BaseModel) -> bytes:
    return (json.dumps(value.model_dump(mode="json", exclude_computed_fields=True), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _write_checksums(root: Path) -> None:
    temporary = root / ".checksums.sha256.tmp"
    temporary.write_text("".join(f"{_sha(root / name)}  {name}\n" for name in (MANIFEST_PATH, RESULT_PATH)), encoding="utf-8", newline="\n")
    os.replace(temporary, root / CHECKSUMS_PATH)
