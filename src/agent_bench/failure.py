"""Immutable evidence for failures before a normal run artifact can exist."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import Field, TypeAdapter, field_validator, model_validator

from agent_bench.backend import (
    BackendPreflightReport,
    FailureClass,
    ResolvedBackendInvocation,
)
from agent_bench.capture import CaptureCapabilities
from agent_bench.models import Identifier, PersistedModel, Sha256, canonical_sha256

FAILURE_MANIFEST = "manifest.json"
FAILURE_EVENTS = "events.jsonl"
FAILURE_STDOUT = "stdout.log"
FAILURE_STDERR = "stderr.log"
FAILURE_ENVIRONMENT = "environment.json"
FAILURE_CHECKSUMS = "checksums.sha256"
_RUN_ID_ADAPTER = TypeAdapter(Identifier)


class FailedRunEvidenceError(RuntimeError):
    """Raised when failed-run evidence cannot be created or verified."""


class FailedRunEvent(PersistedModel):
    """One deterministic event in a pre-task failure record."""

    event_id: str = Field(min_length=1)
    run_id: Identifier
    sequence: int = Field(ge=1)
    timestamp_utc: datetime
    event_type: Literal["preflight_check", "failure"]
    payload: dict[str, object]
    record_digest: Sha256

    @field_validator("timestamp_utc")
    @classmethod
    def require_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("failure event timestamp must include an offset")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def validate_record_digest(self) -> FailedRunEvent:
        expected = canonical_sha256(
            self.model_dump(
                mode="json",
                exclude={"record_digest"},
                exclude_computed_fields=True,
            )
        )
        if self.record_digest != expected:
            raise ValueError("failure event digest mismatch")
        return self

    @classmethod
    def create(cls, **values: object) -> FailedRunEvent:
        draft = cls.model_construct(
            schema_version="1.0.0",
            record_digest="0" * 64,
            **values,
        )
        content = draft.model_dump(
            mode="json",
            exclude={"record_digest"},
            exclude_computed_fields=True,
        )
        return cls.model_validate(
            {**content, "record_digest": canonical_sha256(content)}
        )


class FailureEnvironmentRecord(PersistedModel):
    """Safe observed environment and configured command at failure time."""

    run_id: Identifier
    backend_profile_digest: Sha256
    preflight: BackendPreflightReport
    invocation: ResolvedBackendInvocation
    capture_capabilities: CaptureCapabilities


class FailedRunManifest(PersistedModel):
    """Index for one sealed failed-run evidence directory."""

    failure_manifest_id: str = Field(min_length=1)
    run_id: Identifier
    failure_class: FailureClass
    reason: str = Field(min_length=1)
    created_at: datetime
    evidence_state: Literal["sealed"] = "sealed"
    events_path: Literal["events.jsonl"] = FAILURE_EVENTS
    stdout_path: Literal["stdout.log"] = FAILURE_STDOUT
    stderr_path: Literal["stderr.log"] = FAILURE_STDERR
    environment_path: Literal["environment.json"] = FAILURE_ENVIRONMENT
    checksums_path: Literal["checksums.sha256"] = FAILURE_CHECKSUMS
    record_digest: Sha256

    @field_validator("created_at")
    @classmethod
    def require_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("failed-run creation timestamp must include an offset")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def validate_record_digest(self) -> FailedRunManifest:
        expected = canonical_sha256(
            self.model_dump(
                mode="json",
                exclude={"record_digest"},
                exclude_computed_fields=True,
            )
        )
        if self.record_digest != expected:
            raise ValueError("failed-run manifest digest mismatch")
        return self

    @classmethod
    def create(cls, **values: object) -> FailedRunManifest:
        draft = cls.model_construct(
            schema_version="1.0.0",
            record_digest="0" * 64,
            **values,
        )
        content = draft.model_dump(
            mode="json",
            exclude={"record_digest"},
            exclude_computed_fields=True,
        )
        return cls.model_validate(
            {**content, "record_digest": canonical_sha256(content)}
        )


@dataclass(frozen=True)
class FailedRunEvidence:
    root: Path
    manifest: FailedRunManifest
    environment: FailureEnvironmentRecord
    events: tuple[FailedRunEvent, ...]


def preserve_failed_run(
    *,
    runs_root: Path,
    run_id: str,
    failure_class: FailureClass,
    reason: str,
    environment: FailureEnvironmentRecord,
    stdout: bytes = b"",
    stderr: bytes = b"",
    created_at: datetime | None = None,
) -> FailedRunEvidence:
    """Seal evidence at ``runs/<run-id>/failure`` without overwriting anything."""
    run_id = _RUN_ID_ADAPTER.validate_python(run_id)
    if environment.run_id != run_id:
        raise FailedRunEvidenceError("failure environment run ID mismatch")
    root = runs_root.expanduser().resolve()
    final_run = root / run_id
    if final_run.exists():
        raise FailedRunEvidenceError(f"run evidence already exists: {final_run}")
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{run_id}.failure.incomplete-", dir=root))
    failure = staging / "failure"
    failure.mkdir()
    timestamp = (created_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    events = _failure_events(run_id, failure_class, reason, environment.preflight, timestamp)
    manifest = FailedRunManifest.create(
        failure_manifest_id=f"{run_id}-failure-v1",
        run_id=run_id,
        failure_class=failure_class,
        reason=reason,
        created_at=timestamp,
    )
    try:
        _write_new(failure / FAILURE_EVENTS, b"".join(_json_line(event) for event in events))
        _write_new(failure / FAILURE_STDOUT, stdout)
        _write_new(failure / FAILURE_STDERR, stderr)
        _write_new(failure / FAILURE_ENVIRONMENT, _json_line(environment))
        _write_new(failure / FAILURE_MANIFEST, _json_line(manifest))
        _write_checksums(failure)
        verified = _verify_failure_root(failure)
        staging.rename(final_run)
        return FailedRunEvidence(
            root=final_run / "failure",
            manifest=verified.manifest,
            environment=verified.environment,
            events=verified.events,
        )
    except Exception as exc:
        if isinstance(exc, FailedRunEvidenceError):
            raise
        raise FailedRunEvidenceError(
            f"could not preserve failed-run evidence; incomplete data retained at {staging}: {exc}"
        ) from exc


def verify_failed_run(path: Path) -> FailedRunEvidence:
    """Verify schemas, identities, and the exact failure checksum inventory."""
    resolved = path.expanduser().resolve()
    failure = resolved if resolved.name == "failure" else resolved / "failure"
    return _verify_failure_root(failure)


def _verify_failure_root(root: Path) -> FailedRunEvidence:
    try:
        manifest = FailedRunManifest.model_validate_json(
            (root / FAILURE_MANIFEST).read_bytes()
        )
        environment = FailureEnvironmentRecord.model_validate_json(
            (root / FAILURE_ENVIRONMENT).read_bytes()
        )
        event_lines = (root / FAILURE_EVENTS).read_bytes().splitlines()
        events = tuple(FailedRunEvent.model_validate_json(line) for line in event_lines)
        checksums = _read_checksums(root / FAILURE_CHECKSUMS)
    except Exception as exc:
        raise FailedRunEvidenceError(f"invalid failed-run evidence: {exc}") from exc
    expected_paths = {
        FAILURE_MANIFEST,
        FAILURE_EVENTS,
        FAILURE_STDOUT,
        FAILURE_STDERR,
        FAILURE_ENVIRONMENT,
    }
    if set(checksums) != expected_paths:
        raise FailedRunEvidenceError("failed-run checksum inventory is not exact")
    for relative, expected in checksums.items():
        if sha256_path(root / relative) != expected:
            raise FailedRunEvidenceError(f"checksum mismatch for {relative}")
    if environment.run_id != manifest.run_id:
        raise FailedRunEvidenceError("failure environment run ID mismatch")
    if any(event.run_id != manifest.run_id for event in events):
        raise FailedRunEvidenceError("failure event run ID mismatch")
    if [event.sequence for event in events] != list(range(1, len(events) + 1)):
        raise FailedRunEvidenceError("failure event sequence is not contiguous")
    return FailedRunEvidence(root, manifest, environment, events)


def _failure_events(
    run_id: str,
    failure_class: FailureClass,
    reason: str,
    preflight: BackendPreflightReport,
    timestamp: datetime,
) -> tuple[FailedRunEvent, ...]:
    events: list[FailedRunEvent] = []
    for check in preflight.checks:
        sequence = len(events) + 1
        events.append(
            FailedRunEvent.create(
                event_id=f"{run_id}:failure:{sequence:06d}",
                run_id=run_id,
                sequence=sequence,
                timestamp_utc=timestamp,
                event_type="preflight_check",
                payload=check.model_dump(mode="json", exclude_computed_fields=True),
            )
        )
    sequence = len(events) + 1
    events.append(
        FailedRunEvent.create(
            event_id=f"{run_id}:failure:{sequence:06d}",
            run_id=run_id,
            sequence=sequence,
            timestamp_utc=timestamp,
            event_type="failure",
            payload={"failure_class": failure_class, "reason": reason},
        )
    )
    return tuple(events)


def _write_new(path: Path, content: bytes) -> None:
    with path.open("xb") as output:
        output.write(content)
        output.flush()
        os.fsync(output.fileno())


def _write_checksums(root: Path) -> None:
    names = (
        FAILURE_ENVIRONMENT,
        FAILURE_EVENTS,
        FAILURE_MANIFEST,
        FAILURE_STDERR,
        FAILURE_STDOUT,
    )
    content = "".join(f"{sha256_path(root / name)}  {name}\n" for name in names)
    _write_new(root / FAILURE_CHECKSUMS, content.encode("ascii"))


def _read_checksums(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        digest, separator, relative = line.partition("  ")
        if (
            not separator
            or len(digest) != 64
            or relative in result
            or "/" in relative
        ):
            raise FailedRunEvidenceError("invalid failed-run checksum listing")
        result[relative] = digest
    return result


def _json_line(value: PersistedModel) -> bytes:
    return (
        json.dumps(
            value.model_dump(mode="json", exclude_computed_fields=True),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
