"""Early, generic supervisor evidence for controlled benchmark launches.

The record is written before backend or harness construction.  It is deliberately
small and does not replace sealed run or FailedRunEvidence artifacts.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agent_bench.models import Identifier, Sha256, canonical_sha256


SUPERVISOR_SCHEMA_VERSION: Literal["1.0.0"] = "1.0.0"
_SAFE_ENVIRONMENT_KEYS = (
    "HOME", "LANG", "LC_ALL", "PATH", "PYTHONPATH", "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME",
)


class SupervisorError(RuntimeError):
    """Raised after deterministic startup evidence has been written."""


class SupervisorRecord(BaseModel):
    """Immutable one-stage supervisor record with a canonical digest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0.0"] = SUPERVISOR_SCHEMA_VERSION
    run_id: Identifier
    stage: Literal["initialized", "dry_startup_complete", "failure"]
    timestamp_utc: datetime
    supervisor_pid: int = Field(ge=1)
    argv: tuple[str, ...]
    cwd: str
    python_executable: str
    python_version: str
    python_sha256: Sha256 | None = None
    intended_output_root: str
    sanitized_environment: dict[str, str | None]
    environment_digest: Sha256
    failure_type: str | None = None
    failure_message: str | None = None
    record_digest: Sha256

    @field_validator("timestamp_utc")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must include a UTC offset")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def _valid(self) -> "SupervisorRecord":
        expected = canonical_sha256(self.model_dump(mode="json", exclude={"record_digest"}))
        if expected != self.record_digest:
            raise ValueError("supervisor record digest mismatch")
        if self.stage == "failure" and (not self.failure_type or self.failure_message is None):
            raise ValueError("failure record requires failure type and message")
        if self.stage != "failure" and (self.failure_type is not None or self.failure_message is not None):
            raise ValueError("non-failure record cannot include failure details")
        return self

    @classmethod
    def create(cls, **values: object) -> "SupervisorRecord":
        draft = cls.model_construct(schema_version=SUPERVISOR_SCHEMA_VERSION, record_digest="0" * 64, **values)
        content = draft.model_dump(mode="json", exclude={"record_digest"})
        return cls.model_validate({**content, "record_digest": canonical_sha256(content)})


class SupervisorEvidence:
    """Append-only early-launch evidence rooted outside an eventual run artifact."""

    def __init__(self, root: Path, run_id: str, base: dict[str, object]) -> None:
        self.root = root
        self.run_id = run_id
        self._base = base

    def mark(self, stage: Literal["initialized", "dry_startup_complete", "failure"], *, error: BaseException | None = None) -> SupervisorRecord:
        record = SupervisorRecord.create(
            **self._base,
            stage=stage,
            timestamp_utc=datetime.now(timezone.utc),
            failure_type=type(error).__name__ if error is not None else None,
            failure_message=str(error) if error is not None else None,
        )
        sequence = len(tuple(self.root.glob("*.json"))) + 1
        _write_new(self.root / f"{sequence:02d}-{stage}.json", _json_bytes(record))
        if error is not None:
            _write_new(self.root / "stderr.log", traceback.format_exc().encode("utf-8"))
        return record


def initialize_supervisor(*, output_root: Path, run_id: str, argv: tuple[str, ...]) -> SupervisorEvidence:
    """Create the fsynced initialization record before any child process exists."""
    final = output_root.expanduser().resolve() / "supervisor" / run_id
    if final.exists():
        raise SupervisorError(f"supervisor evidence already exists: {final}")
    final.mkdir(parents=True, exist_ok=False)
    environment = {key: os.environ.get(key) for key in _SAFE_ENVIRONMENT_KEYS}
    executable = Path(sys.executable)
    evidence = SupervisorEvidence(final, run_id, {
        "run_id": run_id,
        "supervisor_pid": os.getpid(),
        "argv": argv,
        "cwd": str(Path.cwd()),
        "python_executable": str(executable),
        "python_version": sys.version,
        "python_sha256": _sha256_file(executable) if executable.is_file() else None,
        "intended_output_root": str(output_root.expanduser().resolve()),
        "sanitized_environment": environment,
        "environment_digest": canonical_sha256(environment),
    })
    evidence.mark("initialized")
    return evidence


def run_startup_diagnostic(*, output_root: Path, run_id: str, argv: tuple[str, ...], initialize: Callable[[], None] | None = None) -> SupervisorEvidence:
    """Exercise supervisor initialization without creating backend/harness objects."""
    evidence = initialize_supervisor(output_root=output_root, run_id=run_id, argv=argv)
    try:
        if initialize is not None:
            initialize()
        evidence.mark("dry_startup_complete")
    except Exception as exc:
        evidence.mark("failure", error=exc)
        raise SupervisorError(f"startup diagnostic failed: {exc}") from exc
    return evidence


def _json_bytes(value: BaseModel) -> bytes:
    return (json.dumps(value.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def _write_new(path: Path, content: bytes) -> None:
    with path.open("xb") as output:
        output.write(content)
        output.flush()
        os.fsync(output.fileno())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
