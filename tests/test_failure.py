from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from agent_bench.backend import (
    BackendPreflightReport,
    PreflightCheck,
    ResolvedBackendInvocation,
)
from agent_bench.capture import fixed_proxy_capture_capabilities
from agent_bench.failure import (
    FailedRunEvidenceError,
    FailureEnvironmentRecord,
    preserve_failed_run,
    verify_failed_run,
)


def _environment(run_id: str) -> FailureEnvironmentRecord:
    report = BackendPreflightReport(
        profile_id="fixture-backend-v1",
        passed=False,
        primary_failure_class="template_hash_mismatch",
        checks=(
            PreflightCheck(
                check_id="chat-template",
                passed=False,
                failure_class="template_hash_mismatch",
                message="fixture template mismatch",
                evidence={"expected": "a" * 64, "observed": "b" * 64},
            ),
        ),
    )
    invocation = ResolvedBackendInvocation(
        profile_id="fixture-backend-v1",
        run_seed=1001,
        executable=Path("/opt/fixture/llama-server"),
        argv=("/opt/fixture/llama-server", "--port", "18080"),
        working_directory=Path("/opt/fixture/llama.cpp"),
        environment={"HOME": "/tmp/fixture-home"},
        stdout_artifact="failure/stdout.log",
        stderr_artifact="failure/stderr.log",
    )
    return FailureEnvironmentRecord(
        run_id=run_id,
        backend_profile_digest="c" * 64,
        preflight=report,
        invocation=invocation,
        capture_capabilities=fixed_proxy_capture_capabilities(),
    )


def test_failed_preflight_evidence_is_sealed_checksums_and_non_overwriting(
    tmp_path: Path,
) -> None:
    created = datetime(2026, 9, 3, 12, 0, tzinfo=timezone.utc)
    evidence = preserve_failed_run(
        runs_root=tmp_path / "runs",
        run_id="failed-run",
        failure_class="template_hash_mismatch",
        reason="fixture template mismatch",
        environment=_environment("failed-run"),
        stdout=b"version output\n",
        stderr=b"diagnostic\n",
        created_at=created,
    )

    assert evidence.root == tmp_path / "runs" / "failed-run" / "failure"
    assert {path.name for path in evidence.root.iterdir()} == {
        "manifest.json", "events.jsonl", "stdout.log", "stderr.log",
        "environment.json", "checksums.sha256",
    }
    verified = verify_failed_run(evidence.root)
    assert verified.manifest.failure_class == "template_hash_mismatch"
    assert verified.events[-1].payload["failure_class"] == "template_hash_mismatch"
    with pytest.raises(FailedRunEvidenceError, match="already exists"):
        preserve_failed_run(
            runs_root=tmp_path / "runs",
            run_id="failed-run",
            failure_class="template_hash_mismatch",
            reason="retry forbidden",
            environment=_environment("failed-run"),
            created_at=created,
        )


def test_failed_run_events_are_deterministic_and_tampering_is_detected(
    tmp_path: Path,
) -> None:
    created = datetime(2026, 9, 3, 12, 0, tzinfo=timezone.utc)
    first = preserve_failed_run(
        runs_root=tmp_path / "one",
        run_id="same-run",
        failure_class="template_hash_mismatch",
        reason="fixture template mismatch",
        environment=_environment("same-run"),
        created_at=created,
    )
    second = preserve_failed_run(
        runs_root=tmp_path / "two",
        run_id="same-run",
        failure_class="template_hash_mismatch",
        reason="fixture template mismatch",
        environment=_environment("same-run"),
        created_at=created,
    )
    assert (first.root / "events.jsonl").read_bytes() == (
        second.root / "events.jsonl"
    ).read_bytes()
    assert (first.root / "manifest.json").read_bytes() == (
        second.root / "manifest.json"
    ).read_bytes()

    (first.root / "stderr.log").write_bytes(b"tampered")
    with pytest.raises(FailedRunEvidenceError, match="checksum mismatch"):
        verify_failed_run(first.root)


def test_failed_run_rejects_path_like_run_id_before_writing(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        preserve_failed_run(
            runs_root=tmp_path / "runs",
            run_id="../escape",
            failure_class="precondition_failed",
            reason="invalid run id",
            environment=_environment("safe-run"),
        )

    assert not (tmp_path / "escape").exists()
