from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from agent_bench.manual_review import (
    CriterionResult, ManualReview, ManualReviewError, aggregate_reviews,
    load_protocol, review_queue, save_review, validate_review_against_protocol,
)


def _review(tmp_path: Path, *, revision: int = 1, outcome: str = "PASS", regression: str = "PASS") -> ManualReview:
    protocol, digest = load_protocol(Path("subjects/pocket-ledger-v1"))
    task = protocol["tasks"]["entry-delete"]
    return ManualReview.create(review_id=f"fixture-review-r{revision}", experiment_id="fixture", run_id="fixture-run",
        semantic_task="entry-delete", review_protocol_id=protocol["review_protocol_id"], review_protocol_digest=digest,
        reviewed_at=datetime(2026, 9, 5, tzinfo=timezone.utc), reviewer_id="local", blind_review_id="blind-fixture",
        functional_outcome=outcome, task_criteria=tuple(CriterionResult(criterion_id=value, outcome="PASS") for value in task["criteria"]),
        regression_criteria=tuple(CriterionResult(criterion_id=value, outcome="PASS") for value in protocol["common_regression_criteria"]),
        regression_outcome=regression, review_completeness="complete", source_artifact_manifest_sha256="a" * 64, revision=revision)


def test_review_digest_protocol_and_immutable_revisions(tmp_path: Path) -> None:
    review = _review(tmp_path)
    validate_review_against_protocol(review, Path("subjects/pocket-ledger-v1"))
    path = save_review(tmp_path, review)
    assert ManualReview.model_validate_json(path.read_bytes()).record_digest == review.record_digest
    with pytest.raises(ManualReviewError, match="use deliberate amend"):
        save_review(tmp_path, review)
    amended = _review(tmp_path, revision=2, outcome="MOSTLY_PASS")
    assert save_review(tmp_path, amended, amend=True).name == "revision-002.json"


def test_protocol_rejects_incomplete_criteria_and_major_regression_pass(tmp_path: Path) -> None:
    review = _review(tmp_path)
    incomplete = review.model_copy(update={"task_criteria": review.task_criteria[:-1]})
    with pytest.raises(ManualReviewError, match="incomplete"):
        validate_review_against_protocol(incomplete, Path("subjects/pocket-ledger-v1"))
    with pytest.raises(ValueError, match="major regression"):
        _review(tmp_path, regression="MAJOR_REGRESSION")


def test_unreviewable_requires_reason_and_persisted_reviews_cannot_be_incomplete(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="short reason"):
        CriterionResult(criterion_id="app_loads", outcome="UNREVIEWABLE")
    review = _review(tmp_path)
    with pytest.raises(ValueError, match="must be complete"):
        ManualReview.model_validate(review.model_copy(update={"review_completeness": "incomplete"}).model_dump(mode="json", exclude={"definition_digest"}))
