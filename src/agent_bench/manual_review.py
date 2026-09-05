"""Versioned, human-authored functional acceptance review evidence (M10)."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agent_bench.config import load_experiment
from agent_bench.executor import ExperimentState
from agent_bench.matrix import expand_experiment
from agent_bench.models import Identifier, PersistedModel, Sha256, canonical_sha256
from agent_bench.preservation import restore_artifact, verify_artifact

REVIEW_SCHEMA_VERSION = "1.0.0"
REVIEW_DIRECTORY = "manual-review-v1"
OUTCOMES = ("PASS", "MOSTLY_PASS", "PARTIAL", "FAIL", "UNREVIEWABLE")
REGRESSION_OUTCOMES = ("PASS", "MINOR_REGRESSION", "MAJOR_REGRESSION", "UNREVIEWABLE")


class ManualReviewError(RuntimeError):
    pass


class CriterionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    criterion_id: Identifier
    outcome: Literal["PASS", "FAIL", "UNREVIEWABLE"]
    notes: str | None = None

    @model_validator(mode="after")
    def unavailable_has_reason(self) -> "CriterionResult":
        if self.outcome == "UNREVIEWABLE" and not (self.notes and self.notes.strip()):
            raise ValueError("UNREVIEWABLE criterion requires a short reason")
        return self


class ManualReview(PersistedModel):
    review_id: Identifier
    experiment_id: Identifier
    run_id: Identifier
    semantic_task: Identifier
    review_protocol_id: Identifier
    review_protocol_digest: Sha256
    reviewed_at: datetime
    reviewer_id: str = Field(min_length=1)
    blind_review_id: Identifier
    functional_outcome: Literal["PASS", "MOSTLY_PASS", "PARTIAL", "FAIL", "UNREVIEWABLE"]
    task_criteria: tuple[CriterionResult, ...]
    regression_criteria: tuple[CriterionResult, ...]
    regression_outcome: Literal["PASS", "MINOR_REGRESSION", "MAJOR_REGRESSION", "UNREVIEWABLE"]
    review_completeness: Literal["complete", "incomplete"]
    notes: str | None = None
    automated_acceptance_evidence: Literal["not_implemented"] = "not_implemented"
    source_artifact_manifest_sha256: Sha256
    revision: int = Field(ge=1)
    record_digest: Sha256

    @field_validator("reviewed_at")
    @classmethod
    def utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("reviewed_at must include UTC offset")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def digest_and_policy(self) -> "ManualReview":
        if self.review_completeness != "complete":
            raise ValueError("a persisted manual review must be complete")
        if self.regression_outcome == "MAJOR_REGRESSION" and self.functional_outcome == "PASS":
            raise ValueError("a major regression cannot receive functional PASS")
        expected = canonical_sha256(self.model_dump(mode="json", exclude={"record_digest", "definition_digest"}))
        if self.record_digest != expected:
            raise ValueError("manual review record digest mismatch")
        return self

    @classmethod
    def create(cls, **values: object) -> "ManualReview":
        draft = cls.model_construct(schema_version=REVIEW_SCHEMA_VERSION, record_digest="0" * 64, **values)
        content = draft.model_dump(mode="json", exclude={"record_digest", "definition_digest"})
        return cls.model_validate({**content, "record_digest": canonical_sha256(content)})


def protocol_path(subject_root: Path) -> Path:
    return subject_root / "review-protocol-v1.yaml"


def load_protocol(subject_root: Path) -> tuple[dict[str, object], str]:
    raw = yaml.safe_load(protocol_path(subject_root).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != REVIEW_SCHEMA_VERSION:
        raise ManualReviewError("unsupported review protocol")
    tasks = raw.get("tasks")
    if not isinstance(tasks, dict) or len(tasks) != 5:
        raise ManualReviewError("review protocol must define exactly five tasks")
    return raw, canonical_sha256(raw)


def review_root(experiment_output: Path) -> Path:
    return experiment_output / REVIEW_DIRECTORY


def review_queue(experiment_output: Path, experiment_definition: Path, subject_root: Path) -> list[dict[str, object]]:
    protocol, digest = load_protocol(subject_root)
    state = ExperimentState.model_validate_json((experiment_output / "experiment-state.json").read_bytes())
    experiment = load_experiment(experiment_definition)
    definitions = {item.run_id: item for item in expand_experiment(experiment)}
    records = latest_reviews(experiment_output)
    queue: list[dict[str, object]] = []
    for progress in state.runs:
        definition = definitions[progress.run_id]
        blind = _blind_id(progress.run_id, digest)
        review = records.get(progress.run_id)
        queue.append({"run_id": progress.run_id, "blind_review_id": blind, "blind_sort": _blind_sort(progress.run_id, digest),
                      "semantic_task": definition.semantic_task_id, "state": progress.state,
                      "reviewed": review is not None, "review_outcome": review.functional_outcome if review else None,
                      "priority_flags": priority_flags(experiment_output, progress.run_id, definition.semantic_task_id, protocol)})
    return sorted(queue, key=lambda item: (str(item["blind_sort"]), str(item["run_id"])))


def priority_flags(experiment_output: Path, run_id: str, task: str, protocol: dict[str, object]) -> tuple[str, ...]:
    flags: list[str] = []
    metrics_path = experiment_output / "analysis" / run_id / "metrics-v1" / "metrics.json"
    if not metrics_path.is_file(): return ("metrics_not_available",)
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    term = data["termination"]["termination_class"]
    git = data["git_result"]
    if term == "no_changes": flags.append("termination_no_changes")
    if git["files_changed"]["value"] == 0: flags.append("zero_changed_files")
    if data["behavior"]["tool_calls_failed"]["value"]:
        flags.append("tool_failures_observed")
    if data["behavior"]["agent_invoked_test_calls"]["value"] == 0:
        flags.append("no_agent_invoked_test")
    if data["timing"]["wall_time_seconds"]["value"] and data["timing"]["wall_time_seconds"]["value"] > 600:
        flags.append("long_wall_time")
    if data["tokens"]["input_tokens_total"]["value"] and data["tokens"]["input_tokens_total"]["value"] > 250000:
        flags.append("high_input_tokens")
    return tuple(flags)


def latest_reviews(experiment_output: Path) -> dict[str, ManualReview]:
    records: dict[str, ManualReview] = {}
    for path in sorted((review_root(experiment_output) / "records").glob("*/*.json")) if (review_root(experiment_output) / "records").is_dir() else []:
        review = ManualReview.model_validate_json(path.read_bytes())
        if review.run_id not in records or review.revision > records[review.run_id].revision:
            records[review.run_id] = review
    return records


def save_review(experiment_output: Path, review: ManualReview, *, amend: bool = False) -> Path:
    root = review_root(experiment_output) / "records" / review.run_id
    existing = latest_reviews(experiment_output).get(review.run_id)
    if existing and not amend:
        raise ManualReviewError("review exists; use deliberate amend to create a new immutable revision")
    expected_revision = (existing.revision + 1) if existing else 1
    if review.revision != expected_revision:
        raise ManualReviewError(f"expected review revision {expected_revision}")
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"revision-{review.revision:03d}.json"
    if path.exists(): raise ManualReviewError("review revision already exists")
    _write_new(path, review.model_dump_json(exclude_none=True, by_alias=False, exclude={"definition_digest"}) + "\n")
    return path


def validate_review_against_protocol(review: ManualReview, subject_root: Path) -> None:
    protocol, digest = load_protocol(subject_root)
    if review.review_protocol_id != protocol["review_protocol_id"] or review.review_protocol_digest != digest:
        raise ManualReviewError("review protocol identity does not match the checked-in protocol")
    task = protocol["tasks"].get(review.semantic_task)
    if not isinstance(task, dict): raise ManualReviewError("unknown semantic task for review")
    expected_task = tuple(task["criteria"])
    expected_regression = tuple(protocol["common_regression_criteria"])
    if len(review.task_criteria) != len(expected_task) or {item.criterion_id for item in review.task_criteria} != set(expected_task):
        raise ManualReviewError("task criteria are incomplete or do not match the canonical acceptance specification")
    if len(review.regression_criteria) != len(expected_regression) or {item.criterion_id for item in review.regression_criteria} != set(expected_regression):
        raise ManualReviewError("regression criteria are incomplete or do not match the common checklist")


def prepare_review_copy(experiment_output: Path, run_id: str, destination: Path, subject_root: Path) -> dict[str, object]:
    protocol, digest = load_protocol(subject_root)
    artifact = experiment_output / "artifacts" / run_id
    manifest = verify_artifact(artifact)
    restore_artifact(artifact, destination)
    fixture = protocol["fixture"]
    html = "<!doctype html><meta charset=utf-8><script>localStorage.clear();sessionStorage.clear();localStorage.setItem(" + json.dumps(str(fixture["storage_key"])) + "," + json.dumps(json.dumps(fixture["entries"], separators=(",", ":"))) + ");location.replace('index.html')</script>"
    (destination / "review-fixture.html").write_text(html, encoding="utf-8", newline="\n")
    return {"run_id": run_id, "blind_review_id": _blind_id(run_id, digest), "destination": str(destination.resolve()),
            "fixture_url": "review-fixture.html", "source_artifact_manifest_sha256": hashlib.sha256((artifact / "manifest.json").read_bytes()).hexdigest(),
            "result_commit": manifest.result_commit}


def aggregate_reviews(experiment_output: Path, experiment_definition: Path) -> list[dict[str, object]]:
    definitions = {item.run_id: item for item in expand_experiment(load_experiment(experiment_definition))}
    rows: dict[tuple[str, str], list[ManualReview]] = defaultdict(list)
    for run_id, review in latest_reviews(experiment_output).items():
        item = definitions[run_id]
        keys = {"all": "all", "harness": item.harness_id, "semantic_task": item.semantic_task_id,
                "prompt_variant": item.prompt_id.rsplit("-", 1)[-1], "repetition": str(item.repetition_index),
                "harness_task": f"{item.harness_id} × {item.semantic_task_id}",
                "harness_prompt_variant": f"{item.harness_id} × {item.prompt_id.rsplit('-', 1)[-1]}",
                "harness_task_prompt_variant": f"{item.harness_id} × {item.semantic_task_id} × {item.prompt_id.rsplit('-', 1)[-1]}"}
        for grouping, key in keys.items(): rows[(grouping, key)].append(review)
    output=[]
    for (grouping,key), values in sorted(rows.items()):
        counts=Counter(item.functional_outcome for item in values)
        evaluable=sum(counts[x] for x in OUTCOMES if x != "UNREVIEWABLE")
        output.append({"grouping": grouping, "group_key": key, "reviewed": len(values), "evaluable": evaluable,
                       "outcomes": {name: counts[name] for name in OUTCOMES}, "strict_success": counts["PASS"],
                       "practical_success": counts["PASS"]+counts["MOSTLY_PASS"]})
    return output


def build_quality_report(experiment_output: Path, experiment_definition: Path, output: Path | None = None) -> Path:
    """Seal a separate M10 summary; M9 report files are never rewritten."""
    state = ExperimentState.model_validate_json((experiment_output / "experiment-state.json").read_bytes())
    root = output or review_root(experiment_output) / "quality-report-v1"
    if root.exists(): raise ManualReviewError("quality report destination already exists")
    root.mkdir(parents=True)
    payload = {"schema_version": REVIEW_SCHEMA_VERSION, "report_id": f"{state.experiment_id}-manual-review-v1",
               "experiment_id": state.experiment_id, "reviewed_run_count": len(latest_reviews(experiment_output)),
               "aggregates": aggregate_reviews(experiment_output, experiment_definition),
               "execution_quality_boundary": "M9 execution metrics are separate; efficiency comparisons are meaningful only after filtering by these human outcomes."}
    payload["record_digest"] = canonical_sha256(payload)
    path = root / "summary.json"
    _write_new(path, json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    return root


def _blind_sort(run_id: str, digest: str) -> str: return canonical_sha256({"run_id": run_id, "review_protocol_digest": digest})
def _blind_id(run_id: str, digest: str) -> str: return "blind-" + _blind_sort(run_id, digest)[:12]
def _write_new(path: Path, payload: str) -> None:
    fd, tmp = tempfile.mkstemp(prefix=".review-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.link(tmp, path)
    except FileExistsError as exc: raise ManualReviewError("review record already exists") from exc
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
