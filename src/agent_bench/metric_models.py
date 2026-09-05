"""Versioned persisted models for deterministic M4 run metrics."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent_bench.models import Identifier, Sha256, canonical_sha256

METRICS_SCHEMA_VERSION: Literal["1.0.0"] = "1.0.0"
METRIC_SPEC_VERSION: Literal["1.0.2"] = "1.0.2"

Availability = Literal["available", "unavailable", "not_applicable"]
MetricMethod = Literal[
    "manifest_exact",
    "normalized_event_exact",
    "backend_exact",
    "api_exact",
    "tokenizer_reconstructed",
    "deterministically_calculated",
    "git_native",
    "not_available",
]
UnavailableReason = Literal[
    "source_not_exposed",
    "capture_incomplete",
    "event_not_observed",
    "not_applicable",
    "ambiguous_evidence",
    "native_execution_timestamp_not_exposed",
    "invalid_source",
    "zero_denominator",
]
TerminationClass = Literal[
    "precondition_failed",
    "preservation_failed",
    "timeout",
    "process_killed",
    "context_overflow",
    "output_truncation",
    "model_backend_error",
    "harness_crash",
    "invalid_harness_output",
    "no_changes",
    "success",
    "unknown_other",
]


class MetricsModel(BaseModel):
    """Strict immutable base for all persisted metrics structures."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0.0"] = METRICS_SCHEMA_VERSION


class MetricProvenance(MetricsModel):
    """Evidence and calculation method for one metric value."""

    method: MetricMethod
    source_event_ids: tuple[str, ...] = ()
    source_artifact_paths: tuple[str, ...] = ()
    source_methods: tuple[str, ...] = ()


class ScalarMetric(MetricsModel):
    """A number with explicit units, availability, and provenance."""

    value: int | float | None
    units: str = Field(min_length=1)
    availability: Availability
    unavailable_reason: UnavailableReason | None = None
    provenance: MetricProvenance

    @model_validator(mode="after")
    def validate_availability(self) -> ScalarMetric:
        if self.availability == "available":
            if self.value is None or self.unavailable_reason is not None:
                raise ValueError("available metrics require a value and no reason")
        else:
            if self.value is not None or self.unavailable_reason is None:
                raise ValueError("unavailable metrics require a null value and reason")
        return self


class TimingMetrics(MetricsModel):
    wall_time_seconds: ScalarMetric
    llm_time_seconds: ScalarMetric
    tool_execution_time_seconds: ScalarMetric
    shell_execution_time_seconds: ScalarMetric
    time_to_first_llm_request_seconds: ScalarMetric
    time_to_first_tool_call_seconds: ScalarMetric
    time_to_first_edit_seconds: ScalarMetric
    time_to_first_test_command_seconds: ScalarMetric


class RequestTokenUsage(MetricsModel):
    request_index: int = Field(ge=1)
    request_event_id: str
    response_event_id: str | None = None
    input_tokens: ScalarMetric
    output_tokens: ScalarMetric
    total_tokens: ScalarMetric


class TokenMetrics(MetricsModel):
    input_tokens_total: ScalarMetric
    output_tokens_total: ScalarMetric
    reasoning_tokens_total: ScalarMetric
    visible_answer_tokens_total: ScalarMetric
    total_tokens: ScalarMetric
    tokens_per_llm_request: tuple[RequestTokenUsage, ...]
    mean_tokens_per_llm_request: ScalarMetric
    tokens_before_first_edit: ScalarMetric
    reasoning_tokens_before_first_edit: ScalarMetric


class ContextRequestPoint(MetricsModel):
    request_index: int = Field(ge=1)
    request_event_id: str
    elapsed_seconds: float | None = Field(default=None, ge=0)
    context_used_tokens: ScalarMetric
    context_max_tokens: ScalarMetric
    context_utilization_percent: ScalarMetric
    context_growth_tokens: ScalarMetric


class CompactionPoint(MetricsModel):
    compaction_index: int = Field(ge=1)
    compaction_event_id: str
    elapsed_seconds: float | None = Field(default=None, ge=0)
    tokens_before_compaction: ScalarMetric
    tokens_after_compaction: ScalarMetric
    context_max_tokens: ScalarMetric
    before_utilization_percent: ScalarMetric
    after_utilization_percent: ScalarMetric


class ContextMetrics(MetricsModel):
    context_used_per_request: tuple[ContextRequestPoint, ...]
    peak_context_tokens: ScalarMetric
    peak_context_utilization_percent: ScalarMetric
    net_context_growth_tokens: ScalarMetric
    number_of_compactions: ScalarMetric
    context_at_first_compaction_tokens: ScalarMetric
    context_utilization_at_first_compaction_percent: ScalarMetric
    compactions: tuple[CompactionPoint, ...]


class ToolCategoryCounts(MetricsModel):
    read: int = Field(ge=0)
    search: int = Field(ge=0)
    edit: int = Field(ge=0)
    write: int = Field(ge=0)
    test: int = Field(ge=0)
    shell: int = Field(ge=0)
    other: int = Field(ge=0)


class BehaviorMetrics(MetricsModel):
    llm_request_count: ScalarMetric
    llm_response_count: ScalarMetric
    tool_calls_total: ScalarMetric
    tool_calls_by_category: ToolCategoryCounts | None
    tool_calls_by_category_availability: Availability
    tool_calls_by_category_unavailable_reason: UnavailableReason | None = None
    tool_calls_by_category_provenance: MetricProvenance
    tool_calls_successful: ScalarMetric
    tool_calls_failed: ScalarMetric
    unknown_outcome_tool_calls: ScalarMetric
    read_calls: ScalarMetric
    search_calls: ScalarMetric
    edit_calls: ScalarMetric
    write_calls: ScalarMetric
    shell_calls: ScalarMetric
    agent_invoked_test_calls: ScalarMetric
    calls_before_first_edit: ScalarMetric
    calls_after_last_edit: ScalarMetric
    exact_duplicate_tool_calls: ScalarMetric
    repeated_reads_of_unchanged_files: ScalarMetric
    repeated_identical_shell_commands: ScalarMetric
    turns_with_reasoning_but_no_action: ScalarMetric
    # Added in metrics spec v1.0.2. Null denotes an older immutable metrics-v1
    # record; new calculations always populate these source-aware values.
    reasoning_only_responses: ScalarMetric | None = None
    length_finished_responses: ScalarMetric | None = None
    length_finished_without_tool_call: ScalarMetric | None = None
    requests_before_first_model_tool_call: ScalarMetric | None = None
    output_tokens_before_first_model_tool_call: ScalarMetric | None = None
    requests_before_first_model_edit_call: ScalarMetric | None = None
    output_tokens_before_first_model_edit_call: ScalarMetric | None = None

    @model_validator(mode="after")
    def validate_categories(self) -> BehaviorMetrics:
        if self.tool_calls_by_category_availability == "available":
            if self.tool_calls_by_category is None:
                raise ValueError("available category counts require values")
            if self.tool_calls_by_category_unavailable_reason is not None:
                raise ValueError("available category counts cannot have a reason")
        elif self.tool_calls_by_category is not None:
            raise ValueError("unavailable category counts must be null")
        return self


class DerivedMetrics(MetricsModel):
    tokens_per_tool_call: ScalarMetric
    tokens_per_edit: ScalarMetric
    reads_per_edit: ScalarMetric
    searches_per_edit: ScalarMetric
    seconds_per_edit: ScalarMetric
    failed_tool_call_rate: ScalarMetric
    reasoning_to_output_ratio: ScalarMetric


class GitResultMetrics(MetricsModel):
    files_changed: ScalarMetric
    files_created: ScalarMetric
    files_deleted: ScalarMetric
    files_renamed: ScalarMetric
    lines_added: ScalarMetric
    lines_deleted: ScalarMetric
    binary_files_changed: ScalarMetric
    source_files_changed: ScalarMetric
    test_files_changed: ScalarMetric
    configuration_files_changed: ScalarMetric


class TerminationResult(MetricsModel):
    termination_class: TerminationClass
    underlying_termination_class: TerminationClass | None = None
    reason: str
    source_event_ids: tuple[str, ...] = ()
    source_artifact_paths: tuple[str, ...] = ()


class MetricsInputIdentity(MetricsModel):
    artifact_manifest_sha256: Sha256
    run_manifest_sha256: Sha256
    raw_events_sha256: Sha256
    normalized_events_sha256: Sha256
    source_snapshot_sha256: Sha256
    git_diff_sha256: Sha256
    git_tracked_numstat_sha256: Sha256
    git_untracked_numstat_sha256: Sha256


class RunMetrics(MetricsModel):
    """Complete immutable deterministic measurements for one preserved run."""

    metrics_id: str = Field(min_length=1)
    metric_spec_version: Literal["1.0.0", "1.0.1", "1.0.2"] = METRIC_SPEC_VERSION
    calculator_name: Literal["agent-bench-metrics"] = "agent-bench-metrics"
    calculator_version: Literal["1.0.0", "1.0.1", "1.0.2"] = "1.0.2"
    calculator_configuration_digest: Sha256
    run_id: Identifier
    input_identity: MetricsInputIdentity
    timing: TimingMetrics
    tokens: TokenMetrics
    context: ContextMetrics
    behavior: BehaviorMetrics
    derived: DerivedMetrics
    git_result: GitResultMetrics
    termination: TerminationResult
    validation_status: Literal["valid", "valid_with_diagnostics"]
    diagnostics: tuple[str, ...] = ()
    record_digest: Sha256

    @model_validator(mode="after")
    def validate_digest(self) -> RunMetrics:
        expected = canonical_sha256(
            self.model_dump(mode="json", exclude={"record_digest"})
        )
        if self.record_digest != expected:
            raise ValueError("record_digest does not match metrics content")
        return self

    @classmethod
    def create(cls, **values: object) -> RunMetrics:
        content = {"schema_version": METRICS_SCHEMA_VERSION, **values}
        draft = cls.model_construct(**content, record_digest="0" * 64)
        canonical_content = draft.model_dump(mode="json", exclude={"record_digest"})
        return cls.model_validate(
            {
                **canonical_content,
                "record_digest": canonical_sha256(canonical_content),
            }
        )

    def canonical_json_bytes(self) -> bytes:
        """Serialize with stable ordering and no calculation-time metadata."""
        return (
            json.dumps(
                self.model_dump(mode="json"),
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
