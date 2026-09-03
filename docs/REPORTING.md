# Agent Bench Deterministic Reporting

Status: Milestone M0 reporting specification  
Specification version: 1.0.0

## 1. Purpose and boundary

Reports expose deterministic metrics, context trajectories, termination outcomes, and direct references to preserved evidence. Report generation consumes sealed manifests, normalized events, metrics, Git summaries, and artifact manifests; it never calls an LLM.

M0 defines report data and chart semantics only. It does not implement plotting, HTML, a server, or a UI.

Every report records its schema/specification version, generator version, input artifact IDs/digests, filters, grouping keys, and generation timestamp. Regeneration writes a new versioned artifact and does not overwrite sealed inputs.

## 2. General rules

- Values use metric definitions from `METRICS.md` without reinterpreting them.
- `N/A` remains distinct from zero and carries its machine-readable reason.
- Counts contributing to every aggregate are displayed.
- Failed and no-change runs are included by default and visually identified.
- No missing value is imputed.
- Exact and reconstructed token methods are labeled; incompatible methods are not silently pooled.
- Chart coordinates are stored as deterministic report data so different renderers can reproduce them.
- Stable sorting uses experiment matrix position, then `run_id`, unless a chart specifies another order.

## 3. Per-run report

### 3.1 Run summary

Each run summary includes:

- run and experiment identities;
- baseline commit and result/source-snapshot identity;
- harness, native version, profile ID/version, and relevant native settings;
- prompt ID, semantic task ID, variant, and SHA256;
- fixed model/backend/hardware identities;
- task timing and termination class, including underlying/supporting error events;
- token/context summary with source methods;
- tool-call/category/outcome summary;
- Git/result summary;
- preservation/checksum status;
- unavailable metric reasons; and
- links or artifact-relative references to exact definitions, prompt, raw streams, normalized events, metrics, Git evidence, complete source snapshot, build artifacts, and build/launch commands.

Links target immutable/downloadable artifacts, not transient worktree paths. Sensitive host paths and secret values are never exposed.

### 3.2 Context versus absolute elapsed time

The x-axis is elapsed task time in seconds from `task_start`. The y-axis is context utilization percent; an optional companion chart uses absolute context tokens.

Each observed LLM request contributes one point:

```text
(request.elapsed_ns / 1_000_000_000,
 100 * request.context_tokens / configured_max_context_tokens)
```

Points are ordered by request index and elapsed time. Missing context counts produce a visible gap/marker and are not treated as zero. The chart includes the task terminal time even if the last request occurred earlier, but it does not synthesize a new measured context point there.

### 3.3 Context versus normalized run progress

For a run with positive `wall_time`, each measured request maps to:

```text
progress_percent = 100 * request.elapsed_ns / wall_time_ns
```

Values are retained without clamping and flagged if source inconsistency puts a task request outside `[0, 100]`. A non-positive or unavailable wall time makes the normalized series unavailable.

The displayed per-run curve uses the common-grid interpolation in section 5. The raw request points remain available so interpolation is not mistaken for measurement.

### 3.4 Context versus LLM request index

The x-axis is the one-based normalized `request_index`; the y-axis is utilization percent, with optional absolute token count. No interpolation occurs between missing integer indices for table data. A renderer may connect adjacent available points but must preserve gaps where an intervening observation is unavailable.

### 3.5 Important event markers

Markers are emitted only for normalized observable events with source references:

- first tool call;
- first edit attempt;
- first test command;
- every compaction start/end;
- output truncation;
- context overflow;
- run completion;
- run failure;
- timeout; and
- process termination.

Absolute-time charts place markers at elapsed seconds. Normalized-time charts map marker elapsed time through the same wall-time formula. Request-index charts attach a marker to a directly correlated request; otherwise the marker is placed between the greatest preceding and least following request indices in a separately labeled event lane, not assigned a fabricated index.

Multiple events at the same coordinate remain distinct in machine-readable data and may be visually stacked. Marker tooltips/labels include event kind, time, outcome, and source reference.

## 4. Cross-run report

Cross-run reports support filtering and grouping by:

- harness;
- harness profile/settings;
- prompt ID;
- semantic task ID and prompt variant;
- harness/profile/prompt combination;
- repetition;
- termination class; and
- compatible token-count method.

Every filter is stored in report metadata. Default views do not drop failed runs.

### 4.1 Individual-run absolute-time overlays

Each run retains its own elapsed-seconds x-axis starting at zero. Lines end at the run's terminal boundary; no value is carried beyond that run. Request observations are not resampled for individual overlays unless a renderer explicitly uses a documented display-only operation.

Longer and shorter runs therefore occupy different x ranges. Termination markers disclose why each line ended.

### 4.2 Individual-run normalized-time overlays

Each run is mapped to 0–100% using its own positive wall time and the interpolation algorithm in section 5. Failures are normalized to their actual terminal boundary exactly like successes, remain labeled by termination class, and are not stretched to a hypothetical successful duration.

Runs without valid wall time or context observations are listed as excluded with reasons; they are not rendered as zero lines.

### 4.3 Request-index comparisons

Runs are aligned by one-based LLM request index. For each integer index, only runs with an available observation at that exact index contribute. There is no interpolation across request indices for aggregate request-index tables or curves. The contributor count is stored at every index.

### 4.4 Repeated-run summaries

For a selected group, reports may show:

- every individual run line;
- pointwise median;
- pointwise 25th and 75th percentiles (interquartile band); and
- another explicitly named percentile pair when selected in report configuration.

Aggregates are pointwise and do not construct a synthetic median run. Termination-rate/count tables accompany curves so failed or unavailable runs are visible.

## 5. Normalized 0–100% interpolation

### 5.1 Common grid

The default progress grid is the 101 exact integer percentages `0, 1, ..., 100`. A report may configure another sorted grid only if it records every exact grid value and uses the same algorithm.

### 5.2 Input preparation

For each run:

1. Require positive `wall_time_ns` and at least one request with available utilization.
2. Compute each point's rational progress from integer elapsed and wall nanoseconds; retain full precision until final serialization.
3. Sort by progress, then request index, then normalized event sequence.
4. If multiple observations have exactly the same progress, keep the last by that ordering for interpolation and retain all raw points for audit.
5. Do not interpolate across an explicitly unavailable request observation. It divides the curve into independent available segments.

### 5.3 Boundary handling

Context is observed only at request boundaries. To make a full 0–100% comparison grid without claiming extra measurements:

- from 0% through the first available observation, use the first observed value as a labeled boundary carry;
- from the last available observation through 100%, use the last observed value as a labeled boundary carry; and
- if an unavailable observation creates an internal gap, values inside that gap remain unavailable until the next available observation; they are not carried or interpolated across the gap.

Boundary-carried values have `value_method = boundary_carried`, distinct from measured points and interpolated points.

### 5.4 Interior interpolation

Between two adjacent available observations with no unavailable observation between them, use piecewise linear interpolation. For grid progress `g` between points `(p0, y0)` and `(p1, y1)`, where `p1 > p0`:

```text
y(g) = y0 + (y1 - y0) * (g - p0) / (p1 - p0)
```

Arithmetic uses decimal/rational precision sufficient to avoid platform-dependent binary rounding. Serialized percentages round once to six decimal places using round-half-even. Measured grid coincidences preserve the measured value and use `value_method = measured`; other interior points use `linear_interpolated`.

A run with exactly one available observation and no internal unavailable points uses that value across the grid, with the one measured coordinate identified and all other values labeled boundary-carried. This makes the lack of trajectory explicit in metadata.

### 5.5 Failed and truncated runs

Timeout, crash, truncation, overflow, no-change, and other terminal outcomes use their observed task terminal time as 100%. They participate in overlays and aggregates when the selected metric is available. Each grid record carries termination class so a normalized failure trajectory cannot be mistaken for successful progress toward task completion.

Precondition failures have no task wall time and are excluded from curves while remaining in summary/termination tables.

## 6. Median and percentile calculation

At each normalized grid point or exact request index:

1. Select available values from runs in the group that pass explicit filters.
2. Partition by token-count method compatibility when required by report configuration.
3. Sort numeric values ascending, retaining run IDs for audit.
4. Compute median, 25th percentile, and 75th percentile using Hyndman-Fan type 7: for probability `p`, `h = (n - 1) * p + 1`; interpolate linearly between the surrounding one-based order statistics.
5. Serialize using six decimal places and round-half-even.

The report stores `n_available`, `n_total`, excluded run IDs/reasons, and termination-class counts at each point. For one contributor, median/Q1/Q3 equal that value but the sample count makes the zero-width spread clear. With zero contributors the aggregate is unavailable.

Aggregate curves never fill an unavailable point using neighboring aggregate points.

## 7. Deterministic summary tables

Cross-run summary tables include, where available:

- run counts and termination counts/rates;
- wall/LLM/tool time;
- request and token/context metrics;
- peak context and compaction metrics;
- tool calls/categories/outcomes and duplicate/repeated-access metrics;
- Git/result counts; and
- preservation/availability status.

For each numeric metric and group, tables may report individual values, contributor count, median, Q1, and Q3 using the same percentile rule. Arithmetic mean is omitted by default and, if requested later, must be explicitly labeled and versioned. Categorical values use counts and denominators, not invented numeric scores.

Group keys include both human labels and immutable IDs/versions. Native harness settings are displayable beside any normalized conceptual category so semantic differences remain visible.

## 8. Artifact references and downloads

Per-run and cross-run report data references:

- sealed run manifest;
- exact prompt artifact;
- raw event/log artifacts;
- normalized event stream;
- deterministic metrics record;
- Git change summary/diff evidence;
- environment snapshots;
- artifact manifest/checksum listing;
- complete preserved source snapshot;
- build/executable artifacts when available; and
- build/launch command records when available.

Every reference includes artifact-relative path, SHA256, byte size, media type, and availability. A renderer may provide a download control but may not rewrite, regenerate in place, or expose redacted secrets.

## 9. Validation

Report generation validates input schema compatibility and digests before calculation. It fails or marks affected output unavailable for:

- inconsistent run/experiment IDs;
- changed input digests;
- invalid wall-time/progress coordinates;
- duplicate request indices without deterministic ordering;
- context utilization inconsistent with source values;
- interpolation across declared gaps;
- aggregate contributor counts inconsistent with run lists; or
- percentile results inconsistent with the specified algorithm.

Rendering choices such as color, font, or image format do not alter machine-readable coordinates or summary values.
