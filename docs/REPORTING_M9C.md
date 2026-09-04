# Deterministic reporting (M9C)

M9C turns sealed experiment evidence into a new, derived `report-schema-v1`
artifact. It never changes a run artifact, `metrics-v1`, `context-analysis-v2`,
timing-provenance artifact, result ref, or `experiment-state.json`. It uses no
LLM and does not read raw reasoning or raw request bodies for report content.

## Commands

```bash
python -m agent_bench.cli report build runs/pocket-ledger-v1-qwen38 \
  --experiment-definition experiments/pocket-ledger-v1.yaml
python -m agent_bench.cli report status runs/pocket-ledger-v1-qwen38 --json
python -m agent_bench.cli report verify runs/pocket-ledger-v1-qwen38/report-v1
python -m agent_bench.cli report export-public runs/pocket-ledger-v1-qwen38 \
  --output published-results/pocket-ledger-v1-qwen38
```

`build` creates a new destination exclusively (default:
`<experiment-output>/report-v1`). Use `--output` for another versioned derived
report; no existing report directory is overwritten. Supplying the immutable
experiment YAML verifies the definition digest and provides canonical matrix,
task, prompt-variant, repetition, and seed labels. Without it, the report keeps
only evidence-derived labels and marks unavailable identity fields as `NULL`.

## Input verification and partial experiments

For every completed row, ingestion verifies the sealed run artifact, metrics
artifact, context-analysis-v2 artifact, their cross-links, and M9B's persistent
result Git ref/tree/ancestry. Timing-provenance-v1 is verified when present.
An invalid completed row is retained as `evidence_status = invalid`, with its
reason, rather than silently ingested. Failed, interrupted, invalid, and pending
matrix rows remain visible even when no normal metrics exist.

The summary and HTML prominently show `PARTIAL EXPERIMENT — completed / planned`
until every planned row is terminal and verified. An ordinary harness exit or
Agent Bench `success` means execution/preservation success only; it is not a
task-correctness or manual-review result.

## Layout and schemas

```text
report-v1/
  parquet/
    experiments.parquet       runs.parquet          metrics.parquet
    requests.parquet          context_points.parquet tools.parquet
    timing.parquet            failures.parquet      artifacts.parquet
    git_change_metrics.parquet curves.parquet        summaries.parquet
  agent-bench.duckdb
  charts.json
  presentation.json
  summary.json
  raw-archival-manifest.json
  report.html
  report-manifest.json
  checksums.sha256
```

All Parquet schemas use a stable, fixed column order. Rows always carry
`experiment_id` and, where applicable, `run_id`, execution index, canonical
matrix index, harness/profile, task, prompt variant, repetition, and seed.
`metrics` stores explicit availability, reason, units, and method. `requests`
and `context_points` retain hashes and numeric observations but never raw request
bodies or reasoning. `artifacts` contains only relative provenance and content
hashes, never execution-host paths.

## Offline dashboard and run explorer

`report.html` is a self-contained, dependency-free dashboard driven by the
sealed, sanitized `presentation.json` companion. It is a human-facing view of
the same deterministic report data, not a new analysis layer and not a source
of measurements. It presents, in order:

- a partial/complete experiment header and execution KPI cards;
- matrix dimensions alongside the fixed model, template, llama.cpp, GPU,
  server, sampling, and generation configuration read from versioned sources;
- individual-run comparison charts and deterministic Type 7 summaries grouped
  by harness, task, prompt variant, and their supported combinations;
- three context views (absolute task-relative time, normalized elapsed task
  time, and real task request index), with labelled observed markers and
  median/Q1/Q3 bands only for like-for-like normalized series with `N > 1`;
- exact prompt text and hash, profile/deviation identity, sanitized meaningful
  invocation flags, metric availability/provenance, request hashes, normalized
  tool records, capture capabilities, and preservation hashes for a selected
  run;
- separate failure/interruption and pending/planned sections; pending rows are
  never displayed as failures.

The report deliberately preserves unavailable values and their reasons. It
does not invent component token attribution, task quality, manual review,
tool-execution timestamps, or a comparison winner. Full host paths, secrets,
raw request bodies, and reasoning are excluded; path values in presentation
metadata are replaced with a host-path-redacted label. `presentation.json` is
covered by the report manifest and checksums and is included in public export.

The DuckDB database materializes those tables and exposes `all_runs`,
`successful_runs`, `failed_runs`, `per_harness_metrics`, `per_task_metrics`,
`per_prompt_variant_metrics`, `per_repetition_metrics`, `context_series`,
`tool_usage`, and `git_changes` views.

Examples:

```sql
SELECT harness, median(wall_time_seconds) FROM successful_runs GROUP BY harness;
SELECT semantic_task, harness, median(input_tokens) FROM successful_runs GROUP BY 1, 2;
SELECT harness, median(peak_context_tokens) FROM successful_runs GROUP BY harness;
SELECT prompt_variant, median(wall_time_seconds) FROM successful_runs GROUP BY prompt_variant;
SELECT state, count(*) FROM all_runs GROUP BY state;
SELECT run_id, request_index, delta_vs_first_task_tokens FROM context_series;
SELECT harness, category, count(*) FROM runs JOIN tool_usage USING (experiment_id, run_id) GROUP BY 1, 2;
```

## Context and timing semantics

The static HTML contains all three context views: individual points against
absolute task-relative time, normalized elapsed task time, and real task request
index. Auxiliary/title requests are retained as separate overhead in Parquet and
summary data but do not set task-time zero.

For M9C, normalized elapsed task time is **not semantic completion**:

```text
0%   = first real task inference request
100% = observed task terminal boundary
progress = (request_elapsed - first_task_request_elapsed)
           / (wall_time - first_task_request_elapsed) * 100
```

The common grid is 0 through 100. Values before/after observable context are
labelled `boundary_carried`; interior values are linear interpolation;
explicitly unavailable observations create `unavailable_gap` values and are
never bridged. Per-point summaries use Hyndman-Fan type 7 median/Q1/Q3 with
round-half-even serialization to six decimal places. N=1 remains an individual
line with N=1; no variance is invented.

Timing rows retain non-interchangeable semantics. True harness execution timing
is present only for explicit native start/end evidence. Pi observation and Hermes
SQLite/export timing remain observation values, while unavailable execution time
retains `native_execution_timestamp_not_exposed`. Aggregate charts/tables do not
combine timing values with different semantic labels or methods.

Context component decomposition is `unavailable` unless the existing sealed
analysis proves exact token attribution; system/harness/tool overhead is never
subtracted from the authoritative proxy/API context total.

## Integrity, publication, and privacy

`report-manifest.json` has report schema/generator/input identities, included
and excluded run IDs, calculation rules, and deterministic file hashes.
`checksums.sha256` covers it and every derived file. The report is rebuildable,
not primary immutable run evidence.

`export-public` copies only the derived static HTML, sanitized Parquet/DuckDB,
summary, integrity records, and archival metadata. It excludes raw logs,
reasoning, request bodies, authorization values, cookies, personal HOME paths,
and arbitrary environment values. A deterministic privacy audit rejects known
secret/header/path markers before publication. It does not add, commit, push, or
upload anything.

`raw-archival-manifest.json` prepares per-run expected raw-bundle names, sealed
artifact hashes, persistent result refs/commits, and future release locations.
It deliberately does not create or upload a raw archive in M9C.
