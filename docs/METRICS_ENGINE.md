# Deterministic Metrics Engine

Status: Milestone M4 implemented contract
Schema version: 1.0.0
Metric specification version: 1.0.0

## 1. Scope and inputs

M4 calculates deterministic measurements from a verified, sealed M3 run
artifact. It does not invoke an LLM, run a harness, change the preserved result,
render charts, or produce cross-run analysis. The authoritative inputs are the
M2 artifact manifest and Git evidence, the M3 run manifest, and the immutable
raw and normalized event streams. All are identified by SHA256 in `RunMetrics`.

The only capture-completeness declaration implemented in M4 is for the
versioned FakeHarness fixture. Real adapters must provide an explicit normalized
capture-capability contract in M5 or later before a missing native observation
can be interpreted as an observed zero.

## 2. Persisted metric representation

`RunMetrics` and every nested persisted model contain `schema_version =
1.0.0`. The top-level record also contains stable metric, run, calculator,
specification, and configuration identities; SHA256 identities for both
manifests, both event streams, the source snapshot, and the Git diff;
deterministic diagnostics; and a canonical-content record digest.

Every scalar is a `ScalarMetric` containing a typed number or `null`, units,
availability, an unavailable reason where applicable, and provenance. Provenance
names the calculation/source method plus contributing normalized event IDs and
artifact paths. Exact backend/API values and tokenizer-reconstructed values are
not merged anonymously: their methods remain on individual values and aggregate
provenance records the contributing method set. Reconstructed counts require a
tokenizer identity and digest.

Unavailable values are never encoded as zero. The reason vocabulary distinguishes
unexposed sources, incomplete capture, absent events, not-applicable values,
ambiguous evidence, invalid source evidence, and zero denominators. Per-request
context points remain present when an individual count is unavailable; complete
aggregates such as peak context become unavailable rather than treating the
remaining points as a full-run peak.

## 3. Deterministic calculation rules

The formulas and edge cases in `METRICS.md` are authoritative. The implemented
calculator configuration is itself hashed and fixes these additional details:

- overlapping LLM and tool intervals are summed as correlated operation
  durations rather than unioned as wall-clock occupancy;
- request/tool correlation IDs and request indices must be unique;
- an incomplete start/end interval makes the corresponding duration aggregate
  unavailable rather than producing a partial sum;
- exact duplicate calls hash canonical JSON containing primary category, native
  tool name, and exact normalized native arguments;
- worktree paths use conservative project-relative POSIX lexical normalization;
- a repeated read requires successful reads, the same path/content SHA256, and
  no successful observed edit/write to that path between them;
- shell repetition includes the exact working directory and captured environment
  subset;
- direct derived metrics use only the formulas in `METRICS.md`; unavailable
  operands and zero denominators stay unavailable; and
- formal termination uses the documented precedence without consulting an LLM.

Context utilization is `100 * context_tokens / configured_max_context_tokens`.
Values above 100 percent remain unchanged evidence. Growth is the signed
difference between adjacent request-index points. Compaction metrics require a
direct compaction-start record and do not infer compaction from a context drop.

## 4. Complete Git/result evidence

The tracked result diff, status, untracked/ignored inventories, and complete
source snapshot remain the basis for file counts. M4 adds one necessary
preservation-time evidence files: `git/tracked-numstat.json` and
`git/untracked-numstat.json`. A normal Git diff cannot contain untracked or
ignored result files, so M2 invokes Git's
`diff --no-index --numstat` for each preserved non-tracked file and records the
Git version, algorithm, per-path text counts or binary status, and explicit
unavailability. This record is schema-versioned and covered by the run artifact
checksum inventory.

Tracked and non-tracked numstat come only from these preservation-time records;
the patch remains human/audit evidence rather than a recalculation dependency.
Binary paths increment
`binary_files_changed` but contribute neither added nor deleted text lines.
Inventory disagreement or unsupported evidence makes line metrics unavailable.
Rename detection is explicitly disabled for the preserved status and diff, so a
rename is deterministically represented as create/delete and `files_renamed` is
zero. No later similarity inference relabels those paths.

Path classifier version 1 lowercases names/suffixes and applies these exact
rules:

- source suffixes: `.c`, `.cc`, `.cpp`, `.cs`, `.css`, `.go`, `.h`, `.hpp`,
  `.html`, `.java`, `.js`, `.jsx`, `.kt`, `.php`, `.py`, `.rb`, `.rs`,
  `.scala`, `.sh`, `.swift`, `.ts`, `.tsx`, and `.vue`;
- test paths: a `tests` or `test` path component, a basename beginning `test_`,
  or a basename ending `_test.py`, `.test.js`, or `.spec.ts`; and
- configuration: `.editorconfig`, `.env`, `.gitignore`, `cargo.toml`,
  `dockerfile`, `go.mod`, `package.json`, `pyproject.toml`, `requirements.txt`,
  or `tox.ini`, plus `.ini`, `.json`, `.toml`, `.yaml`, and `.yml` suffixes.

A file can be both source and test under these independent labels. These are
descriptive file counts, not application-quality claims.

## 5. Termination precedence

Exactly one primary class is selected in this order:

1. `precondition_failed`;
2. `preservation_failed`;
3. `timeout`;
4. `process_killed`;
5. `context_overflow`;
6. `output_truncation`;
7. `model_backend_error`;
8. `harness_crash`;
9. `invalid_harness_output`;
10. `no_changes`;
11. `success`; and
12. `unknown_other`.

A sealed M3 artifact cannot itself represent preservation or precondition
failure, but those classes remain in the persisted vocabulary for later
lifecycle integration. Failure signals outrank ordinary completion. `success`
means only ordinary execution with at least one deterministically observed file
change; it is not a correctness judgment.

## 6. Separate immutable analysis artifact

Metric calculation does not modify the M2/M3 artifact. `metrics calculate`
stores a new artifact at:

```text
<analysis-root>/<run-id>/metrics-v1/
    manifest.json
    metrics.json
    checksums.sha256
```

The metrics manifest links the source artifact manifest ID and SHA256, source
run-manifest SHA256, metrics record digest, and metrics-file SHA256. The checksum
file covers both persisted JSON records. Existing destinations are rejected,
and calculation-time timestamps are deliberately absent from identity-bearing
content. Consequently, recalculating identical inputs and configuration creates
byte-identical `metrics.json` content and the same SHA256; publishing a second
revision requires a new versioned destination.

## 7. CLI

```text
agent-bench metrics calculate SOURCE_ARTIFACT ANALYSIS_ROOT
agent-bench metrics show RUN_OR_METRICS_ARTIFACT
```

`calculate` verifies the source, calculates metrics, and seals the separate
analysis artifact. `show` validates and prints an existing metrics artifact, or
calculates in memory from a sealed run artifact without writing it.

## 8. Deferred work

M4 does not implement llama.cpp/backend management, a logging proxy, real
harness adapters, report rendering, charts, cross-run interpolation, Parquet,
DuckDB, or qualitative analysis. M5 must supply exact backend/request capture
and real capture-capability declarations before real-harness metrics can be
complete.
