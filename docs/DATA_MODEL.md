# Agent Bench Persisted Data Model

Status: Milestone M0 logical model with M1–M5 concrete subsets
Specification version: 1.0.0

## 1. Conventions

This document defines logical persisted entities. It does not commit M0 to concrete Pydantic classes or a storage engine.

Every persisted record has a top-level `schema_version` string. Initial implementations use a semantic version scoped to that record type, beginning at `1.0.0`. Readers must reject unsupported major versions; migrations write a new record or dataset and never rewrite sealed evidence.

Common conventions:

- IDs are non-empty stable strings unique within their stated scope. `run_id` is globally unique.
- Timestamps are RFC 3339 UTC values with a `Z` suffix and sufficient precision to preserve source ordering.
- Durations are integer nanoseconds where captured from monotonic clocks; wall timestamps are not subtracted when a monotonic duration is available.
- Paths persisted for reproducibility use an explicit path kind: artifact-relative, worktree-relative, or execution-host absolute.
- SHA256 values are lowercase 64-character hexadecimal strings and identify exact bytes.
- Unknown and unavailable are represented explicitly, never as zero, an empty string, or an estimate.
- Extensible mappings must contain JSON/YAML-compatible values and have documented canonical serialization when hashed.
- Definitions store requested/configured state. Manifests and events store resolved/observed state. They must not be silently merged.

## 2. Immutability rules

| Record | Immutability |
|---|---|
| ExperimentDefinition | Immutable once any run is allocated; changes create a new experiment ID |
| FixedEnvironment | Immutable once referenced by an allocated run |
| ModelIdentity | Immutable identity record; observed verification is captured separately in the manifest/snapshot |
| BackendIdentity | Immutable identity record |
| HardwareIdentity | Immutable expected hardware profile |
| PromptDefinition | Immutable once referenced; content hash is authoritative |
| HarnessDefinition | Immutable once referenced |
| HarnessProfile | Immutable content-addressed/versioned bundle |
| RunDefinition | Immutable after allocation |
| RunManifest | Appendable only through defined lifecycle states until sealed; immutable after sealing |
| CaptureCapabilities | Immutable versioned declaration for one capture boundary combination |
| BackendEndpointObservation | Immutable exact `/metrics` or `/slots` response observation |
| EnvironmentSnapshot | Immutable observation |
| RawEvent | Append-only during capture; immutable after raw stream sealing |
| NormalizedEvent and event subtypes | Immutable output of one identified normalization version |
| GitChangeSummary | Immutable derived record for one baseline/result pair and algorithm version |
| ArtifactManifest | Appendable until preservation verification; immutable after sealing |
| RunMetrics | Immutable output of one metric-definition/implementation version |
| ManualReview | Append-only revision records; prior revisions immutable |
| FailedRunEvidence | Immutable after checksum sealing; existing run/failure destinations are never overwritten |

Generated normalized/metrics datasets can be recomputed only into a new versioned location with new records. Raw records and preserved application artifacts are never rewritten by recomputation.

## 3. Relationships

```text
ExperimentDefinition
 ├── 1 FixedEnvironment
 │    ├── 1 ModelIdentity
 │    ├── 1 BackendIdentity
 │    └── 1 HardwareIdentity
 ├── 1..* PromptDefinition
 ├── 1..* HarnessDefinition
 │    └── 1..* HarnessProfile
 └── 1..* RunDefinition
      └── 1 RunManifest
           ├── 0..1 CaptureCapabilities
           ├── 1 EnvironmentSnapshot
           ├── 0..* RawEvent
           ├── 0..* NormalizedEvent
           │    ├── LLMRequestEvent / LLMResponseEvent
           │    └── ToolCallEvent / other event kinds
           ├── 0..1 GitChangeSummary
           ├── 1 ArtifactManifest
           ├── 0..* RunMetrics versions
           └── 0..* ManualReview revisions

FailedRunEvidence is an alternate pre-task terminal branch keyed by `run_id`.
It exists when preflight/startup/readiness prevents creation of the ordinary
sealed result topology; it never masquerades as a successful ArtifactManifest.
```

An experiment references definitions by ID plus immutable digest/version. A run definition selects one harness profile, prompt, and repetition from that experiment. A run manifest binds those requested selections to observed identities and artifacts.

## 4. Definition entities

### 4.1 ExperimentDefinition

Declares one benchmark matrix.

Required fields:

- `schema_version`;
- `experiment_id`;
- `name` and optional description;
- `created_at`;
- `fixed_environment_id` and digest;
- `baseline_repository` and unresolved input reference plus resolved-commit policy;
- ordered harness/profile selections;
- ordered prompt IDs and digests;
- positive `repetitions`;
- execution-order policy (`sequential`, `interleaved`, or deterministic `shuffled`);
- matrix-order algorithm/version and seed when shuffled;
- task limits and preservation policy IDs/versions; and
- definition digest over canonical content.

It does not contain observed run results.

### 4.2 FixedEnvironment

Declares values that are constant across a benchmark-v1 experiment.

Required fields:

- `schema_version` and `fixed_environment_id`;
- `model_identity_id` and digest;
- `backend_identity_id` and digest;
- `hardware_identity_id` and digest;
- structured server configuration reference;
- structured request/generation configuration, including controllability metadata;
- backend restart policy;
- readiness and warmup policy IDs/versions;
- required hardware preconditions;
- relevant environment allowlist policy; and
- definition digest.

Server-start and per-request parameters are distinct fields.

### 4.3 ModelIdentity

Identifies the one fixed GGUF.

Required fields:

- `schema_version` and `model_identity_id`;
- family/name (`Qwen 3.8 27B` for benchmark v1);
- quantization identity (`Q4` plus the exact GGUF-reported variant when available);
- expected filename, byte size, and SHA256;
- expected relevant GGUF metadata as key/value entries;
- identity-source description; and
- definition digest.

The execution-host absolute path is observed per run and belongs in `RunManifest`; it is not portable identity.

### 4.4 BackendIdentity

Identifies the fixed llama.cpp build and invocation contract.

Required fields:

- `schema_version` and `backend_identity_id`;
- implementation name (`llama.cpp`);
- executable kind (`llama-server`);
- expected source commit when available;
- expected build/version output and build metadata;
- executable digest policy and expected digest when pinned;
- structured invocation template version;
- readiness, shutdown, restart, and logging policies; and
- definition digest.

Resolved executable path, argv, working directory, environment, process ID, and observed version output are per-run manifest/snapshot data.

### 4.5 HardwareIdentity

Declares the fixed expected machine/GPU profile and preconditions.

Required fields:

- `schema_version` and `hardware_identity_id`;
- host/platform identity policy;
- CPU architecture and relevant CPU constraints;
- GPU vendor/model/count and expected GPU UUIDs when the deployment pins them;
- required memory/VRAM and driver/runtime constraints;
- fixed power/performance settings when applicable;
- allowed competing-process policy;
- precondition thresholds; and
- definition digest.

Dynamic utilization, temperature, free memory, and process observations belong in `EnvironmentSnapshot`.

### 4.6 PromptDefinition

Identifies byte-exact task input.

Required fields:

- `schema_version`;
- `prompt_id`;
- `semantic_task_id`;
- `variant_label`;
- UTF-8 encoding declaration;
- exact content or artifact-relative content path;
- content byte length and SHA256;
- optional task metadata that does not alter delivered bytes; and
- definition digest.

New content, including whitespace-only changes, requires a new prompt definition/digest.

### 4.7 HarnessDefinition

Identifies one supported harness release.

Required fields:

- `schema_version` and `harness_id`;
- display name (`OpenCode`, `Hermes`, or `Pi` initially);
- upstream source/project identity;
- executable/package identity and version resolution policy;
- supported raw capture sources;
- adapter/normalizer compatibility identifiers; and
- definition digest.

No adapter implementation is part of M0.

### 4.8 HarnessProfile

Identifies a clean, versioned configuration bundle.

Required fields:

- `schema_version` and `profile_id`;
- `harness_id` and compatible harness version/range;
- profile version and kind (`controlled_default` or `benchmark_specific`);
- immutable bundle artifact reference, byte size, and SHA256;
- upstream defaults source/version;
- complete deviations from upstream defaults;
- native setting names/values with secret placeholders;
- optional normalized conceptual setting labels with semantic caveats;
- required isolated path mappings; and
- definition digest.

The profile source is copied, never mounted mutably into a run.

### 4.9 RunDefinition

Declares one matrix cell and repetition before execution.

Required fields:

- `schema_version` and globally unique `run_id`;
- `experiment_id` and definition digest;
- `harness_id`, `profile_id`, and immutable digests;
- `prompt_id`, prompt SHA256, and semantic task ID;
- one-based `repetition_index`;
- matrix position/order;
- matrix-order seed when applicable;
- requested generation seed and controllability status;
- configured limits;
- planned baseline reference and fixed-environment ID/digest; and
- allocation timestamp.

Expansion must never reuse a run ID or overwrite an existing destination.

## 5. Run evidence entities

### 5.1 RunManifest

The authoritative index for one realized run.

M3 implements the execution-linked subset described in `EVENTS_AND_RUNS.md`.
Its `execution_complete` record and sealed M2 artifact together provide the
implemented evidence; backend, hardware, metric, and full sealing fields remain
unavailable until their owning milestones rather than being fabricated.

Required fields:

- `schema_version` and `run_id`;
- `run_definition` reference/digest;
- lifecycle state and state timestamps;
- resolved baseline repository and full commit ID;
- resolved prompt/profile paths and verified digests;
- observed model path, filename, size, SHA256, and GGUF metadata;
- observed backend executable path, digest, version/build/commit, full argv array,
  intended run/server seed, working directory, and redacted environment;
- configured and backend-observed request/generation parameters as distinct structures;
- hardware identity reference and environment snapshot reference;
- isolated directory mapping recorded in redacted/portable form;
- raw stream, normalized dataset, metrics, Git summary, source snapshot, and artifact-manifest references;
- backend/task/preservation timing boundaries;
- primary termination classification and supporting event references;
- preservation verification status;
- manifest creation/sealing timestamps and digest; and
- implementation/version identities for every producer.

Lifecycle updates are append-only state transitions or an atomically replaced draft with an audit trail. The sealed manifest is immutable.

### 5.2 EnvironmentSnapshot

Captures observed execution conditions at a named phase such as preflight, task start, or task end.

Required fields:

- `schema_version`, `snapshot_id`, `run_id`, phase, and UTC timestamp;
- monotonic elapsed time when task timing has begun;
- OS/kernel, architecture, hostname policy, locale, timezone, Python/runtime, container/cgroup, and relevant library/driver versions;
- CPU and memory observations;
- GPU identity/UUID, utilization, exact reported total/used/free VRAM,
  temperature, power/performance state, and competing GPU processes where
  available;
- allowlisted non-secret environment variables;
- redacted-secret presence metadata;
- collection command/tool versions and per-field availability/errors; and
- snapshot digest.

Current-occupancy decisions require a newly collected live snapshot. The
snapshot records source command argv and UTC collection time; historical output
may be retained as evidence but cannot satisfy a later run's preflight.

Multiple snapshots may exist; the manifest identifies those required by policy.

### 5.2.1 CaptureCapabilities

M5 implements a dedicated immutable capability declaration and references it
from `RunManifest`. Each named observation uses exactly one method:
`api_exact`, `proxy_exact`, `harness_exact`, `reconstructed`, or `unavailable`.
Fields cover raw requests/responses, request parameters, input/output/reasoning
tokens, context, reasoning content, finish reason, tools/results, compaction,
serialized-history validation, and empty historical think-block detection.
Notes qualify conditional API exposure. An unavailable capability is never
upgraded merely because one run happens to contain a similarly named field.
The field is optional only for backward compatibility with pre-M5 run
manifests; controlled runs created after M5 must reference a declaration.

### 5.2.2 FailedRunEvidence

M5 implements the alternate immutable layout `runs/<run-id>/failure/` for
precondition, backend identity/hash/port/GPU, startup, readiness, and applicable
preservation failures. Its versioned manifest records the primary class and
reason; `events.jsonl` records every deterministic preflight check plus the
terminal failure; `environment.json` contains the resolved invocation,
preflight report, profile digest, and CaptureCapabilities; stdout/stderr are
retained; and `checksums.sha256` covers every other evidence file. Creation is
exclusive and checksum verification precedes publication.

### 5.2.3 BackendEndpointObservation

M5 represents a sampled llama-server `/metrics` or `/slots` response as an
immutable versioned record containing its UTC observation time, HTTP status,
content type, exact body bytes encoded as base64, body SHA256, parsed JSON when
valid, and `llama_server_endpoint_exact` provenance. It does not infer context
values that the endpoint did not expose. Request correlation is deferred until
the real run controller has a harness boundary.

### 5.3 RawEvent

An append-only envelope around one source record without semantic rewriting.

The concrete M3 1.0.0 envelope, including its stable capture sequence and record
digest, is specified in `EVENTS_AND_RUNS.md`. Source-native timing, redaction,
and external payload references remain future optional extensions.

Required fields:

- `schema_version`, `raw_event_id`, and `run_id`;
- source stream (`harness`, `backend`, `proxy`, `stdout`, `stderr`, `os`, `git`, or `hardware`);
- source-native sequence/index where available;
- capture sequence assigned by Agent Bench;
- source timestamp exactly as emitted, if any;
- capture UTC timestamp and monotonic elapsed nanoseconds, if available;
- payload encoding/content type;
- exact payload bytes or an artifact reference plus offset/length/hash;
- redaction metadata describing any policy-authorized pre-persistence redaction;
- parse status without changing the payload; and
- record digest.

If secrets can appear at a boundary, redaction occurs before durable raw persistence. Such a record is raw relative to the safe capture boundary and explicitly says that redaction occurred.

### 5.4 NormalizedEvent

The common event envelope derived deterministically from raw evidence.

M3 implements direct common-event mapping with integrity-bearing raw references.
It omits unsupported raw observations and does not populate specialized fields
that FakeHarness does not expose.

Required fields:

- `schema_version`, `event_id`, and `run_id`;
- `event_kind`;
- normalized per-run sequence;
- UTC timestamp and/or elapsed nanoseconds with clock source;
- zero or more `raw_event_refs`, including byte/record location where possible;
- normalizer name, version, and configuration digest;
- confidence class limited to `direct`, `parsed`, or `deterministically_reconstructed`;
- event-specific payload;
- parent/correlation IDs where supported; and
- event digest.

Allowed kinds initially include run start/end, LLM request/response, reasoning, tool call start/end, file read/search/edit/write, shell command, test execution, compaction start/end, output truncation, context overflow, harness/backend error, timeout, and process termination. Unsupported observations are omitted, not fabricated.

### 5.5 LLMRequestEvent

A specialized `NormalizedEvent` with `event_kind = llm_request`.

Required event payload fields:

- one-based `request_index`;
- request/correlation ID when exposed;
- request start timestamp and elapsed nanoseconds;
- endpoint/protocol identity;
- exact request-payload artifact reference and digest when capturable;
- configured/requested parameters and backend-observed parameters separately;
- `context_tokens` and `context_token_source` (`backend_exact`, `api_exact`, or `tokenizer_reconstructed`) when available;
- compatible tokenizer identity/version/digest when reconstructed;
- configured maximum context tokens;
- context-utilization percent as a derived value or derivation reference;
- compaction state when observable; and
- explicit unavailable reasons for absent fields.

Heuristic token estimates are prohibited.

### 5.6 LLMResponseEvent

A specialized `NormalizedEvent` with `event_kind = llm_response`.

Required event payload fields:

- response/correlation ID and associated request event ID;
- completion timestamp and duration when available;
- outcome (`success`, `error`, `cancelled`, `truncated`, or `unknown`);
- `output_tokens`, `reasoning_tokens`, and `visible_response_tokens` with individual source/method fields when available;
- token-accounting semantics indicating whether reasoning is a subset of output;
- finish/stop reason as observed;
- response-payload artifact reference and digest when capturable;
- error reference when applicable; and
- explicit unavailable reasons.

### 5.7 ToolCallEvent

A correlated representation of a tool invocation. It may be stored as one completed event or linked start/end normalized events, but the chosen representation is fixed by schema version.

Required payload fields:

- stable `tool_call_id` within the run;
- tool name and normalized category;
- start/end timestamps or elapsed nanoseconds when observed;
- exact native argument artifact/reference and a canonical argument digest;
- normalized path/command fields only when deterministically parsed;
- outcome (`success`, `failure`, `cancelled`, `timeout`, or `unknown`);
- native result/status/exit code where available;
- raw source references for both call and result;
- parent LLM response/turn ID when observable; and
- content identity before/after for file operations when available.

Initial categories are `read`, `search`, `edit`, `write`, `shell`, `test`, and `other`. A test may also be a shell call; aggregate definitions prevent double-counting categories ambiguously.

### 5.8 GitChangeSummary

Describes the deterministic comparison between the resolved baseline tree and the preserved result tree.

Required fields:

- `schema_version`, `run_id`, baseline commit/tree ID, and result tree/snapshot identity;
- Git version and exact comparison commands/options or algorithm version;
- counts of files changed, created, deleted, renamed, and type-changed;
- text lines added/deleted;
- binary-file counts and unavailable line-count indicators;
- per-path status and line statistics;
- untracked/ignored-file preservation treatment;
- dirty/submodule state when applicable;
- source/test/config classification rule version when such counts are produced; and
- input and record digests.

### 5.9 ArtifactManifest

Indexes every preserved file needed to inspect or reproduce a run.

Required fields:

- `schema_version`, `artifact_manifest_id`, and `run_id`;
- manifest lifecycle state (`collecting`, `verifying`, or `sealed`);
- one entry per artifact with artifact-relative path, role/layer, media type, byte size, SHA256, creation source, required/optional status, and availability;
- references for the complete source snapshot, raw streams, normalized events, metrics, commands, logs, environment evidence, and Git evidence;
- result Git pin/reference where applicable;
- build and launch command artifact references when available;
- checksum-file identity;
- verification algorithm, timestamp, and result;
- missing-artifact explanations; and
- sealed manifest digest.

Paths must be unique and traversal-safe. A manifest cannot be sealed when any required artifact is missing or fails verification.

## 6. Derived and post-run entities

### 6.1 RunMetrics

Contains deterministic metrics for one run.

Required fields:

- `schema_version`, `metrics_id`, and `run_id`;
- metric-definition specification version;
- calculator implementation/version and configuration digest;
- normalized dataset and Git summary input digests;
- no calculation timestamp in identity-bearing metric content; an operational
  storage time, if introduced later, belongs in separate non-identity metadata;
- one record per metric containing name, typed value or unavailable status, units, source references, method/source class, and edge-case notes;
- primary termination classification and supporting references;
- validation status; and
- metrics record digest.

Zero and unavailable are distinct. Recalculation under changed definitions writes a new `RunMetrics` record.
M4 implements this as structured versioned metric groups whose scalar values
carry units, availability/reason, source method, normalized-event references,
and artifact references. It stores the result in a separately checksummed
immutable analysis artifact linked to the sealed source artifact by manifest and
run-manifest SHA256. See `METRICS_ENGINE.md` for the concrete 1.0.0 contract.

### 6.2 ManualReview

Stores human assessment without changing run evidence.

Required fields:

- `schema_version`, `review_id`, `run_id`, and reviewed artifact/source-snapshot digest;
- reviewer identity or pseudonymous ID;
- revision number and superseded-review reference when applicable;
- creation timestamp;
- tested/not-tested state and test environment/commands when tested;
- tri-state `application_launches`, `requested_feature_works`, and `regressions_observed` (`yes`, `no`, or `not_assessed`);
- usability and visual-quality scores with named scale/version;
- free-text notes;
- attachments as artifact references; and
- review digest.

Corrections append a new revision. They never alter the preserved source or deterministic metrics.

### 6.3 QualitativeAnalysis

Although optional, qualitative output has its own persisted schema rather than sharing `RunMetrics`.

Required fields include `schema_version`, analysis ID/version, run/source references, selected excerpt digests, analyst or model identity, exact analysis prompt/configuration, generated interpretation, timestamp, and record digest. It is immutable and explicitly labeled non-deterministic when an LLM is used.

## 7. Availability and provenance

Every optional measured field uses one of:

- `available` with an exact typed value;
- `unavailable_source_not_exposed`;
- `unavailable_capture_failed`;
- `unavailable_not_applicable`;
- `unavailable_event_not_observed`;
- `unavailable_ambiguous`; or
- `unavailable_invalid_source`.

Metric implementations may add a more specific reason while preserving this stable category. They do not coerce unavailable to zero. Each derived field references its input record IDs/digests and a method version sufficient to reproduce it.

## 8. Canonicalization and integrity

Before hashing structured records, implementations use one versioned canonical serialization: UTF-8, normalized field names, deterministic map-key ordering, documented list ordering, no insignificant whitespace, and a defined representation for timestamps and numbers. The canonicalization algorithm/version is stored with the digest.

References across records include both ID and expected digest where practical. A matching ID with a different digest is an integrity error. Sealing a run writes and verifies the artifact manifest last, after all referenced immutable artifacts exist.
