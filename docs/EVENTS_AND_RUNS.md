# Common Events and Fake Run Lifecycle

Status: Milestone M3 implemented contract  
Schema version: 1.0.0

## 1. Scope

M3 implements the first complete single-run path using a deterministic,
test-only FakeHarness. It connects M1 `RunDefinition` records to M2 Git
isolation and preservation, adds raw and normalized event streams, allocates
fresh filesystem state, enforces a wall-clock deadline, and persists a minimal
run manifest.

M3 does not run an LLM, llama.cpp, OpenCode, Pi, or Hermes. M4 subsequently
calculates metrics and formal termination without changing this M3 evidence.

## 2. Raw event envelope

Raw records are newline-delimited canonical JSON in `raw/events.jsonl`. Each
`RawEvent` 1.0.0 record contains:

- `raw_event_id`, `run_id`, and a one-based capture `sequence`;
- `timestamp_utc` serialized as RFC 3339 UTC;
- exact runner-monotonic `elapsed_ns` and its required `elapsed_seconds`
  representation, or both null outside task timing;
- `source`, `event_type`, and an extensible JSON `payload`; and
- `record_digest`, the SHA256 of the canonical record excluding that field.

The writer creates a new file exclusively, serializes each immutable Pydantic
record in one lock-protected append, flushes it, and strictly increments capture
sequence. On sealing it flushes, calls `fsync`, closes the stream, and rejects
later emission. Sequence, not timestamp, is the authoritative within-run order.
Readers reject invalid digests, mixed run IDs, and non-increasing sequences.

The current source vocabulary is `runner`, `harness`, `backend`, `proxy`, `git`,
`system`, `stdout`, `stderr`, and `hardware`. Event-type strings and payload keys
remain extensible. An unrecognized raw event remains intact in raw storage.

## 3. Normalized events and provenance

`normalized/events.jsonl` is generated only from the sealed raw stream. The M3
normalizer is named `agent-bench-common`, version 1.0.0, and stores a digest of
its fixed event-type mapping.

Each `NormalizedEvent` 1.0.0 record contains:

- deterministic event ID and contiguous one-based normalized sequence;
- run ID and one common `event_kind`;
- the source UTC timestamp and elapsed values without recalculation;
- `clock_source = runner_monotonic`;
- confidence (`direct` for M3's directly emitted common events);
- the directly supported payload without adding absent fields;
- one or more raw references containing raw ID, sequence, and record digest;
- normalizer identity/configuration digest; and
- an integrity digest over the normalized record.

The supported common kinds are run start/end, LLM request/response, reasoning,
tool start/end, file read/search/edit/write, shell command, test execution,
compaction start/end, output truncation, context overflow, harness/backend error,
timeout, and process termination. Unsupported raw event types are omitted from
normalized output rather than inferred. Given identical raw bytes, normalization
to a new destination is byte-identical.

## 4. Minimal adapter boundary

An adapter implements three members: `adapter_id`, `adapter_version`, and
`run(context)`. The immutable context supplies exactly the current shared needs:

- the selected `RunDefinition` and generic `RunLimits`;
- exact prompt content;
- isolated workspace, HOME, XDG, and harness-state paths;
- an event sink; and
- a runner-owned cancellation signal.

The return value states whether execution completed normally and whether output
was directly observed as truncated. Adapter exceptions remain observable and are
captured as harness errors. Harness-specific configuration, parsing, and process
methods are intentionally not part of the M3 interface.

## 5. FakeHarness scenarios

FakeHarness never invokes an LLM or external harness. Every scenario writes a
fresh deterministic session record into its isolated harness state and records
that all isolated paths existed during execution.

| Scenario | Deterministic behavior | Observed M3 outcome |
|---|---|---|
| `success` | Reasons, reads a tracked file, records tool pairs, modifies that file, and creates an untracked file | `success` |
| `no_change` | Reasons and reads without mutating the workspace | `no_changes` |
| `failed_tool` | Records a failed edit call followed by a successful recovery read | `no_changes` |
| `timeout` | Waits for the runner-owned cancellation signal | `timeout` |
| `crash` | Emits reasoning and raises `FakeHarnessCrash` | `harness_crash` |
| `output_truncation` | Emits a fake LLM-response-like record and direct truncation signal | `output_truncation` |
| `reasoning_without_action` | Emits one reasoning/response turn and no tool event | `no_changes` |
| `metrics` | Emits directly sourced synthetic LLM/token/context/compaction, duplicate/read/edit/shell/test evidence and a mixed Git result | `success` |

The fake response records explicitly mark token counts not applicable. M3 does
not invent model or token observations.

## 6. Isolation and timeout behavior

Each run creates unique temporary directories for the detached Git worktree and
for `HOME`, `XDG_CONFIG_HOME`, `XDG_CACHE_HOME`, `XDG_DATA_HOME`,
`XDG_STATE_HOME`, harness state, raw events, normalized events, and the draft
run manifest. The adapter receives
an explicit mapping of the isolated HOME/XDG/state paths; the process-wide user
environment is not mutated.

`RunLimits` currently contains only positive `wall_timeout_seconds`, configured
on the experiment and copied into every expanded run definition. M3 invokes the
adapter in a dedicated thread. At the deadline it signals cancellation and
requires the adapter to stop within a one-second shutdown grace period. FakeHarness
does so cooperatively. Failure to stop is an infrastructure lifecycle failure,
not a completed benchmark result; the worktree and isolated state remain for
recovery. Future process-backed adapters remain responsible for mapping the same
cancellation boundary to process termination.

## 7. Single-run lifecycle

The M3 coordinator performs:

1. verify exact prompt bytes against the selected run definition;
2. resolve the configured baseline reference to a commit;
3. reject reused artifact destinations and result refs;
4. create fresh runtime paths and the exact detached worktree;
5. start raw capture and emit `run_start` with the isolation/baseline boundary;
6. invoke one adapter with the configured wall timeout;
7. capture normal completion, direct truncation, crash, or runner timeout;
8. record process termination and the observed execution outcome;
9. seal raw JSONL and deterministically generate normalized JSONL;
10. create the integrity-digested M3 run manifest;
11. pass runtime evidence to M2 preservation alongside the worktree result;
12. verify the sealed artifact and result ref; and
13. only then remove the worktree and temporary HOME/XDG/state directory.

Crash, timeout, truncation, and no-change outcomes follow the same preservation
path as success. Infrastructure, normalization, or preservation failure retains
temporary state and raises `RunLifecycleError` with recovery paths.

## 8. Persisted layout and run manifest

The M2 artifact gains checksummed M3 evidence while retaining its existing
source/Git layout:

```text
artifacts/<run-id>/
    manifest.json
    checksums.sha256
    raw/events.jsonl
    normalized/events.jsonl
    run/manifest.json
    run/harness-state/session.json
    source/...
    git/...
    build/
```

The M3 `RunManifest` records the run-definition digest and selected identities,
adapter/scenario identity, baseline, execution-host isolation paths, artifact
event paths, task UTC/monotonic boundaries, direct observed outcome, and terminal
raw-event references. M5 adds a dedicated optional `CaptureCapabilities` record;
FakeHarness runs populate their exact fixture capabilities, while future real
adapters must supply an honest backend/harness combination. Its lifecycle state
is `execution_complete`; the adjacent
M2 `manifest.json` remains authoritative for preservation status and result Git
identity. Both manifests and all M3 evidence are covered by `checksums.sha256`.

Execution-time event paths in the run manifest are historical host observations
and need not exist after cleanup. Artifact-relative raw/normalized paths are the
durable references.

## 9. Execution outcome is not an M4 metric

M3 persists direct outcome groundwork: `success`, `no_changes`, `timeout`,
`harness_crash`, and `output_truncation`. The raw `run_end` record explicitly
states that formal classification is deferred to M4. No timing aggregate, token
count, behavior count, Git metric, precedence calculation, or benchmark quality
judgment is performed in M3.

For ordinary FakeHarness completion only, M3 distinguishes `success` from
`no_changes` using the recorded Git porcelain observation, including untracked
and ignored paths. M4 now provides the specified complete Git comparison and
formal termination classifier in a separate analysis artifact.

## 10. Diagnostic CLI

The command below chooses exactly one run ID from the expanded experiment and
does not execute the rest of the matrix:

```text
agent-bench fake-run EXPERIMENT_PATH RUN_ID OUTPUT_ROOT --scenario success
```

It prints the run ID, scenario, observed outcome, raw and normalized event paths,
and sealed artifact path.

## 11. M5 boundary

M5 does not execute a real harness. It supplies proxy-origin `llm_request` and
`llm_response` raw records already accepted by the M3 normalizer, plus auxiliary
raw records for safe upstream request evidence, streamed chunks, empty-think
validation, and backend errors. Unrecognized auxiliary records remain preserved
raw. Exact API `prompt_tokens` on a correlated response can supply the request's
context count to M4; no heuristic token estimate is introduced.
