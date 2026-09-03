# Agent Bench Specification

Status: Milestone M0 architecture specification  
Specification version: 1.0.0

## 1. Purpose

Agent Bench is a deterministic framework for comparing coding-agent harnesses while holding the model, inference backend, hardware, and task inputs fixed. Its purpose is to measure observable harness behavior: how a harness uses the model, tools, context, time, and repository while attempting the same coding task.

Benchmark v1 has four goals:

1. Produce reproducible runs from an exact Git baseline and byte-identical prompt.
2. Preserve enough evidence to independently inspect, analyze, launch, and manually test every result later.
3. Normalize harness-specific telemetry into a common, evidence-linked event representation.
4. Calculate and report deterministic measurements without using an LLM.

The initial harnesses are OpenCode, Hermes, and Pi. Harness-specific capture and interpretation belong at the system boundary; shared metrics operate only on normalized events and preserved Git/result evidence.

## 2. Scope and non-goals

### 2.1 Benchmark-v1 scope

Benchmark v1 varies only:

- harness;
- harness profile/settings;
- prompt; and
- repetition.

Prompt is a first-class dimension. Several immutable prompt variants may share a semantic task ID while differing in precision or openness.

### 2.2 Explicit non-goals

Benchmark v1 does not:

- compare models, quantizations, inference backends, GPUs, or other hardware;
- treat settings with different harness-native semantics as equivalent;
- judge subjective application quality automatically;
- use an LLM to generate or alter deterministic metrics;
- infer events or measurements that are not supported by captured evidence;
- modify a configured baseline repository directly;
- reuse personal harness configuration, memories, sessions, skills, caches, or state;
- discard failed, timed-out, crashed, truncated, context-overflow, malformed, or no-change runs;
- rely on a Git diff as the sole preservation mechanism; or
- introduce DuckDB, Parquet, a web service, charts, HTML reports, or execution machinery in M0.

Adapters, llama.cpp lifecycle management, a logging proxy, worktree creation, execution orchestration, artifact building, manual-review UI, and rendered reports are later milestones.

## 3. Fixed benchmark-v1 environment

Every run in a benchmark-v1 experiment uses one fixed environment:

- one Qwen 3.8 27B model;
- one fixed Q4 GGUF file identified by its SHA256 digest;
- llama.cpp and `llama-server` from one fixed build/commit;
- one fixed hardware profile;
- one fixed set of server-start parameters; and
- one fixed set of request/generation parameters.

These properties are recorded in every run manifest but are not matrix dimensions. Validation must fail before task execution if a run cannot prove it is using the experiment's fixed environment. Configured/requested generation values and values observed at the backend boundary are separate records; disagreement is retained rather than reconciled silently.

The actual model path, digest, llama.cpp build, hardware identity, server arguments, and generation values are deployment inputs and are intentionally not selected in M0.

## 4. Conceptual architecture

The architecture has six boundaries:

1. **Definitions** describe the experiment, fixed environment, prompts, harnesses, profiles, and individual runs.
2. **Run control** validates inputs, prepares isolation, starts capture/backend processes, invokes one fresh harness session, enforces limits, and records termination.
3. **Preservation** freezes raw evidence, the complete resulting source tree, Git state, relevant build outputs, commands, identities, and checksums before temporary resources may be removed.
4. **Normalization** deterministically maps harness-specific raw records to a common event vocabulary while retaining source references.
5. **Metrics and reporting** derive versioned deterministic outputs from normalized events and preserved result evidence.
6. **Post-run assessment** stores manual application review and optional qualitative reasoning analysis in separate data layers.

Only the package/CLI placeholder and these contracts are created in M0. The boundaries are specified now so later components do not weaken isolation, preservation, or metric provenance.

## 5. Experiment definitions

An experiment references exactly one fixed environment, one immutable Git baseline, one or more harness/profile selections, one or more prompt definitions, and a positive repetition count. The expanded run matrix is the Cartesian product of harness/profile, prompt, and repetition unless an explicitly versioned selection rule says otherwise.

Execution order may be sequential, interleaved, or deterministically shuffled. A shuffled order records its algorithm and seed. A seed used for model generation is distinct from the matrix-order seed. When a harness cannot control a requested seed, the run records that limitation rather than claiming equivalence.

Each prompt definition has a stable prompt ID, semantic task ID, variant label, exact UTF-8 bytes, and SHA256. Prompt bytes cannot change within an experiment. Every compared run for a prompt ID receives those exact bytes.

## 6. Harnesses and profiles

The initial harness set is:

- OpenCode;
- Hermes; and
- Pi.

A harness definition identifies the harness and executable/version. A harness profile identifies an immutable, versioned configuration bundle and the harness/version for which it was created.

Each harness has a controlled benchmark default profile based on a clean upstream installation, modified only as required to connect to the benchmark endpoint, supply authentication safely, and support unavoidable non-interactive operation. Deviations from upstream defaults are documented. Default profiles must not contain personal memories, previous sessions, SOUL files, user skills, unrelated plugins, project state, or prior benchmark state.

Harness-native settings are stored verbatim. A normalized conceptual label may also be stored when meaningful, but it never replaces the native value and must document semantic limitations.

The immutable source profile is copied into the run's isolated environment. A run never modifies the source profile. Every run starts a fresh harness session; session reuse is prohibited even between repetitions with otherwise identical settings.

## 7. Per-run isolation

Every run receives a globally unique run ID and separate temporary directories for:

- the Git worktree;
- `HOME`;
- `XDG_CONFIG_HOME`;
- `XDG_CACHE_HOME`;
- `XDG_DATA_HOME`;
- harness-specific configuration/state;
- sessions and memories; and
- controllable caches and temporary files.

Harness processes receive an explicit environment-variable allowlist, not the invoking user's full environment. Allowed non-secret variables that may affect behavior are recorded. Secret values are injected only through an approved mechanism and are never persisted; evidence stores a redacted name/presence record where needed.

No run may read mutable state produced by another run. Cleanup is allowed only for temporary resources and only after preservation and checksum verification succeed. Immutable run results are never automatically deleted or overwritten.

## 8. Git baseline and result isolation

The baseline is identified by repository identity and a resolved full commit object ID. A branch or tag may be an input convenience, but it is resolved before matrix execution and the commit ID is authoritative.

Each run operates in a fresh detached Git worktree at that exact commit. The configured baseline checkout is never used as the agent workspace and is never modified. Submodule and large-file state, when applicable, are resolved and recorded as part of baseline identity.

At completion, Agent Bench records Git status, deterministic diff evidence, result-tree identity, and a complete source snapshot independent of the temporary worktree. Any result commit or Git objects needed to reconstruct the outcome are pinned against garbage collection. Untracked and ignored files relevant to executing the result are handled by an explicit preservation policy and cannot be omitted silently.

## 9. Complete result preservation

Every terminal outcome produces an immutable run directory. At minimum it preserves:

- the experiment and run definitions;
- the resolved fixed environment and environment snapshot;
- exact prompt bytes and prompt SHA256;
- baseline repository and commit identity;
- resulting Git state, status, and diff evidence;
- a complete resulting source snapshot;
- build/executable artifacts when the run or later policy produces them;
- exact build and launch commands when known;
- raw harness, backend, proxy, process, stdout, and stderr records;
- normalized events linked to raw records;
- deterministic metrics and report data;
- termination classification and supporting observations;
- artifact metadata; and
- checksums covering preserved files.

Preservation is successful only after required files are durably written, an artifact manifest is complete, and recorded checksums verify. A preservation failure is itself recorded and prevents automatic worktree deletion. A result must remain available for a human to launch or test later; a Git diff alone is insufficient.

## 10. Run lifecycle

A run proceeds through the following conceptual stages:

1. Validate schemas and experiment configuration.
2. Resolve and verify the immutable baseline commit.
3. Resolve the selected prompt, harness, and profile.
4. Verify prompt/profile integrity hashes.
5. Validate model file identity and SHA256.
6. Validate llama.cpp executable/build/commit identity.
7. Validate fixed hardware and preconditions.
8. Allocate the unique run ID and non-overwriting result destination.
9. Create a temporary worktree at the baseline commit.
10. Create isolated HOME/XDG/harness directories.
11. Copy the immutable profile and prepare a fresh session.
12. Start raw capture.
13. Start a fresh llama-server instance and perform readiness checks.
14. Perform the fixed warmup policy, if enabled, outside task timing.
15. Mark task start and submit the exact prompt to the harness once.
16. Enforce configured time/resource/output limits.
17. Observe task termination and stop capture.
18. Record the primary termination class and all supporting error events.
19. Collect raw records, process outputs, environment metadata, and Git evidence.
20. Freeze the complete resulting source tree and configured artifacts.
21. Preserve commands and resulting Git state and pin required Git objects.
22. Write the artifact manifest and checksums, then verify preservation.
23. Only after successful verification, remove temporary resources.
24. Normalize raw records deterministically.
25. Calculate versioned deterministic metrics.
26. Produce deterministic report data and expose the preserved result for manual review.

Backend startup/readiness/model-load time and optional warmup time are recorded separately and excluded from task wall time. Exact timing boundaries are defined in `METRICS.md`.

## 11. Failures and termination

A failure is a benchmark result, not a missing result. Timeout, harness crash, model/backend error, output truncation, context overflow, process kill, precondition failure, no-change completion, malformed output, preservation failure, and unknown failures retain all evidence available at the point of failure.

Each run has one deterministic primary termination classification and may have multiple error/diagnostic events. The classification rules and precedence are defined in `METRICS.md`. Missing telemetry remains unavailable; normalizers must not fabricate successful tool completions, token counts, or inferred events.

When a process must be killed after a timeout, both observations are retained while the primary class follows the documented precedence. When preservation cannot be verified, temporary workspaces are retained for recovery and the result is never reported as an ordinary successful run.

## 12. Data layers

The following layers are physically and logically distinct:

| Layer | Contents | Mutation policy | May use an LLM? |
|---|---|---|---|
| Raw | Original harness/backend/proxy/OS/Git/hardware records | Append-only during capture, immutable after sealing | No |
| Normalized | Common events deterministically derived from raw records | Immutable generated dataset; regeneration creates a new version | No |
| Metrics | Deterministic measurements and provenance | Immutable generated dataset; regeneration creates a new version | No |
| Manual review | Human-entered post-run application assessment | Append-only revisions, separate from preserved result | Human judgment only |
| Qualitative analysis | Optional interpretation of selected reasoning evidence | Versioned independent artifact | Yes, if explicitly identified |

Qualitative conclusions are never inserted into deterministic metric tables as facts. Manual review never mutates the preserved application. Layer relationships and schemas are specified in `DATA_MODEL.md`.

## 13. Deterministic and qualitative analysis

All numerical and categorical benchmark measurements are calculated by versioned deterministic code from preserved inputs. A metric includes its source/method metadata and is unavailable when its prerequisites are absent. Exact server/API counts and deterministic tokenizer reconstructions are distinguished.

Optional qualitative reasoning analysis may use an LLM after a run, but it records the analysis model, prompt, source excerpts, and provenance in the qualitative layer. It has no authority over normalization, termination classification, metrics, or deterministic report summaries.

## 14. Manual application evaluation

Subjective application quality is evaluated manually after preservation, potentially long after execution. A manual review can record whether the result was tested, whether it launches, whether the requested feature works, observed regressions, usability and visual-quality scores, and notes.

Reviewers operate on a copy or controlled launch of the preserved result. Review records reference the immutable run/artifacts and never edit them. Automated test outcomes produced during the agent run remain deterministic events/metrics and are not substitutes for manual application judgment.

## 15. Context measurement and reporting

At every observable LLM request, normalization captures request index, UTC timestamp, elapsed task time, context/input token count, configured maximum context, utilization percent, and count source/method. Output, reasoning, visible-response tokens, and compaction state are included when exposed.

Exact backend/API counts are preferred. Compatible-tokenizer reconstruction from the exact preserved request payload is allowed and explicitly labeled. Heuristic token estimation is forbidden.

Per-run deterministic report data supports context utilization against:

- absolute elapsed task time;
- normalized run progress from 0% through 100%; and
- LLM request index.

Observable first-tool, first-edit, first-test, compaction, truncation, overflow, completion, failure, and timeout events are available as markers.

Cross-run comparisons support individual overlays, request-index alignment, normalized-time alignment, grouping by harness/profile/prompt and their combinations, and repeated-run median and percentile bands. `REPORTING.md` defines the fixed interpolation and aggregation rules. M0 defines these outputs but does not render charts or reports.

## 16. M0 implementation boundary

M0 consists only of this specification set, package metadata, an importable `agent_bench` package, a version/help-only CLI placeholder, and smoke tests. It intentionally contains no persisted Pydantic models or runtime orchestration because the milestone asks for the contract before implementation and speculative abstractions are prohibited.

Subsequent milestones must select a small coherent portion of this contract, inspect current code, implement it with deterministic fake-backed tests, and stop at that milestone boundary.
