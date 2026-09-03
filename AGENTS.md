# Agent Bench — Development Instructions

## Purpose

Agent Bench is a deterministic benchmarking framework for comparing coding-agent harnesses.

Initial harnesses:
- OpenCode
- Hermes
- Pi

The benchmark must measure harness behavior independently from subjective application quality.

## Core principles

1. Reproducibility is more important than convenience.
2. Raw benchmark data must never be modified after a run.
3. Deterministic measurements must never depend on an LLM.
4. Qualitative analysis of reasoning may optionally use an LLM later, but it is strictly separate from metric generation.
5. Every benchmark run starts from the exact configured Git baseline.
6. Every benchmark run uses a fresh harness session.
7. Each run must be isolated from previous runs.
8. Use Git worktrees for isolated run workspaces.
9. The complete result of every run must be preserved so that a human can execute and inspect it later.
10. Do not automate subjective GUI evaluation. Human evaluation is recorded separately after runs.
11. Never overwrite an existing run or result artifact.
12. Harness-specific behavior belongs in adapters and normalizers. Core metrics must operate only on the common normalized representation.

## Experiment dimensions

Benchmark v1 varies only:

- harness
- harness profile/settings
- prompt
- repetition

Prompt quality is a first-class benchmark dimension.

Different prompts may request the same functional task while differing in precision, completeness, constraints, or openness.

Model, quantization, inference backend, and hardware are fixed for benchmark v1 and are not matrix dimensions.

## Benchmark v1 fixed environment

Benchmark v1 is specifically intended to compare coding harnesses and harness settings.

The following are NOT experiment matrix dimensions:

- model
- quantization
- inference backend
- GPU/hardware

Benchmark v1 uses one fixed environment:

- Qwen 3.8 27B
- one fixed Q4 GGUF, identified by SHA256
- llama.cpp / llama-server
- one fixed llama.cpp build/commit
- one fixed hardware profile
- one fixed set of server parameters
- one fixed set of generation/sampling parameters

These values must be recorded in every run manifest but must not vary within an experiment.

## Harness isolation

Benchmark runs must never use the user's normal mutable harness state.

Each run gets an isolated home/config/state environment.

At minimum isolate:

- HOME
- XDG_CONFIG_HOME
- XDG_CACHE_HOME
- XDG_DATA_HOME
- harness-specific configuration/state directories
- sessions
- memories
- caches where controllable

Each harness must have a versioned benchmark default profile based on a clean upstream/default installation plus only the configuration required to access the benchmark LLM.

Do not import personal user configuration, memories, sessions, skills, SOUL files, or other user customizations into the controlled default profile.

The benchmark must distinguish:

- harness default profile
- benchmark-specific harness profile/settings

A harness profile must be copied into a fresh isolated run environment before execution.

The source profile must never be modified by a run.

## Environment variables

Benchmark harness processes must not inherit the user's complete environment blindly.

Use an explicit environment-variable allowlist.

Record all non-secret environment variables that can affect benchmark behavior.

Secrets must never be written to raw logs, manifests, reports, or exported datasets.

## Prompt handling

Prompt files must be immutable during an experiment.

Each prompt must have:

- a stable prompt ID
- exact UTF-8 content
- SHA256 hash
- semantic task ID where multiple prompt variants request the same task

Prompt variants may include, for example:

- vague/open
- normal
- precise/highly constrained

All compared runs for the same prompt variant must receive byte-identical prompt content.

## Run lifecycle

A run should conceptually perform:

1. Validate experiment configuration.
2. Resolve immutable baseline Git commit.
3. Validate fixed benchmark environment.
4. Validate model identity and SHA256.
5. Validate llama.cpp identity/version/commit.
6. Validate hardware preconditions.
7. Create isolated temporary Git worktree.
8. Create isolated HOME/XDG/harness state.
9. Copy the selected immutable harness profile into the isolated run environment.
10. Create a fresh harness session.
11. Start required capture/logging.
12. Start or prepare the fixed llama.cpp backend according to benchmark policy.
13. Perform defined readiness checks and warmup if configured.
14. Execute exactly the configured prompt.
15. Enforce configured limits.
16. Stop capture.
17. Record termination reason.
18. Collect raw logs.
19. Record Git status and diff.
20. Freeze the complete resulting source tree.
21. Preserve build/executable artifacts when configured.
22. Preserve build and launch commands.
23. Preserve the resulting Git state.
24. Calculate checksums.
25. Remove temporary worktree only after successful preservation.
26. Normalize raw logs.
27. Calculate deterministic metrics.
28. Generate deterministic per-run reports.
29. Make the preserved result available for later manual review.

A failure, crash, timeout, output truncation, context overflow, malformed harness response, or no-change run is itself a valid benchmark result and must still be preserved and analyzed.

## Git baseline

The configured baseline repository must never be modified directly.

Every run must start from the exact configured baseline commit.

Use isolated Git worktrees or an equivalently strong Git-isolation mechanism.

The baseline commit identity must be recorded in every run manifest.

The resulting Git state must be preserved after the run.

Result Git objects must be pinned so that normal Git garbage collection cannot remove benchmark results.

## Result preservation

Every run receives a globally unique immutable run ID.

Every run result must remain manually executable/testable after the benchmark run has completed.

Preserve independently of the temporary worktree:

- experiment configuration
- exact prompt
- prompt SHA256
- baseline commit
- result commit/state
- complete resulting source snapshot
- build/executable artifacts when available
- build command
- launch command
- raw harness output
- raw LLM/API capture
- stdout
- stderr
- normalized events
- deterministic metrics
- environment metadata
- checksums
- termination classification

Do not rely only on a Git diff for preservation.

Temporary worktrees may only be removed after successful result preservation.

Never automatically delete immutable benchmark runs or artifacts.

## Data layers

Keep these layers strictly separate.

### Raw

Unmodified source data from:

- harnesses
- LLM/API proxy
- llama.cpp/backend
- operating-system process monitoring
- Git
- hardware monitoring

Raw data is immutable.

### Normalized

Harness-independent event representation derived deterministically from raw data.

Normalization must preserve references to the corresponding raw source records.

### Metrics

Deterministically calculated numeric or categorical measurements.

Metrics must not depend on an LLM.

### Manual review

Human-entered application assessment performed after one or more benchmark runs.

Manual review must never modify the preserved application result.

### Qualitative reasoning analysis

Optional and separate.

May use an LLM to analyze selected reasoning/thinking excerpts.

Qualitative LLM interpretations must never be mixed into deterministic metric tables as factual measurements.

## Normalized events

Harness-specific logs must be transformed into a common normalized representation.

Where data is available, normalized events should support at least:

- run start
- run end
- LLM request
- LLM response
- reasoning/thinking
- tool call start
- tool call end
- file read
- file search
- file edit
- file write
- shell command
- test execution
- compaction start
- compaction end
- output truncation
- context overflow
- harness error
- backend error
- timeout
- process termination

Do not fabricate events that cannot be derived reliably.

## Deterministic analysis

Metrics should include where source data allows:

### Timing

- wall-clock duration
- LLM duration
- tool execution duration
- shell execution duration
- time to first LLM request
- time to first tool call
- time to first edit
- time to first test command

### Tokens and context

- input tokens
- output tokens
- reasoning tokens when exposed
- visible answer tokens where separable
- total tokens
- tokens per LLM request
- tokens per turn
- context used per request
- context utilization percent
- peak context tokens
- peak context utilization percent
- context growth between requests
- tokens before first edit
- reasoning tokens before first edit
- context at first compaction
- number of compactions
- tokens before and after compaction where observable

### Agent behavior

- number of LLM requests
- total tool calls
- successful tool calls
- failed tool calls
- read calls
- search calls
- edit calls
- write calls
- shell calls
- agent-invoked tests
- calls before first edit
- calls after last edit
- exact duplicate tool calls
- repeated reads of unchanged files
- repeated identical shell commands
- turns containing reasoning but no action where observable

### Derived efficiency metrics

Where mathematically meaningful:

- tokens per tool call
- tokens per edit
- reads per edit
- searches per edit
- seconds per edit
- failed tool-call rate
- reasoning-to-output ratio

Do not imply that lower values are automatically better unless the metric definition explicitly supports that interpretation.

### Git/result metrics

- files changed
- files created
- files deleted
- lines added
- lines deleted
- source files changed
- test files changed
- configuration files changed

## Metric integrity

Do not infer facts that cannot be measured reliably.

If a metric cannot be obtained for one harness, store it as unavailable rather than estimating it.

Where possible, distinguish:

- exact server/API-provided values
- deterministically reconstructed values

Never silently mix the two.

For every persisted metric, retain enough metadata to identify its source and calculation method.

## Context measurement

Every benchmark run must capture context usage at every observable LLM request.

At minimum store:

- request index
- timestamp
- elapsed time from run start
- input/context token count
- configured maximum context
- context-utilization percent
- source/method of token count

When available also store:

- output tokens
- reasoning tokens
- visible response tokens
- compaction state

If exact context-token values are exposed by the server/API, use them as the primary source.

If they are not exposed, deterministic reconstruction from the exact request payload may be used if a compatible tokenizer is available.

Reconstructed values must be explicitly labeled as reconstructed.

Do not use heuristic token estimates.

## Context reporting

Every benchmark run must generate deterministic context-usage charts.

Per-run reports must include:

1. context utilization versus absolute elapsed wall-clock time,
2. context utilization versus normalized run progress from 0% to 100%,
3. context utilization versus LLM request/turn index.

Where useful, also provide absolute context-token charts.

Important events should be marked where observable:

- first tool call
- first edit
- first test command
- compaction
- output truncation
- context overflow
- run completion
- run failure
- timeout

## Cross-run context comparison

Cross-run reports must support:

- overlaying individual runs on an absolute time axis
- overlaying individual runs on a normalized 0–100% time axis
- comparison by LLM request index
- grouping by harness
- grouping by harness profile/settings
- grouping by prompt
- grouping by harness/profile/prompt combination

For repeated runs, support:

- individual-run lines
- median curves
- interquartile or explicitly defined percentile spread

Normalized-time comparison must use a documented deterministic interpolation method onto a common progress grid.

A default grid such as 0%, 1%, 2%, ..., 100% may be used.

The interpolation algorithm, treatment of missing points, and treatment of runs ending in failure must be explicitly specified before implementation.

## Manual evaluation

Subjective GUI/application quality is not automatically evaluated in benchmark v1.

A user may manually test preserved application results later.

Manual review should support at least:

- tested/not tested
- application launches
- requested feature works
- regressions observed
- usability score
- visual-quality score
- free-text notes

Manual-review data must remain separate from deterministic metrics.

Manual review may happen long after the benchmark run.

## Harness comparison integrity

Do not pretend harness settings with different semantics are equivalent.

Store:

- raw native setting
- normalized conceptual category where meaningful
- harness/version to which the setting applies

For example, different harnesses may expose different reasoning controls.

The benchmark may compare such settings but must document semantic differences.

## Default harness profiles

The controlled benchmark must use clean, versioned default profiles.

A default profile should be based on an upstream/fresh-install default and modified only as required to:

- connect to the benchmark LLM endpoint
- provide required authentication without logging secrets
- satisfy unavoidable non-interactive execution requirements

Default profiles must not include:

- personal memories
- old sessions
- personal SOUL files
- user-specific skills
- unrelated plugins
- project-specific state
- prior benchmark state

All deviations from upstream defaults must be documented.

## Fixed inference backend

Benchmark v1 uses llama.cpp / llama-server only.

Backend comparison is explicitly out of scope for benchmark v1.

The benchmark must record:

- llama.cpp executable path
- llama.cpp version/build information
- llama.cpp Git commit when available
- exact command-line arguments
- working directory
- relevant environment variables
- readiness-check method
- shutdown method
- warmup policy
- server logs

The exact backend command must be reproducible from persisted configuration.

Do not hide the command in opaque shell scripts without also recording the fully resolved invocation.

## Fixed model

Benchmark v1 uses:

- Qwen 3.8 27B
- one fixed Q4 GGUF

The exact GGUF file must be identified by:

- full path at execution time
- filename
- file size
- SHA256
- relevant GGUF metadata where available
- quantization identity

Model comparison and quantization comparison are explicitly out of scope for benchmark v1.

## Generation/request parameters

Server-start parameters and per-request generation parameters are separate concepts and must be stored separately.

Where applicable, record:

- temperature
- top_p
- top_k
- min_p
- seed
- maximum output tokens
- stop sequences
- reasoning configuration
- other harness-supplied request parameters

The benchmark should distinguish:

- configured/requested parameter values
- parameter values actually observed at the LLM/backend boundary

A logging proxy may later be used to capture the latter.

## Backend restart and warmup

The benchmark must explicitly define:

- whether llama-server restarts for every run
- whether warmup occurs
- exact warmup request
- whether warmup is included in benchmark timing

For controlled benchmark mode, prefer strong run isolation.

If the backend is restarted per run, model-load/startup time must be measured separately from task execution time.

Benchmark task timing must begin at a precisely documented point after readiness and optional warmup.

## Hardware preconditions

Hardware is fixed for benchmark v1.

Each run should record relevant hardware state before execution where practical, including:

- GPU identity
- GPU UUID where available
- VRAM usage
- GPU utilization
- temperature
- power state where available
- relevant competing GPU processes

If required preconditions are violated, the benchmark should classify the run as invalid/precondition-failed rather than silently continuing.

## Termination classification

Runs must use explicit deterministic termination classes.

At minimum consider:

- success
- timeout
- harness crash
- model/backend error
- context overflow
- output truncation
- no changes
- process killed
- precondition failed
- unknown/other

A run may preserve multiple observed error events while still having one primary termination classification.

## Repetitions and randomness

A single run is not sufficient for comparative conclusions.

Experiment definitions must support repeated runs.

Where a seed is controllable:

- record it
- make seed assignment deterministic
- use comparable seeds across configurations where semantically valid

Where a seed is not controllable, record that fact.

Experiment execution order should be configurable.

Support randomized/interleaved ordering so that all runs from one harness are not necessarily executed consecutively.

The randomization seed for matrix ordering must itself be recorded.

## Storage

Keep raw run data and normalized analytical data separate.

Likely future storage formats include:

- JSON/JSONL for manifests and append-only event data
- Parquet for normalized large analytical tables
- DuckDB for deterministic querying and aggregation

Do not introduce storage dependencies before their milestone requires them.

## Reporting

Reporting must be deterministic.

Per-run reporting should eventually include:

- immutable run identity
- environment identity
- harness/profile
- prompt
- timing summary
- token/context summary
- tool-call summary
- Git/result summary
- termination status
- context charts
- links/references to preserved source and executable artifacts

Cross-run reports should eventually include deterministic comparisons across:

- harness
- profile/settings
- prompt
- repetition

Do not use an LLM to generate numerical benchmark conclusions.

## Development approach

Implement incrementally.

Do not attempt the entire architecture in one change.

Before implementing a feature:

1. inspect the relevant existing code,
2. state the intended scope,
3. make the smallest coherent implementation,
4. add or update tests,
5. run the relevant tests.

Prefer simple explicit code over abstraction without demonstrated need.

## Testing strategy

The benchmarking framework itself must be thoroughly automated and tested.

Use fake harnesses and fixture logs to test:

- experiment-matrix generation
- run lifecycle
- timeout behavior
- crash behavior
- preservation
- normalization
- metrics
- context calculations
- reporting calculations

Tests must not require OpenCode, Hermes, Pi, a GPU, llama.cpp, or a real LLM unless explicitly marked as integration tests.

Unit tests must be deterministic.

## Safety for benchmark subjects

Never modify the configured baseline repository directly.

All agent execution must occur inside isolated worktrees.

Before deleting a temporary worktree, verify that result preservation completed successfully.

Never automatically delete immutable benchmark runs or artifacts.

## Coding standards

- Python 3.12+
- Type hints for public interfaces
- Pydantic models for persisted structured configuration/data where appropriate
- pathlib instead of manual path concatenation
- UTC timestamps internally
- explicit schema/version fields in persisted formats
- JSONL for append-only event streams
- pytest for tests

Introduce Parquet, DuckDB, plotting libraries, FastAPI, or other heavier dependencies only when their milestone requires them.

## Git

Keep changes focused.

Do not rewrite existing commits unless explicitly requested.

Run the relevant tests before considering a task complete.

Do not proceed automatically from one milestone to the next without an explicit instruction.
