# Agent Bench Implementation Roadmap

This roadmap defines implementation order. `AGENTS.md` remains authoritative for
development rules.

## M0 — Specification and project skeleton
Status: completed

Includes:
- architecture/specification
- data-model specification
- metric definitions
- reproducibility requirements
- reporting requirements
- minimal Python package
- smoke tests

## M1 — Configuration models and experiment matrix

Status: completed

Implement:
- typed persisted configuration models
- FixedEnvironment
- PromptDefinition
- HarnessDefinition
- HarnessProfile
- RunDefinition
- ExperimentDefinition
- YAML loading and validation
- prompt loading and SHA256 verification
- deterministic Cartesian matrix generation for:
  - harness
  - harness profile/settings
  - prompt
  - repetition
- deterministic/interleaved execution ordering with recorded ordering seed
- unique deterministic or traceable run IDs

Do not execute harnesses yet.

## M2 — Git isolation and immutable result preservation

Status: completed

Implement and test:
- immutable baseline resolution
- temporary Git worktree creation
- guarantee baseline repository is never directly modified
- result Git state capture
- result commit/ref pinning
- full source snapshot preservation
- artifact manifest
- checksums
- safe worktree cleanup only after preservation succeeds
- restoration/testability of preserved results

Use a fake operation instead of a real coding harness.

## M3 — Common event model and FakeHarness

Status: completed

Implement:
- raw event envelope
- normalized event schema
- timestamps and sequence numbers
- FakeHarness adapter
- deterministic fake behaviors:
  - successful edit
  - no change
  - tool calls
  - failed tool call
  - reasoning event
  - timeout
  - crash
  - output truncation
- complete run lifecycle using FakeHarness

No real LLM or real harness yet.

## M4 — Deterministic metrics engine

Status: completed

Implement metrics from docs/METRICS.md using fixtures and FakeHarness runs.

Include:
- timing
- tokens
- context
- tool behavior
- repeated/duplicate actions
- Git/result metrics
- termination classification

Metrics must remain completely LLM-independent.

## M5 — Fixed llama.cpp environment and LLM logging proxy

Status: completed

Implement:
- fixed llama.cpp backend profile
- exact executable/model identity
- model SHA256
- exact resolved llama-server command
- deterministic repetition seed passed explicitly to llama-server
- backend readiness checks
- restart policy
- warmup policy
- backend stdout/stderr capture
- OpenAI-compatible transparent logging proxy
- capture actual request payloads and responses
- secret redaction
- context/token observations where available
- versioned CaptureCapabilities declarations
- fresh RTX 3090 identity/process telemetry with unavailable-evidence
  fail-closed behavior and no arbitrary idle-VRAM cutoff
- immutable FailedRunEvidence for pre-task backend failures
- deterministic empty-history-think validation support

The proxy must not intentionally alter requests or responses.

## M6 — OpenCode adapter

Status: completed

Implement controlled OpenCode execution:
- isolated HOME/XDG/state
- clean versioned default profile
- fresh session per run
- non-interactive invocation
- raw log capture
- translation into normalized events
- end-to-end run against the fixed llama.cpp environment

## M7 — Pi adapter

Status: completed

Same requirements as M6 for Pi.

## M8 — Hermes adapter

Status: completed

Same requirements as M6 for Hermes.

## M9A — Benchmark Subject / Frozen Baseline / Tasks / Prompts

Status: completed

Implement the independent frozen benchmark subject, its baseline Git commit,
task suite, byte-exact vague/normal/precise prompts, evaluator-only acceptance
criteria, manual-review dimensions, and representative non-executed expansion.

## M9B — Experiment Executor / Resume / Matrix Automation / Progress

Status: completed

Implements the sequential controlled matrix executor, portable v2 identities,
fresh bundle-backed baseline materialization, atomic progress/resume, subset
and dry-run planning, global toolchain preflight, and publication-ready local
payload verification policy. Result commits are transferred before cleanup into
an output-root-scoped bare store, and bootstrap provenance is actionable. The
authoritative one-run host smoke completed with result ref
`refs/agent-bench/results/hermes-hermes-default-v1-keyboard-entry-vague-r001-e041841d2df985953b43d6c4`
at commit `c8f9bb076ce7436ddf400a779a0e1785d94931ad`; its immutable artifacts,
analysis artifacts, persistent Git ancestry, and cleanup all verified. Earlier
smokes remain immutable diagnostic evidence. M9C remains unstarted.

## M9C — Reporting / DuckDB / Parquet / HTML / Charts

Status: completed

Introduce only now as required:
- Parquet/PyArrow
- DuckDB
- plotting library
- HTML reporting

Implement per-run reports:
- context utilization vs absolute elapsed time
- context utilization vs normalized 0–100% run progress
- context utilization vs LLM request index
- event markers
- metric summary
- artifact references

Implement cross-run reports:
- individual overlays on absolute time
- individual overlays on normalized time
- request-index comparisons
- grouping by harness/profile/prompt
- median curves
- interquartile/defined percentile bands
- deterministic tables

All interpolation and aggregation must follow docs/REPORTING.md exactly.

Implemented `report-schema-v1`: verified derived ingestion, stable Parquet
tables, a DuckDB database/views, Type-7 aggregate summaries, provenance-safe
timing/context series and markers, offline static HTML/SVG, checksummed report
manifests, public-safe exports, and raw-archive metadata. M9C was validated
against the read-only M9B v3 smoke as `PARTIAL EXPERIMENT — 1 / 135 completed`;
no model or harness execution was performed.

## M10 — Manual review workflow

Status: completed

Implement:
- list unreviewed preserved results
- show run identity without modifying the result
- display or expose the stored launch command
- enter manual assessment
- persist ManualReview separately
- compare manual scores with deterministic benchmark metrics

Manual GUI/application testing itself remains human-operated.

M10 implements versioned, immutable human functional acceptance records,
canonical task criteria shared across prompt variants, a local-only blinded
browser dashboard with isolated fixture-backed restoration and reset, deterministic
queue ordering, and separate quality aggregations. It does not modify M9 evidence or auto-review
the baseline dataset. Future harness-profile matrices remain a new explicitly
planned experiment: 3 profiles per harness would expand the current 135-cell
default-profile matrix to 405 cells while keeping model/backend fixed.

## M12 — Automated Functional Benchmark Suite v1

Status: completed

Implements a separate, deterministic and headless functional-validation
dimension. M12 adds frozen `taskboard-v1`, its visible Node baseline check, the
evaluator-owned `task-priority-v1` acceptance suite, and the Medium
`combined-filtering-v1` suite over a separately frozen priority-derived
baseline, plus Complex `multi-project-migration-v1` over a frozen filtering
derived baseline. All have recorded untouched-baseline discrimination,
known-good and targeted known-bad validator self-validation, create-only
versioned JSON results, individual outcomes, category counts, separate hard
gates and scoring, and `agent-bench functional baseline-check` /
`agent-bench functional validate` / `agent-bench functional self-check`
commands. `taskboard-functional-v1` now seals all three scenario identities,
prompts, fixture vectors, visible-health checks, evaluator-leakage checks, and
result-schema consistency into one self-validating Functional Suite v1. M12
scenario construction is complete; executor integration is the next separate
step. It deliberately does not change live executor behavior, generate a
composite efficiency score, or run a browser/GPU/real harness.

## M11 — Qualitative reasoning export

Optional later phase.

Implement deterministic selection/export of reasoning material for external LLM analysis without sending full raw benchmark datasets.

Examples:
- runs with unusually high reasoning
- reasoning before first edit
- reasoning-only turns
- selected failure runs
- anomalous/repetitive runs

This component must not generate deterministic benchmark metrics.

## Milestone discipline

Milestones are intentionally incremental. No milestone starts automatically: each
requires explicit instruction, tests, and review before proceeding. Later milestones
may be refined when earlier implementation reveals constraints. This roadmap describes
implementation order, while `AGENTS.md` remains authoritative for development rules.
