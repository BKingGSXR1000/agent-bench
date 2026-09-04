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

## M9 — Analytical storage and deterministic reporting

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

## M10 — Manual review workflow

Implement:
- list unreviewed preserved results
- show run identity without modifying the result
- display or expose the stored launch command
- enter manual assessment
- persist ManualReview separately
- compare manual scores with deterministic benchmark metrics

Manual GUI/application testing itself remains human-operated.

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
