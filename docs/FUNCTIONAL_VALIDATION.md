# Automated Functional Validation v1

M12 adds a deterministic, headless functional dimension. It is intentionally
separate from timing, tokens, tool calls, reasoning, and manual review: it does
not generate an efficiency score.

## Behavior-first evaluator policy

Functional validators test task-required behavior, not an evaluator-preferred
internal architecture. A feature must not fail merely because it uses a
different internal API, DOM structure, storage schema, serialization layout,
or naming convention, unless that contract was explicitly required by the
prompt or was already a frozen public baseline contract.

When the current headless Node evaluator cannot verify behavior without making
one of those assumptions, it emits `manual_review_required`; it must not
invent an implementation-specific automated FAIL. Manual adjudication remains
append-only and separate from automated evidence.

## v2 revalidation and local adjudication

`task-priority-v2` corrects an invalid v1 implementation requirement: priority
is no longer required to appear in `describeTask()`. It proves priority
creation, Low-to-High editing, state, and reload persistence. The
implementation-neutral visual criteria (creation control and per-task display)
are emitted as `manual_review_required`, yielding `needs_review`, never a
false automated failure.

`combined-filtering-v2` retains the same `combined-filtering-v1` task ID and
its eight frozen baseline regressions as hard automated checks. Its remaining
22 search/filter interaction requirements are manual-review evidence: this
evaluator has no implementation-neutral real-browser interaction layer.
`multi-project-migration-v2` similarly retains 11 frozen filtering-baseline
checks and marks its 22 project, migration, and import/export requirements for
manual review. The v1 validators and their sealed artifacts are immutable
historical audit evidence; they are not replaced or rewritten.

Existing sealed snapshots can be assessed without rerunning an agent, model,
or backend:

```sh
agent-bench functional revalidate EXPERIMENT_OUTPUT RUN_ID \
  --experiment-definition experiments/taskboard-functional-easy-v1.yaml
```

This writes a separate create-only
`analysis/RUN_ID/functional-validation-v2/` artifact bound to the run and
sealed snapshot hashes. The report resolves correctness in this order: active
manual PASS, compatible v2 evidence, then v1 evidence. The report server's
one-click **Mark result OK** appends
`EXPERIMENT_OUTPUT/adjudications/RUN_ID/revision-###.json`; Undo appends a
`revoked` revision. Neither action mutates automated evidence.

The revalidation registry maps `task-priority-v1`, `combined-filtering-v1`,
and `multi-project-migration-v1` to their corrected v2 evaluator definitions.
It restores only the sealed source snapshot and starts no harness, model, or
backend. Report precedence is active manual PASS, compatible automated v2,
then automated v1.

## Scenario anatomy

Each evaluator-owned YAML definition under `functional/scenarios/` names a
frozen subject, an external Node validator, the exact untouched-baseline result
vector, and hard-gate groups. The validator is never placed inside the subject
workspace, so a harness receives the product and its visible baseline tests but
not the acceptance assertions.

`task-priority-v1` uses the `taskboard-v1` frozen baseline. The subject has a
small documented JavaScript domain interface shared by its browser UI and its
visible baseline test. The hidden validator exercises that public behavior in
Node without a browser, network, package install, GPU, or visible GUI.

`combined-filtering-v1` is the Medium scenario. It deliberately uses strategy
**B**, the separately frozen `taskboard-priority-v1` derived baseline, rather
than asking agents to add priority and filtering in one run. Its scenario
identity and each result provenance record that strategy and rationale. Priority
is consequently a regression contract in Medium; search, three persisted
filters, AND semantics, active-filter edits/deletes, clear-all, and recoverable
zero results are the requested work.

`multi-project-migration-v1` is the Complex scenario. It uses the third frozen
derived lineage, `taskboard-filtering-v1`, so project membership, archived
project retention, v1-to-v2 persistence migration, and atomic import/export
are measured without reimplementing priority or filters. Its reference spans
project-state, migration, project-scope, import/export, filtering, and board
state modules; solutions remain free to choose another architecture.

Before any Complex benchmark execution, use the future
`experiments/taskboard-functional-complex-v2.yaml` definition and its v2
association. `functional/experiments/taskboard-functional-smoke-v2.yaml`
provides the matching read-only normal/R001 smoke plan.

## Lifecycle

Before benchmark execution, run:

```sh
agent-bench functional baseline-check functional/scenarios/task-priority-v1.yaml \
  --output /new/location/task-priority-baseline.json
```

This reconstructs a new checkout from the checked-in Git bundle, verifies the
baseline identity, and runs the acceptance suite. The command succeeds only if
the exact recorded vector is observed: all baseline-regression tests pass while
the deliberately absent priority tests fail. This is a successful
discrimination check, not a benchmark failure.

Before a scenario is considered validated, prove the validator itself with:

```sh
agent-bench functional self-check functional/scenarios/task-priority-v1.yaml \
  --output /new/location/task-priority-self-check
```

This command starts a new disposable bundle checkout for each evaluator-owned
fixture and never changes either the frozen subject or an agent workspace. It
records one create-only JSON result per fixture. For `task-priority-v1`, the
fixtures are the untouched baseline, a complete reference implementation, a
priority-not-persisted mutation, and a regression-breaking delete mutation.
Reference overlays live under `functional/references/`, outside the subject
workspace exposed to agents.

After preserving a completed agent workspace, run:

```sh
agent-bench functional validate functional/scenarios/task-priority-v1.yaml \
  /preserved/source --run-id RUN_ID --output /new/location/RUN_ID.functional.json
```

The target workspace is read only. Output paths are create-only: a pre-existing
result is rejected rather than replaced. Integration with the live executor is
deliberately deferred; callers must validate an already preserved workspace.

## Functional Suite v1

`functional/suites/taskboard-functional-v1.yaml` seals the Easy, Medium, and
Complex scenarios into one suite. It records exact scenario/validator/prompt
identities, frozen baseline lineage, expected fixture-vector digests, and a
descriptive complexity inventory. Complexity is never used in a score or hard
gate.

Run every suite invariant with:

```sh
agent-bench functional self-check --all --output /new/location/taskboard-suite
```

Use `--json` for machine-readable output. The suite command materializes every
frozen baseline, runs its visible health command, verifies all scenario
self-check fixtures and hard gates, validates prompt identities and contracts,
and checks that evaluator-owned validators/references have not leaked into any
agent-visible bundle. A single mismatch fails the command.

Prompt wording intentionally differs across vague, normal, and precise variants,
but all three are bound to the same scenario ID and hidden acceptance contract.

## Result semantics

The versioned JSON result records scenario/run identity, validator version and
digest, frozen baseline identity, timestamp, individual outcomes, category
counts, numerical score, hard-gate outcomes, and provenance. Historical v1
scores remain audit data. For v2 evidence containing manual requirements, the
report presents automated passed/failed/manual counts instead of treating
passed-over-total as a correctness percentage. `hard_gate_pass` is separate: a
numerical score never masks a critical failure, and a manual requirement never
masks an observed automated failure. Infrastructure absence is recorded as
`unavailable`; malformed/failed validation infrastructure is `error`; neither
is silently reclassified as an ordinary functional failure.

## Benchmark interpretation

Functional correctness is a gating/filtering dimension, separate from reasoning
tokens, reasoning timing, tool calls, context, wall time, and LLM time. A study
may compare reasoning behavior only among Functional PASS runs, or filter by a
documented functional-score threshold. It must never call fewer reasoning tokens
better when the corresponding implementation is functionally worse, and no
composite efficiency score is produced by M12.

## M13 executor lifecycle

Functional validation is opt-in per prompt/task through a pinned
`functional_scenario` association. It records the scenario ID, exact scenario
and validator SHA-256 values, prompt variant, and (when used) suite identity.
Configuration loading verifies that the prompt bytes and variant belong to that
scenario contract and that the frozen subject lineage matches the experiment's
portable baseline. No scenario is inferred from a filename or prompt wording.

For an associated run the executor performs, in order:

```text
verified frozen baseline → visible baseline-health gate → harness execution
→ sealed result preservation → metrics/context analysis → restored sealed
snapshot → hidden functional acceptance → sealed functional analysis artifact
```

Baseline health is only a pre-run visible regression check. It does not run the
M12 missing-feature discrimination vector. A failed health gate prevents the
harness and any LLM request, records create-only precondition evidence under
`functional-preconditions/`, and is an infrastructure-precondition failure.

Post-run validation never trusts the live worktree. It verifies the sealed
artifact, restores `source/source.tar` into a disposable directory, validates
that restored source, and removes the directory. The validator and references
remain evaluator-owned throughout.

The resulting immutable analysis layer is:

```text
analysis/<run-id>/functional-validation-v1/
  functional-validation.json
  manifest.json
  checksums.sha256
```

The result binds the run/experiment IDs, preserved-artifact and source-snapshot
hashes, source run-manifest hash, scenario/validator/suite/prompt identities,
baseline lineage, complete individual outcomes and category counts, hard gates,
and acceptance score when available. `agent-bench functional verify-result` and
`inspect-result` operate on this sealed artifact.

An acceptance `fail` (for example, a lower score or failed hard gate) is a
valid completed benchmark run: it is never mapped to executor `failed`.
Validator `error` or `unavailable`, restore failure, digest mismatch, and
functional-artifact storage failure are analysis-infrastructure failures. They
do not claim an acceptance score, leave the sealed source intact for a later
retry, and are recorded separately under `functional-analysis-failures/` when
the analysis cannot be sealed. Non-functional historical runs neither create
nor require this analysis artifact; their definitions, matrix/run IDs, seeds,
and digests remain unchanged.

## M14 reporting and planned matrix

Reports and matched comparisons expose functional correctness independently:
status, score numerator/denominator/percent, hard-gate result, baseline
regression count/flag, failed-test count, scenario, and tier. Reporting verifies
`functional-validation-v1` and its source link; it never reads loose validator
JSON. Historical non-functional rows are `not_applicable`, while a configured
validator problem is `error` or `unavailable`; neither is a functional FAIL.

Variant comparison can select functional score percent or retain a functional
filter (status, hard gate, tier, and minimum score). Matching remains strict:
same scenario, exact prompt SHA-256, repetition, and seed. Prompt=All first
forms those matched strata and only then aggregates.

`functional/experiments/taskboard-functional-v1.yaml` is a planned-only suite
of three baseline-homogeneous 27-run definitions (Easy, Medium, Complex), for
81 total rows. Separate definitions are necessary because a run lifecycle has
one immutable frozen baseline; mixing the three baseline lineages would weaken
that guarantee. Inspect it without execution:

```sh
agent-bench functional plan functional/experiments/taskboard-functional-v1.yaml
```

Recommended interpretation is: compare correctness first; compare reasoning,
tool behavior, context, and wall/LLM time among hard-gate PASS rows when that
is the question. No composite efficiency score exists, and a faster or shorter
reasoning trace is never called better when functional quality differs.

## Validator self-validation invariant

A functional scenario is not considered validated until all four conditions
hold:

1. baseline health is known-good;
2. the untouched baseline produces its recorded discrimination vector;
3. a known-good implementation passes the complete acceptance suite with every
   hard gate passing; and
4. targeted known-bad implementations are rejected for their recorded reasons.

The M12 priority self-check makes the last condition concrete: the persistence
mutation fails only `priority-persists`, while the regression mutation fails
`baseline-delete` and therefore the baseline-regressions hard gate. The exact
vectors, scores, and hard-gate state are checked from the scenario definition,
not inferred from prose.

The Medium self-check applies the same invariant to a known-good filtering
reference, OR-semantics mutation, filter-persistence mutation, and delete
regression mutation. Its hard gates independently cover baseline regressions,
combined-filter AND semantics, filter persistence, and active-filter
interactions.

The Complex self-check adds targeted project-isolation leakage, migration ID
loss, import-atomicity corruption, and combined-filter regression fixtures.
Its hard gates cover baseline behavior, project isolation, migration integrity,
failed-import atomicity, and project/task round-trip relationships.

## Scenario coverage

The initial easy scenario checks Taskboard's existing initialization, CRUD,
status handling, persistence, and filtering; then Low/Medium/High creation,
display contract, persistence, editing, legacy records without priority, and
invalid stored priority. The implemented Medium scenario is specified in
`functional/scenarios/combined-filtering-v1.md`. The complex project/migration
scenario is implemented and documented in
`functional/scenarios/multi-project-migration-v1.md` with vague, normal, and
precise prompt variants under `subjects/taskboard-filtering-v1/prompts/`.
