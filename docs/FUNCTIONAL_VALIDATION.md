# Automated Functional Validation v1

M12 adds a deterministic, headless functional dimension. It is intentionally
separate from timing, tokens, tool calls, reasoning, and manual review: it does
not generate an efficiency score.

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

## Result semantics

The versioned JSON result records scenario/run identity, validator version and
digest, frozen baseline identity, timestamp, individual outcomes, category
counts, numerical score, hard-gate outcomes, and provenance. Scores are simply
passed tests over all available tests. `hard_gate_pass` is separate: a numerical
score never masks a critical failure. Infrastructure absence is recorded as
`unavailable`; malformed/failed validation infrastructure is `error`; neither
is silently reclassified as an ordinary functional failure.

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
