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

## Scenario coverage

The initial easy scenario checks Taskboard's existing initialization, CRUD,
status handling, persistence, and filtering; then Low/Medium/High creation,
display contract, persistence, editing, legacy records without priority, and
invalid stored priority. Future medium and complex acceptance designs are in
`functional/scenarios/taskboard-v1-future-scenarios.md`; they are not wired as
benchmark scenarios yet.
