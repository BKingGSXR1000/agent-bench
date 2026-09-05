# Manual functional acceptance review (M10)

M10 is separate human-authored evidence answering whether a preserved result
actually solves its semantic task. `completed`, `termination_class`, and M9C
metrics remain execution facts and never become quality scores.

## Protocol

`subjects/pocket-ledger-v1/review-protocol-v1.yaml` is the canonical acceptance
protocol. Its digest is stored in every review. Vague, normal, and precise
prompts for the same semantic task receive exactly the same criteria.

Functional outcomes are: `PASS` (all requested behavior works without a
material task defect), `MOSTLY_PASS` (core behavior works with one minor
defect), `PARTIAL` (meaningful implementation but important behavior missing or
broken), `FAIL` (absent, nonfunctional, seriously incorrect, or no meaningful
implementation), and `UNREVIEWABLE` (technical inability to evaluate; never a
silent failure). A `MAJOR_REGRESSION` prohibits overall `PASS`.

Common regression criteria are application load, existing-entry rendering,
adding an entry, baseline balance update, absence of an obvious fatal runtime
error, and preservation of unrelated functionality. Regression outcomes are
`PASS`, `MINOR_REGRESSION`, `MAJOR_REGRESSION`, and `UNREVIEWABLE`.

No automated acceptance judge, image scoring, or LLM judge is implemented in
M10 v1. Priority flags (no changes, zero changed files, tool failures, no test,
long wall time, high tokens) only prioritize human attention; they are not
scores or automatic failures.

## Isolation and blind review

`review prepare` verifies and restores a sealed source snapshot into a new,
empty disposable directory. It writes `review-fixture.html` only in that copy;
opening it seeds the fixed `pocket-ledger.entries.v1` localStorage data and then
opens the result. Reload that fixture before each script. Serve the disposable
copy with a fresh browser profile, never the reviewer’s normal browser profile.
The artifact, result commit, baseline, and M9 evidence are never changed.

Queue order and opaque blind IDs are deterministic hashes of the run ID and
protocol digest. `review next` exposes task/checklist/flags but not harness,
prompt variant, or repetition. Canonical identities remain in the immutable
record for later aggregation.

## Storage and summaries

Records live separately at `<experiment>/manual-review-v1/records/<run-id>/`.
Each amendment creates a new revision; old records are immutable. Aggregates
show counts by harness, task, prompt variant, repetition, and requested joint
groups. `strict_success = PASS`; `practical_success = PASS + MOSTLY_PASS`.
`PARTIAL` is not success and `UNREVIEWABLE` is reported outside the evaluable
denominator.

Only after reviews exist may efficiency be filtered by functional outcome (for
example, wall time among `PASS` results). M9C’s execution report remains fully
verifiable without any review artifact.

## Normal reviewer workflow

Start the local-only dashboard. It binds only to `127.0.0.1`; it creates no
review record until the reviewer presses **Save review and next**.

```bash
python -m agent_bench.cli review serve runs/pocket-ledger-v1-qwen38-v1
```

The command prints the local URL, review-record root, completed-run review
progress, and Ctrl-C shutdown instructions. The pre-save dashboard displays
only an opaque blind ID, semantic task, fixed acceptance script, progress, and
collapsed optional execution flags. It does not expose harness, profile,
prompt variant, repetition, or canonical run ID. Clicking **Open isolated app**
or **RESET TEST STATE** opens an opaque URL in a separate tab. Each reset clears
both browser storage areas, seeds the fixed Salary/Groceries/Train pass fixture,
and then opens the restored application. It never changes sealed evidence.

Every criterion begins unanswered. PASS, FAIL, and UNREVIEWABLE are deliberate
clickable choices; an UNREVIEWABLE choice requires a short reason. The dashboard
does not save an incomplete checklist, and a major regression cannot be saved
with an overall PASS. `Alt+S` saves a complete review and advances to the next
blind item. Re-review is deliberate and remains an immutable CLI amendment.

## Advanced JSON fallback

The dashboard is the normal workflow. These commands remain for auditable
automation or deliberate revision work; the JSON template begins unset and
must be completed manually before recording.

```bash
python -m agent_bench.cli review status runs/pocket-ledger-v1-qwen38-v1
python -m agent_bench.cli review next runs/pocket-ledger-v1-qwen38-v1 > first-review.json
python -m agent_bench.cli review prepare-blind runs/pocket-ledger-v1-qwen38-v1 BLIND_ID /tmp/pocket-ledger-review-BLIND_ID
python -m agent_bench.cli review record runs/pocket-ledger-v1-qwen38-v1 completed-review.json
python -m agent_bench.cli review summary runs/pocket-ledger-v1-qwen38-v1
python -m agent_bench.cli review report runs/pocket-ledger-v1-qwen38-v1
```

Future profile-matrix experiments keep model/backend fixed and add versioned
harness profiles as a matrix dimension. The current default-profile design is
3 × 1 × 5 × 3 × 3 = 135 runs; three profiles per harness would be 405 runs.
The existing experiment planner remains the authoritative pre-run cost view.
