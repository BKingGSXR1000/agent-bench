# Benchmark Subject v1 — Pocket Ledger

## Decision

Candidates considered: a Tetris-like game (visible but input/rendering complexity
would dominate), a task tracker (good but familiar CRUD can collapse into a
single-file edit), a note organizer (requires persistence/search design), and
Pocket Ledger. Pocket Ledger won because its small dependency-free browser UI is
runnable offline yet has state, input validation, persistence, totals, and list
rendering spread across realistic HTML, CSS, and JavaScript.

The subject is deliberately a compact visible web application, not the prior
README fixture. It has no build step, package manager, lockfile, network,
account, or API key requirement. Python 3.12+ provides the static server and
baseline check; any modern browser can manually review it.

## Frozen baseline

- Subject/version: `pocket-ledger-v1` / `1.0.0`
- Repository: `subjects/pocket-ledger-v1/baseline-repo` (independent Git repo)
- Commit: `7326e6a06de7f693db5ac70a16363e47b620d4fb`
- Tree: `663ea314b9d5f1b378abad28693085d0609386bc`
- Complete canonical Git history: `baseline-v1.bundle`, SHA256
  `1c2c0cd14ec402f29817d7b0c2ee530380ab85dd19e765fdd94338fe6720086d`
- Dependency lock: none (no dependencies)
- Inventory: 5 tracked files, 4,755 bytes, 52 lines; HTML/CSS/JavaScript plus a
  Python static smoke test.

The baseline source is ordinary outer-repository content, not a submodule or
gitlink. The tracked bundle reconstructs its canonical commit in a fresh clone;
the checked-in source tree is compared against that commit by test. Baseline
commands are recorded in `subjects/pocket-ledger-v1/subject.yaml`:
`python3 tests/test_baseline.py` for test/smoke and
`python3 -m http.server 8000` to run. The baseline commit is clean and immutable;
a defect requires a new subject version, never a mutation of this one.

No project-local `AGENTS.md` is included: a tiny browser app has no realistic
special workflow beyond its README. This avoids artificial instructions while
still allowing normal harness discovery of ordinary project files.

## Tasks and isolation

The independent semantic tasks are `entry-delete`, `entry-filter`,
`entry-category`, `monthly-summary`, and `keyboard-entry`. Each starts from the
same frozen baseline. `tasks/tasks.yaml` contains evaluator-only acceptance and
manual dimensions; prompts live under `prompts/` and are the only task material
delivered to a harness. Neither task IDs, acceptance text, nor evaluator rubrics
are inside `baseline-repo`.

Every task has byte-exact vague, normal, and precise variants. Their shared task
ID and SHA256 are validated by the M9A tests. Precise variants add acceptance
clarity, never a larger feature or implementation-location hint.

Future representative matrix: 3 harnesses × 5 tasks × 3 variants × 3
repetitions = **135 runs**. All cells will resolve the same baseline commit and
fixed M5 environment; only harness/profile, prompt and repetition vary. M9B
will execute this, and M10 will use the declared component review dimensions.
