# Hermes benchmark integration

`hermes-default-v1` runs the benchmark-managed Hermes Agent 0.21.0 entrypoint,
not a binary, configuration, Python runtime, Node runtime, or state directory
from the user's `~/.hermes` installation. The tracked identity document is
`toolchains/hermes/0.21.0/identity.json`; the installed source, virtualenv, and
runtime payloads are intentionally not tracked. Preflight verifies the pinned
entrypoint, source tree, virtual environment, CPython, Node, lockfile, and
version outputs before a run.

Each run receives fresh `HOME`, all XDG roots, and `HERMES_HOME`. The source
profile supplies only the fixed local OpenAI-compatible provider/model and
Hermes's documented `reasoning_echo` compatibility declaration. It otherwise
uses upstream one-shot defaults, including its default toolsets and project
discovery. `TERMINAL_CWD` is explicitly the isolated Git worktree because
Hermes one-shot tools otherwise anchor in `HERMES_HOME` despite the process
working directory.

The argv is element-exact:

```text
HERMES_ENTRYPOINT --model qwen3.8-27b --provider agent-bench \
  --usage-file RUN_OWNED_USAGE_JSON --oneshot EXACT_UTF8_PROMPT
```

(`HERMES_ENTRYPOINT` is illustrative; the sealed invocation records the actual
absolute executable.) The run preserves the prompt transport bytes, invocation,
native usage file, isolated SQLite session data, stdout, stderr, and proxy
traffic. Hermes-native `search_files` maps to the common `search` category and
`patch` maps to `edit`.

Hermes 0.21.0 session records are exported from isolated SQLite after one-shot
completion. Their message timestamps establish that tool calls/results were
recorded during the task, but the Agent Bench capture timestamp is an export
observation time, not a tool execution boundary. Hermes therefore contributes
native tool activity and proxy-observed model tool-call timing, while exact
harness tool-execution timing remains unavailable unless a future Hermes
release exposes it. See `timing-provenance-v1`; it must not be compared with
OpenCode execution-clock timing.

The proxy preserves all HTTP activity. Provider/model discovery GETs remain
diagnostic evidence but are excluded from generic inference metrics; POST
completion exchanges are captured without transformation. The separately
versioned `context-analysis-v2` layer records first-task and task-relative
context evidence without subtracting any harness prompt/tool overhead. See
[CONTEXT_ANALYSIS.md](CONTEXT_ANALYSIS.md).

Before any controlled backend or harness process is constructed, Agent Bench
writes and fsyncs a generic `supervisor/<run-id>/01-initialized.json` record
containing the PID, UTC timestamp, argv, cwd, Python identity, output root, and
sanitized environment identity. Startup failures create a separate fsynced
failure record and stderr capture. `hermes supervisor-dry-run` exercises this
boundary without starting llama.cpp, the proxy, Hermes, or allocating GPU work.
