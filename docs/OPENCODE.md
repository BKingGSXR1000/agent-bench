# OpenCode M6 Integration

Status: Milestone M6 implemented contract  
Current adapter/profile/normalizer version: 1.0.2

## Pinned harness identity

The controlled harness is the benchmark-managed local payload
`toolchains/opencode/1.18.25/bin/opencode`, version `1.18.25`, a self-contained
Bun ELF executable of 184,682,624 bytes with SHA256
`d91e0d33676d0839f7cde87924cd4127ea88c9d6784eea9f009a7d08bdc60eeb`.
Its tracked [identity metadata](../toolchains/opencode/1.18.25/identity.json)
records its source provenance and standard host-library dependencies. The large
binary payload is intentionally ignored, like the Pi and Node toolchain payloads.
The user's `/home/bking/.opencode/bin/opencode` installation was only the
one-time materialization source and is not part of the benchmark environment.
Every run checks the benchmark-owned file's existence, executable bit, size,
SHA256, version, and runtime identity in a fresh temporary HOME/XDG environment
before task execution. There is no PATH or personal-installation fallback.

## Controlled default profile

`environment/harnesses/opencode-default-v1/` is the immutable source profile.
Its `opencode.json` SHA256 is
`ef4c1a16e4d50a1f747c14728740dc23bb578b58fec6bd32aa06e2af68a533a4`;
the current parsed profile definition digest is
`80ff2db761a4a9b99241ad476271787c3dab9d58febbc966bd28f76a41c9731d`.

Relative to a fresh upstream configuration, the profile only:

- registers one OpenAI-compatible provider at the fixed Agent Bench proxy;
- selects `agent-bench/qwen3.8-27b` for primary and small-model work;
- disables automatic updates and external session sharing;
- declares no external plugins and invokes `--pure`; and
- uses `--auto` for unavoidable non-interactive permission handling; and
- uses `--thinking` solely so completed reasoning parts appear in native JSON
  capture.

It does not configure instructions, agents, reasoning effort/budget, MCP,
personal plugins, memories, skills, sessions, or project state. OpenCode's
built-in agent and tool descriptions remain upstream behavior. Repository-local
instruction/config discovery is not disabled, because the repository presented
to every compared harness is part of the benchmark subject.

The provider declares reasoning and tool-call support but does not impose a
request temperature or output limit. OpenCode 1.18.25 was observed to issue a
separate small-model title request at temperature 0.5 and to set
`max_tokens = 32000` on all requests. These native choices are captured rather
than normalized away.

## Isolation and exact invocation

Every run creates fresh `HOME`, `XDG_CONFIG_HOME`, `XDG_CACHE_HOME`,
`XDG_DATA_HOME`, `XDG_STATE_HOME`, and harness-state directories. The profile is
copied to `$XDG_CONFIG_HOME/opencode/opencode.json`. OpenCode stores configuration
and installed provider support below XDG config, model cache below XDG cache,
its database/log/repository data below XDG data, and lock/state data below XDG
state. All regular files produced in these roots, plus native stdout, stderr,
session export, invocation, and result records, are preserved in the run artifact.

The complete allowlisted child environment contains only the isolated HOME/XDG
paths, fixed `PATH`, `LANG`, `LC_ALL`, `TZ`, `TERM`, `CI`, `NO_COLOR`,
`OPENCODE_CONFIG`, `OPENCODE_DISABLE_AUTOUPDATE`, and
`AGENT_BENCH_HARNESS_STATE`. It contains no inherited API keys or personal
configuration paths.

The authoritative argv is:

```text
/home/bking/AI/agent-bench/toolchains/opencode/1.18.25/bin/opencode --pure run --format json --thinking --auto --model agent-bench/qwen3.8-27b --dir <isolated-worktree>
```

The exact UTF-8 prompt bytes are written to stdin and followed by EOF. No
`--session` or `--continue` is supplied, so OpenCode creates a fresh session.
The exact byte sequence written is also preserved as
`raw/opencode/stdin-prompt.bin`, independently checksummed with the artifact.
This stdin policy is necessary because inspection of the pinned executable's
bundled run-command source showed that it adds literal quotes around positional
message arguments containing spaces, while piped stdin is appended without that
transformation.

For the pinned OpenCode 1.18.25 executable, `--thinking` is not the normal
`run` default: help describes it as "show thinking blocks", and the bundled
run-command code defaults its local display gate to false. That gate is checked
only when emitting/displaying completed `reasoning` parts. It is not included in
the object passed to `session.prompt`, whose model-affecting fields are the
selected agent/model/variant and message parts. Consequently, the flag neither
enables nor disables model reasoning and does not alter request parameters. It
is retained as an observability-only deviation because JSON reasoning records
would otherwise be omitted from the harness stdout evidence.

## Native capture and normalization

Official `--format json` stdout is retained byte-for-byte. Parsed native records
are also wrapped in immutable raw events. OpenCode `reasoning` parts normalize to
reasoning events. Completed/error `tool_use` parts normalize to paired tool-call
start/end events and a conservative read/search/edit/write/shell/test operation.
Native start/end epoch milliseconds are correlated to task-start UTC and labeled
`harness_wall_clock`; runner and proxy timings retain `runner_monotonic`.

Absolute native file paths are converted to worktree-relative POSIX paths only
when lexically inside the recorded worktree. External absolute paths remain
absolute and therefore cannot qualify as worktree edits. The shell test
classifier recognizes only the versioned explicit command-prefix set. OpenCode
does not expose complete compaction events, so compaction remains unavailable.

`CaptureCapabilities` for this boundary are:

| Observation | Method |
|---|---|
| raw request/response payload | `proxy_exact` |
| request generation parameters | `proxy_exact` |
| input/output/context token counts | `api_exact` when exposed |
| reasoning content and finish reason | `proxy_exact` |
| reasoning-token count | `api_exact` when exposed |
| tool calls/results and session identity | `harness_exact` |
| compaction and post-Jinja serialized prompt | `unavailable` |
| empty historical think detection | `proxy_exact` for request messages only |

## Benchmark-managed executable verification

After the profile was moved to the benchmark-owned executable, one controlled
M6 verification run completed as
`opencode-opencode-default-v1-m6-readme-single-edit-r001-8a935b046dd08b1ac44b7cc4`.
Its sealed artifact is
`/tmp/agent-bench-m6-toolchain-audit/output/artifacts/opencode-opencode-default-v1-m6-readme-single-edit-r001-8a935b046dd08b1ac44b7cc4`
and its metrics artifact is under the matching `output/analysis/.../metrics-v1`
directory. The benchmark-owned executable was recorded as the invoked argv[0];
the personal OpenCode path does not occur in the preserved evidence.

The 117-byte source prompt, captured stdin stream, session export, and every
proxy request task message had SHA256
`03b18403ef4a275d88d1dbaaa9f92f0935a5c38631afa3bcf3c3fbe1526de67f`.
The proxy captured five streamed Qwen requests, the intended sole README edit,
and a successful termination. The sealed metrics record 32.808974701 seconds
wall time, 31,716 input tokens, 564 output tokens, and a 7,998-token peak
context. Both artifact checksum inventories verified; runtime/worktree cleanup
left only their empty parent directories and no OpenCode or llama-server process.

## Historical 1.0.1 real integration observation

The final 1.0.1 adapter/profile/normalizer was exercised exactly once as
`m6-opencode-real-v101-r001` on 2026-09-04. Its immutable run artifact is
`/tmp/agent-bench-m6-v101-real-lebhwmr3/output/artifacts/m6-opencode-real-v101-r001`
and its separately sealed metrics artifact is
`/tmp/agent-bench-m6-v101-real-lebhwmr3/output/analysis/m6-opencode-real-v101-r001/metrics-v1`.
The fixture baseline was
`af7c59f8bef5a8b3066685c30440c64f4ab9364d`; the preserved result commit is
`ccf75741037306b62dda9c250c449a3010eb6b0b`, pinned by
`refs/agent-bench/results/m6-opencode-real-v101-r001`.

The source prompt, preserved stdin byte stream, exported OpenCode user message,
and proxy-observed task message were all exactly 117 bytes with SHA256
`03b18403ef4a275d88d1dbaaa9f92f0935a5c38631afa3bcf3c3fbe1526de67f`.
The exact task text, including its single final newline, remained present in all
four proxy requests. The auxiliary title request added its own separate 40-byte
instruction message but did not alter the task message. OpenCode's session
export validation is `exact_match`.

The run produced 12 native events: three step starts, three reasoning parts, two
completed tool uses (`read` and `edit`), three step finishes, and one final text
part. Both normalized tool paths are the canonical project-relative
`README.md`. The resulting tree changes only that file from `status: pending`
to `status: complete`. M4 records one edit, one changed file, one added line,
one deleted line, first tool use at 22.753158 seconds, and the first edit at the
now-available numeric value 27.393158 seconds.

The proxy captured four streaming requests. All used model `qwen3.8-27b` and
`max_tokens = 32000`; none sent `top_p`, `top_k`, `min_p`, a request seed,
reasoning effort/budget, or `max_completion_tokens`. Request 1 was OpenCode's
title request, used temperature 0.5 and no tools. Requests 2–4 omitted
temperature, used `tool_choice = auto`, and supplied ten built-in tool schemas.
Thus the title request overrode server temperature, while all omitted sampling
values used server defaults and server seed 1001 remained authoritative.

| Request | Purpose/action | Elapsed s | Input | Output | Context utilization | Finish |
|---:|---|---:|---:|---:|---:|---|
| 1 | native title generation | 1.843350548 | 613 | 221 | 0.570126488% | `stop` |
| 2 | task, then `read` | 2.220735356 | 7,541 | 115 | 7.013578869% | `tool_calls` |
| 3 | task history, then `edit` | 22.875016672 | 7,755 | 124 | 7.212611607% | `tool_calls` |
| 4 | final task response | 27.511695097 | 7,898 | 50 | 7.345610119% | `stop` |

Reasoning content was present in every response, while reasoning-token counts
were not exposed. Totals were 23,807 input, 510 output, and 24,317 tokens; peak
context was 7,898 of 107,520 tokens (7.345610119%). Task wall time was
30.523863974 seconds and deterministic summed LLM time was 35.411367542 seconds
because the title and task requests overlapped.

Every request-message history had zero whitespace-only closed think blocks.
Historical assistant messages in requests 3 and 4 contained one and two
non-empty `reasoning_content` values respectively. Final post-Jinja rendered
prompt visibility remains unavailable.

Fresh preflight passed with the pinned model, template, executable, local
libraries, source commit, clean source tree, fixed ports, and RTX 3090 identity.
It observed no conflicting compute process on the RTX 3090. Backend readiness
took 4.484930495 seconds. The proxy shut down without error and the owned backend
exited zero after TERM. Listener-release checks passed for both ports; immediate
backend rebindability was false because of transient TCP state, while a later
read-only check confirmed neither port had a listener. No benchmark process or
GPU allocation remained. The temporary worktree and harness runtime were absent
after successful preservation. Both artifacts and both checksum inventories
verified.

## Historical 1.0.0 integration evidence

The earlier immutable `m6-opencode-real-r001` run remains preserved under
`/tmp/agent-bench-m6-real-9syym5uh/output/`. It demonstrated successful model,
proxy, harness, Git, and preservation integration, but predates the 1.0.1 fixes:
OpenCode added literal quotes around its positional prompt and the normalizer
retained absolute worktree paths. That sealed historical artifact was not
rewritten and is not the final M6 acceptance run.

## M6 limitations

- Post-Jinja rendered prompt bytes and compaction events are unavailable.
- Reasoning-token counts are unavailable unless the backend exposes them.
- OpenCode's automatic title request is included in LLM/token/context metrics.
- Proxy streaming tool-call evidence is retained as exact deltas; complete tool
  execution identity, timing, and outcome come from OpenCode native events.
- M6 provides one-run CLI execution, not the experiment scheduler assigned to a
  later milestone.
