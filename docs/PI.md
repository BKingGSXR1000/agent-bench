# Pi M7 Integration

Status: Milestone M7 completed

Profile revision: `pi-default-v1` `1.0.1`

## Pinned toolchain

Agent Bench uses only the benchmark-managed local npm installation at
`toolchains/pi/0.84.4/`; `node_modules/` is intentionally ignored while
`package.json`, `package-lock.json`, [identity metadata](../toolchains/pi/0.84.4/identity.json),
and the exact [CLI help](../toolchains/pi/0.84.4/cli-help.txt) are tracked. The
local Node executable is ignored as a runtime payload; its [identity metadata](../toolchains/node/26.8.1/identity.json)
is tracked.

- Package: `@earendil-works/pi-coding-agent` `0.84.4`
- Registry integrity: `sha512-jmOlrqUmvhh/siNWFRXjYLJzhKFIHNsAQaysRwzQPQFnPAaV/vhqHsLH/MBsIISA1Rjj7WTUFR3nJrpXoLx39w==`
- Pi entrypoint: `node_modules/@earendil-works/pi-coding-agent/dist/bundle/cli.js`
- Entrypoint SHA256: `5406c369954516fb56879d685e082ff9095cd6e06e41af406f394942377fd4bf`
- Node: `toolchains/node/26.8.1/bin/node`, `v26.8.1`, SHA256
  `19235a9b678f84729464c52623f92de130a165452747c6826d3fdc13df3abcc3`
- npm used for installation: `/home/bking/.local/bin/npm`, `11.19.0`

Pi requires Node 22.19 or newer. The Agent Bench allowlisted PATH deliberately
resolves `/usr/bin/node` 18.19, so the adapter never invokes the Pi launcher
shebang. The benchmark-owned Node executable is a byte-identical local copy of the
runtime used for installation, rather than a path managed by Hermes. Every run first
verifies the pinned absolute Node executable, Pi entrypoint, lockfile, and deterministic
installed `node_modules` tree digest; it then invokes `<pinned-node> <pinned-pi-entrypoint> ...`.
A mismatch fails before the task.

## Installed 0.84.4 behavior

The pinned CLI supports `--print` for one non-interactive task and `--mode json`
for an official JSONL event stream. Its first JSON record is a version-3 session
header, followed by agent/turn/message lifecycle records, streaming message updates,
tool-execution start/end records, and compaction start/end records. The final
`message_end` record contains authoritative assistant content, native usage, and
finish state.

Pi stores its normal configuration under `PI_CODING_AGENT_DIR` (default
`~/.pi/agent`), including `models.json`, `settings.json`, `auth.json`, and
`models-store.json`. Sessions use `PI_CODING_AGENT_SESSION_DIR` (default below
the agent directory) as JSONL files. The clean profile sets both locations into
fresh run-owned XDG directories. Pi's default model reasoning level is `medium`;
automatic compaction is an upstream behavior and its native lifecycle events are
captured when emitted.

The profile uses a custom OpenAI-compatible provider with a literal local placeholder
key, `baseUrl=http://127.0.0.1:18081/v1`, model `qwen3.8-27b`, `max_tokens=16384`
(Pi's documented custom-model default), a 107,520-token context window, and
Pi's documented `qwen-chat-template` compatibility mode. That mode sends
`chat_template_kwargs.enable_thinking=true` and `preserve_thinking=true` for the
upstream Pi default medium-thinking session; it does not impose a reasoning effort
or budget. The profile retains Pi's ordinary extension, skill, prompt-template, theme,
and repository `AGENTS.md`/`CLAUDE.md` discovery behavior. Its fresh run-owned state
prevents personal Pi configuration from being imported. It uses `--offline` only to
disable startup network operations (not model requests to the local proxy), and sets
`PI_TELEMETRY=0` for deterministic, non-personal installation telemetry.

## Final default-profile audit

Revision `1.0.1` removes `--no-extensions`, `--no-skills`,
`--no-prompt-templates`, and `--no-themes`. In Pi 0.84.4 those flags suppress
their respective discovery mechanisms; with fresh run-owned HOME/XDG/Pi roots
they are not needed to prevent personal state and would prevent ordinary
project-local fresh-install behavior. The profile is therefore not a reduced Pi.

`--offline` remains. Pi documents it as disabling startup network operations
(equivalent to `PI_OFFLINE=1`), not as disabling the configured model endpoint.
The profile uses the CLI switch once, rather than duplicating it in the child
environment. It is defensive isolation against catalog/update activity and does
not remove built-in tools, change the system prompt, or alter the local proxy
model request.

The `qwen-chat-template` compatibility declaration and `reasoning: true` are
model/endpoint compatibility metadata. Pi 0.84.4's OpenAI-completions client
generates `chat_template_kwargs.enable_thinking` from its normal session thinking
level and sets `preserve_thinking: true` for this format. Agent Bench supplies no
thinking level, reasoning effort, or reasoning-token budget: Pi's upstream
default medium level remains in effect. These fields select Qwen chat-template
semantics and preserve non-empty historical reasoning over tool turns; they are
not a benchmark-specific reasoning-tuning setting.

The final runtime/profile-semantics verification is preserved at
`/tmp/agent-bench-m7-runtime-audit/output/artifacts/pi-pi-default-v1-m7-readme-single-edit-r001-2b5b449abf838147f513a672`.
It successfully completed the one-line README fixture in 14.086900667 seconds
using the benchmark-owned Node path and an invocation with `--offline` but none
of the four discovery-suppression switches. It made four local-proxy requests,
all with `enable_thinking=true` and `preserve_thinking=true`; the request-history
validator found zero empty historical think blocks. The artifact checksum
verification passed.

## Prompt transport and isolation

The source profile is [pi-default-v1](../environment/harnesses/pi-default-v1/).
Each run receives fresh `HOME`, `XDG_CONFIG_HOME`, `XDG_CACHE_HOME`,
`XDG_DATA_HOME`, `XDG_STATE_HOME`, `PI_CODING_AGENT_DIR`, and
`PI_CODING_AGENT_SESSION_DIR`, with no inherited user environment or Pi state.
All regular files subsequently created in those roots, native stdout/stderr,
session JSONL, exact invocation, and a byte-for-byte prompt transport record are
preserved.

Pi 0.84.4 applies `trim()` to piped stdin, so stdin cannot carry byte-identical
prompts with leading/trailing whitespace. The adapter therefore passes one positional
message after `--`. Inspection with the M7 117-byte fixture confirmed identical
SHA256 `03b18403ef4a275d88d1dbaaa9f92f0935a5c38631afa3bcf3c3fbe1526de67f`
in the source bytes, native Pi session message, and observed OpenAI request message,
including the final newline. No Agent Bench text is added to the user prompt.

## Capture and normalization

The transparent M5 proxy remains authoritative for raw LLM requests/responses,
generation parameters, API usage when exposed, and request-message empty-think
validation. Pi native JSON provides session identity, complete final reasoning text,
tool execution start/result events, and compaction lifecycle events. Native paths are
made worktree-relative only when inside the isolated workspace. Post-Jinja rendered
prompts remain unavailable.

## M7 real integration

The single controlled integration run on 2026-09-04 is
`pi-pi-default-v1-m7-readme-single-edit-r001-7848c852810c2806ca8f672f`.
Its preserved artifact is
`/tmp/agent-bench-m7-real/output/artifacts/pi-pi-default-v1-m7-readme-single-edit-r001-7848c852810c2806ca8f672f`
and its separately sealed M4 metrics artifact is under the matching
`output/analysis/.../metrics-v1` directory. The temporary fixture baseline was
commit `b246399bf314f3ece355ab2c6e61a5f7472cd1f0`; the result changed only
`README.md` from `status: pending` to `status: complete`.

All five streamed proxy requests contained the exact 117-byte task prompt and
the exact session JSONL validation passed. Each request sent `max_tokens=16384`
and `chat_template_kwargs={enable_thinking: true, preserve_thinking: true}`;
none sent a request seed or sampling override, so the fixed llama-server seed
1001 and sampling baseline remained authoritative. The request-history validator
recorded zero empty closed think blocks on all five requests. Historical assistant
reasoning was non-empty where present (0, 1, 2, 3, and 4 entries).

The task completed in 17.958866425 seconds with five LLM requests, 9,386 exact
input tokens, 388 exact output tokens, and a peak context of 2,093/107,520
(1.946614583%). Pi performed one read, two shell commands, and one edit; first
tool and first edit timings were 6.494012486 and 12.775935205 seconds. No native
compaction event or reasoning-token count was exposed. Fresh M5 preflight passed,
the owned backend reached readiness in 4.544144947 seconds, and it was terminated
by the benchmark with exit code zero. No Pi or llama-server process remained.

For an infrastructure-only sanity comparison with the final M6 fixture, the
single Pi run had 17.958866425 seconds wall time, five requests, 9,386 input
tokens, 388 output tokens, four tools, first tool at 6.494012486 seconds, first
edit at 12.775935205 seconds, and peak context 2,093. The M6 OpenCode fixture
had 30.523863974 seconds, four requests, 23,807 input tokens, 510 output tokens,
two tools, first tool at 22.753158 seconds, first edit at 27.393158 seconds,
and peak context 7,898. This one-run observation makes no quality conclusion.
