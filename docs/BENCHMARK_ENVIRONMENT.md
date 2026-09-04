# Agent Bench v1 Fixed Environment

Status: Milestone M5 fixed environment with M6 OpenCode verification
Environment definition version: 1.0.0

The machine-readable deployment profile is `environment/backend-v1.yaml`. Its
validated structured content and the fully resolved per-run invocation are
persisted separately; the argv token array, not a shell rendering, is
authoritative.

## Fixed model and backend identity

- Model: `/mnt/starhunter/AI/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q4_K_XL.gguf`
- Model identity: Qwen 3.8 27B, `UD-Q4_K_XL`
- Model size: 17,923,394,624 bytes
- Model SHA256: `bee238bbeb3dc0a34bde4d0dedbaee1f98c009e8bb4226f03070054c12fb1372`
- Executable: `/home/bking/AI/llama.cpp/build/bin/llama-server`
- Executable size: 17,920 bytes
- Executable SHA256: `92a71ff10ed10f9a24d5af934770f86e1ac6ef0dfccb7d5612f73a2670bb123b`
- Source repository: `/home/bking/AI/llama.cpp`
- Source commit: `dc72703fc69698b1ea68ece8d2dd8a96e6a4e1fe`
- Version/build: `0.1.2-dev`, build 10517, commit `dc72703fc`

Preflight also verifies that the source working tree is clean, checksums the
resolved local `libllama-server-impl`, `libllama-common`, `libmtmd`, `libllama`,
`libggml`, `libggml-base`, `libggml-cpu`, and `libggml-cuda` binaries listed in
the profile, and verifies through `ldd` that these are exactly the local
llama/ggml libraries selected by the executable. System CUDA, C/C++, OpenSSL,
and libc libraries remain outside the pinned local llama.cpp-library set; M5
does not claim checksums for them.

The full model hash is verified before every real run. Size, hash, executable,
version/build, source commit, source cleanliness, local library, or linkage
mismatch fails preflight; Agent Bench never silently selects another file.

## Fixed chat template

The benchmark template is
`environment/templates/qwen38-agent-bench-v1.jinja`, derived from the exact
`tokenizer.chat_template` metadata embedded in the model above.

- Embedded source template SHA256: `12827f24b742ea4e80cdc12dbcf9622227056b9f797252a3149263d4f9aaadce`
- Patch provenance: `/home/bking/AI/chat-templates/qwen38-flash-next/hermes-minimal-empty-think-fix.diff`
- Resulting template SHA256: `2d59a4438d68dc818c5a75db4edcf4c588e0976b113c5c87def7fc9c1168e955`

The only semantic modification adds `and reasoning_content` to the historical
assistant-reasoning guard, preventing messages with no reasoning content from
adding empty `<think>...</think>` blocks. The repository text file has a terminal
LF that the GGUF metadata string lacks; this non-semantic byte is included in
the resulting hash.

The result retains the embedded Unsloth developer-role, leading
system/developer merge, misplaced-role validation, `high` to `xhigh`
normalization, tool function-name validation, and tool-argument validation.
The server explicitly passes this file after `--jinja`; it never relies on the
embedded template. The bounded-medium template is intentionally not used, and
no 2,000-token limit, earlier-tool-call guidance, turn-splitting guidance,
file-reading strategy, reasoning effort, or reasoning budget is added.

## Direct server configuration

Every controlled run owns a new direct llama-server process. The model router,
port 8080, previous KV state, and previous backend session are never used.
The backend binds `127.0.0.1:18080`; the transparent capture proxy defaults to
`127.0.0.1:18081`. Preflight requires both ports to be free. An occupied
required port fails preflight/bind and its
occupant is never stopped.

The fixed server configuration is:

- context 107,520; one slot (`--parallel 1`);
- batch and ubatch 128;
- RTX 3090 only, `split-mode none`, isolated `main-gpu 0`, all GPU layers;
- fit off, Flash Attention on, q8_0 K/V caches, context shift off;
- speculative/MTP decoding explicitly `none`;
- prompt caching and continuous batching on;
- Jinja on with the explicit benchmark template;
- reasoning on with `reasoning-format deepseek`;
- built-in llama.cpp startup warmup on, with no synthetic LLM warmup;
- metrics and slots endpoints on; and
- Web UI off.

The configured sampling baseline is temperature 1.0, top-k 20, top-p 0.95,
and min-p 0.0. The proxy separately records the parameters actually supplied
by a harness and identifies omissions/overrides. M5 does not pass a server-wide
prediction/output limit, reasoning effort, reasoning budget, or reasoning
preservation override. The pinned build's unrestricted `--predict -1` default
therefore remains effective, while future approved harness-profile output and
reasoning treatments remain observable rather than prohibited.

Generation seed is run policy, not a matrix dimension: one-based repetition
`r` receives seed `1000 + r` (1001, 1002, 1003, ...), identically across
harnesses. The run seed is included explicitly as `--seed` in the resolved
llama-server argv and stored as the invocation's intended server seed. That
server/run value and the proxy-observed HTTP request seed are distinct evidence;
a missing or overridden request seed is never reported as matched. The pinned
build advertises `--seed SEED` in its help output.

## GPU and process preconditions

`CUDA_DEVICE_ORDER=PCI_BUS_ID` and
`CUDA_VISIBLE_DEVICES=GPU-63f9c2ad-4dbc-962b-b314-a652bf28fc0d` create a
one-device CUDA namespace in which `main-gpu 0` is the RTX 3090. The V100 UUID
and any V100-only workload are ignored by the conflict decision.

Preflight requires the target UUID to identify an NVIDIA GeForce RTX 3090 with
at least 24,576 MiB total VRAM. Every `nvidia-smi` compute-process record on the
target UUID blocks the run except an explicitly listed desktop/display process
using at most 512 MiB. The versioned exemption list covers Xorg, Xwayland,
GNOME/KDE desktop components, and VS Code's GPU display process. Unknown,
Python, vLLM, llama.cpp, and other non-desktop processes block regardless of
their reported allocation. Preflight records the exact reported total, used,
and free VRAM but applies no fixed available-VRAM or idle-used-VRAM cutoff. Once
identity and process policy pass, the real backend load determines whether the
remaining memory is sufficient. A process that exits before readiness, including
a model-load failure caused by memory pressure, is preserved as
`backend_start_failed`. Agent Bench records observations and never kills a
conflicting process.

## Isolation, readiness, and timing

The backend receives only `HOME`, all five XDG home variables,
`CUDA_DEVICE_ORDER`, `CUDA_VISIBLE_DEVICES`, `LANG`, `LC_ALL`, and `TZ`, with
fresh absolute HOME/XDG directories. It does not inherit the login environment.

Readiness is an HTTP 200 response containing `{"status":"ok"}` from `/health`.
The fixed startup deadline is 900 seconds. llama.cpp's built-in warmup remains
enabled. Backend creation, model loading, built-in warmup, readiness, and proxy
startup occur before benchmark task timing. No additional warmup conversation
is sent. Shutdown sends termination only to the Agent Bench-owned child, waits
10 seconds, and escalates to kill only that same owned child if necessary.
Exit code and shutdown method are evidence.

## Transparent proxy and capabilities

The stdlib HTTP proxy forwards the original method, path, request body, and
response body. It changes only transport-required routing: hop-by-hop headers
are removed and `Host` identifies the upstream backend; streaming transfer
framing may be re-emitted while SSE payload bytes and order are preserved.
Inbound and upstream request bodies, response bodies, decoded response chunks,
status, safe headers, timings, generation parameters, usage, reasoning content,
finish reason, and tool-call structures are captured. Authorization, API-key,
cookie, and recognized structured secret fields are redacted before durable
capture but the unmodified request is forwarded.

The enabled `/metrics` and `/slots` endpoints can be sampled into an immutable
`BackendEndpointObservation`: status, content type, exact body bytes/hash,
timestamp, and JSON content when valid are retained with
`llama_server_endpoint_exact` provenance. M5 supplies the observation mechanism;
request-boundary polling/correlation belongs to real-run integration.

`CaptureCapabilities` is a versioned immutable record referenced by the run
manifest. M5 declares request/response bodies and request parameters
`proxy_exact`; input, output, context, and reasoning usage counts are
`api_exact` only when compatible llama.cpp usage fields are present. Missing
usage and compaction capture remain `unavailable` rather than estimated.
Tool-call structures are observable in
responses and tool results only when they appear in captured subsequent
requests. That does not claim complete tool execution timing/outcome capture;
those metrics remain unavailable until a harness adapter declares
`harness_exact` tool-call and result capabilities.

If host GPU/process telemetry cannot be collected, preflight records the
observation as `availability: unavailable` and fails closed. It never converts
missing/inaccessible process evidence into an assertion that the RTX 3090 is
free. Successful production preflight obtains new live `nvidia-smi` GPU and
compute-process queries, records their argv and a UTC collection timestamp, and
never accepts a historical observation as current occupancy evidence. An empty
live compute-process list means only that no blocking process was reported; it
is not described as proof of zero VRAM use.

The deterministic empty-think validator counts only closed, whitespace-only
`<think>...</think>` blocks in supplied historical message or rendered-prompt
evidence; an open active-generation prefix cannot match. M5 records the
request-message observation and provides rendered-prompt fixtures, but the proxy
cannot see llama.cpp's post-Jinja rendered prompt. Accordingly, M6's real
OpenCode run validates `empty_think_blocks_in_history = 0` only over captured
request messages. The stronger post-Jinja rendered-prompt assertion remains
unavailable pending a boundary that exposes those bytes. Neither M5 nor M6
fabricates that unavailable evidence.

## Failed-run evidence

A preflight, startup, or readiness failure before a normal sealed result exists
is preserved exclusively at `runs/<run-id>/failure/` with versioned
`manifest.json`, `events.jsonl`, `environment.json`, `stdout.log`, `stderr.log`,
and `checksums.sha256`. Existing destinations are rejected and every regular
evidence file is checksummed. Supported classes include `precondition_failed`,
`backend_start_failed`, `backend_readiness_failed`,
`backend_identity_mismatch`, `model_hash_mismatch`, `template_hash_mismatch`,
`benchmark_port_in_use`, `conflicting_gpu_process`, and
`preservation_failed`. This evidence is not mislabeled as a successful normal
run artifact.

## M5 real integration verification

On 2026-09-04, a dedicated diagnostic used the pinned profile with repetition 1
and server seed 1001. Its fresh preflight at `2026-09-04T06:00:00.135772Z`
recorded the RTX 3090 at 1,631 MiB used and 22,494 MiB free with only exempt
desktop/display compute-process records, so preflight passed despite exceeding
the removed 1,024 MiB cutoff. The owned server reached post-built-in-warmup
readiness in 4.477269125 seconds, occupied PID 1051164 and 20,622 MiB while
loaded, served one non-streaming proxy request with HTTP 200, and stopped via
TERM with exit code 0. The post-shutdown snapshot recorded 1,594 MiB used,
22,530 MiB free, and no llama-server compute process.

The real response exposed 63 prompt tokens, 26 completion tokens, reasoning
content, and finish reason `stop`; it did not expose a reasoning-token count, so
that metric remained unavailable. M4 consumed the normalized proxy exchange as
`api_exact`, producing 89 total tokens and 63/107,520 = 0.05859375% context
utilization without diagnostics. Exact `/metrics` and `/slots` observations both
returned HTTP 200; `/slots` reported seed 1001, the fixed samplers, 107,520
context, and no speculation. The request-history validator reported zero empty
think blocks in the supplied messages. The endpoint and proxy still do not
expose the complete post-Jinja rendered prompt, so the empty-historical-block
guard itself cannot yet be claimed as runtime-observed; the fixture-backed
template validation remains the strongest deterministic evidence until a later
harness/backend boundary exposes rendered history.

This was an M5 component integration diagnostic: it exercised M4's deterministic
token/context calculation on normalized real proxy events, but it was not a
sealed normal benchmark run and did not execute a harness. Full lifecycle
integration remains correctly assigned to M6.

## M6 OpenCode integration verification

On 2026-09-04, the single controlled run `m6-opencode-real-r001` started a fresh
owned backend from this profile, routed OpenCode 1.18.25 through the proxy, made
the requested one-line edit in an isolated worktree, preserved a sealed complete
source result, and emitted a separate sealed M4 metrics artifact. The backend
reached readiness in 4.400533378 seconds and shut down through owned TERM with
exit code zero. All five real LLM requests were streaming; their exact API usage
produced 31,726 input tokens, 426 output tokens, and a peak context of 7,997 out
of 107,520. No reasoning-token count or compaction event was exposed.

All five proxy request-history checks found zero empty historical think blocks.
Requests 3–5 carried one, two, and three non-empty historical assistant
`reasoning_content` values respectively. Post-Jinja rendering remains
unavailable. Exact OpenCode behavior, the observed positional-prompt deviation,
and the corrected stdin policy are recorded in `OPENCODE.md`.
