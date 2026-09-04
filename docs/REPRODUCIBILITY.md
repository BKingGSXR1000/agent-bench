# Agent Bench Reproducibility Contract

Status: Milestone M6 fixed-backend and OpenCode implementation contract
Specification version: 1.0.0

## 1. Reproducibility objective

An Agent Bench run is reproducible when a later operator can identify the exact task inputs and software/hardware configuration, reconstruct the isolated starting state, inspect the exact invocation, and verify the preserved result bytes. Reproducibility does not claim that nondeterministic GPU execution or an uncontrollable harness will yield byte-identical model output; it requires that every controllable input and known source of variation be fixed and recorded.

Benchmark v1 treats model, quantization, backend, hardware, and server
configuration as fixed experiment invariants. Sampling defaults are fixed.
Output/reasoning limits deliberately approved as harness-profile treatments
remain separate request-level observations.

## 2. Git baseline identity

Before any run is allocated, a repository reference is resolved to:

- repository origin identity and local repository path used for execution;
- full Git commit object ID;
- commit tree object ID;
- object format/hash algorithm;
- dirty-state prohibition for the baseline source;
- submodule URLs and full commits, when present;
- Git LFS pointer/object status, when present; and
- Git version used to resolve and compare state.

Branch and tag names are informational after resolution. Every run manifest records the full commit/tree identity and verifies its temporary detached worktree before submitting the prompt. The baseline checkout is never the agent workspace.

The complete result is preserved separately, with result-tree/snapshot identity and any necessary Git objects pinned against garbage collection. The implemented M2 procedure is specified in `PRESERVATION.md`.

## 3. Model identity

The fixed model is
`/mnt/starhunter/AI/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q4_K_XL.gguf`, size
17,923,394,624 bytes, SHA256
`bee238bbeb3dc0a34bde4d0dedbaee1f98c009e8bb4226f03070054c12fb1372`.
Before each run, the file is streamed through SHA256 verification; filename or
path alone is insufficient.

The run manifest records:

- execution-host absolute path;
- resolved path after symlink resolution;
- filename and byte size;
- verified SHA256;
- filesystem identity metadata useful for detecting replacement during a run;
- GGUF metadata extraction tool/version and captured metadata; and
- verification time/result.

A hash/size mismatch is `model_hash_mismatch`; missing or invalid input is
`precondition_failed`. Execution never continues with a different file.

## 4. llama.cpp identity

M5 pins `/home/bking/AI/llama.cpp/build/bin/llama-server`, executable SHA256
`92a71ff10ed10f9a24d5af934770f86e1ac6ef0dfccb7d5612f73a2670bb123b`,
source repository `/home/bking/AI/llama.cpp`, commit
`dc72703fc69698b1ea68ece8d2dd8a96e6a4e1fe`, and version/build markers
`0.1.2-dev`, build 10517, commit `dc72703fc`. The fixed backend definition and
each observed run record:

- resolved `llama-server` executable path;
- executable byte size and SHA256 when policy pins the binary;
- exact version/build output bytes;
- llama.cpp source commit when available;
- build configuration, compiler, target architecture, and relevant linked runtime/library versions when available;
- container/image digest if a container supplies the executable; and
- the command/tool used to obtain each identity field.

Missing required identity fields cause a precondition failure. Optional unavailable fields remain explicitly unavailable and cannot be filled from assumptions.

## 5. Exact llama-server invocation representation

M5 fixes the invocation in `environment/backend-v1.yaml` and documents every
argument in `BENCHMARK_ENVIRONMENT.md`. It is represented structurally and
persisted both configured and resolved:

```yaml
schema_version: 1.0.0
executable: /absolute/resolved/path/to/llama-server
argv:
  - /absolute/resolved/path/to/llama-server
  - "<one argument token>"
  - "<its value token>"
run_seed: 1001
working_directory: /absolute/resolved/working/directory
environment:
  allowed_non_secret_name: exact_value
secret_environment:
  secret_name: present_redacted
stdin_policy: closed
stdout_artifact: raw/backend/stdout.log
stderr_artifact: raw/backend/stderr.log
```

`argv` is the authoritative token array passed to process creation; a display-only shell-escaped command is also generated with a named escaping algorithm for human use. Arguments are never stored only as an opaque script or a shell string. Defaults relied upon from llama.cpp are materialized into the resolved configuration when they can affect results, with their source/version.

For repetition `r`, the resolved invocation records `run_seed = 1000 + r` and
passes the same value through the pinned build's explicit `--seed` option. This
server/run seed is not conflated with the independently captured seed field in
an HTTP generation request.

Server-start parameters are separate from per-request generation parameters. The configured template, fully resolved invocation, and observed process identity all receive digests.

## 6. Request and generation parameters

The fixed request configuration represents every applicable value, including:

- temperature;
- `top_p`, `top_k`, and `min_p`;
- seed;
- maximum output tokens when supplied by a harness profile (M5 intentionally
  has no server-wide output cap);
- stop sequences as an ordered array of exact strings/bytes;
- reasoning configuration;
- chat template and tokenizer identity;
- context maximum;
- streaming/non-streaming behavior; and
- any other harness- or backend-specific request property.

For each field, records distinguish:

- configured experiment value;
- harness-requested value;
- value observed at the proxy/backend boundary;
- backend-applied/effective value when exposed;
- source/method; and
- controllability/availability.

Unknown defaults are not invented. A conflict is preserved and flagged. Exact request payload bytes are captured when the safe logging boundary permits it, with secrets redacted before persistence.

## 7. Hardware and environment metadata

The fixed hardware profile declares required identity and precondition thresholds. Preflight captures, where applicable:

- machine/host identity policy, operating system, kernel, architecture, and container/cgroup identity;
- CPU model/count and system memory;
- GPU vendor, model, UUID, driver/runtime, and count;
- VRAM total/free/used, GPU utilization, temperature, power/performance state, and clocks;
- relevant competing GPU processes;
- storage/filesystem information that can materially affect the run;
- locale, timezone, Python/runtime, and relevant library versions; and
- the collector commands/tools and their versions.

Dynamic values are timestamped snapshots, not part of the immutable expected
hardware identity. Production preflight performs fresh live `nvidia-smi` GPU and
compute-process queries and records their commands and UTC observation time;
cached or historical output is not an accepted source for current occupancy.
The fixed RTX identity and non-exempt target-GPU process policy are enforced.
Exact total/used/free VRAM is evidence, but no hard free/used threshold is
applied; the real backend load is the capacity test. Failed required
preconditions yield `precondition_failed` rather than an unmarked run under
different conditions.

## 8. Fresh harness homes and state isolation

Every run creates new directories that did not exist for any earlier run. At minimum these are assigned explicitly:

- `HOME`;
- `XDG_CONFIG_HOME`;
- `XDG_CACHE_HOME`;
- `XDG_DATA_HOME`;
- `XDG_STATE_HOME`;
- harness-native config/state/session/memory directories; and
- controllable temporary/cache directories.

The selected immutable profile is copied into this environment and its copied digest is verified. The source profile is read-only and never modified. No personal home/config fallback is permitted. Harness invocation receives a fresh session identifier and must not resume, import, or discover an earlier session.

The manifest records artifact-relative mappings and, in a restricted host-only record when necessary, resolved absolute paths. Temporary isolation data is removed only after result preservation is sealed and checksums pass.

For OpenCode 1.18.25, the versioned `opencode-default-v1` profile is copied to
`$XDG_CONFIG_HOME/opencode/opencode.json`; all five HOME/XDG locations and
harness state are fresh. The sole permitted runtime binary is the locally
provisioned benchmark payload `toolchains/opencode/1.18.25/bin/opencode`, pinned
by the tracked `toolchains/opencode/1.18.25/identity.json`; the user's personal
OpenCode installation is never consulted at runtime. Its presence, executable
bit, SHA256, size, version, and runtime identity are checked before a task.
The executable is invoked with `--pure`, official JSON output, thinking
visibility, non-interactive auto approval, the fixed model, and the isolated
worktree. Prompt bytes are supplied through stdin because this version's
positional-message path adds literal quote characters around arguments containing
spaces. Native stdout/stderr, the session export, copied XDG state, and the
complete allowlisted environment are preserved. See `OPENCODE.md`.

## 9. Environment-variable allowlisting

Processes are constructed from a small versioned allowlist rather than inheriting the caller's environment. The policy groups names into:

- fixed recorded values;
- run-derived recorded values such as isolated paths;
- required secrets injected but never persisted;
- intentionally omitted names; and
- backend/harness-specific exceptions with justification.

All allowed non-secret names and exact values that may affect behavior are stored in the manifest. The effective child environment is validated against policy before launch. Unexpected variables fail preflight instead of being silently inherited.

## 10. Secret handling and redaction

Secrets are never written to raw logs, manifests, normalized data, metrics, reports, checksums lists containing values, or exported datasets. Redaction occurs before durable capture at known ingress points.

M5 redacts authorization/API-key/cookie header values and recognized structured
secret fields before durable proxy capture while forwarding the original
request. General registered-secret fingerprint scanning remains a later
hardening step and is not claimed by the M5 capture capability.

If safe redaction cannot be guaranteed for a required capture source, the run fails preflight or that capture is disabled and explicitly marked unavailable according to policy. Post-hoc mutation of sealed raw logs is prohibited.

## 11. Backend restart policy

Controlled benchmark-v1 mode starts a fresh llama-server process for every run. It does not reuse a process, KV cache, request queue, or backend session between repetitions. The shutdown method, grace period, escalation signals, and observed exit are recorded.

Model load, startup, and readiness durations are measured separately from task wall time. A server that fails readiness produces a precondition/model-backend failure result without invoking the harness task.

This policy favors isolation. Any future shared-server mode is a different explicitly named protocol and cannot be mixed into benchmark-v1 comparisons.

## 12. Readiness and warmup policy

One versioned readiness policy and one versioned warmup policy are fixed for an experiment.

Readiness records:

- exact endpoint/probe request bytes;
- expected deterministic response conditions;
- polling schedule and deadline;
- start/success/failure timestamps; and
- all probe responses/log references.

Readiness is HTTP 200 with JSON status `ok` from `/health`, with a 900-second
startup deadline. llama.cpp's built-in startup/model warmup is enabled. No
synthetic LLM warmup conversation is sent. Backend creation, model loading,
built-in warmup, readiness, and capture-proxy readiness precede `task_start` and
are excluded from task token/tool metrics.

## 13. Timekeeping

UTC wall timestamps use RFC 3339 with `Z`. Duration and ordering within a process use a monotonic clock with nanosecond integer representation. Each record identifies clock source and capture process.

The manifest records:

- task/backend/preservation phase boundaries;
- host UTC offset/timezone for operator context while canonical timestamps remain UTC;
- clock synchronization source/status when available;
- monotonic-to-UTC anchor pairs for cross-process correlation; and
- detected clock adjustments or uncertainty.

Metrics avoid subtracting independent wall clocks. Cross-process durations require a shared monotonic source or a documented deterministic correlation; otherwise they are unavailable.

## 14. Randomness and ordering

Every controllable random source has a separately named seed. At minimum:

- matrix ordering uses a recorded algorithm/version and seed when shuffled;
- generation uses a recorded requested seed and observed effective seed when available;
- harness-native random behavior uses named seeds when the harness exposes control; and
- any deterministic fixture/test generator records its seed.

For controlled generation, one-based repetition `r` maps to `1000 + r`; the
same repetition receives the same intended seed across harnesses. The proxy
records the request-observed seed separately. An uncontrollable or omitted seed
is marked as such, not assigned a fictional effective value.

## 15. Version pinning

Runs identify and pin, as applicable:

- Agent Bench source commit, package version, and dirty state;
- Python interpreter and dependency lock/resolution artifact;
- harness executable/package and profile bundle;
- adapter, normalizer, metric, classifier, and report algorithm versions;
- Git and system tools used for capture/comparison;
- model/tokenizer/chat-template artifacts;
- llama.cpp source/build/binary;
- OS/container image and GPU driver/runtime; and
- schema and canonicalization versions.

Human-friendly version strings alone are insufficient when a commit, artifact digest, or lock digest is available. A dirty Agent Bench implementation is either prohibited in controlled mode or fully snapshotted and prominently marked according to the experiment policy.

## 16. Run checksums and sealing

Every preserved artifact entry contains path, byte size, SHA256, role, and producer. Paths are artifact-root-relative, normalized, unique, and forbidden from escaping the root.

Sealing order is:

1. close raw streams;
2. preserve source/Git state and other required artifacts;
3. write derived normalized/metric/report data for the specified versions;
4. compute per-file hashes;
5. write a canonical checksum listing;
6. write the artifact manifest referencing that listing;
7. compute the manifest digest according to the non-self-referential canonicalization rule; and
8. read back and verify every required file.

The manifest/checksum representation defines how its own digest field is omitted during hashing to avoid recursion. In the M2 subset, the sorted checksum listing includes `manifest.json` and excludes only itself; the manifest identifies the listing and does not contain its own digest. Later full-run sealing may add a separate non-self-referential sealed-manifest digest. Any verification failure yields `preservation_failed` and prevents temporary-worktree deletion.

Failures before a normal result artifact exists use the exclusive immutable
`runs/<run-id>/failure/` layout defined in `DATA_MODEL.md` and
`BENCHMARK_ENVIRONMENT.md`. Its checksum inventory covers the manifest, event
stream, environment, stdout, and stderr; verification precedes publication.

## 17. Replay record

Preservation includes machine-readable build and launch command records when known. Each uses executable plus argv arrays, working directory, allowlisted/redacted environment, required artifact references, and expected exit/readiness behavior. Display shell commands are convenience renderings only.

A later replay must operate on a copy of preserved immutable artifacts. Replay output and manual review are new linked records and never alter the original run.
