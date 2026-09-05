# Hermes reasoning screen v1

Status: prepared only. This document and its experiment definition do not
start a model, harness, backend, or proxy.

## Scope

`pocket-ledger-v1-hermes-reasoning-screen-v1` is a Hermes-only, first-pass
screen of the one llama.cpp request field that changes Qwen 3.8 reasoning:
`reasoning_effort`. It contains 45 future rows: three candidate profiles × the
15 byte-pinned Pocket Ledger prompt variants × repetition R001. It uses the
same portable frozen baseline bundle, fixed Qwen 3.8 27B Q4 GGUF, fixed
template, llama.cpp build, RTX 3090 profile, server settings, prompt bytes,
seed 1001, and 900-second task limit as the completed Pocket Ledger v1 work.

The existing completed `hermes-default-v1` R001 results are a read-only
control. They are intentionally not placed in the execution matrix and must
not be rerun merely for this comparison.

## Exact reasoning mapping

The fixed server has `--reasoning on`; this enables Qwen reasoning parsing and
visibility. It is not a per-profile reasoning budget or a request-side effort
setting. The candidate profile's copied `config.yaml` adds Hermes's custom
provider `extra_body`, which Hermes merges into each OpenAI-compatible request
before sending it through the Agent Bench proxy. The sealed request body is the
authoritative verification point.

| Profile | Exact request field | Fixed template result |
| --- | --- | --- |
| completed control `hermes-default-v1` | absent | thinking enabled; template defaults the omitted value to `xhigh` |
| `hermes-reasoning-off-v1` | `reasoning_effort: "none"` | llama.cpp maps it to `enable_thinking=false` |
| `hermes-reasoning-low-v1` | `reasoning_effort: "low"` | thinking enabled; embedded template adds its low-effort instruction |
| `hermes-reasoning-medium-v1` | `reasoning_effort: "medium"` | thinking enabled; no additional effort instruction |

An explicit `high` request is intentionally not a candidate. llama.cpp passes
non-`none` effort through to the template; the template then normalizes `high`
to `xhigh`. With `--reasoning on`, an omitted field has thinking enabled and
the same template defaults the value to `xhigh`. Thus the requests differ only
at the HTTP field level, while the rendered prompt/model reasoning semantics
are identical. Existing `hermes-default-v1` R001 remains the control labelled
**default / effective xhigh**; its immutable evidence is not rewritten.

All profiles retain Hermes's existing `reasoning_echo: true` provider
compatibility declaration. It is history replay/preservation behavior, not an
independent reasoning-strength control, and it is not varied in this screen.
No profile varies tools, one-shot mode, project discovery, endpoint, model,
template, server argv, sampling configuration, model output cap, seed, or
timeout.

The fixed template is
`environment/templates/qwen38-agent-bench-v1.jinja`, SHA256
`2d59a4438d68dc818c5a75db4edcf4c588e0976b113c5c87def7fc9c1168e955`.
It is the embedded Unsloth template with only M5's empty-historical-thinking
guard. No bounded-medium or agent-behavior guidance template is used.

## Request and response evidence

For every actual inference POST, the proxy records the normalized request
parameters, configured context, body hash/reference, and response usage. The
screen verifies that `reasoning_effort` is exactly the profile's declared value
on every candidate request, while all other observed request parameters are
reported rather than assumed. Metadata discovery GETs remain diagnostic and do
not count as inference.

The current proxy capture also records whether an observed final assistant
response has visible answer content, reasoning content, tool calls, and an
explicit finish reason. This makes the following deterministic metrics
available only where their exact response fields are exposed:

- `reasoning_only_responses`: a final response with nonempty
  `reasoning_content`, no visible answer, and no tool call;
- `length_finished_responses` and the no-tool-call subset: explicit
  `finish_reason` of `length`, `max_tokens`, or `max_output_tokens`; and
- request count/output tokens before the first model-emitted tool call and
  model-emitted edit call.

These are response/model-call observations, not harness tool-execution timing.
`context-analysis-v2` continues to supply request-by-request input context,
context utilization, first task request, and peak context. Empty historical
think-block validation remains the fixed-template validation; the expected
count is zero, and no unavailable post-Jinja inspection is claimed as visible.

## Result comparison

After completed candidate runs exist, build their normal deterministic M9C
report grouped by `harness_profile`, `harness_profile × semantic_task`, and
`harness_profile × prompt_variant`. Its individual lines and normalized-time
curves are retained; medians and Type-7 Q1–Q3 appear only for like-for-like
series with more than one available value.

For the cross-experiment default-control comparison, run the read-only command
below. It validates every selected sealed run, metrics artifact, and
context-analysis artifact and prints individual R001 rows plus profile-level
Type-7 summaries. It neither writes into a run nor changes a metrics artifact.

```text
python -m agent_bench.cli report reasoning-screen \
  --control-root runs/pocket-ledger-v1-qwen38-v1 \
  --screen-root runs/pocket-ledger-v1-hermes-reasoning-screen-v1
```

The comparison is restricted to deterministic behavior/resource observations:
wall time, request and token counts, context, tool calls, response behavior,
and tool-call/edit-call position. It makes no task-quality claim.

Based on the existing Hermes task-duration distribution (45 completed default
runs: 144.973–626.424 seconds, median 266.761 seconds), the **total serial
task time** for the 45 new rows is estimated at 6,524–28,189 seconds
(about 1.81–7.83 hours), with a median-based estimate of 12,004 seconds
(about 3.33 hours). This excludes per-run backend startup, readiness,
preflight, and preservation overhead.

## Future budget/output/step sweep — design only

Do not combine a future reasoning budget/output/step sweep with this effort
screen. The pinned `llama-server` supports `--reasoning-budget N`, whose
current default is `-1` (unrestricted); benchmark-v1 does not set that option.
Some non-benchmark local startup scripts use `--reasoning-budget 1024`, but
they use a different multi-GPU/model setup and are not benchmark evidence. No
local source confirming a `768` budget was found during this preparation.

Before a later sweep, define and validate independently:

1. whether the budget is server-wide or can be set at the request boundary;
2. its exact unit and termination behavior for this Qwen/template/backend;
3. its relationship to Hermes's observed `max_tokens` (the completed default
   Hermes traffic showed `65536`, while the experiment's retained generation
   record says `16384`); and
4. whether a model output/step cap is a backend argv value, a request value, or
   both.

That later experiment must vary exactly one such control at a time, pin the
new profile/backend identity, capture the actual wire values, and include a
separate stable default control. It is intentionally not created or executed
here.
