# Context analysis v2

`context-analysis-v2` is a separate immutable deterministic analysis layer. It
is derived from a sealed run's preserved proxy-visible request/response evidence;
it never changes raw events, run artifacts, or `metrics-v1` records.

Only `POST` requests to `/chat/completions`, `/completions`, or `/responses` are
model inference requests. Other captured HTTP requests, including provider/model
discovery GETs, remain as `metadata_discovery` diagnostic evidence and do not
contribute to LLM timing, token, context, or request-count metrics.

For each inference request the layer retains a reference and SHA256 for the
exact redacted proxy-visible body, message-array and tool-schema hashes, roles,
model-request index, captured HTTP index, elapsed time, API-exact usage where
provided, configured maximum context, utilization, and deltas from both the
previous inference request and the first real task request. The complete
redacted request body in the immutable raw event remains the authoritative
record for content, reasoning fields, tool definitions, tool choice, model
parameters, and provider-specific fields.

Purpose is classified only from deterministic evidence: an exact preserved task
prompt (with a separately labelled terminal-line-ending transport normalization)
or an explicit known title-generator marker. Other model calls are
`other_internal`; no speculative planning, summarization, or compaction label is
invented. The analysis records the first task request and auxiliary model calls
before it. Auxiliary calls remain included in total run metrics.

Component token attribution is deliberately unavailable unless a future version
uses the exact pinned tokenizer/template with recorded method and identity. No
character-ratio estimate is allowed. Consequently `non_user_initial_context`
is unavailable unless the user-task component is exactly measurable. API input
context totals remain authoritative and include all harness/system/tool/history
context; they are never reduced by this analysis.

The analysis stores hashes of stable configuration evidence available in the
sealed run state, plus `rendered_prompt_observation=unavailable_at_proxy_boundary`
when no separately rendered pre-template prompt exists. Request-level rows are
sufficient for M9 to produce raw and task-relative curves without reparsing raw
logs.

The command is:

```text
agent-bench context calculate SEALED_ARTIFACT ANALYSIS_OUTPUT_ROOT
```

It writes a new `RUN_ID/context-analysis-v2` directory with checksums and source
artifact hashes. Existing analysis directories are never overwritten.
