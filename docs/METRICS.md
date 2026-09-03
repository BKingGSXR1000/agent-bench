# Agent Bench Deterministic Metrics

Status: Milestone M4 implemented metric specification
Specification version: 1.0.0

The concrete persisted schema, provenance, storage, and current capture boundary
are documented in `METRICS_ENGINE.md`.

M4 persists the concrete names requested by the implementation milestone:
aggregate token fields use the `_total` suffix, LLM/tool behavior counts use
`*_count` or `tool_calls_*`, and compaction fields use
`number_of_compactions`, `context_at_first_compaction_tokens`, and
`context_utilization_at_first_compaction_percent`. The shorter names in the
tables below remain the metric-definition names. Timing is calculated from
monotonic nanoseconds and persisted in the explicitly named `*_seconds` fields
with `units = seconds`.

## 1. Metric contract

Metrics are deterministic functions of sealed raw/normalized records and preserved Git/result evidence. They never depend on an LLM or subjective judgment. Each persisted metric records its definition version, typed value, units, source record references/digests, calculation method, and availability.

The following rules apply to every metric:

- `0` means the observed, supported count or duration is exactly zero.
- `unavailable` means prerequisites are absent, invalid, ambiguous, or not exposed; it is never converted to zero.
- A metric may use exact backend/API data or a deterministic reconstruction from exact source bytes. The method is recorded and the two source classes are not silently mixed.
- Heuristic token estimation is forbidden.
- Events are ordered primarily by monotonic elapsed time, then normalized sequence. UTC is for correlation and display, not duration arithmetic when monotonic time exists.
- Negative durations or decreasing timestamps caused by invalid source data make the affected metric unavailable and emit a validation error.
- Counts include only events that the normalizer can derive reliably. If capture coverage is known to be incomplete for a category, that category's count is unavailable rather than a lower bound presented as complete.
- A run can have partial metrics after failure if their source data remains valid.

## 2. Timing boundaries and interval rules

`task_start` is the monotonic boundary immediately before submitting the byte-exact prompt to the fresh harness session, after backend readiness and any configured warmup. `task_end` is the earliest authoritative terminal boundary for the task: clean harness completion, timeout trigger, unrecoverable crash/error, context overflow, or forced process termination when no earlier boundary exists.

Backend startup, model load, readiness, and warmup are separate phases and are excluded from task metrics. They may have their own future phase metrics.

For aggregate active time, every valid completed interval is summed. Overlapping LLM intervals or overlapping tool intervals are each counted because the metric represents cumulative component time, not exclusive wall-clock occupancy. A report must not imply that component sums add to wall time. Unpaired/incomplete intervals are excluded only when a valid duration cannot be recovered; the aggregate is unavailable if capture completeness cannot be established, otherwise the count of excluded intervals is reported beside it.

## 3. Timing metrics

| Metric | Exact definition | Source data | Units | Unavailable and edge cases |
|---|---|---|---|---|
| `wall_time` | `task_end.monotonic_ns - task_start.monotonic_ns` | Run start/end normalized events or manifest timing boundaries | nanoseconds; reports may display seconds | Unavailable if either boundary or a comparable monotonic clock is absent. Zero is valid only when equal verified boundaries exist. Failure runs use their terminal boundary and remain measurable. |
| `llm_time` | Sum of duration for all completed correlated LLM request/response operations in the task interval | `LLMRequestEvent`, `LLMResponseEvent`, or exact proxy/backend timings | nanoseconds | Unavailable if LLM timing coverage is incomplete or clocks cannot be correlated. Cancelled/error responses count through their observed completion. Overlaps are summed. |
| `tool_execution_time` | Sum of durations of all completed correlated tool calls, across every category | `ToolCallEvent` start/end data | nanoseconds | Unavailable when tool timing capture is incomplete. Failed/cancelled calls count through observed end. Overlaps are summed. |
| `shell_execution_time` | Sum of completed tool-call durations whose normalized category is `shell` or `test` executed through a shell | `ToolCallEvent` | nanoseconds | Same interval rules as tool time. A test-shell call contributes once, not twice. |
| `time_to_first_llm_request` | First LLM request start elapsed time minus `task_start` | `LLMRequestEvent`, task boundary | nanoseconds | Unavailable if no request is observed or capture cannot prove request coverage. Never represented as wall time. |
| `time_to_first_tool_call` | Earliest tool-call start elapsed time minus `task_start` | `ToolCallEvent`, task boundary | nanoseconds | Unavailable with reason `event_not_observed` when complete capture shows no call; unavailable with `capture_failed` when coverage is incomplete. |
| `time_to_first_edit` | Earliest start time of an `edit` or `write` call that targets the worktree and is deterministically classified as a mutation, minus `task_start` | `ToolCallEvent` and normalized target path | nanoseconds | A failed mutation attempt still qualifies as the first edit attempt and is marked as such; reports may separately show first successful edit. Calls outside the worktree do not qualify. Unavailable if absent/ambiguous. |
| `time_to_first_test_command` | Earliest start of a normalized `test` event, minus `task_start` | Test-execution event or `ToolCallEvent` classified by the versioned command classifier | nanoseconds | Test-looking prose does not qualify. If the command cannot be classified deterministically, unavailable/absent according to coverage. Failed tests still qualify. |

## 4. Token and context source rules

Token values have one of these methods:

1. `backend_exact`: emitted by the fixed backend at the request boundary;
2. `api_exact`: emitted by the protocol/provider and proven to have the required semantics; or
3. `tokenizer_reconstructed`: counted from the exact preserved payload with the exact compatible tokenizer/template implementation and recorded identity/digest.

The priority is backend exact, API exact, then compatible deterministic reconstruction. If two exact sources disagree, both observations are preserved and the canonical metric is unavailable pending a versioned resolution rule; one is not chosen silently.

`context_tokens` means the complete tokenized input actually presented to the model for one request, including system/developer/user content, tool schemas, prior messages, and protocol/template tokens. It excludes tokens generated by that response. Counts of user prompt text alone are not context counts.

`output_tokens` means the complete generated token sequence for the response. `reasoning_tokens`, when exposed, is recorded separately along with whether it is a subset of `output_tokens`. Therefore reasoning tokens are never automatically added to output tokens.

## 5. Token and context metrics

| Metric | Exact definition | Source data | Units | Unavailable and edge cases |
|---|---|---|---|---|
| `input_tokens` | Sum of canonical `context_tokens` over all LLM requests | `LLMRequestEvent` | tokens | Repeated context is intentionally counted again because it was processed again. Unavailable if any request count is unavailable or request coverage is incomplete. |
| `output_tokens` | Sum of canonical complete generated output tokens over all responses | `LLMResponseEvent` | tokens | Includes truncated/error response output if exactly counted. Unavailable if any response count or response coverage is incomplete. |
| `reasoning_tokens` | Sum of separately exposed reasoning-token counts | `LLMResponseEvent` | tokens | Unavailable unless every relevant response exposes a semantically compatible exact/reconstructed reasoning count. Zero is valid only when complete exposure reports zero. |
| `visible_response_tokens` | Sum of visible/non-reasoning response tokens when separable | `LLMResponseEvent` | tokens | Unavailable when the protocol cannot separate visible output. It is not reconstructed by subtracting unless the accounting relationship is explicit and exact. |
| `total_tokens` | `input_tokens + output_tokens`; reasoning is not added because it is a subset or component of generated output under the normalized contract | Request/response token metrics | tokens | Unavailable if either operand is unavailable or output semantics do not meet the normalized contract. |
| `tokens_per_llm_request` | Ordered vector of request `context_tokens` and associated response `output_tokens`; aggregate mean is `total_tokens / llm_requests` when requested | LLM events | tokens/request | Vector elements retain individual availability. Aggregate unavailable for zero requests or incomplete totals. |
| `context_used_per_request` | Ordered records `(request_index, elapsed_ns, context_tokens, source_method)` | `LLMRequestEvent` | tokens | Each missing observation remains unavailable; available requests are not discarded. Duplicate request indices are a validation error. |
| `context_utilization_percent_per_request` | `100 * context_tokens / configured_max_context_tokens` for each request | `LLMRequestEvent`, fixed environment | percent | Unavailable if either value is absent or max is non-positive. Values over 100% are retained as evidence and flagged, not clamped. |
| `peak_context_tokens` | Maximum available canonical `context_tokens` across requests | `LLMRequestEvent` | tokens | Unavailable if no request has a valid count. If some request counts are missing, value is marked partial and is not a valid complete-run peak; canonical metric is unavailable while an explicitly named observed lower bound may be reported. |
| `peak_context_utilization_percent` | Maximum valid per-request utilization | Derived per-request values | percent | Same completeness rule as peak context. |
| `context_growth_per_request` | For request `i > 1`, `context_tokens[i] - context_tokens[i-1]` in request-index order | `LLMRequestEvent` | tokens | First request has `not_applicable`. Missing either adjacent count makes that element unavailable. Negative values are valid and often indicate compaction. |
| `net_context_growth` | Last request context minus first request context | `LLMRequestEvent` | tokens | Requires at least two completely counted endpoint requests; one request yields `not_applicable`. |
| `tokens_before_first_edit` | Sum of `context_tokens + output_tokens` for LLM operations whose response completion is at or before the first qualifying edit-call start | LLM events and first-edit event | tokens | Counts billed/processed tokens, including repeated input context. Unavailable if no first edit, relevant tokens are missing, or event correlation/order is ambiguous. The response that issued the edit is included because it completed before the call began. |
| `reasoning_tokens_before_first_edit` | Sum of exposed reasoning tokens for the same completed responses included above | LLM response events and first-edit event | tokens | Unavailable unless relevant reasoning coverage is complete and a first edit exists. |
| `first_compaction_context_tokens` | Pre-compaction context count explicitly emitted for the earliest compaction-start event; alternatively the context of a directly linked immediately preceding request when the normalizer records that deterministic linkage | Compaction event plus linked request/raw source | tokens | No temporal guess is permitted. Unavailable if no compaction occurred, the pre-value is absent, or linkage is ambiguous. |
| `compaction_count` | Number of distinct normalized compaction-start events with unique correlation IDs | Normalized compaction events | compactions | Zero only when capture supports compaction and none occurred; otherwise unavailable. |

Token values from different methods may appear in per-request records, but an aggregate records the set of methods used. Cross-run tables must make method differences visible and may exclude incompatible methods from a comparison.

## 6. Agent-behavior metrics

### 6.1 Tool category rules

Each invocation has exactly one primary category for totals:

- `read`: obtains contents or metadata of identified files without broad matching;
- `search`: enumerates paths or matches content/patterns;
- `edit`: changes existing file content through a patch/editor operation;
- `write`: creates/replaces file content through a write operation;
- `test`: executes a command deterministically recognized as a test by a versioned classifier or emitted test event;
- `shell`: executes a shell/process command not primarily classified as `test`; or
- `other`: none of the above.

Secondary attributes may state that a test used a shell or an edit also created a file, but primary-category totals sum to `total_tool_calls`. Harness-specific normalizers map native tools conservatively and version their rules.

| Metric | Exact definition | Source data | Units | Unavailable and edge cases |
|---|---|---|---|---|
| `llm_requests` | Count of unique normalized LLM request events | `LLMRequestEvent` | requests | Zero only with complete LLM capture; duplicate/corrupt identities make the metric unavailable until deterministically resolved. |
| `total_tool_calls` | Count of unique tool-call invocations, regardless of outcome | `ToolCallEvent` | calls | Start/end records with one correlation ID count once. Unpaired starts count as calls with unknown/cancelled outcome when capture is otherwise complete. |
| `tool_calls_by_category` | Count by the single primary category | `ToolCallEvent` | calls | An unclassifiable but observed call is `other`; incomplete tool capture makes all comparative totals unavailable. |
| `successful_tool_calls` | Calls with normalized outcome `success` | `ToolCallEvent` | calls | Native success must have a documented mapping. Unknown outcome is not success. |
| `failed_tool_calls` | Calls with outcome `failure` or `timeout` | `ToolCallEvent` | calls | Cancelled and unknown are reported separately, not forced into failure. |
| `read_calls` | Calls with primary category `read` | `ToolCallEvent` | calls | Same coverage rules as category totals. |
| `search_calls` | Calls with primary category `search` | `ToolCallEvent` | calls | Same coverage rules. |
| `edit_calls` | Calls with primary category `edit` | `ToolCallEvent` | calls | Counts attempts, including failures. |
| `write_calls` | Calls with primary category `write` | `ToolCallEvent` | calls | Counts attempts, including failures. |
| `shell_calls` | Calls with primary category `shell` plus `test` calls whose mechanism is a shell, exposed as a non-overlapping query over the `uses_shell` attribute | `ToolCallEvent` | calls | The primary-category table still sums without duplication. The metric definition explicitly includes test-shell invocations once. |
| `agent_invoked_tests` | Calls with primary category `test` that were initiated by the harness during task timing | Test/ToolCall events and classifier version | calls | Setup checks and Agent Bench's own post-run verification are excluded. Reruns count separately. Ambiguous commands remain non-test and are listed for audit. |
| `calls_before_first_edit` | Tool invocations whose start precedes the first qualifying edit start | `ToolCallEvent` | calls | The first edit itself is excluded. Unavailable if first edit does not occur. |
| `calls_after_last_edit` | Tool invocations whose start is strictly after the last qualifying edit-call end/start boundary selected by schema (completed end preferred) and before task end | `ToolCallEvent` | calls | Unavailable if no edit exists or the last edit lacks a usable boundary. |
| `exact_duplicate_tool_calls` | Number of calls after the first in each group sharing identical normalized harness/tool identity and byte-identical canonical native arguments | `ToolCallEvent` canonical argument digests | calls | Calls need not be adjacent. Different path spellings or argument ordering are duplicates only if canonicalization rules explicitly make them identical. Redacted/absent arguments make this unavailable for affected coverage. |
| `repeated_identical_shell_commands` | Number of shell/test calls after the first with byte-identical executed argv/command representation and equivalent working directory/environment subset | `ToolCallEvent` | calls | Textually equal commands in different working directories are not identical. Unavailable when resolved command/cwd is missing. |
| `repeated_reads_of_unchanged_files` | For each successful read after the first successful read of the same normalized worktree path, count it when content identity is identical and no observed successful mutation to that path occurred between reads | Read/edit/write events plus file content hashes or snapshot identities | reads | Reads of different ranges still count when they access the same unchanged file; the argument-level duplicate metric handles identical ranges. Requires complete mutation coverage and content identity. Otherwise unavailable, never guessed from call order alone. |
| `reasoning_only_turns` | Count of observable harness turns containing reasoning output but no LLM-issued tool call/action and no visible final answer/action | Correlated reasoning, response, and tool events | turns | Unavailable unless turn boundaries and reasoning/action separation are exposed completely. |

## 7. Derived efficiency metrics

These are descriptive, not automatic quality rankings.

| Metric | Formula | Units | Edge cases |
|---|---|---|---|
| `tokens_per_tool_call` | `total_tokens / total_tool_calls` | tokens/call | Unavailable if inputs unavailable or denominator is zero. |
| `tokens_per_edit` | `total_tokens / (edit_calls + write_calls)` | tokens/mutation call | Unavailable for zero denominator. Counts attempts according to component definitions. |
| `reads_per_edit` | `read_calls / (edit_calls + write_calls)` | reads/mutation call | Unavailable for zero denominator. |
| `searches_per_edit` | `search_calls / (edit_calls + write_calls)` | searches/mutation call | Unavailable for zero denominator. |
| `seconds_per_edit` | `(wall_time / 1e9) / (edit_calls + write_calls)` | seconds/mutation call | Unavailable for zero denominator or unavailable wall time. |
| `failed_tool_call_rate` | `failed_tool_calls / total_tool_calls` | ratio in `[0,1]` | Unavailable for zero calls or incomplete outcomes. |
| `reasoning_to_output_ratio` | `reasoning_tokens / output_tokens` | ratio | Unavailable if counts unavailable or output is zero. Reports must state that reasoning may be a subset of output. |

No report labels a lower or higher derived efficiency value as better unless a future version defines and justifies such an interpretation.

## 8. Git/result metrics

Git/result metrics compare the exact baseline tree to the complete preserved result tree using one recorded Git/algorithm version. Worktree state, including relevant untracked files, must be represented in the preserved result before comparison.

| Metric | Exact definition | Source data | Units | Unavailable and edge cases |
|---|---|---|---|---|
| `files_changed` | Number of unique result-relative paths with added, deleted, modified, renamed, copied, or type-changed status. A rename counts as one changed logical entry and retains old/new paths. | `GitChangeSummary` per-path status | files | Unavailable if either tree/snapshot or comparison is incomplete. Submodule entries count as files with type metadata. |
| `files_created` | Count of added paths, excluding the destination of a detected rename | Git status/name-status evidence | files | Copy status is created only when the recorded comparison algorithm classifies it as a copy. |
| `files_deleted` | Count of deleted paths, excluding the source of a detected rename | Git evidence | files | Same completeness requirement. |
| `files_renamed` | Count of rename entries under the recorded deterministic rename-detection configuration | Git evidence | files | Rename detection must use fixed options; otherwise renames appear as create/delete according to the chosen algorithm and are not relabeled later. |
| `lines_added` | Sum of added text lines reported by the recorded diff algorithm across text files | Git numstat/diff evidence | lines | Binary files contribute neither zero nor guessed lines; they are excluded and `binary_files_changed` is reported. Unavailable if text diff is incomplete. |
| `lines_deleted` | Sum of deleted text lines under the same rules | Git evidence | lines | Same binary and completeness rules. |
| `source_files_changed` | Changed paths matching the versioned source-path classifier | Git summary | files | Unavailable if classifier/configuration is absent. |
| `test_files_changed` | Changed paths matching the versioned test-path classifier | Git summary | files | A path may carry both source/test labels only if classifier rules explicitly permit it. |
| `configuration_files_changed` | Changed paths matching the versioned configuration-path classifier | Git summary | files | Unavailable rather than inferred ad hoc. |

Line counts describe textual diff magnitude, not semantic work or quality. Line-ending-only changes count according to the fixed diff configuration.

## 9. Termination classification

Every run stores all observed error/status events and exactly one primary `termination_class`. Classification uses only explicit process status, limit enforcement, backend/harness records, context/truncation signals, preservation verification, and Git comparison. It does not use an LLM.

Classes:

| Class | Deterministic condition |
|---|---|
| `precondition_failed` | A required baseline/model/backend/hardware/configuration preflight check failed before task submission. |
| `preservation_failed` | Required result preservation or checksum verification failed; temporary evidence must be retained. |
| `timeout` | Agent Bench's configured task deadline fired before an authoritative clean completion. Subsequent forced kill remains supporting evidence. |
| `process_killed` | A required task process ended by external/explicit kill without a prior timeout or more specific captured cause. |
| `context_overflow` | Backend/protocol/harness emitted a recognized context-limit failure that prevented ordinary completion. |
| `output_truncation` | A configured/output/backend truncation signal occurred and no higher-precedence class applies, even if partial output or changes exist. |
| `model_backend_error` | A model inference, llama-server, transport, or backend protocol error prevented ordinary completion and no more specific class applies. |
| `harness_crash` | Harness exited abnormally or emitted a recognized fatal error, without evidence assigning the cause to a higher class. |
| `invalid_harness_output` | Harness completed but required protocol/output structure was malformed such that completion cannot be treated as ordinary. |
| `no_changes` | Harness completed ordinarily, but deterministic baseline/result comparison found zero changed files. |
| `success` | Harness completed ordinarily, required preservation verified, no higher class applies, and the result contains at least one changed file. This means execution success, not task correctness. |
| `unknown_other` | A terminal outcome exists but available evidence cannot place it in another class. A reason string and evidence references are required. |

Precedence is the table order from top to bottom, except `preservation_failed` is evaluated after collection and supersedes the execution outcome as the primary class while retaining that execution outcome as `underlying_termination_class`. `precondition_failed` runs never reach task timing. `no_changes` and `success` require complete Git comparison; otherwise `unknown_other` applies.

A nonfatal backend warning does not force `model_backend_error`; the event must be linked to failure of ordinary completion. Likewise, observed truncation that is explicitly outside the benchmark task stream is not classified as task output truncation.

## 10. Metric availability in reports

Reports display unavailable values as `N/A` with a machine-readable reason, never as blank or zero. Aggregates omit unavailable run values and display the contributing run count and excluded reasons. They do not impute missing values.

Failure-class runs are included whenever a metric is valid. Reports may filter by termination class, but default comparisons visibly retain failures so a harness cannot appear better because failed runs disappeared.

## 11. Validation invariants

Metric generation must reject or flag at least:

- duplicated supposedly unique event IDs or request indices;
- source references whose digests do not match;
- task end before task start;
- impossible negative durations;
- non-positive maximum context;
- token method without tokenizer/backend identity;
- utilization inconsistent with its token counts beyond exact numeric serialization tolerance;
- tool outcome totals exceeding total calls;
- category counts not summing to total calls; and
- Git summary totals inconsistent with per-path evidence.

Validation failure does not modify source evidence. It produces unavailable affected metrics and a versioned diagnostic artifact.
