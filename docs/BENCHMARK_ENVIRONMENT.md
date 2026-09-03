# Agent Bench v1 Fixed Environment

Status: pre-M5 deployment input definition  
Environment definition version: 1.0.0

## Qwen 3.8 chat template

The fixed Agent Bench v1 chat template is derived from the exact
`tokenizer.chat_template` metadata embedded in:

`/mnt/starhunter/AI/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q4_K_XL.gguf`

- Embedded source template SHA256: `12827f24b742ea4e80cdc12dbcf9622227056b9f797252a3149263d4f9aaadce`
- Patch provenance: `/home/bking/AI/chat-templates/qwen38-flash-next/hermes-minimal-empty-think-fix.diff`
- Resulting template: `environment/templates/qwen38-agent-bench-v1.jinja`
- Resulting template SHA256: `2d59a4438d68dc818c5a75db4edcf4c588e0976b113c5c87def7fc9c1168e955`

The only semantic modification adds an `and reasoning_content` guard when
rendering historical assistant reasoning. This prevents messages with no actual
reasoning content from adding empty `<think>...</think>` blocks to later prompts.
The repository text file also has a conventional terminal LF; the embedded GGUF
metadata string does not. This byte-level newline difference has no Jinja
semantic effect and is included in the resulting SHA256 above.

The result retains the embedded Unsloth behavior, including developer-role
support, merging of leading system/developer messages, validation of misplaced
system/developer messages, `high` to `xhigh` reasoning normalization, tool
function-name validation, and tool-call argument validation.

The bounded-medium template at
`/home/bking/AI/qwen38-27b-rtx3090/chat_templates/qwen3.8-medium-bounded.jinja`
is intentionally not used. Agent Bench v1 does not add its reasoning-token
limit, earlier-tool-use guidance, turn-splitting guidance, file-reading strategy,
or any other behavioral instruction.
