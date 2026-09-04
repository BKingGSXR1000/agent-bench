# Portable benchmark identities

M1 sealed definitions use `identity_version: 1.0.0`. Their definition and run
identities include `baseline_repository`, so they are historical local evidence
and are never rewritten. They remain readable and verifiable under their
original schema.

Future published matrices use `identity_version: 2.0.0`. A v2 definition must
include `portable_baseline`, whose intrinsic values are subject ID/version,
baseline commit, baseline tree, and baseline bundle SHA256. The v2 experiment
and matrix identities additionally include fixed-environment semantic/content
identities, harness/profile semantic identities, prompt ID/content SHA256,
repetition-derived seed, and run limits. The run ID is derived only from those
values.

The following are operational evidence and excluded from v2 semantic identity:

- Agent Bench clone root, baseline checkout/worktree, output root, and temporary paths;
- model, llama-server, template, harness, Node, Python, and profile local paths;
- local runtime payload locations and host-specific environment paths.

Their byte/content identities remain intrinsic: model/template/executable
SHA256 and size, llama.cpp commit/build identity, package integrity/lock/tree
digest, and profile content hashes. Each sealed local run still records actual
resolved paths for forensic reproduction. A changed pinned hash, commit, tree,
prompt SHA, harness/profile semantic digest, or fixed-environment semantic
digest changes the v2 matrix/run identity.

`subjects/pocket-ledger-v1/baseline-v1.bundle` is the portable reconstruction
source. The executor verifies its SHA256, clones it into a new run-owned
repository, checks out the recorded detached commit, verifies the recorded
tree and clean status, and compares it with the tracked baseline source before
the existing M2 worktree lifecycle starts. The clone is scratch state and is
removed only after immutable result verification.
