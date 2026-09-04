# Benchmark-managed toolchains

Agent Bench never uses a personal installation, ambient `PATH`, shell startup
file, or mutable global package as a fallback. Large payloads are deliberately
outside normal Git history; their checked-in identity manifests, locks, hashes,
and installation logic are the source of truth. Run the following from a fresh
clone on Linux x86_64:

```bash
python -m agent_bench.cli toolchains install --component node --component opencode
python -m agent_bench.cli toolchains install --component pi
python -m agent_bench.cli toolchains verify
```

The installer downloads only the named pinned assets, verifies their download
hashes before materialization, then verifies the installed identity. It never
selects `latest`, `~/.opencode`, `~/.hermes`, a global npm package, or a `PATH`
binary. A missing or drifted payload remains `MISSING_OR_DRIFTED` until fixed.

## Automated components

- **OpenCode 1.18.25** — official `v1.18.25` Linux x64 tarball, archive SHA256
  `58a3729a6f3432dd6d2917fcc4a949788891a035818646ad480e12c947f56e78`,
  materialized at `toolchains/opencode/1.18.25/bin/opencode`. Its final Bun
  ELF must be 184682624 bytes and SHA256
  `d91e0d33676d0839f7cde87924cd4127ea88c9d6784eea9f009a7d08bdc60eeb`.
- **Node 26.8.1** — official `node-v26.8.1-linux-x64.tar.xz`, archive SHA256
  `3e301118d7df53d563b7e96c1617545f26e2f76f9724be668d6cab65c15dda5d`,
  materialized with its bundled npm at `toolchains/node/26.8.1`. `bin/node`
  must be SHA256 `19235a9b678f84729464c52623f92de130a165452747c6826d3fdc13df3abcc3`.
- **Pi 0.84.4** — only benchmark Node's bundled npm invokes `npm ci
  --ignore-scripts` against checked-in `toolchains/pi/0.84.4/package-lock.json`.
  The package is exactly `@earendil-works/pi-coding-agent@0.84.4`; its registry
  integrity, lock hash, entrypoint hash, and dependency-tree digest are tracked.

The 17.9 GB model is deliberately opt-in:

```bash
python -m agent_bench.cli toolchains install --component qwen --include-model
```

It downloads immutable Unsloth revision `f975863083b62f54a5e6fac11671c750c2bbc59c` from `environment/model-v1.json`
and verifies GGUF SHA256 `bee238bbeb3dc0a34bde4d0dedbaee1f98c009e8bb4226f03070054c12fb1372`.
Use `--model-destination PATH` only if the normal fixed profile location is not
writable; update the operational backend location separately before a run.

## Exact manual materialization

**Hermes 0.21.0** and the CUDA llama.cpp reference build remain manual in M9B:
they involve host Python/CUDA toolchains and cannot honestly promise a
byte-identical generic rebuild. `toolchains install` reports this explicitly.

For Hermes, clone `https://github.com/NousResearch/hermes-agent` at commit
`b3576a29c3f1a71f087c749a69fbb28f4e9628d6` into
`toolchains/hermes/0.21.0/source`; use pinned `uv 0.12.7` and `uv sync --locked
--python 3.11 --extra all`; then require `toolchains verify` to accept the
recorded source, lock, CPython, venv, and entrypoint identities. No content
from `~/.hermes` is accepted at runtime.

For llama.cpp, follow `environment/llama-cpp-build-v1.json`: clone
`https://github.com/ggml-org/llama.cpp` at
`dc72703fc69698b1ea68ece8d2dd8a96e6a4e1fe`, configure CUDA 12.9, GCC 13.3,
Release, CUDA architectures 70 and 86, and build `llama-server`. A
source-equivalent rebuild is not the reference binary: benchmark-v1 strictly
requires every executable/shared-library hash in `environment/backend-v1.yaml`.

## Archival policy

Normal Git stores source, subject bundles, definitions, locks, hashes, portable
provenance, and bootstrap logic—not binaries, model weights, `node_modules`,
Hermes source/venv, or Python runtime. Future GitHub Release assets may be
listed in `alternate_mirrors` without changing semantic identity: content hashes
remain authoritative.

The only intended tracked Hermes toolchain metadata file is
`toolchains/hermes/0.21.0/identity.json`; its `source/` and `venv/` payloads
are ignored and must never appear as thousands of ordinary Git changes.
