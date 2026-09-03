# Git Isolation and Result Preservation

Status: Milestone M2 implemented contract  
Schema version: 1.0.0

## 1. Scope

M2 provides the filesystem and Git boundary needed before run execution exists. It
resolves an immutable baseline, creates a detached temporary worktree, applies an
injected local filesystem operation, preserves the complete result, verifies it,
and only then removes the worktree. The injected operation is a test seam, not a
harness abstraction or benchmark execution loop.

M2 does not create harness adapters, isolated harness homes, event streams, run
manifests, metrics, artifact builds, or benchmark lifecycle orchestration. Those
remain assigned to later milestones.

## 2. Baseline and worktree safety

`resolve_baseline` accepts a local Git repository and a reference. It resolves the
reference with `^{commit}` to a full commit object ID without checking out or
changing the baseline repository. A missing repository, non-Git directory,
unresolvable reference, or non-commit reference is rejected.

Each operation runs in a newly allocated worktree created by `git worktree add
--detach` at the resolved commit. Agent Bench verifies both the exact `HEAD` and
the absence of a symbolic branch. Artifact and worktree roots must be outside the
baseline repository. No M2 operation writes application files into the baseline
checkout.

Worktree removal is an explicit final action. It occurs only after all of the
following have succeeded:

1. source and Git evidence capture;
2. result-commit and result-ref creation;
3. manifest and checksum creation;
4. checksum, archive, manifest, and result-ref verification; and
5. atomic promotion of the incomplete artifact directory.

## 3. Result Git identity

For run ID `<run-id>`, the result is pinned at exactly:

```text
refs/agent-bench/results/<run-id>
```

The ref is created atomically only when it does not already exist. Existing refs
and artifact directories are never overwritten.

The result commit has the baseline commit as its parent and captures changes and
deletions to files tracked by the baseline. It is produced with a temporary Git
index, so it does not stage or alter the run worktree's index. Commit identity is
deterministic for the same run ID, baseline, and tracked result tree.

Untracked and ignored files are intentionally not added to this Git-derived
commit. They are captured in the complete source snapshot and inventoried
separately. Consequently, reconstruction and later application testing use the
snapshot; the result commit supplies a durable Git identity for the tracked
result state and protects its Git objects from garbage collection.

## 4. Complete source snapshot

`source/source.tar` is an uncompressed POSIX PAX tar archive. Traversal order is
byte-sorted. File contents, executable/mode bits, empty directories, and safe
relative symbolic links are preserved. Archive modification times, owner IDs,
and owner names are normalized so host metadata does not make otherwise
identical source trees produce different archives.

The snapshot includes tracked, untracked, and Git-ignored source-tree content.
In particular, directory names such as `node_modules`, `dist`, `build`, `vendor`,
and `generated` are not excluded merely because they are large or generated.

The versioned `m2-default-v1` policy excludes only:

- the worktree `.git` administrative file or directory;
- Agent Bench temporary directories named `.agent-bench-tmp` or `.bench`;
- Python cache directories named `__pycache__`;
- files ending in `.pyc`; and
- pytest cache directories named `.pytest_cache`.

Every excluded top-level path is recorded in `source/excluded.txt`, and the
manifest records the policy and count of excluded file or link entries. M2 fails
closed on special filesystem objects and symbolic links that are absolute or
would escape the source root. It does not silently omit them.

## 5. Artifact layout

A successful result is stored at `<artifacts-root>/<run-id>/`:

```text
manifest.json
checksums.sha256
source/source.tar
source/excluded.txt
git/baseline.txt
git/result.txt
git/status.txt
git/diff.patch
git/untracked.txt
git/ignored.txt
build/
```

The versioned manifest records the run and experiment IDs, baseline repository
and commit, result commit and ref, snapshot and diff paths and SHA256 values,
checksum path, UTC creation timestamp, preserved file/byte counts, excluded-file
count, exclusion policy, preservation status, and build metadata. M2 performs no
build, so build artifact paths are empty and build/launch commands are null.
Successful manifests use the data-model lifecycle state `sealed`; incomplete
manifests transition from `verifying` to `failed` when recovery bookkeeping can
be completed.

`checksums.sha256` uses sorted artifact-relative paths and SHA256 and covers every
regular artifact file other than the checksum listing itself, including
`manifest.json`. This avoids a recursive checksum field: the manifest names the
listing and the listing authenticates the manifest. The full per-entry artifact
roles and run-level records specified in `DATA_MODEL.md` will be added with the
later data-layer milestones.

The `git/*.txt` inventories use one JSON-quoted Git path per line, preserving
unusual UTF-8 paths without treating filenames as shell syntax. `git/diff.patch`
is a full-index binary-capable diff against the immutable baseline.

## 6. Verification, restoration, and failure recovery

Verification checks:

- schema and preservation status;
- every listed file checksum;
- manifest snapshot and diff hashes;
- baseline/result text identities;
- safe, unique archive members and the exact file/byte totals;
- absence of policy-excluded archive members; and
- the result ref and commit when the baseline repository remains available.

Restoration always verifies first, then extracts into a new or empty directory
outside the artifact. It restores the complete snapshot without `.git`; it does
not mutate the immutable artifact or recreate a worktree registration.

Collection occurs in a uniquely named hidden `.incomplete-*` directory. Any
operation, preservation, checksum, or verification failure retains the temporary
worktree. Preservation-stage failures also retain the incomplete artifact and
mark its manifest `failed` when a manifest exists. Recovery evidence is never
automatically deleted. A failed run ID is not silently retried over an existing
artifact or pinned result ref.

## 7. Diagnostic CLI

M2 adds three diagnostic commands; none executes a benchmark:

```text
agent-bench git baseline REPOSITORY REFERENCE
agent-bench artifact verify ARTIFACT
agent-bench artifact restore ARTIFACT DESTINATION
```

The first prints the resolved repository and immutable commit. The latter two
verify or restore a previously preserved M2 artifact.
