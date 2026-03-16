# GGML Subtree Maintenance Strategy

## Status

This document is a **future maintenance strategy**, not an instruction to execute immediately.

Current state:

- `VoxCPM.cpp` is still a temporary directory under the top-level `ggbond` repository.
- `VoxCPM.cpp` is not yet being published as its own standalone Git repository.
- Therefore, this document should be treated as a design note for the post-v1 stage.

Recommended trigger for adoption:

- after the first standalone `VoxCPM.cpp` release
- when `VoxCPM.cpp` becomes its own Git repository
- when there is a clear need to carry local `ggml` patches while still consuming upstream `ggml` updates

## Goal

Use `ggml` as a maintained subtree so that `VoxCPM.cpp` can:

- preserve a clear upstream relationship to `ggml`
- keep local performance or correctness patches when needed
- continue to pull future upstream `ggml` fixes and optimizations
- avoid the long-term maintenance problems of an unmanaged vendored copy

This strategy is meant to balance:

- upstream compatibility
- local performance work
- traceability of local changes
- manageable upgrade cost

## Why Subtree

Compared with copying `ggml` into `third_party` without structure, `git subtree` provides a better long-term model:

- upstream sync remains possible
- local modifications can live in normal Git history
- future merges from upstream are more structured
- it is easier to explain where the embedded code came from

Compared with a submodule, subtree may be a better fit if:

- `VoxCPM.cpp` wants a self-contained repository layout
- contributors should not need extra submodule initialization steps
- local patching of `ggml` is expected

## Core Principles

### 1. Prefer local fixes before `ggml` patches

Not every performance issue should be solved inside `ggml`.

Before modifying subtree-managed `ggml`, check whether the issue can be fixed in:

- `VoxCPM.cpp` operator composition
- graph construction
- threading configuration
- model-specific custom operators
- data layout or pre/post-processing

Reason:

- local fixes are easier to maintain
- local fixes reduce future upstream merge conflicts
- many issues are model-specific rather than generic `ggml` problems

This matches the AudioVAE investigation:

- the first real bottleneck was a model-local depthwise operator that was effectively single-threaded
- the best immediate fix was local, not a deep `ggml` patch

### 2. Patch `ggml` only for shared, durable value

Modify `ggml` only when at least one of these is true:

- the problem is clearly inside `ggml` itself
- the fix benefits multiple models or modules
- the fix is likely to remain useful across upstream updates
- the performance gain is large enough to justify long-term merge cost

Examples of justified `ggml` patches:

- SIMD optimization in generic `im2col`
- new fused operator that multiple models need
- backend correctness fix
- generic threading bug

Examples of weak candidates:

- model-specific workaround with no general benefit
- patching `ggml` to preserve a single module's unusual semantics if that logic can remain local

### 3. Keep local `ggml` patches small and isolated

Every local subtree patch should be:

- narrowly scoped
- easy to explain in one paragraph
- easy to test independently
- easy to drop or rewrite during upstream sync

Avoid broad refactors of upstream code unless absolutely necessary.

### 4. Treat upstream sync as a regular maintenance task

Do not let subtree diverge for too long.

If local `ggml` changes accumulate for a long time without upstream sync:

- merge cost rises
- conflicts become harder
- performance comparisons become less meaningful
- it becomes harder to tell whether a local patch is still needed

## Recommended Repository Layout

After `VoxCPM.cpp` becomes its own repository, keep `ggml` under:

- `third_party/ggml`

Keep all `ggml`-specific maintenance notes under:

- `docs/ggml/`

Suggested future docs:

- `docs/ggml/upstream-sync-log.md`
- `docs/ggml/local-patches.md`
- `docs/ggml/perf-notes.md`

## Recommended Patch Tracking Policy

Every local `ggml` patch should be documented in a short registry.

Suggested format for `docs/ggml/local-patches.md`:

For each patch, record:

- patch title
- date added
- owner
- affected files
- reason for patch
- benchmark or correctness evidence
- whether it should eventually be proposed upstream
- likely merge risk during future sync

Suggested patch categories:

- `perf`
- `correctness`
- `backend`
- `temporary-workaround`

If a patch is only a temporary workaround, mark it explicitly.

## Recommended Upstream Sync Workflow

Once subtree maintenance begins, use a consistent workflow.

High-level steps:

1. identify the upstream `ggml` commit or release to import
2. sync subtree into `third_party/ggml`
3. resolve conflicts carefully
4. rerun correctness tests
5. rerun representative performance tests
6. update patch tracking docs

What to verify after each sync:

- AudioVAE trace tests
- MiniCPM and other model trace or regression tests
- CPU threading behavior
- backend-specific functionality that your project uses
- performance-sensitive paths that previously relied on local patches

## Decision Rules For Future AudioVAE-Style Issues

When a future slowdown appears, use this decision order:

1. confirm the bottleneck with measurement
2. check whether the issue is local to `VoxCPM.cpp`
3. fix locally if possible
4. only then decide whether a subtree-level `ggml` patch is warranted

For example:

- If the issue is a single-threaded custom operator in local code, fix it locally first.
- If the issue is clearly inside generic `ggml` `im2col`, then a subtree patch may be justified.

## Risks Of The Subtree Approach

Subtree is useful, but not free.

Main risks:

- local patches can still conflict with upstream sync
- performance patches can become stale after upstream changes
- generic fixes may need revalidation across all supported backends
- developers may start patching `ggml` too casually because the code is nearby

The main discipline required is:

- do not patch upstream code casually
- document every local patch
- keep sync cadence healthy

## Recommendation For VoxCPM.cpp

Once `VoxCPM.cpp` becomes its own repository, adopting `ggml` as a subtree is a good long-term choice if both of these become true:

- `VoxCPM.cpp` wants to stay self-contained
- `VoxCPM.cpp` expects to carry a small number of local `ggml` patches over time

It is **not** recommended to move to subtree solely because a third-party directory exists.

It **is** recommended if the project enters a stage where:

- upstream `ggml` updates matter
- local `ggml` performance work becomes recurring
- correctness and provenance both need to be preserved

## Final Policy

Future policy recommendation:

- keep `ggml` vendored in `third_party/ggml`
- once `VoxCPM.cpp` is a standalone repo, convert it to a managed subtree
- prefer local fixes before `ggml` patches
- patch `ggml` only for high-value generic improvements
- document every local `ggml` patch
- revalidate both correctness and performance after every upstream sync

This gives the project a practical path to:

- preserve traceability
- absorb future upstream performance gains
- keep local optimizations when necessary
- avoid turning `third_party/ggml` into an unstructured fork
