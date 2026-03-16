# AudioVAE Performance Investigation Notes

## Summary

This document summarizes:

- what was actually observed while investigating AudioVAE performance
- which parts of the external optimization proposal are correct or useful
- which parts are incomplete, misleading, or too risky for the current repo state
- what change was implemented and why it is the current best fix

The short version:

- AudioVAE was slow primarily because its custom depthwise convolution path was effectively single-threaded.
- A direct switch to `ggml_conv_1d_dw()` was not a safe fix in this repo: it hit a CPU `F16` assertion and did not preserve current trace semantics.
- The implemented fix keeps the existing numerically-correct depthwise behavior, but parallelizes it across ggml worker threads.
- This produced a real speedup while keeping the AudioVAE trace tests passing.

## What Was Actually Slow

The main bottleneck was in [`src/audio-vae.cpp`](/home/orangepi/Codes/ggbond/VoxCPM.cpp/src/audio-vae.cpp), in the custom depthwise convolution used by residual units.

Before the fix:

- `depthwise_conv_custom()` returned immediately for all workers except `ith == 0`
- `ggml_map_custom3(..., 1, ...)` requested only a single task
- the hottest part of AudioVAE encode therefore ran almost entirely single-threaded

This mattered most on encode because the early encoder stages operate on very long sequences. During investigation, encode was much slower than decode for exactly this reason.

## Evaluation Of The External Proposal

### 1. "im2col has no SIMD optimization"  
Judgment: partially correct, but not the best first fix for this repo.

What is true:

- The F32 `im2col` implementation in [`third_party/ggml/src/ggml-cpu/ops.cpp`](/home/orangepi/Codes/ggbond/VoxCPM.cpp/third_party/ggml/src/ggml-cpu/ops.cpp#L6112) is a straightforward nested-loop implementation.
- It is a plausible optimization target if we are willing to modify ggml internals.

What is missing:

- This was not the most immediate and highest-confidence bottleneck in the current AudioVAE implementation.
- Our own depthwise op was provably underutilizing CPU threads, which is a much more direct explanation for poor performance.
- Optimizing ggml `im2col` first would require third-party changes, architecture-specific SIMD work, and careful regression validation.

Conclusion:

- The observation is technically useful.
- It is not the best first fix for the current issue.

### 2. "Snake activation is unfused and expensive"  
Judgment: directionally correct, but secondary.

What is true:

- `snake_activation()` in [`src/audio-vae.cpp`](/home/orangepi/Codes/ggbond/VoxCPM.cpp/src/audio-vae.cpp#L113) creates several graph nodes and repeats broadcasts.
- A fused snake operator could reduce graph size and memory traffic.

What is missing:

- We did not find evidence that Snake was the primary reason AudioVAE was slower than expected.
- Even after the depthwise fix, Snake remains a reasonable next optimization target, but it was not the blocker that explained the observed large encode slowdown.

Conclusion:

- Good medium-priority optimization idea.
- Not the first thing to change.

### 3. "Use `ggml_conv_1d_dw()`"  
Judgment: good idea in principle, not safe in this repo right now.

What was tested:

- We tried replacing the custom AudioVAE depthwise implementation with `ggml_conv_1d_dw()`.

What actually happened:

- On CPU, the current ggml path hit `GGML_ASSERT(src0->type == GGML_TYPE_F16)` in the F16 `im2col` path.
- Even after trying to adapt tensor types at the AudioVAE layer, the resulting behavior did not preserve current AudioVAE trace outputs.
- In other words: the native path was not a drop-in replacement for this model in the current repository state.

Conclusion:

- As a long-term goal, moving to native ggml ops is still attractive.
- As an immediate fix, it was not correct or safe.

### 4. "Add SIMD to im2col / add GPU support / explore FFT or direct conv"  
Judgment: mostly long-term ideas, not immediate fixes.

These are valid performance workstreams, but they are:

- much higher implementation cost
- harder to validate
- more invasive because they affect `third_party/ggml`
- not necessary to achieve the first meaningful speedup

Conclusion:

- Good future roadmap items.
- Not the right first move for the bug we were actually seeing.

## What Was Implemented

The implemented fix was:

1. keep the existing custom depthwise convolution semantics
2. parallelize the work across ggml worker threads
3. request multiple tasks from ggml instead of forcing a single task

Relevant code:

- depthwise work partitioning in [`src/audio-vae.cpp`](/home/orangepi/Codes/ggbond/VoxCPM.cpp/src/audio-vae.cpp#L27)
- multi-task scheduling in [`src/audio-vae.cpp`](/home/orangepi/Codes/ggbond/VoxCPM.cpp/src/audio-vae.cpp#L320)

The key change is that the work is now partitioned over `channels * batch`, instead of only letting one worker execute the full kernel.

This approach was chosen because it satisfies all of the following:

- preserves current trace correctness
- avoids third-party ggml modifications
- directly addresses the measured bottleneck
- gives an immediate CPU speedup

## Verification Results

The fix was validated with existing AudioVAE trace tests:

- `ctest -R test_audio_vae --output-on-failure` passed
- encode and decode trace tolerances remained within the existing threshold

Thread scaling spot check:

- `VOXCPM_TEST_THREADS=1`: encode compute around `18189.9 ms`
- `VOXCPM_TEST_THREADS=8`: encode compute around `5021.69 ms`

This confirmed that the hot path was no longer effectively single-threaded.

## About Thread Configuration

During validation, a test-only environment variable `VOXCPM_TEST_THREADS` was added in [`tests/test_audio_vae.cpp`](/home/orangepi/Codes/ggbond/VoxCPM.cpp/tests/test_audio_vae.cpp#L32).

Important clarification:

- This environment variable is only a convenience for test-time thread sweeps.
- The real ggml thread control still happens through `VoxCPMBackend`, which calls `ggml_backend_cpu_set_n_threads(...)` in [`src/backend.cpp`](/home/orangepi/Codes/ggbond/VoxCPM.cpp/src/backend.cpp#L17).

So the environment variable is not the actual threading mechanism. It is only a convenient way to choose the `n_threads` value passed into the backend during tests.

Recommended interpretation:

- production/runtime code: set thread count explicitly through backend/config/API
- tests/benchmarks: environment-variable override is acceptable as a lightweight convenience

## Recommended Next Steps

Priority order for future work:

1. Fuse or simplify `snake_activation()` if further CPU latency reduction is needed.
2. Add more targeted profiling around encode residual units to quantify where time still goes after the thread fix.
3. Revisit `ggml_conv_1d_dw()` only after confirming current ggml semantics and type expectations can reproduce AudioVAE traces.
4. Consider deeper ggml work such as SIMD `im2col` only if AudioVAE and other models would clearly benefit enough to justify third-party maintenance cost.

## Final Recommendation

For the current repository, the best answer to "why AudioVAE was slow and what should we do first?" is:

- The largest immediate problem was not generic ggml `im2col`, but our own effectively single-threaded custom depthwise implementation.
- The best fix was to parallelize that custom operator while keeping exact current semantics.
- The external proposal contains several useful ideas, but it over-indexes on deep ggml surgery before fixing the simpler, proven bottleneck in local code.
