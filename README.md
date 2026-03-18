# VoxCPM.cpp

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Standalone C++ inference project for VoxCPM models built on top of `ggml`.

- **GGUF Weights**: https://huggingface.co/bluryar/VoxCPM-GGUF
- VoxCPM Official Repository: https://github.com/OpenBMB/VoxCPM

[中文文档](README_zh.md)

## Status

This directory now serves as the standalone repository root for `VoxCPM.cpp`.

- `third_party/ggml` is intended to be maintained as a vendored subtree.
- `third_party/json`, `third_party/llama.cpp`, `third_party/whisper.cpp`, and `third_party/SenseVoice.cpp` are kept only as local references and are ignored by this repository.
- `CMakeLists.txt` already supports downloading `nlohmann_json` with `FetchContent` when `third_party/json` is absent.

## Build

```bash
cmake -B build
cmake --build build
```

## Tests

```bash
cd build
ctest --output-on-failure
```

For configurable model/trace test paths and open-source collaboration setup, see [docs/TEST_SETUP.md](docs/TEST_SETUP.md).

## ggml Maintenance

The project keeps local provenance for the current `ggml` import and patch flow:

- upstream: `https://github.com/ggerganov/ggml.git`
- current local base commit before repository split: `4773cde162a55f0d10a6a6d7c2ea4378e30e0b01`
- current local patch: Vulkan header compatibility adjustment in `src/ggml-vulkan/ggml-vulkan.cpp`

See `docs/ggml_subtree_maintenance_strategy.md` for the longer-term maintenance approach.

## Benchmark

### Model Size & Compression

| Model | Quant | Size (MB) | Compression |
|-------|-------|-----------|-------------|
| voxcpm1.5 | F32 | 3392 | 1.00x (baseline) |
| voxcpm1.5 | F16 | 1700 | 1.99x |
| voxcpm1.5 | Q8_0 | 942 | 3.60x |
| voxcpm1.5 | Q4_K | 582 | 5.82x |
| voxcpm-0.5b | F32 | 2779 | 1.00x (baseline) |
| voxcpm-0.5b | F16 | 1394 | 1.99x |
| voxcpm-0.5b | Q8_0 | 766 | 3.62x |
| voxcpm-0.5b | Q4_K | 477 | 5.82x |

### CPU Inference Performance (RTF - lower is better)

| Model | Quant | Model Only | Without Encode | Full Pipeline |
|-------|-------|------------|----------------|---------------|
| voxcpm1.5 | Q4_K | 2.395 | 3.395 | 5.598 |
| voxcpm1.5 | **Q4_K+AudioVAE-F16** | **1.873** | **2.848** | 4.433 |
| voxcpm1.5 | **Q8_0** | 2.086 | 2.982 | **4.291** |
| voxcpm1.5 | Q8_0+AudioVAE-F16 | 2.285 | 3.321 | 5.248 |
| voxcpm1.5 | F16 | 3.257 | 4.366 | 6.263 |
| voxcpm1.5 | F16+AudioVAE-F16 | 2.980 | 3.915 | 5.374 |
| voxcpm1.5 | F32 | 4.820 | 5.737 | 7.494 |
| voxcpm-0.5b | **Q4_K** | **1.826** | **2.219** | **3.609** |
| voxcpm-0.5b | Q4_K+AudioVAE-F16 | 1.895 | 2.295 | 3.915 |
| voxcpm-0.5b | Q8_0 | 2.155 | 2.546 | 3.873 |
| voxcpm-0.5b | Q8_0+AudioVAE-F16 | 1.913 | 2.284 | 3.638 |
| voxcpm-0.5b | F16 | 2.558 | 2.931 | 4.086 |
| voxcpm-0.5b | F16+AudioVAE-F16 | 2.685 | 3.057 | 4.409 |
| voxcpm-0.5b | F32 | 3.691 | 4.055 | 5.260 |

### CUDA Inference Performance (RTF - lower is better)

| Model | Variant | AudioVAE | Model Only | Without Encode | Full Pipeline | Total Time (s) |
|-------|---------|----------|------------|----------------|---------------|----------------|
| voxcpm1.5 | Q4_K | mixed | 0.342 | 0.432 | 0.622 | 2.189 |
| voxcpm1.5 | Q4_K+AudioVAE-F16 | f16 | 0.336 | 0.426 | 0.596 | 2.192 |
| voxcpm1.5 | Q8_0 | mixed | **0.320** | **0.411** | 0.596 | 2.002 |
| voxcpm1.5 | Q8_0+AudioVAE-F16 | f16 | **0.308** | **0.397** | **0.559** | 2.148 |
| voxcpm1.5 | F16 | mixed | 0.352 | 0.442 | 0.648 | 1.970 |
| voxcpm1.5 | F16+AudioVAE-F16 | f16 | 0.347 | 0.438 | 0.655 | **1.885** |
| voxcpm1.5 | F32 (baseline) | original | 0.414 | 0.503 | 0.686 | 2.305 |
| voxcpm-0.5b | Q4_K | mixed | 0.401 | 0.442 | **0.550** | 2.067 |
| voxcpm-0.5b | Q4_K+AudioVAE-F16 | f16 | 0.396 | 0.437 | 0.555 | 1.953 |
| voxcpm-0.5b | Q8_0 | mixed | 0.430 | 0.470 | 0.623 | **1.644** |
| voxcpm-0.5b | Q8_0+AudioVAE-F16 | f16 | 0.417 | 0.456 | 0.595 | 1.809 |
| voxcpm-0.5b | F16 | mixed | **0.390** | **0.428** | 0.567 | 1.678 |
| voxcpm-0.5b | F16+AudioVAE-F16 | f16 | 0.392 | 0.430 | 0.565 | 1.718 |
| voxcpm-0.5b | F32 (baseline) | original | 0.500 | 0.539 | 0.680 | 1.903 |

**RTF Definitions:**
- **Model Only**: Pure model inference (prefill + decode loop), excludes AudioVAE
- **Without Encode**: Model + AudioVAE decode (deployment scenario with offline prompt encoding)
- **Full Pipeline**: End-to-end including AudioVAE encode + model + decode

### Key Findings

#### CPU

1. **CPU winners now depend on model and pipeline stage**: `voxcpm1.5 Q4_K+AudioVAE-F16` leads on model-only and without-encode RTF, while `voxcpm1.5 Q8_0` has the best full-pipeline RTF; `voxcpm-0.5b Q4_K` remains the strongest overall CPU choice.
2. **AudioVAE-F16 matters on CPU for 1.5B**: `Q4_K+AudioVAE-F16` gives the best `voxcpm1.5` model-only and without-encode RTF, while `Q8_0` gives the best full-pipeline RTF.
3. **Q4_K remains strongest on 0.5B CPU runs**: `voxcpm-0.5b Q4_K` has the best overall CPU RTF, with `Q8_0+AudioVAE-F16` close behind on full-pipeline performance.
4. **F32 is slowest on this CPU setup**: both `voxcpm1.5` and `voxcpm-0.5b` show the worst CPU RTF with F32 baseline weights.

#### CUDA

1. **CUDA is substantially faster than CPU**: full-pipeline RTF drops from `3.83-15.02` on CPU to `0.55-0.69` on CUDA in this benchmark set.
2. **Best CUDA variant depends on metric**: for `voxcpm1.5`, `Q8_0+AudioVAE-F16` gives the best RTF, while `F16+AudioVAE-F16` gives the shortest total time; for `voxcpm-0.5b`, `Q4_K` gives the best full-pipeline RTF, while `Q8_0` gives the shortest total time.
3. **CUDA no longer clearly favors Q4_K**: unlike CPU, `Q4_K` is not consistently the fastest on CUDA; `Q8_0` and `F16` are often competitive or better.
4. **AudioVAE F16 can help on CUDA**: forcing AudioVAE to `F16` improves several CUDA runs, especially for `voxcpm1.5 Q8_0` and `voxcpm-0.5b Q8_0`.

### Deployment Recommendations

| Scenario | Recommended Config |
|----------|-------------------|
| Production | **voxcpm-0.5b Q4_K** (477 MB, RTF 3.609) |
| Balanced accuracy | **voxcpm1.5 Q8_0** (942 MB, RTF 4.291) |
| Best 1.5B offline prompt pipeline | voxcpm1.5 Q4_K+AudioVAE-F16 (647 MB, RTF 2.848 without encode) |
| Max accuracy baseline | voxcpm1.5 F32 (3392 MB, RTF 7.494) |

### Deployment Recommendations (CUDA)

| Scenario | Recommended Config |
|----------|-------------------|
| Lowest full-pipeline RTF | **voxcpm-0.5b Q4_K** (477 MB, RTF 0.550) |
| Best 1.5B latency/RTF balance | **voxcpm1.5 Q8_0+AudioVAE-F16** (984 MB, RTF 0.559) |
| Smallest CUDA-friendly 1.5B model | voxcpm1.5 Q4_K+AudioVAE-F16 (647 MB, RTF 0.596) |
| Max accuracy baseline | voxcpm1.5 F32 (3392 MB, RTF 0.686) |

**CPU test environment:**
- CPU: 12th Gen Intel(R) Core(TM) i5-12600K
- Threads: 8
- Backend: CPU
- Benchmark source: `logs/benchmark_summary_cpu_20260318_092142.txt`

**CUDA test environment:**
- Backend: CUDA
- GPU: NVIDIA GeForce RTX 4060 Ti
- CUDA device: `CUDA0`
- Compute capability: 8.9
- CUDA VMM: yes
- CPU host: 12th Gen Intel(R) Core(TM) i5-12600K
- Threads: 8
- Inference timesteps: 10
- CFG value: 2.0
- Benchmark source: `logs/benchmark_summary_cuda_20260318_092028.txt`
