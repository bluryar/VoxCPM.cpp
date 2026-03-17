# VoxCPM.cpp

Standalone C++ inference project for VoxCPM models built on top of `ggml`.

VoxCPM Official Repository: https://github.com/OpenBMB/VoxCPM

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

### Inference Performance (RTF - lower is better)

| Model | Quant | Model Only | Without Encode | Full Pipeline |
|-------|-------|------------|----------------|---------------|
| voxcpm1.5 | **Q4_K** | **2.86** | **3.91** | 4.98 |
| voxcpm1.5 | Q8_0 | 3.48 | 4.51 | 5.90 |
| voxcpm1.5 | F32 | 3.93 | 4.66 | 5.85 |
| voxcpm1.5 | F16 | 10.28 | 11.56 | 15.02 |
| voxcpm-0.5b | **Q4_K** | **2.16** | **2.72** | 3.83 |
| voxcpm-0.5b | Q8_0 | 2.76 | 3.25 | 4.68 |
| voxcpm-0.5b | F32 | 3.67 | 4.00 | 5.19 |
| voxcpm-0.5b | F16 | 6.78 | 7.36 | 9.92 |

**RTF Definitions:**
- **Model Only**: Pure model inference (prefill + decode loop), excludes AudioVAE
- **Without Encode**: Model + AudioVAE decode (deployment scenario with offline prompt encoding)
- **Full Pipeline**: End-to-end including AudioVAE encode + model + decode

### Key Findings

1. **Q4_K performs best**: 27-37% faster than F32, while saving 83% storage
2. **F16 is slowest**: 2.6-2.8x slower than F32, likely due to memory bandwidth or SIMD optimization issues
3. **Q8_0 vs F32**: Q8_0 is comparable or slightly faster than F32, with 72% size reduction
4. **Model comparison**: 0.5B is ~30-40% faster than 1.5B, with 18% smaller size

### Deployment Recommendations

| Scenario | Recommended Config |
|----------|-------------------|
| Production | **voxcpm-0.5b Q4_K** (477 MB, RTF 2.72) |
| Balanced accuracy | voxcpm1.5 Q4_K (582 MB, RTF 3.91) |
| Max accuracy | voxcpm1.5 F32 (3392 MB, RTF 4.66) |
| Avoid | F16 quantization (slowest, no benefit) |

**Test environment:**
- Platform: Orange Pi 6 Plus
- SoC: CIX P1 CD8160
- CPU: 12 cores (2x A720 @ 2.6GHz + 6x A720 @ 2.5GHz + 4x A520 @ 1.8GHz)
- Arch: aarch64 with SVE, SVE2, BF16, i8mm
- Threads: 8
- Backend: CPU
