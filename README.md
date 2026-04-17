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
- `VoxCPM2` is now supported on a preliminary basis. The current C++ runtime can load exported `VoxCPM2` GGUF weights, run end-to-end inference, use the new reference-mode plumbing, and produce 48kHz output through the `AudioVAE V2` path, but quality and parity are still under active validation.

## Refactor Preview

A larger Torch-to-GGML runtime refactor is planned. The design direction is documented in:

- [docs/voxcpm_torch_to_ggml_complete_refactor_cookbook_zh.md](docs/voxcpm_torch_to_ggml_complete_refactor_cookbook_zh.md)
- [docs/torch_to_ggml_migration_guide_zh.md](docs/torch_to_ggml_migration_guide_zh.md)

Why this refactor is needed:

- The current codebase already runs, but much of the implementation still reflects a direct "translate PyTorch modules into C++ pieces" path.
- That approach is good for bringing a model up quickly, but it makes shape contracts, ownership boundaries, persistent state, graph lifetime, and backend placement harder to reason about.
- It also tends to introduce avoidable host/backend round-trips such as `tensor_get -> std::vector -> tensor_set`, which become increasingly costly once the model grows or multi-backend execution is involved.

The refactor target is not a cosmetic rewrite. The goal is to move `VoxCPM.cpp` toward a more mature `ggml` runtime with:

- explicit GGUF and module-level contracts
- a shared `WeightStore` and backend-aware loader/runtime skeleton
- clear separation of weights, persistent state, compute memory, and output buffers
- graph caching keyed by real rebuild conditions instead of ad hoc shape guesses
- backend-resident hot-path data flow between modules whenever possible

In short, the project is moving away from a host-side module translation style and toward a contract-first, backend-aware runtime architecture that is easier to verify, optimize, and extend across CPU/CUDA/Vulkan paths.

To help accelerate that work, I also plan to use `ClaudeCode Opus 4.6` for a larger code rewrite pass focused on improving maintainability, clarifying runtime/module boundaries, and reducing the amount of legacy glue code that accumulated during the initial bring-up phase.

## Build

### CPU Build

```bash
cmake -B build
cmake --build build
```

### CUDA Build

Enable the ggml CUDA backend at configure time only if you want to run with `--backend cuda`:

```bash
cmake -B build-cuda \
  -DVOXCPM_CUDA=ON \
  -DVOXCPM_BUILD_BENCHMARK=OFF \
  -DVOXCPM_BUILD_TESTS=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-cuda
```

If you want to keep both CPU and CUDA builds, use separate build directories such as `build` and `build-cuda`.

Important:

- `-DVOXCPM_CUDA=ON` is only needed when you want to use `--backend cuda`.
- CPU-only and Vulkan builds do not need CUDA enabled.
- `-DCMAKE_CUDA_ARCHITECTURES=89` is only an example for RTX 40-series GPUs.
- You should set `-DCMAKE_CUDA_ARCHITECTURES` to match your own GPU architecture.
- Common values:
  - `86` for many RTX 30-series GPUs
  - `89` for many RTX 40-series GPUs

If you are unsure, check your GPU model first instead of copying `89` blindly.

## Inference

### Basic CPU Inference

```bash
./build/examples/voxcpm_tts \
  --model-path ./models/quantized/voxcpm1.5-q8_0-audiovae-f16.gguf \
  --prompt-audio ./examples/tai_yi_xian_ren.wav \
  --prompt-text "对，这就是我，万人敬仰的太乙真人。" \
  --text "大家好，我现在正在大可奇奇体验AI科技。" \
  --output ./out.wav \
  --backend cpu \
  --threads 8
```

### Prompted Inference

```bash
./build/examples/voxcpm_tts \
  --model-path ./models/quantized/voxcpm1.5-q8_0-audiovae-f16.gguf \
  --prompt-audio ./examples/tai_yi_xian_ren.wav \
  --prompt-text "对，这就是我，万人敬仰的太乙真人。" \
  --text "大家好，我现在正在大可奇奇体验AI科技。" \
  --output ./out.wav \
  --backend cpu \
  --threads 8 \
  --inference-timesteps 10 \
  --cfg-value 2.0
```

### CUDA Inference

```bash
./build-cuda/examples/voxcpm_tts \
  --model-path ./models/quantized/voxcpm1.5-q8_0-audiovae-f16.gguf \
  --prompt-audio ./examples/tai_yi_xian_ren.wav \
  --prompt-text "对，这就是我，万人敬仰的太乙真人。" \
  --text "大家好，我现在正在大可奇奇体验AI科技。" \
  --output ./out.wav \
  --backend cuda \
  --threads 8 \
  --inference-timesteps 10 \
  --cfg-value 2.0
```

`voxcpm_tts` currently supports `--backend {cpu|cuda|vulkan|auto}`.

## OpenAI-Compatible TTS Server

`voxcpm-server` now exposes a single-port HTTP API for:

- `POST /v1/voices`
- `GET /v1/voices/{id}`
- `DELETE /v1/voices/{id}`
- `POST /v1/audio/speech`

### Full Endpoint List

#### `GET /healthz`

Health check.

Example response:

```json
{
  "status": "ok"
}
```

#### `POST /v1/voices`

Registers a reusable voice entry by uploading:

- multipart field `id`: required, unique voice id
- multipart field `text`: required, transcript for the reference audio
- multipart file `audio`: required, reference audio file

Success response: `201 Created`

Returned JSON fields:

- `id`
- `prompt_text`
- `prompt_audio_length`
- `sample_rate`
- `patch_size`
- `feat_dim`
- `created_at`
- `updated_at`

#### `GET /v1/voices/{id}`

Returns metadata for a previously registered voice id.

Success response: `200 OK`

Returned JSON fields:

- `id`
- `prompt_text`
- `prompt_audio_length`
- `sample_rate`
- `patch_size`
- `feat_dim`
- `created_at`
- `updated_at`

#### `DELETE /v1/voices/{id}`

Deletes a registered voice id.

Success response: `200 OK`

Example response:

```json
{
  "id": "taiyi",
  "deleted": true
}
```

#### `POST /v1/audio/speech`

Synthesizes speech from text using a registered voice id.

JSON request fields:

- `model`: required string, must match the configured `--model-name`
- `input`: required string, 1 to 4096 characters
- `voice`: required
  - string voice id, for example `"taiyi"`
  - or object form `{ "id": "taiyi" }`
- `response_format`: optional, defaults to `mp3`
  - supported values: `mp3`, `opus`, `flac`, `wav`, `pcm`
- `speed`: optional float, range `0.25` to `4.0`
- `stream_format`: optional, `audio` or `sse`
- `instructions`: accepted for compatibility, but non-empty values currently return an error

Response behavior:

- `stream_format=audio` or omitted:
  - returns raw audio bytes
  - `Content-Type` matches `response_format`
- `stream_format=sse`:
  - returns `text/event-stream`
  - each `audio.delta` event contains a self-contained chunk encoded with the requested `response_format`
  - emits:
    - `event: audio.delta`
    - `event: audio.completed`

Queue behavior:

- one synthesis request runs at a time per server process
- additional requests wait in a bounded queue controlled by `--max-queue`
- when the queue is full, the server returns `503`

Supported output formats:

- `mp3`: `audio/mpeg`
- `opus`: `audio/ogg; codecs=opus`
- `flac`: `audio/flac`
- `wav`: `audio/wav`
- `pcm`: `application/octet-stream`

Build-time support:

- `VOXCPM_ENABLE_MP3=ON|OFF`
  - toggles MP3 support
  - the runtime tries the native encoder first and falls back to `ffmpeg` if that path cannot initialize
- `VOXCPM_ENABLE_OPUS=ON|OFF`
  - toggles the Opus path, which uses an Ogg Opus `ffmpeg` fallback when enabled
  - if `ffmpeg` is unavailable when CMake configures the build, support is disabled and `/v1/audio/speech` returns `501` for `response_format=opus`

Example outputs:

- `speech.mp3`
- `speech.opus`

### Build

For CUDA deployment:

```bash
cmake -B build-cuda \
  -DVOXCPM_CUDA=ON \
  -DVOXCPM_BUILD_BENCHMARK=OFF \
  -DVOXCPM_BUILD_TESTS=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=89

cmake --build build-cuda -j8
```

This CUDA build is only required if you plan to launch the server with `--backend cuda`.
If you want `--backend cpu`, a normal CPU build is enough:

```bash
cmake -B build -DVOXCPM_BUILD_BENCHMARK=OFF -DVOXCPM_BUILD_TESTS=OFF
cmake --build build -j8
```

If you want to explicitly control audio encoder support, add:

```bash
cmake -B build -DVOXCPM_BUILD_BENCHMARK=OFF -DVOXCPM_BUILD_TESTS=OFF \
  -DVOXCPM_ENABLE_MP3=ON \
  -DVOXCPM_ENABLE_OPUS=ON
```

These options default to `ON`.

### Start The Server

The server auto-creates `--voice-dir` if it does not exist.

CUDA example:

```bash
./build-cuda/examples/voxcpm-server \
  --host 127.0.0.1 \
  --port 8080 \
  --model-path ./models/quantized/voxcpm1.5-q8_0-audiovae-f16.gguf \
  --model-name voxcpm-1.5 \
  --threads 8 \
  --backend cuda \
  --voice-dir ./runtime/voices \
  --max-queue 8 \
  --disable-auth
```

CPU example:

```bash
./build/examples/voxcpm-server \
  --host 127.0.0.1 \
  --port 8080 \
  --model-path ./models/quantized/voxcpm1.5-q8_0-audiovae-f16.gguf \
  --model-name voxcpm-1.5 \
  --threads 8 \
  --backend cpu \
  --voice-dir ./runtime/voices \
  --max-queue 8 \
  --disable-auth
```

### Register A Voice

```bash
curl -X POST http://127.0.0.1:8080/v1/voices \
  -F "id=taiyi" \
  -F "text=对，这就是我，万人敬仰的太乙真人。" \
  -F "audio=@./examples/tai_yi_xian_ren.wav"
```

Example response:

```json
{
  "created_at": "2026-03-18T11:32:51Z",
  "feat_dim": 64,
  "id": "taiyi",
  "patch_size": 4,
  "prompt_audio_length": 43,
  "prompt_text": "对，这就是我，万人敬仰的太乙真人。",
  "sample_rate": 44100,
  "updated_at": "2026-03-18T11:32:51Z"
}
```

### Synthesize Speech

```bash
curl -X POST http://127.0.0.1:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "voxcpm-1.5",
    "input": "大家好，我现在正在大可奇奇体验AI科技。",
    "voice": "taiyi",
    "response_format": "wav",
    "speed": 1.0,
    "stream_format": "audio"
  }' \
  --output ./voxcpm_taiyi.wav
```

MP3 example:

```bash
curl -X POST http://127.0.0.1:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "voxcpm-1.5",
    "input": "大家好，我现在正在大可奇奇体验AI科技。",
    "voice": "taiyi",
    "response_format": "mp3",
    "stream_format": "audio"
  }' \
  --output ./voxcpm_taiyi.mp3
```

Opus example:

```bash
curl -X POST http://127.0.0.1:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "voxcpm-1.5",
    "input": "大家好，我现在正在大可奇奇体验AI科技。",
    "voice": "taiyi",
    "response_format": "opus",
    "stream_format": "audio"
  }' \
  --output ./voxcpm_taiyi.opus
```

### Notes

- The current server accepts a voice id string such as `"taiyi"` in the `voice` field.
- `instructions` is accepted for compatibility but is not implemented in VoxCPM v1.
- `stream_format` supports `audio` and `sse`; `audio.delta` events carry the same encoded bytes and `format` value as the non-streaming path.
- `opus` is emitted as an Ogg Opus container, not a raw Opus packet stream.
- If you run the server under systemd, `CUDA_VISIBLE_DEVICES` only controls GPU visibility.
  It does not change `response_format`, MIME type selection, or which audio encoder is used.
- If you only want local offline inference, `examples/voxcpm_tts` is still the simplest entry point.
- When auth is enabled, every API route above requires `Authorization: Bearer <api-key>`.
- If a request asks for an encoder that was disabled at build time, the server returns `501` with a message such as `this build does not include opus encoder support`.
- Error responses use the shape:

```json
{
  "error": {
    "message": "Human-readable message",
    "type": "invalid_request_error",
    "code": "bad_request"
  }
}
```

## Benchmark Scripts

### Export Quantized Weights

```bash
./scripts/export_quantized_weights.sh
```

This exports:
- `Q4_K`
- `Q8_0`
- `F16`
- the corresponding `+AudioVAE-F16` variants
- `F32` baseline copy

and writes a manifest like `logs/quantized_weights_manifest_*.tsv`.

### Benchmark Exported Weights

CPU:

```bash
./scripts/benchmark_exported_weights.sh \
  --weights-file ./logs/quantized_weights_manifest_*.tsv \
  --backend cpu
```

CUDA:

```bash
./scripts/benchmark_exported_weights.sh \
  --weights-file ./logs/quantized_weights_manifest_*.tsv \
  --backend cuda
```

If `--weights-file` is omitted, the script will automatically pick the latest manifest under `logs/`.

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

## TODO

1. Add a WASM demo so users can try VoxCPM directly in the browser.
2. Continue improving inference performance. Based on the benchmark report from `https://github.com/DakeQQ/Text-to-Speech-TTS-ONNX`, there is still a noticeable gap between the current performance here and their reported results.
3. Expand server-side test coverage for OpenAI-compatible TTS and voice-management flows.
4. Continue closing the remaining quality and parity gaps in the preliminary `VoxCPM2` support.
5. Carry out a larger maintainability-oriented rewrite pass with `ClaudeCode Opus 4.6`.

## WASM Playground

A browser-oriented WASM playground scaffold now lives in:

- `wasm/`
- `web/packages/voxcpm-web/`
- `web/playground/`

See [docs/wasm_playground.md](docs/wasm_playground.md) for the Emscripten build flow and web demo setup.

## Preview

I also plan to create a dedicated GGML inference repository for `https://huggingface.co/fishaudio/s2-pro`.

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
