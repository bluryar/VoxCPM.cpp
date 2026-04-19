# VoxCPM.cpp CUDA Build Guide

This file is a focused walkthrough for building and running VoxCPM.cpp with the ggml CUDA backend.

## What this is for

- Build the project so `--backend cuda` works.
- Keep a separate build directory from CPU builds.
- Verify both the TTS example and the OpenAI-compatible server on CUDA.

## CUDA Service Spec

For the CUDA service path, current request budgeting is:

- `seq_len <= 256`: output-pool fast path, decode cap 128 steps
- `257-512`: fallback path, decode cap 96 steps
- `>512`: chunked prefill path, decode cap 64 steps

## Prerequisites

- NVIDIA GPU
- NVIDIA driver installed
- CUDA toolkit installed
- `cmake`, `make` or `ninja`, and a C++ toolchain
- `ffmpeg` if you want `response_format=opus`

If you are unsure about your GPU architecture, check it first. The common values in this repo are:

- `86` for many RTX 30-series GPUs
- `89` for many RTX 40-series GPUs

## Configure

Use a separate build directory:

```bash
cmake -B build-cuda \
  -DVOXCPM_CUDA=ON \
  -DVOXCPM_BUILD_BENCHMARK=OFF \
  -DVOXCPM_BUILD_TESTS=ON \
  -DVOXCPM_ENABLE_MP3=ON \
  -DVOXCPM_ENABLE_OPUS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89
```

Notes:

- `-DVOXCPM_CUDA=ON` enables the ggml CUDA backend.
- Keep `build-cuda` separate from `build` so CPU and CUDA artifacts do not overwrite each other.
- If your GPU is not an RTX 40-series card, replace `89` with the architecture that matches your card.
- `VOXCPM_ENABLE_MP3` and `VOXCPM_ENABLE_OPUS` are independent of CUDA. Leave them `ON` unless you want to disable those audio formats.

If you want a smaller configure step for a quick inference-only build, you can also disable tests:

```bash
cmake -B build-cuda \
  -DVOXCPM_CUDA=ON \
  -DVOXCPM_BUILD_BENCHMARK=OFF \
  -DVOXCPM_BUILD_TESTS=OFF \
  -DVOXCPM_ENABLE_MP3=ON \
  -DVOXCPM_ENABLE_OPUS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89
```

## Build

```bash
cmake --build build-cuda -j8
```

If the configure step was already done and you only changed source files, rerun the same build command.

## Quick Sanity Check

Confirm the CUDA-enabled binaries exist:

```bash
ls build-cuda/examples/voxcpm_tts
ls build-cuda/examples/voxcpm-server
```

## Run the TTS example on CUDA

Use the CUDA backend explicitly:

```bash
./build-cuda/examples/voxcpm_tts \
  --model-path ./models/voxcpm1.5-q4_k-audiovae-f16.gguf \
  --prompt-audio ./examples/tai_yi_xian_ren.wav \
  --prompt-text "对，这就是我，万人敬仰的太乙真人。" \
  --text "大家好，我现在正在大可奇奇体验AI科技。" \
  --output ./out.wav \
  --backend cuda \
  --threads 8
```

If you want to test a more typical generation path:

```bash
./build-cuda/examples/voxcpm_tts \
  --model-path ./models/voxcpm1.5-q4_k-audiovae-f16.gguf \
  --prompt-audio ./examples/tai_yi_xian_ren.wav \
  --prompt-text "对，这就是我，万人敬仰的太乙真人。" \
  --text "大家好，我现在正在大可奇奇体验AI科技。" \
  --output ./out.wav \
  --backend cuda \
  --threads 8 \
  --inference-timesteps 10 \
  --cfg-value 2.0
```

## Run the server on CUDA

Start the OpenAI-compatible server:

```bash
./build-cuda/examples/voxcpm-server \
  --host 127.0.0.1 \
  --port 8080 \
  --model-path ./models/voxcpm1.5-q4_k-audiovae-f16.gguf \
  --model-name voxcpm-1.5 \
  --threads 8 \
  --backend cuda \
  --voice-dir ./runtime/voices \
  --max-queue 8 \
  --output-sample-rate 24000 \
  --disable-auth
```

Register a voice:

```bash
curl -X POST http://127.0.0.1:8080/v1/voices \
  -F "id=taiyi" \
  -F "text=对，这就是我，万人敬仰的太乙真人。" \
  -F "audio=@./examples/tai_yi_xian_ren.wav"
```

Request `mp3`:

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

Request `opus`:

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

## If you need to rebuild from scratch

If your CUDA build directory got stale, it is often easiest to delete only that build tree and reconfigure:

```bash
rm -rf build-cuda
cmake -B build-cuda \
  -DVOXCPM_CUDA=ON \
  -DVOXCPM_BUILD_BENCHMARK=OFF \
  -DVOXCPM_BUILD_TESTS=ON \
  -DVOXCPM_ENABLE_MP3=ON \
  -DVOXCPM_ENABLE_OPUS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-cuda -j8
```

## Common problems

- Build config finishes but CUDA binaries do not appear:
  - Make sure you ran `cmake --build build-cuda`, not `build`.
- `--backend cuda` fails at runtime:
  - Check that the NVIDIA driver and CUDA toolkit match your system.
  - Make sure the GPU architecture passed to `CMAKE_CUDA_ARCHITECTURES` matches your card.
- `response_format=opus` returns 501:
  - `ffmpeg` was not found when CMake configured the build, or the Opus path was disabled.
- `response_format=mp3` returns 501:
  - Reconfigure with `-DVOXCPM_ENABLE_MP3=ON`.
- `CUDA_VISIBLE_DEVICES` in your systemd unit only selects which GPU the process can see.
  - It does not affect `response_format`, `mp3`/`opus` encoding, or OpenAI-compatible audio MIME selection.
- If an OpenAI-compatible client assumes 24 kHz PCM playback, start `voxcpm-server` with `--output-sample-rate 24000`.
  - `pcm` responses have no embedded sample-rate header, so the server-side output rate must match the client expectation.
  - The same override also applies to `wav`, `mp3`, and `opus`.

## Minimal checklist

1. Configure `build-cuda` with `-DVOXCPM_CUDA=ON`.
2. Build with `cmake --build build-cuda -j8`.
3. Run `voxcpm_tts` with `--backend cuda`.
4. Run `voxcpm-server` with `--backend cuda`.
5. Test `mp3` and `opus` requests.
