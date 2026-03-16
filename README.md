# VoxCPM.cpp

Standalone C++ inference project for VoxCPM models built on top of `ggml`.

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

## ggml Maintenance

The project keeps local provenance for the current `ggml` import and patch flow:

- upstream: `https://github.com/ggerganov/ggml.git`
- current local base commit before repository split: `4773cde162a55f0d10a6a6d7c2ea4378e30e0b01`
- current local patch: Vulkan header compatibility adjustment in `src/ggml-vulkan/ggml-vulkan.cpp`

See `docs/ggml_subtree_maintenance_strategy.md` for the longer-term maintenance approach.
