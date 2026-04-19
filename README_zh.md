# VoxCPM.cpp

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

基于 `ggml` 构建的 VoxCPM 模型独立 C++ 推理项目。

- **GGUF 权重**：https://huggingface.co/bluryar/VoxCPM-GGUF
- VoxCPM 官方仓库：https://github.com/OpenBMB/VoxCPM

[English](README.md)

## 状态

此目录现作为 `VoxCPM.cpp` 独立仓库的根目录。

- `third_party/ggml` 作为供应商子树维护。
- `third_party/json`、`third_party/llama.cpp`、`third_party/whisper.cpp` 和 `third_party/SenseVoice.cpp` 仅作为本地参考，被仓库忽略。
- `CMakeLists.txt` 已支持在 `third_party/json` 缺失时通过 `FetchContent` 下载 `nlohmann_json`。
- `VoxCPM2` 现已进入初步支持状态。当前 C++ runtime 已经可以加载导出的 `VoxCPM2` GGUF 权重、完成端到端推理、打通新的 reference 模式链路，并通过 `AudioVAE V2` 路径输出 48kHz 音频，但质量和数值对齐仍在持续验证中。

## 重构预告

仓库接下来会推进一轮更完整的 Torch-to-GGML runtime 重构，设计方向见以下两份文档：

- [docs/voxcpm_torch_to_ggml_complete_refactor_cookbook_zh.md](docs/voxcpm_torch_to_ggml_complete_refactor_cookbook_zh.md)
- [docs/torch_to_ggml_migration_guide_zh.md](docs/torch_to_ggml_migration_guide_zh.md)

这次重构的原因主要有几点：

- 当前代码已经可以运行，但整体实现仍然带有较强的“把 PyTorch 模块直接拆成若干 C++ 片段”的迁移痕迹。
- 这种写法适合快速跑通模型，但会让 shape 契约、对象所有权、持久状态、graph 生命周期以及 backend 放置策略逐渐缠在一起，后续维护和排错成本越来越高。
- 热路径里如果频繁出现 `tensor_get -> std::vector -> tensor_set` 这类 host/backend 往返，随着模型变复杂、后端变多，性能和结构问题都会被进一步放大。

这轮重构不是表层整理，而是准备把 `VoxCPM.cpp` 推向更成熟的 `ggml` runtime 形态，包括：

- 先固定 GGUF 契约和模块级输入输出契约
- 用共享 `WeightStore` 和 backend-aware 的 loader/runtime 骨架统一权重与执行路径
- 明确拆分权重、持久 state、compute 内存和 output buffer
- 用真实重建条件驱动 graph cache，而不是只靠 shape 猜测
- 让模块热路径尽量直接传递 backend-resident tensor 或 state handle，减少无意义的数据回传

一句话说，这次重构的目标不是继续堆补丁，而是把项目从“host 侧模块翻译式实现”推进到“契约先行、后端感知、可验证可扩展”的 runtime 架构。

为了加快这项工作，我也计划使用 `ClaudeCode Opus 4.6` 发起一轮更大规模的代码重写，重点改善代码可维护性、梳理 runtime 与模块边界，并减少模型初期跑通阶段遗留下来的胶水层代码。

## 构建

### CPU 构建

```bash
cmake -B build
cmake --build build
```

### CUDA 构建

只有当你打算使用 `--backend cuda` 时，才需要在配置阶段启用 ggml 的 CUDA backend：

```bash
cmake -B build-cuda \
  -DVOXCPM_CUDA=ON \
  -DVOXCPM_BUILD_BENCHMARK=OFF \
  -DVOXCPM_BUILD_TESTS=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-cuda
```

如果你希望同时保留 CPU 和 CUDA 两套构建，建议使用不同的构建目录，例如 `build` 和 `build-cuda`。

重要提醒：

- `-DVOXCPM_CUDA=ON` 只是在你要使用 `--backend cuda` 时才需要。
- 如果你跑的是 `--backend cpu` 或其他非 CUDA 后端，不需要开启 CUDA 构建。
- `-DCMAKE_CUDA_ARCHITECTURES=89` 只是 RTX 40 系列常见示例，不应盲目照抄。
- 你应该根据自己的显卡架构设置 `-DCMAKE_CUDA_ARCHITECTURES`。
- 常见取值：
  - `86`：很多 RTX 30 系列
  - `89`：很多 RTX 40 系列

如果不确定，先确认自己的显卡型号，再决定这个值。

## 推理用法

### 基础 CPU 推理

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

### 带 Prompt 的推理

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

### CUDA 推理

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

`voxcpm_tts` 当前支持 `--backend {cpu|cuda|vulkan|auto}`。

## OpenAI 兼容 TTS 服务

`voxcpm-server` 现在提供单端口 HTTP 服务，同时支持：

- `POST /v1/voices`
- `GET /v1/voices/{id}`
- `DELETE /v1/voices/{id}`
- `POST /v1/audio/speech`

### 完整接口列表

#### `GET /healthz`

健康检查接口。

示例返回：

```json
{
  "status": "ok"
}
```

#### `POST /v1/voices`

注册可复用的 voice，上传内容为：

- multipart 字段 `id`：必填，唯一 voice id
- multipart 字段 `text`：必填，参考音频对应文本
- multipart 文件 `audio`：必填，参考音频文件

成功返回：`201 Created`

返回 JSON 字段：

- `id`
- `prompt_text`
- `prompt_audio_length`
- `sample_rate`
- `patch_size`
- `feat_dim`
- `created_at`
- `updated_at`

#### `GET /v1/voices/{id}`

查询已注册 voice 的元数据。

成功返回：`200 OK`

返回 JSON 字段：

- `id`
- `prompt_text`
- `prompt_audio_length`
- `sample_rate`
- `patch_size`
- `feat_dim`
- `created_at`
- `updated_at`

#### `DELETE /v1/voices/{id}`

删除已注册的 voice。

成功返回：`200 OK`

示例返回：

```json
{
  "id": "taiyi",
  "deleted": true
}
```

#### `POST /v1/audio/speech`

使用已注册的 voice id 从文本合成语音。

JSON 请求字段：

- `model`：必填字符串，必须和启动参数 `--model-name` 一致
- `input`：必填字符串，长度范围 `1` 到 `4096`
- `voice`：必填
  - 可以直接传字符串 voice id，例如 `"taiyi"`
  - 也可以传对象形式 `{ "id": "taiyi" }`
- `response_format`：可选，默认 `mp3`
  - 支持 `mp3`、`opus`、`flac`、`wav`、`pcm`
- `speed`：可选浮点数，范围 `0.25` 到 `4.0`
- `stream_format`：可选，支持 `audio` 或 `sse`
- `instructions`：为兼容接口而保留，但只要传非空值目前就会报错

返回行为：

- `stream_format=audio` 或省略：
  - 直接返回音频二进制
  - `Content-Type` 与 `response_format` 对应
- `stream_format=sse`：
  - 返回 `text/event-stream`
  - 每个 `audio.delta` 事件都包含一个按请求格式编码好的独立音频片段
  - 发送：
    - `event: audio.delta`
    - `event: audio.completed`

服务端输出采样率：

- 启动 `voxcpm-server` 时可以加 `--output-sample-rate HZ`，在编码前先把合成音频重采样到指定采样率
- 如果省略该参数，服务器会使用模型 AudioVAE 的输出采样率
- 对 OpenAI-compatible 的 `pcm` 返回，若客户端按 24 kHz 解释播放，请显式设置 `--output-sample-rate 24000`
- 这个覆盖同样适用于 `wav`、`mp3` 和 `opus`

队列行为：

- 单个 server 进程同一时间只处理一个合成请求
- 额外请求进入等待队列，长度由 `--max-queue` 控制
- 队列满时返回 `503`

输出格式：

- `mp3`：`audio/mpeg`
- `opus`：`audio/ogg; codecs=opus`
- `flac`：`audio/flac`
- `wav`：`audio/wav`
- `pcm`：`application/octet-stream`

构建选项：

- `VOXCPM_ENABLE_MP3=ON|OFF`
  - 控制 MP3 支持
  - 运行时会优先尝试原生编码器，初始化失败时再回退到 `ffmpeg`
- `VOXCPM_ENABLE_OPUS=ON|OFF`
  - 控制 Opus 路径，启用后会使用 Ogg Opus 的 `ffmpeg` fallback
  - 如果配置 CMake 时找不到 `ffmpeg`，会自动关闭该支持，`response_format=opus` 请求返回 `501`

输出示例：

- `speech.mp3`
- `speech.opus`

### 构建

CUDA 部署推荐这样构建：

```bash
cmake -B build-cuda \
  -DVOXCPM_CUDA=ON \
  -DVOXCPM_BUILD_BENCHMARK=OFF \
  -DVOXCPM_BUILD_TESTS=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=89

cmake --build build-cuda -j8
```

这套 CUDA 构建只在你准备以 `--backend cuda` 启动服务时才需要。
如果你只想用 `--backend cpu`，普通 CPU 构建就够了：

```bash
cmake -B build -DVOXCPM_BUILD_BENCHMARK=OFF -DVOXCPM_BUILD_TESTS=OFF
cmake --build build -j8
```

如果想显式控制音频编码支持，可以再加上：

```bash
cmake -B build -DVOXCPM_BUILD_BENCHMARK=OFF -DVOXCPM_BUILD_TESTS=OFF \
  -DVOXCPM_ENABLE_MP3=ON \
  -DVOXCPM_ENABLE_OPUS=ON
```

这两个选项默认都是 `ON`。

### 启动服务

服务会在启动时自动创建 `--voice-dir`，目录不存在也没关系。

CUDA 示例：

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
  --output-sample-rate 24000 \
  --disable-auth
```

CPU 示例：

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
  --output-sample-rate 24000 \
  --disable-auth
```

### 注册 voice

```bash
curl -X POST http://127.0.0.1:8080/v1/voices \
  -F "id=taiyi" \
  -F "text=对，这就是我，万人敬仰的太乙真人。" \
  -F "audio=@./examples/tai_yi_xian_ren.wav"
```

示例返回：

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

### 合成语音

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

MP3 示例：

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

Opus 示例：

```bash
curl -X POST http://127.0.0.1:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "voxcpm-1.5",
    "input": "大家好，我现在正在大可奇奇体验AI科技。",
    "voice": "taiyi",
    "response_format": "opus",
    "stream_format": "sse"
  }' \
  --output ./voxcpm_taiyi.opus
```

### 说明

- `voice` 字段当前直接传已注册的 voice id，例如 `"taiyi"`。
- `instructions` 为兼容 OpenAI 请求体而保留，但 VoxCPM v1 还未实现。
- `stream_format` 支持 `audio` 和 `sse`；`audio.delta` 的 `format` 字段和实际字节内容会与非流式路径一致。
- `opus` 输出是 Ogg Opus 容器，不是裸 Opus packet。
- 如果你只需要本地离线推理，`examples/voxcpm_tts` 仍然是最简单的入口。
- 如果启用了鉴权，上面所有接口都需要带 `Authorization: Bearer <api-key>`。
- 如果请求了构建时禁用的编码器，服务器会返回 `501`，例如 `this build does not include opus encoder support`。
- 错误返回格式如下：

```json
{
  "error": {
    "message": "可读错误信息",
    "type": "invalid_request_error",
    "code": "bad_request"
  }
}
```

## Benchmark 脚本

### 导出量化权重

```bash
./scripts/export_quantized_weights.sh
```

这个脚本会导出：
- `Q4_K`
- `Q8_0`
- `F16`
- 对应的 `+AudioVAE-F16` 变体
- `F32` baseline 拷贝

并生成类似 `logs/quantized_weights_manifest_*.tsv` 的 manifest 文件。

### 对导出权重做 Benchmark

CPU：

```bash
./scripts/benchmark_exported_weights.sh \
  --weights-file ./logs/quantized_weights_manifest_*.tsv \
  --backend cpu
```

CUDA：

```bash
./scripts/benchmark_exported_weights.sh \
  --weights-file ./logs/quantized_weights_manifest_*.tsv \
  --backend cuda
```

如果不传 `--weights-file`，脚本会自动选取 `logs/` 下最新的 manifest。

## 测试

```bash
cd build
ctest --output-on-failure
```

测试模型/trace 路径配置和开源协作说明请见 [docs/TEST_SETUP.md](docs/TEST_SETUP.md)。

## ggml 维护

项目保持当前 `ggml` 导入和补丁流程的本地溯源：

- 上游：`https://github.com/ggerganov/ggml.git`
- 仓库拆分前的本地基础提交：`4773cde162a55f0d10a6a6d7c2ea4378e30e0b01`
- 当前本地补丁：`src/ggml-vulkan/ggml-vulkan.cpp` 中的 Vulkan 头文件兼容性调整

详见 `docs/ggml_subtree_maintenance_strategy.md`。

## TODO

1. 准备添加一个 WASM 用例，让用户可以直接在网页上试用 VoxCPM 模型。
2. 继续优化推理性能。根据 `https://github.com/DakeQQ/Text-to-Speech-TTS-ONNX` 的报告，我们和它们当前展示的性能表现相比仍然有一段差距。
3. 继续补全 OpenAI 兼容 TTS 服务和 voice 管理链路的测试覆盖。
4. 继续收敛 `VoxCPM2` 初步支持中的质量问题和数值对齐差距。
5. 使用 `ClaudeCode Opus 4.6` 推进一轮以可维护性为目标的大重写。

## 预告

接下来我也计划为 `https://huggingface.co/fishaudio/s2-pro` 单独创建一个 GGML 推理仓库。

## 基准测试

### 模型大小与压缩比

| Model | Quant | Size (MB) | Compression |
|-------|-------|-----------|-------------|
| voxcpm1.5 | F32 | 3392 | 1.00x (基准) |
| voxcpm1.5 | F16 | 1700 | 1.99x |
| voxcpm1.5 | Q8_0 | 942 | 3.60x |
| voxcpm1.5 | Q4_K | 582 | 5.82x |
| voxcpm-0.5b | F32 | 2779 | 1.00x (基准) |
| voxcpm-0.5b | F16 | 1394 | 1.99x |
| voxcpm-0.5b | Q8_0 | 766 | 3.62x |
| voxcpm-0.5b | Q4_K | 477 | 5.82x |

### CPU 推理性能 (RTF - 越低越好)

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

### CUDA 推理性能 (RTF - 越低越好)

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

**RTF 定义：**
- **Model Only**：纯模型推理（prefill + decode loop），不含 AudioVAE
- **Without Encode**：模型 + AudioVAE decode（离线预计算 prompt 特征的部署场景）
- **Full Pipeline**：端到端完整流程，包含 AudioVAE encode + 模型 + decode

### 关键发现

#### CPU

1. **CPU 最优配置现在取决于模型和指标**：`voxcpm1.5 Q4_K+AudioVAE-F16` 在 model-only 和 without-encode 指标上最好，`voxcpm1.5 Q8_0` 在完整流水线指标上最好，而 `voxcpm-0.5b Q4_K` 仍然是整体最稳妥的 CPU 选择。
2. **1.5B 在 CPU 上明显受益于 AudioVAE-F16**：`Q4_K+AudioVAE-F16` 在 `voxcpm1.5` 上拿到了最好的 `Model Only` 和 `Without Encode` RTF，而 `Q8_0` 拿到了最好的完整流水线 RTF。
3. **0.5B 的 CPU 最优仍然是 Q4_K**：`voxcpm-0.5b Q4_K` 的整体 CPU RTF 最好，`Q8_0+AudioVAE-F16` 在完整流水线指标上非常接近。
4. **这台 CPU 上 F32 最慢**：无论是 `voxcpm1.5` 还是 `voxcpm-0.5b`，F32 baseline 都是最慢的 CPU 配置。

#### CUDA

1. **CUDA 明显快于 CPU**：在本轮测试中，完整流水线 RTF 从 CPU 的 `3.83-15.02` 下降到 CUDA 的 `0.55-0.69`。
2. **CUDA 下最佳配置取决于评价指标**：对 `voxcpm1.5`，`Q8_0+AudioVAE-F16` 的 RTF 最好，而 `F16+AudioVAE-F16` 的总耗时最短；对 `voxcpm-0.5b`，`Q4_K` 的完整流水线 RTF 最好，而 `Q8_0` 的总耗时最短。
3. **CUDA 不再明显偏爱 Q4_K**：和 CPU 不同，Q4_K 在 CUDA 上并不总是最快，`Q8_0` 和 `F16` 经常同样有竞争力，甚至更好。
4. **AudioVAE F16 在 CUDA 上有帮助**：把 AudioVAE 强制导出为 `F16` 后，多组 CUDA 测试结果变好，尤其是 `voxcpm1.5 Q8_0` 和 `voxcpm-0.5b Q8_0`。

### 部署建议

| 场景 | 推荐配置 |
|------|---------|
| 生产部署 | **voxcpm-0.5b Q4_K** (477 MB, RTF 3.609) |
| 平衡精度 | **voxcpm1.5 Q8_0** (942 MB, RTF 4.291) |
| 1.5B 离线 prompt 场景 | voxcpm1.5 Q4_K+AudioVAE-F16 (647 MB, Without Encode RTF 2.848) |
| 最高精度基线 | voxcpm1.5 F32 (3392 MB, RTF 7.494) |

### CUDA 部署建议

| 场景 | 推荐配置 |
|------|---------|
| 最低完整流水线 RTF | **voxcpm-0.5b Q4_K** (477 MB, RTF 0.550) |
| 1.5B 最佳延迟/RTF 平衡 | **voxcpm1.5 Q8_0+AudioVAE-F16** (984 MB, RTF 0.559) |
| 1.5B 较小且适合 CUDA 的模型 | voxcpm1.5 Q4_K+AudioVAE-F16 (647 MB, RTF 0.596) |
| 最高精度基线 | voxcpm1.5 F32 (3392 MB, RTF 0.686) |

**CPU 测试环境：**
- CPU：12th Gen Intel(R) Core(TM) i5-12600K
- 线程：8
- 后端：CPU
- 基准结果来源：`logs/benchmark_summary_cpu_20260318_092142.txt`

**CUDA 测试环境：**
- 后端：CUDA
- GPU：NVIDIA GeForce RTX 4060 Ti
- CUDA 设备：`CUDA0`
- Compute capability：8.9
- CUDA VMM：yes
- 主机 CPU：12th Gen Intel(R) Core(TM) i5-12600K
- 线程：8
- Inference timesteps：10
- CFG value：2.0
- 基准结果来源：`logs/benchmark_summary_cuda_20260318_092028.txt`
