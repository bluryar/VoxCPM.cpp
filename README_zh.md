# VoxCPM.cpp

基于 `ggml` 构建的 VoxCPM 模型独立 C++ 推理项目。

VoxCPM 官方仓库：https://github.com/OpenBMB/VoxCPM

[English](README.md)

## 状态

此目录现作为 `VoxCPM.cpp` 独立仓库的根目录。

- `third_party/ggml` 作为供应商子树维护。
- `third_party/json`、`third_party/llama.cpp`、`third_party/whisper.cpp` 和 `third_party/SenseVoice.cpp` 仅作为本地参考，被仓库忽略。
- `CMakeLists.txt` 已支持在 `third_party/json` 缺失时通过 `FetchContent` 下载 `nlohmann_json`。

## 构建

```bash
cmake -B build
cmake --build build
```

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

### 推理性能 (RTF - 越低越好)

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

**RTF 定义：**
- **Model Only**：纯模型推理（prefill + decode loop），不含 AudioVAE
- **Without Encode**：模型 + AudioVAE decode（离线预计算 prompt 特征的部署场景）
- **Full Pipeline**：端到端完整流程，包含 AudioVAE encode + 模型 + decode

### 关键发现

1. **Q4_K 表现最佳**：比 F32 快 27-37%，同时节省 83% 存储空间
2. **F16 反而最慢**：比 F32 慢 2.6-2.8 倍，可能是内存带宽或 SIMD 优化不足
3. **Q8_0 vs F32**：Q8_0 性能接近或略优于 F32，同时节省 72% 空间
4. **模型对比**：0.5B 比 1.5B 快约 30-40%，体积减少 18%

### 部署建议

| 场景 | 推荐配置 |
|------|---------|
| 生产部署 | **voxcpm-0.5b Q4_K** (477 MB, RTF 2.72) |
| 平衡精度 | voxcpm1.5 Q4_K (582 MB, RTF 3.91) |
| 最高精度 | voxcpm1.5 F32 (3392 MB, RTF 4.66) |
| 避免使用 | F16 量化（最慢且无优势）|

**测试环境：**
- 平台：Orange Pi 6 Plus
- SoC：CIX P1 CD8160
- CPU：12 核 (2x A720 @ 2.6GHz + 6x A720 @ 2.5GHz + 4x A520 @ 1.8GHz)
- 架构：aarch64，支持 SVE、SVE2、BF16、i8mm
- 线程：8
- 后端：CPU
