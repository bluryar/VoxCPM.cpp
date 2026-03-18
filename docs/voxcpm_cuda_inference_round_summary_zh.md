# VoxCPM CUDA 推理修复阶段总结

## 1. 背景
本轮工作的目标有两部分：

1. 让 `VoxCPM.cpp` 能够真正支持 `--backend cuda`。
2. 修复 CUDA 路径上的正确性和性能问题，尤其是：
   - `MiniCPM` prefill 阶段的 KV cache 写入崩溃
   - `AudioVAE` 编解码在 CUDA 上的算子兼容性问题
   - `AudioVAE decode` 的严重性能问题

这份文档基于当前工作区 `git status` / `git diff` 的结果，总结本轮已经落地、已经验证、以及尝试后回退的内容。

## 2. 当前工作区状态
截至当前，`git status --short` 显示：

### 2.1 与本轮 CUDA 修复直接相关的文件
- `CMakeLists.txt`
- `examples/voxcpm_imatrix.cpp`
- `examples/voxcpm_tts.cpp`
- `include/voxcpm/audio-vae.h`
- `src/audio-vae.cpp`
- `src/backend.cpp`
- `src/minicpm.cpp`
- `tests/CMakeLists.txt`
- `tests/test_audio_vae.cpp`
- `third_party/ggml/src/ggml-cuda/conv-transpose-1d.cu`
- `third_party/ggml/src/ggml-cuda/conv-transpose-1d.cuh`
- `third_party/ggml/src/ggml-cuda/im2col.cu`
- `third_party/ggml/src/ggml.c`
- `tests/test_cuda_backend.cpp`（新增）

### 2.2 本轮顺手带上的轻微代码整理
- `src/voxcpm.cpp`
  - 只有一处小改动：`state.residual_hidden = std::move(residual_hidden);`
  - 不属于 CUDA 主修复核心，但保留无害。

### 2.3 当前工作区中的其他改动 / 非本轮核心改动
- `scripts/export_quantized_weights.sh`
  - 将若干 `bc` 改成了 `awk` 计算格式化输出。
  - 这不是 CUDA 修复链路的一部分，应视为独立改动。

### 2.4 当前工作区中的生成产物
- `out5.wav`
- `voxcpm_stream_single_final.wav`

这两个是运行生成物，不属于源码变更。

## 3. 已落地的核心改动

### 3.1 CMake 增加 CUDA 开关
文件：`CMakeLists.txt`

新增：
- `option(VOXCPM_CUDA "Enable CUDA backend support through ggml" OFF)`
- `set(GGML_CUDA ${VOXCPM_CUDA} CACHE BOOL ... FORCE)`

效果：
- 可以通过 `-DVOXCPM_CUDA=ON` 打开 ggml 的 CUDA backend 编译。

### 3.2 CLI 支持 `--backend cuda`
文件：
- `examples/voxcpm_tts.cpp`
- `src/backend.cpp`

改动：
- CLI 参数解析新增 `cuda`
- 帮助信息同步更新
- backend 初始化逻辑新增 CUDA 设备发现与选择
- `BackendType::Auto` 已调整为优先选 CUDA，再选 Vulkan，再退回 CPU

效果：
- 现在可以直接使用：
  - `./build/examples/voxcpm_tts --backend cuda ...`

### 3.3 修复 MiniCPM KV 写入在 CUDA 上的 `sum.cu` 崩溃
文件：`src/minicpm.cpp`

问题：
- `ggml_sum()` 直接作用在 view-backed tensor 上。
- CUDA `sum` 要求输入是 contiguously allocated tensor。

修复：
- 将：
  - `ggml_sum(ctx, k_write)`
  - `ggml_sum(ctx, v_write)`
- 改为：
  - `ggml_sum(ctx, ggml_cont(ctx, k_write))`
  - `ggml_sum(ctx, ggml_cont(ctx, v_write))`

效果：
- prefill 阶段不再在：
  - `third_party/ggml/src/ggml-cuda/sum.cu`
  - `GGML_ASSERT(ggml_is_contiguously_allocated(src0))`
 处崩溃。

### 3.4 修复 CUDA `im2col` 长宽度 launch 失败
文件：`third_party/ggml/src/ggml-cuda/im2col.cu`

问题：
- 原实现直接把 `OW` 放进 `grid.y`。
- 长音频 / 长宽度场景下会超过 CUDA grid 限制，报：
  - `CUDA error: invalid configuration argument`

修复：
- 重写 `im2col` / `im2col_3d` 的 launch 索引
- 将 `(OW, N*OH)` 和 `(OW, N*OD*OH)` 折叠到合法 grid 范围

效果：
- 修复了 `AudioVAE encode` 期间在 `IM2COL` 上的 CUDA launch 错误。

### 3.5 AudioVAE 增加 backend-aware depthwise 路径
文件：
- `include/voxcpm/audio-vae.h`
- `src/audio-vae.cpp`
- `examples/voxcpm_tts.cpp`
- `examples/voxcpm_imatrix.cpp`

改动：
- `AudioVAE::encode()` / `decode()` 改为显式接收 `const VoxCPMBackend &`
- `AudioVAE::causal_conv1d_dw()` 在 `BackendType::CUDA` 时改走原生 `ggml_conv_1d_dw`
- CPU / 其他 backend 继续保留 `ggml_map_custom3()` 的旧实现

原因：
- 旧的 depthwise conv 是 CPU 自定义算子，CUDA backend 无法直接执行。
- 为了不影响 CPU trace 和既有行为，只对 CUDA 单独分支。

### 3.6 修复 `ggml_conv_1d_dw()` 的 `im2col` 类型选择
文件：`third_party/ggml/src/ggml.c`

问题：
- `ggml_conv_1d_dw()` 原来强制把 `im2col` 输出设成 `GGML_TYPE_F16`
- 这对 AudioVAE CUDA 路径并不稳

修复：
- 当输入或权重是 `F32` 时，`im2col` 输出改用 `F32`
- 仅在全链路安全时保留 `F16`

效果：
- 使 CUDA depthwise 路径在 AudioVAE 上更稳定。

### 3.7 修复 CUDA `conv_transpose_1d` 的严重性能问题
文件：
- `third_party/ggml/src/ggml-cuda/conv-transpose-1d.cu`
- `third_party/ggml/src/ggml-cuda/conv-transpose-1d.cuh`

问题：
- 旧 CUDA kernel 是“每个输出元素扫描整段输入”的 naive 实现
- 在 AudioVAE decode 上表现为：
  - GPU 利用率打满
  - 单个 CPU 核心打满
  - `Decoding waveform from N latent patches...` 阶段异常慢

修复：
- 将 CUDA `conv_transpose_1d` 改成按输入块驱动、使用 shared memory 的结构化实现
- 同时支持 `F16 weight + F32 input`
- 将 block size 调整为 `128`

效果：
- 用户实测：
  - 之前 `AudioVAE decode` 明显卡住
  - 修复后：`AudioVAE decode: 0.286s`
- 用户一次实际运行结果：
  - `AudioVAE encode: 0.690s`
  - `Model inference: 2.222s`
  - `AudioVAE decode: 0.286s`
  - `Total: 3.197s`
  - `Full pipeline RTF: 1.052`

这说明：
- AudioVAE decode 的“异常慢”问题已经被修复
- 新瓶颈已经转移到 `Model inference`（prefill + decode loop）

## 4. 新增测试

### 4.1 新增 CUDA backend 回归测试
文件：`tests/test_cuda_backend.cpp`

新增覆盖：
- `ggml_cpy(view) -> ggml_cont -> ggml_sum` 在 CUDA 上可运行
- 长宽度 `im2col` 在 CUDA 上可运行
- `conv_transpose_1d` 在 `F16 weight + F32 input` 下与 reference 对齐

配套改动：
- `tests/CMakeLists.txt` 新增 `test_cuda_backend`

### 4.2 AudioVAE CUDA smoke 测试
文件：`tests/test_audio_vae.cpp`

新增：
- `AudioVAE encode CUDA smoke`
- `AudioVAE decode CUDA smoke`

说明：
- 这些测试依赖 trace / fixture / model 文件
- 在当前缺少依赖的环境里会 skip

## 5. 已尝试但已回退的改动

### 5.1 decode loop 融合大图尝试
文件涉及：`src/voxcpm.cpp`（后已回退）

尝试内容：
- 试图把以下步骤合并成单个 decode fused graph：
  - `decode_front_half`
  - `stop_predictor`
  - `patch -> locenc -> lm_embed`
  - `base_lm_step`
  - `residual_lm_step`

目标：
- 减少 decode loop 的 host 同步
- 减少 `tensor_get / tensor_set` 往返
- 降低 CPU 单核驱动小图的开销

结果：
- 该尝试引入了 stop 判定异常
- 用户反馈“无法正确 stop”
- 即使将 stop predictor 单独拆回，也未恢复正确行为
- 因此这条 decode 融合改动已整体回退

当前状态：
- `src/voxcpm.cpp` 已恢复为原来的逐段 decode 路径
- 没有保留 fused decode step 逻辑

结论：
- decode loop 的大图融合目前不是安全优化路径
- 如果后续继续优化 `Model inference`，应采用更保守的切入方式

## 6. 当前结论

### 6.1 已确认解决的问题
- CUDA backend 接线完成，可通过 `--backend cuda` 运行
- `MiniCPM` prefill 阶段的 `sum.cu` 崩溃已修复
- `AudioVAE encode` 的 `IM2COL invalid configuration argument` 已修复
- `AudioVAE decode` 的严重性能问题已修复

### 6.2 当前仍需继续优化的问题
- 主模型推理仍是主要瓶颈：
  - `prefill`
  - `decode loop`
- 这部分的优化不能以破坏 stop / decode 语义为代价

### 6.3 后续推荐方向
建议下一轮从以下更保守的方向继续：

1. 优化 `prefill` 图构建与显存分配开销
2. 优化 decode loop 中稳定子图的缓存与复用
3. 避免把 stop 判定和 decode state 更新边界跨阶段融合
4. 优先做 profiling / 分阶段计时，再决定是否进一步合图

## 7. 当前建议提交边界
如果要按“已验证有效”来分批提交，建议：

### 建议保留并提交
- `CMakeLists.txt`
- `src/backend.cpp`
- `examples/voxcpm_tts.cpp`
- `examples/voxcpm_imatrix.cpp`
- `include/voxcpm/audio-vae.h`
- `src/audio-vae.cpp`
- `src/minicpm.cpp`
- `third_party/ggml/src/ggml.c`
- `third_party/ggml/src/ggml-cuda/im2col.cu`
- `third_party/ggml/src/ggml-cuda/conv-transpose-1d.cu`
- `third_party/ggml/src/ggml-cuda/conv-transpose-1d.cuh`
- `tests/CMakeLists.txt`
- `tests/test_audio_vae.cpp`
- `tests/test_cuda_backend.cpp`

### 可单独评估是否提交
- `src/voxcpm.cpp`
  - 当前仅剩一个 `std::move` 小优化

### 与本轮 CUDA 主线无关，建议单独处理
- `scripts/export_quantized_weights.sh`
- `out5.wav`
- `voxcpm_stream_single_final.wav`
