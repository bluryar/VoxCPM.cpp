# VoxCPM.cpp CUDA 优化总结

本文档总结本轮围绕 `voxcpm_tts --backend cuda` 的优化工作，重点记录：

- 哪些优化最终保留
- 哪些优化尝试过但已经回退
- 用户侧 RTX 4060 Ti benchmark 的实际结论
- 参考项目 `third_party/Text-to-Speech-TTS-ONNX` 与 `third_party/VoxCPM` 分别提供了什么价值

这版文档以“当前稳定代码状态”为准。

## 目标

本轮优化的目标不是单纯追求“更低量化”或“更低理论 FLOPs”，而是尽量把 `VoxCPM.cpp` 的执行形态向 ONNX/TensorRT 风格的快路径靠拢，同时保持：

- CUDA 默认路径稳定
- stop token 语义不回退
- `--inference-timesteps` 继续保持可调
- benchmark 口径可重复
- 通用 GGUF 模型仍可加载

## 参考项目如何使用

本轮主要参考了两个目录：

- `third_party/Text-to-Speech-TTS-ONNX`
- `third_party/VoxCPM`

二者作用不同。

### 1. `third_party/Text-to-Speech-TTS-ONNX`

这是性能方向上最有指导意义的参考。

从 `third_party/Text-to-Speech-TTS-ONNX/VoxCPM/Export_VoxCPM_ONNX.py` 可以提炼出几条关键思路：

1. `LocDiT` 的 CFG cond/uncond 两路会尽量合并，一次 decoder forward 完成。
2. `qkv`、`gate_up` 这类投影会提前融合。
3. `t / dt` 等只依赖 `timesteps` 的时间常量会预先展开，而不是在主图里反复现算。
4. 对固定 shape、固定 timesteps 的路径，会进一步走更专门的 fast path。

也就是说，它更像“性能模板”。

### 2. `third_party/VoxCPM`

这个目录更适合做语义对照。

它帮助确认 C++ 图改写后是否仍然符合原始 PyTorch 模型行为，例如：

- `UnifiedCFM` 的 CFG 逻辑
- `LocDiT` 的 cond/uncond 结构
- `stop token` 的触发行为

它更像“正确性参照”，而不是直接的性能实现模板。

## 最终保留的优化

以下优化目前已经保留在代码中，并经过本地编译、单测以及用户侧 CUDA benchmark 验证。

### 1. 默认回到稳定的单 backend CUDA 路径

实验性 scheduler 不再作为默认路径启用，避免 `ggml_backend_sched_alloc_splits` 一类不稳定行为影响推理。

当前策略是：

- 默认优先稳定
- 性能优化主要在图结构和模块边界上做
- 不再把 scheduler 当成主优化方向

### 2. decode benchmark 与日志口径

`examples/voxcpm_tts.cpp` 已支持以下环境变量：

- `VOXCPM_BENCHMARK_DECODE_STEPS=N`
- `VOXCPM_BENCHMARK_IGNORE_STOP=1`
- `VOXCPM_LOG_DECODE_TIMING=1`
- `VOXCPM_LOG_DECODE_TRANSFERS=1`
- `VOXCPM_PROFILE_FRONT_HALF=1`

作用是把“自然 stop 波动”和“纯 decode 性能变化”拆开看。

### 3. `LocDiT` CFG pair 单次 decoder forward

原始 C++ 路径里，cond/uncond 两路会分别跑 decoder。

现在 `LocDiTModel::forward_cfg_pair_projected()` 会把两路拼成一条序列，用一次 non-causal decoder forward 算完，再拆回 cond/uncond 两路输出。这一项是本轮收益最大的优化之一。

### 4. `LocDiT` 的 `qkv / gate_up` 融合

在 `locdit.` 路径上，GPU 会优先使用：

- `qkv_proj`
- `gate_up_proj`

这样减少了 decoder 内部 matmul 的数量，也更接近 ONNX 的执行形态。

### 5. `UnifiedCFM` 时间表预计算

当前仍然保留外部可调的 `--inference-timesteps`，但在内部会对常用步数做缓存。

当前预计算范围：

- 8
- 9
- 10
- 11
- 12

缓存内容是按 `timesteps` 预先算好的 CFG time table。运行时直接把这张小表作为 graph input 喂给 `UnifiedCFM`，避免每个 Euler 子步在图里反复生成 `t / dt` 相关时间嵌入。

这不是“锁死 timesteps”，而是“按 timesteps 建缓存”。

### 6. `patch_to_lm_embed` 并入 `decode_front_half`

`patch -> locenc.forward_patch -> enc_to_lm_proj` 这段已经被并进 `decode_front_half` cached graph。

结果是 decode 日志里的 `patch_embed_ms` 基本可以降到 `0`，少了一次独立 graph launch 和一段中间往返。

### 7. dense `f16/f32` 路径兼容修复

引入 `qkv` 融合后，`f16 / f32` GGUF 的 CUDA 路径曾触发：

`GGML_ASSERT(ggml_is_contiguous(a))`

这个问题已经修复。当前 dense 路径可以正常跑通 benchmark。

## 已尝试但回退的优化

下面这些优化都实际做过、编译过、跑过 benchmark，但最终没有保留在默认代码路径里。

### 1. `base_step + residual_step` 合并图

曾尝试把：

- `base_lm_step`
- `residual_lm_step`

合成一张 state graph，目的是减少 graph launch 和 host 侧拼接。

结果：

- 局部 timing 看起来略有下降
- 但 stop 语义出现明显回退
- 用户 benchmark 中甚至出现过过早在 `step 3` 观测到 stop

因此这条优化已经回退。

### 2. `cond_proj / feat_cond` 前移缓存

曾尝试在 decode state 里额外缓存 `prefix_feat_cond_proj`，并让 `decode_front_half` 直接吃 projected cond，同时产出下一步的 projected cond。

结果：

- 功能正确
- 但在 RTX 4060 Ti 上没有得到净收益
- `q4_k`、`q8_0`、`f16` 基本都出现了持平或回退

原因判断：

- 这项改动并没有真正减少每步总算量
- 它只是把 `cond_proj` 从“本步入口”挪到“上一步出口”
- 同时还让 `decode_front_half` 图多了额外输出分支
- GGML CUDA 没吃到类似 ONNX/TensorRT 那种明显收益

因此这条优化已经回退。

### 3. `LocDiT final norm -> out_proj` 融合

曾尝试把 `decoder.norm.weight` 折叠进 `locdit.out_proj.weight`，让 decoder 末尾只保留无权重 RMSNorm。

结果：

- 功能正确
- dense 路径可用
- 但在当前 GGML quantized CUDA 路径上没有形成净收益

原因判断：

- 在 ONNX/TensorRT 的 dense 快路径里，这种融合通常是划算的
- 但在当前 `q4_k / q8_0` 场景下，它可能把原本更适合量化 matmul 的部分，变成了一份新的 fused dense/F16 小矩阵
- 对 GGML CUDA 而言，这不一定更快，反而可能更慢

因此这条优化也已经回退。

## 推荐 benchmark 口径

为了避免 stop 步数波动污染性能判断，建议统一使用固定口径：

```bash
VOXCPM_BENCHMARK_DECODE_STEPS=24 \
VOXCPM_BENCHMARK_IGNORE_STOP=1 \
VOXCPM_PROFILE_FRONT_HALF=1 \
VOXCPM_LOG_DECODE_TIMING=1 \
./build/examples/voxcpm_tts \
  --prompt-audio ./examples/dabin.wav \
  --prompt-text "可哪怕位于堂堂超一品官职,在十二郡一言九鼎的大柱国口干舌燥了,这少年还是没什么反应" \
  --text "测试一下，这是一个流式音频" \
  --output ./voxcpm_stream_single_final.wav \
  --model-path ./models/quantized/voxcpm1.5-q4_k.gguf \
  --threads 8 \
  --inference-timesteps 10 \
  --cfg-value 2.0 \
  --backend cuda
```

推荐优先观察：

- `front_half_ms`
- `base_step_ms`
- `residual_step_ms`
- `it/s`
- `Model inference`

不要只看最终 `RTF`，因为不同 run 的 stop 位置和输出时长会影响 RTF。

## 用户侧 CUDA benchmark 结论

以下结论来自用户在 RTX 4060 Ti 上的固定口径 benchmark。

### 1. 当前稳定版本中，`patch_embed_ms` 已经基本归零

这说明 `patch_to_lm_embed` 并进 `decode_front_half` 是有效的，相关 graph launch 开销已经基本吃掉。

### 2. 低风险、图级别优化基本已经挖到头

当前主要热点已经收敛到：

- `front_half_ms`
- `base_step_ms`
- `residual_step_ms`

也就是说，剩余瓶颈更多是模型本体计算，不再是外围小图和小拷贝。

### 3. 在这张 4060 Ti 上，`q8_0` 比 `q4_k` 更快

这是一个非常重要的实际结论。

用户侧固定 24-step、`timesteps=10` benchmark 的代表性结果为：

- `q4_k`: `Model inference ≈ 1.318s`, `Without AudioVAE RTF ≈ 0.343`
- `q8_0`: `Model inference ≈ 1.273s`, `Without AudioVAE RTF ≈ 0.332`
- `f16`: `Model inference ≈ 1.337s`, `Without AudioVAE RTF ≈ 0.348`
- `f32`: `Model inference ≈ 1.672s`, `Without AudioVAE RTF ≈ 0.435`

这说明在当前 `ggml-cuda` kernel 形态下：

- 更低比特量化不一定更快
- `q8_0` 很可能是这张卡上的更优速度档位
- `f16` 也很有竞争力

### 4. `timesteps=8` 是非常有价值的速度档位

用户侧 benchmark 也显示：

- `timesteps=8` 会显著降低 `UnifiedCFM` 与 `front_half_ms`
- 整体 `Model inference` 和 `RTF` 都会明显改善

如果场景允许略微牺牲一些生成保真度，那么 `--inference-timesteps 8` 是很实用的提速选项。

## 内存占用说明

本轮最终保留的新增缓存，内存开销都不大。

### 1. `UnifiedCFM` 8~12 步时间表缓存

大小大致为：

- `hidden_size * timesteps * sizeof(float)`

整体只是几十到几百 KiB 量级，不会成为主要内存压力。

### 2. 其余保留优化

像 `qkv / gate_up` 融合权重、CFG pair 常量张量、`decode_front_half` cached graph 这类额外开销，相比模型权重、compute arena 和 KV cache 都是小头。

## 当前代码状态

截至本文档撰写时，当前稳定版本已经通过：

- `cmake --build build -j8`
- `ctest --test-dir build --output-on-failure -R "test_minicpm|test_locdit|test_unified_cfm|test_voxcpm"`
- CPU CLI smoke

本轮最新已回退的两项优化是：

- `cond_proj / feat_cond` 前移缓存
- `LocDiT final norm -> out_proj` 融合

文档中的性能结论已经按“回退后保留的稳定版本”重新整理。

## 后续仍可继续做的事

如果后面还要继续追性能，剩下更值得做的方向已经不太像“小修小补”了，而是更重的路线：

1. 为 `LocDiT` 做更激进的固定 shape 专用 fast path。
2. 针对常用 `timesteps=8/10` 做更专项的图或算子路径。
3. 直接深入 `ggml-cuda` 侧，做更底层的 kernel 优化。

换句话说：

- 低风险图级优化差不多已经收口
- `third_party/Text-to-Speech-TTS-ONNX` 依然有参考价值
- 但剩下想继续逼近 ONNX/TensorRT，代价会明显提高
