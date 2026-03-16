# VoxCPM Decode 路径重构总结

## 背景

这一轮改动的目标，不是重写 VoxCPM 的推理结构，而是围绕 `decode` 热路径做一轮保守优化：

- 降低 host 侧反复分配和拷贝的次数
- 把 decode 执行路径整理成更少、更清晰的阶段
- 给现有 ggml/gallocr 执行模型补上更合理的 `reserve`
- 修正 stop 头在量化时的策略
- 增加一个能覆盖“多步 decode 稳定性”的测试，避免单步测试通过但真实 CLI 崩溃

这批改动已经进入当前 `HEAD`，因此 `git diff` 现在为空。

## 涉及文件

- [`examples/voxcpm_tts.cpp`](/home/orangepi/Codes/ggbond/VoxCPM.cpp/examples/voxcpm_tts.cpp)
- [`include/voxcpm/voxcpm.h`](/home/orangepi/Codes/ggbond/VoxCPM.cpp/include/voxcpm/voxcpm.h)
- [`src/voxcpm.cpp`](/home/orangepi/Codes/ggbond/VoxCPM.cpp/src/voxcpm.cpp)
- [`include/voxcpm/minicpm.h`](/home/orangepi/Codes/ggbond/VoxCPM.cpp/include/voxcpm/minicpm.h)
- [`src/minicpm.cpp`](/home/orangepi/Codes/ggbond/VoxCPM.cpp/src/minicpm.cpp)
- [`src/quantize.cpp`](/home/orangepi/Codes/ggbond/VoxCPM.cpp/src/quantize.cpp)
- [`tests/test_voxcpm.cpp`](/home/orangepi/Codes/ggbond/VoxCPM.cpp/tests/test_voxcpm.cpp)

## 主要改动

### 1. `voxcpm_tts` 的 host 侧内存复用

[`voxcpm_tts.cpp`](/home/orangepi/Codes/ggbond/VoxCPM.cpp/examples/voxcpm_tts.cpp#L478) 里的 `generate_noise()` 被改成了 `fill_noise()`：

- 以前：每步 decode 返回一个新的 `std::vector<float>`
- 现在：复用已有 `noise` buffer，原地填充

[`patch_major_to_latent()`](/home/orangepi/Codes/ggbond/VoxCPM.cpp/examples/voxcpm_tts.cpp#L516) 也增加了输出参数版本：

- 以前：每次都新建 latent vector
- 现在：可复用已有输出 buffer

另外新增了 [`append_stream_frame()`](/home/orangepi/Codes/ggbond/VoxCPM.cpp/examples/voxcpm_tts.cpp#L544)：

- streaming 路径不再每步重建完整 recent frames 序列
- 改成维护一个最近窗口的小缓存

这一部分的意义主要是减少：

- 频繁的 host 内存分配
- decode 循环中的临时 `vector` 拷贝
- streaming 模式下的整段 recent frames rebuild

### 2. `VoxCPMRuntime::decode()` 被整理成更少的阶段

当前 `decode()` 主路径在 [`src/voxcpm.cpp`](/home/orangepi/Codes/ggbond/VoxCPM.cpp/src/voxcpm.cpp#L659)。

这一轮没有做“全图融合”，而是做了一个更保守的阶段化整理：

- [`run_decode_front_half()`](/home/orangepi/Codes/ggbond/VoxCPM.cpp/src/voxcpm.cpp#L467)
  - 负责 `lm_to_dit_proj + res_to_dit_proj + UnifiedCFM`
- [`run_locenc_patch_to_lm_embed()`](/home/orangepi/Codes/ggbond/VoxCPM.cpp/src/voxcpm.cpp#L519)
  - 负责 `LocEnc + enc_to_lm_proj`
- [`run_base_lm_decode_step()`](/home/orangepi/Codes/ggbond/VoxCPM.cpp/src/voxcpm.cpp#L545)
  - 负责 `base_lm forward_step + FSQ`

对应的接口声明在 [`include/voxcpm/voxcpm.h`](/home/orangepi/Codes/ggbond/VoxCPM.cpp/include/voxcpm/voxcpm.h#L121)。

这样做的收益是：

- decode 路径更容易读和排查
- 比完全碎片化的小 helper 链更少图执行点
- 为后续进一步优化留下清晰的切分边界

### 3. 给热路径 helper 补上 `reserve_compute_memory()`

在 [`src/voxcpm.cpp`](/home/orangepi/Codes/ggbond/VoxCPM.cpp/src/voxcpm.cpp#L191) 到 [`src/voxcpm.cpp`](/home/orangepi/Codes/ggbond/VoxCPM.cpp/src/voxcpm.cpp#L565) 这一批 helper 图执行代码里，统一补上了：

```cpp
backend_->reserve_compute_memory(graph);
```

这类改动不是改变模型数值行为，而是改善 allocator 行为：

- 提前为当前图形状保留 compute buffer
- 减少运行过程中 gallocr 的隐式扩容
- 让 decode 热路径的内存行为更稳定一些

### 4. Stop 头量化策略改成保留

[`src/quantize.cpp`](/home/orangepi/Codes/ggbond/VoxCPM.cpp/src/quantize.cpp#L169) 中新增了：

```cpp
if (has_prefix(name, "stop.")) {
    return true;
}
```

这代表：

- `stop.*` 张量现在不再参与普通量化策略
- stop 头会保留原始类型

这么做的原因是 stop 头对精度较敏感，量化后容易带来：

- stop 判停不稳定
- 量化模型和全精度模型的 stop 行为偏差更大

### 5. `MiniCPMModel` 的配套接口和依赖整理

[`include/voxcpm/minicpm.h`](/home/orangepi/Codes/ggbond/VoxCPM.cpp/include/voxcpm/minicpm.h#L112) 增加了一个更方便的 `forward_step()` 重载：

- 允许只传 `position + kv_cache`
- 不必总是显式传 `positions tensor`

[`src/minicpm.cpp`](/home/orangepi/Codes/ggbond/VoxCPM.cpp/src/minicpm.cpp#L507) 里还把：

```cpp
ggml_tensor* kv_sync = ggml_add(ctx, ggml_sum(ctx, k_write), ggml_sum(ctx, v_write));
```

提前到了 `flash_attn` 之前。

这部分的意义是：

- 给新的 decode 分段 helper 提供更顺手的接口
- 显式保留 KV write 的图依赖关系

### 6. 测试从单步 trace 扩展到多步稳定性

[`tests/test_voxcpm.cpp`](/home/orangepi/Codes/ggbond/VoxCPM.cpp/tests/test_voxcpm.cpp#L1182) 新增了：

- `VoxCPM multi-step decode remains stable`

同时加了两个辅助函数：

- [`all_finite()`](/home/orangepi/Codes/ggbond/VoxCPM.cpp/tests/test_voxcpm.cpp#L149)
- [`make_deterministic_noise()`](/home/orangepi/Codes/ggbond/VoxCPM.cpp/tests/test_voxcpm.cpp#L155)

这个测试的重点不是对齐某一步 trace，而是补上此前缺失的检查：

- 连续 decode 多步是否仍稳定
- 输出是否出现 NaN/Inf
- `current_position` 是否正常推进

它解决的是一个很现实的问题：

- 以前单步 trace 测试通过，并不代表真实 CLI 多步推理一定不崩

## 当前实际效果

从代码意图上看，这一轮已经实现了以下收益：

- `voxcpm_tts` decode 循环中的 host 临时分配更少
- streaming 路径的 recent frames 维护更轻量
- runtime decode 的阶段边界更清晰
- 计算图执行前的 compute buffer reserve 更完整
- stop 头不再被量化破坏
- 测试能覆盖多步 decode 稳定性

## 明确没有做的事情

这一轮没有做以下高风险改动：

- 没有把整个 decode 全融合成单张大图
- 没有引入 `ggml_backend_sched_t`
- 没有重写 `attention` 的 `concat` 路径
- 没有改 KV cache 布局
- 没有做多 backend 混合调度

这些都仍然属于后续潜在优化方向。

## 当前验证状态

当前代码已经验证通过：

- `build/tests/test_voxcpm --reporter compact`
- `build/tests/test_minicpm --reporter compact`
- `build-vulkan/tests/test_voxcpm --reporter compact`
- `build/examples/voxcpm_tts --backend cpu` 实际命令行推理

## 后续建议

如果后面继续优化，推荐按这个顺序推进：

1. 先加更细粒度 profiling，把每个 decode 子阶段耗时打出来
2. 再评估是否要做更进一步的图复用
3. 最后才考虑高风险的 `attention` / scheduler / 多 backend 重构

原因很简单：

- 当前这轮已经把低风险、可验证的部分收拾干净
- 后续想继续提速，瓶颈更可能在真正的模型计算本身，而不是纯粹的 host 侧临时分配
