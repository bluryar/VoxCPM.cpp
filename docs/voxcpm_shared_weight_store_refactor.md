# VoxCPM 共享权重池与低内存加载重构总结

## Context

本文记录一次针对 VoxCPM GGML 运行时高内存问题的重构实践。

重构前，`test_voxcpm` 在执行 `Prefill/Decode` trace 测试时，进程峰值内存曾被观察到接近 `14 GiB`。模型文件本身只有约 `3.4 GiB`，说明问题不在单份权重，而在加载架构。

本次重构的目标是：

- 将“每个模块各自加载整份 GGUF”的模式改为“单份 GGUF metadata + 单份权重 buffer + 多模块共享 tensor 指针”
- 保持现有数值路径不变，不放宽 trace 容差
- 确保 `VoxCPM.cpp/tests` 下所有现有测试继续通过

最终结果：

- 全量测试 `8/8` 通过
- `test_voxcpm` 的近似峰值 RSS 降到约 `4.15 GiB`

---

## 一、问题根因

### 1.1 现象

运行 `VoxCPMRuntime::load_from_gguf()` 时，会同时加载：

- `base_lm_`
- `residual_lm_`
- `feat_encoder_`
- `feat_decoder_estimator_`
- `fsq_layer_`
- `components_`

进一步展开后，`LocEnc` 和 `LocDiT` 内部还各自包含一个 `MiniCPMModel`，而 `components_` 又包含：

- `enc_to_lm_proj`
- `lm_to_dit_proj`
- `res_to_dit_proj`
- `stop_token`
- `embed_tokens`

如果这些对象都各自加载 GGUF，就会出现同一个模型被重复加载多次。

### 1.2 根因

重构前的典型加载模式是：

1. `gguf_init_from_file(..., no_alloc=true, ctx=&ggml_ctx_ptr)`
2. 在这个包含整份 GGUF tensor metadata 的 `ggml_context` 上调用 `ggml_backend_alloc_ctx_tensors(...)`
3. 把对应权重数据读入这个 buffer

问题在于：

- `ggml_ctx_ptr` 不是“该模块私有的少量 tensor context”
- 而是“整份 GGUF 的 tensor metadata context”

因此，只要一个模块调用一次 `ggml_backend_alloc_ctx_tensors(ggml_ctx_ptr, ...)`，就等于给整份模型重新分配一遍权重 buffer。

如果多个模块都这么做，就会出现多份完整权重常驻内存。

### 1.3 为什么 trace 文件不是主因

`trace_Prefill.jsonl` 和 `trace_Decode_one.jsonl` 体积确实很大，系统也会产生 page cache，但这只是次要因素。

真正的大头是：

- 同一个 `voxcpm1.5.gguf`
- 被不同子模块反复 `alloc_ctx_tensors`

---

## 二、为什么共享权重池是最优首选

这次设计不是拍脑袋决定的，而是对照了三类参考：

- [GGML_BEST_PRACTICES.md](./GGML_BEST_PRACTICES.md)
- `third_party/ggml/examples/gpt-2/main-backend.cpp`
- `vendor/llama.cpp` 的 model loader 设计

它们共同体现出的最佳实践是：

1. 只保留一份模型 metadata
2. 只分配一份 persistent weight buffer
3. 模块内部只持有 tensor 指针，而不是再拥有一份独立模型存储

对于当前 VoxCPM 仓库，第一阶段最划算的优化不是直接上 `mmap`，而是先把重复加载去掉。

原因很简单：

- 风险最低
- 对数值行为零侵入
- 收益最大
- 与 GGML 官方示例和 `llama.cpp` 的方向一致

---

## 三、最终方案

### 3.1 新增共享权重池 `VoxCPMWeightStore`

新增文件：

- [include/voxcpm/weight-store.h](../include/voxcpm/weight-store.h)
- [src/weight-store.cpp](../src/weight-store.cpp)

职责非常明确：

1. 只调用一次 `gguf_init_from_file(..., no_alloc=true, ctx=&ctx)`
2. 只调用一次 `ggml_backend_alloc_ctx_tensors(shared_ctx, backend)`
3. 只打开一次 GGUF 文件并加载全部 tensor 数据
4. 对外提供：
   - `get_tensor(name)`
   - `has_tensor(name)`
   - `get_u32/get_f32/get_bool`
   - `get_i32_array/get_f32_array`
   - `buffer_size()/tensor_count()/owns_storage()`

这让所有模块都可以共享同一份：

- `gguf_context`
- `ggml_context`
- `ggml_backend_buffer_t`

### 3.2 模块新增 `load_from_store(...)`

以下模块都新增了 shared-load 接口：

- [MiniCPMModel](../include/voxcpm/minicpm.h)
- [LocEncModel](../include/voxcpm/localenc.h)
- [LocDiTModel](../include/voxcpm/locdit.h)
- [FSQ](../include/voxcpm/fsq.h)
- [LinearProjection / StopTokenPredictor / Embedding / VoxCPMComponents](../include/voxcpm/components.h)
- [AudioVAE](../include/voxcpm/audio-vae.h)

shared-load 的原则是：

- 只从 `VoxCPMWeightStore` 里拿 tensor 指针
- 不再在 shared path 上调用新的 `gguf_init_from_file`
- 不再在 shared path 上调用新的 `ggml_backend_alloc_ctx_tensors`
- 模块析构时不释放共享权重，由 `shared_ptr<VoxCPMWeightStore>` 统一托管生命周期

兼容层 `load_from_gguf(...)` 仍然保留，对单模块测试和独立调用完全兼容。

### 3.3 Runtime 统一走共享 store

[src/voxcpm.cpp](../src/voxcpm.cpp) 现在的加载流程是：

1. 先创建 `weight_store_`
2. 加载整份 GGUF 到共享 store
3. `base_lm_ / residual_lm_ / feat_encoder_ / feat_decoder_estimator_ / fsq_layer_ / components_`
   全部从同一个 `weight_store_` 绑定
4. `LocEnc` 内部的 `MiniCPM`
5. `LocDiT` 内部的 `MiniCPM`

也都继续沿用同一份 store

这一点非常关键。否则即使外层共享了，内层子模块重新开一份权重，也会把收益吃掉。

---

## 四、测试与验证

### 4.1 测试范围

本次重构后验证通过的测试包括：

- `test_audio_vae`
- `test_fsq`
- `test_components`
- `test_minicpm`
- `test_localenc`
- `test_locdit`
- `test_unified_cfm`
- `test_voxcpm`

命令：

```bash
ctest --test-dir /home/orangepi/Codes/ggbond/VoxCPM.cpp/build --output-on-failure -j1
```

结果：

```text
100% tests passed, 0 tests failed out of 8
```

### 4.2 新增的结构性验证

[tests/test_voxcpm.cpp](../tests/test_voxcpm.cpp) 新增了一个 smoke test，专门断言：

- runtime 自己持有一份共享 store
- `base_lm` / `residual_lm` / `locenc` / `locdit` / `fsq`
  都指向同一个 store
- `LocEnc` 内部的 `MiniCPM` 和外层共用同一个 store
- `LocDiT` 内部的 `MiniCPM` 和外层共用同一个 store
- `components_` 以及内部的 projection / stop / embedding
  全都指向同一个 store

这类测试很重要，因为它能防止未来重构时“某个子模块偷偷又恢复独立加载”。

### 4.3 Trace 数值没有回归

本次重构没有修改推理数值路径，只修改权重持有方式。

`test_voxcpm` 仍然维持原来的 trace 对齐输出，包括：

- `input range`
- `expected range`
- `actual range`
- `max abs error`
- `avg abs error`
- `mismatch rate`

实测 `Prefill/Decode` 的误差阈值都保持在原标准内，没有放宽。

---

## 五、内存收益

### 5.1 测量方法

这台机器上没有 `GNU /usr/bin/time -v`，因此使用了 `/proc/<pid>/status` 轮询 `VmRSS` 的方式做近似峰值统计。

测量命令思路：

1. 后台启动 `build/tests/test_voxcpm`
2. 周期性读取 `/proc/$pid/status`
3. 记录最大 `VmRSS`

### 5.2 结果

重构前：

- 用户现场观察约 `14 GiB`

重构后：

- `peak_kb = 4347444`
- 约 `4.15 GiB`

虽然这个数值是近似峰值，不是 GNU `time -v` 的官方统计，但已经足够说明：

- 共享权重池解决了绝大多数重复常驻权重内存
- 当前内存占用已经从“明显异常”回到“基本合理”

---

## 六、实现上的几个关键经验

### 6.1 共享的是“权重存储”，不是“所有状态”

可以共享的：

- GGUF metadata
- tensor metadata context
- 权重 backend buffer
- 各个权重 tensor 指针

不应该共享的：

- graph context
- graph allocator上的中间结果
- KV cache
- 每次推理时创建的输入/输出 tensor
- MiniCPM 的 auxiliary tensor 之外的临时执行状态

换句话说，**persistent weights 要共享，runtime state 不要共享**。

### 6.2 `LocEnc/LocDiT` 的内部 `MiniCPM` 容易漏

只改 runtime 外层不够。

如果：

- `LocEncModel::load_from_store(...)` 绑定了共享权重
- 但内部 `encoder_` 仍然 `load_from_gguf(...)`

那么依然会多开一份权重。

这是这类架构里最容易漏掉的地方。

### 6.3 `components_` 也必须一次性绑定

`VoxCPMComponents` 看起来只是几个“小组件”，但如果：

- `enc_to_lm_proj`
- `lm_to_dit_proj`
- `res_to_dit_proj`
- `stop_token`
- `embed_tokens`

分别独立加载 GGUF，那么内存会非常夸张。

正确做法是：

- `VoxCPMComponents::from_store(...)`
- 内部所有小组件都从同一份 store 取 tensor

### 6.4 共享 store 后要补结构测试

共享 store 最怕“表面跑得通，但实际上某个模块偷偷重新加载了模型”。

因此建议每个聚合 runtime 都配一个结构性测试，断言：

- store 地址唯一
- 子模块共享同一地址

这种测试成本很低，但收益非常高。

---

## 七、后续模块开发规范

以后新增任何 GGUF 模块时，都建议遵循下面这套规范：

### 7.1 模块接口

建议同时保留两套入口：

- `load_from_gguf(...)`
- `load_from_store(const std::shared_ptr<VoxCPMWeightStore>& ...)`

其中：

- `load_from_store(...)` 是 runtime 聚合路径的主入口
- `load_from_gguf(...)` 是兼容单模块测试和独立调试的包装层

### 7.2 运行时聚合

凡是像 `VoxCPMRuntime` 这样的顶层 runtime：

- 必须只创建一份共享 `VoxCPMWeightStore`
- 必须让所有子模块都从这份 store 绑定

禁止出现：

- runtime 外层共享
- 但内层子模块又重新 `gguf_init_from_file`

### 7.3 新模块检查清单

新增模块时，至少检查：

- 是否新增了 `load_from_store(...)`
- shared path 下是否避免了新的 `ggml_backend_alloc_ctx_tensors`
- 是否只保存 tensor 指针，而不是再拥有一份整模型 context/buffer
- 是否给聚合 runtime 加了 store 唯一性断言
- 是否现有 trace 数值测试全部通过

---

## 八、尚未做的事

本次重构**没有**实现 `mmap` 权重加载。

原因不是做不到，而是当前阶段没有必要。

共享权重池已经解决了主要问题：

- 先去掉重复加载
- 再考虑更复杂的 mmap / lazy loading / backend-specific optimization

后续如果要做第二阶段优化，建议顺序仍然是：

1. 先保持共享 store 架构不变
2. 再把 store 的底层存储从“完整读入 buffer”演进到“mmap + 必要映射”

这样风险最小。

---

## 九、结论

这次重构的核心结论是：

> 在 GGML 多模块工程里，最危险的内存陷阱不是计算图，而是“多个模块各自给整份 GGUF 重新分配一遍权重 buffer”。

最有效的第一步优化不是盲目上复杂方案，而是：

> 做一份共享的 `WeightStore`，让所有模块只绑定 tensor 指针。

这套方案：

- 符合 GGML 最佳实践
- 与 `llama.cpp` 的模型加载思想一致
- 对数值实现零侵入
- 对内存收益极大

对后续 VoxCPM 继续扩展新模块、做更完整 runtime 聚合，这会是一条值得长期坚持的基础架构规范。
