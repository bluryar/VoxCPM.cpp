# AudioVAE 量化与性能追踪阶段总结

## 1. 背景

本轮工作的目标不是“强行量化 AudioVAE”，而是先把 `AudioVAE` 从“只能用 F32 权重、量化工具也碰不到”的状态，推进到：

1. 可以稳定导出混合类型 GGUF。
2. 可以稳定推理并合成音频。
3. 可以通过 `imatrix` 覆盖到真正参与 `mul_mat` 的 AudioVAE 规则卷积权重。
4. 用真实 benchmark 判断这条路在 CPU 上到底有没有收益。

结论先写在前面：

- 功能链路已经打通。
- 量化模型已经可以正常推理。
- `imatrix` 已经覆盖到 AudioVAE encoder 和 decoder 的规则卷积权重。
- 但在当前 8 线程 CPU 环境下，`AudioVAE` 的 CPU 时延没有得到优化，反而变慢。

---

## 2. 已实现的改动

### 2.1 AudioVAE 普通卷积改成 weight-first `mul_mat`

文件：`src/audio-vae.cpp`

原来的普通 `conv1d` 路径会把权重放在 `mul_mat` 的 `src1` 一侧，无法复用现有的 low-bit 权重量化路径。

现在已经改成：

- 先解析卷积权重形状
- 把普通卷积权重整理为 2D 视图
- 用 `ggml_im2col(...)` 生成激活列块
- 调用 `ggml_mul_mat(weight_2d, activations)`

这样 AudioVAE 的规则卷积、pointwise 卷积、`fc_mu` 权重已经能进入现有量化链路。

### 2.2 深度卷积改成兼容 `F16` 权重

AudioVAE 的 depthwise conv 仍是自定义 kernel，但已经从“只读 `float *`”改成兼容：

- `F32` 权重
- `F16` 权重
- `F32` bias
- `F16` bias

输出和累加仍保留 `F32`。

### 2.3 `transpose_conv_1d` 继续走原生 `ggml`，但量化策略固定为 `F16`

目前 `ggml_conv_transpose_1d` 在 CPU 路径只支持：

- `F16` kernel
- `F32` kernel

因此 AudioVAE decoder 的转置卷积没有做低比特量化，而是固定导出为 `F16`。

### 2.4 AudioVAE 量化策略变成混合策略

文件：`src/quantize.cpp`

当前默认策略：

- `*.bias`、`*.alpha`：保留 `F32`
- depthwise conv 权重：`F16`
- transpose conv 权重：`F16`
- 其他 AudioVAE 规则卷积 / pointwise / `fc_mu`：跟随请求量化类型

另外有一个重要例外：

- `audio_vae.encoder.fc_logvar.weight` 当前不参与运行图
- 因此它不会出现在 `imatrix` 统计里
- 为避免低比特类型要求 `imatrix` 条目却永远拿不到，这个张量被固定降到 `F16`

### 2.5 量化后的规则卷积权重按 2D layout 写入 GGUF

规则卷积源权重本来是 3D：

- `[kernel, in_channels, out_channels]`

为了适配 K-quant 的 block 对齐限制，量化输出时会改写成 2D layout：

- `[kernel * in_channels, out_channels]`

运行时已经支持同时识别：

- 原始 3D 权重
- 量化后的 2D 权重
- 部分单输出卷积的 2D 权重

### 2.6 `voxcpm_imatrix` 已覆盖 AudioVAE encoder 和 decoder

文件：`examples/voxcpm_imatrix.cpp`

这轮还补了一个很关键的点：

- prompt audio 的 `AudioVAE::encode()` 图，现在会被 collector 观察
- decode 循环结束后，会额外构建一次 AudioVAE decoder 图并观察

因此 `imatrix` 不再只覆盖 LM，而是也覆盖了 AudioVAE 的规则卷积 `mul_mat`。

---

## 3. 新增的导出模式

除了默认的混合模式，现在还支持：

- `--audio-vae-mode mixed`
- `--audio-vae-mode f16`

其中 `f16` 的含义是：

- AudioVAE 全部权重统一走 `F16`
- 其他模块仍按 `--type` 的量化策略处理

示例：

```bash
./build/examples/voxcpm_quantize \
  --input models/voxcpm1.5.gguf \
  --output /tmp/voxcpm1.5-audiovae-f16-q4k.gguf \
  --type Q4_K \
  --audio-vae-mode f16 \
  --threads 8
```

这个模式的目的不是“已经证明更快”，而是作为一个干净基线，用来回答：

- “AudioVAE 只改成 F16，会不会比 mixed low-bit 更适合 CPU？”

目前答案是否定的，见第 5 节。

---

## 4. 兼容性与踩坑记录

### 4.1 `voxcpm_tts` 崩溃并不一定是模型坏了，可能是二进制没重编

本轮出现过一次典型现象：

- `voxcpm_tts` 在 `Encoding prompt audio...` 阶段崩溃
- backtrace 落到 `ggml_im2col` 的 shape assert

根因不是模型损坏，而是：

- `libvoxcpm` 已经更新
- `voxcpm_tts` 还是旧二进制
- 旧二进制没有链接到新的 `audio-vae.cpp` 实现

重新编译 `voxcpm_tts` 后，同一份量化模型就能正常推理。

因此凡是改动了 `src/audio-vae.cpp`、`src/quantize.cpp`、`examples/voxcpm_imatrix.cpp` 之后，都应该至少重编：

- `voxcpm_tts`
- `voxcpm_quantize`
- `voxcpm_imatrix`
- `voxcpm_benchmark`

### 4.2 `imatrix` 收不到 AudioVAE 条目，有两种常见原因

第一种：

- `mul_mat` 的 `src0` 是 reshape 出来的 view
- 没有把原始权重名传播过去
- collector 会把它当匿名 tensor，统计不到

这个问题已经修过。

第二种：

- `fc_logvar` 根本不在运行图里
- 所以不可能出现在 `imatrix` 文件中

这个问题不是 collector 漏采，而是运行路径本身不包含它。

### 4.3 图上“看起来更合理”的优化，未必会更快

曾经尝试过一个 `kernel=1` pointwise conv fast path：

- 目标是绕过 `im2col`
- 直接把输入 reshape 后做 `mul_mat`

真实 benchmark 显示它让 `audio_vae.encode` 变得更慢，因此该改动已经撤回，没有保留在当前实现中。

这个经验很重要：

- AudioVAE 现在的瓶颈不是“只要减少一个 op 就会变快”
- 而是 CPU cache、layout 转换、线程划分、激活打包等开销的综合结果

---

## 5. 已验证的数据

### 5.1 Mixed Q4_K 导出统计

真实模型 dry-run 结果：

- `tensors: total=690, quantized=426, preserved=264`
- `audio_vae detail: total=223, quantized=25, f16=51, preserved=147`

### 5.2 AudioVAE=F16 + Q4_K 导出统计

真实模型导出结果：

- `tensors: total=690, quantized=426, preserved=264`
- `audio_vae detail: total=223, quantized=0, f16=76, preserved=147`
- 输出模型大小约 `643.40 MiB`

### 5.3 `imatrix` 覆盖情况

最初最小样本收集时，条目数只有：

- `351`

补齐 AudioVAE encoder / decoder 图观察后，条目数增长到：

- `390`

并且可以在 `imatrix` 文件中看到典型 AudioVAE 权重条目，例如：

- `audio_vae.encoder.block.0.weight`
- `audio_vae.encoder.fc_mu.weight`
- `audio_vae.decoder.model.1.weight`
- `audio_vae.decoder.model.<final_conv_idx>.weight`

这里最后一项不应再理解为固定的 `model.8.weight`。
在当前实现里，decoder 最终卷积的索引由 `decoder_rates.size()` 动态推导：

- `final_conv_idx = num_decoder_blocks() + 3`

因此不同版本或不同结构的 AudioVAE，最终卷积层编号可能不同。

### 5.4 推理烟测

以下两种模型都已经真实合成成功：

- mixed `Q4_K`
- `AudioVAE=F16 + Q4_K`

输出 wav 已经实测写出。

### 5.5 CPU benchmark 结果

同一台机器、8 线程、`short` 场景下的最新对比：

| 模型 | audio_vae.encode | audio_vae.decode |
|------|------------------|------------------|
| 原始 FP32 | `315.223 ms` | `717.293 ms` |
| mixed Q4_K | `390.964 ms` | `848.697 ms` |
| AudioVAE=F16 + Q4_K | `607.293 ms` | `896.437 ms` |

结论非常明确：

- mixed Q4_K 比 FP32 慢
- AudioVAE=F16 + Q4_K 比 mixed Q4_K 还慢
- 因此“把 AudioVAE 改成 F16 导出”并不是 CPU 加速解法

---

## 6. 当前结论

截至本阶段，已经可以确认：

1. AudioVAE 不再被 “F32-only 权重 + 无法量化” 锁死。
2. mixed 量化链路、`imatrix` 覆盖、真实推理都已经打通。
3. `AudioVAE=F16` 基线也已经建立并验证可推理。
4. CPU 性能问题并不主要来自量化策略配置，而更像是 `ggml` 算子级实现开销。

更直白地说：

- 问题已经从“能不能量化”切换成了“`ggml` CPU 内核是否适合这类 AudioVAE 卷积 workload”

---

## 7. 下一阶段建议

下一步不应继续在“规则卷积用 Q4_K 还是 F16”这个层面反复试错，而应该直接进入 `ggml` 算子级调查。

建议把工作重点放到：

1. `im2col` 的物化与类型转换开销
2. low-bit `mul_mat` 对激活侧的 repack 开销
3. `conv_transpose_1d` 的 CPU 实现

对应调查报告见：

- `docs/audio_vae_ggml_operator_investigation_zh.md`
