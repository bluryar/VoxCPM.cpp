# AudioVAE 的 ggml 算子级优化调查报告

## 1. 目标与边界

本报告的目标是回答一个更具体的问题：

- 为什么 AudioVAE 在 CPU 上“已经能量化，但没有变快”？
- 下一步如果要在 `third_party/ggml` 侧做原型，最值得先改哪几个算子？

本报告基于当前仓库本地源码阅读完成，重点看了：

- `third_party/ggml/src/ggml.c`
- `third_party/ggml/src/ggml-cpu/ggml-cpu.c`
- `third_party/ggml/src/ggml-cpu/ops.cpp`

本阶段结论是：

- 暂时**不建议**先做“大面积图改写”
- 更建议先在 `ggml` CPU 内核上做小而准的原型
- 先做测量，再做 fused / cached / specialized kernel

---

## 2. AudioVAE 当前在 ggml 上的实际调用路径

### 2.1 普通卷积

当前规则卷积的计算链路是：

1. `ggml_pad_ext` 做左侧 causal padding
2. `ggml_im2col` 把激活展开
3. `ggml_mul_mat(weight, activations)`
4. `ggml_add` 加 bias

这条路径的好处是：

- 能复用现有 `mul_mat` 量化能力

坏处是：

- 先物化 `im2col`
- 再执行 `mul_mat`
- 两步之间还有激活类型、布局和缓存开销

### 2.2 深度卷积

depthwise conv 当前不是 `ggml` 原生 op，而是项目侧自定义 kernel。

这条路径目前已经支持：

- `F32` / `F16` 权重
- `F32` / `F16` bias

它不是当前 mixed-Q4_K slowdown 的主嫌疑，因为：

- depthwise 权重体积很小
- 也没有走低比特 `mul_mat`

### 2.3 转置卷积

decoder 的上采样路径走：

- `ggml_conv_transpose_1d`

这部分仍然是 AudioVAE decode 侧最重的候选热点之一。

---

## 3. 源码级观察

## 3.1 `ggml_im2col` 本身是严格 shape-sensitive 的

位置：

- `third_party/ggml/src/ggml.c`

1D 情况下有这个断言：

- `GGML_ASSERT(b->ne[1] == a->ne[1]);`

这里的语义是：

- 输入激活 `b` 的 channel 数必须等于 kernel `a` 的 input channel 数

这也是为什么旧版 `voxcpm_tts` 二进制在加载新格式量化模型时会直接在 `ggml_im2col` 处崩掉。

这条观察说明：

- 只要我们继续复用 `im2col`，运行时 shape 解释必须绝对严格
- 一旦后续想做新的 fused conv op，可以考虑绕开这段 shape-only kernel tensor 的构造

## 3.2 CPU 侧 `im2col` 是实打实的“物化激活矩阵”

位置：

- `third_party/ggml/src/ggml-cpu/ops.cpp`
- `ggml_compute_forward_im2col_f32`
- `ggml_compute_forward_im2col_f16`

当前实现特征：

1. 会把输入显式写入目标 buffer，而不是 lazy view。
2. F16 路径会在写出时做 `FP32 -> F16` 转换。
3. 线程划分是 `for (iic = ith; iic < IC; iic += nth)`，也就是按 channel 分摊。
4. 对 1D pointwise (`kernel=1`) 没有专门分支。

这意味着：

- 对很多 AudioVAE 小卷积而言，`im2col` 本身的 copy/gather 成本未必小于真正的算术成本
- pointwise conv 明明本质更像线性层，但还是走了通用 `im2col`

## 3.3 CPU 侧 `mul_mat` 在 low-bit 路径下会 repack 激活

位置：

- `third_party/ggml/src/ggml-cpu/ggml-cpu.c`
- `ggml_compute_forward_mul_mat`

关键行为：

1. `src0` 是权重，`src1` 是激活。
2. 若 `src1->type != vec_dot_type`，会把 `src1` 转换/打包到 `params->wdata`。
3. 这条分支里还有明确断言：`GGML_ASSERT(src1->type == GGML_TYPE_F32);`

对我们当前 AudioVAE mixed 路径的影响是：

- 权重是 low-bit `Q4_K/Q5_K/...`
- 激活是 `F32`
- 因此每个规则卷积 `mul_mat` 都要额外做一次激活 repack

这非常像当前 slowdown 的核心来源之一：

- 低比特权重省下来的带宽和算术，不足以抵消激活 repack + `im2col` 物化

## 3.4 `conv_transpose_1d` 只支持 `F16/F32` kernel，而且每次都会重排数据

位置：

- `third_party/ggml/src/ggml-cpu/ops.cpp`
- `ggml_compute_forward_conv_transpose_1d_f16_f32`
- `ggml_compute_forward_conv_transpose_1d_f32`

当前实现特征：

1. 只支持：
  - `src0=F16, src1=F32, dst=F32`
  - `src0=F32, src1=F32, dst=F32`
2. 每次调用时：
  - 先清空 `params->wdata`
  - 把 kernel 从 `(K x Cout x Cin)` 重排到 `(Cin x K x Cout)`
  - 把 source 从 `(L x Cin)` 重排到 `(Cin x L)`
  - 再执行嵌套 dot 累加
3. 每次调用还会先 `memset(dst->data, 0, ggml_nbytes(dst))`

这说明 decode 侧至少有三个潜在成本：

1. kernel 重排是重复的
2. source 重排是重复的
3. 输出清零和逐点累加写回可能有较大 cache 压力

## 3.5 当前证据说明“问题不只是权重位宽”

真实 benchmark 已经给出一个很重要的反证：

- FP32 最快
- mixed Q4_K 更慢
- AudioVAE=F16 + Q4_K 还要更慢

如果问题主要只是“low-bit 精度不合适”，那么 `AudioVAE=F16` 本该更接近 FP32。

但它依然明显慢于 FP32，说明当前 CPU 开销还有别的主因，例如：

- `conv_transpose_1d` 的实现方式
- `im2col` 的物化
- 小矩阵场景下线程/缓存/重排开销

---

## 4. 为什么当前 mixed 路径会慢

综合源码与实测，当前最可信的解释是：

### 4.1 encoder 侧：`im2col + activation repack + small matmul`

规则卷积低比特化以后，encoder 每层大致变成：

1. 生成 `im2col`
2. 把 `F32` 激活 repack 到 `vec_dot_type`
3. 执行小到中等规模的 low-bit `mul_mat`

对 AudioVAE 这种：

- 层很多
- 规则卷积 kernel 小
- patch 长度不算特别大

的 workload 来说，这三步叠加非常容易吃掉 low-bit 的收益。

### 4.2 decoder 侧：`conv_transpose_1d` 仍然很重

mixed Q4_K 模型里，decoder 最大块的上采样权重仍然是 `F16`。

因此 decode 侧没有享受到“权重显著缩小后直接加速”的效果，反而还要承担：

- 内核重排
- source 重排
- 累加写回

### 4.3 `AudioVAE=F16` 更慢，说明不是“只要降低精度就能更快”

`AudioVAE=F16` 模式主要减少的是：

- 模型加载大小
- 权重内存体积

但它没有减少：

- `im2col`
- `conv_transpose_1d` 的重排逻辑
- `F32` 输出累加

因此在当前 CPU 后端上，它没有变成更快的折中方案。

---

## 5. 候选优化方向

以下方向按建议优先级排序。

## 5.1 P0：先补可观测性，而不是直接改内核

第一优先级不是“马上写新 kernel”，而是先把热点量化清楚。

建议在 `ggml` CPU 路径加最小测量：

1. `GGML_OP_IM2COL` 总耗时
2. `GGML_OP_MUL_MAT` 中 `src1` repack 耗时
3. `GGML_OP_CONV_TRANSPOSE_1D` 总耗时
4. 每个 op 的输入输出 shape、类型、调用次数

目标：

- 判断 encoder 真正慢在 `im2col` 还是 `mul_mat pack`
- 判断 decoder 真正慢在 kernel 重排还是 dot 累加

如果没有这一层测量，后续很容易反复做“看起来合理、实际更慢”的优化。

## 5.2 P1：给 `conv_transpose_1d` 加“预打包 kernel 缓存”

这是当前我最看好的第一批原型之一。

理由：

1. decoder 的 kernel 是固定权重，不该每次 forward 都重新 permute。
2. 当前实现每次调用都会重排 kernel 到 `params->wdata`。
3. 这类优化对数值语义影响最小，适合作为 subtree 维护的首个 `ggml` patch。

可以考虑的原型：

- 在权重加载后，为 `conv_transpose_1d` 维护一个 CPU 侧预打包版本
- forward 时只重排 source，不再重排 kernel

风险：

- 需要设计缓存生命周期
- 可能涉及 `ggml_tensor` 外挂 metadata 或上层显式缓存

## 5.3 P1：给 1D 规则卷积做 fused kernel，直接绕开 `im2col`

这条路线也很值得做，但复杂度比上一个高。

目标不是一开始就支持所有 low-bit，而是分阶段：

1. 先做 `F16/F32` 权重版本
2. 再做 `Q8_0`
3. 最后再看 `Q4_K/Q5_K`

原因：

- 先证明“绕开 `im2col` 物化”本身能不能带来收益
- 如果连 `F16/Q8_0` 版本都不快，就没必要继续给 `Q4_K` 做更复杂实现

建议支持的第一批场景：

- 1D causal conv
- `stride=1`
- `dilation=1`
- `kernel=1/3/7`

这已经足够覆盖 AudioVAE 大部分普通卷积。

## 5.4 P1：专门做 pointwise conv (`kernel=1`) 的原生 CPU kernel

graph 侧做 pointwise fast path 已经试过，效果不好，原因大概率是：

- 仍然需要显式 `permute/cont/reshape`

但这并不说明 pointwise conv 没有优化空间，而是说明：

- 这种优化应该下沉到内核层，而不是只在图上拼 op

最理想的 pointwise kernel 应该能直接吃：

- `[T, C, B]` 或其现有 stride 布局

避免额外：

- `ggml_cont`
- `ggml_permute`
- `im2col`

## 5.5 P2：让 `mul_mat` 的 activation repack 更便宜

如果继续保留 “规则卷积 = `im2col + mul_mat`” 这条思路，那么另一个方向是降低 `mul_mat` 对激活侧的 repack 成本。

候选思路：

1. 更细粒度的 pack cache
2. 对小矩阵做不同的 chunking 策略
3. 对 `Q4_K/Q5_K x F32` 小矩阵场景加专用内核

但这条路线的风险也更大：

- `ggml_compute_forward_mul_mat` 是全局核心路径
- 一旦改坏，影响面会远大于 `conv_transpose_1d`

因此我不建议把它作为第一刀。

## 5.6 P3：低比特 `conv_transpose_1d`

这是最诱人的方向之一，但不建议现在就做。

原因：

1. 设计和测试成本高
2. 需要新的 low-bit dot kernel 组合
3. 可能牵涉到更多类型分发和 pack 格式

更实际的顺序应该是：

1. 先把现有 F16/F32 `conv_transpose_1d` 做快
2. 确认 decode 侧到底还有多少剩余瓶颈
3. 再决定值不值得继续做 low-bit transpose conv

---

## 6. 推荐原型顺序

建议按下面顺序推进：

### 阶段 1：只做测量

目标：

- 在 `ggml` CPU 路径里记录 `IM2COL`、`MUL_MAT` pack、`CONV_TRANSPOSE_1D` 的时间占比

产出：

- 一份带真实 hotspot 占比的数据表

### 阶段 2：先改 `conv_transpose_1d` 的 kernel 预打包

目标：

- 不改变外部 API
- 只减少每次 forward 的重复 kernel 重排

判断标准：

- 若 decode 明显下降，则说明这条路线有价值

### 阶段 3：做规则 conv 的 fused F16/Q8_0 原型

目标：

- 先验证“去掉 `im2col` 物化”本身的收益

不建议一上来就做：

- `Q4_K` fused conv

因为那样很难判断到底是“low-bit 算法问题”还是“fused 思路本身就不划算”。

### 阶段 4：再看要不要进 low-bit fused conv 或 low-bit transpose conv

只有当前三阶段已经证明：

- 真正的瓶颈位置明确
- F16/Q8_0 原型有正收益

才值得继续做更深的 low-bit kernel。

---

## 7. 对 subtree 维护方式的建议

既然 `ggml` 后续准备作为 subtree 维护，建议尽量按下面原则推进：

1. 第一批 patch 只做：
  - instrumentation
  - `conv_transpose_1d` kernel cache
2. 尽量不改公共 API，优先改 CPU backend 内部实现
3. 每个 patch 都附带：
  - 独立 benchmark
  - AudioVAE workload 对照数据
4. 若必须新增实验接口，优先放在本 repo 上层 wrapper，而不是直接扩散到 `ggml.h`

这样后续无论是：

- 自己长期维护 patch queue
- 还是尝试向上游整理提交

成本都会低很多。

---

## 8. 当前判断

当前最值得继续的方向不是：

- 再换一种 AudioVAE 权重量化类型

而是：

1. 测清 `IM2COL / MUL_MAT pack / CONV_TRANSPOSE_1D` 三者的真实占比
2. 优先给 `conv_transpose_1d` 做低风险 CPU 优化原型
3. 再评估是否需要 fused regular conv

换句话说，真正的下一阶段任务应该是：

- 从“模型策略优化”切换到“算子实现优化”

