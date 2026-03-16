# MiniCPM GGML 模块设计与当前实现状态

## 1. 目标

本模块实现 VoxCPM 中共享的 MiniCPM Transformer 主干，覆盖以下变体：

- `base_lm`：24 层
- `residual_lm`：8 层
- `locenc`：8 层
- `locdit`：8 层

当前实现已经落地到：

- [include/voxcpm/minicpm.h](/home/orangepi/Codes/ggbond/VoxCPM.cpp/include/voxcpm/minicpm.h)
- [src/minicpm.cpp](/home/orangepi/Codes/ggbond/VoxCPM.cpp/src/minicpm.cpp)
- [tests/test_minicpm.cpp](/home/orangepi/Codes/ggbond/VoxCPM.cpp/tests/test_minicpm.cpp)

文档用途已经从“实现计划”切换为“当前实现说明 + 验证结果 + 未完成项”。

## 2. 当前完成情况

已经完成：

- `MiniCPMConfig` 参数定义与 GGUF 加载
- `MiniCPMModel` 权重加载
- `MiniCPMKVCache` 独立上下文与持久化 buffer
- `forward` 预填充路径
- `forward_step` 单步解码路径
- LongRoPE
- GQA attention
- SwiGLU MLP
- muP 配置读取与残差缩放
- BaseLM / ResidualLM / LocEnc / LocDiT 变体参数适配
- 基础图构建测试
- trace 数值对齐测试

已经修复的关键问题：

- 测试里错误访问不透明 `ggml_cgraph`
- causal mask 广播维度构图错误
- GGUF 中 variant-specific config key 读取错误
- KV cache 批量写入视图布局错误
- attention 路径只依赖 `ggml_cpy` 副作用，导致图依赖不显式的问题

## 3. 结构设计

### 3.1 内存架构

模块遵循 ggml 的 `no_alloc=true + backend buffer` 两阶段模式：

- `weight_ctx_` + `weight_buffer_`
  - 持久化模型权重
- `ctx_kv_` + `buffer_kv_`
  - 持久化 KV cache
- `graph ctx`
  - 每次推理重建计算图
- `aux_ctx_` + `aux_buffer_`
  - 持久化 `pos_tensor` 与 LongRoPE factors

这与 `AudioVAE`、`FSQ` 的实现模式保持一致。

### 3.2 核心类

#### `MiniCPMConfig`

定义在 [config.h](/home/orangepi/Codes/ggbond/VoxCPM.cpp/include/voxcpm/config.h)。

当前关键字段：

- `hidden_size`
- `intermediate_size`
- `n_layer`
- `n_heads`
- `n_kv_heads`
- `vocab_size`
- `max_length`
- `rms_norm_eps`
- `rope_freq_base`
- `scale_emb`
- `dim_model_base`
- `scale_depth`
- `use_mup`
- `rope_original_max`
- `rope_short_factor`
- `rope_long_factor`

#### `MiniCPMWeights`

每层包含：

- `input_layernorm`
- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `post_layernorm`
- `gate_proj`
- `up_proj`
- `down_proj`

#### `MiniCPMKVCache`

KV cache 按层存储，每层两块张量：

- `K`: `[head_dim, max_length, n_kv_heads]`
- `V`: `[head_dim, max_length, n_kv_heads]`

当前对外提供：

- 读取全前缀：`get_k/get_v`
- 写入单步槽位：`get_k_slot/get_v_slot`
- 写入批量区间：`get_k_batch/get_v_batch`
- 数值测试专用原始访问：`raw_k_cache/raw_v_cache`

注意：

- `get_k_batch/get_v_batch` 当前返回的是 3D view，不是早期设计里假定的 2D contiguous view
- 这是为了匹配底层 `[D, T, H]` 布局，避免写 cache 时打乱 head 维顺序

#### `MiniCPMModel`

公开接口：

- `load_from_gguf(...)`
- `forward(...)`
- `forward_step(...)`
- `config()`
- `weights()`
- `get_pos_tensor()`

## 4. GGUF 参数加载

### 4.1 已确认的基础参数

基于 `models/voxcpm1.5.dump` 已确认：

- `llama.context_length = 32768`
- `llama.embedding_length = 1024`
- `llama.block_count = 24`
- `llama.feed_forward_length = 4096`
- `llama.attention.head_count = 16`
- `llama.attention.head_count_kv = 2`
- `llama.attention.layer_norm_rms_epsilon = 1e-5`
- `llama.rope.freq_base = 10000`
- `llama.vocab_size = 73448`

### 4.2 BaseLM 映射

当前实现会优先读取实际 `voxcpm_lm_config_*`：

- `voxcpm_lm_config_hidden_size`
- `voxcpm_lm_config_intermediate_size`
- `voxcpm_lm_config_num_attention_heads`
- `voxcpm_lm_config_num_key_value_heads`
- `voxcpm_lm_config_num_hidden_layers`
- `voxcpm_lm_config_max_position_embeddings`
- `voxcpm_lm_config_vocab_size`
- `voxcpm_lm_config_scale_emb`
- `voxcpm_lm_config_dim_model_base`
- `voxcpm_lm_config_scale_depth`
- `voxcpm_lm_config_use_mup`
- `voxcpm_lm_config_rope_scaling_original_max_position_embeddings`
- `voxcpm_lm_config_rope_scaling_short_factor`
- `voxcpm_lm_config_rope_scaling_long_factor`

### 4.3 其他变体映射

`residual_lm`：

- `voxcpm_residual_lm_num_layers`

`locenc`：

- `voxcpm_encoder_config_hidden_dim`
- `voxcpm_encoder_config_ffn_dim`
- `voxcpm_encoder_config_num_heads`
- `voxcpm_encoder_config_num_layers`

`locdit`：

- `voxcpm_dit_config_hidden_dim`
- `voxcpm_dit_config_ffn_dim`
- `voxcpm_dit_config_num_heads`
- `voxcpm_dit_config_num_layers`

## 5. GGUF 权重命名

### 5.1 BaseLM

无前缀，典型 key：

- `token_embd.weight`
- `blk.{i}.attn_norm.weight`
- `blk.{i}.attn_q.weight`
- `blk.{i}.attn_k.weight`
- `blk.{i}.attn_v.weight`
- `blk.{i}.attn_output.weight`
- `blk.{i}.ffn_norm.weight`
- `blk.{i}.ffn_gate.weight`
- `blk.{i}.ffn_up.weight`
- `blk.{i}.ffn_down.weight`
- `output_norm.weight`

### 5.2 ResidualLM / LocEnc / LocDiT

采用前缀形式：

- `residual_lm.blk.{i}.*`
- `locenc.blk.{i}.*`
- `locdit.blk.{i}.*`

其中：

- `residual_lm.output_norm.weight`
- `locenc.output_norm.weight`
- `locdit.output_norm.weight`

也已按真实命名加载。

## 6. 张量布局与算子约定

### 6.1 主要布局

当前实现中的关键布局：

- 隐状态：`[hidden_size, seq_len]`
- `q`：`[head_dim, n_heads, n_tokens]`
- `k/v`：`[head_dim, n_kv_heads, n_tokens]`
- cache 存储：`[head_dim, max_length, n_kv_heads]`

对应关系：

- batch 当前测试和主要推理路径按 `B=1`
- trace 中 Torch 的 `[B, T, C]` 与 GGML 的 `[C, T]` 在 `B=1` 时 flat memory 顺序一致
- trace 中 Torch 的 `[B, H, T, D]` 与 GGML cache 的 `[D, T, H]` 在 `B=1` 时也可以按顺序重排后直接比较

### 6.2 RMSNorm

使用：

- `ggml_rms_norm`
- 再乘权重 `ggml_mul`

### 6.3 LongRoPE

使用：

- `ggml_rope_ext`
- `GGML_ROPE_TYPE_NEOX`

并根据 `seq_len` 与 `rope_original_max` 切换：

- `rope_short_factor`
- `rope_long_factor`

额外按长度计算 `attn_factor`。

### 6.4 GQA Attention

当前路径：

1. `q/k/v` 线性投影
2. reshape 成 head 形式
3. 对 `q/k` 应用 RoPE
4. 当前 token 的 `k/v` 写入 KV cache
5. 如果 `n_past > 0`，显式把 `past_kv` 与 `current_kv` 用 `ggml_concat` 拼接
6. 调用 `ggml_flash_attn_ext`
7. 经过 `o_proj`

关键实现细节：

- 不能只依赖 `ggml_cpy` 副作用来更新 cache
- 当前实现通过 `kv_sync` 把 cache 写操作挂到 attention 输出图上，保证图执行时写入生效

### 6.5 Causal Mask

当前 causal mask 使用显式构图：

- 构造 `key_positions`
- 构造 `query_positions`
- `ggml_repeat`
- `ggml_sub`
- `ggml_step`
- `ggml_scale(-1e9f)`

最终 cast 成 `F16` 传给 `ggml_flash_attn_ext`。

这是对早期广播构图错误的修复。

## 7. KV Cache 设计

### 7.1 内存规模

MiniCPM 配置：

- `n_heads = 16`
- `n_kv_heads = 2`
- `head_dim = 64`

则每层单个 cache：

- `64 * 32768 * 2 * 4 bytes = 16 MB`

每层 `K + V`：

- `32 MB`

因此：

- `base_lm` 24 层约 `768 MB`
- `residual_lm` 8 层约 `256 MB`

### 7.2 当前行为

`forward`：

- `n_past = 0`
- 将当前序列直接写入 cache 前缀

`forward_step`：

- `n_past = position`
- 先读取已有 cache 前缀
- 再与当前步 `k/v` 显式拼接
- 同时把当前步写回持久化 cache

## 8. 已有测试

测试文件：

- [tests/test_minicpm.cpp](/home/orangepi/Codes/ggbond/VoxCPM.cpp/tests/test_minicpm.cpp)

### 8.1 结构测试

已覆盖：

- `MiniCPMConfig defaults`
- `MiniCPMKVCache init and views`
- `BaseLM weights from GGUF`
- `variant-specific config from GGUF`
- `forward graph builds`
- `decode graph builds`

### 8.2 Trace 数值测试

已覆盖：

- `MiniCPM base_lm forward trace aligns with torch`
- `MiniCPM residual_lm forward trace aligns with torch`
- `MiniCPM base_lm forward_step trace aligns with torch`
- `MiniCPM residual_lm forward_step trace aligns with torch`

### 8.3 当前数值测试范围

`forward`：

- 对比 `output_0`
- 对比 cache 的首层和末层前缀

`forward_step`：

- 对比 `output_0`
- 对比全层、完整 `8192` 长度 `K cache`
- 对比全层、完整 `8192` 长度 `V cache`

其中 `forward_step` 使用的是 `forward_step_one` trace，但校验范围是全层完整 cache，不是抽样。

## 9. 当前容差口径

当前测试采用的是工程容差，而不是 ONNX 常见的 `atol + rtol allclose`：

- 单点绝对误差阈值：`0.05f`
- 允许超阈值元素比例：`<= 5%`

也就是说当前判定条件是：

`abs(actual - expected) <= 0.05f`

不满足该条件的点，整体比例不能超过 5%。

### 9.1 已观测结果

`base_lm.forward output`

- `max abs error = 0.525192`
- `avg abs error = 0.0082125`
- `mismatch rate (>0.05) = 0.679688%`

`residual_lm.forward output`

- `max abs error = 0.304274`
- `avg abs error = 0.0047021`
- `mismatch rate (>0.05) = 0.149414%`

`base_lm.forward_step output`

- `max abs error = 0.0319953`
- `avg abs error = 0.00465066`
- `mismatch rate (>0.05) = 0%`

`base_lm.forward_step full key cache`

- `max abs error = 0.030992`
- `mismatch rate (>0.05) = 0%`

`base_lm.forward_step full value cache`

- `max abs error = 0.0308211`
- `mismatch rate (>0.05) = 0%`

`residual_lm.forward_step output`

- `max abs error = 0.0592747`
- `avg abs error = 0.00418836`
- `mismatch rate (>0.05) = 0.0976562%`

`residual_lm.forward_step full key cache`

- `max abs error = 0.0213978`
- `mismatch rate (>0.05) = 0%`

`residual_lm.forward_step full value cache`

- `max abs error = 0.0121033`
- `mismatch rate (>0.05) = 0%`

解释：

- 绝大多数误差来自输出 tensor
- `forward_step` 的全量 cache 对齐已经比较稳定
- `forward` 输出如果改成严格全元素 `atol=0.05` 判定，目前不会通过
- 当前还没有切换到 `atol + rtol` 口径

## 10. 大 trace 解析策略

`forward_step` trace 文件很大：

- `trace_MiniCPM_base_lm_forward_step_one.jsonl` 约 `492 MB`
- `trace_MiniCPM_residual_lm_forward_step_one.jsonl` 约 `165 MB`

因此当前测试没有用 DOM 全量解析，而是使用 `nlohmann::json::sax_parse` 做流式解析，逐步提取：

- `inputs_embeds`
- `position_id`
- `past_key_values.layers[*].key`
- `past_key_values.layers[*].value`
- `outputs.output_0`
- `outputs.output_1.layers[*].key`
- `outputs.output_1.layers[*].value`

这样可以在测试里做完整 `forward_step` 校验而不依赖极大内存。

## 11. 已知未完成项

仍然未完成：

- `forward` 的全层全 cache 校验
- 连续多步 `forward_step` trace 校验
- ONNX 风格 `atol + rtol` 评分口径
- 更细的逐层中间激活对齐
- 上层 `voxcpm` 主模型路径中的更完整集成验证

## 12. 建议的后续工作

建议优先级：

1. 补 `atol + rtol` 版本的数值测试统计
2. 对 `trace_MiniCPM_*_forward_step.jsonl` 做多步连续解码对齐
3. 将 `forward` 的 cache 校验从首尾层扩展到全层
4. 在更高层 `voxcpm` 入口做集成级对齐测试

## 13. 当前结论

当前 MiniCPM GGML 模块已经达到“可加载、可构图、可运行、可做 trace 对齐”的状态。

结论分两部分：

- 工程实现层面：BaseLM / ResidualLM / LocEnc / LocDiT 的 MiniCPM 主干已经实现并通过本地测试
- 数值验证层面：`forward_step` 的输出和全量 KV cache 对齐结果已经较好，`forward` 输出也在当前放宽口径下通过

如果后续要把这套结果直接对接 ONNX 导出验证，需要再补一套明确的 `atol + rtol` 规则，而不是复用当前 `abs_tol + mismatch_rate` 规则。
