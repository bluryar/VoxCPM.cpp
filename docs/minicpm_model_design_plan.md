# MiniCPM GGML 模块完整实现计划

## Context

实现 MiniCPM Transformer 模块的 C++ GGML 版本，用于 VoxCPM 的 BaseLM (24层)、ResidualLM (8层)、LocEnc (8层)、LocDiT (8层)。

**参考实现**：
- Python GGML: `/home/orangepi/Codes/ggbond/examples/minicpm_ggml.py`
- PyTorch 原版: `/home/orangepi/Codes/ggbond/vendor/VoxCPM/src/voxcpm/modules/minicpm4/model.py`
- GPT-2 GGML: `/home/orangepi/Codes/ggbond/VoxCPM.cpp/third_party/ggml/examples/gpt-2/main-backend.cpp`
- llama.cpp KV Cache: `/home/orangepi/Codes/ggbond/vendor/llama.cpp/src/llama-kv-cache.cpp`

**复用模块**：FSQ (`src/fsq.cpp`), AudioVAE (`src/audio-vae.cpp`) 的实现模式。

---

## 一、架构概览

### 内存架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ggml_backend_t                              │
│                    (CPU / CUDA / Metal)                             │
└─────────────────────────────────────────────────────────────────────┘
         ↑                    ↑                    ↑
         │                    │                    │
    alloc_ctx            alloc_ctx            alloc_ctx
         │                    │                    │
┌────────┴────────┐  ┌────────┴────────┐  ┌────────┴────────┐
│    ctx_w        │  │    ctx_kv       │  │  ctx_graph      │
│   buffer_w      │  │   buffer_kv     │  │  (临时)         │
│                 │  │                 │  │                 │
│  [模型权重]      │  │  [KV Cache]     │  │  [计算图节点]    │
│  embed_tokens   │  │  k_cache[0..n]  │  │  每次推理重建    │
│  layers[].*     │  │  v_cache[0..n]  │  │                 │
│  lm_head        │  │                 │  │                 │
│                 │  │  持久化存储      │  │                 │
│  WEIGHTS 用途   │  │  跨调用保持      │  │                 │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

### 核心组件

| 组件 | 功能 | 关键实现 |
|------|------|----------|
| RMSNorm | 层归一化 | `x * rsqrt(mean(x²) + eps) * weight` |
| LongRoPE | 位置编码 | `ggml_rope_ext` + short/long factor 动态切换 |
| GQA Attention | 分组查询注意力 | Q heads=16, KV heads=2, `ggml_flash_attn_ext` |
| SwiGLU MLP | 前馈网络 | `down(silu(gate(x)) * up(x))` |
| muP Scaling | 残差缩放 | `scale_depth / sqrt(n_layers)` |
| KV Cache | 缓存优化 | 独立 Context + Buffer，`ggml_view_*` + `ggml_cpy` |

---

## 二、文件结构

```
VoxCPM.cpp/
├── include/voxcpm/
│   └── minicpm.h          # 新建 - 所有类定义
├── src/
│   └── minicpm.cpp        # 新建 - 实现
└── tests/
    └── test_minicpm.cpp   # 新建 - 测试
```

---

## 三、类设计

### 3.1 MiniCPMConfig (已存在于 config.h)

**GGUF Metadata Keys** (从 `models/voxcpm1.5.dump` 验证):

| GGUF Key | Config 字段 | 值 |
|----------|------------|-----|
| `llama.embedding_length` | hidden_size | 1024 |
| `llama.feed_forward_length` | intermediate_size | 4096 |
| `llama.block_count` | n_layer (BaseLM) | 24 |
| `llama.attention.head_count` | n_heads | 16 |
| `llama.attention.head_count_kv` | n_kv_heads | 2 |
| `llama.vocab_size` | vocab_size | 73448 |
| `llama.context_length` | max_length | 32768 |
| `llama.attention.layer_norm_rms_epsilon` | rms_norm_eps | 1e-5 |
| `llama.rope.freq_base` | rope_freq_base | 10000.0 |
| `voxcpm_lm_config_scale_emb` | scale_emb | 12 |
| `voxcpm_lm_config_scale_depth` | scale_depth | 1.4 |
| `voxcpm_lm_config_dim_model_base` | dim_model_base | 256 |
| `voxcpm_lm_config_use_mup` | use_mup | 0 (false) |
| `voxcpm_lm_config_rope_scaling_type` | rope_type | "longrope" |
| `voxcpm_lm_config_rope_scaling_original_max_position_embeddings` | rope_original_max | 32768 |
| `voxcpm_lm_config_rope_scaling_long_factor` | rope_long_factor | [32 floats] |
| `voxcpm_lm_config_rope_scaling_short_factor` | rope_short_factor | [32 floats] |
| `voxcpm_residual_lm_num_layers` | n_layer (ResidualLM) | 8 |

```cpp
struct MiniCPMConfig {
    int hidden_size = 1024;
    int intermediate_size = 4096;
    int n_layer = 8;
    int n_heads = 16;
    int n_kv_heads = 2;
    int vocab_size = 73448;
    int max_length = 32768;
    float rms_norm_eps = 1e-5f;
    float rope_freq_base = 10000.0f;
    int scale_emb = 12;
    float scale_depth = 1.4f;
    int rope_original_max = 32768;
    std::vector<float> rope_long_factor;
    std::vector<float> rope_short_factor;
    int head_dim() const { return hidden_size / n_heads; }
};
```

### 3.2 MiniCPMWeights

```cpp
struct MiniCPMLayerWeights {
    ggml_tensor* input_layernorm;    // [hidden_size]
    ggml_tensor* q_proj;             // [hidden_size, n_heads * head_dim]
    ggml_tensor* k_proj;             // [hidden_size, n_kv_heads * head_dim]
    ggml_tensor* v_proj;             // [hidden_size, n_kv_heads * head_dim]
    ggml_tensor* o_proj;             // [n_heads * head_dim, hidden_size]
    ggml_tensor* post_layernorm;     // [hidden_size]
    ggml_tensor* gate_proj;          // [hidden_size, intermediate_size]
    ggml_tensor* up_proj;            // [hidden_size, intermediate_size]
    ggml_tensor* down_proj;          // [intermediate_size, hidden_size]
};

struct MiniCPMWeights {
    ggml_tensor* embed_tokens = nullptr;  // [vocab_size, hidden_size]
    std::vector<MiniCPMLayerWeights> layers;
    ggml_tensor* norm = nullptr;          // [hidden_size]
    ggml_tensor* lm_head = nullptr;       // [hidden_size, vocab_size] (可选)
};
```

### 3.3 MiniCPMKVCache

```cpp
class MiniCPMKVCache {
public:
    MiniCPMKVCache(int n_layer, int n_kv_heads, int max_length, int head_dim);
    ~MiniCPMKVCache();

    // 初始化 (在模型加载后调用)
    void init(ggml_backend_t backend);

    // 清空 Cache
    void clear();

    // === 读取接口 (用于 Attention) ===

    // 获取 K/V 视图用于 Attention
    // 返回: [head_dim, seq_len, n_kv_heads] 的视图
    // 在 graph context 中调用
    ggml_tensor* get_k(ggml_context* ctx, int layer, int seq_len);
    ggml_tensor* get_v(ggml_context* ctx, int layer, int seq_len);

    // === 写入接口 ===

    // 获取单个位置用于写入 (Decode 模式)
    // 返回: [head_dim * n_kv_heads, 1] 的 2D 视图
    ggml_tensor* get_k_slot(ggml_context* ctx, int layer, int position);
    ggml_tensor* get_v_slot(ggml_context* ctx, int layer, int position);

    // 获取批量写入位置 (Prefill 模式)
    // 返回: [head_dim * n_kv_heads, n_tokens] 的 2D 视图，从 position 0 开始
    ggml_tensor* get_k_batch(ggml_context* ctx, int layer, int n_tokens);
    ggml_tensor* get_v_batch(ggml_context* ctx, int layer, int n_tokens);

    int max_length() const { return max_length_; }
    int n_layer() const { return n_layer_; }

private:
    int n_layer_;
    int n_kv_heads_;
    int max_length_;
    int head_dim_;

    // 独立的 Context 和 Buffer (持久化)
    ggml_context* ctx_kv_ = nullptr;
    ggml_backend_buffer_t buffer_kv_ = nullptr;

    // 每层的 K/V Cache 张量
    // 形状: [head_dim, max_length, n_kv_heads]
    // 例如: [64, 32768, 2] for MiniCPM
    std::vector<ggml_tensor*> k_caches_;
    std::vector<ggml_tensor*> v_caches_;
};
```

### 3.4 MiniCPMModel

```cpp
class MiniCPMModel {
public:
    explicit MiniCPMModel(const MiniCPMConfig& config);
    ~MiniCPMModel();

    // 权重加载
    bool load_from_gguf(gguf_context* gguf_ctx,
                         const std::string& prefix,  // "", "residual_lm.", "locenc.", "locdit."
                         ggml_context* weight_ctx,
                         ggml_backend_t backend);

    // === 推理接口 ===

    // Prefill 模式: 处理整个序列
    // input: [hidden_size, seq_len, batch]
    // output: [hidden_size, seq_len, batch]
    // positions: [seq_len] 位置索引
    // kv_cache: 用于存储 K/V
    ggml_tensor* forward(ggml_context* ctx,
                          ggml_tensor* input,
                          ggml_tensor* positions,
                          MiniCPMKVCache& kv_cache,
                          int n_tokens,
                          bool is_causal = true);

    // Decode 模式: 单步生成
    // input: [hidden_size, batch]
    // output: [hidden_size, batch]
    // position: 当前位置
    ggml_tensor* forward_step(ggml_context* ctx,
                               ggml_tensor* input,
                               int position,
                               MiniCPMKVCache& kv_cache,
                               bool is_causal = true);

    // === 访问器 ===
    const MiniCPMConfig& config() const { return config_; }
    const MiniCPMWeights& weights() const { return weights_; }
    ggml_tensor* get_pos_tensor() const { return pos_tensor_; }

private:
    MiniCPMConfig config_;
    MiniCPMWeights weights_;
    ggml_backend_buffer_t weight_buffer_ = nullptr;

    // RoPE factors (持久化)
    ggml_tensor* rope_long_factor_ = nullptr;
    ggml_tensor* rope_short_factor_ = nullptr;
    ggml_backend_buffer_t rope_buffer_ = nullptr;

    // 位置张量 (复用)
    ggml_tensor* pos_tensor_ = nullptr;

    // 预计算值
    float residual_scale_ = 1.0f;

    // === 内部方法 ===

    ggml_tensor* layer_forward(ggml_context* ctx,
                                ggml_tensor* hidden,
                                ggml_tensor* positions,
                                const MiniCPMLayerWeights& lw,
                                MiniCPMKVCache& kv_cache,
                                int layer_idx,
                                int n_tokens,
                                int n_past,
                                bool is_causal);

    ggml_tensor* attention_forward(ggml_context* ctx,
                                    ggml_tensor* hidden,
                                    ggml_tensor* positions,
                                    const MiniCPMLayerWeights& lw,
                                    MiniCPMKVCache& kv_cache,
                                    int layer_idx,
                                    int n_tokens,
                                    int n_past,
                                    bool is_causal);

    ggml_tensor* mlp_forward(ggml_context* ctx,
                              ggml_tensor* hidden,
                              const MiniCPMLayerWeights& lw);

    ggml_tensor* apply_rope(ggml_context* ctx,
                             ggml_tensor* x,
                             ggml_tensor* positions,
                             int seq_len);

    void init_rope_factors(ggml_context* ctx, ggml_backend_t backend);
    void init_pos_tensor(ggml_context* ctx, ggml_backend_t backend);
};
```

---

## 四、GGUF 权重命名 (已验证)

从 `models/voxcpm1.5.dump` 验证的实际 GGUF 张量名称：

### BaseLM (24层, 无前缀)

| GGUF Tensor Name | MiniCPMLayerWeights 字段 | 形状 |
|-----------------|-------------------------|------|
| `token_embd.weight` | embed_tokens | [1024, 73448] |
| `blk.{i}.attn_norm.weight` | input_layernorm | [1024] |
| `blk.{i}.ffn_norm.weight` | post_layernorm | [1024] |
| `blk.{i}.attn_q.weight` | q_proj | [1024, 1024] |
| `blk.{i}.attn_k.weight` | k_proj | [1024, 128] |
| `blk.{i}.attn_v.weight` | v_proj | [1024, 128] |
| `blk.{i}.attn_output.weight` | o_proj | [1024, 1024] |
| `blk.{i}.ffn_gate.weight` | gate_proj | [1024, 4096] |
| `blk.{i}.ffn_up.weight` | up_proj | [1024, 4096] |
| `blk.{i}.ffn_down.weight` | down_proj | [4096, 1024] |
| `output_norm.weight` | norm | [1024] |

### ResidualLM (8层, `residual_lm.` 前缀)

| GGUF Tensor Name | 形状 |
|-----------------|------|
| `residual_lm.blk.{i}.attn_norm.weight` | [1024] |
| `residual_lm.blk.{i}.ffn_norm.weight` | [1024] |
| `residual_lm.blk.{i}.attn_q.weight` | [1024, 1024] |
| `residual_lm.blk.{i}.attn_k.weight` | [1024, 128] |
| `residual_lm.blk.{i}.attn_v.weight` | [1024, 128] |
| `residual_lm.blk.{i}.attn_output.weight` | [1024, 1024] |
| `residual_lm.blk.{i}.ffn_gate.weight` | [1024, 4096] |
| `residual_lm.blk.{i}.ffn_up.weight` | [1024, 4096] |
| `residual_lm.blk.{i}.ffn_down.weight` | [4096, 1024] |
| `residual_lm.output_norm.weight` | [1024] |

### LocEnc (8层, `locenc.` 前缀)

| GGUF Tensor Name | 形状 | 说明 |
|-----------------|------|------|
| `locenc.in_proj.weight` | [64, 1024] | 输入投影 |
| `locenc.in_proj.bias` | [1024] | |
| `locenc.special_token` | [1024] | 特殊 token |
| `locenc.blk.{i}.*` | - | 同上 |
| `locenc.output_norm.weight` | [1024] | |

### LocDiT (8层, `locdit.` 前缀)

| GGUF Tensor Name | 形状 | 说明 |
|-----------------|------|------|
| `locdit.cond_proj.weight` | [64, 1024] | 条件投影 |
| `locdit.cond_proj.bias` | [1024] | |
| `locdit.blk.{i}.*` | - | 同上 |

---

## 五、关键实现细节

### 5.1 张量布局约定

```
输入/输出:    [hidden_size, seq_len, batch]     (GGML 约定)
Q/K/V 投影后: [head_dim, n_heads, seq_len]      (RoPE 前)
Flash Attn:   [head_dim, seq_len, n_heads]      (flash_attn_ext 输入)
KV Cache:     [head_dim, max_len, n_kv_heads]   (持久化存储)
```

### 5.2 KV Cache 内存布局

MiniCPM 的 GQA 配置: n_heads=16, n_kv_heads=2, head_dim=64

```
K/V Cache 形状: [head_dim, max_len, n_kv_heads] = [64, 32768, 2]

内存大小计算:
- 每层 K Cache: 64 * 32768 * 2 * 4 bytes = 16 MB
- 每层 V Cache: 64 * 32768 * 2 * 4 bytes = 16 MB
- 24层总计: 24 * 32 MB = 768 MB (BaseLM)
- 8层总计: 8 * 32 MB = 256 MB (ResidualLM/LocEnc/LocDiT)
```

### 5.3 RMSNorm 实现

```cpp
ggml_tensor* rms_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight, float eps) {
    // x: [hidden_size, seq_len]
    // output: [hidden_size, seq_len]
    ggml_tensor* norm = ggml_rms_norm(ctx, x, eps);
    return ggml_mul(ctx, norm, weight);
}
```

### 5.4 LongRoPE 实现

```cpp
ggml_tensor* MiniCPMModel::apply_rope(ggml_context* ctx, ggml_tensor* x,
                                       ggml_tensor* positions, int seq_len) {
    float attn_factor = 1.0f;
    if (seq_len > config_.rope_original_max) {
        float scale = (float)seq_len / config_.rope_original_max;
        attn_factor = sqrtf(1.0f + logf(scale) / logf((float)config_.rope_original_max));
    }

    // 选择 short/long factor
    ggml_tensor* freq_factors = (seq_len <= config_.rope_original_max)
        ? rope_short_factor_
        : rope_long_factor_;

    return ggml_rope_ext(ctx, x, positions, freq_factors,
                         config_.head_dim(),           // n_dims
                         GGML_ROPE_TYPE_NEOX,          // mode
                         config_.rope_original_max,    // freq_base 通过 original_max 计算
                         config_.rope_freq_base,
                         1.0f,                         // freq_scale
                         0.0f,                         // ext_factor (LongRoPE 不用 YaRN)
                         attn_factor,
                         32.0f,                        // beta_fast
                         1.0f                          // beta_slow
                         );
}
```

### 5.5 GQA Attention 实现

```cpp
ggml_tensor* MiniCPMModel::attention_forward(
    ggml_context* ctx,
    ggml_tensor* hidden,          // [hidden_size, n_tokens]
    ggml_tensor* positions,       // [n_tokens]
    const MiniCPMLayerWeights& lw,
    MiniCPMKVCache& kv_cache,
    int layer_idx,
    int n_tokens,
    int n_past,
    bool is_causal)
{
    const int head_dim = config_.head_dim();
    const int n_heads = config_.n_heads;
    const int n_kv_heads = config_.n_kv_heads;

    // 1. Q/K/V 投影
    ggml_tensor* q = ggml_mul_mat(ctx, lw.q_proj, hidden);  // [n_heads*head_dim, n_tokens]
    ggml_tensor* k = ggml_mul_mat(ctx, lw.k_proj, hidden);  // [n_kv_heads*head_dim, n_tokens]
    ggml_tensor* v = ggml_mul_mat(ctx, lw.v_proj, hidden);  // [n_kv_heads*head_dim, n_tokens]

    // 2. Reshape: [head_dim, n_heads, n_tokens]
    q = ggml_reshape_3d(ctx, q, head_dim, n_heads, n_tokens);
    k = ggml_reshape_3d(ctx, k, head_dim, n_kv_heads, n_tokens);
    v = ggml_reshape_3d(ctx, v, head_dim, n_kv_heads, n_tokens);

    // 3. Apply RoPE
    q = apply_rope(ctx, q, positions, n_tokens);
    k = apply_rope(ctx, k, positions, n_tokens);

    // 4. 存储 K/V 到 Cache
    {
        // 获取 Cache 写入位置
        ggml_tensor* k_cache_slot = kv_cache.get_k_batch(ctx, layer_idx, n_tokens);
        ggml_tensor* v_cache_slot = kv_cache.get_v_batch(ctx, layer_idx, n_tokens);

        // Reshape k/v 为 [head_dim * n_kv_heads, n_tokens] 用于复制
        k = ggml_reshape_2d(ctx, k, head_dim * n_kv_heads, n_tokens);
        v = ggml_reshape_2d(ctx, v, head_dim * n_kv_heads, n_tokens);

        // 使用 ggml_cpy 复制到 Cache
        ggml_build_forward_expand(gf, ggml_cpy(ctx, k, k_cache_slot));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, v, v_cache_slot));
    }

    // 5. 读取完整 K/V Cache 用于 Attention
    const int total_len = n_past + n_tokens;
    ggml_tensor* k_all = kv_cache.get_k(ctx, layer_idx, total_len);  // [head_dim, total_len, n_kv_heads]
    ggml_tensor* v_all = kv_cache.get_v(ctx, layer_idx, total_len);

    // 6. Permute for flash_attn_ext: [head_dim, seq, n_heads]
    q = ggml_permute(ctx, q, 0, 2, 1, 3);      // [head_dim, n_tokens, n_heads]
    k_all = ggml_permute(ctx, k_all, 0, 2, 1, 3);  // [head_dim, total_len, n_kv_heads]
    v_all = ggml_permute(ctx, v_all, 0, 2, 1, 3);

    // 7. Flash Attention (GQA 自动处理: K/V 广播到所有 Q heads)
    float scale = 1.0f / sqrtf((float)head_dim);
    ggml_tensor* attn = ggml_flash_attn_ext(ctx, q, k_all, v_all,
        nullptr,  // mask (causal 通过 n_past 实现)
        scale,
        0.0f,     // max_bias
        0.0f      // logit_softcap
    );

    // 8. Output projection
    attn = ggml_reshape_2d(ctx, attn, n_heads * head_dim, n_tokens);
    return ggml_mul_mat(ctx, lw.o_proj, attn);
}
```

### 5.6 SwiGLU MLP 实现

```cpp
ggml_tensor* MiniCPMModel::mlp_forward(ggml_context* ctx,
                                         ggml_tensor* hidden,
                                         const MiniCPMLayerWeights& lw) {
    // hidden: [hidden_size, seq_len]

    // gate = silu(gate_proj(x))
    ggml_tensor* gate = ggml_mul_mat(ctx, lw.gate_proj, hidden);
    gate = ggml_silu(ctx, gate);

    // up = up_proj(x)
    ggml_tensor* up = ggml_mul_mat(ctx, lw.up_proj, hidden);

    // hidden = gate * up
    ggml_tensor* mlp_hidden = ggml_mul(ctx, gate, up);

    // output = down_proj(hidden)
    return ggml_mul_mat(ctx, lw.down_proj, mlp_hidden);
}
```

### 5.7 Layer Forward 实现

```cpp
ggml_tensor* MiniCPMModel::layer_forward(
    ggml_context* ctx,
    ggml_tensor* hidden,
    ggml_tensor* positions,
    const MiniCPMLayerWeights& lw,
    MiniCPMKVCache& kv_cache,
    int layer_idx,
    int n_tokens,
    int n_past,
    bool is_causal)
{
    // 1. Attention
    ggml_tensor* residual = hidden;
    hidden = rms_norm(ctx, hidden, lw.input_layernorm, config_.rms_norm_eps);
    hidden = attention_forward(ctx, hidden, positions, lw, kv_cache,
                                layer_idx, n_tokens, n_past, is_causal);
    hidden = ggml_add(ctx, hidden, residual);

    // 2. MLP
    residual = hidden;
    hidden = rms_norm(ctx, hidden, lw.post_layernorm, config_.rms_norm_eps);
    hidden = mlp_forward(ctx, hidden, lw);
    hidden = ggml_add(ctx, hidden, residual);

    // 3. muP Residual Scaling (如果启用)
    if (config_.use_mup && residual_scale_ != 1.0f) {
        // scale = scale_depth / sqrt(n_layers)
        // 注意: 这里应用于残差连接后，具体逻辑需参考 PyTorch 实现
    }

    return hidden;
}
```

### 5.8 Decode 模式 Attention

```cpp
ggml_tensor* attention_forward_decode(
    ggml_context* ctx,
    ggml_tensor* hidden,          // [hidden_size, 1]
    ggml_tensor* positions,       // [1]
    const MiniCPMLayerWeights& lw,
    MiniCPMKVCache& kv_cache,
    int layer_idx,
    int position)                 // 当前位置
{
    // ... Q/K/V 投影和 RoPE (同 Prefill) ...

    // 存储 K/V 到单个位置
    {
        ggml_tensor* k_slot = kv_cache.get_k_slot(ctx, layer_idx, position);
        ggml_tensor* v_slot = kv_cache.get_v_slot(ctx, layer_idx, position);

        // k: [head_dim * n_kv_heads, 1]
        k = ggml_reshape_2d(ctx, k, head_dim * n_kv_heads, 1);
        v = ggml_reshape_2d(ctx, v, head_dim * n_kv_heads, 1);

        ggml_build_forward_expand(gf, ggml_cpy(ctx, k, k_slot));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, v, v_slot));
    }

    // 读取所有历史 K/V
    ggml_tensor* k_all = kv_cache.get_k(ctx, layer_idx, position + 1);
    ggml_tensor* v_all = kv_cache.get_v(ctx, layer_idx, position + 1);

    // ... Flash Attention ...
}
```

---

## 六、KV Cache 设计详解

### 6.1 核心设计原则

| 原则 | 说明 |
|------|------|
| **独立 Context** | KV Cache 使用独立的 `ctx_kv`，不与权重或计算图共享 |
| **持久化 Buffer** | `buffer_kv` 跨多次推理调用保持，不随 graph 销毁 |
| **视图访问** | 通过 `ggml_view_*` 创建指向 Cache 的视图，每次推理重建 |
| **no_alloc=true** | Context 只存元数据，数据由 Buffer 管理 |

### 6.2 KV Cache 初始化

```cpp
void MiniCPMKVCache::init(ggml_backend_t backend) {
    // 1. 创建独立 Context (仅元数据)
    size_t ctx_size = ggml_tensor_overhead() * n_layer_ * 2 + 1024;
    ggml_init_params params = {
        .mem_size   = ctx_size,
        .mem_buffer = NULL,
        .no_alloc   = true,  // 必须！
    };
    ctx_kv_ = ggml_init(params);

    // 2. 创建每层的 K/V 张量
    for (int i = 0; i < n_layer_; i++) {
        // 形状: [head_dim, max_length, n_kv_heads]
        // 例如: [64, 32768, 2]
        k_caches_.push_back(ggml_new_tensor_3d(ctx_kv_, GGML_TYPE_F32,
            head_dim_, max_length_, n_kv_heads_));
        v_caches_.push_back(ggml_new_tensor_3d(ctx_kv_, GGML_TYPE_F32,
            head_dim_, max_length_, n_kv_heads_));
    }

    // 3. 分配独立 Buffer
    buffer_kv_ = ggml_backend_alloc_ctx_tensors(ctx_kv_, backend);
}
```

### 6.3 KV Cache 视图访问

```cpp
// 读取视图 (用于 Attention)
ggml_tensor* MiniCPMKVCache::get_k(ggml_context* ctx, int layer, int seq_len) {
    ggml_tensor* k = k_caches_[layer];
    // 创建 3D 视图: [head_dim, seq_len, n_kv_heads]
    // 只包含前 seq_len 个位置
    return ggml_view_3d(ctx, k,
        head_dim_, seq_len, n_kv_heads_,
        k->nb[1],  // stride in dim 1
        k->nb[2],  // stride in dim 2
        0          // offset from start
    );
}

// 写入位置 (Prefill 模式)
ggml_tensor* MiniCPMKVCache::get_k_batch(ggml_context* ctx, int layer, int n_tokens) {
    ggml_tensor* k = k_caches_[layer];
    // 创建 2D 视图: [head_dim * n_kv_heads, n_tokens]
    return ggml_view_2d(ctx, k,
        head_dim_ * n_kv_heads_,
        n_tokens,
        k->nb[1],  // row stride
        0          // offset from start
    );
}

// 写入位置 (Decode 模式)
ggml_tensor* MiniCPMKVCache::get_k_slot(ggml_context* ctx, int layer, int position) {
    ggml_tensor* k = k_caches_[layer];
    // 创建 2D 视图: [head_dim * n_kv_heads, 1]
    size_t offset = position * k->nb[1];
    return ggml_view_2d(ctx, k,
        head_dim_ * n_kv_heads_,
        1,
        k->nb[1],
        offset
    );
}
```

### 6.4 Graph 生命周期

```cpp
// 典型推理循环
while (generating) {
    // 1. 创建临时 graph context
    ggml_context* ctx = create_graph_context();

    // 2. 构建计算图 (使用 ggml_view 引用 KV Cache)
    //    注意: 视图是 graph context 的一部分
    ggml_cgraph* graph = build_forward_graph(ctx, model, kv_cache, ...);

    // 3. 分配计算图 Buffer
    ggml_backend_sched_alloc_graph(sched, graph);

    // 4. 设置输入数据
    set_input_tensors(graph, ...);

    // 5. 执行
    ggml_backend_sched_graph_compute(sched, graph);

    // 6. 重置 scheduler
    ggml_backend_sched_reset(sched);

    // 7. 销毁 graph context (视图随之销毁)
    ggml_free(ctx);

    // 注意: KV Cache Buffer 保持不变！
}
```

---

## 七、实现步骤

### Phase 1: 核心实现 (src/minicpm.cpp)

1. **MiniCPMKVCache 类**
   - 构造函数/析构函数
   - `init()` - 分配 KV Cache Buffer
   - `get_k()`/`get_v()` - 读取视图
   - `get_k_slot()`/`get_v_slot()` - 写入位置
   - `get_k_batch()`/`get_v_batch()` - 批量写入

2. **MiniCPMModel 类**
   - 构造函数/析构函数
   - `load_from_gguf()` - 权重加载
   - `init_rope_factors()` - 初始化 RoPE factors
   - `rms_norm()` - RMSNorm
   - `apply_rope()` - LongRoPE
   - `attention_forward()` - Attention 前向
   - `mlp_forward()` - MLP 前向
   - `layer_forward()` - 单层前向
   - `forward()` - Prefill 模式
   - `forward_step()` - Decode 模式

### Phase 2: 测试验证 (tests/test_minicpm.cpp)

1. 权重加载测试
2. KV Cache 初始化和内存分配
3. RMSNorm 单元测试
4. RoPE 单元测试
5. KV Cache 读写测试
6. Attention 单元测试
7. MLP 单元测试
8. 完整 forward 测试 (Prefill + Decode)
9. Trace 对比验证 (与 Python 实现)

---

## 八、验证方法

```bash
cd /home/orangepi/Codes/ggbond/VoxCPM.cpp
cmake -B build && cmake --build build
cd build && ctest -R test_minicpm --output-on-failure
```

---

## 九、关键注意事项

### 必须遵守

1. **no_alloc=true**: 所有 Context 必须使用 `no_alloc=true`
2. **ggml_set_input**: 标记所有输入张量
3. **ggml_set_output**: 标记所有输出张量
4. **权重 Buffer 独立**: 使用 `GGML_BACKEND_BUFFER_USAGE_WEIGHTS`
5. **KV Cache Buffer 独立**: 单独的 Context 和 Buffer

### 注意事项

1. **LongRoPE**: 根据 seq_len 选择 short/long factor
2. **GQA**: K/V 头数 < Q 头数，`ggml_flash_attn_ext` 自动处理广播
3. **muP Scaling**: `scale_depth / sqrt(n_layers)` 应用于残差连接
4. **Decode 优化**: 避免图依赖链导致的性能下降
5. **KV Cache 偏移**: Decode 模式需要正确计算 position offset

### 调试技巧

```cpp
// 打印张量信息
void print_tensor_info(const char* name, ggml_tensor* t) {
    printf("%s: shape=[%lld, %lld, %lld, %lld], type=%s, nb=[%zu, %zu, %zu, %zu]\n",
           name, t->ne[0], t->ne[1], t->ne[2], t->ne[3],
           ggml_type_name(t->type), t->nb[0], t->nb[1], t->nb[2], t->nb[3]);
}

// 验证张量数据
void verify_tensor_data(ggml_tensor* t, const std::vector<float>& expected, float eps = 1e-5) {
    std::vector<float> actual(ggml_nelements(t));
    ggml_backend_tensor_get(t, actual.data(), 0, ggml_nbytes(t));
    for (size_t i = 0; i < actual.size(); i++) {
        if (fabsf(actual[i] - expected[i]) > eps) {
            printf("Mismatch at index %zu: expected %f, got %f\n", i, expected[i], actual[i]);
        }
    }
}
```
