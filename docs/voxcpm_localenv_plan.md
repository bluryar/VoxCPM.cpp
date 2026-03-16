# LocEnc (Local Encoder) GGML 模块实现计划

## Context

实现 LocEnc（Local Encoder）模块的 C++ GGML 版本。LocEnc 是 VoxCPM 的关键组件，负责将音频特征编码为局部表示。

**参考实现**：
- Python: `/home/orangepi/Codes/ggbond/vendor/VoxCPM/src/voxcpm/modules/locenc/local_encoder.py`
- 复用模块: `MiniCPMModel` ([src/minicpm.cpp](../VoxCPM.cpp/src/minicpm.cpp))

**与 MiniCPM 的关系**：
- LocEnc 内部使用 `MiniCPMModel` 作为核心 Transformer
- 主要区别：输入投影层 + 特殊 CLS token + 非因果注意力

---

## 一、架构分析

### 1.0 PyTorch → GGML 算子映射

分析 Python 实现，列出所需算子及 GGML 对应实现：

#### LocEnc 特有操作 (local_encoder.py)

| PyTorch 操作 | 功能 | GGML 算子 | 签名 |
|-------------|------|----------|------|
| `nn.Linear(input_dim, hidden_size)` | 输入投影 | `ggml_mul_mat` + `ggml_add` | `(ctx, weight, input)` + `(ctx, result, bias)` |
| `tensor.expand(B, T, 1, -1)` | 扩展 special_token | `ggml_repeat` | `(ctx, src, target)` |
| `torch.cat([cls, x], dim=2)` | 拼接 CLS token | `ggml_concat` | `(ctx, a, b, dim)` |
| `rearrange(x, "b t p c -> (b t) p c")` | 重塑形状 | `ggml_reshape_3d` / `ggml_view_3d` | `(ctx, x, ...)` |
| `outputs[:, 0, :]` | 提取 CLS 输出 | `ggml_view_1d` / `ggml_view_2d` | `(ctx, x, ne0, offset)` |

#### MiniCPM Transformer 操作 (model.py)

| PyTorch 操作 | 功能 | GGML 算子 | 使用位置 |
|-------------|------|----------|---------|
| **RMSNorm** | | | |
| `hidden.pow(2).mean(-1)` | 均值平方 | `ggml_rms_norm` | 已封装 |
| `* weight` | 缩放 | `ggml_mul` | RMSNorm 内部 |
| **RoPE** | | | |
| `rotate_half` | 旋转半边 | `ggml_rope_ext` 内部实现 | 已封装 |
| `cos`, `sin` | 三角函数 | `ggml_rope_ext` 内部计算 | 已封装 |
| **Attention** | | | |
| `nn.Linear` (q/k/v/o) | 线性投影 | `ggml_mul_mat` | Q/K/V 投影 |
| `view`, `transpose` | 形状变换 | `ggml_reshape_3d`, `ggml_permute` | 多头重塑 |
| `scaled_dot_product_attention` | 缩放点积注意力 | `ggml_flash_attn_ext` | 核心 Attention |
| **MLP (SwiGLU)** | | | |
| `nn.SiLU` | SiLU 激活 | `ggml_silu` | gate 分支 |
| `gate * up` | 元素乘 | `ggml_mul` | 门控融合 |
| `down_proj` | 下投影 | `ggml_mul_mat` | 输出投影 |
| **残差连接** | | | |
| `residual + hidden` | 加法 | `ggml_add` | 残差连接 |
| `* scale` | 缩放 (muP) | `ggml_scale` | 可选缩放 |

#### GGML 算子完整签名参考

```cpp
// 线性投影
ggml_tensor* ggml_mul_mat(ctx, weight, input);  // [M, K] @ [K, N] -> [M, N]
ggml_tensor* ggml_add(ctx, a, b);               // element-wise add

// 形状操作
ggml_tensor* ggml_reshape_2d(ctx, x, ne0, ne1);
ggml_tensor* ggml_reshape_3d(ctx, x, ne0, ne1, ne2);
ggml_tensor* ggml_view_1d(ctx, x, ne0, offset);
ggml_tensor* ggml_view_2d(ctx, x, ne0, ne1, stride, offset);
ggml_tensor* ggml_permute(ctx, x, axis0, axis1, axis2, axis3);

// 拼接/复制
ggml_tensor* ggml_concat(ctx, a, b, dim);       // 沿 dim 拼接
ggml_tensor* ggml_repeat(ctx, src, target);     // 广播复制

// 归一化
ggml_tensor* ggml_rms_norm(ctx, x, eps);        // RMSNorm

// 激活函数
ggml_tensor* ggml_silu(ctx, x);                 // SiLU/Swish

// Attention
ggml_tensor* ggml_flash_attn_ext(ctx, q, k, v, mask, scale, max_bias, logit_softcap);

// RoPE
ggml_tensor* ggml_rope_ext(ctx, x, positions, freq_factors, n_dims, mode, ...);

// 算术运算
ggml_tensor* ggml_mul(ctx, a, b);               // element-wise multiply
ggml_tensor* ggml_scale(ctx, a, scale);         // 标量缩放
```

### 1.1 复用 MiniCPMModel 策略

**关键发现**: `minicpm.cpp` 中的 `MiniCPMModel` 可以直接复用！

已验证的复用点：

| 功能 | 复用方式 | 已实现位置 |
|------|---------|-----------|
| Transformer 层 | 直接使用 `MiniCPMModel` | `minicpm.cpp:564-582` |
| 非因果注意力 | `forward(..., is_causal=false)` | `minicpm.cpp:568` |
| GGUF 权重加载 | `load_from_gguf("locenc", ...)` | `minicpm.cpp:280-284` |
| KV Cache | `MiniCPMKVCache` | `minicpm.cpp:113-215` |

**minicpm.cpp 已支持的 LocEnc 配置读取**:
```cpp
// minicpm.cpp:280-284
} else if (prefix == "locenc.") {
    if (get_u32_kv(gguf_ctx, "voxcpm_encoder_config_hidden_dim", u32)) config_.hidden_size = u32;
    if (get_u32_kv(gguf_ctx, "voxcpm_encoder_config_ffn_dim", u32)) config_.intermediate_size = u32;
    if (get_u32_kv(gguf_ctx, "voxcpm_encoder_config_num_heads", u32)) config_.n_heads = u32;
    if (get_u32_kv(gguf_ctx, "voxcpm_encoder_config_num_layers", u32)) config_.n_layer = u32;
}
```

**结论**: LocEnc 实现只需：
1. 加载特有权重 (`in_proj`, `special_token`)
2. 实现 `forward_patch()` 中的输入投影和 CLS token 拼接
3. 调用 `MiniCPMModel::forward(ctx, input, nullptr, kv_cache, false)`

### 1.2 Python 实现逻辑

```python
class VoxCPMLocEnc(nn.Module):
    def __init__(self, config, input_dim=64):
        self.special_token = nn.Parameter(...)  # [1, 1, 1, hidden_size]
        self.in_proj = nn.Linear(input_dim, config.hidden_size, bias=True)
        self.encoder = MiniCPMModel(config)  # vocab_size=0

    def forward(self, x):
        # x: [B, T, P, D] - batch, time, patches, dim
        x = self.in_proj(x)                    # [B, T, P, hidden_size]
        x = torch.cat([special_tokens, x], 2)  # [B, T, P+1, hidden_size]
        x = rearrange(x, "b t p c -> (b t) p c")  # 展平 batch 和 time
        outputs, _ = self.encoder(x, is_causal=False)  # 非因果！
        return outputs[:, 0, :]  # 返回 CLS token 输出
```

### 1.3 LocEnc vs BaseLM 关键区别

| 特性 | BaseLM | LocEnc |
|-----|--------|--------|
| 输入来源 | Token embedding | 音频特征投影 |
| 输入维度 | vocab_size | 64 (feat_dim) |
| 特殊 token | 无 | CLS token (prepend) |
| 注意力类型 | 因果 (causal) | 非因果 (non-causal) |
| 输出 | 完整序列 | 仅 CLS token |
| 权重前缀 | "" | "locenc." |

### 1.4 权重清单 (GGUF - 已验证)

从 `models/voxcpm1.5.dump` 提取的实际张量名称：

| GGUF Key | 形状 | 字节数 | 说明 |
|----------|------|--------|------|
| `locenc.in_proj.weight` | [64, 1024] | 65536 | 输入投影权重 (feat_dim → hidden_size) |
| `locenc.in_proj.bias` | [1024] | 1024 | 输入投影偏置 |
| `locenc.special_token` | [1024] | 1024 | CLS 特殊 token |
| `locenc.blk.{i}.attn_norm.weight` | [1024] | 1024 | Attention 层归一化 |
| `locenc.blk.{i}.attn_q.weight` | [1024, 1024] | 1048576 | Q 投影 |
| `locenc.blk.{i}.attn_k.weight` | [1024, 128] | 131072 | K 投影 (GQA: n_kv_heads=2) |
| `locenc.blk.{i}.attn_v.weight` | [1024, 128] | 131072 | V 投影 |
| `locenc.blk.{i}.attn_output.weight` | [1024, 1024] | 1048576 | Output 投影 |
| `locenc.blk.{i}.ffn_norm.weight` | [1024] | 1024 | FFN 层归一化 |
| `locenc.blk.{i}.ffn_gate.weight` | [1024, 4096] | 4194304 | Gate 投影 |
| `locenc.blk.{i}.ffn_up.weight` | [1024, 4096] | 4194304 | Up 投影 |
| `locenc.blk.{i}.ffn_down.weight` | [4096, 1024] | 4194304 | Down 投影 |
| `locenc.output_norm.weight` | [1024] | 1024 | 输出归一化 |

**注意**: Transformer 层权重共 8 层 (i=0..7)，与 MiniCPM 权重命名一致。

---

## 二、文件结构

```
VoxCPM.cpp/
├── include/voxcpm/
│   └── localenc.h          # 新建 - LocEnc 类定义
├── src/
│   └── localenc.cpp        # 新建 - LocEnc 实现
└── tests/
    └── test_localenc.cpp   # 新建 - 测试
```

---

## 三、类设计

### 3.1 LocEncWeights

```cpp
struct LocEncWeights {
    ggml_tensor* in_proj_weight = nullptr;   // [hidden_size, feat_dim] = [1024, 64]
    ggml_tensor* in_proj_bias = nullptr;     // [hidden_size] = [1024]
    ggml_tensor* special_token = nullptr;    // [hidden_size] = [1024]
};
```

### 3.2 LocEncModel

**设计原则**: 组合模式，内部持有 `MiniCPMModel` 实例。

```cpp
class LocEncModel {
public:
    explicit LocEncModel() = default;  // 配置从 GGUF 加载
    ~LocEncModel();

    LocEncModel(const LocEncModel&) = delete;
    LocEncModel& operator=(const LocEncModel&) = delete;

    // 权重加载 - 复用 MiniCPMModel::load_from_gguf("locenc", ...)
    bool load_from_gguf(const std::string& gguf_path,
                        VoxCPMContext& weight_ctx,
                        VoxCPMContext& graph_ctx,
                        VoxCPMBackend& backend);

    // === 核心推理接口 ===

    // Patch 模式: 处理单个时间步的 patch 特征
    // input: [feat_dim, n_patches] (已验证形状)
    // 返回: [hidden_size] (CLS token 输出)
    ggml_tensor* forward_patch(VoxCPMContext& ctx,
                                ggml_tensor* input,
                                MiniCPMKVCache& kv_cache);

    // === 访问器 ===
    const MiniCPMConfig& config() const { return encoder_.config(); }
    const LocEncWeights& weights() const { return weights_; }
    MiniCPMModel& encoder() { return encoder_; }
    const MiniCPMModel& encoder() const { return encoder_; }

private:
    LocEncWeights weights_;          // LocEnc 特有权重
    MiniCPMModel encoder_;           // 复用 MiniCPM (非因果模式)

    ggml_context* weight_ctx_ = nullptr;
    ggml_backend_buffer_t weight_buffer_ = nullptr;
};
```

**关键复用点**:
- `encoder_` 是 `MiniCPMModel` 实例
- `encoder_.forward(..., is_causal=false)` 实现非因果注意力
- `encoder_.load_from_gguf(path, "locenc", ...)` 加载 Transformer 权重

---

## 四、关键实现细节

### 4.1 输入投影 + CLS Token 拼接

```cpp
ggml_tensor* LocEncModel::forward_patch(VoxCPMContext& ctx,
                                         ggml_tensor* input,  // [hidden_size, n_patches]
                                         MiniCPMKVCache& kv_cache) {
    ggml_context* raw = ctx.raw_context();
    const int n_patches = input->ne[1];
    const int hidden_size = config_.hidden_size;

    // 1. 输入投影 (如果输入维度 != hidden_size)
    // input: [feat_dim, n_patches] -> [hidden_size, n_patches]
    ggml_tensor* projected = input;
    if (input->ne[0] != hidden_size) {
        projected = ggml_mul_mat(raw, weights_.in_proj_weight, input);
        projected = ggml_add(raw, projected, weights_.in_proj_bias);
    }

    // 2. Prepend special token (CLS)
    // special_token: [hidden_size] -> [hidden_size, 1]
    ggml_tensor* cls = ggml_reshape_2d(raw, weights_.special_token, hidden_size, 1);

    // 3. 拼接: [hidden_size, 1] + [hidden_size, n_patches] -> [hidden_size, n_patches+1]
    ggml_tensor* full_input = ggml_concat(raw, cls, projected, 1);

    // 4. 调用 MiniCPM (非因果注意力)
    ggml_tensor* output = encoder_.forward(ctx, full_input, nullptr, kv_cache, false);

    // 5. 提取 CLS token 输出 (位置 0)
    // output: [hidden_size, n_patches+1] -> [hidden_size]
    return ggml_view_1d(raw, output, hidden_size, 0);
}
```

### 4.2 批量时间步处理

```cpp
ggml_tensor* LocEncModel::forward(VoxCPMContext& ctx,
                                   ggml_tensor* input,  // [hidden_size, n_patches, n_time, batch]
                                   MiniCPMKVCache& kv_cache,
                                   int n_time,
                                   int batch) {
    ggml_context* raw = ctx.raw_context();

    // 方案 1: 简单实现 - 循环处理每个时间步
    // 注: GGML 图构建时不支持动态循环，需要在执行层面处理
    // 这需要多次图构建和执行

    // 方案 2: 批量展平 - 将 [B, T, P, D] 展平为 [(B*T), P, D]
    // 然后 prepend CLS，处理后再 reshape

    // 对于推理场景，通常 batch=1, T=1 或 T=生成步数
    // 建议先实现单时间步版本，后续优化批量处理
    return forward_patch(ctx, input, kv_cache);
}
```

### 4.3 权重加载

```cpp
bool LocEncModel::load_from_gguf(const std::string& gguf_path,
                                  VoxCPMContext& weight_ctx,
                                  VoxCPMContext& graph_ctx,
                                  VoxCPMBackend& backend) {
    // 1. 加载 GGUF 文件
    gguf_context* gguf_ctx = ...;  // 同 minicpm.cpp

    // 2. 从 GGUF metadata 更新配置
    uint32_t u32;
    if (get_u32_kv(gguf_ctx, "voxcpm_encoder_config_hidden_dim", u32))
        config_.hidden_size = static_cast<int>(u32);
    if (get_u32_kv(gguf_ctx, "voxcpm_encoder_config_ffn_dim", u32))
        config_.intermediate_size = static_cast<int>(u32);
    if (get_u32_kv(gguf_ctx, "voxcpm_encoder_config_num_heads", u32))
        config_.n_heads = static_cast<int>(u32);
    if (get_u32_kv(gguf_ctx, "voxcpm_encoder_config_num_layers", u32))
        config_.n_layer = static_cast<int>(u32);

    // 3. 加载 LocEnc 特有权重
    weights_.in_proj_weight = ggml_get_tensor(ggml_ctx, "locenc.in_proj.weight");
    weights_.in_proj_bias = ggml_get_tensor(ggml_ctx, "locenc.in_proj.bias");
    weights_.special_token = ggml_get_tensor(ggml_ctx, "locenc.special_token");

    // 4. 加载 MiniCPM encoder 权重 (前缀 "locenc.")
    // MiniCPMModel::update_config_from_gguf 已支持 "locenc." 前缀
    encoder_.load_from_gguf(gguf_path, "locenc", weight_ctx, graph_ctx, backend);

    // 5. 分配权重 buffer
    weight_buffer_ = ggml_backend_alloc_ctx_tensors(weight_ctx_, backend.raw_backend());
    ggml_backend_buffer_set_usage(weight_buffer_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // 6. 读取权重数据
    load_weight_data(file, gguf_ctx);

    return true;
}
```

### 4.4 批量时间步推理 (完整实现)

LocEnc 的典型输入是 `[B, T, P, D]` 形状，需要展平后处理：

```cpp
// 方案: 在 voxcpm.cpp 主模块中循环调用
// 因为 GGML 图构建时不支持动态循环

// 对于每个时间步 t:
for (int t = 0; t < n_time; ++t) {
    // 1. 提取当前时间步的 patch 特征
    // input: [feat_dim, n_patches, n_time, batch]
    ggml_tensor* patch_t = ggml_view_3d(ctx, input, feat_dim, n_patches, batch,
                                        input->nb[1], input->nb[2], t * input->nb[2]);

    // 2. 调用 LocEnc forward_patch
    ggml_tensor* output_t = locenc.forward_patch(ctx, patch_t, kv_cache);

    // 3. 存储到输出张量
    // output: [hidden_size, n_time, batch]
    ggml_tensor* out_slot = ggml_view_2d(ctx, output, hidden_size, batch,
                                          output->nb[1], t * output->nb[1]);
    ggml_build_forward_expand(graph, ggml_cpy(ctx, output_t, out_slot));
}
```

**注意**: 这种循环方式需要多次图执行。更高效的方式是将所有时间步的数据展平为一个大 batch，但这需要更大的 KV Cache。

---

## 五、GGUF 配置映射 (已验证)

从 `models/voxcpm1.5.dump` 提取的实际 metadata：

| GGUF Key | 值 | Config 字段 |
|----------|-----|------------|
| `voxcpm_encoder_config_hidden_dim` | 1024 | hidden_size |
| `voxcpm_encoder_config_ffn_dim` | 4096 | intermediate_size |
| `voxcpm_encoder_config_num_heads` | 16 | n_heads |
| `voxcpm_encoder_config_num_layers` | 8 | n_layer |

**注意**:
- `n_kv_heads` 需要从全局 MiniCPM 配置继承 (默认 2)
- `rms_norm_eps` 需要从全局配置继承 (默认 1e-5)
- `vocab_size` 在 encoder 中设为 0 (无 embedding)

---

## 六、实现步骤

### Phase 1: 头文件定义 (include/voxcpm/localenc.h)

```cpp
#ifndef VOXCPM_LOCALENC_H
#define VOXCPM_LOCALENC_H

#include "voxcpm/common.h"
#include "voxcpm/minicpm.h"  // 复用 MiniCPMModel 和 MiniCPMKVCache
#include <string>

namespace voxcpm {

class VoxCPMBackend;

struct LocEncWeights {
    ggml_tensor* in_proj_weight = nullptr;   // [hidden_size, feat_dim] = [1024, 64]
    ggml_tensor* in_proj_bias = nullptr;     // [hidden_size] = [1024]
    ggml_tensor* special_token = nullptr;    // [hidden_size] = [1024]
};

class LocEncModel {
public:
    LocEncModel() = default;
    ~LocEncModel();

    LocEncModel(const LocEncModel&) = delete;
    LocEncModel& operator=(const LocEncModel&) = delete;

    bool load_from_gguf(const std::string& gguf_path,
                        VoxCPMContext& weight_ctx,
                        VoxCPMContext& graph_ctx,
                        VoxCPMBackend& backend);

    ggml_tensor* forward_patch(VoxCPMContext& ctx,
                                ggml_tensor* input,
                                MiniCPMKVCache& kv_cache);

    const MiniCPMConfig& config() const { return encoder_.config(); }
    const LocEncWeights& weights() const { return weights_; }
    MiniCPMModel& encoder() { return encoder_; }

private:
    bool load_own_weights(FILE* file, gguf_context* gguf_ctx);

    LocEncWeights weights_;
    MiniCPMModel encoder_;  // 复用！

    ggml_context* weight_ctx_ = nullptr;
    ggml_backend_buffer_t weight_buffer_ = nullptr;
};

}  // namespace voxcpm

#endif  // VOXCPM_LOCALENC_H
```

### Phase 2: 核心实现 (src/localenc.cpp)

关键实现要点：

1. **构造/析构**: 管理 `weight_buffer_` 和 `weight_ctx_`

2. **load_from_gguf()**:
   - 加载 LocEnc 特有权重 (`in_proj`, `special_token`)
   - 调用 `encoder_.load_from_gguf(path, "locenc", ...)` 加载 Transformer
   - 合并权重到单一 Buffer

3. **forward_patch()**:
   ```cpp
   ggml_tensor* LocEncModel::forward_patch(VoxCPMContext& ctx,
                                            ggml_tensor* input,
                                            MiniCPMKVCache& kv_cache) {
       ggml_context* raw = ctx.raw_context();
       const int hidden_size = encoder_.config().hidden_size;

       // 1. 输入投影: [feat_dim, n_patches] -> [hidden_size, n_patches]
       ggml_tensor* projected = ggml_mul_mat(raw, weights_.in_proj_weight, input);
       projected = ggml_add(raw, projected, weights_.in_proj_bias);

       // 2. Prepend CLS token
       ggml_tensor* cls = ggml_reshape_2d(raw, weights_.special_token, hidden_size, 1);
       ggml_tensor* full_input = ggml_concat(raw, cls, projected, 1);

       // 3. 调用 MiniCPM (非因果!)
       ggml_tensor* output = encoder_.forward(ctx, full_input, nullptr, kv_cache, false);

       // 4. 提取 CLS 输出
       return ggml_view_1d(raw, output, hidden_size, 0);
   }
   ```

### Phase 3: 测试验证 (tests/test_localenc.cpp)

参考 `test_minicpm.cpp` 结构：
- 配置加载测试
- 权重加载测试
- 图构建测试
- Trace 验证测试

---

## 七、Trace 测试数据

### 7.1 Trace 文件位置

`vendor/VoxCPM/trace/trace_LocalEnc.jsonl`

### 7.2 输入输出格式

```json
{
    "call_id": 0,
    "module": "LocalEnc",
    "inputs": {
        "arg_0": [/* [1, 100, 4, 64] bfloat16 值 */]
    },
    "outputs": {
        "output": [/* [1, 100, 1024] bfloat16 值 */]
    }
}
```

### 7.3 张量形状解读

| 字段 | Python 形状 | GGML 形状 | 元素数 |
|------|------------|-----------|--------|
| input (arg_0) | [1, 100, 4, 64] | [64, 4, 100, 1] | 25,600 |
| output | [1, 100, 1024] | [1024, 100, 1] | 102,400 |

**解读**:
- batch = 1
- time = 100 (时间步数/帧数)
- patches = 4 (每帧的 patch 数)
- dim = 64 (特征维度)
- hidden_size = 1024 (输出隐藏维度)

### 7.4 数据布局转换

Python (row-major) 到 GGML (column-major):
```
Python: [B, T, P, D] = [1, 100, 4, 64]
展平顺序: B → T → P → D (行优先)

GGML: [D, P, T, B] = [64, 4, 100, 1]
展平顺序: D → P → T → B (列优先)

注意: 对于 batch=1 的情况，两者的展平数据可以直接对应！
```

---

## 八、测试用例设计

### 8.1 测试文件结构

参考 `tests/test_minicpm.cpp` 的模式：

```cpp
// tests/test_localenc.cpp

constexpr const char* kModelPath = "models/voxcpm1.5.gguf";
constexpr const char* kTracePath = "vendor/VoxCPM/trace/trace_LocalEnc.jsonl";

// 1. 配置加载测试
TEST_CASE("LocEnc config loads from GGUF", "[localenc][config]") {
    // 验证 hidden_dim=1024, ffn_dim=4096, num_heads=16, num_layers=8
}

// 2. 权重加载测试
TEST_CASE("LocEnc weights load from GGUF", "[localenc][weights]") {
    // 验证 in_proj.weight [64, 1024], special_token [1024]
    // 验证 8 层 Transformer 权重
}

// 3. 图构建测试
TEST_CASE("LocEnc forward graph builds", "[localenc][graph]") {
    // 输入: [64, 4, 1] (单时间步)
    // 输出: [1024]
}

// 4. Trace 验证测试 (与 PyTorch 对比)
TEST_CASE("LocEnc forward aligns with torch", "[localenc][trace]") {
    // 输入: [1, 100, 4, 64] -> 展平处理
    // 输出: [1, 100, 1024]
    // tolerance: 0.05, max_mismatch_rate: 0.05
}
```

### 8.2 Trace 数据加载

```cpp
// 从 trace_LocalEnc.jsonl 加载测试数据
struct LocalEncTrace {
    std::vector<float> input;   // [1, 100, 4, 64] = 25,600 元素
    std::vector<float> output;  // [1, 100, 1024] = 102,400 元素
};

LocalEncTrace load_trace(const std::string& path) {
    // 解析 JSONL，提取 input 和 output 数组
    // 注意: bfloat16 在 JSON 中表示为 float
}
```

### 8.3 多时间步测试策略

```cpp
// 测试循环处理多个时间步
TEST_CASE("LocEnc multi-timestep forward", "[localenc][integration]") {
    VoxCPMBackend backend(BackendType::CPU, 2);
    LocEncModel locenc;
    locenc.load_from_gguf(kModelPath, ...);

    const int n_time = 100;
    const int n_patches = 4;
    const int feat_dim = 64;

    // 为每个时间步创建独立的 KV Cache
    MiniCPMKVCache kv_cache(locenc.config().n_layer, ...);
    kv_cache.init(backend);

    std::vector<float> all_outputs(n_time * hidden_size);

    for (int t = 0; t < n_time; ++t) {
        // 构建当前时间步的图
        VoxCPMContext graph_ctx(ContextType::Graph, 4096, 32768);

        ggml_tensor* input_t = graph_ctx.new_tensor_2d(GGML_TYPE_F32, feat_dim, n_patches);
        ggml_set_input(input_t);

        ggml_tensor* output_t = locenc.forward_patch(graph_ctx, input_t, kv_cache);
        ggml_set_output(output_t);

        // 执行图
        ggml_cgraph* graph = graph_ctx.new_graph();
        graph_ctx.build_forward(graph, output_t);
        backend.alloc_graph(graph);

        // 设置输入数据并执行
        backend.tensor_set(input_t, trace.input.data() + t * n_patches * feat_dim, ...);
        backend.compute(graph);

        // 收集输出
        backend.tensor_get(output_t, all_outputs.data() + t * hidden_size, ...);

        // 清理 KV Cache 为下一时间步准备
        kv_cache.clear();
    }

    // 与 trace 对比
    validate_with_tolerance(all_outputs, trace.output, ...);
}
```

---

## 九、GGML 最佳实践检查清单

- [ ] `no_alloc=true` 所有 Context
- [ ] `ggml_set_input()` 所有输入张量
- [ ] `ggml_set_output()` 所有输出张量
- [ ] 权重 Buffer 独立 + WEIGHTS 用途标记
- [ ] KV Cache Buffer 独立（复用 MiniCPMKVCache）
- [ ] 复用 MiniCPMModel 的 forward 方法

---

## 十、关键注意事项

### 1. 非因果注意力

LocEnc 使用 `is_causal=false`，这意味着：
- 不需要 causal mask
- 每个 token 可以看到所有其他 token
- `MiniCPMModel::forward()` 已支持此参数

### 2. 特殊 Token 处理

- CLS token 通过 `ggml_concat` prepend 到输入序列
- 最终输出只取位置 0 (CLS token 的输出)

### 3. 张量形状约定

```
Python: [B, T, P, D] (batch, time, patches, dim)
GGML:   [D, P, T, B] (dim, patches, time, batch)
```

### 4. 复用策略

LocEnc 不需要重新实现 Transformer 层，直接复用 `MiniCPMModel`：
- 权重加载时使用 "locenc." 前缀
- forward 时传入 `is_causal=false`
- KV Cache 使用相同结构

---

## 十一、调试技巧

```cpp
// 打印张量信息
void print_tensor(const char* name, ggml_tensor* t) {
    printf("%s: shape=[%lld, %lld, %lld, %lld]\n",
           name, t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
}

// 验证 CLS token 提取
// 在 forward_patch 返回前:
ggml_tensor* cls_out = ggml_view_1d(raw, output, hidden_size, 0);
print_tensor("CLS output", cls_out);
```
