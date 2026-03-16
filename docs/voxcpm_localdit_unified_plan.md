# LocalDIT 和 UnifiedCFM 模块开发计划

## Context

实现 LocDiT (Local Diffusion Transformer) 和 UnifiedCFM (Conditional Flow Matching) 两个模块的 C++ GGML 版本。这两个模块是 VoxCPM 生成流程的核心组件：

- **LocDiT**: 基于 DiT 的扩散模型骨干网络，用于预测 CFM 速度场
- **UnifiedCFM**: CFM 求解器，使用 Euler 积分从噪声生成音频特征

**依赖关系**: UnifiedCFM 依赖 LocDiT 作为速度估计器 (estimator)

**参考实现**:
- Python: `vendor/VoxCPM/src/voxcpm/modules/locdit/local_dit.py`, `unified_cfm.py`
- GGML Python: `examples/locdit_ggml.py`, `examples/unified_cfm_ggml.py`
- 复用模块: `src/minicpm.cpp` (MiniCPMModel 作为 Transformer backbone)

---

## 一、架构分析

### 1.1 LocDiT 结构

```
输入: x [C, T], mu [hidden_size], t (timestep), cond [C, T'], dt (delta time)
    │
    ├─► in_proj(x) ──────────────────────────────────┐
    │                                                   │
    ├─► cond_proj(cond) ──────────────────────────────►│
    │                                                   │
    ├─► time_embeddings(t) ─► time_mlp ───────────────┤
    │                                                   │
    └─► time_embeddings(dt) ─► delta_time_mlp ────────┤
                                                        │
    combined = (mu + t_emb + dt_emb) ──────────────────►│
                                                        ▼
    x = concat([combined, cond, projected_x], dim=seq) ─► MiniCPMModel (非因果)
                                                        │
    output = out_proj(hidden[:, prefix+1:]) ────────────►
    │
输出: [C, T]
```

### 1.2 UnifiedCFM 结构

```
输入: z (噪声), mu (条件), n_timesteps, patch_size, cond (前缀条件)
    │
    ▼
t_span = linspace(1, 0, n_timesteps+1) + sway_sampling
    │
    ▼ (Euler 循环)
┌─────────────────────────────────────────────────────────┐
│ for step in range(1, n_timesteps+1):                    │
│     if step <= zero_init_steps:                         │
│         dphi_dt = 0                                     │
│     else:                                               │
│         dphi_dt_cond = estimator(x, mu, t, cond, dt)    │
│         dphi_dt_uncond = estimator(x, 0, t, cond, dt)   │
│         st_star = <dphi_dt_cond, dphi_dt_uncond> / ||dphi_dt_uncond||²  │
│         dphi_dt = CFG(dphi_dt_cond, dphi_dt_uncond, st_star, cfg_value)│
│     x = x - dt * dphi_dt                                │
│     t = t - dt                                          │
└─────────────────────────────────────────────────────────┘
    │
输出: x (生成的特征)
```

### 1.3 PyTorch → GGML 算子映射

#### LocDiT 特有操作

| PyTorch 操作 | GGML 算子 | 说明 |
|-------------|----------|------|
| `nn.Linear` | `ggml_mul_mat` + `ggml_add` | 线性投影 |
| `SinusoidalPosEmb` | `ggml_timestep_embedding` | 内置算子！或用 `ggml_sin/cos/exp` 组合 |
| `nn.SiLU` | `ggml_silu` | 内置算子 |
| `torch.cat` | `ggml_concat` | 沿维度拼接 |
| `x.transpose(1, 2)` | `ggml_permute` | 维度置换 |

#### UnifiedCFM 操作

| PyTorch 操作 | GGML 算子 | 说明 |
|-------------|----------|------|
| `x * dt` | `ggml_scale` | 标量缩放 |
| `x - y` | `ggml_sub` | 元素减法 |
| `x * y` | `ggml_mul` | 元素乘法 |
| `x.sum()` | `ggml_sum` | 归约求和 |
| `x.clamp(min, max)` | `ggml_clamp` | 数值限制 |

---

## 二、文件结构

```
VoxCPM.cpp/
├── include/voxcpm/
│   ├── locdit.h           # 新建 - LocDiT 类定义
│   └── unified_cfm.h      # 新建 - UnifiedCFM 类定义
├── src/
│   ├── locdit.cpp         # 新建 - LocDiT 实现
│   └── unified_cfm.cpp    # 新建 - UnifiedCFM 实现
└── tests/
    ├── test_locdit.cpp    # 新建 - LocDiT 测试
    └── test_unified_cfm.cpp # 新建 - UnifiedCFM 测试
```

---

## 三、类设计

### 3.1 LocDiTWeights

```cpp
struct LocDiTWeights {
    // 投影层
    ggml_tensor* in_proj_weight = nullptr;    // [hidden_size, feat_dim]
    ggml_tensor* in_proj_bias = nullptr;      // [hidden_size]
    ggml_tensor* cond_proj_weight = nullptr;  // [hidden_size, feat_dim]
    ggml_tensor* cond_proj_bias = nullptr;    // [hidden_size]
    ggml_tensor* out_proj_weight = nullptr;   // [feat_dim, hidden_size]
    ggml_tensor* out_proj_bias = nullptr;     // [feat_dim]

    // 时间嵌入 MLP
    ggml_tensor* time_mlp_linear1_weight = nullptr;  // [hidden_size, hidden_size]
    ggml_tensor* time_mlp_linear1_bias = nullptr;    // [hidden_size]
    ggml_tensor* time_mlp_linear2_weight = nullptr;  // [hidden_size, hidden_size]
    ggml_tensor* time_mlp_linear2_bias = nullptr;    // [hidden_size]

    // Delta 时间嵌入 MLP
    ggml_tensor* delta_time_mlp_linear1_weight = nullptr;
    ggml_tensor* delta_time_mlp_linear1_bias = nullptr;
    ggml_tensor* delta_time_mlp_linear2_weight = nullptr;
    ggml_tensor* delta_time_mlp_linear2_bias = nullptr;
};
```

### 3.2 LocDiTModel

```cpp
class LocDiTModel {
public:
    LocDiTModel() = default;
    ~LocDiTModel();

    LocDiTModel(const LocDiTModel&) = delete;
    LocDiTModel& operator=(const LocDiTModel&) = delete;

    // 权重加载
    bool load_from_gguf(const std::string& gguf_path,
                        VoxCPMContext& weight_ctx,
                        VoxCPMContext& graph_ctx,
                        VoxCPMBackend& backend);

    // 核心推理接口
    // x: [feat_dim, T] 输入特征
    // mu: [hidden_size] 条件向量
    // t_emb: [hidden_size] 时间嵌入 (预计算)
    // cond: [feat_dim, T'] 前缀条件
    // 返回: [feat_dim, T] 输出特征
    ggml_tensor* forward(VoxCPMContext& ctx,
                         ggml_tensor* x,
                         ggml_tensor* mu,
                         ggml_tensor* t_emb,
                         ggml_tensor* cond);

    // 时间嵌入计算 (供 UnifiedCFM 调用)
    ggml_tensor* compute_time_embedding(VoxCPMContext& ctx, float t);
    ggml_tensor* compute_delta_time_embedding(VoxCPMContext& ctx, float dt);

    // 访问器
    const MiniCPMConfig& config() const { return decoder_.config(); }
    int hidden_size() const { return config().hidden_size; }
    int feat_dim() const { return feat_dim_; }

private:
    // Sinusoidal 位置嵌入
    ggml_tensor* sinusoidal_embedding(VoxCPMContext& ctx, float t, int dim, float scale = 1000.0f);

    // TimestepEmbedding MLP
    ggml_tensor* timestep_mlp(VoxCPMContext& ctx,
                               ggml_tensor* t_emb,
                               ggml_tensor* linear1_w, ggml_tensor* linear1_b,
                               ggml_tensor* linear2_w, ggml_tensor* linear2_b);

    bool init_scratch_cache(VoxCPMBackend& backend);
    bool load_own_weights(FILE* file, gguf_context* gguf_ctx);

    LocDiTWeights weights_;
    MiniCPMModel decoder_;  // 复用 MiniCPM (非因果模式)

    int feat_dim_ = 64;
    int hidden_size_ = 1024;

    ggml_context* weight_ctx_ = nullptr;
    ggml_backend_buffer_t weight_buffer_ = nullptr;
    VoxCPMBackend* backend_ = nullptr;
    std::unique_ptr<MiniCPMKVCache> scratch_kv_cache_;
};
```

### 3.3 CFMConfig

```cpp
struct CFMConfig {
    float sigma_min = 1e-6f;
    float inference_cfg_rate = 2.0f;  // CFG 强度
    int n_timesteps = 10;             // Euler 积分步数
    float sway_sampling_coef = 1.0f;  // Sway 采样系数
    bool use_cfg_zero_star = true;    // CFG-Zero* 优化
};
```

### 3.4 UnifiedCFM

```cpp
class UnifiedCFM {
public:
    UnifiedCFM(LocDiTModel& estimator, const CFMConfig& config);
    ~UnifiedCFM() = default;

    // CFM 推理接口
    // z: [feat_dim, T] 初始噪声
    // mu: [hidden_size] 条件向量
    // cond: [feat_dim, T'] 前缀条件
    // 返回: [feat_dim, T] 生成的特征
    ggml_tensor* forward(VoxCPMContext& ctx,
                         ggml_tensor* z,
                         ggml_tensor* mu,
                         ggml_tensor* cond,
                         int patch_size,
                         float temperature = 1.0f);

    // 设置参数
    void set_cfg_value(float cfg) { config_.inference_cfg_rate = cfg; }
    void set_n_timesteps(int n) { config_.n_timesteps = n; }

private:
    // Euler 积分求解
    ggml_tensor* solve_euler(VoxCPMContext& ctx,
                              ggml_tensor* x,
                              const std::vector<float>& t_span,
                              ggml_tensor* mu,
                              ggml_tensor* cond);

    // CFG-Zero* 缩放因子
    ggml_tensor* optimized_scale(VoxCPMContext& ctx,
                                  ggml_tensor* positive,
                                  ggml_tensor* negative,
                                  float eps = 1e-4f);

    // 计算带 CFG 的速度
    ggml_tensor* compute_velocity_with_cfg(VoxCPMContext& ctx,
                                            ggml_tensor* x,
                                            ggml_tensor* mu,
                                            ggml_tensor* cond,
                                            float t,
                                            float dt);

    // 计算时间跨度 (含 sway sampling)
    std::vector<float> compute_t_span(int n_timesteps, float sway_coef);

    LocDiTModel& estimator_;
    CFMConfig config_;
    int feat_dim_;
};
```

---

## 四、GGUF 权重命名 (已验证)

从 `models/voxcpm1.5.dump` 提取的 LocDiT 相关张量：

| GGUF Key | 形状 | 说明 |
|----------|------|------|
| `locdit.in_proj.weight` | [64, 1024] | 输入投影 |
| `locdit.in_proj.bias` | [1024] | |
| `locdit.cond_proj.weight` | [64, 1024] | 条件投影 |
| `locdit.cond_proj.bias` | [1024] | |
| `locdit.out_proj.weight` | [1024, 64] | 输出投影 |
| `locdit.out_proj.bias` | [64] | |
| `locdit.time_mlp.linear_1.weight` | [1024, 1024] | 时间 MLP 层1 |
| `locdit.time_mlp.linear_1.bias` | [1024] | |
| `locdit.time_mlp.linear_2.weight` | [1024, 1024] | 时间 MLP 层2 |
| `locdit.time_mlp.linear_2.bias` | [1024] | |
| `locdit.delta_time_mlp.linear_1.weight` | [1024, 1024] | Delta时间 MLP 层1 |
| `locdit.delta_time_mlp.linear_1.bias` | [1024] | |
| `locdit.delta_time_mlp.linear_2.weight` | [1024, 1024] | Delta时间 MLP 层2 |
| `locdit.delta_time_mlp.linear_2.bias` | [1024] | |
| `locdit.blk.{i}.attn_norm.weight` | [1024] | Transformer 层 (共8层) |
| `locdit.blk.{i}.attn_q.weight` | [1024, 1024] | |
| `locdit.blk.{i}.attn_k.weight` | [1024, 128] | |
| `locdit.blk.{i}.attn_v.weight` | [1024, 128] | |
| `locdit.blk.{i}.attn_output.weight` | [1024, 1024] | |
| `locdit.blk.{i}.ffn_norm.weight` | [1024] | |
| `locdit.blk.{i}.ffn_gate.weight` | [1024, 4096] | |
| `locdit.blk.{i}.ffn_up.weight` | [1024, 4096] | |
| `locdit.blk.{i}.ffn_down.weight` | [4096, 1024] | |
| `locdit.output_norm.weight` | [1024] | |

**注意**: Transformer 层权重通过 `MiniCPMModel::load_from_gguf(path, "locdit", ...)` 加载。

---

## 五、关键实现细节

### 5.1 SinusoidalPosEmb 实现

**方案 A: 使用内置 `ggml_timestep_embedding` (推荐)**

```cpp
ggml_tensor* LocDiTModel::sinusoidal_embedding(VoxCPMContext& ctx, float t, int dim, float scale) {
    ggml_context* raw = ctx.raw_context();

    // 创建时间步张量
    ggml_tensor* t_tensor = ggml_new_tensor_1d(raw, GGML_TYPE_F32, 1);
    ggml_set_input(t_tensor);

    // 使用内置算子
    ggml_tensor* emb = ggml_timestep_embedding(raw, t_tensor, dim, 10000);

    // 设置输入数据
    float t_scaled = t * scale;
    ggml_backend_tensor_set(t_tensor, &t_scaled, 0, sizeof(float));

    return emb;
}
```

**方案 B: 手动组合 (灵活性更高)**

```cpp
ggml_tensor* LocDiTModel::sinusoidal_embedding(VoxCPMContext& ctx, float t, int dim, float scale) {
    ggml_context* raw = ctx.raw_context();
    const int half_dim = dim / 2;

    // emb = exp(arange(0, half_dim) * -log(10000) / (half_dim - 1))
    // 计算: log(10000) / (half_dim - 1)
    float emb_val = std::log(10000.0f) / static_cast<float>(half_dim - 1);

    // arange(0, half_dim)
    ggml_tensor* arange = ggml_arange(raw, 0.0f, static_cast<float>(half_dim), 1.0f);

    // emb = arange * (-emb_val)
    ggml_tensor* emb = ggml_scale(raw, arange, -emb_val);

    // emb = exp(emb)
    emb = ggml_exp(raw, emb);

    // emb = emb * (scale * t)
    emb = ggml_scale(raw, emb, scale * t);

    // sin_emb = sin(emb), cos_emb = cos(emb)
    ggml_tensor* sin_emb = ggml_sin(raw, emb);
    ggml_tensor* cos_emb = ggml_cos(raw, emb);

    // concat(sin, cos)
    return ggml_concat(raw, sin_emb, cos_emb, 0);
}
```

### 5.2 TimestepEmbedding MLP

```cpp
ggml_tensor* LocDiTModel::timestep_mlp(VoxCPMContext& ctx,
                                        ggml_tensor* t_emb,
                                        ggml_tensor* linear1_w,
                                        ggml_tensor* linear1_b,
                                        ggml_tensor* linear2_w,
                                        ggml_tensor* linear2_b) {
    ggml_context* raw = ctx.raw_context();

    // Linear 1: [hidden_size] @ [hidden_size, hidden_size]^T -> [hidden_size]
    ggml_tensor* x = ggml_mul_mat(raw, linear1_w, t_emb);
    x = ggml_add(raw, x, linear1_b);

    // SiLU activation
    x = ggml_silu(raw, x);

    // Linear 2
    x = ggml_mul_mat(raw, linear2_w, x);
    x = ggml_add(raw, x, linear2_b);

    return x;
}
```

### 5.3 LocDiT Forward

```cpp
ggml_tensor* LocDiTModel::forward(VoxCPMContext& ctx,
                                   ggml_tensor* x,       // [feat_dim, T]
                                   ggml_tensor* mu,      // [hidden_size]
                                   ggml_tensor* t_emb,   // [hidden_size] (已计算)
                                   ggml_tensor* cond) {  // [feat_dim, T']
    ggml_context* raw = ctx.raw_context();
    const int hidden_size = config().hidden_size;

    // 1. 输入投影: [feat_dim, T] -> [hidden_size, T]
    ggml_tensor* x_proj = ggml_mul_mat(raw, weights_.in_proj_weight, x);
    x_proj = ggml_add(raw, x_proj, weights_.in_proj_bias);

    // 2. 条件投影: [feat_dim, T'] -> [hidden_size, T']
    ggml_tensor* cond_proj = ggml_mul_mat(raw, weights_.cond_proj_weight, cond);
    cond_proj = ggml_add(raw, cond_proj, weights_.cond_proj_bias);
    const int prefix_len = cond->ne[1];

    // 3. 组合条件: mu + t_emb (broadcast)
    // mu: [hidden_size], t_emb: [hidden_size]
    ggml_tensor* combined = ggml_add(raw, mu, t_emb);
    combined = ggml_reshape_2d(raw, combined, hidden_size, 1);

    // 4. 拼接序列: [combined(1), cond(T'), x(T)]
    // 先拼接 combined 和 cond
    ggml_tensor* full_seq = ggml_concat(raw, combined, cond_proj, 1);
    // 再拼接 x
    full_seq = ggml_concat(raw, full_seq, x_proj, 1);

    // 5. MiniCPM forward (非因果)
    scratch_kv_cache_->clear();
    ggml_tensor* hidden = decoder_.forward(ctx, full_seq, nullptr, *scratch_kv_cache_, false);

    // 6. 提取输出部分 (跳过 prefix + 1 个 token)
    // hidden: [hidden_size, 1 + T' + T]
    const int skip_len = prefix_len + 1;
    ggml_tensor* output_hidden = ggml_view_2d(raw, hidden, hidden_size, x->ne[1],
                                              hidden->nb[1], skip_len * hidden->nb[1]);

    // 7. 输出投影: [hidden_size, T] -> [feat_dim, T]
    ggml_tensor* output = ggml_mul_mat(raw, weights_.out_proj_weight, output_hidden);
    output = ggml_add(raw, output, weights_.out_proj_bias);

    return output;
}
```

### 5.4 CFM Euler Solver

```cpp
ggml_tensor* UnifiedCFM::solve_euler(VoxCPMContext& ctx,
                                      ggml_tensor* x,
                                      const std::vector<float>& t_span,
                                      ggml_tensor* mu,
                                      ggml_tensor* cond) {
    ggml_context* raw = ctx.raw_context();

    float t = t_span[0];
    float dt = t_span[0] - t_span[1];

    const int n_steps = t_span.size() - 1;
    const int zero_init_steps = std::max(1, static_cast<int>(n_steps * 0.04));

    // 创建零张量 (用于 CFG-Zero*)
    std::vector<float> zero_data(x->ne[0] * x->ne[1], 0.0f);

    for (int step = 1; step <= n_steps; ++step) {
        ggml_tensor* dphi_dt = nullptr;

        if (config_.use_cfg_zero_star && step <= zero_init_steps) {
            // CFG-Zero*: 初始步使用零速度
            dphi_dt = ggml_dup_tensor(raw, x);
            ggml_set_input(dphi_dt);
            ggml_backend_tensor_set(dphi_dt, zero_data.data(), 0, zero_data.size() * sizeof(float));
        } else {
            // 计算 CFG 速度
            dphi_dt = compute_velocity_with_cfg(ctx, x, mu, cond, t, dt);
        }

        // Euler 步: x = x - dt * dphi_dt
        ggml_tensor* scaled = ggml_scale(raw, dphi_dt, dt);
        x = ggml_sub(raw, x, scaled);

        t = t - dt;
        if (step < n_steps) {
            dt = t - t_span[step + 1];
        }
    }

    return x;
}
```

### 5.5 CFG-Zero* 缩放因子

```cpp
ggml_tensor* UnifiedCFM::optimized_scale(VoxCPMContext& ctx,
                                          ggml_tensor* positive,
                                          ggml_tensor* negative,
                                          float eps) {
    ggml_context* raw = ctx.raw_context();

    // dot_product = sum(positive * negative)
    ggml_tensor* product = ggml_mul(raw, positive, negative);
    ggml_tensor* dot_product = ggml_sum(raw, product);

    // squared_norm = sum(negative * negative)
    ggml_tensor* neg_squared = ggml_mul(raw, negative, negative);
    ggml_tensor* squared_norm = ggml_sum(raw, neg_squared);

    // st_star = dot_product / (squared_norm + eps)
    ggml_tensor* eps_tensor = ggml_new_tensor_1d(raw, GGML_TYPE_F32, 1);
    ggml_set_input(eps_tensor);
    float eps_val = eps;
    ggml_backend_tensor_set(eps_tensor, &eps_val, 0, sizeof(float));

    ggml_tensor* denom = ggml_add(raw, squared_norm, eps_tensor);
    ggml_tensor* st_star = ggml_div(raw, dot_product, denom);

    // Clamp for stability
    st_star = ggml_clamp(raw, st_star, -5.0f, 5.0f);

    return st_star;
}
```

---

## 六、实现步骤

### Phase 1: LocDiT 头文件 (include/voxcpm/locdit.h)

1. 定义 `LocDiTWeights` 结构体
2. 定义 `LocDiTModel` 类
3. 声明所有公共接口

### Phase 2: LocDiT 核心实现 (src/locdit.cpp)

1. `sinusoidal_embedding()` - 时间嵌入
2. `timestep_mlp()` - MLP 实现
3. `load_from_gguf()` - 权重加载
4. `forward()` - 核心前向传播

### Phase 3: UnifiedCFM 头文件 (include/voxcpm/unified_cfm.h)

1. 定义 `CFMConfig` 结构体
2. 定义 `UnifiedCFM` 类

### Phase 4: UnifiedCFM 实现 (src/unified_cfm.cpp)

1. `compute_t_span()` - 时间跨度计算
2. `optimized_scale()` - CFG-Zero* 缩放
3. `compute_velocity_with_cfg()` - CFG 速度计算
4. `solve_euler()` - Euler 求解器
5. `forward()` - CFM 推理入口

### Phase 5: 测试 (tests/test_locdit.cpp, tests/test_unified_cfm.cpp)

1. 权重加载测试
2. 时间嵌入测试
3. LocDiT forward 测试 (对比 trace)
4. CFM Euler 积分测试
5. 完整生成流程测试

---

## 七、Trace 测试数据

### 7.1 LocDiT Trace

**文件**: `tests/fixtures/trace/trace_LocalDit.jsonl`

**输入格式**:
```json
{
    "call_id": 0,
    "module": "LocalDit",
    "inputs": {
        "arg_0": [/* x: [1, T, 64] */],
        "arg_1": [/* mu: [1, 1024] */],
        "arg_2": [/* t: scalar */],
        "arg_3": [/* cond: [1, T', 64] */],
        "arg_4": [/* dt: scalar */]
    },
    "outputs": {
        "output": [/* [1, T, 64] */]
    }
}
```

### 7.2 UnifiedCFM Trace

**文件**: `tests/fixtures/trace/trace_UnifiedCFM.jsonl`

**输入格式**:
```json
{
    "call_id": 0,
    "module": "UnifiedCFM",
    "inputs": {
        "arg_0": [/* mu: [1, 1024] */],
        "arg_1": [/* n_timesteps: int */],
        "arg_2": [/* patch_size: int */],
        "arg_3": [/* cond: [1, T', 64] */]
    },
    "outputs": {
        "output": [/* [1, T, 64] */]
    }
}
```

---

## 八、GGML 最佳实践检查清单

- [ ] `no_alloc=true` 所有 Context
- [ ] `ggml_set_input()` 所有输入张量
- [ ] `ggml_set_output()` 所有输出张量
- [ ] 权重 Buffer 独立 + WEIGHTS 用途标记
- [ ] KV Cache Buffer 独立（复用 MiniCPMKVCache）
- [ ] 复用 MiniCPMModel 的 forward 方法
- [ ] 每次推理前调用 `scratch_kv_cache_->clear()`

---

## 九、关键注意事项

### 1. 张量形状约定

```
Python (row-major): [B, T, C] (batch, time, channels)
GGML (column-major): [C, T, B] (channels, time, batch)

LocDiT 输入/输出:
- x: [feat_dim, T] = [64, T]
- mu: [hidden_size] = [1024]
- cond: [feat_dim, T'] = [64, T']
```

### 2. 时间嵌入的 Scale

Python 实现使用 `scale=1000`:
```python
emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
```

确保 C++ 实现使用相同的 scale 值。

### 3. CFG-Zero* 的数值稳定性

当 `||negative||` 很小时，`st_star` 可能变得非常大，导致数值不稳定。必须 clamp 到合理范围（如 [-5, 5]）。

### 4. 多图执行模式

CFM 的 Euler 积分需要多次图执行：

```cpp
for (int step = 1; step <= n_steps; ++step) {
    // 1. 构建图
    VoxCPMContext graph_ctx(ContextType::Graph, ...);

    // 2. 计算速度
    ggml_tensor* dphi_dt = ...;

    // 3. 执行图
    backend.compute(graph);

    // 4. 读取中间结果用于下一步
    // ...
}
```

### 5. 复用 MiniCPMModel

LocDiT 的 Transformer 部分直接复用 `MiniCPMModel`:
- 权重加载时使用 `"locdit."` 前缀
- forward 时传入 `is_causal=false`

---

## 十、验证方法

```bash
cd /home/orangepi/Codes/ggbond/VoxCPM.cpp
cmake -B build && cmake --build build
cd build && ctest -R "test_locdit|test_unified_cfm" --output-on-failure
```

---

## 十一、预期测试结果

| 测试 | 预期 |
|------|------|
| LocDiT 权重加载 | 所有权重正确加载，形状匹配 |
| 时间嵌入 | 与 Python 实现误差 < 1e-5 |
| LocDiT forward | 与 trace 对比，tolerance=0.05, max_mismatch_rate=0.05 |
| CFM Euler | 生成的特征形状正确，数值范围合理 |
| 完整生成 | 与 Python 参考实现输出对齐 |
