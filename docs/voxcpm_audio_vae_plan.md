# AudioVAE 模块开发计划

## Context

实现 AudioVAE (Audio Variational Autoencoder) 的 GGML C++ 版本，用于音频编解码：

- **Encoder**: 音频波形 → 潜在表示 (64-dim latent)
- **Decoder**: 潜在表示 → 音频波形

**目标**: 通过 torch 版本导出的 trace 数据测试 GGML 实现的数值精度。

**测试数据**:
- `vendor/VoxCPM/trace/trace_AudioVAE_encode.jsonl`: encode trace
- `vendor/VoxCPM/trace/trace_AudioVAE_decode.jsonl`: decode trace

**参考实现**:
- Python: `vendor/VoxCPM/src/voxcpm/modules/audiovae/audio_vae.py`
- GGML 模式: `VoxCPM.cpp/src/fsq.cpp`, `VoxCPM.cpp/src/minicpm.cpp`

---

## 一、架构分析

### 1.1 AudioVAE 整体结构

```
Encoder:
音频 [B, 1, T]
    │
    ▼ WNCausalConv1d(1→64, k=7) [初始化卷积]
    │
    ▼ EncBlock(stride=2): 64→128, T→T/2
    ▼ EncBlock(stride=3): 128→256, T/2→T/6
    ▼ EncBlock(stride=6): 256→512, T/6→T/36
    ▼ EncBlock(stride=7): 512→1024, T/36→T/252
    ▼ EncBlock(stride=7): 1024→2048, T/252→T/1764
    │
    ├─► fc_mu: Conv1d(2048→64, k=3) → mu [B, 64, T/1764]
    └─► fc_logvar: Conv1d(2048→64, k=3) → logvar [B, 64, T/1764]

Decoder:
潜在 z [B, 64, T/1764]
    │
    ▼ DWConv(64→64, k=7, groups=64) + PWConv(64→2048, k=1)
    │
    ▼ DecBlock(stride=7): 2048→1024, T→T*7
    ▼ DecBlock(stride=7): 1024→512, T*7→T*49
    ▼ DecBlock(stride=6): 512→256, T*49→T*294
    ▼ DecBlock(stride=3): 256→128, T*294→T*882
    ▼ DecBlock(stride=2): 128→64, T*882→T*1764
    │
    ▼ Snake1d(64) + Conv1d(64→1, k=7) + Tanh
    │
音频 [B, 1, T]
```

### 1.2 核心组件

| 组件 | 功能 | GGML 实现挑战 |
|------|------|---------------|
| **Snake1d** | 可学习激活函数 `x + sin(αx)²/α` | 需要组合 sin, mul, add, div |
| **CausalConv1d** | 左侧填充卷积 (因果) | 需显式 pad + conv(padding=0) |
| **CausalTransposeConv1d** | 输出裁剪转置卷积 | conv_transpose + slice |
| **CausalResidualUnit** | 带膨胀卷积的残差块 | dilated conv 支持 |
| **CausalEncoderBlock** | 下采样编码块 | 组合多个残差单元 |
| **CausalDecoderBlock** | 上采样解码块 | 转置卷积 + 残差单元 |

### 1.3 PyTorch → GGML 算子映射

| PyTorch 操作 | GGML 算子 | 说明 |
|-------------|----------|------|
| `F.pad(x, (pad*2, 0))` | `ggml_pad` + `ggml_view` | 左侧填充 |
| `nn.Conv1d` | `ggml_conv_1d` | 一维卷积 |
| `nn.ConvTranspose1d` | `ggml_conv_transpose_1d` | 转置卷积 |
| `weight_norm` | 预折叠 | 转换时折叠为单个权重 |
| `sin(x)` | `ggml_sin` | 正弦函数 |
| `x.pow(2)` | `ggml_sqr` | 平方 |
| `x.reciprocal()` | `ggml_reciprocal` | 倒数 |
| `nn.Tanh()` | `ggml_tanh` | Tanh 激活 |

### 1.4 Trace 数据规格

**encode trace**:
- 输入: `[1, 381024]` 音频 (约 8.64s @ 44100Hz)
- 输出: `[1, 64, 216]` 潜在表示
- 下采样率: 381024 / 216 = 1764 (hop_length)

**decode trace**:
- 输入: `[1, 64, 36]` 潜在表示
- 输出: `[1, 1, 63504]` 音频
- 上采样率: 63504 / 36 = 1764 (hop_length)

---

## 二、文件结构

```
VoxCPM.cpp/
├── include/voxcpm/
│   └── audio-vae.h         # 新建 - AudioVAE 类定义
├── src/
│   └── audio-vae.cpp       # 新建 - AudioVAE 实现
└── tests/
    └── test_audio_vae.cpp  # 新建 - AudioVAE 测试
```

---

## 三、类设计

### 3.1 Snake 层权重

```cpp
struct SnakeWeights {
    ggml_tensor* alpha = nullptr;  // [channels, 1, 1] 可学习参数
};
```

### 3.2 CausalResidualUnit 权重

```cpp
struct ResidualUnitWeights {
    // Snake + CausalConv1d (dilated)
    ggml_tensor* conv1_weight = nullptr;  // [kernel, 1, dim, dim] 或展开形状
    ggml_tensor* conv1_bias = nullptr;    // [dim]
    ggml_tensor* snake1_alpha = nullptr;  // [dim]

    // Snake + CausalConv1d (kernel=1)
    ggml_tensor* conv2_weight = nullptr;  // [1, 1, dim, dim]
    ggml_tensor* conv2_bias = nullptr;    // [dim]
    ggml_tensor* snake2_alpha = nullptr;  // [dim]
};
```

### 3.3 CausalEncoderBlock 权重

```cpp
struct EncoderBlockWeights {
    // 3 个 CausalResidualUnit (dilation=1,3,9)
    std::array<ResidualUnitWeights, 3> residual_units;

    // Snake + 下采样卷积
    ggml_tensor* snake_alpha = nullptr;      // [input_dim]
    ggml_tensor* downconv_weight = nullptr;  // [kernel, 1, input_dim, output_dim]
    ggml_tensor* downconv_bias = nullptr;    // [output_dim]
};
```

### 3.4 CausalDecoderBlock 权重

```cpp
struct DecoderBlockWeights {
    // Snake + 上采样转置卷积
    ggml_tensor* snake_alpha = nullptr;        // [input_dim]
    ggml_tensor* upconv_weight = nullptr;      // [kernel, 1, output_dim, input_dim]
    ggml_tensor* upconv_bias = nullptr;        // [output_dim]

    // 3 个 CausalResidualUnit (dilation=1,3,9)
    std::array<ResidualUnitWeights, 3> residual_units;
};
```

### 3.5 AudioVAEWeights

```cpp
struct AudioVAEWeights {
    // === Encoder ===
    // 初始卷积
    ggml_tensor* encoder_init_weight = nullptr;  // [7, 1, 1, 64]
    ggml_tensor* encoder_init_bias = nullptr;    // [64]

    // Encoder blocks (5 个)
    std::vector<EncoderBlockWeights> encoder_blocks;

    // fc_mu 和 fc_logvar
    ggml_tensor* fc_mu_weight = nullptr;     // [3, 1, 2048, 64]
    ggml_tensor* fc_mu_bias = nullptr;       // [64]
    ggml_tensor* fc_logvar_weight = nullptr; // [3, 1, 2048, 64]
    ggml_tensor* fc_logvar_bias = nullptr;   // [64]

    // === Decoder ===
    // 初始 depthwise + pointwise 卷积
    ggml_tensor* decoder_dw_weight = nullptr;   // [7, 1, 64, 64] groups=64
    ggml_tensor* decoder_pw_weight = nullptr;   // [1, 1, 64, 2048]
    ggml_tensor* decoder_pw_bias = nullptr;     // [2048]

    // Decoder blocks (5 个)
    std::vector<DecoderBlockWeights> decoder_blocks;

    // 最终卷积
    ggml_tensor* final_snake_alpha = nullptr;  // [64]
    ggml_tensor* final_conv_weight = nullptr;  // [7, 1, 64, 1]
    ggml_tensor* final_conv_bias = nullptr;    // [1]
};
```

### 3.6 AudioVAEModel

```cpp
class AudioVAEModel {
public:
    explicit AudioVAEModel(const AudioVAEConfig& config = AudioVAEConfig());
    ~AudioVAEModel();

    // 禁止拷贝
    AudioVAEModel(const AudioVAEModel&) = delete;
    AudioVAEModel& operator=(const AudioVAEModel&) = delete;

    // 权重加载
    bool load_from_gguf(const std::string& gguf_path,
                        VoxCPMContext& weight_ctx,
                        VoxCPMContext& graph_ctx,
                        VoxCPMBackend& backend);

    // 编码接口
    // audio: [1, T] 音频数据 (范围 [-1, 1])
    // 返回: [latent_dim, T/hop_length] 潜在表示
    ggml_tensor* encode(VoxCPMContext& ctx, ggml_tensor* audio);

    // 解码接口
    // z: [latent_dim, T'] 潜在表示
    // 返回: [1, T'*hop_length] 音频数据
    ggml_tensor* decode(VoxCPMContext& ctx, ggml_tensor* z);

    // 访问器
    const AudioVAEConfig& config() const { return config_; }
    int hop_length() const { return config_.hop_length(); }

private:
    // === 组件实现 ===
    ggml_tensor* snake_forward(VoxCPMContext& ctx, ggml_tensor* x, ggml_tensor* alpha);
    ggml_tensor* causal_conv1d(VoxCPMContext& ctx, ggml_tensor* x,
                                ggml_tensor* weight, ggml_tensor* bias, int padding);
    ggml_tensor* causal_conv_transpose1d(VoxCPMContext& ctx, ggml_tensor* x,
                                          ggml_tensor* weight, ggml_tensor* bias,
                                          int stride, int padding, int output_padding);
    ggml_tensor* residual_unit_forward(VoxCPMContext& ctx, ggml_tensor* x,
                                        const ResidualUnitWeights& weights,
                                        int dilation, int kernel, int groups);
    ggml_tensor* encoder_block_forward(VoxCPMContext& ctx, ggml_tensor* x,
                                        const EncoderBlockWeights& weights, int stride, int groups);
    ggml_tensor* decoder_block_forward(VoxCPMContext& ctx, ggml_tensor* x,
                                        const DecoderBlockWeights& weights, int stride, int groups);

    // 权重加载辅助
    bool load_encoder_weights(FILE* file, gguf_context* gguf_ctx, ggml_context* ggml_ctx);
    bool load_decoder_weights(FILE* file, gguf_context* gguf_ctx, ggml_context* ggml_ctx);
    bool load_tensor_data(FILE* file, gguf_context* gguf_ctx, int tensor_idx,
                          ggml_tensor* tensor, ggml_backend_buffer_t buffer);

    AudioVAEConfig config_;
    AudioVAEWeights weights_;

    ggml_context* weight_ctx_ = nullptr;
    ggml_backend_buffer_t weight_buffer_ = nullptr;
};
```

---

## 四、GGUF 权重命名映射

从 `models/voxcpm1.5.dump` 提取的 AudioVAE 张量命名模式：

### 4.1 Encoder 权重

| GGUF Key | 形状 | 说明 |
|----------|------|------|
| `audio_vae.encoder.block.0.weight` | [7, 1, 1, 64] | 初始卷积 |
| `audio_vae.encoder.block.0.bias` | [64] | |
| `audio_vae.encoder.block.{i}.block.{j}.block.{k}.weight` | 多种 | 残差单元卷积 |
| `audio_vae.encoder.block.{i}.block.{j}.block.{k}.alpha` | [dim] | Snake 参数 |
| `audio_vae.encoder.block.{i}.block.4.weight` | [kernel, 1, in, out] | 下采样卷积 |
| `audio_vae.encoder.fc_mu.weight` | [3, 1, 2048, 64] | mu 投影 |
| `audio_vae.encoder.fc_logvar.weight` | [3, 1, 2048, 64] | logvar 投影 |

### 4.2 Decoder 权重

| GGUF Key | 形状 | 说明 |
|----------|------|------|
| `audio_vae.decoder.model.0.weight` | [7, 1, 64, 64] | Depthwise 卷积 |
| `audio_vae.decoder.model.1.weight` | [1, 64, 2048] | Pointwise 卷积 |
| `audio_vae.decoder.model.{i}.block.{j}.weight` | 多种 | 解码块卷积 |
| `audio_vae.decoder.model.7.alpha` | [64] | 最终 Snake |
| `audio_vae.decoder.model.8.bias` | [1] | 最终偏置 |

---

## 五、关键实现细节

### 5.1 Snake 激活函数

```cpp
// snake(x, alpha) = x + (1/alpha) * sin(alpha * x)^2
ggml_tensor* AudioVAEModel::snake_forward(VoxCPMContext& ctx, ggml_tensor* x, ggml_tensor* alpha) {
    ggml_context* raw = ctx.raw_context();

    // alpha * x (broadcast alpha over time dimension)
    ggml_tensor* ax = ggml_mul(raw, x, alpha);

    // sin(alpha * x)^2
    ggml_tensor* sin_ax = ggml_sin(raw, ax);
    ggml_tensor* sin_sq = ggml_sqr(raw, sin_ax);

    // 1/alpha (添加 epsilon 避免除零)
    constexpr float eps = 1e-9f;
    ggml_tensor* alpha_eps = ggml_add(raw, alpha, ggml_new_tensor_1d(raw, GGML_TYPE_F32, 1));
    // 设置 eps 值需要在外部完成
    ggml_tensor* inv_alpha = ggml_reciprocal(raw, alpha_eps);

    // x + sin^2 / alpha
    ggml_tensor* result = ggml_add(raw, x, ggml_mul(raw, sin_sq, inv_alpha));

    return result;
}
```

### 5.2 CausalConv1d (左侧填充)

```cpp
ggml_tensor* AudioVAEModel::causal_conv1d(VoxCPMContext& ctx, ggml_tensor* x,
                                           ggml_tensor* weight, ggml_tensor* bias, int padding) {
    ggml_context* raw = ctx.raw_context();

    if (padding > 0) {
        // 因果卷积：只在左侧填充 (pad*2, 0) -> 左侧填充 2*pad
        // GGML 的 ggml_pad 是两侧填充，需要手动实现左侧填充
        // 方案：先两侧填充 pad*2，然后切片去掉右侧 pad*2

        // 创建填充后的张量
        int64_t padded_len = x->ne[1] + padding * 2;

        // 使用 ggml_pad (两侧填充)
        x = ggml_pad(raw, x, /*pad_0=*/0, /*pad_1=*/padding * 2);

        // 切片去掉右侧填充: [..., :-(padding*2)]
        // 相当于取 [0, padded_len - padding*2)
        x = ggml_view_2d(raw, x, x->ne[0], padded_len - padding * 2,
                         x->nb[1], 0);
    }

    // 使用 padding=0 的卷积
    ggml_tensor* result = ggml_conv_1d(raw, weight, x, /*stride=*/1, /*dilation=*/1, /*padding=*/0);

    // 添加偏置
    if (bias) {
        result = ggml_add(raw, result, bias);
    }

    return result;
}
```

### 5.3 CausalTransposeConv1d (输出裁剪)

```cpp
ggml_tensor* AudioVAEModel::causal_conv_transpose1d(VoxCPMContext& ctx, ggml_tensor* x,
                                                     ggml_tensor* weight, ggml_tensor* bias,
                                                     int stride, int padding, int output_padding) {
    ggml_context* raw = ctx.raw_context();

    // 标准转置卷积
    ggml_tensor* result = ggml_conv_transpose_1d(raw, weight, x, stride, padding);

    // 因果裁剪: [..., :-(padding*2 - output_padding)]
    int crop = padding * 2 - output_padding;
    if (crop > 0) {
        result = ggml_view_2d(raw, result, result->ne[0], result->ne[1] - crop,
                              result->nb[1], 0);
    }

    // 添加偏置
    if (bias) {
        result = ggml_add(raw, result, bias);
    }

    return result;
}
```

### 5.4 CausalResidualUnit

```cpp
ggml_tensor* AudioVAEModel::residual_unit_forward(VoxCPMContext& ctx, ggml_tensor* x,
                                                    const ResidualUnitWeights& weights,
                                                    int dilation, int kernel, int groups) {
    ggml_context* raw = ctx.raw_context();

    // Snake + CausalConv1d (dilated)
    ggml_tensor* h = snake_forward(ctx, x, weights.snake1_alpha);
    int pad = ((kernel - 1) * dilation) / 2;
    h = ggml_conv_1d(raw, weights.conv1_weight, h, /*stride=*/1, dilation, pad);
    if (weights.conv1_bias) {
        h = ggml_add(raw, h, weights.conv1_bias);
    }

    // Snake + CausalConv1d (kernel=1)
    h = snake_forward(ctx, h, weights.snake2_alpha);
    h = ggml_conv_1d(raw, weights.conv2_weight, h, /*stride=*/1, /*dilation=*/1, /*padding=*/0);
    if (weights.conv2_bias) {
        h = ggml_add(raw, h, weights.conv2_bias);
    }

    // 残差连接
    // 处理可能的长度不匹配
    int64_t x_len = x->ne[1];
    int64_t h_len = h->ne[1];
    if (x_len > h_len) {
        int diff = x_len - h_len;
        x = ggml_view_2d(raw, x, x->ne[0], h_len, x->nb[1], (diff / 2) * x->nb[1]);
    }

    return ggml_add(raw, x, h);
}
```

### 5.5 Encoder Forward

```cpp
ggml_tensor* AudioVAEModel::encode(VoxCPMContext& ctx, ggml_tensor* audio) {
    ggml_context* raw = ctx.raw_context();

    // audio: [1, T] -> 添加通道维度 -> [1, T, 1]
    // 但 GGML 卷积输入格式为 [C_out, C_in, K] 的权重
    // 输入格式为 [C, T, B]

    ggml_tensor* h = audio;

    // 初始卷积: [1, T] -> [64, T]
    h = causal_conv1d(ctx, h, weights_.encoder_init_weight, weights_.encoder_init_bias, /*padding=*/3);

    // Encoder blocks
    std::vector<int> channels = config_.encoder_channels();  // [64, 128, 256, 512, 1024, 2048]
    int groups = 1;  // encoder 不使用 depthwise

    for (size_t i = 0; i < weights_.encoder_blocks.size(); ++i) {
        int stride = config_.encoder_rates[i];
        h = encoder_block_forward(ctx, h, weights_.encoder_blocks[i], stride, groups);
    }

    // fc_mu: 仅返回 mu (确定性编码)
    h = ggml_conv_1d(raw, weights_.fc_mu_weight, h, /*stride=*/1, /*dilation=*/1, /*padding=*/1);
    h = ggml_add(raw, h, weights_.fc_mu_bias);

    return h;
}
```

### 5.6 Decoder Forward

```cpp
ggml_tensor* AudioVAEModel::decode(VoxCPMContext& ctx, ggml_tensor* z) {
    ggml_context* raw = ctx.raw_context();

    ggml_tensor* h = z;

    // 初始 depthwise + pointwise 卷积
    // Depthwise: [64, T] -> [64, T]
    h = ggml_conv_1d(raw, weights_.decoder_dw_weight, h, /*stride=*/1, /*dilation=*/1, /*padding=*/3);
    // Pointwise: [64, T] -> [2048, T]
    h = ggml_mul_mat(raw, weights_.decoder_pw_weight, h);
    h = ggml_add(raw, h, weights_.decoder_pw_bias);

    // Decoder blocks
    for (size_t i = 0; i < weights_.decoder_blocks.size(); ++i) {
        int stride = config_.decoder_rates[i];
        int groups = config_.depthwise ? channels_after_block : 1;
        h = decoder_block_forward(ctx, h, weights_.decoder_blocks[i], stride, groups);
    }

    // 最终 Snake + Conv + Tanh
    h = snake_forward(ctx, h, weights_.final_snake_alpha);
    h = ggml_conv_1d(raw, weights_.final_conv_weight, h, /*stride=*/1, /*dilation=*/1, /*padding=*/3);
    if (weights_.final_conv_bias) {
        h = ggml_add(raw, h, weights_.final_conv_bias);
    }
    h = ggml_tanh(raw, h);

    return h;
}
```

---

## 六、实现步骤

### Phase 1: 头文件 (`include/voxcpm/audio-vae.h`)

1. 定义权重结构体 (SnakeWeights, ResidualUnitWeights, EncoderBlockWeights, DecoderBlockWeights, AudioVAEWeights)
2. 定义 AudioVAEModel 类
3. 声明所有公共接口和私有方法

### Phase 2: 核心组件实现 (`src/audio-vae.cpp`)

1. `snake_forward()` - Snake 激活函数
2. `causal_conv1d()` - 因果卷积
3. `causal_conv_transpose1d()` - 因果转置卷积
4. `residual_unit_forward()` - 残差单元
5. `encoder_block_forward()` - 编码块
6. `decoder_block_forward()` - 解码块

### Phase 3: 权重加载 (`src/audio-vae.cpp`)

1. `load_from_gguf()` - 主加载函数
2. `load_encoder_weights()` - 编码器权重
3. `load_decoder_weights()` - 解码器权重
4. `load_tensor_data()` - 张量数据加载辅助

### Phase 4: Forward 接口 (`src/audio-vae.cpp`)

1. `encode()` - 编码接口
2. `decode()` - 解码接口

### Phase 5: 测试 (`tests/test_audio_vae.cpp`)

1. 配置和权重加载测试
2. Encode trace 测试
3. Decode trace 测试
4. 数值误差验证

---

## 七、测试策略

### 7.1 容差设置

根据其他模块的经验，AudioVAE 的容差设置：

| 参数 | 值 | 原因 |
|------|-----|------|
| 绝对容差 | 0.05f | 考虑 float32 精度 + 卷积累积误差 |
| 最大不匹配率 | 0.05f (5%) | 允许少量元素超出容差 |

### 7.2 测试用例

```cpp
TEST_CASE("AudioVAE encode with trace validation", "[audio_vae][encode][trace]") {
    // 1. 加载模型和 trace
    // 2. 构建计算图
    // 3. 执行 encode
    // 4. 与 trace 输出对比
    // 5. 验证误差在容差范围内
}

TEST_CASE("AudioVAE decode with trace validation", "[audio_vae][decode][trace]") {
    // 1. 加载模型和 trace
    // 2. 构建计算图
    // 3. 执行 decode
    // 4. 与 trace 输出对比
    // 5. 验证误差在容差范围内
}
```

### 7.3 张量形状转换

**Encode 输入**:
- PyTorch: `[B=1, T=381024]` 音频
- GGML: `[C=1, T=381024]` (batch=1 时布局相同)

**Encode 输出**:
- PyTorch: `[B=1, C=64, T=216]`
- GGML: `[C=64, T=216, B=1]`

**Decode 输入**:
- PyTorch: `[B=1, C=64, T=36]`
- GGML: `[C=64, T=36, B=1]`

**Decode 输出**:
- PyTorch: `[B=1, C=1, T=63504]`
- GGML: `[C=1, T=63504, B=1]`

---

## 八、GGML 最佳实践检查清单

- [ ] `no_alloc=true` 所有 graph contexts
- [ ] `ggml_set_input()` 所有输入张量
- [ ] `ggml_set_output()` 所有输出张量
- [ ] 权重 Buffer 独立 + `GGML_BACKEND_BUFFER_USAGE_WEIGHTS`
- [ ] Context 大小仅包含元数据开销
- [ ] 使用 VoxCPMBackend 和 VoxCPMContext 封装

---

## 九、关键注意事项

### 1. 权重归一化折叠

PyTorch 的 `weight_norm` 将权重重参数化为 `W = g * V / ||V||`。GGUF 转换时应该已经折叠为单个权重张量，无需在 GGML 中处理。

### 2. 张量内存布局

- PyTorch Conv1d 权重: `[out_channels, in_channels, kernel_size]`
- GGML conv_1d 权重: `[out_channels, in_channels, kernel_size]` (相同!)

### 3. 分组卷积 (Depthwise)

Decoder 初始卷积使用 `groups=input_channel` 的 depthwise 卷积，需确保 GGML 的 `ggml_conv_1d` 正确支持 groups 参数。

### 4. 长度对齐

音频长度必须对齐到 `hop_length` 的倍数。PyTorch 的 `preprocess()` 会自动填充，GGML 版本需要在调用前手动处理。

---

## 十、验证方法

```bash
cd /home/orangepi/Codes/ggbond/VoxCPM.cpp
cmake -B build && cmake --build build
cd build && ctest -R test_audio_vae --output-on-failure
```

---

## 十一、预期测试结果

| 测试 | 预期 |
|------|------|
| AudioVAE 权重加载 | 所有权重正确加载，形状匹配 |
| encode trace 验证 | 与 trace 对比，tolerance=0.05, max_mismatch_rate=0.05 |
| decode trace 验证 | 与 trace 对比，tolerance=0.05, max_mismatch_rate=0.05 |
