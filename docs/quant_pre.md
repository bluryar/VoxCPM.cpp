# VoxCPM.cpp 模型量化研究报告

## 一、当前状态分析

### 模型文件信息

- **文件路径**：`/home/orangepi/Codes/ggbond/VoxCPM.cpp/models/voxcpm1.5.dump`
- **文件格式**：GGUF
- **张量数量**：690 个
- **当前类型**：全部为 `GGML_TYPE_F32` (32-bit 浮点)
- **预估大小**：约 4GB (FP32)

VoxCPM 是一个文本转语音 (TTS) 模型，由以下核心模块组成：

```
┌─────────────────────────────────────────────────────────────────┐
│                        VoxCPM 推理流程                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  音频输入 ──→ [AudioVAE Encoder] ──→ 潜在表示 (64-dim)           │
│                  ↓                                              │
│            [LocEnc] 8层 ──→ 局部特征编码                         │
│                  ↓                                              │
│            [proj.enc_to_lm]                                     │
│                  ↓                                              │
│            [BaseLM] 24层 ──→ 语言建模                           │
│                  ↓                                              │
│     ┌───────────┼───────────┐                                   │
│     ↓           ↓           ↓                                   │
│ [FSQ]    [ResidualLM]   [proj.lm_to_dit]                        │
│ 量化器      8层残差            │                                │
│     │           │              │                                │
│     └───────────┴──────────────┘                                │
│                  ↓                                              │
│            [LocDiT] 8层 ──→ CFM 生成                            │
│                  ↓                                              │
│            [AudioVAE Decoder] ──→ 音频输出                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 代码基础

VoxCPM.cpp 已具备良好的量化基础：

| 组件 | 文件 | 量化支持情况 |
|------|------|-------------|
| 权重加载 | `src/weight-store.cpp` | ✅ 已支持 GGUF 格式，可直接加载量化模型 |
| 矩阵乘法 | `src/minicpm.cpp` | ✅ `ggml_mul_mat()` 自动支持量化权重 |
| 后端管理 | `src/backend.cpp` | ✅ 支持多后端，量化权重无需特殊处理 |

**关键结论**：GGML 的 `ggml_mul_mat()` 已内置支持量化权重，推理代码无需修改即可使用量化模型！

---

## 二、GGML 支持的量化类型

### 2.1 基础量化类型

| 类型 | 枚举值 | 比特/权重 | 块大小 | 特点 |
|------|--------|----------|--------|------|
| Q4_0 | `GGML_TYPE_Q4_0` | 4.5 | 32 | 对称量化，最简单 |
| Q4_1 | `GGML_TYPE_Q4_1` | 5.0 | 32 | 非对称量化（scale + min） |
| Q5_0 | `GGML_TYPE_Q5_0` | 5.5 | 32 | 5-bit 对称 |
| Q5_1 | `GGML_TYPE_Q5_1` | 6.0 | 32 | 5-bit 非对称 |
| Q8_0 | `GGML_TYPE_Q8_0` | 8.5 | 32 | 8-bit 对称，高精度 |

### 2.2 K-Quant 类型（推荐）

K-Quant 使用超块（super-block）架构，精度更高：

| 类型 | 枚举值 | 比特/权重 | 超块大小 | 特点 |
|------|--------|----------|----------|------|
| Q2_K | `GGML_TYPE_Q2_K` | 2.6 | 256 | 极低比特，需要 importance matrix |
| Q3_K | `GGML_TYPE_Q3_K` | 3.4 | 256 | 低比特 |
| Q4_K | `GGML_TYPE_Q4_K` | 4.5 | 256 | **推荐**，平衡精度和压缩 |
| Q5_K | `GGML_TYPE_Q5_K` | 5.5 | 256 | 高精度 |
| Q6_K | `GGML_TYPE_Q6_K` | 6.6 | 256 | 更高精度 |

### 2.3 Importance-Weighted 类型（IQ）

使用重要性矩阵的高精度量化：

| 类型 | 比特/权重 | 特点 |
|------|----------|------|
| IQ1_S | 1.56 | 需要 importance matrix |
| IQ1_M | 1.75 | 需要 importance matrix |
| IQ2_XXS | 2.06 | 需要 importance matrix |
| IQ3_XXS | 3.06 | 需要 importance matrix |
| IQ4_NL | 4.50 | 非线性量化 |
| IQ4_XS | 4.25 | 需要 importance matrix |

### 2.4 其他类型

| 类型 | 比特/权重 | 特点 |
|------|----------|------|
| F16 | 16 | 半精度浮点，无损压缩 |
| BF16 | 16 | Brain Float 16 |
| MXFP4 | 4.0 | MXFP4 (E8M0 scale) |

---

## 三、llama.cpp 量化原理详解

### 3.1 llama.cpp 量化架构概览

llama.cpp 是功能最完整、最成熟的 GGML 量化实现。其量化工具位于：
- CLI 工具：`tools/quantize/quantize.cpp`
- 核心实现：`src/llama-quant.cpp`

**量化流程**：

```
输入 GGUF (FP32/FP16)
        ↓
[llama_model_loader] 加载模型元数据和权重
        ↓
[llama_tensor_get_type] 智能选择每个张量的量化类型
        ↓
[llama_tensor_dequantize_impl] 如需反量化到 FP32
        ↓
[llama_tensor_quantize_impl] 多线程并行量化
        ↓
[ggml_validate_row_data] 验证量化数据有效性
        ↓
输出 GGUF (量化格式)
```

### 3.2 智能张量类型选择

llama.cpp 最大的特色是 **根据张量名称、层位置、模型架构自动选择最佳量化类型**：

```cpp
// llama-quant.cpp: llama_tensor_get_type()
static ggml_type llama_tensor_get_type(quantize_state_impl & qs, ggml_type new_type, const ggml_tensor * tensor, llama_ftype ftype) {
    const std::string name = ggml_get_name(tensor);

    // 1. 输出层和 Embedding 层使用更高精度
    if (name == "output.weight" || name == "token_embd.weight") {
        if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_XXS) {
            new_type = GGML_TYPE_Q5_K;  // IQ2_XXS 对输出层使用 Q5_K
        } else if (new_type != GGML_TYPE_Q8_0) {
            new_type = GGML_TYPE_Q6_K;  // 默认使用 Q6_K
        }
    }

    // 2. Attention V 层：根据 GQA 比例调整
    else if (name.find("attn_v.weight") != std::string::npos) {
        if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K) {
            new_type = qs.model.hparams.n_gqa() >= 4 ? GGML_TYPE_Q4_K : GGML_TYPE_Q3_K;
        }
        // 70B 模型 attn_v 是 8x 共享，使用更高精度
        if (qs.model.type == LLM_TYPE_70B) {
            if (new_type == GGML_TYPE_Q3_K || new_type == GGML_TYPE_Q4_K) {
                new_type = GGML_TYPE_Q5_K;
            }
        }
    }

    // 3. FFN down 层：根据层位置调整精度
    else if (name.find("ffn_down") != std::string::npos) {
        auto use_more_bits = [](int i_layer, int n_layers) -> bool {
            return i_layer < n_layers/8 || i_layer >= 7*n_layers/8 || (i_layer - n_layers/8)%3 == 2;
        };
        // 首尾 1/8 的层使用更高精度
        if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M && use_more_bits(i_layer, n_layer)) {
            new_type = GGML_TYPE_Q6_K;
        }
    }

    return new_type;
}
```

**分层量化策略**：

| 张量类型 | Q4_K_M 默认 | 调整条件 |
|----------|-------------|----------|
| token_embd.weight | Q6_K | 输出层精度 |
| output.weight | Q6_K | 输出层精度 |
| attn_v.weight | Q4_K → Q5_K/Q6_K | GQA>=4 或 70B 模型 |
| ffn_down (首尾层) | Q4_K → Q6_K | 层位置在首尾 1/8 |
| ffn_gate/up | Q4_K | 一般精度 |

### 3.3 多线程并行量化

```cpp
// llama-quant.cpp: llama_tensor_quantize_impl()
static size_t llama_tensor_quantize_impl(
    enum ggml_type new_type,
    const float * f32_data,
    void * new_data,
    const int64_t chunk_size,
    int64_t nrows,
    int64_t n_per_row,
    const float * imatrix,
    std::vector<std::thread> & workers,
    const int nthread
) {
    std::mutex mutex;
    int64_t counter = 0;
    size_t new_size = 0;
    bool valid = true;

    auto compute = [&]() {
        const int64_t nrows_per_chunk = chunk_size / n_per_row;
        size_t local_size = 0;

        while (true) {
            std::unique_lock<std::mutex> lock(mutex);
            int64_t first_row = counter;
            counter += nrows_per_chunk;
            if (first_row >= nrows) {
                if (local_size > 0) new_size += local_size;
                break;
            }
            lock.unlock();

            // 调用 GGML 核心量化函数
            const int64_t this_nrow = std::min(nrows - first_row, nrows_per_chunk);
            size_t this_size = ggml_quantize_chunk(
                new_type, f32_data, new_data,
                first_row * n_per_row, this_nrow, n_per_row, imatrix
            );
            local_size += this_size;

            // 验证量化数据
            const size_t row_size = ggml_row_size(new_type, n_per_row);
            void * this_data = (char *)new_data + first_row * row_size;
            if (!ggml_validate_row_data(new_type, this_data, this_size)) {
                valid = false;
                break;
            }
        }
    };

    // 启动工作线程
    for (int it = 0; it < nthread - 1; ++it) {
        workers.emplace_back(compute);
    }
    compute();
    for (auto & w : workers) { w.join(); }

    return new_size;
}
```

### 3.4 Importance Matrix 支持

低比特量化（IQ1_S, IQ2_XXS, Q2_K 等）需要 importance matrix：

```cpp
// 检查是否需要 importance matrix
static bool tensor_type_requires_imatrix(const ggml_tensor * t, const ggml_type dst_type, const llama_ftype ftype) {
    return (
        dst_type == GGML_TYPE_IQ2_XXS || dst_type == GGML_TYPE_IQ2_XS ||
        dst_type == GGML_TYPE_IQ3_XXS || dst_type == GGML_TYPE_IQ1_S  ||
        dst_type == GGML_TYPE_IQ2_S   || dst_type == GGML_TYPE_IQ1_M
    );
}
```

**生成 importance matrix**：

```bash
# 使用校准数据生成 importance matrix
./llama-imatrix --model model-f32.gguf --data calibration.txt --output imatrix.dat

# 使用 importance matrix 量化
./llama-quantize --imatrix imatrix.dat model-f32.gguf model-iq4_xs.gguf IQ4_XS
```

### 3.5 支持的量化类型

llama.cpp 支持最广泛的量化类型：

```cpp
// tools/quantize/quantize.cpp
static const std::vector<quant_option> QUANT_OPTIONS = {
    // 基础量化
    { "Q4_0",  LLAMA_FTYPE_MOSTLY_Q4_0,  " 4.34G, +0.4685 ppl @ Llama-3-8B" },
    { "Q5_0",  LLAMA_FTYPE_MOSTLY_Q5_0,  " 5.21G, +0.1316 ppl @ Llama-3-8B" },
    { "Q8_0",  LLAMA_FTYPE_MOSTLY_Q8_0,  " 7.96G, +0.0026 ppl @ Llama-3-8B" },

    // K-Quant（推荐）
    { "Q4_K_S", LLAMA_FTYPE_MOSTLY_Q4_K_S, " 4.37G, +0.2689 ppl" },
    { "Q4_K_M", LLAMA_FTYPE_MOSTLY_Q4_K_M, " 4.58G, +0.1754 ppl" },  // 推荐
    { "Q5_K_M", LLAMA_FTYPE_MOSTLY_Q5_K_M, " 5.33G, +0.0569 ppl" },
    { "Q6_K",   LLAMA_FTYPE_MOSTLY_Q6_K,   " 6.14G, +0.0217 ppl" },

    // IQ 系列（需要 imatrix）
    { "IQ1_S",  LLAMA_FTYPE_MOSTLY_IQ1_S,  " 1.56 bpw" },
    { "IQ2_XXS",LLAMA_FTYPE_MOSTLY_IQ2_XXS," 2.06 bpw" },
    { "IQ3_XXS",LLAMA_FTYPE_MOSTLY_IQ3_XXS," 3.06 bpw" },
    { "IQ4_NL", LLAMA_FTYPE_MOSTLY_IQ4_NL, " 4.50 bpw" },
    { "IQ4_XS", LLAMA_FTYPE_MOSTLY_IQ4_XS, " 4.25 bpw" },

    // 浮点
    { "F16",    LLAMA_FTYPE_MOSTLY_F16,    "14.00G, +0.0020 ppl" },
    { "BF16",   LLAMA_FTYPE_MOSTLY_BF16,   "14.00G, -0.0050 ppl" },
};
```

### 3.6 高级选项

```bash
# 不量化输出层（保持更高精度）
./llama-quantize --leave-output-tensor model.gguf model-q4_k.gguf Q4_K

# 纯量化（不使用混合精度）
./llama-quantize --pure model.gguf model-q4_k.gguf Q4_K

# 指定特定张量的量化类型
./llama-quantize --tensor-type attn_q=q8_0 model.gguf model-q4_k.gguf Q4_K

# 裁剪层
./llama-quantize --prune-layers 0,1,2 model.gguf model-pruned.gguf Q4_K

# 预估量化后大小（不实际量化）
./llama-quantize --dry-run model.gguf Q4_K
```

---

## 四、参考实现对比分析

### 4.1 三种量化实现对比

| 特性 | llama.cpp | SenseVoice.cpp | whisper.cpp |
|------|-----------|----------------|-------------|
| **文件格式** | GGUF（现代格式） | GGUF（现代格式） | Legacy GGML 二进制 |
| **元数据处理** | `gguf_context` API 自动处理 | `gguf_context` API 自动处理 | 手动读取/写入 struct |
| **量化方式** | 全量加载后量化 | 全量加载后量化 | 流式逐张量处理 |
| **多线程支持** | ✅ 多线程量化 | ✅ 多线程量化 | ❌ 单线程 |
| **数据验证** | ✅ `ggml_validate_row_data()` | ✅ `ggml_validate_row_data()` | ❌ 无验证 |
| **智能类型选择** | ✅ 根据张量名称/位置自动调整 | ❌ 固定类型 | ❌ 固定类型 |
| **分层量化** | ✅ 首尾层更高精度 | ❌ | ❌ |
| **Importance Matrix** | ✅ 完整支持 | ✅ 支持 | ❌ 不支持 |
| **内存占用** | 较高（全量加载） | 较高（全量加载） | 较低（流式处理） |
| **代码复杂度** | 高（功能最全） | 中等 | 简单 |
| **量化类型支持** | 最全（包括 IQ 系列） | 中等 | 基础类型 |

**推荐使用**：
- **llama.cpp**：功能最完整，支持智能分层量化，推荐用于生产环境
- **SenseVoice.cpp**：适合 GGUF 格式的简单量化需求
- **whisper.cpp**：仅适用于传统 GGML 格式

### 4.2 SenseVoice.cpp 量化实现详解

**文件位置**：`VoxCPM.cpp/third_party/SenseVoice.cpp/examples/`

**核心特点**：
1. 使用 GGUF 格式，元数据自动处理
2. 多线程量化，速度更快
3. 支持数据验证
4. 使用 `gguf_context` 管理张量信息

**关键代码结构**：

```cpp
// quantize.cc - 主程序
// 1. 加载 GGUF 模型 (no_alloc=false，需要实际数据)
struct gguf_init_params gguf_params = {
    .no_alloc = false,  // 重要：需要加载实际数据
    .ctx = &ctx,
};
struct gguf_context* gguf_ctx = gguf_init_from_file(fname_inp.c_str(), gguf_params);

// 2. 定义跳过列表
const std::vector<std::string> to_skip = {
    "embed.weight",
    "encoder.*.fsmn_block.weight",
    "encoder.encoders0.0.norm1.weight",
    // ... 更多跳过规则
};

// 3. 调用核心量化函数
sense_voice_ggml_quantize0(ctx, gguf_ctx, fname_inp, fname_out, ftype, 4, { ".*" }, to_skip);
```

**核心量化实现** (`common-ggml.cc`)：

```cpp
bool sense_voice_ggml_quantize0(...) {
    // 创建输出 GGUF context
    struct gguf_context* ctx_out = gguf_init_empty();
    gguf_set_kv(ctx_out, gguf_input);
    gguf_set_val_u32(ctx_out, "general.quantization_version", GGML_QNT_VERSION);
    gguf_set_val_u32(ctx_out, "general.file_type", ftype);

    // 多线程量化
    static size_t sense_voice_tensor_quantize_internal(
        enum ggml_type new_type,
        const float * f32_data,
        void * new_data,
        const int64_t chunk_size,
        int64_t nrows,
        int64_t n_per_row,
        const float * imatrix,
        std::vector<std::thread> & workers,
        const int nthread
    ) {
        // 分块并行量化
        auto compute = [&]() {
            while (true) {
                std::unique_lock<std::mutex> lock(mutex);
                int64_t first_row = counter; counter += nrows_per_chunk;
                if (first_row >= nrows) break;
                lock.unlock();

                size_t this_size = ggml_quantize_chunk(
                    new_type, f32_data, new_data,
                    first_row * n_per_row, this_nrow, n_per_row, imatrix
                );

                // 验证量化数据
                if (!ggml_validate_row_data(new_type, this_data, this_size)) {
                    valid = false;
                    break;
                }
            }
        };
        // 启动多线程...
    }
}
```

**量化条件判断**：

```cpp
// 判断是否需要量化
bool quantize = name.rfind("weight") == name.size() - 6;  // 以 "weight" 结尾
quantize &= (ggml_n_dims(tensor) >= 2);                    // 至少 2 维
quantize &= name.find("_norm.weight") == std::string::npos; // 跳过 norm 层

// 检查跳过列表
for (const auto& s : to_skip) {
    if (std::regex_match(name, std::regex(s))) {
        quantize = false;
        break;
    }
}
```

### 4.3 whisper.cpp 量化实现详解

**文件位置**：`VoxCPM.cpp/third_party/whisper.cpp/examples/quantize/`

**核心特点**：
1. 使用传统 GGML 二进制格式（非 GGUF）
2. 流式处理，内存占用低
3. 单线程处理，代码简单
4. 手动处理元数据

**文件格式验证**：

```cpp
// quantize.cpp
// 验证 magic number
uint32_t magic;
finp.read((char*)&magic, sizeof(magic));
if (magic != GGML_FILE_MAGIC) {
    fprintf(stderr, "invalid model file (bad magic)\n");
    return false;
}
```

**手动处理超参数**：

```cpp
// 读取超参数
finp.read((char*)&hparams.n_vocab,       sizeof(hparams.n_vocab));
finp.read((char*)&hparams.n_audio_ctx,   sizeof(hparams.n_audio_ctx));
finp.read((char*)&hparams.n_audio_state, sizeof(hparams.n_audio_state));
// ... 更多参数

// 更新 ftype 并写入
const int32_t ftype_dst = GGML_QNT_VERSION * GGML_QNT_VERSION_FACTOR + ftype;
fout.write((const char*)&ftype_dst, sizeof(ftype_dst));
```

**核心量化实现** (`common-ggml.cpp`)：

```cpp
bool ggml_common_quantize_0(
    std::ifstream& finp,
    std::ofstream& fout,
    const ggml_ftype ftype,
    const std::vector<std::string>& to_quant,
    const std::vector<std::string>& to_skip
) {
    // 流式处理：逐张量读取、量化、写入
    while (true) {
        // 1. 读取张量头信息
        int32_t n_dims, length, ttype;
        finp.read(reinterpret_cast<char*>(&n_dims), sizeof(n_dims));
        finp.read(reinterpret_cast<char*>(&length), sizeof(length));
        finp.read(reinterpret_cast<char*>(&ttype),  sizeof(ttype));

        if (finp.eof()) break;

        // 2. 读取维度信息
        int32_t ne[4] = {1, 1, 1, 1};
        for (int i = 0; i < n_dims; ++i) {
            finp.read(reinterpret_cast<char*>(&ne[i]), sizeof(ne[i]));
        }

        // 3. 读取张量名称
        std::string name(length, 0);
        finp.read(&name[0], length);

        // 4. 判断是否量化
        bool quantize = false;
        for (const auto& s : to_quant) {
            if (std::regex_match(name, std::regex(s))) {
                quantize = true;
                break;
            }
        }
        for (const auto& s : to_skip) {
            if (std::regex_match(name, std::regex(s))) {
                quantize = false;
                break;
            }
        }
        quantize &= (n_dims == 2);  // 只量化 2D 张量

        // 5. 处理数据
        if (quantize) {
            // 读取并转换为 F32
            if (ttype == GGML_TYPE_F16) {
                data_f16.resize(nelements);
                finp.read(reinterpret_cast<char*>(data_f16.data()), nelements * sizeof(ggml_fp16_t));
                for (int i = 0; i < nelements; ++i) {
                    data_f32[i] = ggml_fp16_to_fp32(data_f16[i]);
                }
            } else {
                data_f32.resize(nelements);
                finp.read(reinterpret_cast<char*>(data_f32.data()), nelements * sizeof(float));
            }

            // 量化
            size_t cur_size = ggml_quantize_chunk(
                (ggml_type)ttype,
                data_f32.data(),
                work.data(),
                0,
                nelements / ne[0],  // nrows
                ne[0],              // n_per_row
                nullptr             // imatrix
            );

            // 写入
            fout.write(reinterpret_cast<char*>(work.data()), cur_size);
        } else {
            // 直接复制
            data_u8.resize(nelements * bpe);
            finp.read(reinterpret_cast<char*>(data_u8.data()), nelements * bpe);
            fout.write(reinterpret_cast<char*>(data_u8.data()), data_u8.size());
        }
    }
}
```

**跳过列表**：

```cpp
const std::vector<std::string> to_skip = {
    "encoder.conv1.bias",
    "encoder.conv2.bias",
    "encoder.positional_embedding",
    "decoder.positional_embedding",
};
```

### 4.4 实现风格对比总结

| 方面 | SenseVoice.cpp | whisper.cpp |
|------|----------------|-------------|
| **适合场景** | GGUF 格式模型 | 传统 GGML 格式 |
| **内存效率** | 全量加载，内存占用高 | 流式处理，内存占用低 |
| **量化速度** | 多线程，更快 | 单线程，较慢 |
| **代码复杂度** | 中等（需要管理 GGUF context） | 简单（直接文件 I/O） |
| **数据验证** | 有（更安全） | 无 |
| **VoxCPM 适用性** | ✅ 高（GGUF 格式） | ❌ 低（格式不同） |

---

## 五、量化方法详解

### 方法 1：使用 llama.cpp 预量化（最简单）

**原理**：使用 llama.cpp 的 `llama-quantize` 工具预先将 GGUF 模型量化。

**步骤**：

```bash
# 1. 进入 llama.cpp 目录
cd /home/orangepi/Codes/ggbond/vendor/llama.cpp

# 2. 构建项目
cmake -B build && cmake --build build

# 3. 量化模型
./build/bin/llama-quantize \
    /home/orangepi/Codes/ggbond/VoxCPM.cpp/models/voxcpm1.5.dump \
    /home/orangepi/Codes/ggbond/VoxCPM.cpp/models/voxcpm1.5-Q4_K.gguf \
    Q4_K
```

**支持的量化类型命令**：

```bash
# Q4_K 量化 (推荐)
./llama-quantize model.gguf model-Q4_K.gguf Q4_K

# Q5_K 量化 (更高精度)
./llama-quantize model.gguf model-Q5_K.gguf Q5_K

# Q8_0 量化 (最高精度)
./llama-quantize model.gguf model-Q8_0.gguf Q8_0

# F16 (无损压缩)
./llama-quantize model.gguf model-F16.gguf F16
```

**优点**：
- ✅ 无需修改 VoxCPM.cpp 代码
- ✅ 支持所有量化类型
- ✅ 量化后文件可直接使用
- ✅ 经过广泛测试

**缺点**：
- ❌ 需要额外步骤
- ❌ 需要管理多个模型文件

---

### 方法 2：使用 SenseVoice.cpp 风格的量化工具

**原理**：参考 SenseVoice.cpp 的 `examples/quantize/` 实现，为 VoxCPM.cpp 创建专用量化工具。

**推荐原因**：VoxCPM.cpp 使用 GGUF 格式，与 SenseVoice.cpp 格式相同，可直接复用代码。

**关键代码结构**：

```
VoxCPM.cpp/
├── examples/
│   └── quantize/
│       ├── quantize.cpp          # 主程序入口
│       ├── CMakeLists.txt        # 构建配置
│       └── readme.md             # 使用说明
├── src/
│   └── quantize.cpp              # 量化核心实现 (可选)
└── include/voxcpm/
    └── quantize.h                # 量化 API (可选)
```

**VoxCPM.cpp 专用量化工具模板**：

```cpp
// examples/quantize/quantize.cpp
#include "ggml.h"
#include "gguf.h"
#include <fstream>
#include <regex>
#include <thread>

// VoxCPM 模型超参数
struct voxcpm_hparams {
    int32_t n_vocab;
    int32_t lm_hidden_size;
    int32_t lm_num_layers;
    int32_t lm_num_heads;
    int32_t patch_size;
    int32_t feat_dim;
    // ...
};

// 跳过列表（VoxCPM 特定）
const std::vector<std::string> to_skip = {
    "token_embd.weight",        // 词嵌入保持高精度
    "output.weight",            // 输出层保持高精度
    ".*_norm.weight",           // Norm 层
    ".*.ln.*.weight",           // Layer Norm
    ".*.bias",                  // Bias 不量化
};

// 量化函数（复用 SenseVoice 风格）
bool voxcpm_quantize(
    const std::string& fname_inp,
    const std::string& fname_out,
    ggml_ftype ftype,
    int nthread
) {
    // 1. 加载 GGUF 模型
    struct ggml_context* ctx = nullptr;
    struct gguf_init_params params = {
        .no_alloc = false,
        .ctx = &ctx,
    };
    struct gguf_context* gguf_ctx = gguf_init_from_file(fname_inp.c_str(), params);

    // 2. 创建输出 context
    struct gguf_context* ctx_out = gguf_init_empty();
    gguf_set_kv(ctx_out, gguf_ctx);
    gguf_set_val_u32(ctx_out, "general.quantization_version", GGML_QNT_VERSION);
    gguf_set_val_u32(ctx_out, "general.file_type", ftype);

    // 3. 遍历张量并量化（多线程）
    const int n_tensors = gguf_get_n_tensors(gguf_ctx);
    std::vector<std::thread> workers;
    workers.reserve(nthread);

    for (int i = 0; i < n_tensors; ++i) {
        const std::string name = gguf_get_tensor_name(gguf_ctx, i);
        struct ggml_tensor* tensor = ggml_get_tensor(ctx, name.c_str());

        // 判断是否量化
        bool quantize = should_quantize(tensor, name);

        if (quantize) {
            // 多线程量化
            size_t new_size = tensor_quantize_internal(...);
        }

        // 写入输出
        gguf_set_tensor_type(ctx_out, name.c_str(), new_type);
        gguf_set_tensor_data(ctx_out, name.c_str(), new_data);
    }

    // 4. 写入文件
    // ...

    return true;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s model-f32.gguf model-quant.gguf type\n", argv[0]);
        return 1;
    }

    const std::string fname_inp = argv[1];
    const std::string fname_out = argv[2];
    const ggml_ftype ftype = ggml_parse_ftype(argv[3]);

    return voxcpm_quantize(fname_inp, fname_out, ftype, 4) ? 0 : 1;
}
```

**使用方式**：

```bash
# 构建
cd VoxCPM.cpp && cmake -B build && cmake --build build

# 量化
./build/bin/voxcpm-quantize models/voxcpm1.5.dump models/voxcpm1.5-Q4_K.gguf Q4_K
```

---

### 方法 3：使用 whisper.cpp 风格（流式处理）

**适用场景**：如果 VoxCPM 模型使用传统 GGML 二进制格式（非 GGUF），可参考此方法。

**核心优势**：
- 流式处理，内存占用低
- 适合处理超大模型

**注意**：VoxCPM.cpp 使用 GGUF 格式，不建议使用此方法。但若需要处理超大模型或内存受限场景，可借鉴流式处理思想。

---

### 方法 4：运行时量化（load_and_quantize）

**原理**：在加载模型时动态进行量化，用户只需保存一个 FP32 模型。

**在 weight-store.cpp 中添加**：

```cpp
bool VoxCPMWeightStore::load_and_quantize(
    const std::string& gguf_path,
    VoxCPMBackend& backend,
    ggml_type quantize_type
) {
    // 1. 先加载 FP32 模型
    if (!load_from_file(gguf_path, backend)) {
        return false;
    }

    // 2. 遍历张量进行量化
    const int n_tensors = gguf_get_n_tensors(gguf_ctx_);
    for (int i = 0; i < n_tensors; ++i) {
        const char* name = gguf_get_tensor_name(gguf_ctx_, i);
        ggml_tensor* tensor = ggml_get_tensor(ggml_ctx_, name);

        if (should_quantize(tensor, name)) {
            // 在原位置量化（节省内存）
            quantize_tensor_inplace(tensor, quantize_type);
        }
    }

    return true;
}
```

**优点**：
- ✅ 单一模型文件
- ✅ 可动态选择量化类型

**缺点**：
- ❌ 启动时量化开销
- ❌ 需要额外内存
- ❌ 首次加载较慢

---

### 方法 5：混合精度量化

**原理**：针对不同类型的层使用不同量化精度。

**推荐配置**：

```cpp
ggml_type get_quantize_type_for_tensor(const std::string& name, ggml_type base_type) {
    // 1. Embedding 层保持较高精度
    if (name.find("token_embd") != std::string::npos ||
        name.find("output") != std::string::npos) {
        return GGML_TYPE_Q8_0;  // 或 F16
    }

    // 2. Attention K/V 保持较高精度
    if (name.find("_k.weight") != std::string::npos ||
        name.find("_v.weight") != std::string::npos) {
        return GGML_TYPE_Q8_0;  // 或 Q5_K
    }

    // 3. FFN 层可用较低精度
    if (name.find("ffn_") != std::string::npos) {
        return GGML_TYPE_Q4_K;
    }

    // 4. 其他层使用基础类型
    return base_type;
}
```

**分层量化策略**：

| 层类型 | 推荐类型 | 理由 |
|--------|----------|------|
| Token Embedding | Q8_0 / F16 | 词嵌入对生成质量影响大 |
| Attention Q | Q4_K / Q5_K | 一般精度即可 |
| Attention K/V | Q8_0 / Q5_K | 保持注意力精度 |
| Attention O | Q4_K | 一般精度 |
| FFN gate/up | Q4_K | 可用较低精度 |
| FFN down | Q4_K / Q5_K | 输出层稍高精度 |
| Norm 层 | F32 | 参数极少，不量化 |

---

## 六、GGML 量化核心 API 参考

### 5.1 量化函数

```cpp
// 核心量化函数
size_t ggml_quantize_chunk(
    enum ggml_type type,        // 目标量化类型
    const float * src,          // 源 FP32 数据
    void * dst,                 // 目标缓冲区
    int64_t start,              // 起始行索引
    int64_t nrows,              // 要量化的行数
    int64_t n_per_row,          // 每行元素数
    const float * imatrix       // 重要性矩阵 (可选)
);
```

### 5.2 辅助函数

```cpp
// 计算量化后的一行大小
size_t ggml_row_size(enum ggml_type type, int64_t ne);

// 计算量化后的总大小
size_t ggml_nbytes(const struct ggml_tensor * tensor);

// 验证量化数据有效性
bool ggml_validate_row_data(enum ggml_type type, const void * data, size_t size);

// 检查是否为量化类型
bool ggml_is_quantized(enum ggml_type type);

// 获取类型名称
const char * ggml_type_name(enum ggml_type type);
```

### 5.3 类型映射（ftype → ggml_type）

```cpp
enum ggml_ftype ggml_ftype_from_string(const char* str) {
    static const std::map<std::string, ggml_ftype> map = {
        {"q4_0", GGML_FTYPE_MOSTLY_Q4_0},
        {"q4_1", GGML_FTYPE_MOSTLY_Q4_1},
        {"q5_0", GGML_FTYPE_MOSTLY_Q5_0},
        {"q5_1", GGML_FTYPE_MOSTLY_Q5_1},
        {"q8_0", GGML_FTYPE_MOSTLY_Q8_0},
        {"q2_k", GGML_FTYPE_MOSTLY_Q2_K},
        {"q3_k", GGML_FTYPE_MOSTLY_Q3_K},
        {"q4_k", GGML_FTYPE_MOSTLY_Q4_K},
        {"q5_k", GGML_FTYPE_MOSTLY_Q5_K},
        {"q6_k", GGML_FTYPE_MOSTLY_Q6_K},
    };
    auto it = map.find(str);
    return it != map.end() ? it->second : GGML_FTYPE_UNKNOWN;
}

ggml_type ggml_type_from_ftype(ggml_ftype ftype) {
    switch (ftype) {
        case GGML_FTYPE_MOSTLY_Q4_0: return GGML_TYPE_Q4_0;
        case GGML_FTYPE_MOSTLY_Q4_1: return GGML_TYPE_Q4_1;
        case GGML_FTYPE_MOSTLY_Q4_K: return GGML_TYPE_Q4_K;
        // ... 其他映射
        default: return GGML_TYPE_F32;
    }
}
```

---

## 七、预期收益估算

假设 VoxCPM 1.5 模型大小为 **4GB (FP32)**：

| 量化类型 | 模型大小 | 显存占用 | 压缩比 | 精度损失 |
|----------|----------|----------|--------|----------|
| FP32 (原始) | 4.0 GB | 4.0 GB | 1x | 无 |
| F16 | 2.0 GB | 2.0 GB | 2x | 极小 |
| Q8_0 | 1.1 GB | 1.1 GB | 3.7x | 小 |
| Q6_K | 825 MB | 825 MB | 4.9x | 小 |
| Q5_K | 730 MB | 730 MB | 5.5x | 中等 |
| Q4_K | 600 MB | 600 MB | 6.7x | 中等 |
| Q4_0 | 575 MB | 575 MB | 7.0x | 较大 |
| Q3_K | 450 MB | 450 MB | 8.9x | 较大 |
| Q2_K | 340 MB | 340 MB | 11.8x | 大 |

**推荐选择**：
- **通用场景**：Q4_K（~600MB，精度损失可接受）
- **高精度场景**：Q5_K 或 Q8_0
- **极端压缩场景**：Q2_K（需 importance matrix）

---

## 八、实施建议

### 7.1 快速验证（推荐首次尝试）

```bash
# 使用 llama.cpp 工具直接量化
cd /home/orangepi/Codes/ggbond/vendor/llama.cpp
cmake -B build && cmake --build build

./build/bin/llama-quantize \
    /home/orangepi/Codes/ggbond/VoxCPM.cpp/models/voxcpm1.5.dump \
    /home/orangepi/Codes/ggbond/VoxCPM.cpp/models/voxcpm1.5-Q4_K.gguf \
    Q4_K
```

### 7.2 完整工具开发

如需开发专用量化工具，参考以下文件：

| 参考项目 | 文件路径 | 适用性 |
|----------|----------|--------|
| SenseVoice.cpp 量化工具 | `VoxCPM.cpp/third_party/SenseVoice.cpp/examples/quantize/` | ✅ 推荐（GGUF 格式相同） |
| SenseVoice 量化实现 | `VoxCPM.cpp/third_party/SenseVoice.cpp/examples/common-ggml.cc` | ✅ 推荐 |
| whisper.cpp 量化工具 | `VoxCPM.cpp/third_party/whisper.cpp/examples/quantize/` | ⚠️ 参考（格式不同） |
| llama.cpp 量化工具 | `vendor/llama.cpp/tools/quantize/` | ✅ 功能最全 |
| GGML 量化 API | `VoxCPM.cpp/third_party/ggml/src/ggml-quants.h` | ✅ 底层 API |

### 7.3 验证步骤

1. **构建测试**
   ```bash
   cd VoxCPM.cpp && cmake -B build && cmake --build build
   ```

2. **加载量化模型**
   - 直接使用现有的 `VoxCPMWeightStore::load_from_file()`
   - 无需修改代码，GGUF 已支持量化类型

3. **运行推理**
   ```bash
   ./build/bin/voxcpm-inference models/voxcpm1.5-Q4_K.gguf --input test.wav
   ```

4. **对比结果**
   - 比较 FP32 和量化模型的输出质量
   - 测量显存占用和推理速度

---

## 九、注意事项

### 8.1 不应量化的张量

| 张量类型 | 原因 |
|----------|------|
| Embedding 权重 | 对生成质量影响大 |
| 输出层权重 | 对最终输出影响大 |
| Norm 层权重 | 参数极少，量化收益小 |
| Bias 向量 | 1D 张量，不支持量化 |
| 位置编码 | 参数少，保持精度 |

### 8.2 GPU 兼容性

部分量化类型在不同后端的支持情况：

| 类型 | CPU | CUDA | Metal (Apple) |
|------|-----|------|---------------|
| Q4_0 | ✅ | ✅ | ✅ |
| Q4_K | ✅ | ✅ | ✅ |
| Q5_K | ✅ | ✅ | ✅ |
| Q8_0 | ✅ | ✅ | ✅ |
| IQ 系列 | ✅ | 部分 | 部分 |

### 8.3 Importance Matrix

低比特量化（Q2_K, IQ1_S 等）需要 importance matrix：

```bash
# 生成 importance matrix
./llama-imatrix --model model.gguf --data calibration.txt --output imatrix.dat

# 使用 importance matrix 量化
./llama-quantize --imatrix imatrix.dat model.gguf model-Q2_K.gguf Q2_K
```

---

## 十、总结

| 方法 | 复杂度 | 推荐度 | 适用场景 |
|------|--------|--------|----------|
| llama.cpp 预量化 | ⭐ | ⭐⭐⭐⭐⭐ | 快速验证、生产使用 |
| SenseVoice 风格工具 | ⭐⭐⭐ | ⭐⭐⭐⭐ | GGUF 格式定制化 |
| whisper 风格（流式） | ⭐⭐⭐ | ⭐⭐ | 内存受限场景 |
| 运行时量化 | ⭐⭐⭐⭐ | ⭐⭐⭐ | 单模型文件需求 |
| 混合精度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 极致性能优化 |

**最佳实践**：
1. **首选**：使用 llama.cpp `llama-quantize` 工具预量化
2. **如需定制**：参考 SenseVoice.cpp 实现（GGUF 格式相同）
3. **显存优化**：使用 Q4_K 或 Q5_K 平衡精度和大小
