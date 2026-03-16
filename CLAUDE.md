# GGBond Project Memory

## 一、GGML 核心架构：两阶段模型

### 阶段1：元数据定义
- 定义张量形状、类型、操作关系
- 构建计算图，不分配实际数据存储
- Context 使用 `no_alloc=true`，只存元数据

### 阶段2：分配与执行
- 在后端设备上分配内存
- 填充数据、执行计算

### 优势
- 设备无关：同一份元数据可用于 CPU/CUDA/Metal
- 内存优化：后端可复用内存
- 延迟绑定：数据位置运行时决定

---

## 二、Context 详解

### 本质
`ggml_context` 是内存池，存储**张量元数据**（非数据本身）：
- 张量形状（ne0, ne1, ne2, ne3）
- 数据类型（FP32, FP16, INT8 等）
- 操作类型和依赖关系
- 计算图结构

### 源码结构
```c
struct ggml_context {
    size_t mem_size;           // 内存池总大小
    void * mem_buffer;         // 内存池起始地址
    bool   mem_buffer_owned;   // 是否拥有内存
    bool   no_alloc;           // 关键：true时不分配tensor数据
    int    n_objects;          // 已分配对象数量
    struct ggml_object * objects_begin;  // 对象链表头
    struct ggml_object * objects_end;    // 对象链表尾
};
```

### Context 大小计算
```
mem_size = n_tensors * ggml_tensor_overhead() + ggml_graph_overhead() + 安全余量
```
**重要：不包含张量数据大小！**

---

## 三、最佳实践

### 1. Context 创建
```cpp
struct ggml_init_params params = {
    .mem_size = n_tensors * ggml_tensor_overhead() + ggml_graph_overhead(),
    .mem_buffer = buffer_ptr,
    .no_alloc = true,  // 生产环境必须为 true
};
```

### 2. 标记输入输出张量
```cpp
ggml_set_input(input_tensor);   // 必须，scheduler 需要
ggml_set_output(output_tensor); // 必须，allocator 需要
```

### 3. Buffer 分离原则
| Buffer 类型 | 用途 | 标记 |
|------------|------|------|
| 权重 Buffer | 模型权重 | `GGML_BACKEND_BUFFER_USAGE_WEIGHTS` |
| KV Cache Buffer | 解码器状态 | 独立的 context 和 buffer |
| 计算 Buffer | 中间结果 | 由 `ggml_backend_sched` 管理 |

```cpp
// 标记权重 Buffer
ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
```

### 4. 使用 `ggml_backend_sched`（推荐）
```cpp
// 创建 scheduler
sched = ggml_backend_sched_new(backends.data(), nullptr, backends.size(),
                                MAX_NODES, false, true);

// Meta buffer 大小
meta.resize(ggml_tensor_overhead() * MAX_NODES + ggml_graph_overhead());

// 预分配
ggml_backend_sched_alloc_graph(sched, graph);
ggml_backend_sched_reset(sched);

// 执行循环
ggml_backend_sched_graph_compute(sched, graph);
ggml_backend_sched_reset(sched);
```

### 5. 图执行模式
```
Reset → Alloc → Compute → Reset
```

### 6. Backend 初始化（GPU 优先）
```cpp
// 1. 优先 GPU
ggml_backend_t backend_gpu = init_gpu_backend(params);
if (backend_gpu) result.push_back(backend_gpu);

// 2. 添加 CPU/ACCEL 后端
for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
    auto dev = ggml_backend_dev_get(i);
    if (dev_type == GGML_BACKEND_DEVICE_TYPE_ACCEL ||
        dev_type == GGML_BACKEND_DEVICE_TYPE_CPU) {
        result.push_back(ggml_backend_dev_init(dev, nullptr));
    }
}
```

---

## 四、VoxCPM 模型架构

### 推理流程
```
音频输入 → AudioVAE Encoder → 潜在表示(64-dim)
                ↓
          LocEnc (8层) → 局部特征编码
                ↓
          proj.enc_to_lm
                ↓
          BaseLM (24层) → 语言建模
                ↓
    ┌───────────┼───────────┐
    ↓           ↓           ↓
  FSQ      ResidualLM   proj.lm_to_dit
  量化器      8层残差           │
    │           │              │
    └───────────┴──────────────┘
                ↓
          LocDiT (8层) → CFM 生成
                ↓
          AudioVAE Decoder → 音频输出
```

### 模块参数
| 模块 | 层数 | 隐藏维度 | 注意力头 |
|------|------|----------|----------|
| LocEnc | 8 | 1024 | 16 |
| BaseLM | 24 | 4096 | 32 |
| ResidualLM | 8 | 4096 | 32 |
| LocDiT | 8 | 1024 | 16 |
| AudioVAE | - | 512 (latent=64) | - |

### 关键操作
- **LocEnc/LocDiT**: Local attention + 1D convolution
- **FSQ**: Finite Scalar Quantization (3 levels: [8,8,4])
- **CFM**: Conditional Flow Matching 用于生成

---

## 五、常见陷阱

### 1. Context 大小包含张量数据
**错误**！只包含元数据开销。

### 2. 忘记 `ggml_set_input/output`
Scheduler 无法正确工作。

### 3. 混用 Buffer
权重、KV Cache、计算 Buffer 必须分离。

### 4. 不使用 `no_alloc=true`
生产代码必须使用两阶段模型。

### 5. Metal 后端矩阵乘法
Metal 需要使用 `ggml_mul_mat_pad`：
```cpp
#if defined(GGML_USE_METAL)
#define ggml_mul_mat ggml_mul_mat_pad
#endif
```

### 6. 动态 Feature Buffer 重分配
每次推理销毁重建会导致内存碎片化和性能开销。**建议**：预分配最大 feature buffer。

---

## 六、SenseVoice.cpp 参考实现

完全符合 GGML 最佳实践，可作为参考：

### 最佳实践检查清单
- [x] `no_alloc=true` 所有 graph contexts
- [x] `ggml_set_input()` 所有输入张量
- [x] `ggml_set_output()` 所有输出张量
- [x] 权重 Buffer 独立 + WEIGHTS 用途标记
- [x] KV Cache Buffer 独立
- [x] 使用 `ggml_backend_sched`
- [x] 预分配模式
- [x] Context 大小仅元数据开销
- [x] 多后端支持（GPU优先）

---

参考文档:

1. 最佳实践：/home/orangepi/Codes/ggbond/VoxCPM.cpp/docs/GGML_BEST_PRACTICES.md
2. VoxCPM.cpp 架构设计：/home/orangepi/Codes/ggbond/VoxCPM.cpp/docs/logical-hatching-zebra.md