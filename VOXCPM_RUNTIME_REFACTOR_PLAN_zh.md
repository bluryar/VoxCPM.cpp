# VoxCPM.cpp Runtime 重构总计划

本文档用于约束 VoxCPM.cpp 后续 Torch -> GGML 迁移和 runtime 重构的主方向，避免实现重新滑回“模块各自 materialize host vector、热路径反复 Host/Device 往返、权重/状态/计算图生命周期混乱”的旧路径。

这不是“另开一个全新仓库从零开发”的计划，而是“在当前仓库内并行落一套新 runtime 骨架，并逐步替换 legacy 路径”的计划。

## 1. 目标

本轮重构的最终目标只有四条：

1. 把 VoxCPM.cpp 收敛成共享权重池 + 持久状态对象 + 图缓存 + backend-aware 前向骨架的 runtime。
2. 明显降低峰值内存占用，让常驻内存更接近“模型权重 + 必要 KV/State/Output + 有界 compute arena”。
3. 明显降低 decode 热路径的 Host/Device 通信量，尽量把热路径收敛到 backend-resident tensor/state。
4. 让后续 CPU / CUDA / Vulkan / Metal 等扩展建立在同一套抽象边界上，而不是每个后端各写一条前向链。

## 2. 结论与仓库策略

结论：不再新建一个新的 VoxCPM.cpp 仓库。

原因：

- 当前 `VoxCPM.cpp` 已经是独立仓库，另开仓库并不能解决当前核心问题。
- 现有仓库已经有可复用资产：测试、examples、benchmark、WASM、backend 接线、共享权重池实现。
- 真正要重建的是 runtime 架构，不是工程外壳。
- 同仓双轨更利于 trace 对拍、行为回归、性能回归和逐模块替换。

因此采用下面的策略：

1. 保留当前可运行实现作为 `legacy` 参考路径。
2. 在当前仓库内新增更成熟的 runtime 分层。
3. 用模块级 trace 和测试逐块迁移，不做一次性大爆炸替换。
4. 等 `prefill/decode_step/AudioVAE` 新链路闭环并对齐后，再切默认入口。

## 3. 不可退让的五条红线

1. 不允许每个模块各自重新加载 GGUF，整个模型只能有一份共享 `WeightStore`。
2. 不允许把 `KV / persistent state / output` 偷放进 compute allocator 或 graph 临时输出里长期借用。
3. 不允许把模块边界默认实现成 `tensor_get -> std::vector<float> -> tensor_set`。
4. 不允许在布局契约不清楚时直接翻译算子，必须先明确 shape / stride / contiguous / broadcast 契约。
5. 不允许为了“看起来更快”先做高风险优化，再回头补 contract、state 和 output 边界。

补充说明：

- 本项目默认采用 `ggml_context(no_alloc=true) + backend buffer` 的 runtime 设计，这是一种工程选择，不是对 GGML 本体能力的限制。
- `ggml_context` 在 GGML 里既可以只存 metadata，也可以直接持有 tensor data；VoxCPM 这里选择前者，是为了更清晰地分离 `Weights / KV / State / Output / Compute` 的生命周期。
- 对 `mul_mat`、attention、conv、reshape/view/permute` 这类关键算子，不允许只写“通用 shape 规则”；必须把算子级输入输出布局、隐式转置、contiguous 前提和结果视角单独写进 contract。

## 4. 目标架构

目标 runtime 至少拆成下面七层：

1. `Contract`
2. `WeightStore / Loader`
3. `Backend / Memory`
4. `Output Buffer`
5. `Persistent State`
6. `Graph Cache / Scheduler`
7. `Runtime API`

推荐目录落点：

```text
include/voxcpm/
  contract.h
  weight-store.h
  backend.h
  context.h
  output.h
  state.h
  graph-cache.h
  runtime.h
  minicpm.h
  localenc.h
  locdit.h
  unified_cfm.h
  fsq.h
  components.h
  audio-vae.h

src/
  weight-store.cpp
  backend.cpp
  context.cpp
  output.cpp
  state.cpp
  graph-cache.cpp
  runtime.cpp
  minicpm.cpp
  localenc.cpp
  locdit.cpp
  unified_cfm.cpp
  fsq.cpp
  components.cpp
  audio-vae.cpp
```

### 4.1 数据边界

必须显式区分五类 buffer：

```cpp
enum class BufferUsage {
    Weights,
    KVCache,
    State,
    Output,
    Compute,
};
```

设计约束：

- `Weights`：只读持久，统一由 `WeightStore` 管理。
- `KVCache`：跨 step 持久可写，不进入 compute arena。
- `State`：`lm_hidden / residual_hidden / prefix_patch / cross-state` 等跨图持久对象。
- `Output`：最终用户输出、跨阶段稳定可读结果、inspectable outputs。
- `Compute`：单图生命周期临时结果。

补充说明：

- `ggml_context` 在新 runtime 中默认只承担 metadata owner 的角色，但这属于项目内约定，不应被表述为 GGML 的唯一工作模式。
- 简单 demo 或临时实验允许使用 context 内部分配数据；正式 runtime 默认不这样做。

### 4.2 热路径 API 边界

新 runtime 的默认边界应是：

- 模块间优先传 `ggml_tensor *`、view、persistent state handle。
- 只有最终对外输出或显式 fallback/staging 结果才 materialize 到 host。
- `decode_step()` 默认返回 output pool view，而不是中间 `std::vector<float>`。

### 4.3 持久状态

`DecodeState` 必须显式持有：

- `base_kv`
- `residual_kv`
- `lm_hidden`
- `residual_hidden`
- `prefix_patch`
- 必要的 graph cache handles

这些对象必须拥有清晰的 owner、context 和 buffer，不能借 compute arena 冒充长期状态。

## 5. 迁移顺序

### 阶段一：先搭骨架

交付物：

1. 固化 GGUF contract。
2. 补齐 `OutputBuffer`。
3. 补齐 `PersistentState`。
4. 补齐 `GraphCache`。
5. 把 backend buffer usage 扩展到 `Weights/KVCache/State/Output/Compute`。
6. 为后续 runtime 新入口预留 `runtime.h` / `runtime.cpp`。

阶段完成标准：

- 不改变模型数值路径。
- 新骨架可编译。
- `WeightStore` 仍保持单份共享加载。
- 新增测试可以验证 state/output 生命周期独立于 compute graph。

### 阶段二：先迁简单模块

顺序：

1. `Embedding`
2. `LinearProjection`
3. `StopPredictor`
4. `FSQ`

阶段完成标准：

- 模块各自有 trace 对拍。
- 新实现不新增热路径 host round-trip。
- 新旧实现可并存，调用方可切换。

### 阶段三：迁移 backbone

顺序：

1. `MiniCPMModel`
2. `LocEnc`
3. `LocDiT`

阶段完成标准：

- `forward()` 与 `forward_step()` 有明确 graph cache key。
- `LocEnc.forward_patch()` 与 `forward_sequence()` 的布局契约固定。
- KV cache、position、mask 语义通过 trace/测试验证。

### 阶段四：迁移生成链

顺序：

1. `UnifiedCFM`
2. `prefill`
3. `decode_step`
4. `AudioVAE`

阶段完成标准：

- `prefill` 会显式 capture `base_kv / residual_kv / lm_hidden / residual_hidden / prefix_patch`。
- `decode_step` 默认以 backend-resident state 为输入输出边界。
- `AudioVAE.decode` 只在最终导出阶段回 host。

### 阶段五：最后做优化

只在前四阶段闭环后推进：

1. batched `LocEnc`
2. 更彻底的 graph reuse
3. 消除残余 host 往返
4. scheduler / offload
5. fused projection / backend-specific repack

## 6. 每阶段必须跟踪的指标

### 6.1 内存指标

必须持续记录：

- 模型加载后常驻 RSS
- `prefill` 峰值 RSS
- 多步 `decode` 过程的峰值 RSS
- 多步 `decode` 过程的 steady-state RSS 漂移

“合理”的最低标准：

- 不因模块重构重新引入多份权重 buffer。
- steady-state `decode` 不应随着 step 数持续线性增长。
- 新实现的 RSS 不得无说明地高于当前共享权重路径基线。
- 所有新增持久内存都必须能解释为 `KV / State / Output` 的必要组成，而不是 graph/host 临时副本泄漏。

### 6.2 Host / Device 通信指标

必须持续记录：

- `host_to_device_bytes`
- `device_to_host_bytes`
- `device_to_device_bytes`
- 每个 decode 子阶段的 transfer delta

“合理”的最低标准：

- 不新增热路径 `tensor_get -> std::vector<float> -> tensor_set`。
- 模块边界不允许用 host vector 传递大 hidden / patch tensor，除非明确标记为过渡 fallback。
- decode 热路径里允许回 host 的默认对象只能是：
  - 最终用户可见输出
  - 显式 host staging / cross-state fallback
  - 少量标量控制参数或调试结果
- 所有优化提交都应该说明 transfer 是下降、持平还是上升；若上升，必须解释原因和后续移除计划。

### 6.3 建议的阶段性目标

以当前可复现 benchmark/log 为 baseline：

1. 阶段一到阶段三：不得增加热路径 transfer bytes。
2. 阶段四完成后：应开始消灭 `decode` 主链上大张量的 host materialization。
3. 阶段五：在相同模型、相同后端、相同输入下，目标把 `decode` 热路径 H2D + D2H 总量较当前 baseline 明显下降，优先追求 30% 到 50% 的降幅。

说明：这里的具体比例是工程目标，不是硬性数值契约。若后端实现、算子支持或 staging 需求限制短期收益，必须在变更说明中写清楚。

## 7. 验证方法

不要一开始就跑整模型，按下面顺序验证：

1. `Embedding`
2. `LinearProjection`
3. `FSQ`
4. `StopPredictor`
5. `LocalEncoder.forward_patch`
6. `LocalEncoder.forward_sequence`
7. `MiniCPM.forward`
8. `MiniCPM.forward_step`
9. `LocDiT.forward`
10. `UnifiedCFM.forward`
11. `prefill`
12. `decode_step`
13. `AudioVAE.decode`
14. 全链路 TTS

每个模块至少对拍：

- 输入张量
- 输出张量
- 中间关键节点

优先排查顺序：

1. 导出后的真实 tensor shape
2. GGML 侧 layout/stride/contiguous 前提，以及关键算子的 operator-specific 语义
3. broadcast 对齐
4. `reshape/view/permute/concat` 是否漏步，是否遗漏隐式转置或结果转置视角
5. 是否误带入训练态逻辑
6. 最后才判断是否缺算子

## 8. 当前仓库的执行策略

为了避免在旧 runtime 上持续打补丁导致边界越来越糊，后续默认执行策略如下：

1. 新架构优先新增文件和新入口，不直接把全部新逻辑继续堆进 `src/voxcpm.cpp`。
2. 若必须修改旧 runtime，只做兼容层、桥接层或 bug fix，不再把它视为最终形态。
3. 所有涉及 `src/voxcpm.cpp`、`include/voxcpm/voxcpm.h`、`include/voxcpm/backend.h` 的改动，都必须检查是否在扩大 host vector 边界。
4. 所有新增优化都必须能解释为减少以下之一：
   - 内存峰值
   - Host/Device 往返
   - graph 重建次数
   - 无意义复制

## 9. 当前已知热点与优先收敛对象

当前代码里最值得优先收敛的是：

- `src/voxcpm.cpp`
  - 仍有大量 `tensor_set/tensor_get` + `std::vector<float>` 热路径串联
- `include/voxcpm/voxcpm.h`
  - `DecodeState` 仍以 host-side vectors 为主
- `include/voxcpm/backend.h`
  - `BufferUsage` 还未完全扩展到目标形态

这意味着下一步最值得做的不是“继续局部提速”，而是：

1. 先补 `state/output/graph-cache/runtime` 骨架。
2. 再把 decode 热路径改造成显式持久 state + output pool。
3. 最后才继续追求 offload 和更深的调度优化。

## 10. 当前任务进度

维护规则：

- 每次推进 runtime 重构任务后，都要更新本节。
- 默认只更新状态、勾选项、里程碑和必要的单行备注。
- 不允许无限增加描述性文字；只有在目标、边界、顺序或验收标准发生变化时，才修改前文文字说明。

当前状态总览：

| 项目 | 状态 | 备注 |
| --- | --- | --- |
| 阶段一：runtime skeleton | 已完成 | `state/output/graph-cache/runtime` 已落地，`BufferUsage` 五分类已补齐 |
| 阶段二：简单模块迁移 | 进行中 | `StopPredictor` 已部分进入新边界，其他模块尚未系统性切换 |
| 阶段三：backbone 迁移 | 未完成 | 已有 graph/state 基础，但未形成模块级迁移闭环 |
| 阶段四：生成链迁移 | 进行中 | `prefill/decode/clone` 已进入 persistent-first 主路径 |
| 阶段五：优化 | 未开始 | 等前四阶段进一步闭环后推进 |

当前阶段性里程碑：

- 已达成：`prefill -> decode -> clone` 已能以 `persistent_state/output_pool` 为主 owner 运行。
- 已达成：`VOXCPM_LAZY_HOST_STATE` 已有链式 decode 与 lazy prefill 测试保护。
- 已达成：`output_pool.latent_seq` 已开始承载 prefix/decode patch timeline。

当前交付清单：

- [x] `BufferUsage` 扩展到 `Weights / KVCache / State / Output / Compute`
- [x] 新增 `state.h/.cpp`
- [x] 新增 `output.h/.cpp`
- [x] 新增 `graph-cache.h/.cpp`
- [x] 新增 `runtime.h/.cpp`
- [x] `VoxCPMDecodeState` 接入 `persistent_state` 与 `output_pool`
- [x] `decode()` 主链优先使用 backend-resident state/output
- [x] `prefill()` 支持 persistent-first + lazy host shadow
- [x] `benchmark_clone_state()` 支持 persistent-only state
- [x] `output_pool.latent_seq` 接入 prefill/decode/clone
- [x] `audio_frame_count` 接入 decode state，`latent_seq` 改为按音频帧时间线维护
- [x] `server_common` 已开始优先消费 `output_pool.latent_seq`
- [x] `server_common` 已把 `generated_steps / stream_recent_frames` 降级为 fallback 数据源
- [x] `server_common` 已通过统一 helper 消费 `output_pool` 导出的 AudioVAE latent
- [x] `server_common` 的 `output_pool -> AudioVAE.decode` 主路径已改为 backend-resident latent view，不再先导出 host latent
- [x] `output_pool` 已显式承载 AudioVAE latent view 的布局契约，`server_common` 不再自行实现这段变换
- [x] synth 主路径已补回归测试，覆盖 `load -> encode_prompt_audio -> synthesize`
- [x] `server_common` 在 `output_pool` 主路径下已调用 `decode(..., export_patch_to_host=false)`，避免每步多余 patch d2h
- [x] 已有 transfer-stats 测试证明 `export_patch_to_host=false` 会降低 decode 的 d2h bytes
- [x] `server_common` 在 `output_pool` 主路径下已跳过 stop logits 发布到 output pool，且有 transfer-stats 测试证明 d2d bytes 下降
- [x] `server_common` 在 `output_pool` 主路径下已跳过 `patch_output` 发布到 output pool，且有 transfer-stats 测试证明 d2d bytes 下降
- [x] `decode` 已支持 `trust_persistent_state`，`server_common` 主路径已跳过冗余 `host shadow -> persistent` 回灌，且有 transfer-stats 测试证明 h2d bytes 下降
- [x] `encode_prompt_audio` 已将 latent 到 `prompt_feat` 的重排前移到图内，去掉一块同尺寸 host 中间副本
- [x] `prefill` 已收缩两处大 host 临时副本（`combined_embed` / `residual_inputs`）
- [x] `prefill` 尾部 patch 发布已优先走 `persistent_state -> output_pool` 的 device-to-device 路径，并减少重复 `prefix_feat_cond` 拷贝
- [x] `prefill` 中 `feat_embed / combined_embed / enc_outputs / residual_inputs` 已缩短生命周期，不再拖到尾部状态发布阶段
- [x] `prefill` 的 lazy persistent path 已直接写入 `persistent_state`，不再先填充 host shadow 再同步后清空
- [x] `prefill` 写入 prompt `latent_seq` 已支持连续音频段批量 h2d，减少逐帧小传输
- [x] `prefill` 的 lazy path 已跳过 `create_decode_state()` 的 host shadow 零初始化与初始 state h2d，同步新增 transfer-stats 测试验证 bootstrap h2d 下降
- [x] `persistent_state/output_pool` 初始化已切到 backend-side clear，`create_decode_state()` 不再为初始零状态执行 host bootstrap h2d，并有测试验证 `host_to_device_bytes == 0`
- [x] `benchmark_clone_state()` 在 persistent-only 源状态下已跳过 host shadow 预分配，不再留下无用 vector capacity
- [x] `benchmark_clone_state()` 已按 output validity 选择性复制 `patch_output/stop_logits/latent_seq`，未发布输出不再发生多余 d2d
- [x] `benchmark_clone_state()` 已改为仅复制 `latent_seq` 的活跃前缀，clone 的 d2d 开销会随 `audio_frame_count` 线性变化
- [x] `output_pool` 已支持按区间导出 `latent_seq`；fallback `export_audio_vae_latent_to_host()` 不再多拿 `frame_offset + frame_count` 的前缀数据
- [x] fallback `export_audio_vae_latent_to_host()` 已改为图内 `view + transpose + cont` 后直接 d2h，去掉 host `patch_major` 中间副本与重排循环
- [x] 未发布的 `patch_output/stop_logits` 现在会直接在 host 侧返回零值，不再为了导出零 buffer 做无意义 d2h
- [x] 公共 `create_decode_state()` 现在默认返回不带 host shadow 的 fresh state，三块 host vectors 不再预分配
- [x] eager `prefill()` 也已去掉三块零值 host shadow 的预分配，改为只在生成真实结果时按需填充
- [x] `benchmark_clone_state()` 对完整 host shadow 也已统一改为“空 fresh state + 按需赋值”，不再走预热 host shadow 的内部创建路径
- [x] `decode` 已将当前 patch 写入 `latent_seq` 的路径切到 `output_pool.patch_output -> latent_seq` 的 device-to-device 发布
- [x] `decode` persistent path 已去掉两组无效 host hidden 分配，并移除冗余 `prefix_feat_cond` host shadow 写回
- [x] `prefill` residual 分支已改为只导出最后一列 hidden，不再为 `residual_hidden` 整段 materialize `residual_outputs`，并有 transfer-stats 测试验证 d2h 下降
- [x] fallback synth 收尾路径已直接拼装最终 AudioVAE latent，不再先构造整份 `decode_frames` host 中间副本再做重排
- [x] fallback 流式 synth 分支已改为在 helper 内按需构造 latent，不再在服务循环里长期保留 `stream_latent` host 中间缓冲
- [x] 非流式 fallback synth 请求已跳过无用的 `stream_recent_frames` prompt context 预拷贝，并预留流式窗口容量以减少重分配
- [x] fallback 流式 chunk 解码已切到图内 `transpose + cont`，不再先在 host 上构造一份 `stream_latent`
- [x] lazy persistent-only `prefill` 已直接把 base `enc_outputs` 的最后一列写入 `persistent_state`，不再额外 materialize 一份 `lm_hidden` 中转
- [x] lazy persistent-only `prefill` 已把 residual 末态改为 direct `tensor_copy` 到 `persistent_state`，不再经过 `residual_hidden` 的 host `d2h + h2d` 中转
- [x] lazy persistent-only `prefill` 已把最后一个 prompt patch 改为 `output_pool.latent_seq -> persistent_state.prefix_patch` 的 direct `d2d` 发布，不再重复执行 prefix patch `h2d`
- [x] 已补一组阶段二模块 benchmark/contract smoke tests，覆盖 `Embedding / enc_to_lm projection / FSQ / StopPredictor / LocEnc patch -> lm embed`
- [x] `StopPredictor` 已补 direct persistent-state benchmark 路径与 h2d 收缩测试，开始具备独立模块边界护栏
- [x] `lm_to_dit projection` 已补 direct persistent-state benchmark 路径与 h2d 收缩测试，阶段二模块边界继续收敛
- [x] `res_to_dit projection` 已补 direct persistent-state benchmark 路径与 h2d 收缩测试，阶段二的 projection 边界开始成体系
- [x] `FSQ` 已补 direct persistent-state benchmark 路径与 h2d 收缩测试，阶段二简单模块的 direct-state 护栏继续补齐
- [x] `Embedding` 已补 direct backend-resident token-tensor benchmark 路径与 h2d 收缩测试，阶段二开始覆盖 backend-resident token 输入边界
- [x] `enc_to_lm projection` 已补 direct backend-resident feature-tensor benchmark 路径与 h2d 收缩测试，阶段二继续覆盖 backend-resident 模块输入边界
- [x] `enc_to_lm projection + FSQ` 已补 direct backend-resident 组合 benchmark 路径与 h2d 收缩测试，阶段二开始形成可组合模块边界
- [x] `LocEnc sequence -> enc_to_lm projection` 已补 direct backend-resident 组合 benchmark 路径与 h2d 收缩测试，阶段二组合边界继续向 prefill 真正模块顺序推进
- [x] `LocEnc sequence -> enc_to_lm projection -> FSQ` 已补 direct backend-resident 组合 benchmark 路径与 h2d 收缩测试，阶段二组合边界继续沿 prefill 主链前推
- [x] `Embedding + mask + LocEnc sequence -> enc_to_lm projection` 已补 direct backend-resident 组合 benchmark 路径与 h2d 收缩测试，阶段二开始覆盖更贴近 prefill 组装边界的组合入口
- [x] `Embedding + mask + LocEnc sequence -> enc_to_lm projection -> FSQ` 已补 direct backend-resident 组合 benchmark 路径与 h2d 收缩测试，阶段二进一步贴近 prefill 的 host-heavy 组装主链
- [x] `prefill` 前半段已接回 `Embedding + mask + LocEnc sequence -> enc_to_lm projection` 组合入口，并用图内 `masked FSQ blend` 替换原 host FSQ 混合循环
- [x] 已补 `prefill` 前半段主链替换的专门 transfer/regression 测试，锁定“旧 host 组装链 vs 新组合入口”结果一致，且减少一整块前半段 `d2h` staging
- [x] `prefill` 的 `base_lm.forward -> residual_inputs` 已接回融合路径，在同一条图里完成 `base_lm.forward + masked FSQ blend + residual add`，并有专门 transfer/regression 测试验证结果一致且 `h2d` staging 下降
- [x] eager `prefill` 末态发布已去掉尾部统一 `sync_host_state_to_persistent` 作为主路径，改为按结果点位 direct publish；现有 transfer 测试已更新为锁定 prefix patch 那段 eager `h2d` 已被吃掉
- [x] eager `prefill` 的 `residual_hidden` 已改成先 direct publish 到 `persistent_state`、再按需回填 host shadow；现有 eager/lazy transfer 测试已更新为锁定两条路径的 `h2d` 完全收敛
- [x] eager `prefill` 的 `lm_hidden` 已改成 direct publish base last hidden 到 `persistent_state`、再按需回填 host shadow；现有 eager/lazy transfer 测试已更新为锁定额外 `d2h` 至少覆盖 `lm_hidden + residual_hidden`
- [x] `prefill` 已把 `combined_embed` 从真实主链里收成 backend-resident 中转，不再先 `d2h` 到 host 再 `h2d` 回 `base_lm` 后半段；新增专门 transfer/regression 测试锁定 `combined_embed` 级别的双向传输收缩
- [x] `prefill` 的 persistent-state 主路径已把 `residual_inputs -> residual_lm` 收成 backend-resident 直传，不再先整段 materialize `residual_inputs` 到 host 再重新 `h2d`；新增专门 transfer/regression 测试锁定 `residual_inputs` 级别的双向传输收缩
- [x] eager `prefill` 已支持通过显式开关 `VOXCPM_PREFILL_LAZY_PREFIX_SHADOW` 跳过 `prefix_feat_cond` host shadow，改为仅保留 `persistent_state/output_pool` 里的 prefix patch；新增回归测试锁定默认 eager 行为不变，开启后链式 decode 仍与默认路径对齐
- [x] eager `decode` 已支持通过显式开关 `VOXCPM_DECODE_LAZY_PREFIX_SHADOW` 跳过 `prefix_feat_cond` host shadow，改为仅保留 `persistent_state/output_pool` 里的 prefix patch；新增回归测试锁定开启后 decode 结果不变，且减少一整块 patch 级别的 `d2h`
- [x] `output_pool` 已补 backend-resident patch-range -> `latent_seq` 的发布 primitive，并新增 d2d 测试；后续若上游可提供 backend-resident prompt patch 段，可直接替换 `prompt timeline` 当前的 host `h2d`
- [x] 已补 backend-resident prompt patch-range 的 benchmark/helper，`prefill` prompt timeline 尾部状态发布已收敛到复用 `latent_seq` 的统一 finalize helper，并有专门 transfer/regression 测试锁定“backend prompt 输入”相对 host 路径减少整段 prompt-range `h2d`
- [x] `prefill` 已新增显式 backend-resident prompt patch 输入入口，并已接入真实主链；新增专门 transfer/regression 测试锁定“主链 prefill + backend prompt 输入”相对默认 host 路径减少整段 prompt-range `h2d`
- [x] `prefill` 已新增显式 backend-resident feature 输入入口，并已接入真实主链；新增专门 transfer/regression 测试锁定“主链 prefill + backend feature 输入”相对默认 host 路径减少整段 `feat` 与 prompt-range 级别的 `h2d`
- [x] `prefill` 已新增显式 backend-resident `token ids / text_mask / feat / feat_mask` 完整输入入口，并已接入真实主链；新增专门 transfer/regression 测试锁定相对 `feature tensor` 路径进一步减少 `token ids + text_mask + feat_mask` 级别的 `h2d`
- [x] `prefill` 完整输入张量路径已把 `text_mask` 完全留在 backend，`feat_mask` host 镜像已收缩为 prompt span 控制专用；新增测试锁定完整输入路径额外的 `d2h` 不超过一份 `feat_mask`
- [x] `prefill` 已新增显式 backend-resident 完整输入 + `prompt positions` 入口，并已接入真实主链；新增专门 transfer/regression 测试锁定相对 `prefill_with_input_tensors(...)` 路径减少一整份 `feat_mask` 级别的 `d2h`
- [x] `prefill_with_input_tensors(...)` 已收敛为围绕显式 `prompt positions` 主实现的兼容包装；后续 backend-resident `prefill` 主入口可优先围绕 `prefill_with_input_tensors_and_prompt_positions(...)` 继续演进
- [x] 已新增 typed `VoxCPMPrefillTensorInputs` 模块入口，`prefill_from_tensor_inputs(...)` 作为更正式的 backend-resident `prefill` 主入口已落地；原显式 `prompt positions` 函数已收敛为其便利包装，并有专门测试锁定两条入口结果一致
- [x] `prefill_from_tensor_inputs(...)` 现已同时支持“显式 `prompt positions`”与“从 `feat_mask` 派生 `prompt positions`”两种模式；`prefill_with_input_tensors(...)` 已进一步收敛为 typed 模块入口的兼容包装，并有专门测试锁定与模块入口结果一致
- [x] 已修复 `VoxCPMOutputPool::initialize()` 的 metadata context 估算，`voxcpm_tts` CLI 不再在 `create_decode_state()/prefill` 阶段因 patch views 撑爆 `ggml_context` 而崩溃；已用真实 `voxcpm_tts` 命令验证可完成整条推理并产出 wav
- [x] `LocEnc patch` 已补 direct output-pool patch-view benchmark 路径与 h2d 收缩测试，阶段二开始覆盖 runtime-owned patch 输入边界
- [x] `LocEnc patch -> lm embed` 已补 direct output-pool patch-view benchmark 路径与 h2d 收缩测试，阶段二开始覆盖 runtime-owned patch 输入边界
- [x] 已补 `VoxCPM2` GGUF contract 兼容桥接：`kv_channels` 头维、`residual_lm_no_rope` 与 `AudioVAE out_sample_rate`
- [x] 已补 `VoxCPM2` `fusion_concat_proj` 导出/加载与 runtime residual bridge 接口，`LocDiT` 也已支持多 `mu token` 前缀语义
- [ ] `VoxCPM2` 真实 stop 数值对齐：当前已确认新 GGUF 可导出、可推理，但短句 smoke 仍会跑满 `max_len`，不能视为 stop 语义正确
- [ ] `Embedding / LinearProjection / FSQ / StopPredictor` 完整模块级迁移与 trace 对拍
- [ ] `MiniCPM / LocEnc / LocDiT` 模块级迁移闭环
- [ ] `AudioVAE.decode` 切到最终导出前不回 host
- [ ] `prefill` 内部重 host-vector 编排继续收缩
- [ ] `decode` 入口与下游消费方进一步减少 host staging

待处理工作项：

- `prefill` 主线继续收缩
  - [ ] `base_lm.forward -> FSQ -> text/feat mask 混合 -> residual 输入` 继续向图内/直接发布边界收缩，减少整段 `enc_outputs` 的 d2h
  - [ ] `feat_embed` 从 host 中间量继续推进到更 backend-resident 的输入/混合边界
  - [ ] `combined_embed` 不再默认走 host 侧组装，继续收缩 `Embedding + mask + feat` 的 host 编排
  - [ ] `FSQ` 从“整段 host 输入/输出 helper”继续推进到图内或 backend-resident 主路径
  - [ ] eager `prefill` 末态发布继续从 `host shadow -> sync_host_state_to_persistent` 向 direct state publish 收口
  - [ ] `prefill` 继续从“backend-resident 完整输入入口”推进到更完整的全链 backend-resident 边界，评估是否将 prompt span 控制统一收敛到显式 `prompt positions` / backend-resident 元数据边界，并继续收缩 fallback staging

- 阶段二简单模块边界
  - [ ] `Embedding` 补齐 backend-resident 边界、shape/layout contract、模块级验证
  - [ ] `LinearProjection` 补齐 `mul_mat` 布局语义、backend-resident 输出路径、模块级验证
  - [ ] `FSQ` 从 runtime helper 提升为正式模块边界，并补齐独立验证
  - [ ] `StopPredictor` 从“部分接入新边界”推进到“完整模块迁移 + 独立验证”
  - [ ] `LocEnc patch -> lm embed` 收敛到更清晰的新 runtime 边界
  - [ ] 为阶段二模块补齐 benchmark/trace/transfer 回归检查
  - [ ] 把阶段二模块从“helper 集合”推进成“新 runtime skeleton 可组合调用的模块入口”

最近更新：

- `2026-04-02`: 阶段一完成；阶段四已进入 persistent-first 主路径；latent sequence 开始进入 output lifecycle。
- `2026-04-02`: `audio_frame_count` 已进入 decode state；`server_common` 已可优先从 `output_pool.latent_seq` 导出 AudioVAE latent。
- `2026-04-02`: `server_common` 默认不再把 `generated_steps` 作为主时间线，改为优先使用 `output_pool.latent_seq`，host vectors 仅保留 fallback。
- `2026-04-02`: `server_common` 已统一通过 helper 消费 `output_pool` 导出的 AudioVAE latent，后续替换底层 staging 只需改一处。
- `2026-04-02`: `server_common` 的 `output_pool -> AudioVAE.decode` 主路径已改成图内 `view + transpose + cont`，避免在最终波形导出前先回 host latent。
- `2026-04-02`: `output_pool` 已新增 AudioVAE latent view helper，并有测试锁定其布局与既有 host 导出语义一致。
- `2026-04-02`: synth 主路径已新增回归测试，实际覆盖 `load -> encode_prompt_audio -> synthesize`，为后续继续收缩 fallback host 路线提供基线。
- `2026-04-02`: `decode()` 已支持跳过 host patch 导出；`server_common` 在 `output_pool` 主路径下已使用该模式，去掉了 synth 主链里每步一次多余 patch d2h。
- `2026-04-02`: 已新增 transfer-stats 测试，自动验证 `decode(..., export_patch_to_host=false)` 相比默认路径会减少 d2h bytes。
- `2026-04-02`: `decode()` 已支持跳过 stop logits 发布到 output pool；`server_common` 在主路径下已使用该模式，并有 transfer-stats 测试验证 d2d bytes 下降。
- `2026-04-02`: `decode()` 已支持跳过 `patch_output` 发布到 output pool；修正了相应的 patch 生命周期边界，并有 transfer-stats 测试验证 d2d bytes 下降。
- `2026-04-02`: `decode()` 已改用 `VoxCPMDecodeOptions`，并新增 `trust_persistent_state`；`server_common` 主路径已用该模式跳过冗余 `host shadow -> persistent` 同步，且有 transfer-stats 测试验证 h2d bytes 下降。
- `2026-04-02`: `encode_prompt_audio` 已将 AudioVAE latent 到 `prompt_feat` 的重排前移到图内，避免先 d2h 到 `encoded` 再做一次同尺寸 host 重排。
- `2026-04-02`: `prefill` 已减少两处大 host 中间副本的峰值占用，数值路径经现有测试保持稳定。
- `2026-04-03`: 已补 backend-resident prompt patch-range 的 benchmark/helper，`prefill` prompt timeline 现可通过统一 finalize helper 消费 backend-resident prompt patch 段，并有专门 transfer/regression 测试锁定 prompt-range 级别的 `h2d` 收缩。
- `2026-04-03`: `prefill` 已新增显式 backend-resident prompt patch 输入入口并接入真实主链；新增专门 transfer/regression 测试锁定真实 `prefill` 主路径在 backend prompt 输入下减少整段 prompt-range `h2d`。
- `2026-04-03`: `prefill` 已新增显式 backend-resident feature 输入入口并接入真实主链；新增专门 transfer/regression 测试锁定真实 `prefill` 主路径在 backend feature 输入下减少整段 `feat` 与 prompt-range 级别的 `h2d`。
- `2026-04-03`: `prefill` 已新增显式 backend-resident 完整输入入口（`token ids / text_mask / feat / feat_mask`）并接入真实主链；新增专门 transfer/regression 测试锁定相对 `feature tensor` 路径进一步减少 `token ids + text_mask + feat_mask` 级别的 `h2d`。
- `2026-04-03`: `prefill` 完整输入张量路径已把 `text_mask` 完全留在 backend，`feat_mask` host 镜像收缩为 prompt span 控制专用；新增测试锁定这条路径相对 feature-tensor 路径额外的 `d2h` 不超过一份 `feat_mask`。
- `2026-04-03`: `prefill` 已新增显式 `prompt positions` 入口；完整输入张量主链可在不镜像 `feat_mask` 的前提下完成 prompt timeline 发布，并有专门测试锁定相对 `prefill_with_input_tensors(...)` 路径减少一整份 `feat_mask` 级别的 `d2h`。
- `2026-04-03`: `prefill_with_input_tensors(...)` 已实现层收敛为围绕显式 `prompt positions` 主实现的兼容包装；backend-resident `prefill` 的推荐主入口已开始向 `prefill_with_input_tensors_and_prompt_positions(...)` 收口。
- `2026-04-03`: 已新增 typed `VoxCPMPrefillTensorInputs` 模块入口；`prefill_from_tensor_inputs(...)` 已作为更正式的 backend-resident `prefill` 主入口落地，`prefill_with_input_tensors_and_prompt_positions(...)` 现作为便利包装复用同一主实现，并有专门测试锁定两条入口结果一致。
- `2026-04-03`: `prefill_from_tensor_inputs(...)` 已进一步支持“显式 `prompt positions`”与“从 `feat_mask` 派生 `prompt positions`”两种模式；`prefill_with_input_tensors(...)` 已完全退化为 typed 模块入口的兼容包装，并有专门测试锁定与模块入口结果一致。
- `2026-04-03`: 已修复 `VoxCPMOutputPool::initialize()` 的 metadata context 固定大小问题；`voxcpm_tts` 现已用真实 CLI 命令验证通过 `Encoding prompt audio -> prefill -> decode -> AudioVAE decode` 全链并成功保存 wav。
- `2026-04-02`: `prefill` 尾部当前 patch 的发布已优先走 persistent-first 的 d2d 路径，并去掉了 prompt 音频扫描中的重复 `prefix_feat_cond` 覆盖。
- `2026-04-02`: `prefill` 里 `feat_embed / combined_embed / enc_outputs / residual_inputs` 已收进局部作用域，降低这些大 host 中间量跨越到尾部发布阶段的存活时间。
- `2026-04-03`: `prefill` 的 residual 分支已切到 last-hidden 导出路径，不再为了 `residual_hidden` 取整段 `residual_outputs`，并新增 transfer-stats 测试验证 d2h 收缩。
- `2026-04-03`: fallback synth 收尾路径已直接构造最终 AudioVAE latent，去掉 `decode_frames -> latent` 这段整尺寸 host 中间副本；现有 `test_server_common` 集成链保持通过。
- `2026-04-03`: `prefill` 的 persistent-state 主路径已把 `residual_inputs -> residual_lm` 收成 backend-resident 直传，并新增专门 transfer/regression 测试锁定至少一整块 `residual_inputs` 级别的 `d2h + h2d` 收缩。
- `2026-04-03`: eager `prefill` 已支持通过 `VOXCPM_PREFILL_LAZY_PREFIX_SHADOW` 显式跳过 `prefix_feat_cond` host shadow；新增回归测试锁定开启后 `persistent/output` 仍保持权威，且后续 decode 结果与默认 eager 路径一致。
- `2026-04-03`: eager `decode` 已支持通过 `VOXCPM_DECODE_LAZY_PREFIX_SHADOW` 显式跳过 `prefix_feat_cond` host shadow；新增回归测试锁定开启后 `persistent/output` 仍保持权威，且 `d2h` 至少减少一整块 patch。
- `2026-04-03`: `output_pool` 已支持将 backend-resident 连续 patch 段直接发布到 `latent_seq`，并新增 d2d 测试；这为后续把 `prompt timeline` 从 host `h2d` 迁到 backend-resident 输入边界补齐了关键 primitive。
- `2026-04-03`: fallback 流式 synth 分支已去掉循环内长期持有的 `stream_latent` 缓冲，改为在 helper 内按需构造 latent 后立即解码；`test_server_common` 保持通过。
- `2026-04-03`: 非流式 fallback synth 请求不再预先把 prompt context 拷进 `stream_recent_frames`；同时为流式窗口预留容量，减少无意义 host copy 与重分配。
- `2026-04-03`: 已新增隐藏的真实 CLI `decode` 诊断测试，确认 stop 回归不是偶发 badcase；当前已定位到真实样例的一步 `decode` 仍与分解路径存在确定性 `lm_hidden` 偏差，日常测试集保持绿色，后续优先继续钉 `decode` 主链而不是再扩包装层。
- `2026-04-03`: fallback 流式 chunk 解码已前移到图内 `transpose + cont`，去掉每次 chunk 解码前的 host latent 构造；`test_server_common` 保持通过。
- `2026-04-03`: lazy persistent-only `prefill` 已直接发布 base 末态到 `persistent_state`，去掉一份只用于中转的 `lm_hidden` host materialization；`test_runtime_skeleton` 与 `test_voxcpm` 保持通过。
- `2026-04-03`: lazy persistent-only `prefill` 已把 residual 末态切到 direct `tensor_copy` 发布；对应 transfer-stats 测试已更新为验证 lazy 路径相对 eager 同时减少 residual hidden 的 `d2h` 和 `h2d`。
- `2026-04-03`: lazy persistent-only `prefill` 已复用 `output_pool.latent_seq` 中最后一帧 patch，direct `d2d` 发布到 `persistent_state.prefix_patch`；transfer-stats 测试已同步更新为验证 lazy 路径相对 eager 额外减少 prefix patch 的 `h2d`。
- `2026-04-03`: 已补阶段二模块 benchmark/contract smoke tests，锁定 `Embedding / enc_to_lm projection / FSQ / StopPredictor / LocEnc patch -> lm embed` 的基础 shape 与 finite 行为，为后续模块边界收敛提供护栏。
- `2026-04-03`: `StopPredictor` 已新增 direct persistent-state benchmark helper，并有测试验证与 host 路径结果一致且 h2d 更少，阶段二模块边界开始从 smoke test 进入有收益证明的护栏阶段。
- `2026-04-03`: `lm_to_dit projection` 已新增 direct persistent-state benchmark helper，并有测试验证与 host 路径结果一致且 h2d 更少；阶段二里 `LinearProjection` 的模块边界也开始具备收益证明护栏。
- `2026-04-03`: `res_to_dit projection` 已新增 direct persistent-state benchmark helper，并有测试验证与 host 路径结果一致且 h2d 更少；阶段二里 projection 这组模块边界开始形成成体系的 direct-state 护栏。
- `2026-04-03`: `FSQ` 已新增 direct persistent-state benchmark helper，并有测试验证与 host 路径结果一致且 h2d 更少；阶段二简单模块的 direct-state 护栏继续补齐。
- `2026-04-03`: `Embedding` 已新增 direct backend-resident token-tensor benchmark helper，并有测试验证与 host 路径结果一致且 h2d 更少；阶段二开始覆盖 backend-resident token 输入边界。
- `2026-04-03`: `enc_to_lm projection` 已新增 direct backend-resident feature-tensor benchmark helper，并有测试验证与 host 路径结果一致且 h2d 更少；阶段二继续覆盖 backend-resident 模块输入边界。
- `2026-04-03`: `enc_to_lm projection + FSQ` 已新增 direct backend-resident 组合 benchmark helper，并有测试验证与 host 组合路径结果一致且 h2d 更少；阶段二开始形成可组合模块边界。
- `2026-04-03`: `LocEnc sequence -> enc_to_lm projection` 已新增 direct backend-resident 组合 benchmark helper，并有测试验证与 host 组合路径结果一致且 h2d 更少；阶段二组合边界继续向 prefill 的真实模块顺序推进。
- `2026-04-03`: `LocEnc sequence -> enc_to_lm projection -> FSQ` 已新增 direct backend-resident 组合 benchmark helper，并有测试验证与 host 组合路径结果一致且 h2d 更少；阶段二组合边界继续沿 prefill 主链前推。
- `2026-04-03`: `Embedding + mask + LocEnc sequence -> enc_to_lm projection` 已新增 direct backend-resident 组合 benchmark helper，并有测试验证与 host 组合路径结果一致且 h2d 更少；阶段二开始覆盖更贴近 prefill 组装边界的组合入口。
- `2026-04-03`: `Embedding + mask + LocEnc sequence -> enc_to_lm projection -> FSQ` 已新增 direct backend-resident 组合 benchmark helper，并有测试验证与 host 组合路径结果一致且 h2d 更少；阶段二进一步贴近 prefill 的 host-heavy 组装主链。
- `2026-04-03`: `prefill` 前半段主链替换已新增专门的 transfer/regression 测试，锁定“旧 host 组装链 vs 新组合入口”结果一致，并验证前半段 `d2h` staging 下降；`test_runtime_skeleton` 与 `test_voxcpm` 保持通过。
- `2026-04-03`: `prefill` 的 `base_lm.forward -> residual_inputs` 已接回融合路径，在同一条图里完成 `base_lm.forward + masked FSQ blend + residual add`；新增专门 transfer/regression 测试锁定与旧 host 链结果一致，并验证这段 `h2d` staging 下降。
- `2026-04-03`: eager `prefill` 末态发布已改为按结果点位 direct publish，不再把尾部统一 `sync_host_state_to_persistent` 作为主路径；现有 eager/lazy transfer 测试已更新为锁定 eager 相对 lazy 的额外 `h2d` 已从“residual + prefix”收缩到“主要剩 residual”。
- `2026-04-07`: 为 `VoxCPM2` 新 GGUF contract 补齐兼容桥接，`MiniCPM` 现可按权重/metadata 识别 `kv_channels` 与 `residual_lm_no_rope`，`AudioVAE`/CLI/service 也已区分输入 `sample_rate` 与输出 `out_sample_rate`。
- `2026-04-07`: 已补 `VoxCPM2` `fusion_concat_proj` 导出/加载、residual concat bridge 与 `LocDiT V2` 多 `mu token` 前缀；相关 runtime/regression 测试保持通过，但真实 `VoxCPM2` smoke 仍跑满 `max_len`，说明 stop 数值对齐尚未完成。
- `2026-04-07`: 已修复 `decode()` 主链里 `front_half.output_aux0` 被后续 cached graph 复用 compute arena 覆盖的问题；真实 CLI decode 单步诊断现已通过，`VoxCPM2` smoke 也从跑满 `max_len=190` 收敛到在 step `74` 触发 stop、输出约 `12.000s` 音频。
- `2026-04-07`: 已为 `AudioVAE` 接上 `VoxCPM2` decoder-side `sr_cond_model`（含 `sr_bin_boundaries` bucket 语义、GGUF 权重加载与 decode 输入 staging）；新增 `test_audio_vae` 专项回归锁定该桥接，真实 `VoxCPM2` smoke 进一步从 step `74` 收敛到 step `24`，输出约 `4.000s / 48kHz` 音频。后续仍需继续与上游 PyTorch 做同 latent / 同 prompt 的波形级对拍，确认“已从乱码恢复”为真正数值对齐而非仅 stop 改善。
- `2026-04-07`: 已补 `VoxCPM2 reference + prompt` 接口层：CLI/service 支持 `reference_wav_path` 等价装配、`retry_badcase` 参数、reference 特征持久化，并把 runtime prefill 的 prompt timeline 从“所有音频 span”收敛到“序列尾部 continuation 音频 span”；同时补上参考音频的首尾静音裁剪与区分 `reference/prompt` 编码日志。当前 reference/HiFi smoke 仍在 step `143` 左右停下、输出约 `23s`，说明 reference 模式的核心数值对齐尚未完成，下一步需要继续对拍上游 `prompt_cache/ref_continuation` 的 hidden/stop 语义。
- `2026-04-19`: OpenAI-compatible `voxcpm-server` 已新增 `--output-sample-rate`，并把 `wav/mp3/opus/pcm` 的最终输出统一改为先按该采样率重采样后再编码；README / README.cuda / README_zh 已补充 24 kHz PCM 的服务端注意事项。
- `2026-04-18`: service 长文本请求已补 seq-aware graph context 与 chunked prefill fallback；CUDA service decode budget 固定为 `<=256/128 step`, `257-512/96 step`, `>512/64 step`，并在 `README.cuda.md` 记录。
- `2026-04-03`: eager `prefill` 的 `residual_hidden` 已改成先 direct publish 到 `persistent_state`、再按需回填 host shadow；现有 eager/lazy transfer 测试已更新为锁定两条路径的 `h2d` 完全收敛，只剩 eager 为 host shadow 多付出的 `d2h`。
- `2026-04-03`: eager `prefill` 的 `lm_hidden` 已改成 direct publish base last hidden 到 `persistent_state`、再按需回填 host shadow；现有 eager/lazy transfer 测试已进一步更新为锁定 eager 相对 lazy 的额外 `d2h` 至少覆盖 `lm_hidden + residual_hidden`。
- `2026-04-03`: `prefill` 已把 `combined_embed` 从真实主链里收成 backend-resident 中转，避免先 `d2h` 到 host 再 `h2d` 回 `base_lm` 后半段；新增专门 transfer/regression 测试验证这一步至少减少一整块 `combined_embed` 级别的 `d2h + h2d`。
- `2026-04-03`: `LocEnc patch` 已新增 direct output-pool patch-view benchmark helper，并有测试验证与 host 路径结果一致且 h2d 更少；阶段二开始覆盖 runtime-owned patch 输入边界。
- `2026-04-03`: `LocEnc patch -> lm embed` 已新增 direct output-pool patch-view benchmark helper，并有测试验证与 host 路径结果一致且 h2d 更少；阶段二开始覆盖 runtime-owned patch 输入边界。
- `2026-04-02`: `prefill` 的 lazy persistent path 已跳过“host shadow -> persistent -> clear”这段往返，直接把末态写入 `persistent_state`。
- `2026-04-02`: `prefill` 已将 prompt 音频时间线写入 `latent_seq` 的路径改为按连续音频段批量 h2d，减少逐帧小块传输。
- `2026-04-02`: `prefill` 的 lazy path 已改为直接创建不带 host shadow 的 decode state，去掉初始化零向量及其初始 h2d，同步新增 transfer-stats 测试验证 h2d bytes 下降。
- `2026-04-02`: `persistent_state/output_pool` 初始化已改为 backend-side clear；`create_decode_state()` 不再把零值 host shadow 回灌到 persistent/output，新增测试验证 create 时 `host_to_device_bytes == 0`，同时 eager/lazy prefill 的 bootstrap h2d 已收敛。
- `2026-04-02`: `benchmark_clone_state()` 已按源状态决定是否初始化 host shadow；persistent-only clone 不再先分配再清空大 host vectors，并新增测试锁定 `capacity()==0`。
- `2026-04-02`: `output_pool` 已显式记录 `patch_output/stop_logits` 是否真正发布；`benchmark_clone_state()` 现在只复制有效输出与非空 `latent_seq`，并新增 transfer-stats 测试验证未发布输出不会再带来多余 d2d。
- `2026-04-02`: `benchmark_clone_state()` 复制 `latent_seq` 时已改为按活跃帧 patch-view 前缀逐帧 d2d，不再整块复制整条时间线；新增 transfer-stats 测试验证 d2d 会随 `audio_frame_count` 增长。
- `2026-04-02`: `output_pool` 已新增按区间导出 `latent_seq`；`export_audio_vae_latent_to_host()` 现在只做请求区间的 d2h，不再先抓整段前缀，并新增 transfer-stats 测试验证 d2h bytes 精确等于请求片段大小。
- `2026-04-02`: fallback `export_audio_vae_latent_to_host()` 已切到图内 `make_audio_vae_latent_view()` 路径，直接导出最终 latent layout；host `patch_major` 临时向量与重排循环已移除，现有布局/transfer 测试保持通过。
- `2026-04-03`: `output_pool` 在 `patch_output/stop_logits` 未发布时已改为直接返回 host 侧零值；fresh decode state 的未发布输出导出不再产生 d2h，新增测试验证 `device_to_host_bytes == 0`。
- `2026-04-03`: 公共 `create_decode_state()` 已改为默认不预分配 `lm_hidden/residual_hidden/prefix_feat_cond`；fresh state 现在直接以 persistent/output 为主，新增测试锁定三块 host vectors `empty + capacity()==0`。
- `2026-04-03`: eager `prefill()` 已停止通过内部 state 创建预分配三块零值 host shadow；`prefix_feat_cond` 改为仅在需要写最后 prompt patch 时按需分配，现有 prefill/decode/server 测试保持通过。
- `2026-04-03`: `benchmark_clone_state()` 已不再根据 host completeness 选择预热 host shadow 的创建路径；完整 host shadow clone 现在统一从空 fresh state 按需拷贝，新增测试锁定完整 host shadow 仍可正确保留。
- `2026-04-02`: `decode` 当前 patch 追加到 `latent_seq` 已切到 d2d 路径，去掉了这一段 `tensor_get -> host vector -> tensor_set` 的回写链。
- `2026-04-02`: `decode` 在 persistent path 下已不再分配无效 host hidden 向量，也不再写回会立刻被清空或覆盖的 `prefix_feat_cond` host shadow。
- `2026-04-17`: OpenAI-compatible TTS server 已完成 `response_format=mp3/opus` 的实际编码与 SSE `audio.delta` 同格式输出；同时在 `/v1/audio/speech` 请求边界加入 runtime/backend reset，修复同一 core/service 的重复请求 `SEGV`，相关 README / CMake / tests / smoke 已收口。
- `2026-04-20`: 流式 synth 的 chunk AudioVAE decode 现在会在继续下一步 decode 前清理 runtime/state cached graph handles，避免 compute arena 扩容后复用悬挂 tensor data；ASan 复现用例、CUDA service 集成测试与非 CLI runtime skeleton 已验证通过。

## 11. 参考文档

重构过程中，以下文档是主参考而不是可选阅读：

- `docs/torch_to_ggml_migration_guide_zh.md`
- `docs/voxcpm_torch_to_ggml_complete_refactor_cookbook_zh.md`
- `docs/voxcpm_shared_weight_store_refactor.md`
- `docs/voxcpm_decode_refactor_summary_zh.md`
- `docs/voxcpm_cpp_backend.md`

如果后续实现与这些文档发生偏离，必须优先更新计划和契约说明，再继续写代码。
