# VoxCPM.cpp 量化与 imatrix 实施经验总结

## 一、背景

这次工作的目标，已经从“验证 `quant_pre.md` 里的初版设想”扩展成了完整的工程落地：

- 在 `VoxCPM.cpp` 内实现离线 GGUF 量化
- 支持更低比特 K-Quant 和 IQ 系列
- 为需要 `imatrix` 的量化类型提供 VoxCPM 专用 calibration collector
- 让工具链具备可测试、可验证、可对比的使用方式

最终落地的主要内容包括：

- 量化接口：`include/voxcpm/quantize.h`
- 量化实现：`src/quantize.cpp`
- 量化 CLI：`examples/voxcpm_quantize.cpp`
- imatrix 接口：`include/voxcpm/imatrix.h`
- imatrix 实现：`src/imatrix.cpp`
- imatrix CLI：`examples/voxcpm_imatrix.cpp`
- runtime 采集挂点：`include/voxcpm/voxcpm.h`、`src/voxcpm.cpp`
- 测试：`tests/test_quantize.cpp`、`tests/test_imatrix.cpp`

---

## 二、最初方案里哪些成立，哪些需要修正

### 2.1 成立的部分

`quant_pre.md` 的核心方向是对的：

- VoxCPM 使用 GGUF，适合做离线量化再推理
- 主干计算路径大量依赖 `ggml_mul_mat()` / `ggml_get_rows()`
- 最自然的落地点不是运行时动态量化，而是生成量化 GGUF

### 2.2 需要修正的部分

最重要的修正是：

- “推理代码无需修改即可使用量化模型”并不完全成立

真实核查后发现：

- `MiniCPM`、`ResidualLM`、`LocEnc`、`LocDiT`、`FSQ`、`proj.*`、`stop.*` 基本都能直接吃量化权重
- `AudioVAE` 里有自定义 depthwise conv，直接把权重当 `float *` 读取

这意味着：

- `audio_vae.*` 不能直接量化成 `Q4_K/Q5_K/Q8_0/IQ*`
- v1 到当前版本都必须保留 `AudioVAE` 为浮点

这个判断不是“模型结构上看起来复杂”，而是被真实推理实现方式决定的。

---

## 三、最终采用的总体架构

### 3.1 离线量化，而不是运行时量化

最终选择离线量化，有三个直接原因：

- 运行时量化会拖慢首次加载
- 会额外增加内存占用
- GGUF 的 API 已经足够直接支持“读取 -> 改 tensor type/data -> 写回”

核心流程是：

1. `gguf_init_from_file(..., no_alloc=false)` 读入原模型
2. `gguf_init_empty()` 创建输出 context
3. `gguf_set_kv()` 复制 metadata
4. `gguf_add_tensor()` 建立输出 tensor 元数据
5. `gguf_set_tensor_type()` 改 tensor 类型
6. `gguf_set_tensor_data()` 回填新数据
7. `gguf_write_to_file()` 写回量化 GGUF

### 3.2 仓库内实现，而不是依赖外部 llama 工具

虽然 `llama.cpp` 已经有成熟的量化工具，但这里仍然在 `VoxCPM.cpp` 内实现，原因是：

- 张量命名规则必须按 VoxCPM 的真实 tensor 名写
- 必须显式跳过 `audio_vae.*`
- `imatrix` 的 calibration collector 需要挂到 VoxCPM 自己的 prefill/decode 路径
- 这条链路后续需要长期随仓库一起维护

---

## 四、真正决定实现复杂度的关键点

### 4.1 量化策略必须从“权重怎么被读取”反推

这一点比“哪些张量名字像权重”更关键。

当前永远保持浮点的张量包括：

- 所有 `audio_vae.*`
- 所有 1D tensor
- 所有 `*.bias`
- 所有 `*.alpha`
- 所有 norm 权重：`*_norm.weight`、`*.output_norm.weight`、`output_norm.weight`
- `locenc.special_token`

### 4.2 K-Quant / IQ 类型有 block size 和 shape 约束

实现中必须检查：

- 目标 `ggml_type` 是否支持该 row shape
- `tensor->ne[0]` 是否满足 block size 约束

否则不能强制改类型，只能保留浮点。

这也是最小 GGUF 测试早期暴露出来的关键问题之一。

### 4.3 F16 / Q8_0 / 低比特 / IQ 不是“只加 CLI 选项”

真正落地时，底层都要补齐：

- `ggml_ftype -> ggml_type` 映射
- `F32 -> F16` 转换
- `F32 -> Q8_0` 量化
- 低比特类型的混合精度分支
- `imatrix` 参与的量化权重路径

只有 CLI 没有底层转换逻辑是跑不通的。

---

## 五、当前支持的量化能力

### 5.1 基础量化类型

当前 `voxcpm_quantize` 已支持：

- `Q2_K`
- `Q3_K`
- `Q4_K`
- `Q5_K`
- `Q8_0`
- `F16`

### 5.2 IQ 系列

当前还支持：

- `IQ2_XXS`
- `IQ2_XS`
- `IQ2_S`
- `IQ3_XXS`
- `IQ3_S`
- `IQ1_S`
- `IQ1_M`
- `IQ4_NL`
- `IQ4_XS`

### 5.3 哪些 IQ 类型必须提供 `--imatrix`

按 vendored `ggml` 的要求，当前必须传 `--imatrix` 的类型是：

- `IQ2_XXS`
- `IQ2_XS`
- `IQ1_S`

其他 IQ 类型当前不强制，但如果有可用 `imatrix`，通常仍然更稳。

### 5.4 混合精度规则的实际经验

落地后确认：对 VoxCPM 更稳妥的不是“全量一刀切”，而是保守混合量化。

高敏感层通常需要保得更高一些：

- `token_embd.weight` 往往维持到 `Q8_0`
- `proj.*`、`fsq.*`、`stop.*`、`locdit/locenc` 关键投影层适合维持更高精度
- `attn_v.weight` 和部分首尾 `ffn_down.weight` 在更低比特时应上调一档

这套规则对 `Q4_K/Q5_K/Q3_K/Q2_K/IQ*` 都比“全量统一压缩”稳。

---

## 六、imatrix 的落地经验

### 6.1 imatrix 不是推理时依赖，而是离线校准产物

`imatrix.gguf` 的作用只发生在量化阶段：

- `voxcpm_quantize --imatrix ...` 会读取它
- 最终量化模型会写入追踪元数据
- 推理时 `voxcpm_tts` 不会再单独加载 `imatrix.gguf`

### 6.2 collector 要挂在真实运行图上

这次没有做“离线猜测重要性矩阵”，而是给 `VoxCPMRuntime` 增加了 collector 挂点：

- 每次 graph compute 成功后
- 遍历 graph 中的 `GGML_OP_MUL_MAT`
- 按权重 tensor 名累计输入激活平方和

这个设计参考了 `llama-imatrix`，但采集路径是 VoxCPM 自己的：

- `LocEnc`
- `MiniCPM`
- `ResidualLM`
- `LocDiT`
- `FSQ`
- 各投影层

### 6.3 AudioVAE 不在 imatrix 采集目标里

因为 `AudioVAE` 当前本来就不进入量化范围，所以 collector 没必要为它额外做复杂支持。

### 6.4 `imatrix` 读写需要兼容两种格式

当前实现支持：

- GGUF 格式 `imatrix`
- legacy 二进制格式

这样做的好处是：

- 新 collector 产物能直接喂给量化器
- 旧文件如果存在，也不会完全失去可用性

### 6.5 `--save-frequency` 和 `--show-statistics` 很有用

这两个能力是在真正使用 collector 时才发现很重要的：

- `--save-frequency`
  - 长跑任务可以周期性落快照
  - 出现中断时至少保住阶段性结果
  - 也方便比较不同 chunk 数量下的量化效果

- `--show-statistics`
  - 可以快速看 `entries/chunk_count/zero_count_entries`
  - 可以看 top tensors 的激活强度排序
  - 很适合做健康检查，避免拿明显不完整的 `imatrix` 去量化

---

## 七、TTS 校准数据的实际经验

### 7.1 不是“只需要文本”，而是“至少可以只用文本”

对 VoxCPM TTS，当前 collector 支持两种常见模式：

- 纯文本校准
- prompt-audio + prompt-text 条件校准

这意味着：

- 如果线上主要是纯文本 TTS，只给 `--text-file` 就够
- 如果线上主要是带参考音频的风格/说话人提示，就应该把 prompt 条件也带进 calibration

### 7.2 不建议用“模型自己合成的音频”反过来做主校准数据

这次实践里确认，一个常见误区是：

- 先让模型合成一批音频
- 再把这些“模型自己生成的音频 + 文本”拿来校准

这通常不是最佳选择，因为：

- 它不代表真实用户输入分布
- 容易把模型已有偏差又喂回校准流程
- 对激进低比特量化不够稳

更好的原则是：

- 校准数据尽量贴近真实部署时的输入条件分布

### 7.3 dataset 模式比单一全局 prompt 更实用

为适配真实 TTS 场景，这次又把 `voxcpm_imatrix` 升级成支持：

- `--text-file`
- `--dataset-file`

其中 `--dataset-file` 支持每行：

- `text`
- `text<TAB>prompt_text<TAB>prompt_audio`

这个设计的好处是：

- 同一轮校准中可以混合“纯文本样本”和“带参考音频样本”
- 不同样本可以有不同 prompt speaker
- 更接近真实业务流量

---

## 八、测试和验证经验

### 8.1 先做最小 GGUF 集成测试

这是量化工作里最值钱的一步。

最小 GGUF 测试可以快速验证：

- tensor 名到目标类型的映射是否正确
- `general.file_type` 是否写回正确
- `general.quantization_version` 是否写回正确
- 非量化 tensor 字节是否保持不变
- `VoxCPMWeightStore` 是否能加载 mixed tensor types

### 8.2 `test_quantize` 要覆盖的不只是基础类型

这次 `test_quantize` 最终覆盖了：

- `Q4_K/Q5_K/Q8_0/F16`
- `Q3_K/Q2_K`
- `IQ4_NL`
- `IQ2_XXS`
- `imatrix` 元数据写回
- 缺失 `imatrix` 时的失败分支

这点很重要，因为如果只测 `Q4_K`，很容易让低比特或 IQ 分支在后续修改中悄悄退化。

### 8.3 `test_imatrix` 需要覆盖 collector 和 dataset

这次新增的 `test_imatrix` 覆盖了两类核心行为：

- collector 的 `observe -> save -> load -> print_statistics`
- calibration 数据加载器的
  - 纯文本文件
  - TSV dataset
  - 非法行报错

这使得 `--show-statistics` 和 `--dataset-file` 不再只是“手工 smoke 通过”，而是有单测兜底。

### 8.4 真实模型 smoke test 仍然不可替代

单测只能证明“逻辑正确”，不能完全证明“真实模型链路可用”。

所以最后仍然要跑：

- 真实模型 `dry-run`
- 真实模型实际量化
- `voxcpm_tts` 推理 smoke test
- `voxcpm_imatrix --dataset-file` / `--show-statistics` smoke test

这一层验证已经证明：

- 量化后的模型能被当前推理代码加载
- `imatrix` collector 产物能喂给量化器
- dataset 模式和 snapshot 模式都能真正工作

---

## 九、当前确认的已知限制

### 9.1 AudioVAE 仍不能直接量化

根因仍然是自定义卷积直接按浮点读取权重。

如果后续要继续压缩，有两条路线：

- 加载时把量化后的 `audio_vae.*` 反量化回 `F32`
- 改写 `AudioVAE` 卷积路径，让它直接支持量化权重

### 9.2 目前的 dataset 格式仍是轻量版本

当前 `--dataset-file` 只支持：

- `text`
- `text<TAB>prompt_text<TAB>prompt_audio`

还不支持更复杂的 schema，例如：

- 单独的说话人 ID
- 情感标签
- 每样本自定义 decode 参数
- JSONL/CSV 多列格式

### 9.3 还没有自动化音质回归指标

目前自动验证主要覆盖：

- 文件生成
- 类型映射
- metadata
- mixed tensor 加载
- 推理 smoke

还没有接入自动音质指标，例如：

- mel/L2 差异
- 频谱误差
- 主观打分代理指标

---

## 十、后续建议

建议后续按下面顺序推进：

1. 增加脚本化 benchmark
   - 固定文本
   - 固定 prompt
   - 固定线程数
   - 固定推理步数
   - 自动输出模型大小、RTF、主观备注

2. 建立量化档位对比基线
   - FP32
   - F16
   - Q8_0
   - Q5_K
   - Q4_K
   - IQ4_NL
   - IQ3_S
   - IQ2_*（有 `imatrix`）

3. 继续增强 calibration 数据集能力
   - 支持更丰富的 dataset schema
   - 支持多 speaker / 多 prompt 的更系统采样

4. 评估 AudioVAE 压缩路线
   - 优先尝试“加载后反量化到 F32”
   - 如果收益不够，再考虑改卷积实现

---

## 十一、结论

这次实施最终确认了三件最重要的事：

1. VoxCPM.cpp 可以稳定落地 GGUF 离线量化，但前提不是“全模型统一压缩”，而是“根据真实推理路径做保守混合量化”。
2. 更低比特和 IQ 系列在工程上是可用的，但要配合 `imatrix`、测试和真实 smoke 验证，不能只停留在类型枚举层面。
3. 对 TTS 来说，校准质量很大程度取决于输入分布是否贴近真实使用场景；collector、dataset 模式和统计工具的价值，和量化器本身同样重要。

最关键的经验并不是如何调用 `ggml_quantize_chunk()`，而是：

- 先确认权重是怎么被推理代码读取的
- 再决定哪些 tensor 可以量化
- 用最小 GGUF 和单测把规则钉住
- 最后再用真实模型、真实 TTS、真实 calibration 路径做闭环验证
