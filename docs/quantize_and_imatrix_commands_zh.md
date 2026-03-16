# VoxCPM 量化与 imatrix 命令速查（中文）

## 一、哪些量化类型需要 `--imatrix`

### 必须带 `--imatrix` 的类型（3 种）

以下类型在量化时**必须**提供 `--imatrix`，否则会报错：

| 类型     | 说明           |
|----------|----------------|
| `IQ2_XXS` | 超低比特 IQ    |
| `IQ2_XS`  | 低比特 IQ      |
| `IQ1_S`   | 1-bit 标量 IQ  |

### 可选 `--imatrix` 的类型（有则质量更好）

以下类型**不强制** imatrix，但提供 imatrix 通常会提升量化质量：

- **IQ 系列**：`IQ2_S`、`IQ3_XXS`、`IQ3_S`、`IQ1_M`、`IQ4_NL`、`IQ4_XS`
- **K-Quant**：`Q2_K`、`Q3_K`、`Q4_K`、`Q5_K`、`Q8_0`
- **其他**：`F16`

---

## 二、精度从高到低排序

按**量化精度/位宽**从高到低排列（同一行内精度接近，可视为同一档）：

| 顺序 | 类型 | 说明 |
|------|------|------|
| 1（最高） | **F16** | 半精度浮点，无量化 |
| 2 | **Q8_0** | 8-bit |
| 3 | **Q5_K** | 5-bit K-quant |
| 4 | **Q4_K**、**IQ4_NL**、**IQ4_XS** | 4-bit（IQ4_NL 通常略优于 IQ4_XS） |
| 5 | **Q3_K**、**IQ3_S**、**IQ3_XXS** | 3-bit（S > XXS） |
| 6 | **Q2_K**、**IQ2_S**、**IQ2_XS**、**IQ2_XXS** | 2-bit（S > XS > XXS） |
| 7（最低） | **IQ1_M**、**IQ1_S** | 1-bit |

**记忆**：数字越大精度越高（Q8 > Q5 > Q4 > Q3 > Q2）；IQ 同档里后缀 **S > XS > XXS**（越短越精细）。

---

## 三、`voxcpm_quantize` 用法

### 通用格式

```bash
/path/to/voxcpm_quantize \
  --input  INPUT.gguf \
  --output OUTPUT.gguf \
  --type   <TYPE> \
  [--audio-vae-mode {mixed|f16}] \
  [--imatrix PATH] \
  [--threads N] \
  [--dry-run]
```

### 必须带 imatrix 的三种类型（示例）

```bash
# IQ2_XXS（必须 --imatrix）
voxcpm_quantize --input /path/to/voxcpm1.5.gguf \
  --output /tmp/voxcpm1.5-iq2xxs.gguf \
  --type IQ2_XXS \
  --imatrix /tmp/voxcpm.imatrix.gguf \
  --threads 8

# IQ2_XS（必须 --imatrix）
voxcpm_quantize --input /path/to/voxcpm1.5.gguf \
  --output /tmp/voxcpm1.5-iq2xs.gguf \
  --type IQ2_XS \
  --imatrix /tmp/voxcpm.imatrix.gguf \
  --threads 8

# IQ1_S（必须 --imatrix）
voxcpm_quantize --input /path/to/voxcpm1.5.gguf \
  --output /tmp/voxcpm1.5-iq1s.gguf \
  --type IQ1_S \
  --imatrix /tmp/voxcpm.imatrix.gguf \
  --threads 8
```

### 可选 imatrix 的类型（示例）

```bash
# IQ2_S（可选 imatrix）
voxcpm_quantize --input /path/to/voxcpm1.5.gguf \
  --output /tmp/voxcpm1.5-iq2s.gguf \
  --type IQ2_S \
  --imatrix /tmp/voxcpm.imatrix.gguf \
  --threads 8

# IQ3_XXS
voxcpm_quantize --input /path/to/voxcpm1.5.gguf \
  --output /tmp/voxcpm1.5-iq3xxs.gguf \
  --type IQ3_XXS \
  --imatrix /tmp/voxcpm.imatrix.gguf \
  --threads 8

# IQ3_S
voxcpm_quantize --input /path/to/voxcpm1.5.gguf \
  --output /tmp/voxcpm1.5-iq3s.gguf \
  --type IQ3_S \
  --imatrix /tmp/voxcpm.imatrix.gguf \
  --threads 8

# IQ1_M
voxcpm_quantize --input /path/to/voxcpm1.5.gguf \
  --output /tmp/voxcpm1.5-iq1m.gguf \
  --type IQ1_M \
  --imatrix /tmp/voxcpm.imatrix.gguf \
  --threads 8

# IQ4_NL
voxcpm_quantize --input /path/to/voxcpm1.5.gguf \
  --output /tmp/voxcpm1.5-iq4nl.gguf \
  --type IQ4_NL \
  --imatrix /tmp/voxcpm.imatrix.gguf \
  --threads 8

# IQ4_XS
voxcpm_quantize --input /path/to/voxcpm1.5.gguf \
  --output /tmp/voxcpm1.5-iq4xs.gguf \
  --type IQ4_XS \
  --imatrix /tmp/voxcpm.imatrix.gguf \
  --threads 8
```

### 不需要 imatrix 的类型（K-Quant / F16 / Q8_0）

```bash
# Q4_K（默认类型，可不带 imatrix）
voxcpm_quantize --input /path/to/voxcpm1.5.gguf \
  --output /tmp/voxcpm1.5-q4k.gguf \
  --type Q4_K \
  --threads 8

# Q3_K
voxcpm_quantize --input /path/to/voxcpm1.5.gguf \
  --output /tmp/voxcpm1.5-q3k.gguf \
  --type Q3_K \
  --threads 8

# Q2_K
voxcpm_quantize --input /path/to/voxcpm1.5.gguf \
  --output /tmp/voxcpm1.5-q2k.gguf \
  --type Q2_K \
  --threads 8

# Q5_K
voxcpm_quantize --input /path/to/voxcpm1.5.gguf \
  --output /tmp/voxcpm1.5-q5k.gguf \
  --type Q5_K \
  --threads 8

# Q8_0
voxcpm_quantize --input /path/to/voxcpm1.5.gguf \
  --output /tmp/voxcpm1.5-q8_0.gguf \
  --type Q8_0 \
  --threads 8

# F16
voxcpm_quantize --input /path/to/voxcpm1.5.gguf \
  --output /tmp/voxcpm1.5-f16.gguf \
  --type F16 \
  --threads 8

# LM 保持 Q4_K，但整个 AudioVAE 强制导出为 F16
voxcpm_quantize --input /path/to/voxcpm1.5.gguf \
  --output /tmp/voxcpm1.5-audiovae-f16-q4k.gguf \
  --type Q4_K \
  --audio-vae-mode f16 \
  --threads 8
```

### 参数说明

| 参数        | 必填 | 说明 |
|-------------|------|------|
| `--input`  | 是   | 原始 GGUF 模型路径 |
| `--output` | 是*  | 输出 GGUF 路径（`--dry-run` 时可省略） |
| `--type`   | 是   | 见下文「支持的类型」 |
| `--audio-vae-mode` | 否 | `mixed` 或 `f16`；默认 `mixed` |
| `--imatrix`| 见上 | imatrix.gguf 路径；IQ2_XXS/IQ2_XS/IQ1_S 必填 |
| `--threads`| 否   | 线程数，默认 4 |
| `--dry-run`| 否   | 只检查不写文件 |

\* 使用 `--dry-run` 时可不写 `--output`。

### `--audio-vae-mode` 说明

- `mixed`：默认模式。规则卷积按 `--type` 量化，depthwise / transpose conv 走 `F16`，`bias/alpha` 保持 `F32`。
- `f16`：整个 AudioVAE 权重统一导出为 `F16`，其他模块仍按 `--type` 处理。

这个参数适合做两类事情：

1. 建立 “AudioVAE=F16” 的性能基线。
2. 在 mixed 量化没有速度收益时，快速验证问题是否主要来自 low-bit 规则卷积路径。

### 支持的类型（`--type` 取值）

- **K-Quant**：`Q2_K`、`Q3_K`、`Q4_K`、`Q5_K`、`Q8_0`
- **IQ**：`IQ2_XXS`、`IQ2_XS`、`IQ2_S`、`IQ3_XXS`、`IQ3_S`、`IQ1_S`、`IQ1_M`、`IQ4_NL`、`IQ4_XS`
- **浮点**：`F16`

---

## 四、`voxcpm_imatrix` 用法（生成 imatrix）

先生成 imatrix，再在量化时用 `--imatrix` 指向该文件。

### 从纯文本收集（每行一句）

```bash
voxcpm_imatrix \
  --text-file /path/to/lines.txt \
  --output /tmp/voxcpm.imatrix.gguf \
  --model-path /path/to/voxcpm1.5.gguf \
  --threads 8 \
  --max-samples 300 \
  --max-decode-steps 24 \
  --save-frequency 50
```

### 从 TSV 数据集收集（支持每行不同 prompt 音频/文本）

```bash
voxcpm_imatrix \
  --dataset-file /path/to/dataset.tsv \
  --output /tmp/voxcpm.imatrix.gguf \
  --model-path /path/to/voxcpm1.5.gguf \
  --threads 8 \
  --max-samples 300 \
  --save-frequency 50
```

TSV 格式：`text` 或 `text<TAB>prompt_text<TAB>prompt_audio`。

### 带固定 prompt 音频（克隆向）

```bash
voxcpm_imatrix \
  --text-file /path/to/lines.txt \
  --output /tmp/voxcpm.prompted.imatrix.gguf \
  --model-path /path/to/voxcpm1.5.gguf \
  --prompt-audio /path/to/prompt.wav \
  --prompt-text "与提示音频完全一致的文本" \
  --threads 8 \
  --max-samples 300
```

### 查看 imatrix 统计

```bash
voxcpm_imatrix --show-statistics --in-file /tmp/voxcpm.imatrix.gguf
```

### 常用参数

| 参数                 | 说明 |
|----------------------|------|
| `--text-file`        | 每行一句的文本文件 |
| `--dataset-file`     | TSV：text 或 text\\tprompt_text\\tprompt_audio |
| `--output` / `-o`    | 输出 imatrix.gguf |
| `--model-path`       | 原始 GGUF 模型 |
| `--prompt-audio`     | 固定 prompt 音频（可选） |
| `--prompt-text`      | 固定 prompt 文本（可选） |
| `--cfg-value`        | CFG，默认 2.0 |
| `--inference-timesteps` | 推理步数，默认 10 |
| `--threads`         | 线程数，默认 4 |
| `--max-samples`      | 最多跑多少条，0=全部 |
| `--max-decode-steps`| 每条最多 decode 步数，默认 32 |
| `--save-frequency`  | 每 N 条保存一次快照，0=不保存 |
| `--show-statistics` | 与 `--in-file` 一起用，查看统计 |
| `--seed`            | 随机种子，默认 1234 |

---

## 五、`voxcpm_tts` 推理（使用量化模型）

量化后的模型用法与 F32 相同，**不需要**再带 imatrix：

```bash
voxcpm_tts \
  --text "测试一下，这是一个流式音频" \
  --prompt-audio /path/to/dabin.wav \
  --prompt-text "可哪怕位于堂堂超一品官职..." \
  --output /path/to/out.wav \
  --model-path /tmp/voxcpm1.5-iq2xxs.gguf \
  --threads 8 \
  --inference-timesteps 10 \
  --cfg-value 2.0
```

---

## 六、推荐流程小结

1. **若用 IQ2_XXS / IQ2_XS / IQ1_S**：先跑 `voxcpm_imatrix` 生成 imatrix，再在 `voxcpm_quantize` 中加 `--imatrix`。
2. **若用其他 IQ 或低比特 K-Quant**：建议同样先跑 imatrix，量化时加 `--imatrix` 以提升质量。
3. **若只用 Q4_K/Q5_K/Q8_0**：可不做 imatrix，直接 `voxcpm_quantize`。
4. 推理阶段只用 `--model-path` 指向量化后的 GGUF，无需 imatrix。

更细的 imatrix 数据准备与建议见：`imatrix_calibration_guide.md`。
