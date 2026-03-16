# VoxCPM imatrix Calibration Guide

This note captures a practical calibration workflow for `voxcpm_imatrix`, with a bias toward Chinese TTS.

## What `imatrix` is for

`imatrix.gguf` is an offline calibration artifact used during quantization, especially for lower-bit `IQ` formats such as `IQ2_XXS`, `IQ2_XS`, and `IQ1_S`.

It is not needed at inference time. `voxcpm_tts` only loads the final quantized model.

## Recommended workflow

1. Prepare a text file or dataset TSV.
2. If your target use case is voice cloning or prompt-audio TTS, use a stable prompt audio and its exact prompt text during collection.
3. Run `voxcpm_imatrix` on a representative sample set.
4. Inspect the result with `--show-statistics`.
5. Use the generated `imatrix.gguf` in `voxcpm_quantize --imatrix ...`.

## Input formats

`voxcpm_imatrix` now supports two calibration input modes:

- `--text-file`
  One target text per line. Good for plain TTS or for runs that use one shared global `--prompt-audio` and `--prompt-text`.
- `--dataset-file`
  TSV with either:
  - `text`
  - `text<TAB>prompt_text<TAB>prompt_audio`

This dataset mode is the right choice when different samples use different prompt audios or different prompt texts.

## Suggested sample scale

Use diversity first, then scale up.

- Smoke test: `32-64` lines
  Good for checking that collection and quantization work end to end.
- Baseline production calibration: `200-500` lines
  A strong default for Chinese TTS. Usually enough to stabilize most high-traffic layers.
- Higher-stability low-bit calibration: `800-2000` lines
  Recommended when targeting aggressive `IQ2_*` or `IQ1_*` formats.

For VoxCPM TTS, line length matters as much as line count. A good baseline is:

- `200-500` lines total
- average `20-60` Chinese characters per line
- a mix of short, medium, and long utterances

If you only calibrate on short conversational text, low-bit quantization may overfit that rhythm and do worse on narration, numbers, or mixed-language lines.

## Coverage checklist for Chinese TTS

Your calibration file should include all of the following:

- Short conversational lines
- Medium neutral narration
- Long clauses with commas and pauses
- Question sentences
- Exclamation sentences
- Arabic numbers like `2026`, `3.14`, `12:30`, phone-like digit runs
- Chinese numerals like `一百二十三`
- Mixed Chinese and English words
- Names, places, brands, abbreviations
- Punctuation-heavy text with `，。！？；：`
- Quotes and parentheses
- A few emotionally colored lines, but not only emotional lines

Avoid calibration data that is too narrow, for example:

- all news text
- all audiobook-style long narration
- all short chat replies
- mostly repeated sentence patterns

## Prompt-audio recommendations

If your deployment path uses `--prompt-audio` and `--prompt-text`, calibrate with them too.

Use:

- one stable prompt speaker per `imatrix` run
- exact matching prompt text
- prompt audio with clean speech and natural pacing

For a multi-speaker product, generate one shared baseline `imatrix` with no prompt audio first. If a specific speaker profile is especially important, you can additionally generate a speaker-biased `imatrix` and compare quantized quality.

## Example commands

Collect from plain text:

```bash
/home/orangepi/Codes/ggbond/VoxCPM.cpp/build/examples/voxcpm_imatrix \
  --text-file /home/orangepi/Codes/ggbond/VoxCPM.cpp/docs/imatrix_calibration_template_zh.txt \
  --output /tmp/voxcpm.zh.imatrix.gguf \
  --model-path /home/orangepi/Codes/ggbond/VoxCPM.cpp/models/voxcpm1.5.gguf \
  --threads 8 \
  --max-samples 300 \
  --max-decode-steps 24 \
  --save-frequency 50 \
  --show-statistics
```

Collect from dataset TSV with per-sample prompts:

```bash
/home/orangepi/Codes/ggbond/VoxCPM.cpp/build/examples/voxcpm_imatrix \
  --dataset-file /home/orangepi/Codes/ggbond/VoxCPM.cpp/docs/imatrix_calibration_dataset_zh.tsv \
  --output /tmp/voxcpm.zh.dataset.imatrix.gguf \
  --model-path /home/orangepi/Codes/ggbond/VoxCPM.cpp/models/voxcpm1.5.gguf \
  --threads 8 \
  --max-samples 300 \
  --save-frequency 50 \
  --show-statistics
```

Inspect an existing file:

```bash
/home/orangepi/Codes/ggbond/VoxCPM.cpp/build/examples/voxcpm_imatrix \
  --show-statistics \
  --in-file /tmp/voxcpm.zh.imatrix.gguf
```

Collect with voice-cloning prompt:

```bash
/home/orangepi/Codes/ggbond/VoxCPM.cpp/build/examples/voxcpm_imatrix \
  --text-file /home/orangepi/Codes/ggbond/VoxCPM.cpp/docs/imatrix_calibration_template_zh.txt \
  --output /tmp/voxcpm.zh.prompted.imatrix.gguf \
  --model-path /home/orangepi/Codes/ggbond/VoxCPM.cpp/models/voxcpm1.5.gguf \
  --prompt-audio /path/to/prompt.wav \
  --prompt-text "与提示音频完全一致的文本" \
  --threads 8 \
  --max-samples 300
```

## `--save-frequency` guidance

`--save-frequency N` writes snapshots like:

- `imatrix.gguf.at_50`
- `imatrix.gguf.at_100`

This is useful when:

- calibration runs are long
- you want resumable checkpoints
- you want to compare quantization quality at different calibration scales

Recommended values:

- quick experiments: `25` or `50`
- medium runs: `50` or `100`
- large runs: `100` or `200`

## `--show-statistics` guidance

Use `--show-statistics` after a run to check whether the calibration file looks healthy.

Healthy signs:

- `entries` count is close to expected quantizable matrix count
- `chunk_count` matches the number of processed samples
- no large number of `zero_count_entries`
- top tensors are concentrated in core transformer blocks and projections

Warning signs:

- very low `chunk_count`
- many `zero_count_entries`
- extremely tiny files caused by incomplete collection
- calibration built from only a few repetitive lines

## Practical recommendation

If you want one default recipe for Chinese TTS:

- Start with `300` lines
- Use the template file in this folder as a base
- If you need multiple prompt speakers, switch to the dataset TSV template in this folder
- Add `50-100` lines from your real production domain
- Use `--save-frequency 50`
- Inspect with `--show-statistics`
- Quantize first with `IQ4_NL` or `IQ3_S`
- Move to `IQ2_*` only after quality stays acceptable
