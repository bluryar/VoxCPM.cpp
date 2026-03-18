#!/bin/bash

# VoxCPM quantization/export script.
# Usage: ./scripts/export_quantized_weights.sh

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"
QUANTIZE_BIN="${BUILD_DIR}/examples/voxcpm_quantize"
INFERENCE_BENCHMARK_SCRIPT="${SCRIPT_DIR}/benchmark_exported_weights.sh"
OUTPUT_DIR="${PROJECT_ROOT}/models/quantized"
LOG_DIR="${PROJECT_ROOT}/logs"

# Models
declare -a MODELS=("voxcpm1.5.gguf" "voxcpm-0.5b.gguf")
declare -a QUANT_TYPES=("Q4_K" "Q8_0" "F16")
declare -a AUDIO_VAE_MODES=("mixed" "f16")
QUANTIZE_THREADS=4

# Create directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

format_variant_label() {
    local quant="$1"
    local audio_vae_mode="$2"
    if [[ "${audio_vae_mode}" == "f16" ]]; then
        echo "${quant}+AudioVAE-F16"
    else
        echo "${quant}"
    fi
}

format_output_name() {
    local model="$1"
    local quant="$2"
    local audio_vae_mode="$3"
    local base_name=$(basename "${model}" .gguf)
    local quant_lower="${quant,,}"

    if [[ "${audio_vae_mode}" == "f16" ]]; then
        echo "${base_name}-${quant_lower}-audiovae-f16.gguf"
    else
        echo "${base_name}-${quant_lower}.gguf"
    fi
}

get_size_mb() {
    local file="$1"
    du -m "${file}" | cut -f1
}

append_manifest_row() {
    local manifest_file="$1"
    shift
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$@" >> "${manifest_file}"
}

# Check binaries exist
if [[ ! -x "${QUANTIZE_BIN}" ]]; then
    log_error "Quantize binary not found: ${QUANTIZE_BIN}"
    log_info "Please build the project first: cmake --build ${BUILD_DIR}"
    exit 1
fi

EXPORT_LOG_FILE="${LOG_DIR}/quantize_export_$(date +%Y%m%d_%H%M%S).log"
WEIGHTS_MANIFEST="$(mktemp)"
trap 'rm -f "${WEIGHTS_MANIFEST}"' EXIT

echo "model	variant	audio_vae_mode	model_path	original_size	export_time	compression" > "${WEIGHTS_MANIFEST}"
echo "VoxCPM Quantization Export Log" > "${EXPORT_LOG_FILE}"
echo "Date: $(date)" >> "${EXPORT_LOG_FILE}"
echo "======================================" >> "${EXPORT_LOG_FILE}"
echo "" >> "${EXPORT_LOG_FILE}"

MANIFEST_OUTPUT="${LOG_DIR}/quantized_weights_manifest_$(date +%Y%m%d_%H%M%S).tsv"

log_info "Starting VoxCPM quantization export"
log_info "Models: ${MODELS[*]}"
log_info "Quantization types: ${QUANT_TYPES[*]}"
log_info "AudioVAE modes: ${AUDIO_VAE_MODES[*]}"
echo ""

# Main loop
for model in "${MODELS[@]}"; do
    model_path="${PROJECT_ROOT}/models/${model}"

    if [[ ! -f "${model_path}" ]]; then
        log_warn "Model not found: ${model_path}"
        continue
    fi

    original_size=$(get_size_mb "${model_path}")
    log_info "Processing model: ${model} (${original_size} MB)"

    for quant_type in "${QUANT_TYPES[@]}"; do
        for audio_vae_mode in "${AUDIO_VAE_MODES[@]}"; do
            output_name=$(format_output_name "${model}" "${quant_type}" "${audio_vae_mode}")
            output_path="${OUTPUT_DIR}/${output_name}"
            variant_label=$(format_variant_label "${quant_type}" "${audio_vae_mode}")
            export_log="${LOG_DIR}/${output_name%.gguf}_export.log"

            log_info "  Exporting ${variant_label}..."

            quant_start=$(date +%s.%N)
            "${QUANTIZE_BIN}" \
                --input "${model_path}" \
                --output "${output_path}" \
                --type "${quant_type}" \
                --audio-vae-mode "${audio_vae_mode}" \
                --threads "${QUANTIZE_THREADS}" \
                2>&1 | tee "${export_log}"
            quant_end=$(date +%s.%N)
            export_time="$(awk "BEGIN {printf \"%.2fs\", ${quant_end} - ${quant_start}}")"

            if [[ ! -f "${output_path}" ]]; then
                log_error "  Export failed for ${output_name}"
                continue
            fi

            quant_size=$(get_size_mb "${output_path}")
            compression="$(awk "BEGIN {printf \"%.2fx\", ${original_size} / ${quant_size}}")"
            log_info "  Export complete: ${quant_size} MB (took ${export_time})"

            {
                echo "Model: ${model}"
                echo "Variant: ${variant_label}"
                echo "AudioVAE mode: ${audio_vae_mode}"
                echo "Output: ${output_path}"
                echo "Original size: ${original_size} MB"
                echo "Exported size: ${quant_size} MB"
                echo "Compression ratio: ${compression}"
                echo "Export time: ${export_time}"
                echo ""
            } >> "${EXPORT_LOG_FILE}"

            append_manifest_row "${WEIGHTS_MANIFEST}" \
                "${model}" "${variant_label}" "${audio_vae_mode}" "${output_path}" \
                "${original_size}" "${export_time}" "${compression}"
        done
    done

    f32_output_name="$(basename "${model}" .gguf)-f32.gguf"
    f32_output_path="${OUTPUT_DIR}/${f32_output_name}"
    log_info "  Copying F32 baseline..."
    cp "${model_path}" "${f32_output_path}"
    append_manifest_row "${WEIGHTS_MANIFEST}" \
        "${model}" "F32(baseline)" "original" "${f32_output_path}" \
        "${original_size}" "N/A(copy)" "1.00x"
    echo "Model: ${model}" >> "${EXPORT_LOG_FILE}"
    echo "Variant: F32(baseline)" >> "${EXPORT_LOG_FILE}"
    echo "AudioVAE mode: original" >> "${EXPORT_LOG_FILE}"
    echo "Output: ${f32_output_path}" >> "${EXPORT_LOG_FILE}"
    echo "Original size: ${original_size} MB" >> "${EXPORT_LOG_FILE}"
    echo "Exported size: ${original_size} MB" >> "${EXPORT_LOG_FILE}"
    echo "Compression ratio: 1.00x" >> "${EXPORT_LOG_FILE}"
    echo "Export time: N/A(copy)" >> "${EXPORT_LOG_FILE}"
    echo "" >> "${EXPORT_LOG_FILE}"
done

cp "${WEIGHTS_MANIFEST}" "${MANIFEST_OUTPUT}"
log_info "Export stage complete"
log_info "Export log saved to: ${EXPORT_LOG_FILE}"
log_info "Weights manifest saved to: ${MANIFEST_OUTPUT}"
if [[ -x "${INFERENCE_BENCHMARK_SCRIPT}" ]]; then
    log_info "Run inference benchmark separately with:"
    echo "  ${INFERENCE_BENCHMARK_SCRIPT} --weights-file ${MANIFEST_OUTPUT}"
else
    log_warn "Inference benchmark script not found or not executable: ${INFERENCE_BENCHMARK_SCRIPT}"
fi
