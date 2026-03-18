#!/bin/bash

# VoxCPM inference benchmark script for a list of exported GGUF weights.
# Usage:
#   ./scripts/benchmark_exported_weights.sh \
#     [--weights-file /tmp/weights.tsv] \
#     [--backend cpu|cuda|vulkan|auto] \
#     [--threads 8] \
#     [--summary-file /path/to/summary.txt]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"
TTS_BIN="${BUILD_DIR}/examples/voxcpm_tts"
LOG_DIR="${PROJECT_ROOT}/logs"

PROMPT_AUDIO="${PROJECT_ROOT}/examples/dabin.wav"
PROMPT_TEXT="可哪怕位于堂堂超一品官职,在十 二郡一言九鼎的大柱国口干舌燥了,这少年还是没什么反应"
TEST_TEXT="测试一下，这是一个流式音频"
THREADS=8
TIMESTEPS=10
CFG_VALUE=2.0
BACKEND="cpu"
OUTPUT_PREFIX="test"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

extract_timing_info() {
    local log_file="$1"
    local field="$2"

    case "${field}" in
        "vae_encode")
            grep "AudioVAE encode:" "${log_file}" | tail -n 1 | awk '{print $3}' | sed 's/s$//' || echo "N/A"
            ;;
        "model_inference")
            grep "Model inference:" "${log_file}" | tail -n 1 | awk '{print $3}' | sed 's/s$//' || echo "N/A"
            ;;
        "vae_decode")
            grep "AudioVAE decode:" "${log_file}" | tail -n 1 | awk '{print $3}' | sed 's/s$//' || echo "N/A"
            ;;
        "total_time")
            grep "Total:" "${log_file}" | tail -n 1 | awk '{print $2}' | sed 's/s$//' || echo "N/A"
            ;;
        "rtf_model_only")
            grep "Without AudioVAE:" "${log_file}" | tail -n 1 | awk '{print $3}' || echo "N/A"
            ;;
        "rtf_without_encode")
            grep "Without AudioVAE Encode:" "${log_file}" | tail -n 1 | awk '{print $4}' || echo "N/A"
            ;;
        "rtf_full")
            grep "Full pipeline:" "${log_file}" | tail -n 1 | awk '{print $3}' || echo "N/A"
            ;;
        *)
            echo "N/A"
            ;;
    esac
}

get_size_mb() {
    local file="$1"
    du -m "${file}" | cut -f1
}

append_table_row() {
    local table_file="$1"
    shift
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$@" >> "${table_file}"
}

print_ascii_table() {
    local table_file="$1"
    awk -F '\t' '
    BEGIN {
        headers[1]="Model"; headers[2]="Variant"; headers[3]="AudioVAE"; headers[4]="Backend";
        headers[5]="SizeMB"; headers[6]="Compression"; headers[7]="ExportTime"; headers[8]="TotalTime";
        headers[9]="RTF_wo_AudioVAE"; headers[10]="RTF_wo_AudioVAE_Encode"; headers[11]="RTF_Full";
        for (i=1; i<=11; i++) w[i]=length(headers[i]);
    }
    {
        for (i=1; i<=11; i++) {
            cell[NR,i]=$i;
            if (length($i) > w[i]) w[i]=length($i);
        }
        n=NR;
    }
    END {
        sep="+";
        for (i=1; i<=11; i++) {
            for (j=0; j<w[i]+1; j++) sep=sep "-";
            sep=sep "+";
        }
        print sep;
        printf "|";
        for (i=1; i<=11; i++) printf " %-*s |", w[i], headers[i];
        printf "\n";
        print sep;
        for (r=1; r<=n; r++) {
            printf "|";
            for (i=1; i<=11; i++) printf " %-*s |", w[i], cell[r,i];
            printf "\n";
        }
        print sep;
    }' "${table_file}"
}

WEIGHTS_FILE=""
SUMMARY_FILE=""

find_latest_manifest() {
    local latest
    latest="$(find "${LOG_DIR}" -maxdepth 1 -type f -name 'quantized_weights_manifest_*.tsv' | sort | tail -n 1)"
    if [[ -n "${latest}" ]]; then
        echo "${latest}"
    fi
}

validate_backend() {
    case "$1" in
        cpu|cuda|vulkan|auto)
            ;;
        *)
            log_error "--backend must be one of: cpu, cuda, vulkan, auto"
            exit 1
            ;;
    esac
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --weights-file)
            WEIGHTS_FILE="$2"
            shift 2
            ;;
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        --threads)
            THREADS="$2"
            shift 2
            ;;
        --timesteps)
            TIMESTEPS="$2"
            shift 2
            ;;
        --cfg-value)
            CFG_VALUE="$2"
            shift 2
            ;;
        --prompt-audio)
            PROMPT_AUDIO="$2"
            shift 2
            ;;
        --prompt-text)
            PROMPT_TEXT="$2"
            shift 2
            ;;
        --text)
            TEST_TEXT="$2"
            shift 2
            ;;
        --summary-file)
            SUMMARY_FILE="$2"
            shift 2
            ;;
        --help|-h)
            sed -n '1,12p' "$0"
            exit 0
            ;;
        *)
            log_error "Unknown argument: $1"
            exit 1
            ;;
    esac
done

validate_backend "${BACKEND}"

if [[ -z "${WEIGHTS_FILE}" ]]; then
    WEIGHTS_FILE="$(find_latest_manifest)"
fi

if [[ -z "${WEIGHTS_FILE}" ]]; then
    log_error "--weights-file is required, or place a manifest under ${LOG_DIR}"
    exit 1
fi

if [[ ! -f "${WEIGHTS_FILE}" ]]; then
    log_error "Weights manifest not found: ${WEIGHTS_FILE}"
    exit 1
fi

if [[ ! -x "${TTS_BIN}" ]]; then
    log_error "TTS binary not found: ${TTS_BIN}"
    log_info "Please build the project first: cmake --build ${BUILD_DIR}"
    exit 1
fi

mkdir -p "${LOG_DIR}"

if [[ -z "${SUMMARY_FILE}" ]]; then
    SUMMARY_FILE="${LOG_DIR}/benchmark_summary_${BACKEND}_$(date +%Y%m%d_%H%M%S).txt"
fi

TABLE_DATA_FILE="$(mktemp)"
trap 'rm -f "${TABLE_DATA_FILE}"' EXIT

echo "VoxCPM Inference Benchmark Summary" > "${SUMMARY_FILE}"
echo "Date: $(date)" >> "${SUMMARY_FILE}"
echo "Weights manifest: ${WEIGHTS_FILE}" >> "${SUMMARY_FILE}"
echo "Backend: ${BACKEND}" >> "${SUMMARY_FILE}"
echo "Threads: ${THREADS}" >> "${SUMMARY_FILE}"
echo "Timesteps: ${TIMESTEPS}" >> "${SUMMARY_FILE}"
echo "CFG value: ${CFG_VALUE}" >> "${SUMMARY_FILE}"
echo "======================================" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

log_info "Starting inference benchmark"
log_info "Weights manifest: ${WEIGHTS_FILE}"
log_info "Backend: ${BACKEND} | Threads: ${THREADS} | Timesteps: ${TIMESTEPS} | CFG: ${CFG_VALUE}"

while IFS=$'\t' read -r model variant audio_vae_mode model_path original_size export_time compression; do
    [[ -z "${model}" ]] && continue
    [[ "${model}" == "model" ]] && continue

    if [[ ! -f "${model_path}" ]]; then
        log_warn "Skipping missing weight: ${model_path}"
        continue
    fi

    safe_variant="$(echo "${variant}_${audio_vae_mode}" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9_' '_')"
    output_wav="/tmp/${OUTPUT_PREFIX}_${BACKEND}_${safe_variant}.wav"
    log_file="${LOG_DIR}/${model%.gguf}_${BACKEND}_${safe_variant}.log"

    log_info "Benchmarking ${model} | ${variant} | AudioVAE=${audio_vae_mode} | backend=${BACKEND}"

    tts_start=$(date +%s.%N)
    "${TTS_BIN}" \
        --prompt-audio "${PROMPT_AUDIO}" \
        --prompt-text "${PROMPT_TEXT}" \
        --text "${TEST_TEXT}" \
        --output "${output_wav}" \
        --model-path "${model_path}" \
        --threads "${THREADS}" \
        --inference-timesteps "${TIMESTEPS}" \
        --cfg-value "${CFG_VALUE}" \
        --backend "${BACKEND}" \
        2>&1 | tee "${log_file}"
    tts_end=$(date +%s.%N)
    wall_time=$(awk "BEGIN {printf \"%.2f\", ${tts_end} - ${tts_start}}")

    size_mb=$(get_size_mb "${model_path}")
    vae_encode=$(extract_timing_info "${log_file}" "vae_encode")
    model_inference=$(extract_timing_info "${log_file}" "model_inference")
    vae_decode=$(extract_timing_info "${log_file}" "vae_decode")
    total_time=$(extract_timing_info "${log_file}" "total_time")
    rtf_model_only=$(extract_timing_info "${log_file}" "rtf_model_only")
    rtf_without_encode=$(extract_timing_info "${log_file}" "rtf_without_encode")
    rtf_full=$(extract_timing_info "${log_file}" "rtf_full")

    if [[ -f "${output_wav}" ]]; then
        audio_duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "${output_wav}" 2>/dev/null || echo "unknown")
    else
        audio_duration="failed"
    fi

    echo "" >> "${SUMMARY_FILE}"
    echo "Model: ${model} | Variant: ${variant} | AudioVAE: ${audio_vae_mode} | Backend: ${BACKEND}" >> "${SUMMARY_FILE}"
    echo "  Original size: ${original_size} MB" >> "${SUMMARY_FILE}"
    echo "  Exported size: ${size_mb} MB" >> "${SUMMARY_FILE}"
    echo "  Compression ratio: ${compression}" >> "${SUMMARY_FILE}"
    echo "  Export time: ${export_time}" >> "${SUMMARY_FILE}"
    echo "  Inference wall time: ${wall_time}s" >> "${SUMMARY_FILE}"
    echo "" >> "${SUMMARY_FILE}"
    echo "  === Inference Timing ===" >> "${SUMMARY_FILE}"
    echo "  AudioVAE encode:   ${vae_encode}s" >> "${SUMMARY_FILE}"
    echo "  Model inference:   ${model_inference}s" >> "${SUMMARY_FILE}"
    echo "  AudioVAE decode:   ${vae_decode}s" >> "${SUMMARY_FILE}"
    echo "  Total:             ${total_time}s" >> "${SUMMARY_FILE}"
    echo "  Audio duration:    ${audio_duration}s" >> "${SUMMARY_FILE}"
    echo "" >> "${SUMMARY_FILE}"
    echo "  === RTF (Real-Time Factor) ===" >> "${SUMMARY_FILE}"
    echo "  Without AudioVAE:        ${rtf_model_only}" >> "${SUMMARY_FILE}"
    echo "  Without AudioVAE Encode: ${rtf_without_encode}  (model + decode)" >> "${SUMMARY_FILE}"
    echo "  Full pipeline:           ${rtf_full}" >> "${SUMMARY_FILE}"

    append_table_row "${TABLE_DATA_FILE}" \
        "${model}" "${variant}" "${audio_vae_mode}" "${BACKEND}" "${size_mb}" "${compression}" "${export_time}" \
        "${total_time}" "${rtf_model_only}" "${rtf_without_encode}" "${rtf_full}"
done < "${WEIGHTS_FILE}"

log_info "Inference benchmark complete!"
log_info "Summary saved to: ${SUMMARY_FILE}"
log_info "ASCII table:"
echo ""
print_ascii_table "${TABLE_DATA_FILE}"
echo ""
cat "${SUMMARY_FILE}"
