#include "voxcpm/quantize.h"

#include "ggml-cpu.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace voxcpm {

namespace {

static const char* const LLM_KV_QUANTIZE_IMATRIX_FILE = "quantize.imatrix.file";
static const char* const LLM_KV_QUANTIZE_IMATRIX_DATASET = "quantize.imatrix.dataset";
static const char* const LLM_KV_QUANTIZE_IMATRIX_N_ENTRIES = "quantize.imatrix.entries_count";
static const char* const LLM_KV_QUANTIZE_IMATRIX_N_CHUNKS = "quantize.imatrix.chunks_count";

static const char* const LLM_KV_IMATRIX_DATASETS = "imatrix.datasets";
static const char* const LLM_KV_IMATRIX_CHUNK_COUNT = "imatrix.chunk_count";
static const char* const LLM_KV_IMATRIX_CHUNK_SIZE = "imatrix.chunk_size";

struct TensorDataBuffer {
    std::string name;
    std::vector<uint8_t> bytes;
};

struct LayerLayout {
    int max_block_index = -1;
};

struct QuantizationPlan {
    std::unordered_map<std::string, LayerLayout> layouts;
};

struct TensorLayout {
    int n_dims = 0;
    std::array<int64_t, GGML_MAX_DIMS> ne = {1, 1, 1, 1};
};

struct LoadedImatrix {
    std::vector<std::string> datasets;
    std::unordered_map<std::string, std::vector<float>> entries;
    int chunks_count = 0;

    bool empty() const {
        return entries.empty();
    }
};

bool has_prefix(const std::string& value, const std::string& prefix) {
    return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
}

bool has_suffix(const std::string& value, const std::string& suffix) {
    return value.size() >= suffix.size() &&
           value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool remove_suffix(std::string* value, const std::string& suffix) {
    if (!value || !has_suffix(*value, suffix)) {
        return false;
    }
    value->resize(value->size() - suffix.size());
    return true;
}

bool is_float_tensor_type(ggml_type type) {
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16;
}

ggml_type quantize_base_type(ggml_ftype type);

bool is_supported_output_type(ggml_ftype type) {
    return type == GGML_FTYPE_MOSTLY_Q2_K ||
           type == GGML_FTYPE_MOSTLY_Q3_K ||
           type == GGML_FTYPE_MOSTLY_Q4_K ||
           type == GGML_FTYPE_MOSTLY_Q5_K ||
           type == GGML_FTYPE_MOSTLY_Q8_0 ||
           type == GGML_FTYPE_MOSTLY_F16 ||
           type == GGML_FTYPE_MOSTLY_IQ2_XXS ||
           type == GGML_FTYPE_MOSTLY_IQ2_XS ||
           type == GGML_FTYPE_MOSTLY_IQ2_S ||
           type == GGML_FTYPE_MOSTLY_IQ3_XXS ||
           type == GGML_FTYPE_MOSTLY_IQ3_S ||
           type == GGML_FTYPE_MOSTLY_IQ1_S ||
           type == GGML_FTYPE_MOSTLY_IQ1_M ||
           type == GGML_FTYPE_MOSTLY_IQ4_NL ||
           type == GGML_FTYPE_MOSTLY_IQ4_XS;
}

bool is_very_low_bit_k_type(ggml_ftype type) {
    return type == GGML_FTYPE_MOSTLY_Q2_K || type == GGML_FTYPE_MOSTLY_Q3_K;
}

bool is_low_bit_iq_type(ggml_ftype type) {
    return type == GGML_FTYPE_MOSTLY_IQ2_XXS ||
           type == GGML_FTYPE_MOSTLY_IQ2_XS ||
           type == GGML_FTYPE_MOSTLY_IQ2_S ||
           type == GGML_FTYPE_MOSTLY_IQ3_XXS ||
           type == GGML_FTYPE_MOSTLY_IQ3_S ||
           type == GGML_FTYPE_MOSTLY_IQ1_S ||
           type == GGML_FTYPE_MOSTLY_IQ1_M;
}

const char* ggml_type_name_safe(ggml_type type) {
    if (type < 0 || type >= GGML_TYPE_COUNT) {
        return "UNKNOWN";
    }
    return ggml_type_name(type);
}

bool parse_block_index(const std::string& name, const std::string& prefix, int* block_index) {
    if (!has_prefix(name, prefix)) {
        return false;
    }

    size_t offset = prefix.size();
    size_t end = offset;
    while (end < name.size() && std::isdigit(static_cast<unsigned char>(name[end]))) {
        ++end;
    }
    if (end == offset || end >= name.size() || name[end] != '.') {
        return false;
    }

    *block_index = std::stoi(name.substr(offset, end - offset));
    return true;
}

QuantizationPlan build_quantization_plan(gguf_context* gguf_ctx) {
    QuantizationPlan plan;
    const std::array<std::string, 4> prefixes = {
        "blk.",
        "residual_lm.blk.",
        "locenc.blk.",
        "locdit.blk.",
    };

    const int64_t n_tensors = gguf_get_n_tensors(gguf_ctx);
    for (int64_t i = 0; i < n_tensors; ++i) {
        const std::string name = gguf_get_tensor_name(gguf_ctx, i);
        for (const std::string& prefix : prefixes) {
            int block_index = -1;
            if (!parse_block_index(name, prefix, &block_index)) {
                continue;
            }
            LayerLayout& layout = plan.layouts[prefix];
            layout.max_block_index = std::max(layout.max_block_index, block_index);
        }
    }

    return plan;
}

bool is_norm_tensor_name(const std::string& name) {
    return has_suffix(name, "_norm.weight") ||
           has_suffix(name, ".output_norm.weight") ||
           name == "output_norm.weight";
}

bool is_audio_vae_tensor_name(const std::string& name) {
    return has_prefix(name, "audio_vae.");
}

bool is_audio_vae_weight_name(const std::string& name) {
    return is_audio_vae_tensor_name(name) && has_suffix(name, ".weight");
}

bool is_audio_vae_unused_weight(const std::string& name) {
    return name == "audio_vae.encoder.fc_logvar.weight";
}

bool is_audio_vae_depthwise_weight(const ggml_tensor* tensor, const std::string& name) {
    return is_audio_vae_weight_name(name) &&
           ggml_n_dims(tensor) == 3 &&
           tensor->ne[0] == 7 &&
           tensor->ne[1] == 1;
}

bool is_audio_vae_transpose_conv_weight(const std::string& name) {
    static const std::string prefix = "audio_vae.decoder.model.";
    if (!has_prefix(name, prefix)) {
        return false;
    }

    size_t pos = prefix.size();
    size_t end = pos;
    while (end < name.size() && std::isdigit(static_cast<unsigned char>(name[end]))) {
        ++end;
    }
    if (end == pos || end >= name.size() || name[end] != '.') {
        return false;
    }

    return name.compare(end + 1, std::string("block.1.weight").size(), "block.1.weight") == 0;
}

bool is_audio_vae_regular_conv_weight(const ggml_tensor* tensor, const std::string& name) {
    return is_audio_vae_weight_name(name) &&
           !is_audio_vae_depthwise_weight(tensor, name) &&
           !is_audio_vae_transpose_conv_weight(name);
}

TensorLayout make_tensor_layout(const ggml_tensor* tensor) {
    VOXCPM_ASSERT(tensor != nullptr);

    TensorLayout layout;
    layout.n_dims = ggml_n_dims(tensor);
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        layout.ne[static_cast<size_t>(i)] = tensor->ne[i];
    }
    return layout;
}

TensorLayout choose_output_layout(const ggml_tensor* tensor,
                                  const std::string& name,
                                  ggml_type output_type) {
    TensorLayout layout = make_tensor_layout(tensor);
    if (!is_audio_vae_regular_conv_weight(tensor, name) ||
        !ggml_is_quantized(output_type) ||
        ggml_n_dims(tensor) != 3) {
        return layout;
    }

    layout.n_dims = 2;
    layout.ne[0] = tensor->ne[0] * tensor->ne[1];
    layout.ne[1] = tensor->ne[2];
    layout.ne[2] = 1;
    layout.ne[3] = 1;
    return layout;
}

int64_t layout_nelements(const TensorLayout& layout) {
    int64_t total = 1;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        total *= layout.ne[static_cast<size_t>(i)];
    }
    return total;
}

ggml_tensor make_output_tensor_metadata(const ggml_tensor* tensor,
                                        ggml_type output_type,
                                        const TensorLayout& layout) {
    VOXCPM_ASSERT(tensor != nullptr);

    ggml_tensor out = *tensor;
    out.type = output_type;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        out.ne[i] = layout.ne[static_cast<size_t>(i)];
    }

    const int64_t blck_size = ggml_blck_size(output_type);
    VOXCPM_ASSERT(blck_size > 0);
    VOXCPM_ASSERT(out.ne[0] % blck_size == 0);

    out.nb[0] = ggml_type_size(output_type);
    out.nb[1] = out.nb[0] * (out.ne[0] / blck_size);
    for (int i = 2; i < GGML_MAX_DIMS; ++i) {
        out.nb[i] = out.nb[i - 1] * out.ne[i - 1];
    }
    out.data = nullptr;
    return out;
}

bool should_preserve_tensor(const ggml_tensor* tensor, const std::string& name) {
    if (has_prefix(name, "stop.")) {
        return true;
    }
    if (ggml_n_dims(tensor) < 2) {
        return true;
    }
    if (has_suffix(name, ".bias") || has_suffix(name, ".alpha")) {
        return true;
    }
    if (is_norm_tensor_name(name)) {
        return true;
    }
    if (name == "locenc.special_token") {
        return true;
    }
    return false;
}

ggml_type choose_audio_vae_target_type(const ggml_tensor* tensor,
                                       const std::string& name,
                                       const QuantizeOptions& options) {
    VOXCPM_ASSERT(tensor != nullptr);

    if (!is_audio_vae_tensor_name(name)) {
        return GGML_TYPE_COUNT;
    }
    if (has_suffix(name, ".bias") || has_suffix(name, ".alpha")) {
        return tensor->type;
    }
    if (options.audio_vae_mode == AudioVAEQuantizationMode::F16) {
        return GGML_TYPE_F16;
    }
    if (is_audio_vae_unused_weight(name)) {
        return GGML_TYPE_F16;
    }
    if (is_audio_vae_depthwise_weight(tensor, name) || is_audio_vae_transpose_conv_weight(name)) {
        return GGML_TYPE_F16;
    }

    const ggml_type base_type = quantize_base_type(options.file_type);
    VOXCPM_ASSERT(base_type != GGML_TYPE_COUNT);
    return base_type;
}

ggml_type quantize_base_type(ggml_ftype file_type) {
    switch (file_type) {
        case GGML_FTYPE_MOSTLY_Q2_K:
            return GGML_TYPE_Q2_K;
        case GGML_FTYPE_MOSTLY_Q3_K:
            return GGML_TYPE_Q3_K;
        case GGML_FTYPE_MOSTLY_Q4_K:
            return GGML_TYPE_Q4_K;
        case GGML_FTYPE_MOSTLY_Q5_K:
            return GGML_TYPE_Q5_K;
        case GGML_FTYPE_MOSTLY_Q8_0:
            return GGML_TYPE_Q8_0;
        case GGML_FTYPE_MOSTLY_F16:
            return GGML_TYPE_F16;
        case GGML_FTYPE_MOSTLY_IQ2_XXS:
            return GGML_TYPE_IQ2_XXS;
        case GGML_FTYPE_MOSTLY_IQ2_XS:
            return GGML_TYPE_IQ2_XS;
        case GGML_FTYPE_MOSTLY_IQ2_S:
            return GGML_TYPE_IQ2_S;
        case GGML_FTYPE_MOSTLY_IQ3_XXS:
            return GGML_TYPE_IQ3_XXS;
        case GGML_FTYPE_MOSTLY_IQ3_S:
            return GGML_TYPE_IQ3_S;
        case GGML_FTYPE_MOSTLY_IQ1_S:
            return GGML_TYPE_IQ1_S;
        case GGML_FTYPE_MOSTLY_IQ1_M:
            return GGML_TYPE_IQ1_M;
        case GGML_FTYPE_MOSTLY_IQ4_NL:
            return GGML_TYPE_IQ4_NL;
        case GGML_FTYPE_MOSTLY_IQ4_XS:
            return GGML_TYPE_IQ4_XS;
        default:
            return GGML_TYPE_COUNT;
    }
}

ggml_type transformer_sensitive_override_type(ggml_ftype file_type) {
    if (file_type == GGML_FTYPE_MOSTLY_Q4_K ||
        file_type == GGML_FTYPE_MOSTLY_IQ4_NL ||
        file_type == GGML_FTYPE_MOSTLY_IQ4_XS) {
        return GGML_TYPE_Q5_K;
    }
    if (is_very_low_bit_k_type(file_type) || is_low_bit_iq_type(file_type)) {
        return GGML_TYPE_Q4_K;
    }
    return GGML_TYPE_COUNT;
}

bool is_transformer_stack_name(const std::string& name, std::string* matched_prefix, int* block_index) {
    const std::array<std::string, 4> prefixes = {
        "blk.",
        "residual_lm.blk.",
        "locenc.blk.",
        "locdit.blk.",
    };

    for (const std::string& prefix : prefixes) {
        int parsed = -1;
        if (!parse_block_index(name, prefix, &parsed)) {
            continue;
        }
        if (matched_prefix) {
            *matched_prefix = prefix;
        }
        if (block_index) {
            *block_index = parsed;
        }
        return true;
    }

    return false;
}

ggml_type choose_target_type(const ggml_tensor* tensor,
                             const std::string& name,
                             ggml_ftype file_type,
                             const QuantizeOptions& options,
                             const QuantizationPlan& plan) {
    const ggml_type audio_vae_type = choose_audio_vae_target_type(tensor, name, options);
    if (audio_vae_type != GGML_TYPE_COUNT) {
        return audio_vae_type;
    }
    if (should_preserve_tensor(tensor, name)) {
        return tensor->type;
    }

    const ggml_type base_type = quantize_base_type(file_type);
    VOXCPM_ASSERT(base_type != GGML_TYPE_COUNT);

    if (file_type == GGML_FTYPE_MOSTLY_Q8_0 || file_type == GGML_FTYPE_MOSTLY_F16) {
        return base_type;
    }

    if (name == "token_embd.weight") {
        return GGML_TYPE_Q8_0;
    }

    if (has_prefix(name, "proj.") ||
        has_prefix(name, "fsq.") ||
        name == "locenc.in_proj.weight" ||
        name == "locdit.in_proj.weight" ||
        name == "locdit.out_proj.weight" ||
        name == "locdit.cond_proj.weight" ||
        has_prefix(name, "locdit.time_mlp.") ||
        has_prefix(name, "locdit.delta_time_mlp.")) {
        return GGML_TYPE_Q8_0;
    }

    std::string stack_prefix;
    int block_index = -1;
    if (is_transformer_stack_name(name, &stack_prefix, &block_index)) {
        const ggml_type sensitive_type = transformer_sensitive_override_type(file_type);
        if (sensitive_type != GGML_TYPE_COUNT && has_suffix(name, ".attn_v.weight")) {
            return sensitive_type;
        }
        if (has_suffix(name, ".ffn_down.weight")) {
            const auto it = plan.layouts.find(stack_prefix);
            if (it != plan.layouts.end()) {
                const int last_block = it->second.max_block_index;
                if ((block_index == 0 || block_index == last_block) && sensitive_type != GGML_TYPE_COUNT) {
                    return sensitive_type;
                }
            }
        }
    }

    return base_type;
}

bool tensor_can_change_type(const ggml_tensor* tensor,
                            const std::string&,
                            ggml_type dst_type,
                            const TensorLayout& layout) {
    if (tensor->type == dst_type) {
        return true;
    }
    if (dst_type == GGML_TYPE_F32) {
        return true;
    }
    return layout.ne[0] % ggml_blck_size(dst_type) == 0;
}

bool convert_tensor_to_f32(const ggml_tensor* tensor, std::vector<float>* dst) {
    if (!tensor || !dst) {
        return false;
    }

    const int64_t ne = ggml_nelements(tensor);
    if (ne < 0) {
        return false;
    }

    dst->resize(static_cast<size_t>(ne));
    if (tensor->type == GGML_TYPE_F32) {
        std::memcpy(dst->data(), tensor->data, static_cast<size_t>(ne) * sizeof(float));
        return true;
    }
    if (tensor->type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row(static_cast<const ggml_fp16_t*>(tensor->data), dst->data(), ne);
        return true;
    }
    if (tensor->type == GGML_TYPE_BF16) {
        ggml_bf16_to_fp32_row(static_cast<const ggml_bf16_t*>(tensor->data), dst->data(), ne);
        return true;
    }

    return false;
}

bool convert_f32_to_type(const std::vector<float>& src, ggml_type dst_type, std::vector<uint8_t>* dst) {
    if (!dst) {
        return false;
    }

    if (dst_type == GGML_TYPE_F32) {
        dst->resize(src.size() * sizeof(float));
        std::memcpy(dst->data(), src.data(), dst->size());
        return true;
    }
    if (dst_type == GGML_TYPE_F16) {
        dst->resize(src.size() * sizeof(ggml_fp16_t));
        ggml_fp32_to_fp16_row(src.data(), reinterpret_cast<ggml_fp16_t*>(dst->data()), static_cast<int64_t>(src.size()));
        return true;
    }
    if (dst_type == GGML_TYPE_BF16) {
        dst->resize(src.size() * sizeof(ggml_bf16_t));
        ggml_fp32_to_bf16_row_ref(src.data(), reinterpret_cast<ggml_bf16_t*>(dst->data()), static_cast<int64_t>(src.size()));
        return true;
    }

    return false;
}

bool quantize_rows_parallel(ggml_type dst_type,
                            const std::vector<float>& src,
                            const float* imatrix,
                            int64_t n_rows,
                            int64_t n_per_row,
                            int n_threads,
                            std::vector<uint8_t>* dst) {
    if (!dst || n_rows < 0 || n_per_row <= 0) {
        return false;
    }

    const size_t total_size = ggml_row_size(dst_type, n_per_row) * static_cast<size_t>(n_rows);
    dst->assign(total_size, 0);

    const int worker_count = std::max(1, n_threads);
    const int64_t rows_per_chunk = std::max<int64_t>(1, (64 * 1024) / std::max<int64_t>(1, n_per_row));
    std::mutex mutex;
    int64_t next_row = 0;
    bool ok = true;
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(std::max(0, worker_count - 1)));

    auto work = [&]() {
        while (true) {
            int64_t first_row = 0;
            int64_t this_rows = 0;
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (!ok || next_row >= n_rows) {
                    return;
                }
                first_row = next_row;
                this_rows = std::min<int64_t>(rows_per_chunk, n_rows - next_row);
                next_row += this_rows;
            }

            const size_t written = ggml_quantize_chunk(
                dst_type,
                src.data(),
                dst->data(),
                first_row * n_per_row,
                this_rows,
                n_per_row,
                imatrix);
            const size_t expected = ggml_row_size(dst_type, n_per_row) * static_cast<size_t>(this_rows);
            if (written != expected) {
                std::lock_guard<std::mutex> lock(mutex);
                ok = false;
                return;
            }

            void* row_ptr = dst->data() + ggml_row_size(dst_type, n_per_row) * static_cast<size_t>(first_row);
            if (!ggml_validate_row_data(dst_type, row_ptr, expected)) {
                std::lock_guard<std::mutex> lock(mutex);
                ok = false;
                return;
            }
        }
    };

    for (int i = 1; i < worker_count; ++i) {
        workers.emplace_back(work);
    }
    work();
    for (std::thread& worker : workers) {
        worker.join();
    }

    return ok;
}

const float* resolve_imatrix_for_tensor(const LoadedImatrix& imatrix,
                                        const std::string& name,
                                        ggml_type dst_type,
                                        int64_t n_per_row) {
    if (!ggml_quantize_requires_imatrix(dst_type)) {
        const auto it = imatrix.entries.find(name);
        if (it == imatrix.entries.end()) {
            return nullptr;
        }
        if (static_cast<int64_t>(it->second.size()) < n_per_row) {
            throw Error(
                ErrorCode::InvalidArgument,
                "imatrix entry for " + name + " is too small: expected at least " + std::to_string(n_per_row) +
                    " floats, found " + std::to_string(it->second.size()));
        }
        return it->second.data();
    }

    if (imatrix.empty()) {
        throw Error(
            ErrorCode::InvalidArgument,
            "quantization type " + std::string(ggml_type_name_safe(dst_type)) + " requires --imatrix");
    }

    const auto it = imatrix.entries.find(name);
    if (it == imatrix.entries.end()) {
        throw Error(
            ErrorCode::InvalidArgument,
            "missing imatrix entry for tensor " + name + " required by " + ggml_type_name_safe(dst_type));
    }
    if (static_cast<int64_t>(it->second.size()) < n_per_row) {
        throw Error(
            ErrorCode::InvalidArgument,
            "imatrix entry for " + name + " is too small: expected at least " + std::to_string(n_per_row) +
                " floats, found " + std::to_string(it->second.size()));
    }
    return it->second.data();
}

bool copy_or_convert_tensor_data(const ggml_tensor* tensor,
                                 ggml_type dst_type,
                                 const TensorLayout& layout,
                                 const LoadedImatrix& imatrix,
                                 int n_threads,
                                 std::vector<uint8_t>* dst) {
    if (!tensor || !dst) {
        return false;
    }

    const TensorLayout src_layout = make_tensor_layout(tensor);
    const bool same_layout = src_layout.n_dims == layout.n_dims && src_layout.ne == layout.ne;

    if (dst_type == tensor->type && same_layout) {
        dst->resize(ggml_nbytes(tensor));
        std::memcpy(dst->data(), tensor->data, dst->size());
        return true;
    }

    std::vector<float> src_f32;
    if (!convert_tensor_to_f32(tensor, &src_f32)) {
        return false;
    }

    if (dst_type == GGML_TYPE_F32 || dst_type == GGML_TYPE_F16 || dst_type == GGML_TYPE_BF16) {
        return convert_f32_to_type(src_f32, dst_type, dst);
    }

    if (layout.n_dims < 2) {
        return false;
    }

    const int64_t n_per_row = layout.ne[0];
    const int64_t n_rows = layout_nelements(layout) / std::max<int64_t>(1, n_per_row);
    const float* quant_weights =
        resolve_imatrix_for_tensor(imatrix, tensor->name, dst_type, n_per_row);
    return quantize_rows_parallel(dst_type, src_f32, quant_weights, n_rows, n_per_row, n_threads, dst);
}

void fill_stats_for_input_tensor(const ggml_tensor* tensor, QuantizeStats* stats) {
    if (!stats || !tensor) {
        return;
    }
    if (tensor->type >= 0 && tensor->type < GGML_TYPE_COUNT) {
        stats->input_type_counts[static_cast<size_t>(tensor->type)] += 1;
    }
    stats->input_bytes += ggml_nbytes(tensor);
    stats->total_tensors += 1;
}

void fill_stats_for_output_tensor(const std::string& name,
                                  bool quantized,
                                  bool policy_preserved,
                                  bool shape_preserved,
                                  ggml_type output_type,
                                  size_t output_bytes,
                                  QuantizeStats* stats) {
    if (!stats) {
        return;
    }
    if (output_type >= 0 && output_type < GGML_TYPE_COUNT) {
        stats->output_type_counts[static_cast<size_t>(output_type)] += 1;
    }
    stats->output_bytes += output_bytes;
    if (quantized) {
        stats->quantized_tensors += 1;
    } else {
        stats->preserved_tensors += 1;
    }
    if (policy_preserved) {
        stats->skipped_for_policy += 1;
    }
    if (shape_preserved) {
        stats->skipped_for_shape += 1;
    }
    if (has_prefix(name, "audio_vae.")) {
        stats->audio_vae_tensors += 1;
        if (quantized && ggml_is_quantized(output_type)) {
            stats->audio_vae_quantized_tensors += 1;
        } else if (output_type == GGML_TYPE_F16) {
            stats->audio_vae_f16_tensors += 1;
        } else {
            stats->audio_vae_preserved_tensors += 1;
        }
    }
}

int load_legacy_imatrix(const std::string& imatrix_file, LoadedImatrix* out) {
    VOXCPM_ASSERT(out != nullptr);

    std::ifstream in(imatrix_file.c_str(), std::ios::binary);
    if (!in) {
        throw Error(ErrorCode::FileNotFound, "failed to open imatrix file: " + imatrix_file);
    }

    int n_entries = 0;
    in.read(reinterpret_cast<char*>(&n_entries), sizeof(n_entries));
    if (in.fail() || n_entries < 1) {
        throw Error(ErrorCode::InvalidFormat, "no imatrix data in file: " + imatrix_file);
    }

    for (int i = 0; i < n_entries; ++i) {
        int len = 0;
        in.read(reinterpret_cast<char*>(&len), sizeof(len));
        if (in.fail() || len < 1) {
            throw Error(ErrorCode::InvalidFormat, "failed reading imatrix entry name length");
        }

        std::vector<char> name_as_vec(static_cast<size_t>(len) + 1, 0);
        in.read(name_as_vec.data(), len);
        if (in.fail()) {
            throw Error(ErrorCode::InvalidFormat, "failed reading imatrix entry name");
        }

        const std::string name(name_as_vec.data());
        int ncall = 0;
        int nval = 0;
        in.read(reinterpret_cast<char*>(&ncall), sizeof(ncall));
        in.read(reinterpret_cast<char*>(&nval), sizeof(nval));
        if (in.fail() || nval < 1) {
            throw Error(ErrorCode::InvalidFormat, "failed reading imatrix entry size for " + name);
        }

        std::vector<float>& entry = out->entries[name];
        entry.resize(static_cast<size_t>(nval));
        in.read(reinterpret_cast<char*>(entry.data()), nval * sizeof(float));
        if (in.fail()) {
            throw Error(ErrorCode::InvalidFormat, "failed reading imatrix entry payload for " + name);
        }

        if (ncall > 0) {
            for (float& value : entry) {
                value /= ncall;
            }
        }
    }

    int chunks_count = 0;
    if (in.peek() != EOF) {
        in.read(reinterpret_cast<char*>(&chunks_count), sizeof(chunks_count));
        int dataset_len = 0;
        in.read(reinterpret_cast<char*>(&dataset_len), sizeof(dataset_len));
        if (!in.fail() && dataset_len > 0) {
            std::vector<char> dataset_as_vec(static_cast<size_t>(dataset_len));
            in.read(dataset_as_vec.data(), dataset_len);
            if (!in.fail()) {
                out->datasets.emplace_back(dataset_as_vec.begin(), dataset_as_vec.end());
            }
        }
    }

    out->chunks_count = chunks_count;
    return chunks_count;
}

LoadedImatrix load_imatrix_file(const std::string& imatrix_file) {
    if (imatrix_file.empty()) {
        return {};
    }

    LoadedImatrix out;
    ggml_context* ctx_raw = nullptr;
    gguf_init_params params = {
        .no_alloc = false,
        .ctx = &ctx_raw,
    };

    UniqueGGUFContext gguf_ctx(gguf_init_from_file(imatrix_file.c_str(), params));
    UniqueContext ctx(ctx_raw);

    if (!gguf_ctx || !ctx) {
        load_legacy_imatrix(imatrix_file, &out);
        return out;
    }

    const int32_t n_entries = static_cast<int32_t>(gguf_get_n_tensors(gguf_ctx.get()));
    if (n_entries < 1) {
        throw Error(ErrorCode::InvalidFormat, "no imatrix tensor data in file: " + imatrix_file);
    }

    const int dataset_idx = gguf_find_key(gguf_ctx.get(), LLM_KV_IMATRIX_DATASETS);
    const int chunk_count_idx = gguf_find_key(gguf_ctx.get(), LLM_KV_IMATRIX_CHUNK_COUNT);
    const int chunk_size_idx = gguf_find_key(gguf_ctx.get(), LLM_KV_IMATRIX_CHUNK_SIZE);
    if (dataset_idx < 0 || chunk_count_idx < 0 || chunk_size_idx < 0) {
        throw Error(ErrorCode::InvalidFormat, "missing imatrix metadata in file: " + imatrix_file);
    }

    const uint32_t chunk_size = gguf_get_val_u32(gguf_ctx.get(), chunk_size_idx);
    VOXCPM_UNUSED(chunk_size);

    const std::string sums_suffix = ".in_sum2";
    const std::string counts_suffix = ".counts";
    std::map<std::string, std::pair<ggml_tensor*, ggml_tensor*>> sums_counts_for;

    for (ggml_tensor* cur = ggml_get_first_tensor(ctx.get()); cur; cur = ggml_get_next_tensor(ctx.get(), cur)) {
        std::string name = cur->name;
        if (name.empty()) {
            continue;
        }
        if (remove_suffix(&name, sums_suffix)) {
            sums_counts_for[std::move(name)].first = cur;
        } else if (remove_suffix(&name, counts_suffix)) {
            sums_counts_for[std::move(name)].second = cur;
        }
    }

    for (const auto& item : sums_counts_for) {
        const std::string& name = item.first;
        const ggml_tensor* sums = item.second.first;
        const ggml_tensor* counts = item.second.second;
        if (!sums || !counts) {
            throw Error(ErrorCode::InvalidFormat, "mismatched imatrix sums/counts for tensor: " + name);
        }

        const int64_t ne0 = sums->ne[0];
        const int64_t ne1 = sums->ne[1];
        std::vector<float>& entry = out.entries[name];
        entry.resize(static_cast<size_t>(ggml_nelements(sums)));
        for (int64_t row = 0; row < ne1; ++row) {
            const float count = static_cast<const float*>(counts->data)[row];
            for (int64_t col = 0; col < ne0; ++col) {
                const size_t offset = static_cast<size_t>(row * ne0 + col);
                entry[offset] = count > 0.0f ? static_cast<const float*>(sums->data)[offset] / count : 1.0f;
            }
        }
    }

    const int64_t n_datasets = static_cast<int64_t>(gguf_get_arr_n(gguf_ctx.get(), dataset_idx));
    out.datasets.reserve(static_cast<size_t>(std::max<int64_t>(0, n_datasets)));
    for (int64_t i = 0; i < n_datasets; ++i) {
        out.datasets.push_back(gguf_get_arr_str(gguf_ctx.get(), dataset_idx, static_cast<size_t>(i)));
    }
    out.chunks_count = static_cast<int>(gguf_get_val_u32(gguf_ctx.get(), chunk_count_idx));

    return out;
}

void attach_imatrix_metadata(const QuantizeOptions& options, const LoadedImatrix& imatrix, gguf_context* gguf_out) {
    if (options.imatrix_path.empty() || imatrix.empty()) {
        return;
    }

    gguf_set_val_str(gguf_out, LLM_KV_QUANTIZE_IMATRIX_FILE, options.imatrix_path.c_str());
    if (!imatrix.datasets.empty()) {
        gguf_set_val_str(gguf_out, LLM_KV_QUANTIZE_IMATRIX_DATASET, imatrix.datasets.front().c_str());
    }
    gguf_set_val_u32(gguf_out, LLM_KV_QUANTIZE_IMATRIX_N_ENTRIES, static_cast<uint32_t>(imatrix.entries.size()));
    if (imatrix.chunks_count > 0) {
        gguf_set_val_u32(gguf_out, LLM_KV_QUANTIZE_IMATRIX_N_CHUNKS, static_cast<uint32_t>(imatrix.chunks_count));
    }
}

}  // namespace

bool quantize_gguf(const QuantizeOptions& options, QuantizeStats* stats) {
    if (options.input_path.empty()) {
        throw Error(ErrorCode::InvalidArgument, "quantize_gguf requires input_path");
    }
    if (!options.dry_run && options.output_path.empty()) {
        throw Error(ErrorCode::InvalidArgument, "quantize_gguf requires output_path unless dry_run is enabled");
    }
    if (options.n_threads < 1) {
        throw Error(ErrorCode::InvalidArgument, "quantize_gguf requires n_threads >= 1");
    }
    if (!is_supported_output_type(options.file_type)) {
        throw Error(ErrorCode::InvalidArgument, "unsupported quantization type for VoxCPM");
    }

    const LoadedImatrix imatrix = load_imatrix_file(options.imatrix_path);
    QuantizeStats local_stats;
    ggml_context* ggml_ctx_raw = nullptr;
    gguf_init_params params = {
        .no_alloc = false,
        .ctx = &ggml_ctx_raw,
    };

    UniqueGGUFContext gguf_ctx(gguf_init_from_file(options.input_path.c_str(), params));
    UniqueContext ggml_ctx(ggml_ctx_raw);
    if (!gguf_ctx || !ggml_ctx) {
        throw Error(ErrorCode::InvalidFormat, "failed to load GGUF input model");
    }

    const QuantizationPlan plan = build_quantization_plan(gguf_ctx.get());

    UniqueGGUFContext gguf_out(gguf_init_empty());
    if (!gguf_out) {
        throw Error(ErrorCode::OutOfMemory, "failed to allocate output GGUF context");
    }

    gguf_set_kv(gguf_out.get(), gguf_ctx.get());
    gguf_set_val_u32(gguf_out.get(), "general.quantization_version", GGML_QNT_VERSION);
    gguf_set_val_u32(gguf_out.get(), "general.file_type", static_cast<uint32_t>(options.file_type));
    attach_imatrix_metadata(options, imatrix, gguf_out.get());

    std::vector<TensorDataBuffer> owned_buffers;
    owned_buffers.reserve(static_cast<size_t>(gguf_get_n_tensors(gguf_ctx.get())));

    const int64_t n_tensors = gguf_get_n_tensors(gguf_ctx.get());
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char* tensor_name = gguf_get_tensor_name(gguf_ctx.get(), i);
        ggml_tensor* tensor = ggml_get_tensor(ggml_ctx.get(), tensor_name);
        if (!tensor) {
            throw Error(ErrorCode::InvalidFormat, "GGUF tensor missing from ggml context: " + std::string(tensor_name));
        }

        fill_stats_for_input_tensor(tensor, &local_stats);

        if (!is_float_tensor_type(tensor->type)) {
            throw Error(
                ErrorCode::InvalidArgument,
                "quantize_gguf only accepts floating-point input tensors; found " +
                    std::string(ggml_type_name_safe(tensor->type)) + " in " + tensor_name);
        }

        const std::string name(tensor_name);
        const ggml_type desired_type = choose_target_type(tensor, name, options.file_type, options, plan);
        ggml_type output_type = desired_type;
        TensorLayout output_layout = choose_output_layout(tensor, name, output_type);
        bool shape_preserved = false;
        if (output_type != tensor->type && !tensor_can_change_type(tensor, name, output_type, output_layout)) {
            if (is_audio_vae_regular_conv_weight(tensor, name) && ggml_is_quantized(output_type)) {
                output_type = GGML_TYPE_F16;
                output_layout = choose_output_layout(tensor, name, output_type);
            } else {
                shape_preserved = true;
                output_type = tensor->type;
                output_layout = make_tensor_layout(tensor);
            }
        }
        const bool policy_preserved = desired_type == tensor->type && should_preserve_tensor(tensor, name);
        const bool quantized = output_type != tensor->type;

        const ggml_tensor output_tensor = make_output_tensor_metadata(tensor, output_type, output_layout);
        gguf_add_tensor(gguf_out.get(), &output_tensor);

        TensorDataBuffer& buffer = owned_buffers.emplace_back();
        buffer.name = name;
        if (!copy_or_convert_tensor_data(tensor, output_type, output_layout, imatrix, options.n_threads, &buffer.bytes)) {
            throw Error(ErrorCode::InvalidFormat, "failed to convert tensor data for " + name);
        }
        gguf_set_tensor_data(gguf_out.get(), tensor_name, buffer.bytes.data());

        fill_stats_for_output_tensor(
            name,
            quantized,
            policy_preserved,
            shape_preserved,
            output_type,
            buffer.bytes.size(),
            &local_stats);
    }

    if (!options.dry_run) {
        if (!gguf_write_to_file(gguf_out.get(), options.output_path.c_str(), false)) {
            throw Error(ErrorCode::BackendError, "failed to write quantized GGUF file");
        }
    }

    if (stats) {
        *stats = std::move(local_stats);
    }
    return true;
}

}  // namespace voxcpm
