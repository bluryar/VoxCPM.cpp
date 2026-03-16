/**
 * @file minicpm.cpp
 * @brief MiniCPM Transformer Backbone Implementation
 */

#include "voxcpm/minicpm.h"
#include "voxcpm/backend.h"
#include "voxcpm/weight-store.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace voxcpm {

namespace {

constexpr float kCausalMaskNeg = -1.0e9f;

static std::string normalize_prefix(const std::string& prefix) {
    if (prefix.empty()) {
        return "";
    }
    if (prefix.back() == '.') {
        return prefix;
    }
    return prefix + ".";
}

static bool load_tensor_data(FILE* file,
                             gguf_context* gguf_ctx,
                             int tensor_idx,
                             ggml_tensor* tensor,
                             ggml_backend_buffer_t buffer) {
    if (!file || !gguf_ctx || !tensor || !buffer) {
        return false;
    }

    const size_t offset = gguf_get_data_offset(gguf_ctx) + gguf_get_tensor_offset(gguf_ctx, tensor_idx);
    const size_t nbytes = ggml_nbytes(tensor);

    if (fseek(file, static_cast<long>(offset), SEEK_SET) != 0) {
        return false;
    }

    if (ggml_backend_buffer_is_host(buffer)) {
        return fread(tensor->data, 1, nbytes, file) == nbytes;
    }

    std::vector<uint8_t> temp(nbytes);
    if (fread(temp.data(), 1, nbytes, file) != nbytes) {
        return false;
    }
    ggml_backend_tensor_set(tensor, temp.data(), 0, nbytes);
    return true;
}

static std::string layer_tensor_name(const std::string& prefix, int layer, const char* suffix) {
    return prefix + "blk." + std::to_string(layer) + "." + suffix;
}

static bool get_u32_kv(gguf_context* gguf_ctx, const char* key, uint32_t& value) {
    const int idx = gguf_find_key(gguf_ctx, key);
    if (idx < 0) {
        return false;
    }
    value = gguf_get_val_u32(gguf_ctx, idx);
    return true;
}

static bool get_f32_kv(gguf_context* gguf_ctx, const char* key, float& value) {
    const int idx = gguf_find_key(gguf_ctx, key);
    if (idx < 0) {
        return false;
    }
    value = gguf_get_val_f32(gguf_ctx, idx);
    return true;
}

static bool get_f32_array_kv(gguf_context* gguf_ctx, const char* key, std::vector<float>& values) {
    const int idx = gguf_find_key(gguf_ctx, key);
    if (idx < 0) {
        return false;
    }

    const size_t n = gguf_get_arr_n(gguf_ctx, idx);
    const float* data = static_cast<const float*>(gguf_get_arr_data(gguf_ctx, idx));
    if (!data || n == 0) {
        values.clear();
        return true;
    }

    values.assign(data, data + n);
    return true;
}

static std::vector<float> expand_rope_factors(const std::vector<float>& source, int head_dim) {
    const int half_dim = head_dim / 2;
    std::vector<float> out(head_dim, 1.0f);
    for (int i = 0; i < half_dim; ++i) {
        const float v = i < static_cast<int>(source.size()) ? source[i] : 1.0f;
        out[i] = v;
        out[i + half_dim] = v;
    }
    return out;
}

}  // namespace

MiniCPMKVCache::MiniCPMKVCache(int n_layer, int n_kv_heads, int max_length, int head_dim)
    : n_layer_(n_layer),
      n_kv_heads_(n_kv_heads),
      max_length_(max_length),
      head_dim_(head_dim) {
}

MiniCPMKVCache::~MiniCPMKVCache() {
    if (buffer_kv_) {
        ggml_backend_buffer_free(buffer_kv_);
        buffer_kv_ = nullptr;
    }
    if (ctx_kv_) {
        ggml_free(ctx_kv_);
        ctx_kv_ = nullptr;
    }
}

void MiniCPMKVCache::init(VoxCPMBackend& backend) {
    if (ctx_kv_) {
        return;
    }

    const size_t ctx_size = ggml_tensor_overhead() * static_cast<size_t>(n_layer_) * 2 + 1024;
    ggml_init_params params = {
        .mem_size = ctx_size,
        .mem_buffer = nullptr,
        .no_alloc = true,
    };

    ctx_kv_ = ggml_init(params);
    if (!ctx_kv_) {
        throw Error(ErrorCode::OutOfMemory, "Failed to create MiniCPM KV cache context");
    }

    k_caches_.reserve(n_layer_);
    v_caches_.reserve(n_layer_);
    for (int i = 0; i < n_layer_; ++i) {
        k_caches_.push_back(ggml_new_tensor_3d(ctx_kv_, GGML_TYPE_F32, head_dim_, max_length_, n_kv_heads_));
        v_caches_.push_back(ggml_new_tensor_3d(ctx_kv_, GGML_TYPE_F32, head_dim_, max_length_, n_kv_heads_));
    }

    buffer_kv_ = ggml_backend_alloc_ctx_tensors(ctx_kv_, backend.raw_backend());
    if (!buffer_kv_) {
        throw Error(ErrorCode::OutOfMemory, "Failed to allocate MiniCPM KV cache buffer");
    }

    clear();
}

void MiniCPMKVCache::clear() {
    if (buffer_kv_) {
        ggml_backend_buffer_clear(buffer_kv_, 0);
    }
}

ggml_tensor* MiniCPMKVCache::get_k(ggml_context* ctx, int layer, int seq_len) const {
    VOXCPM_ASSERT(layer >= 0 && layer < n_layer_);
    VOXCPM_ASSERT(seq_len >= 0 && seq_len <= max_length_);
    ggml_tensor* k = k_caches_[layer];
    return ggml_view_3d(ctx, k, head_dim_, seq_len, n_kv_heads_, k->nb[1], k->nb[2], 0);
}

ggml_tensor* MiniCPMKVCache::get_v(ggml_context* ctx, int layer, int seq_len) const {
    VOXCPM_ASSERT(layer >= 0 && layer < n_layer_);
    VOXCPM_ASSERT(seq_len >= 0 && seq_len <= max_length_);
    ggml_tensor* v = v_caches_[layer];
    return ggml_view_3d(ctx, v, head_dim_, seq_len, n_kv_heads_, v->nb[1], v->nb[2], 0);
}

ggml_tensor* MiniCPMKVCache::get_k_slot(ggml_context* ctx, int layer, int position) const {
    return get_k_batch(ctx, layer, position, 1);
}

ggml_tensor* MiniCPMKVCache::get_v_slot(ggml_context* ctx, int layer, int position) const {
    return get_v_batch(ctx, layer, position, 1);
}

ggml_tensor* MiniCPMKVCache::get_k_batch(ggml_context* ctx, int layer, int start, int n_tokens) const {
    VOXCPM_ASSERT(layer >= 0 && layer < n_layer_);
    VOXCPM_ASSERT(start >= 0 && start + n_tokens <= max_length_);
    ggml_tensor* k = k_caches_[layer];
    const size_t offset = static_cast<size_t>(start) * k->nb[1];
    return ggml_view_3d(ctx, k, head_dim_, n_kv_heads_, n_tokens, k->nb[2], k->nb[1], offset);
}

ggml_tensor* MiniCPMKVCache::get_v_batch(ggml_context* ctx, int layer, int start, int n_tokens) const {
    VOXCPM_ASSERT(layer >= 0 && layer < n_layer_);
    VOXCPM_ASSERT(start >= 0 && start + n_tokens <= max_length_);
    ggml_tensor* v = v_caches_[layer];
    const size_t offset = static_cast<size_t>(start) * v->nb[1];
    return ggml_view_3d(ctx, v, head_dim_, n_kv_heads_, n_tokens, v->nb[2], v->nb[1], offset);
}

ggml_tensor* MiniCPMKVCache::raw_k_cache(int layer) const {
    VOXCPM_ASSERT(layer >= 0 && layer < n_layer_);
    return k_caches_[layer];
}

ggml_tensor* MiniCPMKVCache::raw_v_cache(int layer) const {
    VOXCPM_ASSERT(layer >= 0 && layer < n_layer_);
    return v_caches_[layer];
}

size_t MiniCPMKVCache::buffer_size() const {
    return buffer_kv_ ? ggml_backend_buffer_get_size(buffer_kv_) : 0;
}

void MiniCPMKVCache::copy_from(const MiniCPMKVCache& other, VoxCPMBackend& backend) {
    VOXCPM_ASSERT(n_layer_ == other.n_layer_);
    VOXCPM_ASSERT(n_kv_heads_ == other.n_kv_heads_);
    VOXCPM_ASSERT(max_length_ == other.max_length_);
    VOXCPM_ASSERT(head_dim_ == other.head_dim_);

    init(backend);

    for (int i = 0; i < n_layer_; ++i) {
        const size_t k_bytes = ggml_nbytes(k_caches_[i]);
        const size_t v_bytes = ggml_nbytes(v_caches_[i]);
        std::vector<uint8_t> temp(std::max(k_bytes, v_bytes));

        backend.tensor_get(other.raw_k_cache(i), temp.data(), 0, k_bytes);
        backend.tensor_set(raw_k_cache(i), temp.data(), 0, k_bytes);

        backend.tensor_get(other.raw_v_cache(i), temp.data(), 0, v_bytes);
        backend.tensor_set(raw_v_cache(i), temp.data(), 0, v_bytes);
    }
}

MiniCPMModel::MiniCPMModel(const MiniCPMConfig& config)
    : config_(config) {
    residual_scale_ = config_.use_mup
        ? (config_.scale_depth / std::sqrt(static_cast<float>(std::max(1, config_.n_layer))))
        : 1.0f;
}

MiniCPMModel::~MiniCPMModel() {
    if (aux_buffer_) {
        ggml_backend_buffer_free(aux_buffer_);
        aux_buffer_ = nullptr;
    }
    if (aux_ctx_) {
        ggml_free(aux_ctx_);
        aux_ctx_ = nullptr;
    }
    if (weight_buffer_) {
        ggml_backend_buffer_free(weight_buffer_);
        weight_buffer_ = nullptr;
    }
    if (weight_ctx_) {
        ggml_free(weight_ctx_);
        weight_ctx_ = nullptr;
    }
}

bool MiniCPMModel::update_config_from_gguf(gguf_context* gguf_ctx, const std::string& prefix) {
    uint32_t u32 = 0;
    float f32 = 0.0f;

    if (get_u32_kv(gguf_ctx, "llama.embedding_length", u32)) config_.hidden_size = static_cast<int>(u32);
    if (get_u32_kv(gguf_ctx, "llama.feed_forward_length", u32)) config_.intermediate_size = static_cast<int>(u32);
    if (get_u32_kv(gguf_ctx, "llama.attention.head_count", u32)) config_.n_heads = static_cast<int>(u32);
    if (get_u32_kv(gguf_ctx, "llama.attention.head_count_kv", u32)) config_.n_kv_heads = static_cast<int>(u32);
    if (get_u32_kv(gguf_ctx, "llama.vocab_size", u32)) config_.vocab_size = static_cast<int>(u32);
    if (get_u32_kv(gguf_ctx, "llama.context_length", u32)) config_.max_length = static_cast<int>(u32);
    if (get_u32_kv(gguf_ctx, "voxcpm_lm_config_scale_emb", u32)) config_.scale_emb = static_cast<int>(u32);
    if (get_u32_kv(gguf_ctx, "voxcpm_lm_config_dim_model_base", u32)) config_.dim_model_base = static_cast<int>(u32);
    if (get_u32_kv(gguf_ctx, "voxcpm_lm_config_use_mup", u32)) config_.use_mup = (u32 != 0);
    if (get_u32_kv(gguf_ctx, "voxcpm_lm_config_rope_scaling_original_max_position_embeddings", u32)) {
        config_.rope_original_max = static_cast<int>(u32);
    }
    if (get_f32_kv(gguf_ctx, "llama.attention.layer_norm_rms_epsilon", f32)) config_.rms_norm_eps = f32;
    if (get_f32_kv(gguf_ctx, "llama.rope.freq_base", f32)) config_.rope_freq_base = f32;
    if (get_f32_kv(gguf_ctx, "voxcpm_lm_config_scale_depth", f32)) config_.scale_depth = f32;
    get_f32_array_kv(gguf_ctx, "voxcpm_lm_config_rope_scaling_short_factor", config_.rope_short_factor);
    get_f32_array_kv(gguf_ctx, "voxcpm_lm_config_rope_scaling_long_factor", config_.rope_long_factor);

    bool has_variant_config = true;
    if (prefix.empty()) {
        const bool has_hidden = get_u32_kv(gguf_ctx, "voxcpm_lm_config_hidden_size", u32);
        if (has_hidden) config_.hidden_size = static_cast<int>(u32);
        const bool has_intermediate = get_u32_kv(gguf_ctx, "voxcpm_lm_config_intermediate_size", u32);
        if (has_intermediate) config_.intermediate_size = static_cast<int>(u32);
        const bool has_heads = get_u32_kv(gguf_ctx, "voxcpm_lm_config_num_attention_heads", u32);
        if (has_heads) config_.n_heads = static_cast<int>(u32);
        const bool has_kv_heads = get_u32_kv(gguf_ctx, "voxcpm_lm_config_num_key_value_heads", u32);
        if (has_kv_heads) config_.n_kv_heads = static_cast<int>(u32);
        const bool has_max_length = get_u32_kv(gguf_ctx, "voxcpm_lm_config_max_position_embeddings", u32);
        if (has_max_length) config_.max_length = static_cast<int>(u32);
        const bool has_vocab = get_u32_kv(gguf_ctx, "voxcpm_lm_config_vocab_size", u32);
        if (has_vocab) config_.vocab_size = static_cast<int>(u32);
        const bool has_layers = get_u32_kv(gguf_ctx, "voxcpm_lm_config_num_hidden_layers", u32) ||
                                get_u32_kv(gguf_ctx, "llama.block_count", u32);
        if (has_layers) config_.n_layer = static_cast<int>(u32);
        has_variant_config = has_hidden && has_intermediate && has_heads && has_kv_heads && has_max_length && has_vocab && has_layers;
    } else if (prefix == "residual_lm.") {
        has_variant_config = get_u32_kv(gguf_ctx, "voxcpm_residual_lm_num_layers", u32);
        if (has_variant_config) {
            config_.n_layer = static_cast<int>(u32);
        }
    } else if (prefix == "locenc.") {
        const bool has_hidden = get_u32_kv(gguf_ctx, "voxcpm_encoder_config_hidden_dim", u32);
        if (has_hidden) config_.hidden_size = static_cast<int>(u32);
        const bool has_intermediate = get_u32_kv(gguf_ctx, "voxcpm_encoder_config_ffn_dim", u32);
        if (has_intermediate) config_.intermediate_size = static_cast<int>(u32);
        const bool has_heads = get_u32_kv(gguf_ctx, "voxcpm_encoder_config_num_heads", u32);
        if (has_heads) config_.n_heads = static_cast<int>(u32);
        const bool has_layers = get_u32_kv(gguf_ctx, "voxcpm_encoder_config_num_layers", u32);
        if (has_layers) config_.n_layer = static_cast<int>(u32);
        has_variant_config = has_hidden && has_intermediate && has_heads && has_layers;
    } else if (prefix == "locdit.") {
        const bool has_hidden = get_u32_kv(gguf_ctx, "voxcpm_dit_config_hidden_dim", u32);
        if (has_hidden) config_.hidden_size = static_cast<int>(u32);
        const bool has_intermediate = get_u32_kv(gguf_ctx, "voxcpm_dit_config_ffn_dim", u32);
        if (has_intermediate) config_.intermediate_size = static_cast<int>(u32);
        const bool has_heads = get_u32_kv(gguf_ctx, "voxcpm_dit_config_num_heads", u32);
        if (has_heads) config_.n_heads = static_cast<int>(u32);
        const bool has_layers = get_u32_kv(gguf_ctx, "voxcpm_dit_config_num_layers", u32);
        if (has_layers) config_.n_layer = static_cast<int>(u32);
        has_variant_config = has_hidden && has_intermediate && has_heads && has_layers;
    }

    if (!has_variant_config) {
        return false;
    }

    residual_scale_ = config_.use_mup
        ? (config_.scale_depth / std::sqrt(static_cast<float>(std::max(1, config_.n_layer))))
        : 1.0f;
    return true;
}

bool MiniCPMModel::init_aux_tensors(VoxCPMBackend& backend) {
    if (aux_ctx_) {
        return true;
    }

    ggml_init_params params = {
        .mem_size = ggml_tensor_overhead() * 3 + 1024,
        .mem_buffer = nullptr,
        .no_alloc = true,
    };
    aux_ctx_ = ggml_init(params);
    if (!aux_ctx_) {
        return false;
    }

    rope_short_factor_ = ggml_new_tensor_1d(aux_ctx_, GGML_TYPE_F32, config_.head_dim());
    rope_long_factor_ = ggml_new_tensor_1d(aux_ctx_, GGML_TYPE_F32, config_.head_dim());
    pos_tensor_ = ggml_new_tensor_1d(aux_ctx_, GGML_TYPE_I32, config_.max_length);

    aux_buffer_ = ggml_backend_alloc_ctx_tensors(aux_ctx_, backend.raw_backend());
    if (!aux_buffer_) {
        return false;
    }

    const std::vector<float> short_factor = expand_rope_factors(config_.rope_short_factor, config_.head_dim());
    const std::vector<float> long_factor = expand_rope_factors(config_.rope_long_factor, config_.head_dim());
    ggml_backend_tensor_set(rope_short_factor_, short_factor.data(), 0, short_factor.size() * sizeof(float));
    ggml_backend_tensor_set(rope_long_factor_, long_factor.data(), 0, long_factor.size() * sizeof(float));

    std::vector<int32_t> positions(config_.max_length);
    for (int i = 0; i < config_.max_length; ++i) {
        positions[i] = i;
    }
    ggml_backend_tensor_set(pos_tensor_, positions.data(), 0, positions.size() * sizeof(int32_t));

    return true;
}

bool MiniCPMModel::load_weight_data(FILE* file, gguf_context* gguf_ctx) {
    const int n_tensors = gguf_get_n_tensors(gguf_ctx);
    for (int i = 0; i < n_tensors; ++i) {
        const char* name = gguf_get_tensor_name(gguf_ctx, i);
        ggml_tensor* tensor = ggml_get_tensor(weight_ctx_, name);
        if (!tensor) {
            continue;
        }
        if (!load_tensor_data(file, gguf_ctx, i, tensor, weight_buffer_)) {
            return false;
        }
    }
    return true;
}

bool MiniCPMModel::load_from_gguf(const std::string& gguf_path,
                                  const std::string& prefix,
                                  VoxCPMContext& weight_ctx,
                                  VoxCPMContext& graph_ctx,
                                  VoxCPMBackend& backend) {
    VOXCPM_UNUSED(weight_ctx);
    VOXCPM_UNUSED(graph_ctx);

    auto store = std::make_shared<VoxCPMWeightStore>();
    if (!store->load_from_file(gguf_path, backend)) {
        return false;
    }
    return load_from_store(store, prefix, backend);
}

bool MiniCPMModel::load_from_store(const std::shared_ptr<VoxCPMWeightStore>& store,
                                   const std::string& prefix,
                                   VoxCPMBackend& backend) {
    if (!store || !store->owns_storage()) {
        return false;
    }

    const std::string prefix_norm = normalize_prefix(prefix);
    if (!update_config_from_gguf(store->gguf(), prefix_norm)) {
        return false;
    }

    shared_store_ = store;
    weights_.layers.assign(config_.n_layer, {});

    auto get_required = [ctx = store->ggml_ctx()](const std::string& name, ggml_tensor** dst) -> bool {
        *dst = ggml_get_tensor(ctx, name.c_str());
        return *dst != nullptr;
    };

    weights_.embed_tokens = nullptr;
    if (prefix_norm.empty() && !get_required("token_embd.weight", &weights_.embed_tokens)) {
        return false;
    }

    const std::string norm_name = prefix_norm.empty() ? "output_norm.weight" : prefix_norm + "output_norm.weight";
    if (!get_required(norm_name, &weights_.norm)) {
        return false;
    }

    bool ok = true;
    for (int i = 0; i < config_.n_layer && ok; ++i) {
        MiniCPMLayerWeights& lw = weights_.layers[static_cast<size_t>(i)];
        ok &= get_required(layer_tensor_name(prefix_norm, i, "attn_norm.weight"), &lw.input_layernorm);
        ok &= get_required(layer_tensor_name(prefix_norm, i, "attn_q.weight"), &lw.q_proj);
        ok &= get_required(layer_tensor_name(prefix_norm, i, "attn_k.weight"), &lw.k_proj);
        ok &= get_required(layer_tensor_name(prefix_norm, i, "attn_v.weight"), &lw.v_proj);
        ok &= get_required(layer_tensor_name(prefix_norm, i, "attn_output.weight"), &lw.o_proj);
        ok &= get_required(layer_tensor_name(prefix_norm, i, "ffn_norm.weight"), &lw.post_layernorm);
        ok &= get_required(layer_tensor_name(prefix_norm, i, "ffn_gate.weight"), &lw.gate_proj);
        ok &= get_required(layer_tensor_name(prefix_norm, i, "ffn_up.weight"), &lw.up_proj);
        ok &= get_required(layer_tensor_name(prefix_norm, i, "ffn_down.weight"), &lw.down_proj);
    }

    return ok && init_aux_tensors(backend);
}

ggml_tensor* MiniCPMModel::rms_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight) const {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, config_.rms_norm_eps), weight);
}

ggml_tensor* MiniCPMModel::apply_rope(ggml_context* ctx,
                                      ggml_tensor* x,
                                      ggml_tensor* positions,
                                      int seq_len) const {
    const ggml_tensor* freq_factors = seq_len > config_.rope_original_max
        ? rope_long_factor_
        : rope_short_factor_;

    float attn_factor = 1.0f;
    if (seq_len > config_.rope_original_max && config_.rope_original_max > 1) {
        const float scale = static_cast<float>(seq_len) / static_cast<float>(config_.rope_original_max);
        attn_factor = std::sqrt(1.0f + std::log(scale) / std::log(static_cast<float>(config_.rope_original_max)));
    }

    return ggml_rope_ext(ctx, x, positions, const_cast<ggml_tensor*>(freq_factors),
                         config_.head_dim(), GGML_ROPE_TYPE_NEOX,
                         config_.rope_original_max, config_.rope_freq_base,
                         1.0f, 0.0f, attn_factor, 32.0f, 1.0f);
}

ggml_tensor* MiniCPMModel::create_causal_mask(ggml_context* ctx,
                                              ggml_tensor* positions,
                                              int total_len) const {
    ggml_tensor* key_positions = ggml_arange(ctx, 0.0f, static_cast<float>(total_len), 1.0f);
    ggml_tensor* query_positions = ggml_cast(ctx, positions, GGML_TYPE_F32);

    key_positions = ggml_reshape_2d(ctx, key_positions, total_len, 1);
    query_positions = ggml_reshape_2d(ctx, query_positions, 1, positions->ne[0]);

    ggml_tensor* target = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, total_len, positions->ne[0]);
    ggml_tensor* key_grid = ggml_repeat(ctx, key_positions, target);
    ggml_tensor* query_grid = ggml_repeat(ctx, query_positions, target);

    ggml_tensor* mask = ggml_sub(ctx, key_grid, query_grid);
    mask = ggml_add1(ctx, mask, ggml_arange(ctx, -0.5f, 0.5f, 1.0f));
    mask = ggml_step(ctx, mask);
    mask = ggml_scale(ctx, mask, kCausalMaskNeg);
    return ggml_cont(ctx, ggml_cast(ctx, mask, GGML_TYPE_F16));
}

ggml_tensor* MiniCPMModel::attention_forward(ggml_context* ctx,
                                             ggml_tensor* hidden,
                                             ggml_tensor* positions,
                                             const MiniCPMLayerWeights& lw,
                                             MiniCPMKVCache& kv_cache,
                                             int layer_idx,
                                             int n_tokens,
                                             int n_past,
                                             bool is_causal) const {
    const int head_dim = config_.head_dim();
    const int total_len = n_past + n_tokens;

    VOXCPM_ASSERT(total_len <= kv_cache.max_length());

    ggml_tensor* q = ggml_mul_mat(ctx, lw.q_proj, hidden);
    ggml_tensor* k = ggml_mul_mat(ctx, lw.k_proj, hidden);
    ggml_tensor* v = ggml_mul_mat(ctx, lw.v_proj, hidden);

    q = ggml_reshape_3d(ctx, q, head_dim, config_.n_heads, n_tokens);
    k = ggml_reshape_3d(ctx, k, head_dim, config_.n_kv_heads, n_tokens);
    v = ggml_reshape_3d(ctx, v, head_dim, config_.n_kv_heads, n_tokens);

    q = apply_rope(ctx, q, positions, total_len);
    k = apply_rope(ctx, k, positions, total_len);

    ggml_tensor* k_write = ggml_cpy(ctx, k, kv_cache.get_k_batch(ctx, layer_idx, n_past, n_tokens));
    ggml_tensor* v_write = ggml_cpy(ctx, v, kv_cache.get_v_batch(ctx, layer_idx, n_past, n_tokens));
    ggml_tensor* kv_sync = ggml_add(ctx, ggml_sum(ctx, k_write), ggml_sum(ctx, v_write));
    kv_sync = ggml_scale(ctx, kv_sync, 0.0f);
    ggml_tensor* k_cur = ggml_permute(ctx, k, 0, 2, 1, 3);
    ggml_tensor* v_cur = ggml_permute(ctx, v, 0, 2, 1, 3);

    ggml_tensor* k_all = k_cur;
    ggml_tensor* v_all = v_cur;
    if (n_past > 0) {
        ggml_tensor* k_past = kv_cache.get_k(ctx, layer_idx, n_past);
        ggml_tensor* v_past = kv_cache.get_v(ctx, layer_idx, n_past);
        k_all = ggml_concat(ctx, k_past, k_cur, 1);
        v_all = ggml_concat(ctx, v_past, v_cur, 1);
    }

    q = ggml_permute(ctx, q, 0, 2, 1, 3);

    ggml_tensor* mask = is_causal ? create_causal_mask(ctx, positions, total_len) : nullptr;

    ggml_tensor* attn = ggml_flash_attn_ext(ctx, q, k_all, v_all, mask,
                                            1.0f / std::sqrt(static_cast<float>(head_dim)),
                                            0.0f, 0.0f);
    attn = ggml_add1(ctx, attn, kv_sync);
    attn = ggml_reshape_2d(ctx, attn, config_.n_heads * head_dim, n_tokens);
    return ggml_mul_mat(ctx, lw.o_proj, attn);
}

ggml_tensor* MiniCPMModel::mlp_forward(ggml_context* ctx,
                                       ggml_tensor* hidden,
                                       const MiniCPMLayerWeights& lw) const {
    ggml_tensor* gate = ggml_silu(ctx, ggml_mul_mat(ctx, lw.gate_proj, hidden));
    ggml_tensor* up = ggml_mul_mat(ctx, lw.up_proj, hidden);
    ggml_tensor* fused = ggml_mul(ctx, gate, up);
    return ggml_mul_mat(ctx, lw.down_proj, fused);
}

ggml_tensor* MiniCPMModel::layer_forward(ggml_context* ctx,
                                         ggml_tensor* hidden,
                                         ggml_tensor* positions,
                                         const MiniCPMLayerWeights& lw,
                                         MiniCPMKVCache& kv_cache,
                                         int layer_idx,
                                         int n_tokens,
                                         int n_past,
                                         bool is_causal) const {
    ggml_tensor* residual = hidden;
    ggml_tensor* normed = rms_norm(ctx, hidden, lw.input_layernorm);
    ggml_tensor* attn_out = attention_forward(ctx, normed, positions, lw, kv_cache,
                                              layer_idx, n_tokens, n_past, is_causal);
    if (config_.use_mup) {
        attn_out = ggml_scale(ctx, attn_out, residual_scale_);
    }
    hidden = ggml_add(ctx, residual, attn_out);

    residual = hidden;
    normed = rms_norm(ctx, hidden, lw.post_layernorm);
    ggml_tensor* mlp_out = mlp_forward(ctx, normed, lw);
    if (config_.use_mup) {
        mlp_out = ggml_scale(ctx, mlp_out, residual_scale_);
    }
    return ggml_add(ctx, residual, mlp_out);
}

ggml_tensor* MiniCPMModel::forward(VoxCPMContext& ctx,
                                   ggml_tensor* input,
                                   ggml_tensor* positions,
                                   MiniCPMKVCache& kv_cache,
                                   bool is_causal) {
    VOXCPM_ASSERT(input != nullptr);

    ggml_context* raw = ctx.raw_context();
    const int n_tokens = input->ne[1] > 0 ? static_cast<int>(input->ne[1]) : 1;

    if (!positions) {
        positions = ggml_view_1d(raw, pos_tensor_, n_tokens, 0);
    }

    ggml_tensor* hidden = input;
    for (int i = 0; i < config_.n_layer; ++i) {
        hidden = layer_forward(raw, hidden, positions, weights_.layers[i], kv_cache, i, n_tokens, 0, is_causal);
    }
    return rms_norm(raw, hidden, weights_.norm);
}

ggml_tensor* MiniCPMModel::forward_step(VoxCPMContext& ctx,
                                        ggml_tensor* input,
                                        int position,
                                        MiniCPMKVCache& kv_cache,
                                        bool is_causal) {
    return forward_step(ctx, input, position, nullptr, kv_cache, is_causal);
}

ggml_tensor* MiniCPMModel::forward_step(VoxCPMContext& ctx,
                                        ggml_tensor* input,
                                        int position,
                                        ggml_tensor* positions,
                                        MiniCPMKVCache& kv_cache,
                                        bool is_causal) {
    VOXCPM_ASSERT(input != nullptr);
    VOXCPM_ASSERT(position >= 0 && position < config_.max_length);

    ggml_context* raw = ctx.raw_context();
    ggml_tensor* hidden = input;
    bool flatten_output = false;
    if (ggml_n_dims(hidden) == 1) {
        hidden = ggml_reshape_2d(raw, hidden, hidden->ne[0], 1);
        flatten_output = true;
    }

    if (!positions) {
        positions = ggml_view_1d(raw, pos_tensor_, 1, static_cast<size_t>(position) * sizeof(int32_t));
    }

    for (int i = 0; i < config_.n_layer; ++i) {
        hidden = layer_forward(raw, hidden, positions, weights_.layers[i], kv_cache, i, 1, position, is_causal);
    }

    hidden = rms_norm(raw, hidden, weights_.norm);
    if (flatten_output) {
        hidden = ggml_reshape_1d(raw, hidden, hidden->ne[0]);
    }
    return hidden;
}

}  // namespace voxcpm
