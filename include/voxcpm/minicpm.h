/**
 * @file minicpm.h
 * @brief MiniCPM Transformer Backbone
 *
 * Implements the shared MiniCPM transformer module used by BaseLM,
 * ResidualLM, LocEnc and LocDiT.
 */

#ifndef VOXCPM_MINICPM_H
#define VOXCPM_MINICPM_H

#include "voxcpm/common.h"
#include "voxcpm/config.h"
#include "voxcpm/context.h"
#include <string>
#include <vector>

namespace voxcpm {

class VoxCPMBackend;
class VoxCPMWeightStore;

struct MiniCPMLayerWeights {
    ggml_tensor* input_layernorm = nullptr;
    ggml_tensor* q_proj = nullptr;
    ggml_tensor* k_proj = nullptr;
    ggml_tensor* v_proj = nullptr;
    ggml_tensor* qkv_proj = nullptr;
    ggml_tensor* o_proj = nullptr;
    ggml_tensor* post_layernorm = nullptr;
    ggml_tensor* gate_proj = nullptr;
    ggml_tensor* up_proj = nullptr;
    ggml_tensor* gate_up_proj = nullptr;
    ggml_tensor* down_proj = nullptr;
};

struct MiniCPMWeights {
    ggml_tensor* embed_tokens = nullptr;
    std::vector<MiniCPMLayerWeights> layers;
    ggml_tensor* norm = nullptr;
    ggml_tensor* lm_head = nullptr;
};

class MiniCPMKVCache {
public:
    MiniCPMKVCache(int n_layer, int n_kv_heads, int max_length, int head_dim);
    ~MiniCPMKVCache();

    MiniCPMKVCache(const MiniCPMKVCache&) = delete;
    MiniCPMKVCache& operator=(const MiniCPMKVCache&) = delete;

    void init(VoxCPMBackend& backend);
    void clear();

    ggml_tensor* get_k(ggml_context* ctx, int layer, int seq_len) const;
    ggml_tensor* get_v(ggml_context* ctx, int layer, int seq_len) const;

    ggml_tensor* get_k_slot(ggml_context* ctx, int layer, int position) const;
    ggml_tensor* get_v_slot(ggml_context* ctx, int layer, int position) const;

    ggml_tensor* get_k_batch(ggml_context* ctx, int layer, int start, int n_tokens) const;
    ggml_tensor* get_v_batch(ggml_context* ctx, int layer, int start, int n_tokens) const;

    ggml_tensor* raw_k_cache(int layer) const;
    ggml_tensor* raw_v_cache(int layer) const;
    void copy_from(const MiniCPMKVCache& other, VoxCPMBackend& backend);

    int max_length() const { return max_length_; }
    int n_layer() const { return n_layer_; }
    int n_kv_heads() const { return n_kv_heads_; }
    int head_dim() const { return head_dim_; }
    size_t buffer_size() const;

private:
    int n_layer_;
    int n_kv_heads_;
    int max_length_;
    int head_dim_;

    ggml_context* ctx_kv_ = nullptr;
    ggml_backend_buffer_t buffer_kv_ = nullptr;
    std::vector<ggml_tensor*> k_caches_;
    std::vector<ggml_tensor*> v_caches_;
};

class MiniCPMModel {
public:
    explicit MiniCPMModel(const MiniCPMConfig& config = MiniCPMConfig());
    ~MiniCPMModel();

    MiniCPMModel(const MiniCPMModel&) = delete;
    MiniCPMModel& operator=(const MiniCPMModel&) = delete;

    bool load_from_gguf(const std::string& gguf_path,
                        const std::string& prefix,
                        VoxCPMContext& weight_ctx,
                        VoxCPMContext& graph_ctx,
                        VoxCPMBackend& backend);
    bool load_from_store(const std::shared_ptr<VoxCPMWeightStore>& store,
                         const std::string& prefix,
                         VoxCPMBackend& backend);

    ggml_tensor* forward(VoxCPMContext& ctx,
                         ggml_tensor* input,
                         ggml_tensor* positions,
                         MiniCPMKVCache& kv_cache,
                         bool is_causal = true,
                         bool write_kv_cache = true,
                         ggml_tensor* attention_mask = nullptr);

    ggml_tensor* forward_step(VoxCPMContext& ctx,
                              ggml_tensor* input,
                              int position,
                              ggml_tensor* positions,
                              MiniCPMKVCache& kv_cache,
                              bool is_causal = true,
                              bool write_kv_cache = true);

    ggml_tensor* forward_step(VoxCPMContext& ctx,
                              ggml_tensor* input,
                              int position,
                              MiniCPMKVCache& kv_cache,
                              bool is_causal = true,
                              bool write_kv_cache = true);

    const MiniCPMConfig& config() const { return config_; }
    const MiniCPMWeights& weights() const { return weights_; }
    ggml_tensor* get_pos_tensor() const { return pos_tensor_; }
    const void* shared_store_token() const { return shared_store_.get(); }
    bool uses_shared_weights() const { return shared_store_ != nullptr; }

private:
    ggml_tensor* rms_norm(ggml_context* ctx,
                          ggml_tensor* x,
                          ggml_tensor* weight) const;
    ggml_tensor* apply_rope(ggml_context* ctx, ggml_tensor* x, ggml_tensor* positions, int seq_len) const;
    ggml_tensor* create_causal_mask(ggml_context* ctx, ggml_tensor* positions, int total_len) const;

    ggml_tensor* attention_forward(ggml_context* ctx,
                                   ggml_tensor* hidden,
                                   ggml_tensor* positions,
                                   ggml_tensor* causal_mask,
                                   ggml_tensor* attention_mask,
                                   const MiniCPMLayerWeights& lw,
                                   MiniCPMKVCache& kv_cache,
                                   int layer_idx,
                                   int n_tokens,
                                   int n_past,
                                   bool is_causal,
                                   bool write_kv_cache) const;

    ggml_tensor* mlp_forward(ggml_context* ctx,
                             ggml_tensor* hidden,
                             const MiniCPMLayerWeights& lw) const;

    ggml_tensor* layer_forward(ggml_context* ctx,
                               ggml_tensor* hidden,
                               ggml_tensor* positions,
                               ggml_tensor* causal_mask,
                               ggml_tensor* attention_mask,
                               const MiniCPMLayerWeights& lw,
                               MiniCPMKVCache& kv_cache,
                               int layer_idx,
                               int n_tokens,
                               int n_past,
                               bool is_causal,
                               bool write_kv_cache) const;

    bool update_config_from_gguf(gguf_context* gguf_ctx, const std::string& prefix);
    bool init_aux_tensors(VoxCPMBackend& backend);
    bool init_fused_projection_tensors(VoxCPMBackend& backend, const std::string& prefix);
    bool load_weight_data(FILE* file, gguf_context* gguf_ctx);

    MiniCPMConfig config_;
    MiniCPMWeights weights_;

    ggml_context* weight_ctx_ = nullptr;
    ggml_backend_buffer_t weight_buffer_ = nullptr;

    ggml_context* aux_ctx_ = nullptr;
    ggml_backend_buffer_t aux_buffer_ = nullptr;
    ggml_context* fused_weight_ctx_ = nullptr;
    ggml_backend_buffer_t fused_weight_buffer_ = nullptr;
    ggml_tensor* rope_long_factor_ = nullptr;
    ggml_tensor* rope_short_factor_ = nullptr;
    ggml_tensor* pos_tensor_ = nullptr;
    std::shared_ptr<VoxCPMWeightStore> shared_store_;

    float residual_scale_ = 1.0f;
};

}  // namespace voxcpm

#endif  // VOXCPM_MINICPM_H
