/**
 * @file locdit.h
 * @brief VoxCPM Local Diffusion Transformer built on top of MiniCPM
 */

#ifndef VOXCPM_LOCDIT_H
#define VOXCPM_LOCDIT_H

#include "voxcpm/context.h"
#include "voxcpm/minicpm.h"

#include <memory>
#include <string>

namespace voxcpm {

class VoxCPMBackend;
class VoxCPMWeightStore;

struct LocDiTWeights {
    ggml_tensor* in_proj_weight = nullptr;
    ggml_tensor* in_proj_bias = nullptr;
    ggml_tensor* cond_proj_weight = nullptr;
    ggml_tensor* cond_proj_bias = nullptr;
    ggml_tensor* out_proj_weight = nullptr;
    ggml_tensor* out_proj_bias = nullptr;

    ggml_tensor* time_mlp_linear1_weight = nullptr;
    ggml_tensor* time_mlp_linear1_bias = nullptr;
    ggml_tensor* time_mlp_linear2_weight = nullptr;
    ggml_tensor* time_mlp_linear2_bias = nullptr;

    ggml_tensor* delta_time_mlp_linear1_weight = nullptr;
    ggml_tensor* delta_time_mlp_linear1_bias = nullptr;
    ggml_tensor* delta_time_mlp_linear2_weight = nullptr;
    ggml_tensor* delta_time_mlp_linear2_bias = nullptr;
};

class LocDiTModel {
public:
    LocDiTModel() = default;
    ~LocDiTModel();

    LocDiTModel(const LocDiTModel&) = delete;
    LocDiTModel& operator=(const LocDiTModel&) = delete;

    bool load_from_gguf(const std::string& gguf_path,
                        VoxCPMContext& weight_ctx,
                        VoxCPMContext& graph_ctx,
                        VoxCPMBackend& backend);
    bool load_from_store(const std::shared_ptr<VoxCPMWeightStore>& store,
                         VoxCPMBackend& backend);

    // x / cond / output: [feat_dim, seq_len, batch]
    // mu: [hidden_size, batch]
    // t / dt: [batch]
    ggml_tensor* forward(VoxCPMContext& ctx,
                         ggml_tensor* x,
                         ggml_tensor* mu,
                         ggml_tensor* t,
                         ggml_tensor* cond,
                         ggml_tensor* dt);

    const MiniCPMConfig& config() const { return decoder_.config(); }
    const LocDiTWeights& weights() const { return weights_; }
    int feat_dim() const { return feat_dim_; }
    const MiniCPMModel& decoder_model() const { return decoder_; }
    const void* shared_store_token() const { return shared_store_.get(); }
    bool uses_shared_weights() const { return shared_store_ != nullptr; }

private:
    ggml_tensor* sinusoidal_embedding(VoxCPMContext& ctx, ggml_tensor* scalar, int dim, float scale) const;
    ggml_tensor* timestep_mlp(VoxCPMContext& ctx,
                              ggml_tensor* input,
                              ggml_tensor* linear1_w,
                              ggml_tensor* linear1_b,
                              ggml_tensor* linear2_w,
                              ggml_tensor* linear2_b) const;

    ggml_tensor* compute_time_embedding(VoxCPMContext& ctx, ggml_tensor* t_scalar) const;
    ggml_tensor* compute_delta_time_embedding(VoxCPMContext& ctx, ggml_tensor* dt_scalar) const;

    ggml_tensor* forward_single(VoxCPMContext& ctx,
                                ggml_tensor* x,
                                ggml_tensor* mu,
                                ggml_tensor* t_scalar,
                                ggml_tensor* cond,
                                ggml_tensor* dt_scalar);

    bool init_scratch_cache(VoxCPMBackend& backend);

    LocDiTWeights weights_;
    MiniCPMModel decoder_;

    int feat_dim_ = 0;

    ggml_context* weight_ctx_ = nullptr;
    ggml_backend_buffer_t weight_buffer_ = nullptr;
    VoxCPMBackend* backend_ = nullptr;
    std::unique_ptr<MiniCPMKVCache> scratch_kv_cache_;
    std::shared_ptr<VoxCPMWeightStore> shared_store_;
};

}  // namespace voxcpm

#endif  // VOXCPM_LOCDIT_H
