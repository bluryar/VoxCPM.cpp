/**
 * @file localenc.h
 * @brief VoxCPM Local Encoder built on top of MiniCPM
 */

#ifndef VOXCPM_LOCALENC_H
#define VOXCPM_LOCALENC_H

#include "voxcpm/context.h"
#include "voxcpm/minicpm.h"

#include <memory>
#include <string>

namespace voxcpm {

class VoxCPMBackend;
class VoxCPMWeightStore;

struct LocEncWeights {
    ggml_tensor* in_proj_weight = nullptr;   // [feat_dim, hidden_size]
    ggml_tensor* in_proj_bias = nullptr;     // [hidden_size]
    ggml_tensor* special_token = nullptr;    // [hidden_size]
};

class LocEncModel {
public:
    LocEncModel() = default;
    ~LocEncModel();

    LocEncModel(const LocEncModel&) = delete;
    LocEncModel& operator=(const LocEncModel&) = delete;

    bool load_from_gguf(const std::string& gguf_path,
                        VoxCPMContext& weight_ctx,
                        VoxCPMContext& graph_ctx,
                        VoxCPMBackend& backend);
    bool load_from_store(const std::shared_ptr<VoxCPMWeightStore>& store,
                         VoxCPMBackend& backend);

    // input: [feat_dim, n_patches] or [hidden_size, n_patches]
    // output: [hidden_size] (CLS token output)
    ggml_tensor* forward_patch(VoxCPMContext& ctx, ggml_tensor* input);

    const MiniCPMConfig& config() const { return encoder_.config(); }
    const LocEncWeights& weights() const { return weights_; }
    int feat_dim() const { return feat_dim_; }
    const MiniCPMModel& encoder_model() const { return encoder_; }
    const void* shared_store_token() const { return shared_store_.get(); }
    bool uses_shared_weights() const { return shared_store_ != nullptr; }

private:
    bool init_scratch_cache(VoxCPMBackend& backend);

    LocEncWeights weights_;
    MiniCPMModel encoder_;

    int feat_dim_ = 0;

    ggml_context* weight_ctx_ = nullptr;
    ggml_backend_buffer_t weight_buffer_ = nullptr;
    VoxCPMBackend* backend_ = nullptr;
    std::unique_ptr<MiniCPMKVCache> scratch_kv_cache_;
    std::shared_ptr<VoxCPMWeightStore> shared_store_;
};

}  // namespace voxcpm

#endif  // VOXCPM_LOCALENC_H
