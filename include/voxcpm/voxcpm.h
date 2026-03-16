#ifndef VOXCPM_VOXCPM_H
#define VOXCPM_VOXCPM_H

#include "voxcpm/components.h"
#include "voxcpm/config.h"
#include "voxcpm/context.h"
#include "voxcpm/fsq.h"
#include "voxcpm/localenc.h"
#include "voxcpm/locdit.h"
#include "voxcpm/minicpm.h"
#include "voxcpm/unified_cfm.h"

#include <memory>
#include <string>
#include <vector>

namespace voxcpm {

class VoxCPMBackend;
class VoxCPMImatrixCollector;
class VoxCPMWeightStore;

struct VoxCPMDecodeState {
    std::unique_ptr<MiniCPMKVCache> base_lm_cache;
    std::unique_ptr<MiniCPMKVCache> residual_lm_cache;
    std::vector<float> lm_hidden;
    std::vector<float> residual_hidden;
    int current_position = 0;
    std::vector<float> prefix_feat_cond;  // [patch_size, feat_dim] in patch-major order
    int streaming_prefix_len = 3;

    VoxCPMDecodeState() = default;
    ~VoxCPMDecodeState() = default;

    VoxCPMDecodeState(const VoxCPMDecodeState&) = delete;
    VoxCPMDecodeState& operator=(const VoxCPMDecodeState&) = delete;
    VoxCPMDecodeState(VoxCPMDecodeState&&) noexcept = default;
    VoxCPMDecodeState& operator=(VoxCPMDecodeState&&) noexcept = default;
};

struct VoxCPMDecodeResult {
    std::vector<float> output_0;  // [patch_size, feat_dim] in patch-major order
    VoxCPMDecodeState output_1;
    bool output_2 = false;
};

class VoxCPMRuntime {
public:
    VoxCPMRuntime() = default;
    ~VoxCPMRuntime() = default;

    VoxCPMRuntime(const VoxCPMRuntime&) = delete;
    VoxCPMRuntime& operator=(const VoxCPMRuntime&) = delete;

    bool load_from_gguf(const std::string& gguf_path,
                        VoxCPMContext& weight_ctx,
                        VoxCPMContext& graph_ctx,
                        VoxCPMBackend& backend);
    bool load_from_store(const std::shared_ptr<VoxCPMWeightStore>& store,
                         VoxCPMBackend& backend);

    VoxCPMDecodeState create_decode_state() const;

    VoxCPMDecodeState prefill(const std::vector<int32_t>& text,
                              const std::vector<int32_t>& text_mask,
                              const std::vector<float>& feat,
                              const std::vector<int32_t>& feat_mask,
                              int seq_len,
                              int streaming_prefix_len = 3);

    VoxCPMDecodeResult decode(VoxCPMDecodeState state,
                              const std::vector<float>& z,
                              int inference_timesteps = 10,
                              float cfg_value = 2.0f);

    // Benchmark helpers exposing stable module boundaries without changing
    // inference semantics.
    std::vector<float> benchmark_encode_feature_sequence(const std::vector<float>& feat, int seq_len);
    std::vector<float> benchmark_run_embedding(const std::vector<int32_t>& token_ids);
    std::vector<float> benchmark_run_enc_to_lm_projection(const std::vector<float>& input, int seq_len);
    std::vector<float> benchmark_run_lm_to_dit_projection(const std::vector<float>& input);
    std::vector<float> benchmark_run_res_to_dit_projection(const std::vector<float>& input);
    std::vector<float> benchmark_run_fsq_2d(const std::vector<float>& input, int seq_len);
    std::vector<float> benchmark_run_base_lm_forward(const std::vector<float>& input,
                                                     int seq_len,
                                                     MiniCPMKVCache& kv_cache,
                                                     bool is_causal = true);
    std::vector<float> benchmark_run_residual_lm_forward(const std::vector<float>& input,
                                                         int seq_len,
                                                         MiniCPMKVCache& kv_cache,
                                                         bool is_causal = true);
    std::vector<float> benchmark_run_unified_cfm(const std::vector<float>& z,
                                                 const std::vector<float>& mu,
                                                 const std::vector<float>& cond,
                                                 int n_timesteps,
                                                 float cfg_value);
    std::vector<float> benchmark_run_stop_predictor(const std::vector<float>& input);
    std::vector<float> benchmark_run_locenc_patch_to_lm_embed(const std::vector<float>& patch);
    std::vector<float> benchmark_run_base_lm_decode_step(const std::vector<float>& curr_embed,
                                                         int position,
                                                         MiniCPMKVCache& kv_cache);
    std::vector<float> benchmark_run_residual_lm_decode_step(const std::vector<float>& input,
                                                             int position,
                                                             MiniCPMKVCache& kv_cache,
                                                             bool is_causal = true);
    std::vector<float> benchmark_run_decode_front_half(const std::vector<float>& z,
                                                       const std::vector<float>& lm_hidden,
                                                       const std::vector<float>& residual_hidden,
                                                       const std::vector<float>& prefix_feat_cond,
                                                       int inference_timesteps,
                                                       float cfg_value);
    VoxCPMDecodeState benchmark_clone_state(const VoxCPMDecodeState& state) const;

    const VoxCPMConfig& config() const { return config_; }
    const MiniCPMModel& base_lm() const { return base_lm_; }
    const MiniCPMModel& residual_lm() const { return residual_lm_; }
    const LocEncModel& feat_encoder() const { return feat_encoder_; }
    const LocDiTModel& feat_decoder_estimator() const { return feat_decoder_estimator_; }
    const FSQ& fsq_layer() const { return fsq_layer_; }
    const VoxCPMComponents* components() const { return components_.get(); }
    const void* shared_store_token() const { return weight_store_.get(); }
    bool uses_shared_weights() const { return weight_store_ != nullptr; }
    void set_imatrix_collector(VoxCPMImatrixCollector* collector) { imatrix_collector_ = collector; }

private:
    bool update_config_from_gguf(const std::string& gguf_path);
    bool update_config_from_store(const VoxCPMWeightStore& store);
    void maybe_collect_graph(ggml_cgraph* graph);

    std::vector<float> run_locenc_patch(const float* patch_data);
    std::vector<float> run_embedding(const std::vector<int32_t>& token_ids);
    std::vector<float> run_projection_1d(LinearProjection& projection,
                                         const std::vector<float>& input,
                                         int in_dim,
                                         int out_dim);
    std::vector<float> run_projection_2d(LinearProjection& projection,
                                         const std::vector<float>& input,
                                         int in_dim,
                                         int seq_len,
                                         int out_dim);
    std::vector<float> run_stop_predictor(const std::vector<float>& input);
    std::vector<float> run_fsq_1d(const std::vector<float>& input);
    std::vector<float> run_fsq_2d(const std::vector<float>& input, int seq_len);
    std::vector<float> run_minicpm_forward(MiniCPMModel& model,
                                           const std::vector<float>& input,
                                           int seq_len,
                                           MiniCPMKVCache& kv_cache,
                                           bool is_causal);
    std::vector<float> run_minicpm_forward_step(MiniCPMModel& model,
                                                const std::vector<float>& input,
                                                int position,
                                                MiniCPMKVCache& kv_cache,
                                                bool is_causal);
    std::vector<float> run_unified_cfm(const std::vector<float>& z,
                                       const std::vector<float>& mu,
                                       const std::vector<float>& cond,
                                       int n_timesteps,
                                       float cfg_value);
    void run_decode_front_half(const std::vector<float>& z,
                               const std::vector<float>& lm_hidden,
                               const std::vector<float>& residual_hidden,
                               const std::vector<float>& prefix_feat_cond,
                               int inference_timesteps,
                               float cfg_value,
                               std::vector<float>& output_0);
    std::vector<float> run_locenc_patch_to_lm_embed(const std::vector<float>& patch);
    std::vector<float> run_base_lm_decode_step(const std::vector<float>& curr_embed,
                                               int position,
                                               MiniCPMKVCache& kv_cache);

    std::vector<float> encode_feature_sequence(const std::vector<float>& feat, int seq_len);

    VoxCPMConfig config_;
    MiniCPMModel base_lm_;
    MiniCPMModel residual_lm_;
    LocEncModel feat_encoder_;
    LocDiTModel feat_decoder_estimator_;
    FSQ fsq_layer_;
    std::unique_ptr<VoxCPMComponents> components_;
    std::unique_ptr<UnifiedCFM> feat_decoder_;
    VoxCPMBackend* backend_ = nullptr;
    VoxCPMImatrixCollector* imatrix_collector_ = nullptr;
    std::shared_ptr<VoxCPMWeightStore> weight_store_;
};

}  // namespace voxcpm

#endif  // VOXCPM_VOXCPM_H
