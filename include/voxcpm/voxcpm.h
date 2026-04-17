#ifndef VOXCPM_VOXCPM_H
#define VOXCPM_VOXCPM_H

#include "voxcpm/components.h"
#include "voxcpm/config.h"
#include "voxcpm/context.h"
#include "voxcpm/fsq.h"
#include "voxcpm/localenc.h"
#include "voxcpm/locdit.h"
#include "voxcpm/minicpm.h"
#include "voxcpm/output.h"
#include "voxcpm/state.h"
#include "voxcpm/unified_cfm.h"

#include <array>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace voxcpm {

class VoxCPMBackend;
class VoxCPMImatrixCollector;
class VoxCPMWeightStore;

struct VoxCPMCachedGraph {
    std::unique_ptr<VoxCPMContext> context;
    ggml_cgraph* graph = nullptr;
    ggml_tensor* input0 = nullptr;
    ggml_tensor* input1 = nullptr;
    ggml_tensor* input2 = nullptr;
    ggml_tensor* input3 = nullptr;
    ggml_tensor* input4 = nullptr;
    ggml_tensor* output = nullptr;
    ggml_tensor* output_aux0 = nullptr;
    std::vector<float> aux_input4;

    void clear() {
        context.reset();
        graph = nullptr;
        input0 = nullptr;
        input1 = nullptr;
        input2 = nullptr;
        input3 = nullptr;
        input4 = nullptr;
        output = nullptr;
        output_aux0 = nullptr;
        aux_input4.clear();
    }
};

struct VoxCPMDecodeState {
    std::unique_ptr<MiniCPMKVCache> base_lm_cache;
    std::unique_ptr<MiniCPMKVCache> residual_lm_cache;
    std::unique_ptr<VoxCPMPersistentState> persistent_state;
    std::unique_ptr<VoxCPMOutputPool> output_pool;
    std::vector<float> lm_hidden;
    std::vector<float> residual_hidden;
    int current_position = 0;
    int audio_frame_count = 0;
    std::vector<float> prefix_feat_cond;  // [patch_size, feat_dim] in patch-major order
    int streaming_prefix_len = 4;
    VoxCPMCachedGraph base_lm_step_graph;
    VoxCPMCachedGraph residual_lm_step_graph;
    int base_lm_step_graph_position = -1;
    int residual_lm_step_graph_position = -1;

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

struct VoxCPMDecodeOptions {
    bool export_patch_to_host = true;
    bool publish_stop_logits_to_output = true;
    bool publish_patch_to_output = true;
    bool trust_persistent_state = false;
};

struct VoxCPMPrefillTensorInputs {
    ggml_tensor* text_src = nullptr;
    ggml_tensor* text_mask_src = nullptr;
    ggml_tensor* feat_src = nullptr;
    ggml_tensor* feat_mask_src = nullptr;
    std::vector<int32_t> prompt_positions;
    int seq_len = 0;
    int streaming_prefix_len = 4;
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
    void reset_request_state();

    VoxCPMDecodeState prefill(const std::vector<int32_t>& text,
                              const std::vector<int32_t>& text_mask,
                              const std::vector<float>& feat,
                              const std::vector<int32_t>& feat_mask,
                              int seq_len,
                              int streaming_prefix_len = 4);
    VoxCPMDecodeState prefill_with_feature_tensor(const std::vector<int32_t>& text,
                                                  const std::vector<int32_t>& text_mask,
                                                  ggml_tensor* feat_src,
                                                  const std::vector<int32_t>& feat_mask,
                                                  int seq_len,
                                                  int streaming_prefix_len = 4);
    VoxCPMDecodeState prefill_with_input_tensors(ggml_tensor* text_src,
                                                 ggml_tensor* text_mask_src,
                                                 ggml_tensor* feat_src,
                                                 ggml_tensor* feat_mask_src,
                                                 int seq_len,
                                                 int streaming_prefix_len = 4);
    // Preferred backend-resident prefill module entry. Callers can pass
    // backend-owned inputs plus explicit prompt span metadata so prompt
    // timeline control no longer depends on mirroring feat_mask to host.
    VoxCPMDecodeState prefill_from_tensor_inputs(const VoxCPMPrefillTensorInputs& inputs);
    // Convenience wrapper for callers that already have the explicit prompt
    // positions but are not yet using the typed module entry above.
    VoxCPMDecodeState prefill_with_input_tensors_and_prompt_positions(ggml_tensor* text_src,
                                                                      ggml_tensor* text_mask_src,
                                                                      ggml_tensor* feat_src,
                                                                      ggml_tensor* feat_mask_src,
                                                                      const std::vector<int32_t>& prompt_positions,
                                                                      int seq_len,
                                                                      int streaming_prefix_len = 4);
    VoxCPMDecodeState prefill_with_prompt_patch_tensor(const std::vector<int32_t>& text,
                                                       const std::vector<int32_t>& text_mask,
                                                       const std::vector<float>& feat,
                                                       ggml_tensor* prompt_patch_src,
                                                       const std::vector<int32_t>& feat_mask,
                                                       int seq_len,
                                                       int streaming_prefix_len = 4);

    VoxCPMDecodeResult decode(VoxCPMDecodeState state,
                              const std::vector<float>& z,
                              int inference_timesteps = 10,
                              float cfg_value = 2.0f,
                              const VoxCPMDecodeOptions& options = {});

    // Benchmark helpers exposing stable module boundaries without changing
    // inference semantics.
    std::vector<float> benchmark_encode_feature_sequence(const std::vector<float>& feat, int seq_len);
    std::vector<float> benchmark_run_locenc_sequence_to_lm_projection(const std::vector<float>& feat, int seq_len);
    std::vector<float> benchmark_run_locenc_sequence_to_lm_projection_from_tensor(ggml_tensor* feat_src, int seq_len);
    std::vector<float> benchmark_run_locenc_sequence_to_lm_projection_fsq(const std::vector<float>& feat, int seq_len);
    std::vector<float> benchmark_run_locenc_sequence_to_lm_projection_fsq_from_tensor(ggml_tensor* feat_src,
                                                                                       int seq_len);
    std::vector<float> benchmark_run_embedding_masked_locenc_sequence_to_lm_projection(
        const std::vector<int32_t>& token_ids,
        const std::vector<float>& feat,
        const std::vector<float>& text_mask,
        const std::vector<float>& feat_mask,
        int seq_len);
    std::vector<float> benchmark_run_embedding_masked_locenc_sequence_to_lm_projection_fsq(
        const std::vector<int32_t>& token_ids,
        const std::vector<float>& feat,
        const std::vector<float>& text_mask,
        const std::vector<float>& feat_mask,
        int seq_len);
    std::vector<float> benchmark_run_embedding_masked_locenc_sequence_to_lm_projection_from_tensors(
        ggml_tensor* token_src,
        ggml_tensor* feat_src,
        ggml_tensor* text_mask_src,
        ggml_tensor* feat_mask_src,
        int seq_len);
    std::vector<float> benchmark_run_embedding_masked_locenc_sequence_to_lm_projection_fsq_from_tensors(
        ggml_tensor* token_src,
        ggml_tensor* feat_src,
        ggml_tensor* text_mask_src,
        ggml_tensor* feat_mask_src,
        int seq_len);
    std::vector<float> benchmark_run_embedding(const std::vector<int32_t>& token_ids);
    std::vector<float> benchmark_run_embedding_from_tensor(ggml_tensor* token_src, int token_count);
    std::vector<float> benchmark_run_enc_to_lm_projection(const std::vector<float>& input, int seq_len);
    std::vector<float> benchmark_run_enc_to_lm_projection_from_tensor(ggml_tensor* input_src, int seq_len);
    std::vector<float> benchmark_run_enc_to_lm_projection_fsq(const std::vector<float>& input, int seq_len);
    std::vector<float> benchmark_run_enc_to_lm_projection_fsq_from_tensor(ggml_tensor* input_src, int seq_len);
    std::vector<float> benchmark_run_masked_fsq_blend(const std::vector<float>& input,
                                                      const std::vector<float>& text_mask,
                                                      const std::vector<float>& feat_mask,
                                                      int seq_len);
    std::pair<std::vector<float>, std::vector<float>> benchmark_run_prefill_base_to_residual_inputs(
        const std::vector<float>& combined_embed,
        const std::vector<float>& text_mask,
        const std::vector<float>& feat_mask,
        int seq_len,
        MiniCPMKVCache& kv_cache,
        bool is_causal = true);
    std::pair<std::vector<float>, std::vector<float>> benchmark_run_prefill_inputs_to_residual_inputs(
        const std::vector<int32_t>& token_ids,
        const std::vector<float>& feat,
        const std::vector<float>& text_mask,
        const std::vector<float>& feat_mask,
        int seq_len,
        MiniCPMKVCache& kv_cache,
        bool is_causal = true);
    std::pair<std::vector<float>, std::vector<float>> benchmark_run_prefill_inputs_to_hidden_states(
        const std::vector<int32_t>& token_ids,
        const std::vector<float>& feat,
        const std::vector<float>& text_mask,
        const std::vector<float>& feat_mask,
        int seq_len,
        MiniCPMKVCache& base_kv_cache,
        MiniCPMKVCache& residual_kv_cache,
        bool is_causal = true);
    VoxCPMDecodeState benchmark_stage_prefill_prompt_timeline(const std::vector<float>& prompt_patches,
                                                              int frame_count,
                                                              bool materialize_prefix_host_shadow = true);
    VoxCPMDecodeState benchmark_stage_prefill_prompt_timeline_from_tensor(ggml_tensor* prompt_patch_src,
                                                                          int frame_count,
                                                                          bool materialize_prefix_host_shadow = true);
    std::vector<float> benchmark_run_lm_to_dit_projection(const std::vector<float>& input);
    std::vector<float> benchmark_run_lm_to_dit_projection_from_state(const VoxCPMDecodeState& state);
    std::vector<float> benchmark_run_res_to_dit_projection(const std::vector<float>& input);
    std::vector<float> benchmark_run_res_to_dit_projection_from_state(const VoxCPMDecodeState& state);
    std::vector<float> benchmark_run_fsq_2d(const std::vector<float>& input, int seq_len);
    std::vector<float> benchmark_run_fsq_from_state(const VoxCPMDecodeState& state);
    std::vector<float> benchmark_run_base_lm_forward(const std::vector<float>& input,
                                                     int seq_len,
                                                     MiniCPMKVCache& kv_cache,
                                                     bool is_causal = true);
    std::vector<float> benchmark_run_residual_lm_forward(const std::vector<float>& input,
                                                         int seq_len,
                                                         MiniCPMKVCache& kv_cache,
                                                         bool is_causal = true);
    std::vector<float> benchmark_run_residual_lm_forward_last_hidden(const std::vector<float>& input,
                                                                     int seq_len,
                                                                     MiniCPMKVCache& kv_cache,
                                                                     bool is_causal = true);
    std::vector<float> benchmark_run_unified_cfm(const std::vector<float>& z,
                                                 const std::vector<float>& mu,
                                                 const std::vector<float>& cond,
                                                 int n_timesteps,
                                                 float cfg_value);
    std::vector<float> benchmark_run_stop_predictor(const std::vector<float>& input);
    std::array<float, 2> benchmark_run_stop_predictor_from_state(const VoxCPMDecodeState& state,
                                                                 bool publish_to_output_pool = false);
    std::vector<float> benchmark_run_locenc_patch(const std::vector<float>& patch);
    std::vector<float> benchmark_run_locenc_patch_to_lm_embed(const std::vector<float>& patch);
    std::vector<float> benchmark_run_locenc_patch_from_output_pool(const VoxCPMDecodeState& state,
                                                                   int frame_index);
    std::vector<float> benchmark_run_locenc_patch_to_lm_embed_from_output_pool(const VoxCPMDecodeState& state,
                                                                               int frame_index);
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
    std::pair<std::vector<float>, std::vector<float>> benchmark_run_decode_front_half_with_curr_embed(
        const std::vector<float>& z,
        const std::vector<float>& lm_hidden,
        const std::vector<float>& residual_hidden,
        const std::vector<float>& prefix_feat_cond,
        int inference_timesteps,
        float cfg_value);
    std::pair<std::vector<float>, std::vector<float>> benchmark_run_decode_front_half_from_state(
        const std::vector<float>& z,
        const VoxCPMDecodeState& state,
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
    void clear_cached_graphs();
    VoxCPMCachedGraph& ensure_locenc_patch_graph();
    VoxCPMCachedGraph& ensure_locenc_sequence_graph(int seq_len);
    VoxCPMCachedGraph& ensure_locenc_sequence_to_lm_projection_graph(int seq_len);
    VoxCPMCachedGraph& ensure_locenc_sequence_to_lm_projection_fsq_graph(int seq_len);
    VoxCPMCachedGraph& ensure_embedding_masked_locenc_sequence_to_lm_projection_graph(int seq_len);
    VoxCPMCachedGraph& ensure_embedding_masked_locenc_sequence_to_lm_projection_fsq_graph(int seq_len);
    VoxCPMCachedGraph& ensure_masked_fsq_blend_graph(int seq_len);
    VoxCPMCachedGraph& ensure_embedding_graph(int token_count);
    VoxCPMCachedGraph& ensure_enc_to_lm_projection_graph(int seq_len);
    VoxCPMCachedGraph& ensure_enc_to_lm_projection_fsq_graph(int seq_len);
    VoxCPMCachedGraph& ensure_fsq_2d_graph(int seq_len);
    VoxCPMCachedGraph& ensure_unified_cfm_graph(int n_timesteps, float cfg_value);
    VoxCPMCachedGraph& ensure_decode_front_half_graph(int n_timesteps, float cfg_value);
    VoxCPMCachedGraph& ensure_state_base_lm_step_graph(VoxCPMDecodeState& state, int position);
    VoxCPMCachedGraph& ensure_state_residual_lm_step_graph(VoxCPMDecodeState& state, int position);
    VoxCPMCachedGraph& ensure_stop_predictor_graph();
    VoxCPMCachedGraph& ensure_locenc_patch_to_lm_embed_graph();

    void run_locenc_patch_into(const float* patch_data, float* output_data);
    std::vector<float> run_locenc_patch(const float* patch_data);
    std::vector<float> run_locenc_patch_from_tensor(ggml_tensor* patch_src);
    std::vector<float> run_locenc_patch_to_lm_embed_from_tensor(ggml_tensor* patch_src);
    std::vector<float> run_embedding(const std::vector<int32_t>& token_ids);
    std::vector<float> run_embedding_from_tensor(ggml_tensor* token_src, int token_count);
    std::vector<float> run_projection_1d(LinearProjection& projection,
                                         const std::vector<float>& input,
                                         int in_dim,
                                         int out_dim);
    std::vector<float> run_projection_1d_from_tensor(LinearProjection& projection,
                                                     ggml_tensor* input_src,
                                                     int in_dim,
                                                     int out_dim);
    std::vector<float> run_projection_2d(LinearProjection& projection,
                                         const std::vector<float>& input,
                                         int in_dim,
                                         int seq_len,
                                         int out_dim);
    std::vector<float> run_stop_predictor(const std::vector<float>& input);
    std::vector<float> run_fsq_1d(const std::vector<float>& input);
    std::vector<float> run_fsq_1d_from_tensor(ggml_tensor* input_src);
    std::vector<float> run_fsq_2d(const std::vector<float>& input, int seq_len);
    std::vector<float> run_masked_fsq_blend(const std::vector<float>& input,
                                            const std::vector<float>& text_mask,
                                            const std::vector<float>& feat_mask,
                                            int seq_len);
    void run_embedding_masked_locenc_sequence_to_lm_projection_into(const std::vector<int32_t>& token_ids,
                                                                    const std::vector<float>& feat,
                                                                    const std::vector<float>& text_mask,
                                                                    const std::vector<float>& feat_mask,
                                                                    int seq_len,
                                                                    ggml_tensor* output_dst);
    void run_embedding_masked_locenc_sequence_to_lm_projection_from_feat_tensor_into(
        const std::vector<int32_t>& token_ids,
        ggml_tensor* feat_src,
        const std::vector<float>& text_mask,
        const std::vector<float>& feat_mask,
        int seq_len,
        ggml_tensor* output_dst);
    void run_embedding_masked_locenc_sequence_to_lm_projection_from_input_tensors_into(
        ggml_tensor* token_src,
        ggml_tensor* feat_src,
        ggml_tensor* text_mask_src,
        ggml_tensor* feat_mask_src,
        int seq_len,
        ggml_tensor* output_dst);
    std::pair<std::vector<float>, std::vector<float>> run_prefill_base_to_residual_inputs(
        const std::vector<float>& combined_embed,
        const std::vector<float>& text_mask,
        const std::vector<float>& feat_mask,
        int seq_len,
        MiniCPMKVCache& kv_cache,
        bool is_causal);
    std::vector<float> run_prefill_base_to_residual_inputs_from_tensor_with_last_hidden_into(
        ggml_tensor* combined_embed_src,
        const std::vector<float>& text_mask,
        const std::vector<float>& feat_mask,
        int seq_len,
        MiniCPMKVCache& kv_cache,
        bool is_causal,
        ggml_tensor* lm_hidden_dst);
    void run_prefill_hidden_states_from_tensor_into(ggml_tensor* combined_embed_src,
                                                    const std::vector<float>& text_mask,
                                                    const std::vector<float>& feat_mask,
                                                    int seq_len,
                                                    MiniCPMKVCache& base_kv_cache,
                                                    MiniCPMKVCache& residual_kv_cache,
                                                    bool is_causal,
                                                    ggml_tensor* lm_hidden_dst,
                                                    ggml_tensor* residual_hidden_dst);
    void run_prefill_hidden_states_from_input_tensors_into(ggml_tensor* combined_embed_src,
                                                           ggml_tensor* text_mask_src,
                                                           ggml_tensor* feat_mask_src,
                                                           int seq_len,
                                                           MiniCPMKVCache& base_kv_cache,
                                                           MiniCPMKVCache& residual_kv_cache,
                                                           bool is_causal,
                                                           ggml_tensor* lm_hidden_dst,
                                                           ggml_tensor* residual_hidden_dst);
    void finalize_prefill_prompt_timeline(VoxCPMDecodeState& state,
                                          int audio_frame_count,
                                          bool materialize_prefix_host_shadow,
                                          const float* last_prompt_patch_fallback);
    std::vector<float> run_prefill_base_to_residual_inputs_with_last_hidden_into(
        const std::vector<float>& combined_embed,
        const std::vector<float>& text_mask,
        const std::vector<float>& feat_mask,
        int seq_len,
        MiniCPMKVCache& kv_cache,
        bool is_causal,
        ggml_tensor* lm_hidden_dst);
    std::vector<float> run_minicpm_forward(MiniCPMModel& model,
                                           const std::vector<float>& input,
                                           int seq_len,
                                           MiniCPMKVCache& kv_cache,
                                           bool is_causal);
    void run_minicpm_forward_last_hidden_into(MiniCPMModel& model,
                                              const std::vector<float>& input,
                                              int seq_len,
                                              MiniCPMKVCache& kv_cache,
                                              bool is_causal,
                                              ggml_tensor* output_dst);
    std::vector<float> run_minicpm_forward_last_hidden(MiniCPMModel& model,
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
                               std::vector<float>& output_0,
                               std::vector<float>* curr_embed = nullptr);
    VoxCPMCachedGraph& run_decode_front_half_graph(const std::vector<float>& z,
                                                   const VoxCPMDecodeState& state,
                                                   int inference_timesteps,
                                                   float cfg_value);
    std::array<float, 2> run_stop_predictor_from_state(const VoxCPMDecodeState& state,
                                                       bool publish_to_output_pool = true);
    std::vector<float> run_locenc_patch_to_lm_embed(const std::vector<float>& patch);
    std::vector<float> run_base_lm_decode_step(const std::vector<float>& curr_embed,
                                               int position,
                                               MiniCPMKVCache& kv_cache);

    std::vector<float> encode_feature_sequence(const std::vector<float>& feat, int seq_len);
    void sync_host_state_to_persistent(VoxCPMDecodeState& state) const;
    void sync_persistent_to_host(VoxCPMDecodeState& state, bool include_prefix_patch = true) const;
    bool should_precompute_cfm_time_table(int n_timesteps) const;
    const std::vector<float>& get_precomputed_cfm_time_table(int n_timesteps);
    VoxCPMDecodeState create_decode_state_internal(bool initialize_host_shadow) const;
    VoxCPMDecodeState prefill_impl(const std::vector<int32_t>& text,
                                   const std::vector<int32_t>& text_mask,
                                   const std::vector<float>* feat,
                                   ggml_tensor* feat_src,
                                   ggml_tensor* prompt_patch_src,
                                   const std::vector<int32_t>& feat_mask,
                                   int seq_len,
                                   int streaming_prefix_len);
    VoxCPMDecodeState prefill_impl_from_input_tensors_with_prompt_positions(
        ggml_tensor* text_src,
        ggml_tensor* text_mask_src,
        ggml_tensor* feat_src,
        ggml_tensor* feat_mask_src,
        const std::vector<int32_t>& prompt_positions,
        int seq_len,
        int streaming_prefix_len);
    std::vector<int32_t> derive_prompt_positions_from_feat_mask_tensor(ggml_tensor* feat_mask_src,
                                                                       int seq_len) const;

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
    VoxCPMCachedGraph locenc_patch_graph_;
    std::unordered_map<int, VoxCPMCachedGraph> locenc_sequence_graphs_;
    std::unordered_map<int, VoxCPMCachedGraph> locenc_sequence_to_lm_projection_graphs_;
    std::unordered_map<int, VoxCPMCachedGraph> locenc_sequence_to_lm_projection_fsq_graphs_;
    std::unordered_map<int, VoxCPMCachedGraph> embedding_masked_locenc_sequence_to_lm_projection_graphs_;
    std::unordered_map<int, VoxCPMCachedGraph> embedding_masked_locenc_sequence_to_lm_projection_fsq_graphs_;
    std::unordered_map<int, VoxCPMCachedGraph> masked_fsq_blend_graphs_;
    std::unordered_map<int, VoxCPMCachedGraph> embedding_graphs_;
    std::unordered_map<int, VoxCPMCachedGraph> enc_to_lm_projection_graphs_;
    std::unordered_map<int, VoxCPMCachedGraph> enc_to_lm_projection_fsq_graphs_;
    std::unordered_map<int, VoxCPMCachedGraph> fsq_2d_graphs_;
    std::unordered_map<std::string, VoxCPMCachedGraph> unified_cfm_graphs_;
    std::unordered_map<std::string, VoxCPMCachedGraph> decode_front_half_graphs_;
    std::unordered_map<int, std::vector<float>> precomputed_cfm_time_tables_;
    VoxCPMCachedGraph stop_predictor_graph_;
    VoxCPMCachedGraph locenc_patch_to_lm_embed_graph_;
};

}  // namespace voxcpm

#endif  // VOXCPM_VOXCPM_H
