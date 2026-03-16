#include "voxcpm/voxcpm.h"

#include "voxcpm/backend.h"
#include "voxcpm/imatrix.h"
#include "voxcpm/weight-store.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace voxcpm {

namespace {

VoxCPMContext make_graph_ctx(int n_tensors, int max_nodes) {
    return VoxCPMContext(ContextType::Graph, n_tensors, max_nodes);
}

std::vector<float> slice_column_major_2d(const std::vector<float>& input,
                                         int row_dim,
                                         int col_idx) {
    std::vector<float> out(static_cast<size_t>(row_dim));
    const size_t offset = static_cast<size_t>(col_idx) * row_dim;
    std::copy_n(input.data() + offset, row_dim, out.data());
    return out;
}

void assign_column_major_2d(std::vector<float>& output,
                            const std::vector<float>& column,
                            int row_dim,
                            int col_idx) {
    const size_t offset = static_cast<size_t>(col_idx) * row_dim;
    std::copy_n(column.data(), row_dim, output.data() + offset);
}

}  // namespace

bool VoxCPMRuntime::update_config_from_gguf(const std::string& gguf_path) {
    auto store = std::make_shared<VoxCPMWeightStore>();
    if (!store->load_from_file(gguf_path, *backend_)) {
        return false;
    }
    return update_config_from_store(*store);
}

bool VoxCPMRuntime::update_config_from_store(const VoxCPMWeightStore& store) {
    uint32_t u32 = 0;
    float f32 = 0.0f;
    const bool has_patch_size = store.get_u32("voxcpm_patch_size", u32);
    if (has_patch_size) {
        config_.patch_size = static_cast<int>(u32);
    }
    const bool has_feat_dim = store.get_u32("voxcpm_feat_dim", u32);
    if (has_feat_dim) {
        config_.feat_dim = static_cast<int>(u32);
    }
    const bool has_max_length = store.get_u32("voxcpm_max_length", u32);
    if (has_max_length) {
        config_.max_length = static_cast<int>(u32);
    }
    const bool has_residual_layers = store.get_u32("voxcpm_residual_lm_num_layers", u32);
    if (has_residual_layers) {
        config_.residual_lm.n_layer = static_cast<int>(u32);
    }
    const bool has_sigma_min = store.get_f32("voxcpm_dit_config_cfm_config_sigma_min", f32);
    if (has_sigma_min) {
        config_.loc_dit.sigma_min = f32;
    }
    const bool has_cfg_rate = store.get_f32("voxcpm_dit_config_cfm_config_inference_cfg_rate", f32);
    if (has_cfg_rate) {
        config_.loc_dit.cfg_rate = f32;
    }

    return has_patch_size && has_feat_dim && has_max_length && has_residual_layers && has_sigma_min && has_cfg_rate;
}

bool VoxCPMRuntime::load_from_gguf(const std::string& gguf_path,
                                   VoxCPMContext& weight_ctx,
                                   VoxCPMContext& graph_ctx,
                                   VoxCPMBackend& backend) {
    VOXCPM_UNUSED(weight_ctx);
    VOXCPM_UNUSED(graph_ctx);

    backend_ = &backend;
    weight_store_ = std::make_shared<VoxCPMWeightStore>();
    if (!weight_store_->load_from_file(gguf_path, backend)) {
        return false;
    }

    return load_from_store(weight_store_, backend);
}

bool VoxCPMRuntime::load_from_store(const std::shared_ptr<VoxCPMWeightStore>& store,
                                    VoxCPMBackend& backend) {
    backend_ = &backend;
    weight_store_ = store;
    if (!weight_store_) {
        return false;
    }

    if (!update_config_from_store(*weight_store_)) {
        return false;
    }

    if (!base_lm_.load_from_store(weight_store_, "", backend)) {
        return false;
    }
    if (!residual_lm_.load_from_store(weight_store_, "residual_lm", backend)) {
        return false;
    }
    if (!feat_encoder_.load_from_store(weight_store_, backend)) {
        return false;
    }
    if (!feat_decoder_estimator_.load_from_store(weight_store_, backend)) {
        return false;
    }
    if (!fsq_layer_.load_from_store(weight_store_)) {
        return false;
    }

    const float scale_emb = base_lm_.config().use_mup ? static_cast<float>(base_lm_.config().scale_emb) : 1.0f;
    components_ = VoxCPMComponents::from_store(weight_store_,
                                               base_lm_.config().hidden_size,
                                               base_lm_.config().vocab_size,
                                               scale_emb);
    if (!components_) {
        return false;
    }

    config_.base_lm = base_lm_.config();
    config_.residual_lm = residual_lm_.config();
    config_.loc_enc.hidden_size = feat_encoder_.config().hidden_size;
    config_.loc_enc.n_layer = feat_encoder_.config().n_layer;
    config_.loc_enc.n_heads = feat_encoder_.config().n_heads;
    config_.loc_enc.n_kv_heads = feat_encoder_.config().n_kv_heads;
    config_.loc_enc.intermediate_size = feat_encoder_.config().intermediate_size;
    config_.loc_enc.feat_dim = feat_encoder_.feat_dim();
    config_.loc_dit.hidden_size = feat_decoder_estimator_.config().hidden_size;
    config_.loc_dit.n_layer = feat_decoder_estimator_.config().n_layer;
    config_.loc_dit.n_heads = feat_decoder_estimator_.config().n_heads;
    config_.loc_dit.n_kv_heads = feat_decoder_estimator_.config().n_kv_heads;
    config_.loc_dit.intermediate_size = feat_decoder_estimator_.config().intermediate_size;
    config_.loc_dit.feat_dim = feat_decoder_estimator_.feat_dim();
    config_.fsq = fsq_layer_.config();

    CFMConfig cfm_config;
    cfm_config.sigma_min = config_.loc_dit.sigma_min;
    cfm_config.inference_cfg_rate = config_.loc_dit.cfg_rate;
    feat_decoder_ = std::make_unique<UnifiedCFM>(feat_decoder_estimator_, cfm_config);
    return true;
}

void VoxCPMRuntime::maybe_collect_graph(ggml_cgraph* graph) {
    if (imatrix_collector_ && backend_ && graph) {
        imatrix_collector_->observe_graph(graph, *backend_);
    }
}

VoxCPMDecodeState VoxCPMRuntime::create_decode_state() const {
    VOXCPM_ASSERT(backend_ != nullptr);

    VoxCPMDecodeState state;
    state.base_lm_cache = std::make_unique<MiniCPMKVCache>(base_lm_.config().n_layer,
                                                           base_lm_.config().n_kv_heads,
                                                           config_.max_length,
                                                           base_lm_.config().head_dim());
    state.residual_lm_cache = std::make_unique<MiniCPMKVCache>(residual_lm_.config().n_layer,
                                                               residual_lm_.config().n_kv_heads,
                                                               config_.max_length,
                                                               residual_lm_.config().head_dim());
    state.base_lm_cache->init(*backend_);
    state.residual_lm_cache->init(*backend_);
    state.lm_hidden.assign(static_cast<size_t>(base_lm_.config().hidden_size), 0.0f);
    state.residual_hidden.assign(static_cast<size_t>(residual_lm_.config().hidden_size), 0.0f);
    state.prefix_feat_cond.assign(static_cast<size_t>(config_.patch_size * config_.feat_dim), 0.0f);
    return state;
}

std::vector<float> VoxCPMRuntime::run_locenc_patch(const float* patch_data) {
    VOXCPM_ASSERT(backend_ != nullptr);

    VoxCPMContext graph_ctx = make_graph_ctx(8192, 65536);
    ggml_tensor* input = graph_ctx.new_tensor_2d(GGML_TYPE_F32, config_.feat_dim, config_.patch_size);
    ggml_set_input(input);

    ggml_tensor* output = feat_encoder_.forward_patch(graph_ctx, input);
    ggml_set_output(output);

    ggml_cgraph* graph = graph_ctx.new_graph();
    graph_ctx.build_forward(graph, output);
    backend_->reserve_compute_memory(graph, "runtime.locenc.patch");
    backend_->alloc_graph(graph, "runtime.locenc.patch");
    backend_->tensor_set(input, patch_data, 0, static_cast<size_t>(config_.feat_dim * config_.patch_size) * sizeof(float));
    VOXCPM_ASSERT(backend_->compute(graph) == GGML_STATUS_SUCCESS);
    maybe_collect_graph(graph);

    std::vector<float> out(static_cast<size_t>(base_lm_.config().hidden_size));
    backend_->tensor_get(output, out.data(), 0, out.size() * sizeof(float));
    return out;
}

std::vector<float> VoxCPMRuntime::encode_feature_sequence(const std::vector<float>& feat, int seq_len) {
    const size_t patch_elems = static_cast<size_t>(config_.patch_size * config_.feat_dim);
    std::vector<float> encoded(static_cast<size_t>(base_lm_.config().hidden_size) * seq_len);
    for (int t = 0; t < seq_len; ++t) {
        const float* patch_ptr = feat.data() + static_cast<size_t>(t) * patch_elems;
        const std::vector<float> hidden = run_locenc_patch(patch_ptr);
        assign_column_major_2d(encoded, hidden, base_lm_.config().hidden_size, t);
    }
    return encoded;
}

std::vector<float> VoxCPMRuntime::run_embedding(const std::vector<int32_t>& token_ids) {
    VOXCPM_ASSERT(backend_ != nullptr);

    VoxCPMContext graph_ctx = make_graph_ctx(4096, 32768);
    ggml_tensor* input = graph_ctx.new_tensor_1d(GGML_TYPE_I32, static_cast<int64_t>(token_ids.size()));
    ggml_set_input(input);

    ggml_tensor* output = components_->embed_tokens()->forward(graph_ctx, input);
    ggml_set_output(output);

    ggml_cgraph* graph = graph_ctx.new_graph();
    graph_ctx.build_forward(graph, output);
    backend_->reserve_compute_memory(graph, "runtime.embedding");
    backend_->alloc_graph(graph, "runtime.embedding");
    backend_->tensor_set(input, token_ids.data(), 0, token_ids.size() * sizeof(int32_t));
    VOXCPM_ASSERT(backend_->compute(graph) == GGML_STATUS_SUCCESS);
    maybe_collect_graph(graph);

    std::vector<float> out(static_cast<size_t>(components_->embed_tokens()->config().hidden_dim) * token_ids.size());
    backend_->tensor_get(output, out.data(), 0, out.size() * sizeof(float));
    return out;
}

std::vector<float> VoxCPMRuntime::run_projection_1d(LinearProjection& projection,
                                                    const std::vector<float>& input,
                                                    int in_dim,
                                                    int out_dim) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(static_cast<int>(input.size()) == in_dim);

    VoxCPMContext graph_ctx = make_graph_ctx(4096, 32768);
    ggml_tensor* input_tensor = graph_ctx.new_tensor_1d(GGML_TYPE_F32, in_dim);
    ggml_set_input(input_tensor);

    ggml_tensor* output = projection.forward(graph_ctx, input_tensor);
    ggml_set_output(output);

    ggml_cgraph* graph = graph_ctx.new_graph();
    graph_ctx.build_forward(graph, output);
    backend_->reserve_compute_memory(graph, "runtime.proj.1d");
    backend_->alloc_graph(graph, "runtime.proj.1d");
    backend_->tensor_set(input_tensor, input.data(), 0, input.size() * sizeof(float));
    VOXCPM_ASSERT(backend_->compute(graph) == GGML_STATUS_SUCCESS);
    maybe_collect_graph(graph);

    std::vector<float> out(static_cast<size_t>(out_dim));
    backend_->tensor_get(output, out.data(), 0, out.size() * sizeof(float));
    return out;
}

std::vector<float> VoxCPMRuntime::run_projection_2d(LinearProjection& projection,
                                                    const std::vector<float>& input,
                                                    int in_dim,
                                                    int seq_len,
                                                    int out_dim) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(static_cast<int>(input.size()) == in_dim * seq_len);

    VoxCPMContext graph_ctx = make_graph_ctx(4096, 32768);
    ggml_tensor* input_tensor = graph_ctx.new_tensor_2d(GGML_TYPE_F32, in_dim, seq_len);
    ggml_set_input(input_tensor);

    ggml_tensor* output = projection.forward(graph_ctx, input_tensor);
    ggml_set_output(output);

    ggml_cgraph* graph = graph_ctx.new_graph();
    graph_ctx.build_forward(graph, output);
    backend_->reserve_compute_memory(graph, "runtime.proj.2d");
    backend_->alloc_graph(graph, "runtime.proj.2d");
    backend_->tensor_set(input_tensor, input.data(), 0, input.size() * sizeof(float));
    VOXCPM_ASSERT(backend_->compute(graph) == GGML_STATUS_SUCCESS);
    maybe_collect_graph(graph);

    std::vector<float> out(static_cast<size_t>(out_dim) * seq_len);
    backend_->tensor_get(output, out.data(), 0, out.size() * sizeof(float));
    return out;
}

std::vector<float> VoxCPMRuntime::run_stop_predictor(const std::vector<float>& input) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(static_cast<int>(input.size()) == base_lm_.config().hidden_size);

    VoxCPMContext graph_ctx = make_graph_ctx(4096, 32768);
    ggml_tensor* input_tensor = graph_ctx.new_tensor_1d(GGML_TYPE_F32, base_lm_.config().hidden_size);
    ggml_set_input(input_tensor);

    ggml_tensor* output = components_->stop_token()->forward(graph_ctx, input_tensor);
    ggml_set_output(output);

    ggml_cgraph* graph = graph_ctx.new_graph();
    graph_ctx.build_forward(graph, output);
    backend_->reserve_compute_memory(graph, "runtime.stop_predictor");
    backend_->alloc_graph(graph, "runtime.stop_predictor");
    backend_->tensor_set(input_tensor, input.data(), 0, input.size() * sizeof(float));
    VOXCPM_ASSERT(backend_->compute(graph) == GGML_STATUS_SUCCESS);
    maybe_collect_graph(graph);

    std::vector<float> out(2);
    backend_->tensor_get(output, out.data(), 0, out.size() * sizeof(float));
    return out;
}

std::vector<float> VoxCPMRuntime::run_fsq_1d(const std::vector<float>& input) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(static_cast<int>(input.size()) == base_lm_.config().hidden_size);

    VoxCPMContext graph_ctx = make_graph_ctx(4096, 32768);
    ggml_tensor* input_tensor = graph_ctx.new_tensor_2d(GGML_TYPE_F32, base_lm_.config().hidden_size, 1);
    ggml_set_input(input_tensor);

    ggml_tensor* output = fsq_layer_.forward(graph_ctx, input_tensor);
    output = ggml_reshape_1d(graph_ctx.raw_context(), output, output->ne[0]);
    ggml_set_output(output);

    ggml_cgraph* graph = graph_ctx.new_graph();
    graph_ctx.build_forward(graph, output);
    backend_->reserve_compute_memory(graph, "runtime.fsq.1d");
    backend_->alloc_graph(graph, "runtime.fsq.1d");
    backend_->tensor_set(input_tensor, input.data(), 0, input.size() * sizeof(float));
    VOXCPM_ASSERT(backend_->compute(graph) == GGML_STATUS_SUCCESS);
    maybe_collect_graph(graph);

    std::vector<float> out(static_cast<size_t>(base_lm_.config().hidden_size));
    backend_->tensor_get(output, out.data(), 0, out.size() * sizeof(float));
    return out;
}

std::vector<float> VoxCPMRuntime::run_fsq_2d(const std::vector<float>& input, int seq_len) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(static_cast<int>(input.size()) == base_lm_.config().hidden_size * seq_len);

    VoxCPMContext graph_ctx = make_graph_ctx(8192, 65536);
    ggml_tensor* input_tensor = graph_ctx.new_tensor_2d(GGML_TYPE_F32, base_lm_.config().hidden_size, seq_len);
    ggml_set_input(input_tensor);

    ggml_tensor* output = fsq_layer_.forward(graph_ctx, input_tensor);
    ggml_set_output(output);

    ggml_cgraph* graph = graph_ctx.new_graph();
    graph_ctx.build_forward(graph, output);
    backend_->reserve_compute_memory(graph, "runtime.fsq.2d");
    backend_->alloc_graph(graph, "runtime.fsq.2d");
    backend_->tensor_set(input_tensor, input.data(), 0, input.size() * sizeof(float));
    VOXCPM_ASSERT(backend_->compute(graph) == GGML_STATUS_SUCCESS);
    maybe_collect_graph(graph);

    std::vector<float> out(static_cast<size_t>(base_lm_.config().hidden_size) * seq_len);
    backend_->tensor_get(output, out.data(), 0, out.size() * sizeof(float));
    return out;
}

std::vector<float> VoxCPMRuntime::run_minicpm_forward(MiniCPMModel& model,
                                                      const std::vector<float>& input,
                                                      int seq_len,
                                                      MiniCPMKVCache& kv_cache,
                                                      bool is_causal) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(static_cast<int>(input.size()) == model.config().hidden_size * seq_len);

    VoxCPMContext graph_ctx = make_graph_ctx(32768, 262144);
    ggml_tensor* input_tensor = graph_ctx.new_tensor_2d(GGML_TYPE_F32, model.config().hidden_size, seq_len);
    ggml_set_input(input_tensor);

    ggml_tensor* output = model.forward(graph_ctx, input_tensor, nullptr, kv_cache, is_causal);
    ggml_set_output(output);

    ggml_cgraph* graph = graph_ctx.new_graph();
    graph_ctx.build_forward(graph, output);
    backend_->reserve_compute_memory(graph, "runtime.minicpm.forward");
    backend_->alloc_graph(graph, "runtime.minicpm.forward");
    backend_->tensor_set(input_tensor, input.data(), 0, input.size() * sizeof(float));
    VOXCPM_ASSERT(backend_->compute(graph) == GGML_STATUS_SUCCESS);
    maybe_collect_graph(graph);

    std::vector<float> out(static_cast<size_t>(model.config().hidden_size) * seq_len);
    backend_->tensor_get(output, out.data(), 0, out.size() * sizeof(float));
    return out;
}

std::vector<float> VoxCPMRuntime::run_minicpm_forward_step(MiniCPMModel& model,
                                                           const std::vector<float>& input,
                                                           int position,
                                                           MiniCPMKVCache& kv_cache,
                                                           bool is_causal) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(static_cast<int>(input.size()) == model.config().hidden_size);

    VoxCPMContext graph_ctx = make_graph_ctx(8192, 65536);
    ggml_tensor* input_tensor = graph_ctx.new_tensor_1d(GGML_TYPE_F32, model.config().hidden_size);
    ggml_tensor* positions_tensor = graph_ctx.new_tensor_1d(GGML_TYPE_I32, 1);
    ggml_set_input(input_tensor);
    ggml_set_input(positions_tensor);

    ggml_tensor* output = model.forward_step(graph_ctx, input_tensor, position, positions_tensor, kv_cache, is_causal);
    ggml_set_output(output);

    ggml_cgraph* graph = graph_ctx.new_graph();
    graph_ctx.build_forward(graph, output);
    backend_->reserve_compute_memory(graph, "runtime.minicpm.forward_step");
    backend_->alloc_graph(graph, "runtime.minicpm.forward_step");
    backend_->tensor_set(input_tensor, input.data(), 0, input.size() * sizeof(float));
    const int32_t position_value = position;
    backend_->tensor_set(positions_tensor, &position_value, 0, sizeof(position_value));
    VOXCPM_ASSERT(backend_->compute(graph) == GGML_STATUS_SUCCESS);
    maybe_collect_graph(graph);

    std::vector<float> out(static_cast<size_t>(model.config().hidden_size));
    backend_->tensor_get(output, out.data(), 0, out.size() * sizeof(float));
    return out;
}

std::vector<float> VoxCPMRuntime::run_unified_cfm(const std::vector<float>& z,
                                                  const std::vector<float>& mu,
                                                  const std::vector<float>& cond,
                                                  int n_timesteps,
                                                  float cfg_value) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(feat_decoder_ != nullptr);
    VOXCPM_ASSERT(static_cast<int>(z.size()) == config_.feat_dim * config_.patch_size);
    VOXCPM_ASSERT(static_cast<int>(mu.size()) == config_.loc_dit.hidden_size);
    VOXCPM_ASSERT(static_cast<int>(cond.size()) == config_.feat_dim * config_.patch_size);

    VoxCPMContext graph_ctx = make_graph_ctx(65536, 524288);
    ggml_tensor* z_tensor = graph_ctx.new_tensor_2d(GGML_TYPE_F32, config_.feat_dim, config_.patch_size);
    ggml_tensor* mu_tensor = graph_ctx.new_tensor_1d(GGML_TYPE_F32, config_.loc_dit.hidden_size);
    ggml_tensor* cond_tensor = graph_ctx.new_tensor_2d(GGML_TYPE_F32, config_.feat_dim, config_.patch_size);
    ggml_set_input(z_tensor);
    ggml_set_input(mu_tensor);
    ggml_set_input(cond_tensor);

    ggml_tensor* output = feat_decoder_->forward(graph_ctx,
                                                 z_tensor,
                                                 mu_tensor,
                                                 config_.patch_size,
                                                 cond_tensor,
                                                 n_timesteps,
                                                 cfg_value);
    ggml_set_output(output);

    ggml_cgraph* graph = graph_ctx.new_graph();
    graph_ctx.build_forward(graph, output);
    backend_->reserve_compute_memory(graph, "runtime.unified_cfm");
    backend_->alloc_graph(graph, "runtime.unified_cfm");
    backend_->tensor_set(z_tensor, z.data(), 0, z.size() * sizeof(float));
    backend_->tensor_set(mu_tensor, mu.data(), 0, mu.size() * sizeof(float));
    backend_->tensor_set(cond_tensor, cond.data(), 0, cond.size() * sizeof(float));
    VOXCPM_ASSERT(backend_->compute(graph) == GGML_STATUS_SUCCESS);
    maybe_collect_graph(graph);

    std::vector<float> out(static_cast<size_t>(config_.feat_dim * config_.patch_size));
    backend_->tensor_get(output, out.data(), 0, out.size() * sizeof(float));
    return out;
}

void VoxCPMRuntime::run_decode_front_half(const std::vector<float>& z,
                                          const std::vector<float>& lm_hidden,
                                          const std::vector<float>& residual_hidden,
                                          const std::vector<float>& prefix_feat_cond,
                                          int inference_timesteps,
                                          float cfg_value,
                                          std::vector<float>& output_0) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(components_ != nullptr);
    VOXCPM_ASSERT(feat_decoder_ != nullptr);
    VOXCPM_ASSERT(static_cast<int>(z.size()) == config_.feat_dim * config_.patch_size);
    VOXCPM_ASSERT(static_cast<int>(lm_hidden.size()) == base_lm_.config().hidden_size);
    VOXCPM_ASSERT(static_cast<int>(residual_hidden.size()) == residual_lm_.config().hidden_size);
    VOXCPM_ASSERT(static_cast<int>(prefix_feat_cond.size()) == config_.feat_dim * config_.patch_size);

    VoxCPMContext graph_ctx = make_graph_ctx(65536, 524288);
    ggml_tensor* z_tensor = graph_ctx.new_tensor_2d(GGML_TYPE_F32, config_.feat_dim, config_.patch_size);
    ggml_tensor* lm_hidden_tensor = graph_ctx.new_tensor_1d(GGML_TYPE_F32, base_lm_.config().hidden_size);
    ggml_tensor* residual_hidden_tensor = graph_ctx.new_tensor_1d(GGML_TYPE_F32, residual_lm_.config().hidden_size);
    ggml_tensor* cond_tensor = graph_ctx.new_tensor_2d(GGML_TYPE_F32, config_.feat_dim, config_.patch_size);
    ggml_set_input(z_tensor);
    ggml_set_input(lm_hidden_tensor);
    ggml_set_input(residual_hidden_tensor);
    ggml_set_input(cond_tensor);

    ggml_tensor* dit_hidden_1 = components_->lm_to_dit_proj()->forward(graph_ctx, lm_hidden_tensor);
    ggml_tensor* dit_hidden_2 = components_->res_to_dit_proj()->forward(graph_ctx, residual_hidden_tensor);
    ggml_tensor* dit_hidden = ggml_add(graph_ctx.raw_context(), dit_hidden_1, dit_hidden_2);
    ggml_tensor* patch = feat_decoder_->forward(graph_ctx,
                                                z_tensor,
                                                dit_hidden,
                                                config_.patch_size,
                                                cond_tensor,
                                                inference_timesteps,
                                                cfg_value);
    ggml_set_output(patch);

    ggml_cgraph* graph = graph_ctx.new_graph();
    graph_ctx.build_forward(graph, patch);
    backend_->reserve_compute_memory(graph, "runtime.decode_front_half");
    backend_->alloc_graph(graph, "runtime.decode_front_half");
    backend_->tensor_set(z_tensor, z.data(), 0, z.size() * sizeof(float));
    backend_->tensor_set(lm_hidden_tensor, lm_hidden.data(), 0, lm_hidden.size() * sizeof(float));
    backend_->tensor_set(residual_hidden_tensor, residual_hidden.data(), 0, residual_hidden.size() * sizeof(float));
    backend_->tensor_set(cond_tensor, prefix_feat_cond.data(), 0, prefix_feat_cond.size() * sizeof(float));
    VOXCPM_ASSERT(backend_->compute(graph) == GGML_STATUS_SUCCESS);
    maybe_collect_graph(graph);

    output_0.resize(static_cast<size_t>(config_.feat_dim * config_.patch_size));
    backend_->tensor_get(patch, output_0.data(), 0, output_0.size() * sizeof(float));
}

std::vector<float> VoxCPMRuntime::run_locenc_patch_to_lm_embed(const std::vector<float>& patch) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(components_ != nullptr);
    VOXCPM_ASSERT(static_cast<int>(patch.size()) == config_.feat_dim * config_.patch_size);

    VoxCPMContext graph_ctx = make_graph_ctx(16384, 131072);
    ggml_tensor* patch_tensor = graph_ctx.new_tensor_2d(GGML_TYPE_F32, config_.feat_dim, config_.patch_size);
    ggml_set_input(patch_tensor);

    ggml_tensor* hidden = feat_encoder_.forward_patch(graph_ctx, patch_tensor);
    ggml_tensor* embed = components_->enc_to_lm_proj()->forward(graph_ctx, hidden);
    ggml_set_output(embed);

    ggml_cgraph* graph = graph_ctx.new_graph();
    graph_ctx.build_forward(graph, embed);
    backend_->reserve_compute_memory(graph, "runtime.locenc_to_lm_embed");
    backend_->alloc_graph(graph, "runtime.locenc_to_lm_embed");
    backend_->tensor_set(patch_tensor, patch.data(), 0, patch.size() * sizeof(float));
    VOXCPM_ASSERT(backend_->compute(graph) == GGML_STATUS_SUCCESS);
    maybe_collect_graph(graph);

    std::vector<float> out(static_cast<size_t>(base_lm_.config().hidden_size));
    backend_->tensor_get(embed, out.data(), 0, out.size() * sizeof(float));
    return out;
}

std::vector<float> VoxCPMRuntime::run_base_lm_decode_step(const std::vector<float>& curr_embed,
                                                          int position,
                                                          MiniCPMKVCache& kv_cache) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(static_cast<int>(curr_embed.size()) == base_lm_.config().hidden_size);

    VoxCPMContext graph_ctx = make_graph_ctx(16384, 131072);
    ggml_tensor* input_tensor = graph_ctx.new_tensor_1d(GGML_TYPE_F32, base_lm_.config().hidden_size);
    ggml_tensor* positions_tensor = graph_ctx.new_tensor_1d(GGML_TYPE_I32, 1);
    ggml_set_input(input_tensor);
    ggml_set_input(positions_tensor);

    ggml_tensor* hidden = base_lm_.forward_step(graph_ctx, input_tensor, position, positions_tensor, kv_cache, true);
    ggml_tensor* hidden_2d = ggml_reshape_2d(graph_ctx.raw_context(), hidden, hidden->ne[0], 1);
    ggml_tensor* fsq_hidden = fsq_layer_.forward(graph_ctx, hidden_2d);
    ggml_tensor* output = ggml_reshape_1d(graph_ctx.raw_context(), fsq_hidden, fsq_hidden->ne[0]);
    ggml_set_output(output);

    ggml_cgraph* graph = graph_ctx.new_graph();
    graph_ctx.build_forward(graph, output);
    backend_->reserve_compute_memory(graph, "runtime.base_lm.decode_step");
    backend_->alloc_graph(graph, "runtime.base_lm.decode_step");
    backend_->tensor_set(input_tensor, curr_embed.data(), 0, curr_embed.size() * sizeof(float));
    const int32_t position_value = position;
    backend_->tensor_set(positions_tensor, &position_value, 0, sizeof(position_value));
    VOXCPM_ASSERT(backend_->compute(graph) == GGML_STATUS_SUCCESS);
    maybe_collect_graph(graph);

    std::vector<float> out(static_cast<size_t>(base_lm_.config().hidden_size));
    backend_->tensor_get(output, out.data(), 0, out.size() * sizeof(float));
    return out;
}

VoxCPMDecodeState VoxCPMRuntime::prefill(const std::vector<int32_t>& text,
                                         const std::vector<int32_t>& text_mask,
                                         const std::vector<float>& feat,
                                         const std::vector<int32_t>& feat_mask,
                                         int seq_len,
                                         int streaming_prefix_len) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(static_cast<int>(text.size()) == seq_len);
    VOXCPM_ASSERT(static_cast<int>(text_mask.size()) == seq_len);
    VOXCPM_ASSERT(static_cast<int>(feat.size()) == seq_len * config_.patch_size * config_.feat_dim);
    VOXCPM_ASSERT(static_cast<int>(feat_mask.size()) == seq_len);

    VoxCPMDecodeState state = create_decode_state();
    state.streaming_prefix_len = streaming_prefix_len;

    const std::vector<float> feat_encoded = encode_feature_sequence(feat, seq_len);
    const std::vector<float> feat_embed = run_projection_2d(*components_->enc_to_lm_proj(),
                                                            feat_encoded,
                                                            feat_encoder_.config().hidden_size,
                                                            seq_len,
                                                            base_lm_.config().hidden_size);
    const std::vector<float> text_embed = run_embedding(text);

    std::vector<float> combined_embed(static_cast<size_t>(base_lm_.config().hidden_size) * seq_len, 0.0f);
    for (int t = 0; t < seq_len; ++t) {
        const float text_scale = text_mask[static_cast<size_t>(t)] != 0 ? 1.0f : 0.0f;
        const float feat_scale = feat_mask[static_cast<size_t>(t)] != 0 ? 1.0f : 0.0f;
        for (int h = 0; h < base_lm_.config().hidden_size; ++h) {
            const size_t idx = static_cast<size_t>(t) * base_lm_.config().hidden_size + h;
            combined_embed[idx] = text_scale * text_embed[idx] + feat_scale * feat_embed[idx];
        }
    }

    std::vector<float> enc_outputs = run_minicpm_forward(base_lm_, combined_embed, seq_len, *state.base_lm_cache, true);
    const std::vector<float> fsq_outputs = run_fsq_2d(enc_outputs, seq_len);
    for (int t = 0; t < seq_len; ++t) {
        const float text_scale = text_mask[static_cast<size_t>(t)] != 0 ? 1.0f : 0.0f;
        const float feat_scale = feat_mask[static_cast<size_t>(t)] != 0 ? 1.0f : 0.0f;
        for (int h = 0; h < base_lm_.config().hidden_size; ++h) {
            const size_t idx = static_cast<size_t>(t) * base_lm_.config().hidden_size + h;
            enc_outputs[idx] = feat_scale * fsq_outputs[idx] + text_scale * enc_outputs[idx];
        }
    }

    state.lm_hidden = slice_column_major_2d(enc_outputs, base_lm_.config().hidden_size, seq_len - 1);

    std::vector<float> residual_inputs = enc_outputs;
    for (int t = 0; t < seq_len; ++t) {
        const float feat_scale = feat_mask[static_cast<size_t>(t)] != 0 ? 1.0f : 0.0f;
        for (int h = 0; h < base_lm_.config().hidden_size; ++h) {
            const size_t idx = static_cast<size_t>(t) * base_lm_.config().hidden_size + h;
            residual_inputs[idx] += feat_scale * feat_embed[idx];
        }
    }

    const std::vector<float> residual_outputs =
        run_minicpm_forward(residual_lm_, residual_inputs, seq_len, *state.residual_lm_cache, true);
    state.residual_hidden =
        slice_column_major_2d(residual_outputs, residual_lm_.config().hidden_size, seq_len - 1);

    const size_t prefix_offset = static_cast<size_t>(seq_len - 1) * config_.patch_size * config_.feat_dim;
    std::copy_n(feat.data() + prefix_offset,
                static_cast<size_t>(config_.patch_size * config_.feat_dim),
                state.prefix_feat_cond.data());
    state.current_position = seq_len;
    return state;
}

VoxCPMDecodeResult VoxCPMRuntime::decode(VoxCPMDecodeState state,
                                         const std::vector<float>& z,
                                         int inference_timesteps,
                                         float cfg_value) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(state.base_lm_cache != nullptr);
    VOXCPM_ASSERT(state.residual_lm_cache != nullptr);
    VOXCPM_ASSERT(static_cast<int>(state.lm_hidden.size()) == base_lm_.config().hidden_size);
    VOXCPM_ASSERT(static_cast<int>(state.residual_hidden.size()) == residual_lm_.config().hidden_size);
    VOXCPM_ASSERT(static_cast<int>(state.prefix_feat_cond.size()) == config_.patch_size * config_.feat_dim);
    VOXCPM_ASSERT(static_cast<int>(z.size()) == config_.feat_dim * config_.patch_size);

    VoxCPMDecodeResult result;
    run_decode_front_half(z,
                          state.lm_hidden,
                          state.residual_hidden,
                          state.prefix_feat_cond,
                          inference_timesteps,
                          cfg_value,
                          result.output_0);
    const int new_position = state.current_position + 1;
    const std::vector<float> stop_logits = run_stop_predictor(state.lm_hidden);
    result.output_2 = stop_logits.size() >= 2 && stop_logits[1] > stop_logits[0];

    const std::vector<float> curr_embed = run_locenc_patch_to_lm_embed(result.output_0);
    std::vector<float> lm_hidden = run_base_lm_decode_step(curr_embed,
                                                           new_position,
                                                           *state.base_lm_cache);

    std::vector<float> residual_input(curr_embed.size(), 0.0f);
    for (size_t i = 0; i < residual_input.size(); ++i) {
        residual_input[i] = lm_hidden[i] + curr_embed[i];
    }

    const std::vector<float> residual_hidden = run_minicpm_forward_step(residual_lm_,
                                                                        residual_input,
                                                                        new_position,
                                                                        *state.residual_lm_cache,
                                                                        true);

    state.lm_hidden = std::move(lm_hidden);
    state.residual_hidden = residual_hidden;
    state.current_position = new_position;
    state.prefix_feat_cond = result.output_0;

    result.output_1 = std::move(state);
    return result;
}

std::vector<float> VoxCPMRuntime::benchmark_encode_feature_sequence(const std::vector<float>& feat, int seq_len) {
    return encode_feature_sequence(feat, seq_len);
}

std::vector<float> VoxCPMRuntime::benchmark_run_embedding(const std::vector<int32_t>& token_ids) {
    return run_embedding(token_ids);
}

std::vector<float> VoxCPMRuntime::benchmark_run_enc_to_lm_projection(const std::vector<float>& input, int seq_len) {
    return run_projection_2d(*components_->enc_to_lm_proj(),
                             input,
                             feat_encoder_.config().hidden_size,
                             seq_len,
                             base_lm_.config().hidden_size);
}

std::vector<float> VoxCPMRuntime::benchmark_run_lm_to_dit_projection(const std::vector<float>& input) {
    return run_projection_1d(*components_->lm_to_dit_proj(),
                             input,
                             base_lm_.config().hidden_size,
                             config_.loc_dit.hidden_size);
}

std::vector<float> VoxCPMRuntime::benchmark_run_res_to_dit_projection(const std::vector<float>& input) {
    return run_projection_1d(*components_->res_to_dit_proj(),
                             input,
                             residual_lm_.config().hidden_size,
                             config_.loc_dit.hidden_size);
}

std::vector<float> VoxCPMRuntime::benchmark_run_fsq_2d(const std::vector<float>& input, int seq_len) {
    return run_fsq_2d(input, seq_len);
}

std::vector<float> VoxCPMRuntime::benchmark_run_base_lm_forward(const std::vector<float>& input,
                                                                int seq_len,
                                                                MiniCPMKVCache& kv_cache,
                                                                bool is_causal) {
    return run_minicpm_forward(base_lm_, input, seq_len, kv_cache, is_causal);
}

std::vector<float> VoxCPMRuntime::benchmark_run_residual_lm_forward(const std::vector<float>& input,
                                                                    int seq_len,
                                                                    MiniCPMKVCache& kv_cache,
                                                                    bool is_causal) {
    return run_minicpm_forward(residual_lm_, input, seq_len, kv_cache, is_causal);
}

std::vector<float> VoxCPMRuntime::benchmark_run_unified_cfm(const std::vector<float>& z,
                                                            const std::vector<float>& mu,
                                                            const std::vector<float>& cond,
                                                            int n_timesteps,
                                                            float cfg_value) {
    return run_unified_cfm(z, mu, cond, n_timesteps, cfg_value);
}

std::vector<float> VoxCPMRuntime::benchmark_run_stop_predictor(const std::vector<float>& input) {
    return run_stop_predictor(input);
}

std::vector<float> VoxCPMRuntime::benchmark_run_locenc_patch_to_lm_embed(const std::vector<float>& patch) {
    return run_locenc_patch_to_lm_embed(patch);
}

std::vector<float> VoxCPMRuntime::benchmark_run_base_lm_decode_step(const std::vector<float>& curr_embed,
                                                                    int position,
                                                                    MiniCPMKVCache& kv_cache) {
    return run_base_lm_decode_step(curr_embed, position, kv_cache);
}

std::vector<float> VoxCPMRuntime::benchmark_run_residual_lm_decode_step(const std::vector<float>& input,
                                                                        int position,
                                                                        MiniCPMKVCache& kv_cache,
                                                                        bool is_causal) {
    return run_minicpm_forward_step(residual_lm_, input, position, kv_cache, is_causal);
}

std::vector<float> VoxCPMRuntime::benchmark_run_decode_front_half(const std::vector<float>& z,
                                                                  const std::vector<float>& lm_hidden,
                                                                  const std::vector<float>& residual_hidden,
                                                                  const std::vector<float>& prefix_feat_cond,
                                                                  int inference_timesteps,
                                                                  float cfg_value) {
    std::vector<float> output;
    run_decode_front_half(z,
                          lm_hidden,
                          residual_hidden,
                          prefix_feat_cond,
                          inference_timesteps,
                          cfg_value,
                          output);
    return output;
}

VoxCPMDecodeState VoxCPMRuntime::benchmark_clone_state(const VoxCPMDecodeState& state) const {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(state.base_lm_cache != nullptr);
    VOXCPM_ASSERT(state.residual_lm_cache != nullptr);

    VoxCPMDecodeState copy = create_decode_state();
    copy.base_lm_cache->copy_from(*state.base_lm_cache, *backend_);
    copy.residual_lm_cache->copy_from(*state.residual_lm_cache, *backend_);
    copy.lm_hidden = state.lm_hidden;
    copy.residual_hidden = state.residual_hidden;
    copy.current_position = state.current_position;
    copy.prefix_feat_cond = state.prefix_feat_cond;
    copy.streaming_prefix_len = state.streaming_prefix_len;
    return copy;
}

}  // namespace voxcpm
