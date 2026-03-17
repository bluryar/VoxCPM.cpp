#include "voxcpm/voxcpm.h"

#include "voxcpm/backend.h"
#include "voxcpm/imatrix.h"
#include "voxcpm/weight-store.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace voxcpm {

namespace {

VoxCPMContext make_graph_ctx(int n_tensors, int max_nodes) {
    return VoxCPMContext(ContextType::Graph, n_tensors, max_nodes);
}

std::string decode_graph_key(int n_timesteps, float cfg_value) {
    uint32_t cfg_bits = 0;
    static_assert(sizeof(cfg_bits) == sizeof(cfg_value), "float size mismatch");
    std::memcpy(&cfg_bits, &cfg_value, sizeof(cfg_bits));

    std::ostringstream oss;
    oss << n_timesteps << ":" << cfg_bits;
    return oss.str();
}

std::vector<float> slice_column_major_2d(const std::vector<float>& input,
                                         int row_dim,
                                         int col_idx) {
    std::vector<float> out(static_cast<size_t>(row_dim));
    const size_t offset = static_cast<size_t>(col_idx) * row_dim;
    std::copy_n(input.data() + offset, row_dim, out.data());
    return out;
}

bool env_flag_enabled(const char* name) {
    const char* raw = std::getenv(name);
    if (!raw || raw[0] == '\0') {
        return false;
    }

    std::string value(raw);
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value == "1" || value == "true" || value == "yes" || value == "on";
}

BackendTransferStats transfer_stats_delta(const BackendTransferStats& before,
                                          const BackendTransferStats& after) {
    BackendTransferStats delta;
    delta.host_to_device_bytes = after.host_to_device_bytes - before.host_to_device_bytes;
    delta.device_to_host_bytes = after.device_to_host_bytes - before.device_to_host_bytes;
    delta.device_to_device_bytes = after.device_to_device_bytes - before.device_to_device_bytes;
    delta.host_to_device_ms = after.host_to_device_ms - before.host_to_device_ms;
    delta.device_to_host_ms = after.device_to_host_ms - before.device_to_host_ms;
    delta.device_to_device_ms = after.device_to_device_ms - before.device_to_device_ms;
    return delta;
}

double bytes_to_mib(size_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

std::string format_transfer_stats(const BackendTransferStats& stats) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3)
        << "h2d=" << bytes_to_mib(stats.host_to_device_bytes) << "MiB/" << stats.host_to_device_ms << "ms"
        << " d2h=" << bytes_to_mib(stats.device_to_host_bytes) << "MiB/" << stats.device_to_host_ms << "ms"
        << " d2d=" << bytes_to_mib(stats.device_to_device_bytes) << "MiB/" << stats.device_to_device_ms << "ms";
    return oss.str();
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
    clear_cached_graphs();
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

    if (backend_ && backend_->is_gpu()) {
        for (int timesteps = 8; timesteps <= 12; ++timesteps) {
            (void) get_precomputed_cfm_time_table(timesteps);
        }
    }

    return true;
}

void VoxCPMRuntime::maybe_collect_graph(ggml_cgraph* graph) {
    if (imatrix_collector_ && backend_ && graph) {
        imatrix_collector_->observe_graph(graph, *backend_);
    }
}

void VoxCPMRuntime::clear_cached_graphs() {
    locenc_patch_graph_.clear();
    for (auto& entry : locenc_sequence_graphs_) {
        entry.second.clear();
    }
    locenc_sequence_graphs_.clear();
    for (auto& entry : embedding_graphs_) {
        entry.second.clear();
    }
    embedding_graphs_.clear();
    for (auto& entry : enc_to_lm_projection_graphs_) {
        entry.second.clear();
    }
    enc_to_lm_projection_graphs_.clear();
    for (auto& entry : fsq_2d_graphs_) {
        entry.second.clear();
    }
    fsq_2d_graphs_.clear();
    for (auto& entry : unified_cfm_graphs_) {
        entry.second.clear();
    }
    unified_cfm_graphs_.clear();
    for (auto& entry : decode_front_half_graphs_) {
        entry.second.clear();
    }
    decode_front_half_graphs_.clear();
    precomputed_cfm_time_tables_.clear();
    stop_predictor_graph_.clear();
    locenc_patch_to_lm_embed_graph_.clear();
}

bool VoxCPMRuntime::should_precompute_cfm_time_table(int n_timesteps) const {
    return n_timesteps >= 8 && n_timesteps <= 12;
}

const std::vector<float>& VoxCPMRuntime::get_precomputed_cfm_time_table(int n_timesteps) {
    VOXCPM_ASSERT(feat_decoder_ != nullptr);

    auto [it, inserted] = precomputed_cfm_time_tables_.try_emplace(n_timesteps);
    if (!inserted && !it->second.empty()) {
        return it->second;
    }

    const std::vector<float> t_span = UnifiedCFM::compute_t_span(n_timesteps, feat_decoder_->config().sway_sampling_coef);
    std::vector<float> t_values;
    t_values.reserve(static_cast<size_t>(n_timesteps));
    for (int step = 0; step < n_timesteps; ++step) {
        t_values.push_back(t_span[static_cast<size_t>(step)]);
    }

    it->second = feat_decoder_estimator_.precompute_cfg_time_table(t_values);
    return it->second;
}

VoxCPMCachedGraph& VoxCPMRuntime::ensure_locenc_patch_graph() {
    VOXCPM_ASSERT(backend_ != nullptr);

    if (locenc_patch_graph_.graph) {
        return locenc_patch_graph_;
    }

    locenc_patch_graph_.context = std::make_unique<VoxCPMContext>(ContextType::Graph, 8192, 65536);
    VoxCPMContext& graph_ctx = *locenc_patch_graph_.context;
    locenc_patch_graph_.input0 = graph_ctx.new_tensor_2d(GGML_TYPE_F32, config_.feat_dim, config_.patch_size);
    ggml_set_input(locenc_patch_graph_.input0);
    locenc_patch_graph_.output = feat_encoder_.forward_patch(graph_ctx, locenc_patch_graph_.input0);
    ggml_set_output(locenc_patch_graph_.output);

    locenc_patch_graph_.graph = graph_ctx.new_graph();
    graph_ctx.build_forward(locenc_patch_graph_.graph, locenc_patch_graph_.output);
    backend_->reserve_compute_memory(locenc_patch_graph_.graph, "runtime.locenc.patch.cached");
    return locenc_patch_graph_;
}

VoxCPMCachedGraph& VoxCPMRuntime::ensure_locenc_sequence_graph(int seq_len) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(seq_len >= 0);

    auto [it, inserted] = locenc_sequence_graphs_.try_emplace(seq_len);
    VoxCPMCachedGraph& cached = it->second;
    if (!inserted && cached.graph) {
        return cached;
    }

    cached.clear();
    cached.context = std::make_unique<VoxCPMContext>(ContextType::Graph, 65536, 524288);
    VoxCPMContext& graph_ctx = *cached.context;
    cached.input0 = graph_ctx.new_tensor_3d(GGML_TYPE_F32, config_.feat_dim, config_.patch_size, seq_len);
    ggml_set_input(cached.input0);
    cached.output = feat_encoder_.forward_sequence(graph_ctx, cached.input0);
    ggml_set_output(cached.output);

    cached.graph = graph_ctx.new_graph();
    graph_ctx.build_forward(cached.graph, cached.output);
    backend_->reserve_compute_memory(cached.graph, "runtime.locenc.sequence.cached");
    return cached;
}

VoxCPMCachedGraph& VoxCPMRuntime::ensure_embedding_graph(int token_count) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(components_ != nullptr);
    VOXCPM_ASSERT(token_count >= 0);

    auto [it, inserted] = embedding_graphs_.try_emplace(token_count);
    VoxCPMCachedGraph& cached = it->second;
    if (!inserted && cached.graph) {
        return cached;
    }

    cached.clear();
    cached.context = std::make_unique<VoxCPMContext>(ContextType::Graph, 4096, 32768);
    VoxCPMContext& graph_ctx = *cached.context;
    cached.input0 = graph_ctx.new_tensor_1d(GGML_TYPE_I32, token_count);
    ggml_set_input(cached.input0);
    cached.output = components_->embed_tokens()->forward(graph_ctx, cached.input0);
    ggml_set_output(cached.output);

    cached.graph = graph_ctx.new_graph();
    graph_ctx.build_forward(cached.graph, cached.output);
    backend_->reserve_compute_memory(cached.graph, "runtime.embedding.cached");
    return cached;
}

VoxCPMCachedGraph& VoxCPMRuntime::ensure_enc_to_lm_projection_graph(int seq_len) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(components_ != nullptr);
    VOXCPM_ASSERT(seq_len >= 0);

    auto [it, inserted] = enc_to_lm_projection_graphs_.try_emplace(seq_len);
    VoxCPMCachedGraph& cached = it->second;
    if (!inserted && cached.graph) {
        return cached;
    }

    cached.clear();
    cached.context = std::make_unique<VoxCPMContext>(ContextType::Graph, 4096, 32768);
    VoxCPMContext& graph_ctx = *cached.context;
    cached.input0 = graph_ctx.new_tensor_2d(GGML_TYPE_F32, feat_encoder_.config().hidden_size, seq_len);
    ggml_set_input(cached.input0);
    cached.output = components_->enc_to_lm_proj()->forward(graph_ctx, cached.input0);
    ggml_set_output(cached.output);

    cached.graph = graph_ctx.new_graph();
    graph_ctx.build_forward(cached.graph, cached.output);
    backend_->reserve_compute_memory(cached.graph, "runtime.enc_to_lm_proj.cached");
    return cached;
}

VoxCPMCachedGraph& VoxCPMRuntime::ensure_fsq_2d_graph(int seq_len) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(seq_len >= 0);

    auto [it, inserted] = fsq_2d_graphs_.try_emplace(seq_len);
    VoxCPMCachedGraph& cached = it->second;
    if (!inserted && cached.graph) {
        return cached;
    }

    cached.clear();
    cached.context = std::make_unique<VoxCPMContext>(ContextType::Graph, 8192, 65536);
    VoxCPMContext& graph_ctx = *cached.context;
    cached.input0 = graph_ctx.new_tensor_2d(GGML_TYPE_F32, base_lm_.config().hidden_size, seq_len);
    ggml_set_input(cached.input0);
    cached.output = fsq_layer_.forward(graph_ctx, cached.input0);
    ggml_set_output(cached.output);

    cached.graph = graph_ctx.new_graph();
    graph_ctx.build_forward(cached.graph, cached.output);
    backend_->reserve_compute_memory(cached.graph, "runtime.fsq.2d.cached");
    return cached;
}

VoxCPMCachedGraph& VoxCPMRuntime::ensure_unified_cfm_graph(int n_timesteps, float cfg_value) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(feat_decoder_ != nullptr);

    const std::string key = decode_graph_key(n_timesteps, cfg_value);
    auto [it, inserted] = unified_cfm_graphs_.try_emplace(key);
    VoxCPMCachedGraph& cached = it->second;
    if (!inserted && cached.graph) {
        return cached;
    }

    cached.clear();
    cached.context = std::make_unique<VoxCPMContext>(ContextType::Graph, 65536, 524288);
    VoxCPMContext& graph_ctx = *cached.context;
    cached.input0 = graph_ctx.new_tensor_2d(GGML_TYPE_F32, config_.feat_dim, config_.patch_size);
    cached.input1 = graph_ctx.new_tensor_1d(GGML_TYPE_F32, config_.loc_dit.hidden_size);
    cached.input2 = graph_ctx.new_tensor_2d(GGML_TYPE_F32, config_.feat_dim, config_.patch_size);
    if (should_precompute_cfm_time_table(n_timesteps)) {
        cached.input4 = graph_ctx.new_tensor_2d(GGML_TYPE_F32, config_.loc_dit.hidden_size, n_timesteps);
        ggml_set_input(cached.input4);
        cached.aux_input4 = get_precomputed_cfm_time_table(n_timesteps);
    }
    ggml_set_input(cached.input0);
    ggml_set_input(cached.input1);
    ggml_set_input(cached.input2);
    cached.output = feat_decoder_->forward(graph_ctx,
                                           cached.input0,
                                           cached.input1,
                                           config_.patch_size,
                                           cached.input2,
                                           n_timesteps,
                                           cfg_value,
                                           1.0f,
                                           feat_decoder_->config().sway_sampling_coef,
                                           feat_decoder_->config().use_cfg_zero_star,
                                           cached.input4);
    ggml_set_output(cached.output);

    cached.graph = graph_ctx.new_graph();
    graph_ctx.build_forward(cached.graph, cached.output);
    backend_->reserve_compute_memory(cached.graph, "runtime.unified_cfm.cached");
    return cached;
}

VoxCPMCachedGraph& VoxCPMRuntime::ensure_decode_front_half_graph(int n_timesteps, float cfg_value) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(components_ != nullptr);
    VOXCPM_ASSERT(feat_decoder_ != nullptr);

    const std::string key = decode_graph_key(n_timesteps, cfg_value);
    auto [it, inserted] = decode_front_half_graphs_.try_emplace(key);
    VoxCPMCachedGraph& cached = it->second;
    if (!inserted && cached.graph) {
        return cached;
    }

    cached.clear();
    cached.context = std::make_unique<VoxCPMContext>(ContextType::Graph, 65536, 524288);
    VoxCPMContext& graph_ctx = *cached.context;
    cached.input0 = graph_ctx.new_tensor_2d(GGML_TYPE_F32, config_.feat_dim, config_.patch_size);
    cached.input1 = graph_ctx.new_tensor_1d(GGML_TYPE_F32, base_lm_.config().hidden_size);
    cached.input2 = graph_ctx.new_tensor_1d(GGML_TYPE_F32, residual_lm_.config().hidden_size);
    cached.input3 = graph_ctx.new_tensor_2d(GGML_TYPE_F32, config_.feat_dim, config_.patch_size);
    if (should_precompute_cfm_time_table(n_timesteps)) {
        cached.input4 = graph_ctx.new_tensor_2d(GGML_TYPE_F32, config_.loc_dit.hidden_size, n_timesteps);
        ggml_set_input(cached.input4);
        cached.aux_input4 = get_precomputed_cfm_time_table(n_timesteps);
    }
    ggml_set_input(cached.input0);
    ggml_set_input(cached.input1);
    ggml_set_input(cached.input2);
    ggml_set_input(cached.input3);

    ggml_tensor* dit_hidden_1 = components_->lm_to_dit_proj()->forward(graph_ctx, cached.input1);
    ggml_tensor* dit_hidden_2 = components_->res_to_dit_proj()->forward(graph_ctx, cached.input2);
    ggml_tensor* dit_hidden = ggml_add(graph_ctx.raw_context(), dit_hidden_1, dit_hidden_2);
    cached.output = feat_decoder_->forward(graph_ctx,
                                           cached.input0,
                                           dit_hidden,
                                           config_.patch_size,
                                           cached.input3,
                                           n_timesteps,
                                           cfg_value,
                                           1.0f,
                                           feat_decoder_->config().sway_sampling_coef,
                                           feat_decoder_->config().use_cfg_zero_star,
                                           cached.input4);
    ggml_set_output(cached.output);

    ggml_tensor* patch_hidden = feat_encoder_.forward_patch(graph_ctx, cached.output);
    cached.output_aux0 = components_->enc_to_lm_proj()->forward(graph_ctx, patch_hidden);
    ggml_set_output(cached.output_aux0);

    cached.graph = graph_ctx.new_graph();
    graph_ctx.build_forward(cached.graph, cached.output_aux0);
    backend_->reserve_compute_memory(cached.graph, "runtime.decode_front_half.cached");
    return cached;
}

VoxCPMCachedGraph& VoxCPMRuntime::ensure_state_base_lm_step_graph(VoxCPMDecodeState& state, int position) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(state.base_lm_cache != nullptr);

    if (state.base_lm_step_graph.graph && state.base_lm_step_graph_position == position) {
        return state.base_lm_step_graph;
    }

    VoxCPMCachedGraph& cached = state.base_lm_step_graph;
    cached.clear();
    state.base_lm_step_graph_position = position;
    cached.context = std::make_unique<VoxCPMContext>(ContextType::Graph, 16384, 131072);
    VoxCPMContext& graph_ctx = *cached.context;
    cached.input0 = graph_ctx.new_tensor_1d(GGML_TYPE_F32, base_lm_.config().hidden_size);
    cached.input1 = graph_ctx.new_tensor_1d(GGML_TYPE_I32, 1);
    ggml_set_input(cached.input0);
    ggml_set_input(cached.input1);

    ggml_tensor* hidden = base_lm_.forward_step(graph_ctx, cached.input0, position, cached.input1, *state.base_lm_cache, true);
    ggml_tensor* hidden_2d = ggml_reshape_2d(graph_ctx.raw_context(), hidden, hidden->ne[0], 1);
    ggml_tensor* fsq_hidden = fsq_layer_.forward(graph_ctx, hidden_2d);
    cached.output = ggml_reshape_1d(graph_ctx.raw_context(), fsq_hidden, fsq_hidden->ne[0]);
    ggml_set_output(cached.output);

    cached.graph = graph_ctx.new_graph();
    graph_ctx.build_forward(cached.graph, cached.output);
    backend_->reserve_compute_memory(cached.graph, "runtime.base_lm.decode_step.state_cached");
    return cached;
}

VoxCPMCachedGraph& VoxCPMRuntime::ensure_state_residual_lm_step_graph(VoxCPMDecodeState& state, int position) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(state.residual_lm_cache != nullptr);

    if (state.residual_lm_step_graph.graph && state.residual_lm_step_graph_position == position) {
        return state.residual_lm_step_graph;
    }

    VoxCPMCachedGraph& cached = state.residual_lm_step_graph;
    cached.clear();
    state.residual_lm_step_graph_position = position;
    cached.context = std::make_unique<VoxCPMContext>(ContextType::Graph, 8192, 65536);
    VoxCPMContext& graph_ctx = *cached.context;
    cached.input0 = graph_ctx.new_tensor_1d(GGML_TYPE_F32, residual_lm_.config().hidden_size);
    cached.input1 = graph_ctx.new_tensor_1d(GGML_TYPE_I32, 1);
    ggml_set_input(cached.input0);
    ggml_set_input(cached.input1);
    cached.output = residual_lm_.forward_step(graph_ctx,
                                              cached.input0,
                                              position,
                                              cached.input1,
                                              *state.residual_lm_cache,
                                              true);
    ggml_set_output(cached.output);

    cached.graph = graph_ctx.new_graph();
    graph_ctx.build_forward(cached.graph, cached.output);
    backend_->reserve_compute_memory(cached.graph, "runtime.residual_lm.decode_step.state_cached");
    return cached;
}

VoxCPMCachedGraph& VoxCPMRuntime::ensure_stop_predictor_graph() {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(components_ != nullptr);

    if (stop_predictor_graph_.graph) {
        return stop_predictor_graph_;
    }

    stop_predictor_graph_.context = std::make_unique<VoxCPMContext>(ContextType::Graph, 4096, 32768);
    VoxCPMContext& graph_ctx = *stop_predictor_graph_.context;
    stop_predictor_graph_.input0 = graph_ctx.new_tensor_1d(GGML_TYPE_F32, base_lm_.config().hidden_size);
    ggml_set_input(stop_predictor_graph_.input0);
    stop_predictor_graph_.output = components_->stop_token()->forward(graph_ctx, stop_predictor_graph_.input0);
    ggml_set_output(stop_predictor_graph_.output);

    stop_predictor_graph_.graph = graph_ctx.new_graph();
    graph_ctx.build_forward(stop_predictor_graph_.graph, stop_predictor_graph_.output);
    backend_->reserve_compute_memory(stop_predictor_graph_.graph, "runtime.stop_predictor.cached");
    return stop_predictor_graph_;
}

VoxCPMCachedGraph& VoxCPMRuntime::ensure_locenc_patch_to_lm_embed_graph() {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(components_ != nullptr);

    if (locenc_patch_to_lm_embed_graph_.graph) {
        return locenc_patch_to_lm_embed_graph_;
    }

    locenc_patch_to_lm_embed_graph_.context = std::make_unique<VoxCPMContext>(ContextType::Graph, 16384, 131072);
    VoxCPMContext& graph_ctx = *locenc_patch_to_lm_embed_graph_.context;
    locenc_patch_to_lm_embed_graph_.input0 =
        graph_ctx.new_tensor_2d(GGML_TYPE_F32, config_.feat_dim, config_.patch_size);
    ggml_set_input(locenc_patch_to_lm_embed_graph_.input0);
    ggml_tensor* hidden = feat_encoder_.forward_patch(graph_ctx, locenc_patch_to_lm_embed_graph_.input0);
    locenc_patch_to_lm_embed_graph_.output = components_->enc_to_lm_proj()->forward(graph_ctx, hidden);
    ggml_set_output(locenc_patch_to_lm_embed_graph_.output);

    locenc_patch_to_lm_embed_graph_.graph = graph_ctx.new_graph();
    graph_ctx.build_forward(locenc_patch_to_lm_embed_graph_.graph, locenc_patch_to_lm_embed_graph_.output);
    backend_->reserve_compute_memory(locenc_patch_to_lm_embed_graph_.graph, "runtime.locenc_to_lm_embed.cached");
    return locenc_patch_to_lm_embed_graph_;
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

void VoxCPMRuntime::run_locenc_patch_into(const float* patch_data, float* output_data) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(output_data != nullptr);

    VoxCPMCachedGraph& cached = ensure_locenc_patch_graph();
    backend_->alloc_graph(cached.graph, "runtime.locenc.patch.cached");
    backend_->tensor_set(cached.input0,
                         patch_data,
                         0,
                         static_cast<size_t>(config_.feat_dim * config_.patch_size) * sizeof(float));
    VOXCPM_ASSERT(backend_->compute(cached.graph) == GGML_STATUS_SUCCESS);
    maybe_collect_graph(cached.graph);
    backend_->tensor_get(cached.output,
                         output_data,
                         0,
                         static_cast<size_t>(base_lm_.config().hidden_size) * sizeof(float));
}

std::vector<float> VoxCPMRuntime::run_locenc_patch(const float* patch_data) {
    std::vector<float> out(static_cast<size_t>(base_lm_.config().hidden_size));
    run_locenc_patch_into(patch_data, out.data());
    return out;
}

std::vector<float> VoxCPMRuntime::encode_feature_sequence(const std::vector<float>& feat, int seq_len) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(static_cast<int>(feat.size()) == seq_len * config_.patch_size * config_.feat_dim);

    VoxCPMCachedGraph& cached = ensure_locenc_sequence_graph(seq_len);
    backend_->alloc_graph(cached.graph, "runtime.locenc.sequence.cached");
    backend_->tensor_set(cached.input0, feat.data(), 0, feat.size() * sizeof(float));
    VOXCPM_ASSERT(backend_->compute(cached.graph) == GGML_STATUS_SUCCESS);
    maybe_collect_graph(cached.graph);

    std::vector<float> encoded(static_cast<size_t>(base_lm_.config().hidden_size) * seq_len);
    backend_->tensor_get(cached.output, encoded.data(), 0, encoded.size() * sizeof(float));
    return encoded;
}

std::vector<float> VoxCPMRuntime::run_embedding(const std::vector<int32_t>& token_ids) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VoxCPMCachedGraph& cached = ensure_embedding_graph(static_cast<int>(token_ids.size()));
    backend_->alloc_graph(cached.graph, "runtime.embedding.cached");
    backend_->tensor_set(cached.input0, token_ids.data(), 0, token_ids.size() * sizeof(int32_t));
    VOXCPM_ASSERT(backend_->compute(cached.graph) == GGML_STATUS_SUCCESS);
    maybe_collect_graph(cached.graph);

    std::vector<float> out(static_cast<size_t>(components_->embed_tokens()->config().hidden_dim) * token_ids.size());
    backend_->tensor_get(cached.output, out.data(), 0, out.size() * sizeof(float));
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

    if (components_ != nullptr &&
        &projection == components_->enc_to_lm_proj() &&
        in_dim == feat_encoder_.config().hidden_size &&
        out_dim == base_lm_.config().hidden_size) {
        VoxCPMCachedGraph& cached = ensure_enc_to_lm_projection_graph(seq_len);
        backend_->alloc_graph(cached.graph, "runtime.enc_to_lm_proj.cached");
        backend_->tensor_set(cached.input0, input.data(), 0, input.size() * sizeof(float));
        VOXCPM_ASSERT(backend_->compute(cached.graph) == GGML_STATUS_SUCCESS);
        maybe_collect_graph(cached.graph);

        std::vector<float> out(static_cast<size_t>(out_dim) * seq_len);
        backend_->tensor_get(cached.output, out.data(), 0, out.size() * sizeof(float));
        return out;
    }

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
    VoxCPMCachedGraph& cached = ensure_stop_predictor_graph();
    backend_->alloc_graph(cached.graph, "runtime.stop_predictor.cached");
    backend_->tensor_set(cached.input0, input.data(), 0, input.size() * sizeof(float));
    VOXCPM_ASSERT(backend_->compute(cached.graph) == GGML_STATUS_SUCCESS);
    maybe_collect_graph(cached.graph);

    std::vector<float> out(2);
    backend_->tensor_get(cached.output, out.data(), 0, out.size() * sizeof(float));
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
    VoxCPMCachedGraph& cached = ensure_fsq_2d_graph(seq_len);
    backend_->alloc_graph(cached.graph, "runtime.fsq.2d.cached");
    backend_->tensor_set(cached.input0, input.data(), 0, input.size() * sizeof(float));
    VOXCPM_ASSERT(backend_->compute(cached.graph) == GGML_STATUS_SUCCESS);
    maybe_collect_graph(cached.graph);

    std::vector<float> out(static_cast<size_t>(base_lm_.config().hidden_size) * seq_len);
    backend_->tensor_get(cached.output, out.data(), 0, out.size() * sizeof(float));
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

    VoxCPMCachedGraph& cached = ensure_unified_cfm_graph(n_timesteps, cfg_value);
    backend_->alloc_graph(cached.graph, "runtime.unified_cfm.cached");
    backend_->tensor_set(cached.input0, z.data(), 0, z.size() * sizeof(float));
    backend_->tensor_set(cached.input1, mu.data(), 0, mu.size() * sizeof(float));
    backend_->tensor_set(cached.input2, cond.data(), 0, cond.size() * sizeof(float));
    if (cached.input4 && !cached.aux_input4.empty()) {
        backend_->tensor_set(cached.input4, cached.aux_input4.data(), 0, cached.aux_input4.size() * sizeof(float));
    }
    VOXCPM_ASSERT(backend_->compute(cached.graph) == GGML_STATUS_SUCCESS);
    maybe_collect_graph(cached.graph);

    std::vector<float> out(static_cast<size_t>(config_.feat_dim * config_.patch_size));
    backend_->tensor_get(cached.output, out.data(), 0, out.size() * sizeof(float));
    return out;
}

void VoxCPMRuntime::run_decode_front_half(const std::vector<float>& z,
                                          const std::vector<float>& lm_hidden,
                                          const std::vector<float>& residual_hidden,
                                          const std::vector<float>& prefix_feat_cond,
                                          int inference_timesteps,
                                          float cfg_value,
                                          std::vector<float>& output_0,
                                          std::vector<float>* curr_embed) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(components_ != nullptr);
    VOXCPM_ASSERT(feat_decoder_ != nullptr);
    VOXCPM_ASSERT(static_cast<int>(z.size()) == config_.feat_dim * config_.patch_size);
    VOXCPM_ASSERT(static_cast<int>(lm_hidden.size()) == base_lm_.config().hidden_size);
    VOXCPM_ASSERT(static_cast<int>(residual_hidden.size()) == residual_lm_.config().hidden_size);
    VOXCPM_ASSERT(static_cast<int>(prefix_feat_cond.size()) == config_.patch_size * config_.feat_dim);

    VoxCPMCachedGraph& cached = ensure_decode_front_half_graph(inference_timesteps, cfg_value);
    backend_->alloc_graph(cached.graph, "runtime.decode_front_half.cached");
    backend_->tensor_set(cached.input0, z.data(), 0, z.size() * sizeof(float));
    backend_->tensor_set(cached.input1, lm_hidden.data(), 0, lm_hidden.size() * sizeof(float));
    backend_->tensor_set(cached.input2, residual_hidden.data(), 0, residual_hidden.size() * sizeof(float));
    backend_->tensor_set(cached.input3,
                         prefix_feat_cond.data(),
                         0,
                         prefix_feat_cond.size() * sizeof(float));
    if (cached.input4 && !cached.aux_input4.empty()) {
        backend_->tensor_set(cached.input4, cached.aux_input4.data(), 0, cached.aux_input4.size() * sizeof(float));
    }
    VOXCPM_ASSERT(backend_->compute(cached.graph) == GGML_STATUS_SUCCESS);
    maybe_collect_graph(cached.graph);

    output_0.resize(static_cast<size_t>(config_.feat_dim * config_.patch_size));
    backend_->tensor_get(cached.output, output_0.data(), 0, output_0.size() * sizeof(float));
    if (curr_embed != nullptr && cached.output_aux0 != nullptr) {
        curr_embed->resize(static_cast<size_t>(base_lm_.config().hidden_size));
        backend_->tensor_get(cached.output_aux0, curr_embed->data(), 0, curr_embed->size() * sizeof(float));
    }
}

std::vector<float> VoxCPMRuntime::run_locenc_patch_to_lm_embed(const std::vector<float>& patch) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(components_ != nullptr);
    VOXCPM_ASSERT(static_cast<int>(patch.size()) == config_.feat_dim * config_.patch_size);
    VoxCPMCachedGraph& cached = ensure_locenc_patch_to_lm_embed_graph();
    backend_->alloc_graph(cached.graph, "runtime.locenc_to_lm_embed.cached");
    backend_->tensor_set(cached.input0, patch.data(), 0, patch.size() * sizeof(float));
    VOXCPM_ASSERT(backend_->compute(cached.graph) == GGML_STATUS_SUCCESS);
    maybe_collect_graph(cached.graph);

    std::vector<float> out(static_cast<size_t>(base_lm_.config().hidden_size));
    backend_->tensor_get(cached.output, out.data(), 0, out.size() * sizeof(float));
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

    const bool log_decode_timing = env_flag_enabled("VOXCPM_LOG_DECODE_TIMING");
    const bool log_decode_transfers = env_flag_enabled("VOXCPM_LOG_DECODE_TRANSFERS");
    using clock = std::chrono::steady_clock;
    const auto decode_start = clock::now();
    const BackendTransferStats transfers_before = backend_->transfer_stats();

    VoxCPMDecodeResult result;
    std::vector<float> curr_embed;
    run_decode_front_half(z,
                          state.lm_hidden,
                          state.residual_hidden,
                          state.prefix_feat_cond,
                          inference_timesteps,
                          cfg_value,
                          result.output_0,
                          &curr_embed);
    const auto front_half_end = clock::now();
    const BackendTransferStats transfers_after_front_half = backend_->transfer_stats();

    const int new_position = state.current_position + 1;
    const std::vector<float> stop_logits = run_stop_predictor(state.lm_hidden);
    result.output_2 = stop_logits.size() >= 2 && stop_logits[1] > stop_logits[0];
    const auto stop_end = clock::now();
    const BackendTransferStats transfers_after_stop = backend_->transfer_stats();

    const auto patch_end = clock::now();
    const BackendTransferStats transfers_after_patch = backend_->transfer_stats();

    VoxCPMCachedGraph& base_step = ensure_state_base_lm_step_graph(state, new_position);
    backend_->alloc_graph(base_step.graph, "runtime.base_lm.decode_step.state_cached");
    backend_->tensor_set(base_step.input0, curr_embed.data(), 0, curr_embed.size() * sizeof(float));
    const int32_t position_value = new_position;
    backend_->tensor_set(base_step.input1, &position_value, 0, sizeof(position_value));
    VOXCPM_ASSERT(backend_->compute(base_step.graph) == GGML_STATUS_SUCCESS);
    maybe_collect_graph(base_step.graph);
    std::vector<float> lm_hidden(static_cast<size_t>(base_lm_.config().hidden_size));
    backend_->tensor_get(base_step.output, lm_hidden.data(), 0, lm_hidden.size() * sizeof(float));
    const auto base_end = clock::now();
    const BackendTransferStats transfers_after_base = backend_->transfer_stats();

    std::vector<float> residual_input(curr_embed.size(), 0.0f);
    for (size_t i = 0; i < residual_input.size(); ++i) {
        residual_input[i] = lm_hidden[i] + curr_embed[i];
    }

    VoxCPMCachedGraph& residual_step = ensure_state_residual_lm_step_graph(state, new_position);
    backend_->alloc_graph(residual_step.graph, "runtime.residual_lm.decode_step.state_cached");
    backend_->tensor_set(residual_step.input0, residual_input.data(), 0, residual_input.size() * sizeof(float));
    backend_->tensor_set(residual_step.input1, &position_value, 0, sizeof(position_value));
    VOXCPM_ASSERT(backend_->compute(residual_step.graph) == GGML_STATUS_SUCCESS);
    maybe_collect_graph(residual_step.graph);
    std::vector<float> residual_hidden(static_cast<size_t>(residual_lm_.config().hidden_size));
    backend_->tensor_get(residual_step.output, residual_hidden.data(), 0, residual_hidden.size() * sizeof(float));
    const auto residual_end = clock::now();
    const BackendTransferStats transfers_after_residual = backend_->transfer_stats();

    if (log_decode_timing) {
        const auto ms = [](clock::time_point begin, clock::time_point end) {
            return std::chrono::duration<double, std::milli>(end - begin).count();
        };
        std::cerr << "[decode_timing]"
                  << " position=" << new_position
                  << " front_half_ms=" << ms(decode_start, front_half_end)
                  << " stop_ms=" << ms(front_half_end, stop_end)
                  << " patch_embed_ms=" << ms(stop_end, patch_end)
                  << " base_step_ms=" << ms(patch_end, base_end)
                  << " residual_step_ms=" << ms(base_end, residual_end)
                  << " total_ms=" << ms(decode_start, residual_end)
                  << "\n";
    }

    if (log_decode_transfers) {
        std::cerr << "[decode_transfer]"
                  << " position=" << new_position
                  << " front_half{" << format_transfer_stats(transfer_stats_delta(transfers_before, transfers_after_front_half)) << "}"
                  << " stop{" << format_transfer_stats(transfer_stats_delta(transfers_after_front_half, transfers_after_stop)) << "}"
                  << " patch_embed{" << format_transfer_stats(transfer_stats_delta(transfers_after_stop, transfers_after_patch)) << "}"
                  << " base_step{" << format_transfer_stats(transfer_stats_delta(transfers_after_patch, transfers_after_base)) << "}"
                  << " residual_step{" << format_transfer_stats(transfer_stats_delta(transfers_after_base, transfers_after_residual)) << "}"
                  << "\n";
    }

    state.lm_hidden = std::move(lm_hidden);
    state.residual_hidden = std::move(residual_hidden);
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
    copy.base_lm_step_graph_position = -1;
    copy.residual_lm_step_graph_position = -1;
    return copy;
}

}  // namespace voxcpm
