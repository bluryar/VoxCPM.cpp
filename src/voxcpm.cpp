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

VoxCPMContext make_sequence_graph_ctx(int seq_len, int min_tensors, int min_nodes) {
    constexpr size_t kSequenceGraphHeadroomBytes = 16 * 1024 * 1024;
    const int safe_seq_len = std::max(0, seq_len);
    // Seq-dependent graphs can land slightly short when ggml expands internal
    // bookkeeping for attention/rope-heavy paths, so keep a fixed cushion.
    const int n_tensors = min_tensors + safe_seq_len * 64 + 1024;
    const int max_nodes = min_nodes + safe_seq_len * 256 + 8192;
    return VoxCPMContext(ContextType::Graph, n_tensors, max_nodes, kSequenceGraphHeadroomBytes);
}

std::string decode_graph_key(int n_timesteps, float cfg_value) {
    uint32_t cfg_bits = 0;
    static_assert(sizeof(cfg_bits) == sizeof(cfg_value), "float size mismatch");
    std::memcpy(&cfg_bits, &cfg_value, sizeof(cfg_bits));

    std::ostringstream oss;
    oss << n_timesteps << ":" << cfg_bits;
    return oss.str();
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

bool has_complete_host_state(const VoxCPMDecodeState& state,
                             int lm_hidden_size,
                             int residual_hidden_size,
                             int patch_elem_count) {
    return static_cast<int>(state.lm_hidden.size()) == lm_hidden_size &&
           static_cast<int>(state.residual_hidden.size()) == residual_hidden_size &&
           static_cast<int>(state.prefix_feat_cond.size()) == patch_elem_count;
}

std::string format_transfer_stats(const BackendTransferStats& stats) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3)
        << "h2d=" << bytes_to_mib(stats.host_to_device_bytes) << "MiB/" << stats.host_to_device_ms << "ms"
        << " d2h=" << bytes_to_mib(stats.device_to_host_bytes) << "MiB/" << stats.device_to_host_ms << "ms"
        << " d2d=" << bytes_to_mib(stats.device_to_device_bytes) << "MiB/" << stats.device_to_device_ms << "ms";
    return oss.str();
}

struct TrailingPromptSpan {
    int start = 0;
    int count = 0;
};

TrailingPromptSpan find_trailing_prompt_span(const std::vector<int32_t>& feat_mask, int seq_len) {
    VOXCPM_ASSERT(static_cast<int>(feat_mask.size()) == seq_len);
    TrailingPromptSpan span;
    int end = seq_len;
    while (end > 0 && feat_mask[static_cast<size_t>(end - 1)] == 0) {
        --end;
    }
    if (end == 0) {
        return span;
    }
    int start = end - 1;
    while (start > 0 && feat_mask[static_cast<size_t>(start - 1)] != 0) {
        --start;
    }
    span.start = start;
    span.count = end - start;
    return span;
}

TrailingPromptSpan find_trailing_prompt_span(const std::vector<float>& feat_mask, int seq_len) {
    VOXCPM_ASSERT(static_cast<int>(feat_mask.size()) == seq_len);
    TrailingPromptSpan span;
    int end = seq_len;
    while (end > 0 && feat_mask[static_cast<size_t>(end - 1)] == 0.0f) {
        --end;
    }
    if (end == 0) {
        return span;
    }
    int start = end - 1;
    while (start > 0 && feat_mask[static_cast<size_t>(start - 1)] != 0.0f) {
        --start;
    }
    span.start = start;
    span.count = end - start;
    return span;
}

ggml_tensor* make_feature_sequence_patch_span_view(ggml_context* ctx,
                                                   ggml_tensor* feat_src,
                                                   const VoxCPMConfig& config,
                                                   int frame_offset,
                                                   int frame_count) {
    if (ctx == nullptr || feat_src == nullptr || frame_offset < 0 || frame_count <= 0) {
        return nullptr;
    }
    if (feat_src->type != GGML_TYPE_F32) {
        return nullptr;
    }
    if (feat_src->ne[0] != config.feat_dim || feat_src->ne[1] != config.patch_size) {
        return nullptr;
    }
    if (frame_offset + frame_count > feat_src->ne[2]) {
        return nullptr;
    }
    return ggml_view_2d(ctx,
                        feat_src,
                        config.feat_dim,
                        config.patch_size * frame_count,
                        feat_src->nb[1],
                        static_cast<size_t>(frame_offset) * feat_src->nb[2]);
}

ggml_tensor* build_residual_fusion_input(VoxCPMContext& ctx,
                                         VoxCPMComponents& components,
                                         ggml_tensor* blended_output,
                                         ggml_tensor* feat_embed_part) {
    VOXCPM_ASSERT(blended_output != nullptr);
    VOXCPM_ASSERT(feat_embed_part != nullptr);

    ggml_context* raw = ctx.raw_context();
    if (LinearProjection* fusion = components.fusion_concat_proj()) {
        ggml_tensor* fused_input = ggml_concat(raw, blended_output, feat_embed_part, 0);
        return fusion->forward(ctx, fused_input);
    }
    return ggml_add(raw, blended_output, feat_embed_part);
}

ggml_tensor* build_dit_mu(VoxCPMContext& ctx,
                          VoxCPMComponents& components,
                          ggml_tensor* lm_hidden,
                          ggml_tensor* residual_hidden) {
    VOXCPM_ASSERT(lm_hidden != nullptr);
    VOXCPM_ASSERT(residual_hidden != nullptr);

    ggml_context* raw = ctx.raw_context();
    ggml_tensor* dit_hidden_1 = components.lm_to_dit_proj()->forward(ctx, lm_hidden);
    ggml_tensor* dit_hidden_2 = components.res_to_dit_proj()->forward(ctx, residual_hidden);
    if (components.fusion_concat_proj()) {
        return ggml_concat(raw, dit_hidden_1, dit_hidden_2, 0);
    }
    return ggml_add(raw, dit_hidden_1, dit_hidden_2);
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

void VoxCPMRuntime::reset_request_state() {
    clear_cached_graphs();
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
    for (auto& entry : locenc_sequence_to_lm_projection_graphs_) {
        entry.second.clear();
    }
    locenc_sequence_to_lm_projection_graphs_.clear();
    for (auto& entry : locenc_sequence_to_lm_projection_fsq_graphs_) {
        entry.second.clear();
    }
    locenc_sequence_to_lm_projection_fsq_graphs_.clear();
    for (auto& entry : embedding_masked_locenc_sequence_to_lm_projection_graphs_) {
        entry.second.clear();
    }
    embedding_masked_locenc_sequence_to_lm_projection_graphs_.clear();
    for (auto& entry : embedding_masked_locenc_sequence_to_lm_projection_fsq_graphs_) {
        entry.second.clear();
    }
    embedding_masked_locenc_sequence_to_lm_projection_fsq_graphs_.clear();
    for (auto& entry : masked_fsq_blend_graphs_) {
        entry.second.clear();
    }
    masked_fsq_blend_graphs_.clear();
    for (auto& entry : embedding_graphs_) {
        entry.second.clear();
    }
    embedding_graphs_.clear();
    for (auto& entry : enc_to_lm_projection_graphs_) {
        entry.second.clear();
    }
    enc_to_lm_projection_graphs_.clear();
    for (auto& entry : enc_to_lm_projection_fsq_graphs_) {
        entry.second.clear();
    }
    enc_to_lm_projection_fsq_graphs_.clear();
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
    cached.context = std::make_unique<VoxCPMContext>(
        make_sequence_graph_ctx(seq_len, 4096, 32768));
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

VoxCPMCachedGraph& VoxCPMRuntime::ensure_embedding_masked_locenc_sequence_to_lm_projection_graph(int seq_len) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(components_ != nullptr);
    VOXCPM_ASSERT(seq_len >= 0);

    auto [it, inserted] = embedding_masked_locenc_sequence_to_lm_projection_graphs_.try_emplace(seq_len);
    VoxCPMCachedGraph& cached = it->second;
    if (!inserted && cached.graph) {
        return cached;
    }

    cached.clear();
    cached.context = std::make_unique<VoxCPMContext>(
        make_sequence_graph_ctx(seq_len, 65536, 524288));
    VoxCPMContext& graph_ctx = *cached.context;
    ggml_context* raw = graph_ctx.raw_context();

    cached.input0 = graph_ctx.new_tensor_1d(GGML_TYPE_I32, seq_len);
    cached.input1 = graph_ctx.new_tensor_3d(GGML_TYPE_F32, config_.feat_dim, config_.patch_size, seq_len);
    cached.input2 = graph_ctx.new_tensor_1d(GGML_TYPE_F32, seq_len);
    cached.input3 = graph_ctx.new_tensor_1d(GGML_TYPE_F32, seq_len);
    ggml_set_input(cached.input0);
    ggml_set_input(cached.input1);
    ggml_set_input(cached.input2);
    ggml_set_input(cached.input3);

    ggml_tensor* text_embed = components_->embed_tokens()->forward(graph_ctx, cached.input0);
    ggml_tensor* feat_hidden = feat_encoder_.forward_sequence(graph_ctx, cached.input1);
    ggml_tensor* feat_embed = components_->enc_to_lm_proj()->forward(graph_ctx, feat_hidden);

    ggml_tensor* text_mask_2d = ggml_reshape_2d(raw, cached.input2, 1, seq_len);
    ggml_tensor* feat_mask_2d = ggml_reshape_2d(raw, cached.input3, 1, seq_len);
    ggml_tensor* text_mask_broadcast = ggml_repeat(raw, text_mask_2d, text_embed);
    ggml_tensor* feat_mask_broadcast = ggml_repeat(raw, feat_mask_2d, feat_embed);

    ggml_tensor* text_part = ggml_mul(raw, text_embed, text_mask_broadcast);
    ggml_tensor* feat_part = ggml_mul(raw, feat_embed, feat_mask_broadcast);
    cached.output = ggml_add(raw, text_part, feat_part);
    ggml_set_output(cached.output);

    cached.graph = graph_ctx.new_graph();
    graph_ctx.build_forward(cached.graph, cached.output);
    backend_->reserve_compute_memory(cached.graph, "runtime.embedding_masked_locenc_to_lm.cached");
    return cached;
}

VoxCPMCachedGraph& VoxCPMRuntime::ensure_locenc_sequence_to_lm_projection_graph(int seq_len) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(components_ != nullptr);
    VOXCPM_ASSERT(seq_len >= 0);

    auto [it, inserted] = locenc_sequence_to_lm_projection_graphs_.try_emplace(seq_len);
    VoxCPMCachedGraph& cached = it->second;
    if (!inserted && cached.graph) {
        return cached;
    }

    cached.clear();
    cached.context = std::make_unique<VoxCPMContext>(ContextType::Graph, 65536, 524288);
    VoxCPMContext& graph_ctx = *cached.context;
    cached.input0 = graph_ctx.new_tensor_3d(GGML_TYPE_F32, config_.feat_dim, config_.patch_size, seq_len);
    ggml_set_input(cached.input0);
    ggml_tensor* encoded = feat_encoder_.forward_sequence(graph_ctx, cached.input0);
    cached.output = components_->enc_to_lm_proj()->forward(graph_ctx, encoded);
    ggml_set_output(cached.output);

    cached.graph = graph_ctx.new_graph();
    graph_ctx.build_forward(cached.graph, cached.output);
    backend_->reserve_compute_memory(cached.graph, "runtime.locenc.sequence_to_lm.cached");
    return cached;
}

VoxCPMCachedGraph& VoxCPMRuntime::ensure_embedding_masked_locenc_sequence_to_lm_projection_fsq_graph(int seq_len) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(components_ != nullptr);
    VOXCPM_ASSERT(seq_len >= 0);

    auto [it, inserted] = embedding_masked_locenc_sequence_to_lm_projection_fsq_graphs_.try_emplace(seq_len);
    VoxCPMCachedGraph& cached = it->second;
    if (!inserted && cached.graph) {
        return cached;
    }

    cached.clear();
    cached.context = std::make_unique<VoxCPMContext>(ContextType::Graph, 65536, 524288);
    VoxCPMContext& graph_ctx = *cached.context;
    ggml_context* raw = graph_ctx.raw_context();

    cached.input0 = graph_ctx.new_tensor_1d(GGML_TYPE_I32, seq_len);
    cached.input1 = graph_ctx.new_tensor_3d(GGML_TYPE_F32, config_.feat_dim, config_.patch_size, seq_len);
    cached.input2 = graph_ctx.new_tensor_1d(GGML_TYPE_F32, seq_len);
    cached.input3 = graph_ctx.new_tensor_1d(GGML_TYPE_F32, seq_len);
    ggml_set_input(cached.input0);
    ggml_set_input(cached.input1);
    ggml_set_input(cached.input2);
    ggml_set_input(cached.input3);

    ggml_tensor* text_embed = components_->embed_tokens()->forward(graph_ctx, cached.input0);
    ggml_tensor* feat_hidden = feat_encoder_.forward_sequence(graph_ctx, cached.input1);
    ggml_tensor* feat_embed = components_->enc_to_lm_proj()->forward(graph_ctx, feat_hidden);

    ggml_tensor* text_mask_2d = ggml_reshape_2d(raw, cached.input2, 1, seq_len);
    ggml_tensor* feat_mask_2d = ggml_reshape_2d(raw, cached.input3, 1, seq_len);
    ggml_tensor* text_mask_broadcast = ggml_repeat(raw, text_mask_2d, text_embed);
    ggml_tensor* feat_mask_broadcast = ggml_repeat(raw, feat_mask_2d, feat_embed);

    ggml_tensor* text_part = ggml_mul(raw, text_embed, text_mask_broadcast);
    ggml_tensor* feat_part = ggml_mul(raw, feat_embed, feat_mask_broadcast);
    ggml_tensor* combined = ggml_add(raw, text_part, feat_part);
    cached.output = fsq_layer_.forward(graph_ctx, combined);
    ggml_set_output(cached.output);

    cached.graph = graph_ctx.new_graph();
    graph_ctx.build_forward(cached.graph, cached.output);
    backend_->reserve_compute_memory(cached.graph, "runtime.embedding_masked_locenc_to_lm_fsq.cached");
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
    cached.context = std::make_unique<VoxCPMContext>(ContextType::Graph, 65536, 524288);
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

VoxCPMCachedGraph& VoxCPMRuntime::ensure_masked_fsq_blend_graph(int seq_len) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(seq_len >= 0);

    auto [it, inserted] = masked_fsq_blend_graphs_.try_emplace(seq_len);
    VoxCPMCachedGraph& cached = it->second;
    if (!inserted && cached.graph) {
        return cached;
    }

    cached.clear();
    cached.context = std::make_unique<VoxCPMContext>(make_sequence_graph_ctx(seq_len, 16384, 131072));
    VoxCPMContext& graph_ctx = *cached.context;
    ggml_context* raw = graph_ctx.raw_context();

    cached.input0 = graph_ctx.new_tensor_2d(GGML_TYPE_F32, base_lm_.config().hidden_size, seq_len);
    cached.input1 = graph_ctx.new_tensor_1d(GGML_TYPE_F32, seq_len);
    cached.input2 = graph_ctx.new_tensor_1d(GGML_TYPE_F32, seq_len);
    ggml_set_input(cached.input0);
    ggml_set_input(cached.input1);
    ggml_set_input(cached.input2);

    ggml_tensor* fsq_out = fsq_layer_.forward(graph_ctx, cached.input0);
    ggml_tensor* text_mask_2d = ggml_reshape_2d(raw, cached.input1, 1, seq_len);
    ggml_tensor* feat_mask_2d = ggml_reshape_2d(raw, cached.input2, 1, seq_len);
    ggml_tensor* text_mask_broadcast = ggml_repeat(raw, text_mask_2d, cached.input0);
    ggml_tensor* feat_mask_broadcast = ggml_repeat(raw, feat_mask_2d, fsq_out);
    ggml_tensor* text_part = ggml_mul(raw, cached.input0, text_mask_broadcast);
    ggml_tensor* feat_part = ggml_mul(raw, fsq_out, feat_mask_broadcast);
    cached.output = ggml_add(raw, text_part, feat_part);
    ggml_set_output(cached.output);

    cached.graph = graph_ctx.new_graph();
    graph_ctx.build_forward(cached.graph, cached.output);
    backend_->reserve_compute_memory(cached.graph, "runtime.masked_fsq_blend.cached");
    return cached;
}

VoxCPMCachedGraph& VoxCPMRuntime::ensure_locenc_sequence_to_lm_projection_fsq_graph(int seq_len) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(components_ != nullptr);
    VOXCPM_ASSERT(seq_len >= 0);

    auto [it, inserted] = locenc_sequence_to_lm_projection_fsq_graphs_.try_emplace(seq_len);
    VoxCPMCachedGraph& cached = it->second;
    if (!inserted && cached.graph) {
        return cached;
    }

    cached.clear();
    cached.context = std::make_unique<VoxCPMContext>(ContextType::Graph, 65536, 524288);
    VoxCPMContext& graph_ctx = *cached.context;
    cached.input0 = graph_ctx.new_tensor_3d(GGML_TYPE_F32, config_.feat_dim, config_.patch_size, seq_len);
    ggml_set_input(cached.input0);
    ggml_tensor* encoded = feat_encoder_.forward_sequence(graph_ctx, cached.input0);
    ggml_tensor* projected = components_->enc_to_lm_proj()->forward(graph_ctx, encoded);
    cached.output = fsq_layer_.forward(graph_ctx, projected);
    ggml_set_output(cached.output);

    cached.graph = graph_ctx.new_graph();
    graph_ctx.build_forward(cached.graph, cached.output);
    backend_->reserve_compute_memory(cached.graph, "runtime.locenc.sequence_to_lm_fsq.cached");
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
    cached.context = std::make_unique<VoxCPMContext>(make_sequence_graph_ctx(seq_len, 8192, 65536));
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

VoxCPMCachedGraph& VoxCPMRuntime::ensure_enc_to_lm_projection_fsq_graph(int seq_len) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(components_ != nullptr);
    VOXCPM_ASSERT(seq_len >= 0);

    auto [it, inserted] = enc_to_lm_projection_fsq_graphs_.try_emplace(seq_len);
    VoxCPMCachedGraph& cached = it->second;
    if (!inserted && cached.graph) {
        return cached;
    }

    cached.clear();
    cached.context = std::make_unique<VoxCPMContext>(make_sequence_graph_ctx(seq_len, 8192, 65536));
    VoxCPMContext& graph_ctx = *cached.context;
    cached.input0 = graph_ctx.new_tensor_2d(GGML_TYPE_F32, feat_encoder_.config().hidden_size, seq_len);
    ggml_set_input(cached.input0);
    ggml_tensor* projected = components_->enc_to_lm_proj()->forward(graph_ctx, cached.input0);
    cached.output = fsq_layer_.forward(graph_ctx, projected);
    ggml_set_output(cached.output);

    cached.graph = graph_ctx.new_graph();
    graph_ctx.build_forward(cached.graph, cached.output);
    backend_->reserve_compute_memory(cached.graph, "runtime.enc_to_lm_proj_fsq.cached");
    return cached;
}

VoxCPMCachedGraph& VoxCPMRuntime::ensure_unified_cfm_graph(int n_timesteps, float cfg_value) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(feat_decoder_ != nullptr);
    VOXCPM_ASSERT(components_ != nullptr);

    const std::string key = decode_graph_key(n_timesteps, cfg_value);
    auto [it, inserted] = unified_cfm_graphs_.try_emplace(key);
    VoxCPMCachedGraph& cached = it->second;
    if (!inserted && cached.graph) {
        return cached;
    }

    cached.clear();
    cached.context = std::make_unique<VoxCPMContext>(ContextType::Graph, 65536, 524288);
    VoxCPMContext& graph_ctx = *cached.context;
    const int mu_dim = config_.loc_dit.hidden_size * (components_->fusion_concat_proj() ? 2 : 1);
    cached.input0 = graph_ctx.new_tensor_2d(GGML_TYPE_F32, config_.feat_dim, config_.patch_size);
    cached.input1 = graph_ctx.new_tensor_1d(GGML_TYPE_F32, mu_dim);
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

    ggml_tensor* dit_hidden = build_dit_mu(graph_ctx, *components_, cached.input1, cached.input2);
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
    VOXCPM_ASSERT(cached.output_aux0 != nullptr);
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
    if (!base_lm_.config().no_rope) {
        cached.input1 = graph_ctx.new_tensor_1d(GGML_TYPE_I32, 1);
    }
    ggml_set_input(cached.input0);
    if (cached.input1 != nullptr) {
        ggml_set_input(cached.input1);
    }

    ggml_tensor* hidden = base_lm_.forward_step(
        graph_ctx, cached.input0, position, cached.input1, *state.base_lm_cache, true);
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
    if (!residual_lm_.config().no_rope) {
        cached.input1 = graph_ctx.new_tensor_1d(GGML_TYPE_I32, 1);
    }
    cached.input2 = graph_ctx.new_tensor_1d(GGML_TYPE_F32, residual_lm_.config().hidden_size);
    ggml_set_input(cached.input0);
    if (cached.input1 != nullptr) {
        ggml_set_input(cached.input1);
    }
    ggml_set_input(cached.input2);

    ggml_tensor* residual_input = build_residual_fusion_input(graph_ctx, *components_, cached.input0, cached.input2);
    cached.output = residual_lm_.forward_step(graph_ctx,
                                              residual_input,
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

VoxCPMDecodeState VoxCPMRuntime::create_decode_state_internal(bool initialize_host_shadow) const {
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
    state.persistent_state = std::make_unique<VoxCPMPersistentState>();
    VOXCPM_ASSERT(state.persistent_state->initialize(*backend_,
                                                     PersistentStateShape{
                                                         base_lm_.config().hidden_size,
                                                         config_.feat_dim,
                                                         config_.patch_size,
                                                     }));
    state.output_pool = std::make_unique<VoxCPMOutputPool>();
    VOXCPM_ASSERT(state.output_pool->initialize(*backend_,
                                                OutputPoolShape{
                                                    config_.feat_dim,
                                                    config_.patch_size,
                                                    std::max(1, config_.max_length),
                                                }));
    if (initialize_host_shadow) {
        state.lm_hidden.assign(static_cast<size_t>(base_lm_.config().hidden_size), 0.0f);
        state.residual_hidden.assign(static_cast<size_t>(residual_lm_.config().hidden_size), 0.0f);
        state.prefix_feat_cond.assign(static_cast<size_t>(config_.patch_size * config_.feat_dim), 0.0f);
    }
    return state;
}

VoxCPMDecodeState VoxCPMRuntime::create_decode_state() const {
    return create_decode_state_internal(false);
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

std::vector<float> VoxCPMRuntime::run_locenc_patch_from_tensor(ggml_tensor* patch_src) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(patch_src != nullptr);
    VOXCPM_ASSERT(static_cast<int>(ggml_nelements(patch_src)) == config_.feat_dim * config_.patch_size);

    VoxCPMCachedGraph& cached = ensure_locenc_patch_graph();
    backend_->alloc_graph(cached.graph, "runtime.locenc.patch.cached");
    backend_->tensor_copy(patch_src, cached.input0);
    VOXCPM_ASSERT(backend_->compute(cached.graph) == GGML_STATUS_SUCCESS);
    maybe_collect_graph(cached.graph);

    std::vector<float> out(static_cast<size_t>(base_lm_.config().hidden_size));
    backend_->tensor_get(cached.output, out.data(), 0, out.size() * sizeof(float));
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

void VoxCPMRuntime::sync_host_state_to_persistent(VoxCPMDecodeState& state) const {
    if (backend_ == nullptr || state.persistent_state == nullptr || !state.persistent_state->is_initialized()) {
        return;
    }

    if (!state.lm_hidden.empty()) {
        state.persistent_state->set_lm_hidden_from_host(*backend_, state.lm_hidden.data(), state.lm_hidden.size());
    }
    if (!state.residual_hidden.empty()) {
        state.persistent_state->set_residual_hidden_from_host(*backend_,
                                                              state.residual_hidden.data(),
                                                              state.residual_hidden.size());
    }
    if (!state.prefix_feat_cond.empty()) {
        state.persistent_state->set_prefix_patch_from_host(*backend_,
                                                           state.prefix_feat_cond.data(),
                                                           state.prefix_feat_cond.size());
    }
}

void VoxCPMRuntime::sync_persistent_to_host(VoxCPMDecodeState& state, bool include_prefix_patch) const {
    if (backend_ == nullptr || state.persistent_state == nullptr || !state.persistent_state->is_initialized()) {
        return;
    }

    const size_t lm_count = static_cast<size_t>(base_lm_.config().hidden_size);
    if (state.lm_hidden.size() != lm_count) {
        state.lm_hidden.resize(lm_count);
    }
    state.persistent_state->get_lm_hidden_to_host(*backend_, state.lm_hidden.data(), state.lm_hidden.size());

    const size_t residual_count = static_cast<size_t>(residual_lm_.config().hidden_size);
    if (state.residual_hidden.size() != residual_count) {
        state.residual_hidden.resize(residual_count);
    }
    state.persistent_state->get_residual_hidden_to_host(*backend_,
                                                        state.residual_hidden.data(),
                                                        state.residual_hidden.size());

    if (include_prefix_patch) {
        const size_t prefix_count = static_cast<size_t>(config_.patch_size * config_.feat_dim);
        if (state.prefix_feat_cond.size() != prefix_count) {
            state.prefix_feat_cond.resize(prefix_count);
        }
        state.persistent_state->get_prefix_patch_to_host(*backend_,
                                                         state.prefix_feat_cond.data(),
                                                         state.prefix_feat_cond.size());
    } else {
        std::vector<float>().swap(state.prefix_feat_cond);
    }
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

std::vector<float> VoxCPMRuntime::run_embedding_from_tensor(ggml_tensor* token_src, int token_count) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(token_src != nullptr);
    VOXCPM_ASSERT(token_src->type == GGML_TYPE_I32);
    VOXCPM_ASSERT(static_cast<int>(ggml_nelements(token_src)) == token_count);

    VoxCPMCachedGraph& cached = ensure_embedding_graph(token_count);
    backend_->alloc_graph(cached.graph, "runtime.embedding.cached");
    backend_->tensor_copy(token_src, cached.input0);
    VOXCPM_ASSERT(backend_->compute(cached.graph) == GGML_STATUS_SUCCESS);
    maybe_collect_graph(cached.graph);

    std::vector<float> out(static_cast<size_t>(components_->embed_tokens()->config().hidden_dim) * token_count);
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

std::vector<float> VoxCPMRuntime::run_projection_1d_from_tensor(LinearProjection& projection,
                                                                ggml_tensor* input_src,
                                                                int in_dim,
                                                                int out_dim) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(input_src != nullptr);
    VOXCPM_ASSERT(static_cast<int>(ggml_nelements(input_src)) == in_dim);

    VoxCPMContext graph_ctx = make_graph_ctx(4096, 32768);
    ggml_tensor* input_tensor = graph_ctx.new_tensor_1d(GGML_TYPE_F32, in_dim);
    ggml_set_input(input_tensor);

    ggml_tensor* output = projection.forward(graph_ctx, input_tensor);
    ggml_set_output(output);

    ggml_cgraph* graph = graph_ctx.new_graph();
    graph_ctx.build_forward(graph, output);
    backend_->reserve_compute_memory(graph, "runtime.proj.1d.from_tensor");
    backend_->alloc_graph(graph, "runtime.proj.1d.from_tensor");
    backend_->tensor_copy(input_src, input_tensor);
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

std::array<float, 2> VoxCPMRuntime::run_stop_predictor_from_state(const VoxCPMDecodeState& state,
                                                                 bool publish_to_output_pool) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VoxCPMCachedGraph& cached = ensure_stop_predictor_graph();
    backend_->alloc_graph(cached.graph, "runtime.stop_predictor.cached");

    if (state.persistent_state != nullptr && state.persistent_state->is_initialized()) {
        backend_->tensor_copy(state.persistent_state->lm_hidden(), cached.input0);
    } else {
        VOXCPM_ASSERT(static_cast<int>(state.lm_hidden.size()) == base_lm_.config().hidden_size);
        backend_->tensor_set(cached.input0, state.lm_hidden.data(), 0, state.lm_hidden.size() * sizeof(float));
    }

    VOXCPM_ASSERT(backend_->compute(cached.graph) == GGML_STATUS_SUCCESS);
    maybe_collect_graph(cached.graph);

    std::array<float, 2> out = {0.0f, 0.0f};
    backend_->tensor_get(cached.output, out.data(), 0, out.size() * sizeof(float));

    if (publish_to_output_pool &&
        state.output_pool != nullptr &&
        state.output_pool->is_initialized()) {
        state.output_pool->publish_stop_logits(*backend_, cached.output);
    }
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

std::vector<float> VoxCPMRuntime::run_fsq_1d_from_tensor(ggml_tensor* input_src) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(input_src != nullptr);
    VOXCPM_ASSERT(static_cast<int>(ggml_nelements(input_src)) == base_lm_.config().hidden_size);

    VoxCPMContext graph_ctx = make_graph_ctx(4096, 32768);
    ggml_tensor* input_tensor = graph_ctx.new_tensor_1d(GGML_TYPE_F32, base_lm_.config().hidden_size);
    ggml_set_input(input_tensor);

    ggml_tensor* input_view = ggml_reshape_2d(graph_ctx.raw_context(),
                                              input_tensor,
                                              base_lm_.config().hidden_size,
                                              1);
    ggml_tensor* output = fsq_layer_.forward(graph_ctx, input_view);
    output = ggml_reshape_1d(graph_ctx.raw_context(), output, output->ne[0]);
    ggml_set_output(output);

    ggml_cgraph* graph = graph_ctx.new_graph();
    graph_ctx.build_forward(graph, output);
    backend_->reserve_compute_memory(graph, "runtime.fsq.1d.from_tensor");
    backend_->alloc_graph(graph, "runtime.fsq.1d.from_tensor");
    backend_->tensor_copy(input_src, input_tensor);
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

std::vector<float> VoxCPMRuntime::run_masked_fsq_blend(const std::vector<float>& input,
                                                       const std::vector<float>& text_mask,
                                                       const std::vector<float>& feat_mask,
                                                       int seq_len) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(static_cast<int>(input.size()) == base_lm_.config().hidden_size * seq_len);
    VOXCPM_ASSERT(static_cast<int>(text_mask.size()) == seq_len);
    VOXCPM_ASSERT(static_cast<int>(feat_mask.size()) == seq_len);

    VoxCPMCachedGraph& cached = ensure_masked_fsq_blend_graph(seq_len);
    backend_->alloc_graph(cached.graph, "runtime.masked_fsq_blend.cached");
    backend_->tensor_set(cached.input0, input.data(), 0, input.size() * sizeof(float));
    backend_->tensor_set(cached.input1, text_mask.data(), 0, text_mask.size() * sizeof(float));
    backend_->tensor_set(cached.input2, feat_mask.data(), 0, feat_mask.size() * sizeof(float));
    VOXCPM_ASSERT(backend_->compute(cached.graph) == GGML_STATUS_SUCCESS);
    maybe_collect_graph(cached.graph);

    std::vector<float> out(static_cast<size_t>(base_lm_.config().hidden_size) * seq_len);
    backend_->tensor_get(cached.output, out.data(), 0, out.size() * sizeof(float));
    return out;
}

void VoxCPMRuntime::run_embedding_masked_locenc_sequence_to_lm_projection_into(
    const std::vector<int32_t>& token_ids,
    const std::vector<float>& feat,
    const std::vector<float>& text_mask,
    const std::vector<float>& feat_mask,
    int seq_len,
    ggml_tensor* output_dst) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(output_dst != nullptr);
    VOXCPM_ASSERT(static_cast<int>(token_ids.size()) == seq_len);
    VOXCPM_ASSERT(static_cast<int>(feat.size()) == config_.feat_dim * config_.patch_size * seq_len);
    VOXCPM_ASSERT(static_cast<int>(text_mask.size()) == seq_len);
    VOXCPM_ASSERT(static_cast<int>(feat_mask.size()) == seq_len);
    VOXCPM_ASSERT(static_cast<int>(ggml_nelements(output_dst)) == base_lm_.config().hidden_size * seq_len);

    VoxCPMCachedGraph& cached = ensure_embedding_masked_locenc_sequence_to_lm_projection_graph(seq_len);
    backend_->alloc_graph(cached.graph, "runtime.embedding_masked_locenc_to_lm.cached");
    backend_->tensor_set(cached.input0, token_ids.data(), 0, token_ids.size() * sizeof(int32_t));
    backend_->tensor_set(cached.input1, feat.data(), 0, feat.size() * sizeof(float));
    backend_->tensor_set(cached.input2, text_mask.data(), 0, text_mask.size() * sizeof(float));
    backend_->tensor_set(cached.input3, feat_mask.data(), 0, feat_mask.size() * sizeof(float));
    VOXCPM_ASSERT(backend_->compute(cached.graph) == GGML_STATUS_SUCCESS);
    maybe_collect_graph(cached.graph);

    backend_->tensor_copy(cached.output, output_dst);
}

void VoxCPMRuntime::run_embedding_masked_locenc_sequence_to_lm_projection_from_feat_tensor_into(
    const std::vector<int32_t>& token_ids,
    ggml_tensor* feat_src,
    const std::vector<float>& text_mask,
    const std::vector<float>& feat_mask,
    int seq_len,
    ggml_tensor* output_dst) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(feat_src != nullptr);
    VOXCPM_ASSERT(output_dst != nullptr);
    VOXCPM_ASSERT(static_cast<int>(token_ids.size()) == seq_len);
    VOXCPM_ASSERT(static_cast<int>(ggml_nelements(feat_src)) == config_.feat_dim * config_.patch_size * seq_len);
    VOXCPM_ASSERT(static_cast<int>(text_mask.size()) == seq_len);
    VOXCPM_ASSERT(static_cast<int>(feat_mask.size()) == seq_len);
    VOXCPM_ASSERT(static_cast<int>(ggml_nelements(output_dst)) == base_lm_.config().hidden_size * seq_len);

    VoxCPMCachedGraph& cached = ensure_embedding_masked_locenc_sequence_to_lm_projection_graph(seq_len);
    backend_->alloc_graph(cached.graph, "runtime.embedding_masked_locenc_to_lm.cached");
    backend_->tensor_set(cached.input0, token_ids.data(), 0, token_ids.size() * sizeof(int32_t));
    backend_->tensor_copy(feat_src, cached.input1);
    backend_->tensor_set(cached.input2, text_mask.data(), 0, text_mask.size() * sizeof(float));
    backend_->tensor_set(cached.input3, feat_mask.data(), 0, feat_mask.size() * sizeof(float));
    VOXCPM_ASSERT(backend_->compute(cached.graph) == GGML_STATUS_SUCCESS);
    maybe_collect_graph(cached.graph);

    backend_->tensor_copy(cached.output, output_dst);
}

void VoxCPMRuntime::run_embedding_masked_locenc_sequence_to_lm_projection_from_input_tensors_into(
    ggml_tensor* token_src,
    ggml_tensor* feat_src,
    ggml_tensor* text_mask_src,
    ggml_tensor* feat_mask_src,
    int seq_len,
    ggml_tensor* output_dst) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(token_src != nullptr);
    VOXCPM_ASSERT(feat_src != nullptr);
    VOXCPM_ASSERT(text_mask_src != nullptr);
    VOXCPM_ASSERT(feat_mask_src != nullptr);
    VOXCPM_ASSERT(output_dst != nullptr);
    VOXCPM_ASSERT(token_src->type == GGML_TYPE_I32);
    VOXCPM_ASSERT(static_cast<int>(ggml_nelements(token_src)) == seq_len);
    VOXCPM_ASSERT(static_cast<int>(ggml_nelements(feat_src)) == config_.feat_dim * config_.patch_size * seq_len);
    VOXCPM_ASSERT(static_cast<int>(ggml_nelements(text_mask_src)) == seq_len);
    VOXCPM_ASSERT(static_cast<int>(ggml_nelements(feat_mask_src)) == seq_len);
    VOXCPM_ASSERT(static_cast<int>(ggml_nelements(output_dst)) == base_lm_.config().hidden_size * seq_len);

    VoxCPMCachedGraph& cached = ensure_embedding_masked_locenc_sequence_to_lm_projection_graph(seq_len);
    backend_->alloc_graph(cached.graph, "runtime.embedding_masked_locenc_to_lm.cached");
    backend_->tensor_copy(token_src, cached.input0);
    backend_->tensor_copy(feat_src, cached.input1);
    backend_->tensor_copy(text_mask_src, cached.input2);
    backend_->tensor_copy(feat_mask_src, cached.input3);
    VOXCPM_ASSERT(backend_->compute(cached.graph) == GGML_STATUS_SUCCESS);
    maybe_collect_graph(cached.graph);

    backend_->tensor_copy(cached.output, output_dst);
}

std::pair<std::vector<float>, std::vector<float>> VoxCPMRuntime::run_prefill_base_to_residual_inputs(
    const std::vector<float>& combined_embed,
    const std::vector<float>& text_mask,
    const std::vector<float>& feat_mask,
    int seq_len,
    MiniCPMKVCache& kv_cache,
    bool is_causal) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(seq_len > 0);
    VOXCPM_ASSERT(static_cast<int>(combined_embed.size()) == base_lm_.config().hidden_size * seq_len);
    VOXCPM_ASSERT(static_cast<int>(text_mask.size()) == seq_len);
    VOXCPM_ASSERT(static_cast<int>(feat_mask.size()) == seq_len);

    VoxCPMContext graph_ctx = make_sequence_graph_ctx(seq_len, 65536, 524288);
    ggml_context* raw = graph_ctx.raw_context();
    ggml_tensor* embed_tensor = graph_ctx.new_tensor_2d(GGML_TYPE_F32, base_lm_.config().hidden_size, seq_len);
    ggml_tensor* text_mask_tensor = graph_ctx.new_tensor_1d(GGML_TYPE_F32, seq_len);
    ggml_tensor* feat_mask_tensor = graph_ctx.new_tensor_1d(GGML_TYPE_F32, seq_len);
    ggml_set_input(embed_tensor);
    ggml_set_input(text_mask_tensor);
    ggml_set_input(feat_mask_tensor);

    ggml_tensor* base_output = base_lm_.forward(graph_ctx, embed_tensor, nullptr, kv_cache, is_causal);
    ggml_tensor* fsq_out = fsq_layer_.forward(graph_ctx, base_output);
    ggml_tensor* text_mask_2d = ggml_reshape_2d(raw, text_mask_tensor, 1, seq_len);
    ggml_tensor* feat_mask_2d = ggml_reshape_2d(raw, feat_mask_tensor, 1, seq_len);
    ggml_tensor* text_mask_broadcast = ggml_repeat(raw, text_mask_2d, base_output);
    ggml_tensor* feat_mask_broadcast = ggml_repeat(raw, feat_mask_2d, base_output);
    ggml_tensor* text_part = ggml_mul(raw, base_output, text_mask_broadcast);
    ggml_tensor* feat_part = ggml_mul(raw, fsq_out, feat_mask_broadcast);
    ggml_tensor* blended_output = ggml_add(raw, text_part, feat_part);
    ggml_tensor* feat_embed_part = ggml_mul(raw, embed_tensor, feat_mask_broadcast);
    ggml_tensor* residual_inputs = build_residual_fusion_input(graph_ctx, *components_, blended_output, feat_embed_part);
    ggml_set_output(blended_output);
    ggml_set_output(residual_inputs);

    ggml_cgraph* graph = graph_ctx.new_graph();
    graph_ctx.build_forward(graph, residual_inputs);
    backend_->reserve_compute_memory(graph, "runtime.prefill.base_to_residual_inputs");
    backend_->alloc_graph(graph, "runtime.prefill.base_to_residual_inputs");
    backend_->tensor_set(embed_tensor, combined_embed.data(), 0, combined_embed.size() * sizeof(float));
    backend_->tensor_set(text_mask_tensor, text_mask.data(), 0, text_mask.size() * sizeof(float));
    backend_->tensor_set(feat_mask_tensor, feat_mask.data(), 0, feat_mask.size() * sizeof(float));
    VOXCPM_ASSERT(backend_->compute(graph) == GGML_STATUS_SUCCESS);
    maybe_collect_graph(graph);

    std::vector<float> blended(static_cast<size_t>(base_lm_.config().hidden_size) * seq_len);
    std::vector<float> residual(static_cast<size_t>(base_lm_.config().hidden_size) * seq_len);
    backend_->tensor_get(blended_output, blended.data(), 0, blended.size() * sizeof(float));
    backend_->tensor_get(residual_inputs, residual.data(), 0, residual.size() * sizeof(float));
    return {std::move(blended), std::move(residual)};
}

std::vector<float> VoxCPMRuntime::run_prefill_base_to_residual_inputs_from_tensor_with_last_hidden_into(
    ggml_tensor* combined_embed_src,
    const std::vector<float>& text_mask,
    const std::vector<float>& feat_mask,
    int seq_len,
    MiniCPMKVCache& kv_cache,
    bool is_causal,
    ggml_tensor* lm_hidden_dst) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(combined_embed_src != nullptr);
    VOXCPM_ASSERT(lm_hidden_dst != nullptr);
    VOXCPM_ASSERT(seq_len > 0);
    VOXCPM_ASSERT(static_cast<int>(ggml_nelements(combined_embed_src)) == base_lm_.config().hidden_size * seq_len);
    VOXCPM_ASSERT(static_cast<int>(text_mask.size()) == seq_len);
    VOXCPM_ASSERT(static_cast<int>(feat_mask.size()) == seq_len);

    VoxCPMContext graph_ctx = make_sequence_graph_ctx(seq_len, 65536, 524288);
    ggml_context* raw = graph_ctx.raw_context();
    ggml_tensor* embed_tensor = graph_ctx.new_tensor_2d(GGML_TYPE_F32, base_lm_.config().hidden_size, seq_len);
    ggml_tensor* text_mask_tensor = graph_ctx.new_tensor_1d(GGML_TYPE_F32, seq_len);
    ggml_tensor* feat_mask_tensor = graph_ctx.new_tensor_1d(GGML_TYPE_F32, seq_len);
    ggml_set_input(embed_tensor);
    ggml_set_input(text_mask_tensor);
    ggml_set_input(feat_mask_tensor);

    ggml_tensor* base_output = base_lm_.forward(graph_ctx, embed_tensor, nullptr, kv_cache, is_causal);
    ggml_tensor* fsq_out = fsq_layer_.forward(graph_ctx, base_output);
    ggml_tensor* text_mask_2d = ggml_reshape_2d(raw, text_mask_tensor, 1, seq_len);
    ggml_tensor* feat_mask_2d = ggml_reshape_2d(raw, feat_mask_tensor, 1, seq_len);
    ggml_tensor* text_mask_broadcast = ggml_repeat(raw, text_mask_2d, base_output);
    ggml_tensor* feat_mask_broadcast = ggml_repeat(raw, feat_mask_2d, base_output);
    ggml_tensor* text_part = ggml_mul(raw, base_output, text_mask_broadcast);
    ggml_tensor* feat_part = ggml_mul(raw, fsq_out, feat_mask_broadcast);
    ggml_tensor* blended_output = ggml_add(raw, text_part, feat_part);
    ggml_tensor* feat_embed_part = ggml_mul(raw, embed_tensor, feat_mask_broadcast);
    ggml_tensor* residual_inputs = build_residual_fusion_input(graph_ctx, *components_, blended_output, feat_embed_part);
    ggml_tensor* blended_last = ggml_view_1d(raw,
                                             blended_output,
                                             base_lm_.config().hidden_size,
                                             static_cast<size_t>(seq_len - 1) *
                                                 static_cast<size_t>(base_lm_.config().hidden_size) * sizeof(float));
    ggml_tensor* blended_last_copy = ggml_cont(raw, blended_last);
    ggml_set_output(residual_inputs);
    ggml_set_output(blended_last_copy);

    ggml_cgraph* graph = graph_ctx.new_graph();
    graph_ctx.build_forward(graph, residual_inputs);
    graph_ctx.build_forward(graph, blended_last_copy);
    backend_->reserve_compute_memory(graph, "runtime.prefill.base_to_residual_inputs.tensor.last_hidden");
    backend_->alloc_graph(graph, "runtime.prefill.base_to_residual_inputs.tensor.last_hidden");
    backend_->tensor_copy(combined_embed_src, embed_tensor);
    backend_->tensor_set(text_mask_tensor, text_mask.data(), 0, text_mask.size() * sizeof(float));
    backend_->tensor_set(feat_mask_tensor, feat_mask.data(), 0, feat_mask.size() * sizeof(float));
    VOXCPM_ASSERT(backend_->compute(graph) == GGML_STATUS_SUCCESS);
    maybe_collect_graph(graph);

    backend_->tensor_copy(blended_last_copy, lm_hidden_dst);

    std::vector<float> residual(static_cast<size_t>(base_lm_.config().hidden_size) * seq_len);
    backend_->tensor_get(residual_inputs, residual.data(), 0, residual.size() * sizeof(float));
    return residual;
}

void VoxCPMRuntime::run_prefill_hidden_states_from_tensor_into(ggml_tensor* combined_embed_src,
                                                               const std::vector<float>& text_mask,
                                                               const std::vector<float>& feat_mask,
                                                               int seq_len,
                                                               MiniCPMKVCache& base_kv_cache,
                                                               MiniCPMKVCache& residual_kv_cache,
                                                               bool is_causal,
                                                               ggml_tensor* lm_hidden_dst,
                                                               ggml_tensor* residual_hidden_dst) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(combined_embed_src != nullptr);
    VOXCPM_ASSERT(lm_hidden_dst != nullptr);
    VOXCPM_ASSERT(residual_hidden_dst != nullptr);
    VOXCPM_ASSERT(seq_len > 0);
    VOXCPM_ASSERT(static_cast<int>(ggml_nelements(combined_embed_src)) == base_lm_.config().hidden_size * seq_len);

    VoxCPMContext base_ctx = make_sequence_graph_ctx(seq_len, 65536, 524288);
    ggml_context* base_raw = base_ctx.raw_context();
    ggml_tensor* embed_tensor = base_ctx.new_tensor_2d(GGML_TYPE_F32, base_lm_.config().hidden_size, seq_len);
    ggml_tensor* text_mask_tensor = base_ctx.new_tensor_1d(GGML_TYPE_F32, seq_len);
    ggml_tensor* feat_mask_tensor = base_ctx.new_tensor_1d(GGML_TYPE_F32, seq_len);
    ggml_set_input(embed_tensor);
    ggml_set_input(text_mask_tensor);
    ggml_set_input(feat_mask_tensor);

    ggml_tensor* base_output = base_lm_.forward(base_ctx, embed_tensor, nullptr, base_kv_cache, is_causal);
    ggml_tensor* fsq_out = fsq_layer_.forward(base_ctx, base_output);
    ggml_tensor* text_mask_2d = ggml_reshape_2d(base_raw, text_mask_tensor, 1, seq_len);
    ggml_tensor* feat_mask_2d = ggml_reshape_2d(base_raw, feat_mask_tensor, 1, seq_len);
    ggml_tensor* text_mask_broadcast = ggml_repeat(base_raw, text_mask_2d, base_output);
    ggml_tensor* feat_mask_broadcast = ggml_repeat(base_raw, feat_mask_2d, base_output);
    ggml_tensor* text_part = ggml_mul(base_raw, base_output, text_mask_broadcast);
    ggml_tensor* feat_part = ggml_mul(base_raw, fsq_out, feat_mask_broadcast);
    ggml_tensor* blended_output = ggml_add(base_raw, text_part, feat_part);
    ggml_tensor* feat_embed_part = ggml_mul(base_raw, embed_tensor, feat_mask_broadcast);
    ggml_tensor* residual_inputs = build_residual_fusion_input(base_ctx, *components_, blended_output, feat_embed_part);
    ggml_tensor* blended_last = ggml_view_1d(base_raw,
                                             blended_output,
                                             base_lm_.config().hidden_size,
                                             static_cast<size_t>(seq_len - 1) *
                                                 static_cast<size_t>(base_lm_.config().hidden_size) * sizeof(float));
    ggml_tensor* blended_last_copy = ggml_cont(base_raw, blended_last);
    ggml_set_output(residual_inputs);
    ggml_set_output(blended_last_copy);

    ggml_cgraph* base_graph = base_ctx.new_graph();
    base_ctx.build_forward(base_graph, residual_inputs);
    base_ctx.build_forward(base_graph, blended_last_copy);
    backend_->reserve_compute_memory(base_graph, "runtime.prefill.hidden_states.base");
    backend_->alloc_graph(base_graph, "runtime.prefill.hidden_states.base");
    backend_->tensor_copy(combined_embed_src, embed_tensor);
    backend_->tensor_set(text_mask_tensor, text_mask.data(), 0, text_mask.size() * sizeof(float));
    backend_->tensor_set(feat_mask_tensor, feat_mask.data(), 0, feat_mask.size() * sizeof(float));
    VOXCPM_ASSERT(backend_->compute(base_graph) == GGML_STATUS_SUCCESS);
    maybe_collect_graph(base_graph);
    backend_->tensor_copy(blended_last_copy, lm_hidden_dst);

    VoxCPMContext residual_ctx = make_sequence_graph_ctx(seq_len, 32768, 262144);
    ggml_context* residual_raw = residual_ctx.raw_context();
    ggml_tensor* residual_input_tensor =
        residual_ctx.new_tensor_2d(GGML_TYPE_F32, residual_lm_.config().hidden_size, seq_len);
    ggml_set_input(residual_input_tensor);
    ggml_tensor* residual_output = residual_lm_.forward(residual_ctx,
                                                        residual_input_tensor,
                                                        nullptr,
                                                        residual_kv_cache,
                                                        is_causal);
    ggml_tensor* residual_last = ggml_view_1d(residual_raw,
                                              residual_output,
                                              residual_lm_.config().hidden_size,
                                              static_cast<size_t>(seq_len - 1) *
                                                  static_cast<size_t>(residual_lm_.config().hidden_size) * sizeof(float));
    ggml_tensor* residual_last_copy = ggml_cont(residual_raw, residual_last);
    ggml_set_output(residual_last_copy);

    ggml_cgraph* residual_graph = residual_ctx.new_graph();
    residual_ctx.build_forward(residual_graph, residual_last_copy);
    backend_->reserve_compute_memory(residual_graph, "runtime.prefill.hidden_states.residual");
    backend_->alloc_graph(residual_graph, "runtime.prefill.hidden_states.residual");
    backend_->tensor_copy(residual_inputs, residual_input_tensor);
    VOXCPM_ASSERT(backend_->compute(residual_graph) == GGML_STATUS_SUCCESS);
    maybe_collect_graph(residual_graph);
    backend_->tensor_copy(residual_last_copy, residual_hidden_dst);
}

void VoxCPMRuntime::run_prefill_hidden_states_from_input_tensors_into(ggml_tensor* combined_embed_src,
                                                                      ggml_tensor* text_mask_src,
                                                                      ggml_tensor* feat_mask_src,
                                                                      int seq_len,
                                                                      MiniCPMKVCache& base_kv_cache,
                                                                      MiniCPMKVCache& residual_kv_cache,
                                                                      bool is_causal,
                                                                      ggml_tensor* lm_hidden_dst,
                                                                      ggml_tensor* residual_hidden_dst) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(combined_embed_src != nullptr);
    VOXCPM_ASSERT(text_mask_src != nullptr);
    VOXCPM_ASSERT(feat_mask_src != nullptr);
    VOXCPM_ASSERT(lm_hidden_dst != nullptr);
    VOXCPM_ASSERT(residual_hidden_dst != nullptr);
    VOXCPM_ASSERT(seq_len > 0);
    VOXCPM_ASSERT(static_cast<int>(ggml_nelements(combined_embed_src)) == base_lm_.config().hidden_size * seq_len);
    VOXCPM_ASSERT(static_cast<int>(ggml_nelements(text_mask_src)) == seq_len);
    VOXCPM_ASSERT(static_cast<int>(ggml_nelements(feat_mask_src)) == seq_len);

    VoxCPMContext base_ctx = make_sequence_graph_ctx(seq_len, 65536, 524288);
    ggml_context* base_raw = base_ctx.raw_context();
    ggml_tensor* embed_tensor = base_ctx.new_tensor_2d(GGML_TYPE_F32, base_lm_.config().hidden_size, seq_len);
    ggml_tensor* text_mask_tensor = base_ctx.new_tensor_1d(GGML_TYPE_F32, seq_len);
    ggml_tensor* feat_mask_tensor = base_ctx.new_tensor_1d(GGML_TYPE_F32, seq_len);
    ggml_set_input(embed_tensor);
    ggml_set_input(text_mask_tensor);
    ggml_set_input(feat_mask_tensor);

    ggml_tensor* base_output = base_lm_.forward(base_ctx, embed_tensor, nullptr, base_kv_cache, is_causal);
    ggml_tensor* fsq_out = fsq_layer_.forward(base_ctx, base_output);
    ggml_tensor* text_mask_2d = ggml_reshape_2d(base_raw, text_mask_tensor, 1, seq_len);
    ggml_tensor* feat_mask_2d = ggml_reshape_2d(base_raw, feat_mask_tensor, 1, seq_len);
    ggml_tensor* text_mask_broadcast = ggml_repeat(base_raw, text_mask_2d, base_output);
    ggml_tensor* feat_mask_broadcast = ggml_repeat(base_raw, feat_mask_2d, base_output);
    ggml_tensor* text_part = ggml_mul(base_raw, base_output, text_mask_broadcast);
    ggml_tensor* feat_part = ggml_mul(base_raw, fsq_out, feat_mask_broadcast);
    ggml_tensor* blended_output = ggml_add(base_raw, text_part, feat_part);
    ggml_tensor* feat_embed_part = ggml_mul(base_raw, embed_tensor, feat_mask_broadcast);
    ggml_tensor* residual_inputs = build_residual_fusion_input(base_ctx, *components_, blended_output, feat_embed_part);
    ggml_tensor* blended_last = ggml_view_1d(base_raw,
                                             blended_output,
                                             base_lm_.config().hidden_size,
                                             static_cast<size_t>(seq_len - 1) *
                                                 static_cast<size_t>(base_lm_.config().hidden_size) * sizeof(float));
    ggml_tensor* blended_last_copy = ggml_cont(base_raw, blended_last);
    ggml_set_output(residual_inputs);
    ggml_set_output(blended_last_copy);

    ggml_cgraph* base_graph = base_ctx.new_graph();
    base_ctx.build_forward(base_graph, residual_inputs);
    base_ctx.build_forward(base_graph, blended_last_copy);
    backend_->reserve_compute_memory(base_graph, "runtime.prefill.hidden_states.base.tensor_masks");
    backend_->alloc_graph(base_graph, "runtime.prefill.hidden_states.base.tensor_masks");
    backend_->tensor_copy(combined_embed_src, embed_tensor);
    backend_->tensor_copy(text_mask_src, text_mask_tensor);
    backend_->tensor_copy(feat_mask_src, feat_mask_tensor);
    VOXCPM_ASSERT(backend_->compute(base_graph) == GGML_STATUS_SUCCESS);
    maybe_collect_graph(base_graph);
    backend_->tensor_copy(blended_last_copy, lm_hidden_dst);

    VoxCPMContext residual_ctx = make_sequence_graph_ctx(seq_len, 32768, 262144);
    ggml_context* residual_raw = residual_ctx.raw_context();
    ggml_tensor* residual_input_tensor =
        residual_ctx.new_tensor_2d(GGML_TYPE_F32, residual_lm_.config().hidden_size, seq_len);
    ggml_set_input(residual_input_tensor);
    ggml_tensor* residual_output = residual_lm_.forward(residual_ctx,
                                                        residual_input_tensor,
                                                        nullptr,
                                                        residual_kv_cache,
                                                        is_causal);
    ggml_tensor* residual_last = ggml_view_1d(residual_raw,
                                              residual_output,
                                              residual_lm_.config().hidden_size,
                                              static_cast<size_t>(seq_len - 1) *
                                                  static_cast<size_t>(residual_lm_.config().hidden_size) * sizeof(float));
    ggml_tensor* residual_last_copy = ggml_cont(residual_raw, residual_last);
    ggml_set_output(residual_last_copy);

    ggml_cgraph* residual_graph = residual_ctx.new_graph();
    residual_ctx.build_forward(residual_graph, residual_last_copy);
    backend_->reserve_compute_memory(residual_graph, "runtime.prefill.hidden_states.residual.tensor_masks");
    backend_->alloc_graph(residual_graph, "runtime.prefill.hidden_states.residual.tensor_masks");
    backend_->tensor_copy(residual_inputs, residual_input_tensor);
    VOXCPM_ASSERT(backend_->compute(residual_graph) == GGML_STATUS_SUCCESS);
    maybe_collect_graph(residual_graph);
    backend_->tensor_copy(residual_last_copy, residual_hidden_dst);
}

std::vector<float> VoxCPMRuntime::run_prefill_base_to_residual_inputs_with_last_hidden_into(
    const std::vector<float>& combined_embed,
    const std::vector<float>& text_mask,
    const std::vector<float>& feat_mask,
    int seq_len,
    MiniCPMKVCache& kv_cache,
    bool is_causal,
    ggml_tensor* lm_hidden_dst) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(lm_hidden_dst != nullptr);
    VOXCPM_ASSERT(seq_len > 0);
    VOXCPM_ASSERT(static_cast<int>(combined_embed.size()) == base_lm_.config().hidden_size * seq_len);
    VOXCPM_ASSERT(static_cast<int>(text_mask.size()) == seq_len);
    VOXCPM_ASSERT(static_cast<int>(feat_mask.size()) == seq_len);

    VoxCPMContext graph_ctx = make_sequence_graph_ctx(seq_len, 65536, 524288);
    ggml_context* raw = graph_ctx.raw_context();
    ggml_tensor* embed_tensor = graph_ctx.new_tensor_2d(GGML_TYPE_F32, base_lm_.config().hidden_size, seq_len);
    ggml_tensor* text_mask_tensor = graph_ctx.new_tensor_1d(GGML_TYPE_F32, seq_len);
    ggml_tensor* feat_mask_tensor = graph_ctx.new_tensor_1d(GGML_TYPE_F32, seq_len);
    ggml_set_input(embed_tensor);
    ggml_set_input(text_mask_tensor);
    ggml_set_input(feat_mask_tensor);

    ggml_tensor* base_output = base_lm_.forward(graph_ctx, embed_tensor, nullptr, kv_cache, is_causal);
    ggml_tensor* fsq_out = fsq_layer_.forward(graph_ctx, base_output);
    ggml_tensor* text_mask_2d = ggml_reshape_2d(raw, text_mask_tensor, 1, seq_len);
    ggml_tensor* feat_mask_2d = ggml_reshape_2d(raw, feat_mask_tensor, 1, seq_len);
    ggml_tensor* text_mask_broadcast = ggml_repeat(raw, text_mask_2d, base_output);
    ggml_tensor* feat_mask_broadcast = ggml_repeat(raw, feat_mask_2d, base_output);
    ggml_tensor* text_part = ggml_mul(raw, base_output, text_mask_broadcast);
    ggml_tensor* feat_part = ggml_mul(raw, fsq_out, feat_mask_broadcast);
    ggml_tensor* blended_output = ggml_add(raw, text_part, feat_part);
    ggml_tensor* feat_embed_part = ggml_mul(raw, embed_tensor, feat_mask_broadcast);
    ggml_tensor* residual_inputs = build_residual_fusion_input(graph_ctx, *components_, blended_output, feat_embed_part);
    ggml_tensor* blended_last = ggml_view_1d(raw,
                                             blended_output,
                                             base_lm_.config().hidden_size,
                                             static_cast<size_t>(seq_len - 1) *
                                                 static_cast<size_t>(base_lm_.config().hidden_size) * sizeof(float));
    ggml_tensor* blended_last_copy = ggml_cont(raw, blended_last);
    ggml_set_output(residual_inputs);
    ggml_set_output(blended_last_copy);

    ggml_cgraph* graph = graph_ctx.new_graph();
    graph_ctx.build_forward(graph, residual_inputs);
    graph_ctx.build_forward(graph, blended_last_copy);
    backend_->reserve_compute_memory(graph, "runtime.prefill.base_to_residual_inputs.last_hidden");
    backend_->alloc_graph(graph, "runtime.prefill.base_to_residual_inputs.last_hidden");
    backend_->tensor_set(embed_tensor, combined_embed.data(), 0, combined_embed.size() * sizeof(float));
    backend_->tensor_set(text_mask_tensor, text_mask.data(), 0, text_mask.size() * sizeof(float));
    backend_->tensor_set(feat_mask_tensor, feat_mask.data(), 0, feat_mask.size() * sizeof(float));
    VOXCPM_ASSERT(backend_->compute(graph) == GGML_STATUS_SUCCESS);
    maybe_collect_graph(graph);

    backend_->tensor_copy(blended_last_copy, lm_hidden_dst);

    std::vector<float> residual(static_cast<size_t>(base_lm_.config().hidden_size) * seq_len);
    backend_->tensor_get(residual_inputs, residual.data(), 0, residual.size() * sizeof(float));
    return residual;
}

std::vector<float> VoxCPMRuntime::run_minicpm_forward(MiniCPMModel& model,
                                                      const std::vector<float>& input,
                                                      int seq_len,
                                                      MiniCPMKVCache& kv_cache,
                                                      bool is_causal) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(static_cast<int>(input.size()) == model.config().hidden_size * seq_len);

    VoxCPMContext graph_ctx = make_sequence_graph_ctx(seq_len, 32768, 262144);
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

void VoxCPMRuntime::run_minicpm_forward_last_hidden_into(MiniCPMModel& model,
                                                         const std::vector<float>& input,
                                                         int seq_len,
                                                         MiniCPMKVCache& kv_cache,
                                                         bool is_causal,
                                                         ggml_tensor* output_dst) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(output_dst != nullptr);
    VOXCPM_ASSERT(seq_len > 0);
    VOXCPM_ASSERT(static_cast<int>(input.size()) == model.config().hidden_size * seq_len);

    VoxCPMContext graph_ctx = make_sequence_graph_ctx(seq_len, 32768, 262144);
    ggml_tensor* input_tensor = graph_ctx.new_tensor_2d(GGML_TYPE_F32, model.config().hidden_size, seq_len);
    ggml_set_input(input_tensor);

    ggml_tensor* output = model.forward(graph_ctx, input_tensor, nullptr, kv_cache, is_causal);
    ggml_tensor* output_last = ggml_view_1d(graph_ctx.raw_context(),
                                            output,
                                            model.config().hidden_size,
                                            static_cast<size_t>(seq_len - 1) *
                                                static_cast<size_t>(model.config().hidden_size) * sizeof(float));
    ggml_set_output(output_last);

    ggml_cgraph* graph = graph_ctx.new_graph();
    graph_ctx.build_forward(graph, output_last);
    backend_->reserve_compute_memory(graph, "runtime.minicpm.forward_last_hidden");
    backend_->alloc_graph(graph, "runtime.minicpm.forward_last_hidden");
    backend_->tensor_set(input_tensor, input.data(), 0, input.size() * sizeof(float));
    VOXCPM_ASSERT(backend_->compute(graph) == GGML_STATUS_SUCCESS);
    maybe_collect_graph(graph);

    backend_->tensor_copy(output_last, output_dst);
}

std::vector<float> VoxCPMRuntime::run_minicpm_forward_last_hidden(MiniCPMModel& model,
                                                                  const std::vector<float>& input,
                                                                  int seq_len,
                                                                  MiniCPMKVCache& kv_cache,
                                                                  bool is_causal) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(seq_len > 0);
    VOXCPM_ASSERT(static_cast<int>(input.size()) == model.config().hidden_size * seq_len);

    VoxCPMContext graph_ctx = make_sequence_graph_ctx(seq_len, 32768, 262144);
    ggml_tensor* input_tensor = graph_ctx.new_tensor_2d(GGML_TYPE_F32, model.config().hidden_size, seq_len);
    ggml_set_input(input_tensor);

    ggml_tensor* output = model.forward(graph_ctx, input_tensor, nullptr, kv_cache, is_causal);
    ggml_tensor* output_last = ggml_view_1d(graph_ctx.raw_context(),
                                            output,
                                            model.config().hidden_size,
                                            static_cast<size_t>(seq_len - 1) *
                                                static_cast<size_t>(model.config().hidden_size) * sizeof(float));
    ggml_set_output(output_last);

    ggml_cgraph* graph = graph_ctx.new_graph();
    graph_ctx.build_forward(graph, output_last);
    backend_->reserve_compute_memory(graph, "runtime.minicpm.forward_last_hidden");
    backend_->alloc_graph(graph, "runtime.minicpm.forward_last_hidden");
    backend_->tensor_set(input_tensor, input.data(), 0, input.size() * sizeof(float));
    VOXCPM_ASSERT(backend_->compute(graph) == GGML_STATUS_SUCCESS);
    maybe_collect_graph(graph);

    std::vector<float> out(static_cast<size_t>(model.config().hidden_size));
    backend_->tensor_get(output_last, out.data(), 0, out.size() * sizeof(float));
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
    VOXCPM_ASSERT(components_ != nullptr);
    VOXCPM_ASSERT(static_cast<int>(z.size()) == config_.feat_dim * config_.patch_size);
    const int expected_mu_size = config_.loc_dit.hidden_size * (components_->fusion_concat_proj() ? 2 : 1);
    VOXCPM_ASSERT(static_cast<int>(mu.size()) == expected_mu_size);
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

VoxCPMCachedGraph& VoxCPMRuntime::run_decode_front_half_graph(const std::vector<float>& z,
                                                              const VoxCPMDecodeState& state,
                                                              int inference_timesteps,
                                                              float cfg_value) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(static_cast<int>(z.size()) == config_.feat_dim * config_.patch_size);

    VoxCPMCachedGraph& cached = ensure_decode_front_half_graph(inference_timesteps, cfg_value);
    backend_->alloc_graph(cached.graph, "runtime.decode_front_half.cached");
    backend_->tensor_set(cached.input0, z.data(), 0, z.size() * sizeof(float));

    if (state.persistent_state != nullptr && state.persistent_state->is_initialized()) {
        backend_->tensor_copy(state.persistent_state->lm_hidden(), cached.input1);
        backend_->tensor_copy(state.persistent_state->residual_hidden(), cached.input2);
        backend_->tensor_copy(state.persistent_state->prefix_patch(), cached.input3);
    } else {
        VOXCPM_ASSERT(static_cast<int>(state.lm_hidden.size()) == base_lm_.config().hidden_size);
        VOXCPM_ASSERT(static_cast<int>(state.residual_hidden.size()) == residual_lm_.config().hidden_size);
        VOXCPM_ASSERT(static_cast<int>(state.prefix_feat_cond.size()) == config_.patch_size * config_.feat_dim);
        backend_->tensor_set(cached.input1, state.lm_hidden.data(), 0, state.lm_hidden.size() * sizeof(float));
        backend_->tensor_set(cached.input2,
                             state.residual_hidden.data(),
                             0,
                             state.residual_hidden.size() * sizeof(float));
        backend_->tensor_set(cached.input3,
                             state.prefix_feat_cond.data(),
                             0,
                             state.prefix_feat_cond.size() * sizeof(float));
    }

    if (cached.input4 && !cached.aux_input4.empty()) {
        backend_->tensor_set(cached.input4, cached.aux_input4.data(), 0, cached.aux_input4.size() * sizeof(float));
    }

    VOXCPM_ASSERT(backend_->compute(cached.graph) == GGML_STATUS_SUCCESS);
    maybe_collect_graph(cached.graph);
    return cached;
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

std::vector<float> VoxCPMRuntime::run_locenc_patch_to_lm_embed_from_tensor(ggml_tensor* patch_src) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(components_ != nullptr);
    VOXCPM_ASSERT(patch_src != nullptr);
    VOXCPM_ASSERT(static_cast<int>(ggml_nelements(patch_src)) == config_.feat_dim * config_.patch_size);

    VoxCPMCachedGraph& cached = ensure_locenc_patch_to_lm_embed_graph();
    backend_->alloc_graph(cached.graph, "runtime.locenc_to_lm_embed.cached");
    backend_->tensor_copy(patch_src, cached.input0);
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

void VoxCPMRuntime::finalize_prefill_prompt_timeline(VoxCPMDecodeState& state,
                                                     int audio_frame_count,
                                                     bool materialize_prefix_host_shadow,
                                                     const float* last_prompt_patch_fallback) {
    const size_t patch_elem_count = static_cast<size_t>(config_.patch_size * config_.feat_dim);
    ggml_tensor* last_prompt_patch_view =
        (state.output_pool != nullptr && state.output_pool->is_initialized() && audio_frame_count > 0)
            ? state.output_pool->latent_patch_view(audio_frame_count - 1)
            : nullptr;

    if (state.persistent_state != nullptr &&
        state.persistent_state->is_initialized() &&
        (last_prompt_patch_view != nullptr || last_prompt_patch_fallback != nullptr)) {
        if (last_prompt_patch_view != nullptr) {
            backend_->tensor_copy(last_prompt_patch_view, state.persistent_state->prefix_patch());
        } else {
            state.persistent_state->set_prefix_patch_from_host(*backend_,
                                                               last_prompt_patch_fallback,
                                                               patch_elem_count);
        }
    }

    if (materialize_prefix_host_shadow &&
        (last_prompt_patch_view != nullptr || last_prompt_patch_fallback != nullptr)) {
        state.prefix_feat_cond.resize(patch_elem_count);
        if (state.persistent_state != nullptr && state.persistent_state->is_initialized()) {
            state.persistent_state->get_prefix_patch_to_host(*backend_,
                                                             state.prefix_feat_cond.data(),
                                                             state.prefix_feat_cond.size());
        } else if (last_prompt_patch_view != nullptr) {
            backend_->tensor_get(last_prompt_patch_view,
                                 state.prefix_feat_cond.data(),
                                 0,
                                 state.prefix_feat_cond.size() * sizeof(float));
        } else {
            std::copy_n(last_prompt_patch_fallback, patch_elem_count, state.prefix_feat_cond.data());
        }
    } else {
        std::vector<float>().swap(state.prefix_feat_cond);
    }

    if (state.output_pool != nullptr &&
        state.output_pool->is_initialized() &&
        (last_prompt_patch_view != nullptr || last_prompt_patch_fallback != nullptr)) {
        if (state.persistent_state != nullptr && state.persistent_state->is_initialized()) {
            state.output_pool->publish_patch_output(*backend_, state.persistent_state->prefix_patch());
        } else if (last_prompt_patch_view != nullptr) {
            state.output_pool->publish_patch_output(*backend_, last_prompt_patch_view);
        } else {
            state.output_pool->publish_patch_output_from_host(*backend_,
                                                              last_prompt_patch_fallback,
                                                              patch_elem_count);
        }
    }
}

VoxCPMDecodeState VoxCPMRuntime::prefill(const std::vector<int32_t>& text,
                                         const std::vector<int32_t>& text_mask,
                                         const std::vector<float>& feat,
                                         const std::vector<int32_t>& feat_mask,
                                         int seq_len,
                                         int streaming_prefix_len) {
    return prefill_impl(text, text_mask, &feat, nullptr, nullptr, feat_mask, seq_len, streaming_prefix_len);
}

VoxCPMDecodeState VoxCPMRuntime::prefill_with_feature_tensor(const std::vector<int32_t>& text,
                                                             const std::vector<int32_t>& text_mask,
                                                             ggml_tensor* feat_src,
                                                             const std::vector<int32_t>& feat_mask,
                                                             int seq_len,
                                                             int streaming_prefix_len) {
    VOXCPM_ASSERT(feat_src != nullptr);
    return prefill_impl(text, text_mask, nullptr, feat_src, nullptr, feat_mask, seq_len, streaming_prefix_len);
}

VoxCPMDecodeState VoxCPMRuntime::prefill_with_input_tensors(ggml_tensor* text_src,
                                                            ggml_tensor* text_mask_src,
                                                            ggml_tensor* feat_src,
                                                            ggml_tensor* feat_mask_src,
                                                            int seq_len,
                                                            int streaming_prefix_len) {
    VOXCPM_ASSERT(text_src != nullptr);
    VOXCPM_ASSERT(text_mask_src != nullptr);
    VOXCPM_ASSERT(feat_src != nullptr);
    VOXCPM_ASSERT(feat_mask_src != nullptr);
    VoxCPMPrefillTensorInputs inputs;
    inputs.text_src = text_src;
    inputs.text_mask_src = text_mask_src;
    inputs.feat_src = feat_src;
    inputs.feat_mask_src = feat_mask_src;
    inputs.seq_len = seq_len;
    inputs.streaming_prefix_len = streaming_prefix_len;
    return prefill_from_tensor_inputs(inputs);
}

VoxCPMDecodeState VoxCPMRuntime::prefill_from_tensor_inputs(const VoxCPMPrefillTensorInputs& inputs) {
    VOXCPM_ASSERT(inputs.text_src != nullptr);
    VOXCPM_ASSERT(inputs.text_mask_src != nullptr);
    VOXCPM_ASSERT(inputs.feat_src != nullptr);
    VOXCPM_ASSERT(inputs.feat_mask_src != nullptr);
    VOXCPM_ASSERT(inputs.seq_len > 0);
    std::vector<int32_t> prompt_positions = inputs.prompt_positions;
    if (prompt_positions.empty()) {
        prompt_positions = derive_prompt_positions_from_feat_mask_tensor(inputs.feat_mask_src, inputs.seq_len);
    }
    return prefill_impl_from_input_tensors_with_prompt_positions(inputs.text_src,
                                                                 inputs.text_mask_src,
                                                                 inputs.feat_src,
                                                                 inputs.feat_mask_src,
                                                                 prompt_positions,
                                                                 inputs.seq_len,
                                                                 inputs.streaming_prefix_len);
}

VoxCPMDecodeState VoxCPMRuntime::prefill_with_input_tensors_and_prompt_positions(
    ggml_tensor* text_src,
    ggml_tensor* text_mask_src,
    ggml_tensor* feat_src,
    ggml_tensor* feat_mask_src,
    const std::vector<int32_t>& prompt_positions,
    int seq_len,
    int streaming_prefix_len) {
    VOXCPM_ASSERT(text_src != nullptr);
    VOXCPM_ASSERT(text_mask_src != nullptr);
    VOXCPM_ASSERT(feat_src != nullptr);
    VOXCPM_ASSERT(feat_mask_src != nullptr);
    VoxCPMPrefillTensorInputs inputs;
    inputs.text_src = text_src;
    inputs.text_mask_src = text_mask_src;
    inputs.feat_src = feat_src;
    inputs.feat_mask_src = feat_mask_src;
    inputs.prompt_positions = prompt_positions;
    inputs.seq_len = seq_len;
    inputs.streaming_prefix_len = streaming_prefix_len;
    return prefill_from_tensor_inputs(inputs);
}

VoxCPMDecodeState VoxCPMRuntime::prefill_with_prompt_patch_tensor(const std::vector<int32_t>& text,
                                                                  const std::vector<int32_t>& text_mask,
                                                                  const std::vector<float>& feat,
                                                                  ggml_tensor* prompt_patch_src,
                                                                  const std::vector<int32_t>& feat_mask,
                                                                  int seq_len,
                                                                  int streaming_prefix_len) {
    VOXCPM_ASSERT(prompt_patch_src != nullptr);
    return prefill_impl(text, text_mask, &feat, nullptr, prompt_patch_src, feat_mask, seq_len, streaming_prefix_len);
}

VoxCPMDecodeState VoxCPMRuntime::prefill_impl(const std::vector<int32_t>& text,
                                              const std::vector<int32_t>& text_mask,
                                              const std::vector<float>* feat,
                                              ggml_tensor* feat_src,
                                              ggml_tensor* prompt_patch_src,
                                              const std::vector<int32_t>& feat_mask,
                                              int seq_len,
                                              int streaming_prefix_len) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(static_cast<int>(text.size()) == seq_len);
    VOXCPM_ASSERT(static_cast<int>(text_mask.size()) == seq_len);
    VOXCPM_ASSERT((feat != nullptr) || (feat_src != nullptr));
    if (feat != nullptr) {
        VOXCPM_ASSERT(static_cast<int>(feat->size()) == seq_len * config_.patch_size * config_.feat_dim);
    }
    if (feat_src != nullptr) {
        VOXCPM_ASSERT(static_cast<int>(ggml_nelements(feat_src)) == seq_len * config_.patch_size * config_.feat_dim);
    }
    VOXCPM_ASSERT(static_cast<int>(feat_mask.size()) == seq_len);
    const bool lazy_host_state = env_flag_enabled("VOXCPM_LAZY_HOST_STATE");

    VoxCPMDecodeState state = create_decode_state_internal(false);
    state.streaming_prefix_len = streaming_prefix_len;
    const bool persistent_only_prefill = lazy_host_state &&
                                         state.persistent_state != nullptr &&
                                         state.persistent_state->is_initialized();
    const bool lazy_prefix_host_shadow = env_flag_enabled("VOXCPM_PREFILL_LAZY_PREFIX_SHADOW") &&
                                         state.persistent_state != nullptr &&
                                         state.persistent_state->is_initialized();
    std::vector<float> text_mask_f(static_cast<size_t>(seq_len), 0.0f);
    std::vector<float> feat_mask_f(static_cast<size_t>(seq_len), 0.0f);
    for (int t = 0; t < seq_len; ++t) {
        text_mask_f[static_cast<size_t>(t)] = text_mask[static_cast<size_t>(t)] != 0 ? 1.0f : 0.0f;
        feat_mask_f[static_cast<size_t>(t)] = feat_mask[static_cast<size_t>(t)] != 0 ? 1.0f : 0.0f;
    }

    std::vector<float> lm_hidden;
    std::vector<float> residual_hidden;

    {
        constexpr int kChunkedPrefillSeqThreshold = 512;
        constexpr int kLongPrefillChunkLen = 192;
        std::vector<float> residual_inputs;
        const bool use_chunked_prefill = feat != nullptr &&
                                         state.persistent_state != nullptr &&
                                         state.persistent_state->is_initialized() &&
                                         seq_len > kChunkedPrefillSeqThreshold;
        if (use_chunked_prefill) {
            for (int chunk_start = 0; chunk_start < seq_len; chunk_start += kLongPrefillChunkLen) {
                const int chunk_len = std::min(kLongPrefillChunkLen, seq_len - chunk_start);
                const std::vector<int32_t> chunk_text =
                    std::vector<int32_t>(text.begin() + chunk_start, text.begin() + chunk_start + chunk_len);
                const size_t feat_offset = static_cast<size_t>(chunk_start) * static_cast<size_t>(config_.patch_size) *
                                           static_cast<size_t>(config_.feat_dim);
                const size_t feat_count = static_cast<size_t>(chunk_len) * static_cast<size_t>(config_.patch_size) *
                                          static_cast<size_t>(config_.feat_dim);
                const std::vector<float> chunk_feat = std::vector<float>(feat->begin() + static_cast<std::ptrdiff_t>(feat_offset),
                                                                         feat->begin() + static_cast<std::ptrdiff_t>(feat_offset + feat_count));
                std::vector<float> chunk_text_mask_f(static_cast<size_t>(chunk_len), 0.0f);
                std::vector<float> chunk_feat_mask_f(static_cast<size_t>(chunk_len), 0.0f);
                for (int t = 0; t < chunk_len; ++t) {
                    const size_t source_pos = static_cast<size_t>(chunk_start + t);
                    chunk_text_mask_f[static_cast<size_t>(t)] =
                        text_mask[source_pos] != 0 ? 1.0f : 0.0f;
                    chunk_feat_mask_f[static_cast<size_t>(t)] =
                        feat_mask[source_pos] != 0 ? 1.0f : 0.0f;
                }

                VoxCPMContext combined_ctx = make_sequence_graph_ctx(chunk_len, 8, 64);
                ggml_tensor* combined_embed_tensor =
                    combined_ctx.new_tensor_2d(GGML_TYPE_F32, base_lm_.config().hidden_size, chunk_len);
                VOXCPM_ASSERT(combined_embed_tensor != nullptr);
                ggml_backend_buffer_t combined_buffer =
                    backend_->alloc_buffer(combined_ctx.raw_context(), BufferUsage::Compute);
                VOXCPM_ASSERT(combined_buffer != nullptr);
                run_embedding_masked_locenc_sequence_to_lm_projection_into(chunk_text,
                                                                           chunk_feat,
                                                                           chunk_text_mask_f,
                                                                           chunk_feat_mask_f,
                                                                           chunk_len,
                                                                           combined_embed_tensor);
                run_prefill_hidden_states_from_tensor_into(combined_embed_tensor,
                                                           chunk_text_mask_f,
                                                           chunk_feat_mask_f,
                                                           chunk_len,
                                                           *state.base_lm_cache,
                                                           *state.residual_lm_cache,
                                                           true,
                                                           state.persistent_state->lm_hidden(),
                                                           state.persistent_state->residual_hidden());
                backend_->free_buffer(combined_buffer);
            }
            if (!persistent_only_prefill) {
                lm_hidden.resize(static_cast<size_t>(base_lm_.config().hidden_size));
                state.persistent_state->get_lm_hidden_to_host(*backend_, lm_hidden.data(), lm_hidden.size());
                residual_hidden.resize(static_cast<size_t>(residual_lm_.config().hidden_size));
                state.persistent_state->get_residual_hidden_to_host(*backend_,
                                                                    residual_hidden.data(),
                                                                    residual_hidden.size());
            }
        } else if (state.persistent_state != nullptr && state.persistent_state->is_initialized()) {
            VoxCPMContext combined_ctx = make_sequence_graph_ctx(seq_len, 8, 64);
            ggml_tensor* combined_embed_tensor =
                combined_ctx.new_tensor_2d(GGML_TYPE_F32, base_lm_.config().hidden_size, seq_len);
            VOXCPM_ASSERT(combined_embed_tensor != nullptr);
            ggml_backend_buffer_t combined_buffer =
                backend_->alloc_buffer(combined_ctx.raw_context(), BufferUsage::Compute);
            VOXCPM_ASSERT(combined_buffer != nullptr);

            if (feat_src != nullptr) {
                run_embedding_masked_locenc_sequence_to_lm_projection_from_feat_tensor_into(text,
                                                                                            feat_src,
                                                                                            text_mask_f,
                                                                                            feat_mask_f,
                                                                                            seq_len,
                                                                                            combined_embed_tensor);
            } else {
                run_embedding_masked_locenc_sequence_to_lm_projection_into(text,
                                                                           *feat,
                                                                           text_mask_f,
                                                                           feat_mask_f,
                                                                           seq_len,
                                                                           combined_embed_tensor);
            }
            run_prefill_hidden_states_from_tensor_into(combined_embed_tensor,
                                                       text_mask_f,
                                                       feat_mask_f,
                                                       seq_len,
                                                       *state.base_lm_cache,
                                                       *state.residual_lm_cache,
                                                       true,
                                                       state.persistent_state->lm_hidden(),
                                                       state.persistent_state->residual_hidden());
            backend_->free_buffer(combined_buffer);
            if (!persistent_only_prefill) {
                lm_hidden.resize(static_cast<size_t>(base_lm_.config().hidden_size));
                state.persistent_state->get_lm_hidden_to_host(*backend_, lm_hidden.data(), lm_hidden.size());
                residual_hidden.resize(static_cast<size_t>(residual_lm_.config().hidden_size));
                state.persistent_state->get_residual_hidden_to_host(*backend_,
                                                                    residual_hidden.data(),
                                                                    residual_hidden.size());
            }
        } else {
            std::vector<float> combined_embed;
            if (feat_src != nullptr) {
                VoxCPMContext mask_ctx = make_sequence_graph_ctx(seq_len, 3, 3);
                ggml_tensor* text_mask_tensor = mask_ctx.new_tensor_1d(GGML_TYPE_F32, seq_len);
                ggml_tensor* feat_mask_tensor = mask_ctx.new_tensor_1d(GGML_TYPE_F32, seq_len);
                VOXCPM_ASSERT(text_mask_tensor != nullptr);
                VOXCPM_ASSERT(feat_mask_tensor != nullptr);
                ggml_backend_buffer_t mask_buffer = backend_->alloc_buffer(mask_ctx.raw_context(), BufferUsage::Compute);
                VOXCPM_ASSERT(mask_buffer != nullptr);
                backend_->tensor_set(text_mask_tensor, text_mask_f.data(), 0, text_mask_f.size() * sizeof(float));
                backend_->tensor_set(feat_mask_tensor, feat_mask_f.data(), 0, feat_mask_f.size() * sizeof(float));
                VoxCPMContext token_ctx = make_sequence_graph_ctx(seq_len, 1, 1);
                ggml_tensor* token_tensor = token_ctx.new_tensor_1d(GGML_TYPE_I32, seq_len);
                VOXCPM_ASSERT(token_tensor != nullptr);
                ggml_backend_buffer_t token_buffer = backend_->alloc_buffer(token_ctx.raw_context(), BufferUsage::Compute);
                VOXCPM_ASSERT(token_buffer != nullptr);
                backend_->tensor_set(token_tensor, text.data(), 0, text.size() * sizeof(int32_t));
                combined_embed = benchmark_run_embedding_masked_locenc_sequence_to_lm_projection_from_tensors(
                    token_tensor,
                    feat_src,
                    text_mask_tensor,
                    feat_mask_tensor,
                    seq_len);
                backend_->free_buffer(token_buffer);
                backend_->free_buffer(mask_buffer);
            } else {
                combined_embed =
                    benchmark_run_embedding_masked_locenc_sequence_to_lm_projection(text, *feat, text_mask_f, feat_mask_f, seq_len);
            }
            auto [enc_outputs, residual_inputs_host] =
                run_prefill_base_to_residual_inputs(combined_embed, text_mask_f, feat_mask_f, seq_len, *state.base_lm_cache, true);

            const size_t hidden_size = static_cast<size_t>(base_lm_.config().hidden_size);
            const float* last_lm_hidden = enc_outputs.data() + static_cast<size_t>(seq_len - 1) * hidden_size;
            lm_hidden.assign(last_lm_hidden, last_lm_hidden + hidden_size);
            residual_inputs = std::move(residual_inputs_host);
            residual_hidden =
                run_minicpm_forward_last_hidden(residual_lm_, residual_inputs, seq_len, *state.residual_lm_cache, true);
        }
    }

    int audio_frame_count = 0;
    const float* last_prompt_patch = nullptr;
    std::vector<float> last_prompt_patch_storage;
    const size_t patch_elem_count = static_cast<size_t>(config_.patch_size * config_.feat_dim);
    if (prompt_patch_src != nullptr) {
        const TrailingPromptSpan prompt_span = find_trailing_prompt_span(feat_mask, seq_len);
        audio_frame_count = prompt_span.count;
        VOXCPM_ASSERT(static_cast<size_t>(ggml_nelements(prompt_patch_src)) == patch_elem_count * static_cast<size_t>(audio_frame_count));
        if (audio_frame_count > 0) {
            if (state.output_pool != nullptr && state.output_pool->is_initialized()) {
                VOXCPM_ASSERT(state.output_pool->publish_patch_range_to_latent_seq(*backend_,
                                                                                   prompt_patch_src,
                                                                                   0,
                                                                                   audio_frame_count));
            } else {
                last_prompt_patch_storage.resize(patch_elem_count);
                const size_t last_offset = static_cast<size_t>(audio_frame_count - 1) * patch_elem_count * sizeof(float);
                backend_->tensor_get(prompt_patch_src,
                                     last_prompt_patch_storage.data(),
                                     last_offset,
                                     patch_elem_count * sizeof(float));
                last_prompt_patch = last_prompt_patch_storage.data();
            }
        }
    } else if (feat_src != nullptr) {
        const TrailingPromptSpan prompt_span = find_trailing_prompt_span(feat_mask, seq_len);
        if (prompt_span.count > 0) {
            const int span_start = prompt_span.start;
            const int span_count = prompt_span.count;
            if (state.output_pool != nullptr && state.output_pool->is_initialized()) {
                VoxCPMContext span_ctx = make_sequence_graph_ctx(span_count, 16, 32);
                ggml_tensor* span_view =
                    make_feature_sequence_patch_span_view(span_ctx.raw_context(), feat_src, config_, span_start, span_count);
                VOXCPM_ASSERT(span_view != nullptr);
                ggml_tensor* span_tensor =
                    span_ctx.new_tensor_2d(GGML_TYPE_F32, config_.feat_dim, config_.patch_size * span_count);
                VOXCPM_ASSERT(span_tensor != nullptr);
                ggml_tensor* span_copy = ggml_cpy(span_ctx.raw_context(), span_view, span_tensor);
                ggml_set_output(span_copy);
                ggml_cgraph* span_graph = span_ctx.new_graph();
                span_ctx.build_forward(span_graph, span_copy);
                backend_->reserve_compute_memory(span_graph, "runtime.prefill.prompt_span.materialize");
                backend_->alloc_graph(span_graph, "runtime.prefill.prompt_span.materialize");
                VOXCPM_ASSERT(backend_->compute(span_graph) == GGML_STATUS_SUCCESS);
                maybe_collect_graph(span_graph);
                VOXCPM_ASSERT(state.output_pool->publish_patch_range_to_latent_seq(*backend_,
                                                                                   span_tensor,
                                                                                   audio_frame_count,
                                                                                   span_count));
            } else {
                last_prompt_patch_storage.resize(patch_elem_count);
                const size_t last_offset =
                    static_cast<size_t>(span_start + span_count - 1) * patch_elem_count * sizeof(float);
                backend_->tensor_get(feat_src,
                                     last_prompt_patch_storage.data(),
                                     last_offset,
                                     patch_elem_count * sizeof(float));
                last_prompt_patch = last_prompt_patch_storage.data();
            }
            audio_frame_count += span_count;
        }
    } else {
        const TrailingPromptSpan prompt_span = find_trailing_prompt_span(feat_mask, seq_len);
        if (prompt_span.count > 0) {
            const int span_start = prompt_span.start;
            const int span_count = prompt_span.count;
            const size_t frame_offset = static_cast<size_t>(span_start) * patch_elem_count;
            const float* frame_data = feat->data() + frame_offset;
            last_prompt_patch = frame_data + static_cast<size_t>(span_count - 1) * patch_elem_count;
            if (state.output_pool != nullptr && state.output_pool->is_initialized()) {
                state.output_pool->write_patch_range_to_latent_seq_from_host(*backend_,
                                                                             frame_data,
                                                                             audio_frame_count,
                                                                             span_count);
            }
            audio_frame_count += span_count;
        }
    }

    if (!persistent_only_prefill) {
        state.lm_hidden = std::move(lm_hidden);
        state.residual_hidden = std::move(residual_hidden);
    }
    finalize_prefill_prompt_timeline(state,
                                     audio_frame_count,
                                     !persistent_only_prefill && !lazy_prefix_host_shadow,
                                     last_prompt_patch);

    state.current_position = seq_len;
    state.audio_frame_count = audio_frame_count;
    if (!persistent_only_prefill &&
        (state.persistent_state == nullptr || !state.persistent_state->is_initialized())) {
        sync_host_state_to_persistent(state);
    }
    if (state.output_pool != nullptr &&
        state.output_pool->is_initialized() &&
        (state.persistent_state == nullptr || !state.persistent_state->is_initialized())) {
        const std::array<float, 2> stop_logits = {0.0f, 0.0f};
        state.output_pool->publish_decode_outputs_from_host(*backend_,
                                                            state.prefix_feat_cond.data(),
                                                            state.prefix_feat_cond.size(),
                                                            stop_logits.data(),
                                                            stop_logits.size());
    }
    if (lazy_host_state &&
        state.persistent_state != nullptr &&
        state.persistent_state->is_initialized()) {
        state.lm_hidden.clear();
        state.residual_hidden.clear();
        state.prefix_feat_cond.clear();
    }
    return state;
}

std::vector<int32_t> VoxCPMRuntime::derive_prompt_positions_from_feat_mask_tensor(ggml_tensor* feat_mask_src,
                                                                                  int seq_len) const {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(feat_mask_src != nullptr);
    VOXCPM_ASSERT(static_cast<int>(ggml_nelements(feat_mask_src)) == seq_len);
    std::vector<float> feat_mask_f(static_cast<size_t>(seq_len), 0.0f);
    backend_->tensor_get(feat_mask_src, feat_mask_f.data(), 0, feat_mask_f.size() * sizeof(float));

    const TrailingPromptSpan prompt_span = find_trailing_prompt_span(feat_mask_f, seq_len);
    std::vector<int32_t> prompt_positions;
    prompt_positions.reserve(static_cast<size_t>(prompt_span.count));
    for (int t = 0; t < prompt_span.count; ++t) {
        prompt_positions.push_back(prompt_span.start + t);
    }
    return prompt_positions;
}

VoxCPMDecodeState VoxCPMRuntime::prefill_impl_from_input_tensors_with_prompt_positions(
    ggml_tensor* text_src,
    ggml_tensor* text_mask_src,
    ggml_tensor* feat_src,
    ggml_tensor* feat_mask_src,
    const std::vector<int32_t>& prompt_positions,
    int seq_len,
    int streaming_prefix_len) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(text_src != nullptr);
    VOXCPM_ASSERT(text_mask_src != nullptr);
    VOXCPM_ASSERT(feat_src != nullptr);
    VOXCPM_ASSERT(feat_mask_src != nullptr);
    VOXCPM_ASSERT(text_src->type == GGML_TYPE_I32);
    VOXCPM_ASSERT(static_cast<int>(ggml_nelements(text_src)) == seq_len);
    VOXCPM_ASSERT(static_cast<int>(ggml_nelements(text_mask_src)) == seq_len);
    VOXCPM_ASSERT(static_cast<int>(ggml_nelements(feat_mask_src)) == seq_len);
    VOXCPM_ASSERT(static_cast<int>(ggml_nelements(feat_src)) == seq_len * config_.patch_size * config_.feat_dim);
    for (int32_t pos : prompt_positions) {
        VOXCPM_ASSERT(pos >= 0);
        VOXCPM_ASSERT(pos < seq_len);
    }

    const bool lazy_host_state = env_flag_enabled("VOXCPM_LAZY_HOST_STATE");

    VoxCPMDecodeState state = create_decode_state_internal(false);
    state.streaming_prefix_len = streaming_prefix_len;
    const bool persistent_only_prefill = lazy_host_state &&
                                         state.persistent_state != nullptr &&
                                         state.persistent_state->is_initialized();
    const bool lazy_prefix_host_shadow = env_flag_enabled("VOXCPM_PREFILL_LAZY_PREFIX_SHADOW") &&
                                         state.persistent_state != nullptr &&
                                         state.persistent_state->is_initialized();

    std::vector<float> lm_hidden;
    std::vector<float> residual_hidden;

    {
        std::vector<float> residual_inputs;
        if (state.persistent_state != nullptr && state.persistent_state->is_initialized()) {
            VoxCPMContext combined_ctx = make_sequence_graph_ctx(seq_len, 8, 64);
            ggml_tensor* combined_embed_tensor =
                combined_ctx.new_tensor_2d(GGML_TYPE_F32, base_lm_.config().hidden_size, seq_len);
            VOXCPM_ASSERT(combined_embed_tensor != nullptr);
            ggml_backend_buffer_t combined_buffer =
                backend_->alloc_buffer(combined_ctx.raw_context(), BufferUsage::Compute);
            VOXCPM_ASSERT(combined_buffer != nullptr);

            run_embedding_masked_locenc_sequence_to_lm_projection_from_input_tensors_into(text_src,
                                                                                           feat_src,
                                                                                           text_mask_src,
                                                                                           feat_mask_src,
                                                                                           seq_len,
                                                                                           combined_embed_tensor);
            run_prefill_hidden_states_from_input_tensors_into(combined_embed_tensor,
                                                              text_mask_src,
                                                              feat_mask_src,
                                                              seq_len,
                                                              *state.base_lm_cache,
                                                              *state.residual_lm_cache,
                                                              true,
                                                              state.persistent_state->lm_hidden(),
                                                              state.persistent_state->residual_hidden());
            backend_->free_buffer(combined_buffer);
            if (!persistent_only_prefill) {
                lm_hidden.resize(static_cast<size_t>(base_lm_.config().hidden_size));
                state.persistent_state->get_lm_hidden_to_host(*backend_, lm_hidden.data(), lm_hidden.size());
                residual_hidden.resize(static_cast<size_t>(residual_lm_.config().hidden_size));
                state.persistent_state->get_residual_hidden_to_host(*backend_,
                                                                    residual_hidden.data(),
                                                                    residual_hidden.size());
            }
        } else {
            std::vector<float> combined_embed =
                benchmark_run_embedding_masked_locenc_sequence_to_lm_projection_from_tensors(text_src,
                                                                                              feat_src,
                                                                                              text_mask_src,
                                                                                              feat_mask_src,
                                                                                              seq_len);
            std::vector<float> text_mask_f(static_cast<size_t>(seq_len), 0.0f);
            std::vector<float> feat_mask_f(static_cast<size_t>(seq_len), 0.0f);
            backend_->tensor_get(text_mask_src, text_mask_f.data(), 0, text_mask_f.size() * sizeof(float));
            backend_->tensor_get(feat_mask_src, feat_mask_f.data(), 0, feat_mask_f.size() * sizeof(float));
            auto [enc_outputs, residual_inputs_host] =
                run_prefill_base_to_residual_inputs(combined_embed,
                                                    text_mask_f,
                                                    feat_mask_f,
                                                    seq_len,
                                                    *state.base_lm_cache,
                                                    true);

            const size_t hidden_size = static_cast<size_t>(base_lm_.config().hidden_size);
            const float* last_lm_hidden = enc_outputs.data() + static_cast<size_t>(seq_len - 1) * hidden_size;
            lm_hidden.assign(last_lm_hidden, last_lm_hidden + hidden_size);
            residual_inputs = std::move(residual_inputs_host);
            residual_hidden =
                run_minicpm_forward_last_hidden(residual_lm_, residual_inputs, seq_len, *state.residual_lm_cache, true);
        }
    }

    int audio_frame_count = 0;
    const float* last_prompt_patch = nullptr;
    std::vector<float> last_prompt_patch_storage;
    const size_t patch_elem_count = static_cast<size_t>(config_.patch_size * config_.feat_dim);
    for (int i = 0; i < static_cast<int>(prompt_positions.size());) {
        const int span_start_pos = prompt_positions[static_cast<size_t>(i)];
        int span_count = 1;
        while (i + span_count < static_cast<int>(prompt_positions.size()) &&
               prompt_positions[static_cast<size_t>(i + span_count)] == span_start_pos + span_count) {
            ++span_count;
        }

        if (state.output_pool != nullptr && state.output_pool->is_initialized()) {
            VoxCPMContext span_ctx = make_sequence_graph_ctx(span_count, 16, 32);
            ggml_tensor* span_view =
                make_feature_sequence_patch_span_view(span_ctx.raw_context(), feat_src, config_, span_start_pos, span_count);
            VOXCPM_ASSERT(span_view != nullptr);
            ggml_tensor* span_tensor =
                span_ctx.new_tensor_2d(GGML_TYPE_F32, config_.feat_dim, config_.patch_size * span_count);
            VOXCPM_ASSERT(span_tensor != nullptr);
            ggml_tensor* span_copy = ggml_cpy(span_ctx.raw_context(), span_view, span_tensor);
            ggml_set_output(span_copy);
            ggml_cgraph* span_graph = span_ctx.new_graph();
            span_ctx.build_forward(span_graph, span_copy);
            backend_->reserve_compute_memory(span_graph, "runtime.prefill.prompt_span.materialize");
            backend_->alloc_graph(span_graph, "runtime.prefill.prompt_span.materialize");
            VOXCPM_ASSERT(backend_->compute(span_graph) == GGML_STATUS_SUCCESS);
            maybe_collect_graph(span_graph);
            VOXCPM_ASSERT(state.output_pool->publish_patch_range_to_latent_seq(*backend_,
                                                                               span_tensor,
                                                                               audio_frame_count,
                                                                               span_count));
        } else {
            last_prompt_patch_storage.resize(patch_elem_count);
            const size_t last_offset =
                static_cast<size_t>(span_start_pos + span_count - 1) * patch_elem_count * sizeof(float);
            backend_->tensor_get(feat_src,
                                 last_prompt_patch_storage.data(),
                                 last_offset,
                                 patch_elem_count * sizeof(float));
            last_prompt_patch = last_prompt_patch_storage.data();
        }

        audio_frame_count += span_count;
        i += span_count;
    }

    if (!persistent_only_prefill) {
        state.lm_hidden = std::move(lm_hidden);
        state.residual_hidden = std::move(residual_hidden);
    }
    finalize_prefill_prompt_timeline(state,
                                     audio_frame_count,
                                     !persistent_only_prefill && !lazy_prefix_host_shadow,
                                     last_prompt_patch);

    state.current_position = seq_len;
    state.audio_frame_count = audio_frame_count;
    if (!persistent_only_prefill &&
        (state.persistent_state == nullptr || !state.persistent_state->is_initialized())) {
        sync_host_state_to_persistent(state);
    }
    if (state.output_pool != nullptr &&
        state.output_pool->is_initialized() &&
        (state.persistent_state == nullptr || !state.persistent_state->is_initialized())) {
        const std::array<float, 2> stop_logits = {0.0f, 0.0f};
        state.output_pool->publish_decode_outputs_from_host(*backend_,
                                                            state.prefix_feat_cond.data(),
                                                            state.prefix_feat_cond.size(),
                                                            stop_logits.data(),
                                                            stop_logits.size());
    }
    if (lazy_host_state &&
        state.persistent_state != nullptr &&
        state.persistent_state->is_initialized()) {
        state.lm_hidden.clear();
        state.residual_hidden.clear();
        state.prefix_feat_cond.clear();
    }
    return state;
}

VoxCPMDecodeResult VoxCPMRuntime::decode(VoxCPMDecodeState state,
                                         const std::vector<float>& z,
                                         int inference_timesteps,
                                         float cfg_value,
                                         const VoxCPMDecodeOptions& options) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(state.base_lm_cache != nullptr);
    VOXCPM_ASSERT(state.residual_lm_cache != nullptr);
    VOXCPM_ASSERT(static_cast<int>(z.size()) == config_.feat_dim * config_.patch_size);

    const bool log_decode_timing = env_flag_enabled("VOXCPM_LOG_DECODE_TIMING");
    const bool log_decode_transfers = env_flag_enabled("VOXCPM_LOG_DECODE_TRANSFERS");
    const bool log_stop_logits = env_flag_enabled("VOXCPM_LOG_STOP_LOGITS");
    const bool lazy_host_state = env_flag_enabled("VOXCPM_LAZY_HOST_STATE");
    using clock = std::chrono::steady_clock;
    const auto decode_start = clock::now();
    const BackendTransferStats transfers_before = backend_->transfer_stats();

    VoxCPMDecodeResult result;
    const bool use_persistent_state = state.persistent_state != nullptr && state.persistent_state->is_initialized();
    const bool lazy_prefix_host_shadow = env_flag_enabled("VOXCPM_DECODE_LAZY_PREFIX_SHADOW") && use_persistent_state;
    const bool host_state_complete = has_complete_host_state(state,
                                                             base_lm_.config().hidden_size,
                                                             residual_lm_.config().hidden_size,
                                                             config_.patch_size * config_.feat_dim);
    std::vector<float> curr_embed_host;
    if (use_persistent_state) {
        if (host_state_complete && !options.trust_persistent_state) {
            // Public decode state still exposes host vectors; if callers changed them,
            // we must treat that host state as authoritative before staying on-device.
            sync_host_state_to_persistent(state);
        }
    } else {
        VOXCPM_ASSERT(host_state_complete);
    }

    VoxCPMCachedGraph& front_half = run_decode_front_half_graph(z, state, inference_timesteps, cfg_value);
    VOXCPM_ASSERT(front_half.output_aux0 != nullptr);
    if (state.output_pool != nullptr && state.output_pool->is_initialized()) {
        if (options.publish_patch_to_output) {
            state.output_pool->publish_patch_output(*backend_, front_half.output);
        }
        if (options.export_patch_to_host) {
            if (options.publish_patch_to_output) {
                result.output_0 = state.output_pool->export_patch_to_host(*backend_);
            } else {
                result.output_0.resize(static_cast<size_t>(config_.feat_dim * config_.patch_size));
                backend_->tensor_get(front_half.output, result.output_0.data(), 0, result.output_0.size() * sizeof(float));
            }
        }
    } else {
        result.output_0.resize(static_cast<size_t>(config_.feat_dim * config_.patch_size));
        backend_->tensor_get(front_half.output, result.output_0.data(), 0, result.output_0.size() * sizeof(float));
    }
    const auto front_half_end = clock::now();
    const BackendTransferStats transfers_after_front_half = backend_->transfer_stats();

    // Preserve the current patch embedding before any later cached graph can
    // reuse compute arena storage owned by the front-half graph.
    if (use_persistent_state) {
        backend_->tensor_copy(front_half.output_aux0, state.persistent_state->residual_hidden());
    } else {
        curr_embed_host.resize(static_cast<size_t>(base_lm_.config().hidden_size));
        backend_->tensor_get(front_half.output_aux0, curr_embed_host.data(), 0, curr_embed_host.size() * sizeof(float));
    }

    if (use_persistent_state &&
        state.output_pool != nullptr &&
        state.output_pool->is_initialized() &&
        !options.publish_patch_to_output) {
        state.output_pool->publish_patch_to_latent_seq(*backend_, front_half.output, state.audio_frame_count);
        backend_->tensor_copy(front_half.output, state.persistent_state->prefix_patch());
    }

    const int decode_position = state.current_position;
    const int new_position = decode_position + 1;
    const std::array<float, 2> stop_logits = run_stop_predictor_from_state(state,
                                                                           options.publish_stop_logits_to_output);
    result.output_2 = stop_logits[1] > stop_logits[0];
    if (log_stop_logits) {
        std::cerr << "[stop_logits]"
                  << " position=" << state.current_position
                  << " phase=pre"
                  << " stop0=" << stop_logits[0]
                  << " stop1=" << stop_logits[1]
                  << " stop=" << (result.output_2 ? 1 : 0)
                  << "\n";
    }
    const auto stop_end = clock::now();
    const BackendTransferStats transfers_after_stop = backend_->transfer_stats();

    const auto patch_end = clock::now();
    const BackendTransferStats transfers_after_patch = backend_->transfer_stats();

    VoxCPMCachedGraph& base_step = ensure_state_base_lm_step_graph(state, decode_position);
    backend_->alloc_graph(base_step.graph, "runtime.base_lm.decode_step.state_cached");
    if (use_persistent_state) {
        backend_->tensor_copy(state.persistent_state->residual_hidden(), base_step.input0);
    } else {
        backend_->tensor_set(base_step.input0, curr_embed_host.data(), 0, curr_embed_host.size() * sizeof(float));
    }
    const int32_t decode_position_value = decode_position;
    if (base_step.input1 != nullptr) {
        backend_->tensor_set(base_step.input1, &decode_position_value, 0, sizeof(decode_position_value));
    }
    VOXCPM_ASSERT(backend_->compute(base_step.graph) == GGML_STATUS_SUCCESS);
    maybe_collect_graph(base_step.graph);
    std::vector<float> lm_hidden;
    if (use_persistent_state) {
        backend_->tensor_copy(base_step.output, state.persistent_state->lm_hidden());
    } else {
        lm_hidden.resize(static_cast<size_t>(base_lm_.config().hidden_size));
        backend_->tensor_get(base_step.output, lm_hidden.data(), 0, lm_hidden.size() * sizeof(float));
    }
    if (log_stop_logits) {
        std::array<float, 2> post_base_stop = {0.0f, 0.0f};
        VoxCPMCachedGraph& stop_cached = ensure_stop_predictor_graph();
        backend_->alloc_graph(stop_cached.graph, "runtime.stop_predictor.cached");
        if (use_persistent_state) {
            backend_->tensor_copy(state.persistent_state->lm_hidden(), stop_cached.input0);
        } else {
            backend_->tensor_set(stop_cached.input0, lm_hidden.data(), 0, lm_hidden.size() * sizeof(float));
        }
        VOXCPM_ASSERT(backend_->compute(stop_cached.graph) == GGML_STATUS_SUCCESS);
        maybe_collect_graph(stop_cached.graph);
        backend_->tensor_get(stop_cached.output, post_base_stop.data(), 0, post_base_stop.size() * sizeof(float));
        std::cerr << "[stop_logits]"
                  << " position=" << decode_position
                  << " phase=post_base"
                  << " stop0=" << post_base_stop[0]
                  << " stop1=" << post_base_stop[1]
                  << " stop=" << (post_base_stop[1] > post_base_stop[0] ? 1 : 0)
                  << "\n";
    }
    const auto base_end = clock::now();
    const BackendTransferStats transfers_after_base = backend_->transfer_stats();

    VoxCPMCachedGraph& residual_step = ensure_state_residual_lm_step_graph(state, decode_position);
    backend_->alloc_graph(residual_step.graph, "runtime.residual_lm.decode_step.state_cached");
    if (use_persistent_state) {
        backend_->tensor_copy(state.persistent_state->lm_hidden(), residual_step.input0);
    } else {
        backend_->tensor_set(residual_step.input0, lm_hidden.data(), 0, lm_hidden.size() * sizeof(float));
    }
    if (residual_step.input1 != nullptr) {
        backend_->tensor_set(residual_step.input1, &decode_position_value, 0, sizeof(decode_position_value));
    }
    if (use_persistent_state) {
        backend_->tensor_copy(state.persistent_state->residual_hidden(), residual_step.input2);
    } else {
        backend_->tensor_set(residual_step.input2, curr_embed_host.data(), 0, curr_embed_host.size() * sizeof(float));
    }
    VOXCPM_ASSERT(backend_->compute(residual_step.graph) == GGML_STATUS_SUCCESS);
    maybe_collect_graph(residual_step.graph);
    std::vector<float> residual_hidden;
    if (use_persistent_state) {
        backend_->tensor_copy(residual_step.output, state.persistent_state->residual_hidden());
    } else {
        residual_hidden.resize(static_cast<size_t>(residual_lm_.config().hidden_size));
        backend_->tensor_get(residual_step.output, residual_hidden.data(), 0, residual_hidden.size() * sizeof(float));
    }
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

    state.current_position = new_position;
    ++state.audio_frame_count;
    if (use_persistent_state) {
        if (state.output_pool != nullptr &&
            state.output_pool->is_initialized() &&
            options.publish_patch_to_output) {
            ggml_tensor* current_patch =
                (options.publish_patch_to_output && state.output_pool->patch_output() != nullptr)
                    ? state.output_pool->patch_output()
                    : front_half.output;
            state.output_pool->publish_patch_to_latent_seq(*backend_,
                                                           current_patch,
                                                           state.audio_frame_count - 1);
            backend_->tensor_copy(current_patch, state.persistent_state->prefix_patch());
        } else if (state.output_pool == nullptr || !state.output_pool->is_initialized()) {
            state.persistent_state->set_prefix_patch_from_host(*backend_,
                                                               state.prefix_feat_cond.data(),
                                                               state.prefix_feat_cond.size());
        }
        if (lazy_host_state) {
            state.lm_hidden.clear();
            state.residual_hidden.clear();
            state.prefix_feat_cond.clear();
        } else {
            sync_persistent_to_host(state, !lazy_prefix_host_shadow);
        }
    } else {
        state.prefix_feat_cond = result.output_0;
        state.lm_hidden = std::move(lm_hidden);
        state.residual_hidden = std::move(residual_hidden);
        sync_host_state_to_persistent(state);
    }

    result.output_1 = std::move(state);
    return result;
}

std::vector<float> VoxCPMRuntime::benchmark_encode_feature_sequence(const std::vector<float>& feat, int seq_len) {
    return encode_feature_sequence(feat, seq_len);
}

std::vector<float> VoxCPMRuntime::benchmark_run_locenc_sequence_to_lm_projection(const std::vector<float>& feat,
                                                                                  int seq_len) {
    return benchmark_run_enc_to_lm_projection(encode_feature_sequence(feat, seq_len), seq_len);
}

std::vector<float> VoxCPMRuntime::benchmark_run_locenc_sequence_to_lm_projection_fsq(const std::vector<float>& feat,
                                                                                      int seq_len) {
    return benchmark_run_enc_to_lm_projection_fsq(encode_feature_sequence(feat, seq_len), seq_len);
}

std::vector<float> VoxCPMRuntime::benchmark_run_locenc_sequence_to_lm_projection_from_tensor(ggml_tensor* feat_src,
                                                                                              int seq_len) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(feat_src != nullptr);
    VOXCPM_ASSERT(static_cast<int>(ggml_nelements(feat_src)) == config_.feat_dim * config_.patch_size * seq_len);

    VoxCPMCachedGraph& cached = ensure_locenc_sequence_to_lm_projection_graph(seq_len);
    backend_->alloc_graph(cached.graph, "runtime.locenc.sequence_to_lm.cached");
    backend_->tensor_copy(feat_src, cached.input0);
    VOXCPM_ASSERT(backend_->compute(cached.graph) == GGML_STATUS_SUCCESS);
    maybe_collect_graph(cached.graph);

    std::vector<float> out(static_cast<size_t>(base_lm_.config().hidden_size) * seq_len);
    backend_->tensor_get(cached.output, out.data(), 0, out.size() * sizeof(float));
    return out;
}

std::vector<float> VoxCPMRuntime::benchmark_run_locenc_sequence_to_lm_projection_fsq_from_tensor(ggml_tensor* feat_src,
                                                                                                  int seq_len) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(feat_src != nullptr);
    VOXCPM_ASSERT(static_cast<int>(ggml_nelements(feat_src)) == config_.feat_dim * config_.patch_size * seq_len);

    VoxCPMCachedGraph& cached = ensure_locenc_sequence_to_lm_projection_fsq_graph(seq_len);
    backend_->alloc_graph(cached.graph, "runtime.locenc.sequence_to_lm_fsq.cached");
    backend_->tensor_copy(feat_src, cached.input0);
    VOXCPM_ASSERT(backend_->compute(cached.graph) == GGML_STATUS_SUCCESS);
    maybe_collect_graph(cached.graph);

    std::vector<float> out(static_cast<size_t>(base_lm_.config().hidden_size) * seq_len);
    backend_->tensor_get(cached.output, out.data(), 0, out.size() * sizeof(float));
    return out;
}

std::vector<float> VoxCPMRuntime::benchmark_run_embedding_masked_locenc_sequence_to_lm_projection(
    const std::vector<int32_t>& token_ids,
    const std::vector<float>& feat,
    const std::vector<float>& text_mask,
    const std::vector<float>& feat_mask,
    int seq_len) {
    VOXCPM_ASSERT(static_cast<int>(token_ids.size()) == seq_len);
    VOXCPM_ASSERT(static_cast<int>(feat.size()) == config_.feat_dim * config_.patch_size * seq_len);
    VOXCPM_ASSERT(static_cast<int>(text_mask.size()) == seq_len);
    VOXCPM_ASSERT(static_cast<int>(feat_mask.size()) == seq_len);

    VoxCPMCachedGraph& cached = ensure_embedding_masked_locenc_sequence_to_lm_projection_graph(seq_len);
    backend_->alloc_graph(cached.graph, "runtime.embedding_masked_locenc_to_lm.cached");
    backend_->tensor_set(cached.input0, token_ids.data(), 0, token_ids.size() * sizeof(int32_t));
    backend_->tensor_set(cached.input1, feat.data(), 0, feat.size() * sizeof(float));
    backend_->tensor_set(cached.input2, text_mask.data(), 0, text_mask.size() * sizeof(float));
    backend_->tensor_set(cached.input3, feat_mask.data(), 0, feat_mask.size() * sizeof(float));
    VOXCPM_ASSERT(backend_->compute(cached.graph) == GGML_STATUS_SUCCESS);
    maybe_collect_graph(cached.graph);

    std::vector<float> out(static_cast<size_t>(base_lm_.config().hidden_size) * seq_len);
    backend_->tensor_get(cached.output, out.data(), 0, out.size() * sizeof(float));
    return out;
}

std::vector<float> VoxCPMRuntime::benchmark_run_embedding_masked_locenc_sequence_to_lm_projection_fsq(
    const std::vector<int32_t>& token_ids,
    const std::vector<float>& feat,
    const std::vector<float>& text_mask,
    const std::vector<float>& feat_mask,
    int seq_len) {
    return benchmark_run_fsq_2d(
        benchmark_run_embedding_masked_locenc_sequence_to_lm_projection(token_ids, feat, text_mask, feat_mask, seq_len),
        seq_len);
}

std::vector<float> VoxCPMRuntime::benchmark_run_embedding_masked_locenc_sequence_to_lm_projection_from_tensors(
    ggml_tensor* token_src,
    ggml_tensor* feat_src,
    ggml_tensor* text_mask_src,
    ggml_tensor* feat_mask_src,
    int seq_len) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(token_src != nullptr);
    VOXCPM_ASSERT(feat_src != nullptr);
    VOXCPM_ASSERT(text_mask_src != nullptr);
    VOXCPM_ASSERT(feat_mask_src != nullptr);
    VOXCPM_ASSERT(token_src->type == GGML_TYPE_I32);
    VOXCPM_ASSERT(static_cast<int>(ggml_nelements(token_src)) == seq_len);
    VOXCPM_ASSERT(static_cast<int>(ggml_nelements(feat_src)) == config_.feat_dim * config_.patch_size * seq_len);
    VOXCPM_ASSERT(static_cast<int>(ggml_nelements(text_mask_src)) == seq_len);
    VOXCPM_ASSERT(static_cast<int>(ggml_nelements(feat_mask_src)) == seq_len);

    VoxCPMCachedGraph& cached = ensure_embedding_masked_locenc_sequence_to_lm_projection_graph(seq_len);
    backend_->alloc_graph(cached.graph, "runtime.embedding_masked_locenc_to_lm.cached");
    backend_->tensor_copy(token_src, cached.input0);
    backend_->tensor_copy(feat_src, cached.input1);
    backend_->tensor_copy(text_mask_src, cached.input2);
    backend_->tensor_copy(feat_mask_src, cached.input3);
    VOXCPM_ASSERT(backend_->compute(cached.graph) == GGML_STATUS_SUCCESS);
    maybe_collect_graph(cached.graph);

    std::vector<float> out(static_cast<size_t>(base_lm_.config().hidden_size) * seq_len);
    backend_->tensor_get(cached.output, out.data(), 0, out.size() * sizeof(float));
    return out;
}

std::vector<float> VoxCPMRuntime::benchmark_run_embedding_masked_locenc_sequence_to_lm_projection_fsq_from_tensors(
    ggml_tensor* token_src,
    ggml_tensor* feat_src,
    ggml_tensor* text_mask_src,
    ggml_tensor* feat_mask_src,
    int seq_len) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(token_src != nullptr);
    VOXCPM_ASSERT(feat_src != nullptr);
    VOXCPM_ASSERT(text_mask_src != nullptr);
    VOXCPM_ASSERT(feat_mask_src != nullptr);
    VOXCPM_ASSERT(token_src->type == GGML_TYPE_I32);
    VOXCPM_ASSERT(static_cast<int>(ggml_nelements(token_src)) == seq_len);
    VOXCPM_ASSERT(static_cast<int>(ggml_nelements(feat_src)) == config_.feat_dim * config_.patch_size * seq_len);
    VOXCPM_ASSERT(static_cast<int>(ggml_nelements(text_mask_src)) == seq_len);
    VOXCPM_ASSERT(static_cast<int>(ggml_nelements(feat_mask_src)) == seq_len);

    VoxCPMCachedGraph& cached = ensure_embedding_masked_locenc_sequence_to_lm_projection_fsq_graph(seq_len);
    backend_->alloc_graph(cached.graph, "runtime.embedding_masked_locenc_to_lm_fsq.cached");
    backend_->tensor_copy(token_src, cached.input0);
    backend_->tensor_copy(feat_src, cached.input1);
    backend_->tensor_copy(text_mask_src, cached.input2);
    backend_->tensor_copy(feat_mask_src, cached.input3);
    VOXCPM_ASSERT(backend_->compute(cached.graph) == GGML_STATUS_SUCCESS);
    maybe_collect_graph(cached.graph);

    std::vector<float> out(static_cast<size_t>(base_lm_.config().hidden_size) * seq_len);
    backend_->tensor_get(cached.output, out.data(), 0, out.size() * sizeof(float));
    return out;
}

std::vector<float> VoxCPMRuntime::benchmark_run_embedding(const std::vector<int32_t>& token_ids) {
    return run_embedding(token_ids);
}

std::vector<float> VoxCPMRuntime::benchmark_run_embedding_from_tensor(ggml_tensor* token_src, int token_count) {
    return run_embedding_from_tensor(token_src, token_count);
}

std::vector<float> VoxCPMRuntime::benchmark_run_enc_to_lm_projection(const std::vector<float>& input, int seq_len) {
    return run_projection_2d(*components_->enc_to_lm_proj(),
                             input,
                             feat_encoder_.config().hidden_size,
                             seq_len,
                             base_lm_.config().hidden_size);
}

std::vector<float> VoxCPMRuntime::benchmark_run_enc_to_lm_projection_from_tensor(ggml_tensor* input_src,
                                                                                  int seq_len) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(input_src != nullptr);
    VOXCPM_ASSERT(static_cast<int>(ggml_nelements(input_src)) == feat_encoder_.config().hidden_size * seq_len);

    VoxCPMCachedGraph& cached = ensure_enc_to_lm_projection_graph(seq_len);
    backend_->alloc_graph(cached.graph, "runtime.enc_to_lm_proj.cached");
    backend_->tensor_copy(input_src, cached.input0);
    VOXCPM_ASSERT(backend_->compute(cached.graph) == GGML_STATUS_SUCCESS);
    maybe_collect_graph(cached.graph);

    std::vector<float> out(static_cast<size_t>(base_lm_.config().hidden_size) * seq_len);
    backend_->tensor_get(cached.output, out.data(), 0, out.size() * sizeof(float));
    return out;
}

std::vector<float> VoxCPMRuntime::benchmark_run_enc_to_lm_projection_fsq(const std::vector<float>& input,
                                                                          int seq_len) {
    const std::vector<float> projected = benchmark_run_enc_to_lm_projection(input, seq_len);
    return run_fsq_2d(projected, seq_len);
}

std::vector<float> VoxCPMRuntime::benchmark_run_enc_to_lm_projection_fsq_from_tensor(ggml_tensor* input_src,
                                                                                      int seq_len) {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(input_src != nullptr);
    VOXCPM_ASSERT(static_cast<int>(ggml_nelements(input_src)) == feat_encoder_.config().hidden_size * seq_len);

    VoxCPMCachedGraph& cached = ensure_enc_to_lm_projection_fsq_graph(seq_len);
    backend_->alloc_graph(cached.graph, "runtime.enc_to_lm_proj_fsq.cached");
    backend_->tensor_copy(input_src, cached.input0);
    VOXCPM_ASSERT(backend_->compute(cached.graph) == GGML_STATUS_SUCCESS);
    maybe_collect_graph(cached.graph);

    std::vector<float> out(static_cast<size_t>(base_lm_.config().hidden_size) * seq_len);
    backend_->tensor_get(cached.output, out.data(), 0, out.size() * sizeof(float));
    return out;
}

std::vector<float> VoxCPMRuntime::benchmark_run_masked_fsq_blend(const std::vector<float>& input,
                                                                 const std::vector<float>& text_mask,
                                                                 const std::vector<float>& feat_mask,
                                                                 int seq_len) {
    return run_masked_fsq_blend(input, text_mask, feat_mask, seq_len);
}

std::pair<std::vector<float>, std::vector<float>> VoxCPMRuntime::benchmark_run_prefill_base_to_residual_inputs(
    const std::vector<float>& combined_embed,
    const std::vector<float>& text_mask,
    const std::vector<float>& feat_mask,
    int seq_len,
    MiniCPMKVCache& kv_cache,
    bool is_causal) {
    return run_prefill_base_to_residual_inputs(combined_embed, text_mask, feat_mask, seq_len, kv_cache, is_causal);
}

std::pair<std::vector<float>, std::vector<float>> VoxCPMRuntime::benchmark_run_prefill_inputs_to_residual_inputs(
    const std::vector<int32_t>& token_ids,
    const std::vector<float>& feat,
    const std::vector<float>& text_mask,
    const std::vector<float>& feat_mask,
    int seq_len,
    MiniCPMKVCache& kv_cache,
    bool is_causal) {
    VOXCPM_ASSERT(backend_ != nullptr);

    VoxCPMContext combined_ctx = make_sequence_graph_ctx(seq_len, 8, 64);
    ggml_tensor* combined_embed_tensor =
        combined_ctx.new_tensor_2d(GGML_TYPE_F32, base_lm_.config().hidden_size, seq_len);
    VOXCPM_ASSERT(combined_embed_tensor != nullptr);
    ggml_backend_buffer_t combined_buffer =
        backend_->alloc_buffer(combined_ctx.raw_context(), BufferUsage::Compute);
    VOXCPM_ASSERT(combined_buffer != nullptr);

    VoxCPMContext hidden_ctx = make_sequence_graph_ctx(1, 8, 64);
    ggml_tensor* lm_hidden_tensor = hidden_ctx.new_tensor_1d(GGML_TYPE_F32, base_lm_.config().hidden_size);
    VOXCPM_ASSERT(lm_hidden_tensor != nullptr);
    ggml_backend_buffer_t hidden_buffer = backend_->alloc_buffer(hidden_ctx.raw_context(), BufferUsage::State);
    VOXCPM_ASSERT(hidden_buffer != nullptr);

    run_embedding_masked_locenc_sequence_to_lm_projection_into(token_ids,
                                                               feat,
                                                               text_mask,
                                                               feat_mask,
                                                               seq_len,
                                                               combined_embed_tensor);
    std::vector<float> residual_inputs = run_prefill_base_to_residual_inputs_from_tensor_with_last_hidden_into(
        combined_embed_tensor,
        text_mask,
        feat_mask,
        seq_len,
        kv_cache,
        is_causal,
        lm_hidden_tensor);

    std::vector<float> lm_hidden(static_cast<size_t>(base_lm_.config().hidden_size), 0.0f);
    backend_->tensor_get(lm_hidden_tensor, lm_hidden.data(), 0, lm_hidden.size() * sizeof(float));

    backend_->free_buffer(hidden_buffer);
    backend_->free_buffer(combined_buffer);
    return {std::move(lm_hidden), std::move(residual_inputs)};
}

std::pair<std::vector<float>, std::vector<float>> VoxCPMRuntime::benchmark_run_prefill_inputs_to_hidden_states(
    const std::vector<int32_t>& token_ids,
    const std::vector<float>& feat,
    const std::vector<float>& text_mask,
    const std::vector<float>& feat_mask,
    int seq_len,
    MiniCPMKVCache& base_kv_cache,
    MiniCPMKVCache& residual_kv_cache,
    bool is_causal) {
    VOXCPM_ASSERT(backend_ != nullptr);

    VoxCPMContext combined_ctx = make_sequence_graph_ctx(seq_len, 8, 64);
    ggml_tensor* combined_embed_tensor =
        combined_ctx.new_tensor_2d(GGML_TYPE_F32, base_lm_.config().hidden_size, seq_len);
    VOXCPM_ASSERT(combined_embed_tensor != nullptr);
    ggml_backend_buffer_t combined_buffer =
        backend_->alloc_buffer(combined_ctx.raw_context(), BufferUsage::Compute);
    VOXCPM_ASSERT(combined_buffer != nullptr);

    VoxCPMContext hidden_ctx = make_sequence_graph_ctx(1, 8, 64);
    ggml_tensor* lm_hidden_tensor = hidden_ctx.new_tensor_1d(GGML_TYPE_F32, base_lm_.config().hidden_size);
    ggml_tensor* residual_hidden_tensor = hidden_ctx.new_tensor_1d(GGML_TYPE_F32, residual_lm_.config().hidden_size);
    VOXCPM_ASSERT(lm_hidden_tensor != nullptr);
    VOXCPM_ASSERT(residual_hidden_tensor != nullptr);
    ggml_backend_buffer_t hidden_buffer = backend_->alloc_buffer(hidden_ctx.raw_context(), BufferUsage::State);
    VOXCPM_ASSERT(hidden_buffer != nullptr);

    run_embedding_masked_locenc_sequence_to_lm_projection_into(token_ids,
                                                               feat,
                                                               text_mask,
                                                               feat_mask,
                                                               seq_len,
                                                               combined_embed_tensor);
    run_prefill_hidden_states_from_tensor_into(combined_embed_tensor,
                                               text_mask,
                                               feat_mask,
                                               seq_len,
                                               base_kv_cache,
                                               residual_kv_cache,
                                               is_causal,
                                               lm_hidden_tensor,
                                               residual_hidden_tensor);

    std::vector<float> lm_hidden(static_cast<size_t>(base_lm_.config().hidden_size), 0.0f);
    std::vector<float> residual_hidden(static_cast<size_t>(residual_lm_.config().hidden_size), 0.0f);
    backend_->tensor_get(lm_hidden_tensor, lm_hidden.data(), 0, lm_hidden.size() * sizeof(float));
    backend_->tensor_get(residual_hidden_tensor, residual_hidden.data(), 0, residual_hidden.size() * sizeof(float));

    backend_->free_buffer(hidden_buffer);
    backend_->free_buffer(combined_buffer);
    return {std::move(lm_hidden), std::move(residual_hidden)};
}

VoxCPMDecodeState VoxCPMRuntime::benchmark_stage_prefill_prompt_timeline(const std::vector<float>& prompt_patches,
                                                                         int frame_count,
                                                                         bool materialize_prefix_host_shadow) {
    VOXCPM_ASSERT(backend_ != nullptr);
    const size_t patch_elem_count = static_cast<size_t>(config_.patch_size * config_.feat_dim);
    VOXCPM_ASSERT(frame_count >= 0);
    VOXCPM_ASSERT(prompt_patches.size() == static_cast<size_t>(frame_count) * patch_elem_count);

    VoxCPMDecodeState state = create_decode_state();
    if (frame_count > 0) {
        VOXCPM_ASSERT(state.output_pool != nullptr);
        VOXCPM_ASSERT(state.output_pool->write_patch_range_to_latent_seq_from_host(*backend_,
                                                                                   prompt_patches.data(),
                                                                                   0,
                                                                                   frame_count));
    }
    finalize_prefill_prompt_timeline(state,
                                     frame_count,
                                     materialize_prefix_host_shadow,
                                     frame_count > 0 ? prompt_patches.data() + (static_cast<size_t>(frame_count - 1) * patch_elem_count)
                                                     : nullptr);
    state.audio_frame_count = frame_count;
    return state;
}

VoxCPMDecodeState VoxCPMRuntime::benchmark_stage_prefill_prompt_timeline_from_tensor(ggml_tensor* prompt_patch_src,
                                                                                     int frame_count,
                                                                                     bool materialize_prefix_host_shadow) {
    VOXCPM_ASSERT(backend_ != nullptr);
    const size_t patch_elem_count = static_cast<size_t>(config_.patch_size * config_.feat_dim);
    VOXCPM_ASSERT(frame_count >= 0);
    VOXCPM_ASSERT(prompt_patch_src != nullptr || frame_count == 0);
    if (prompt_patch_src != nullptr) {
        VOXCPM_ASSERT(ggml_nbytes(prompt_patch_src) ==
                      static_cast<size_t>(frame_count) * patch_elem_count * sizeof(float));
    }

    VoxCPMDecodeState state = create_decode_state();
    if (frame_count > 0) {
        VOXCPM_ASSERT(state.output_pool != nullptr);
        VOXCPM_ASSERT(state.output_pool->publish_patch_range_to_latent_seq(*backend_,
                                                                           prompt_patch_src,
                                                                           0,
                                                                           frame_count));
    }
    finalize_prefill_prompt_timeline(state, frame_count, materialize_prefix_host_shadow, nullptr);
    state.audio_frame_count = frame_count;
    return state;
}

std::vector<float> VoxCPMRuntime::benchmark_run_lm_to_dit_projection(const std::vector<float>& input) {
    return run_projection_1d(*components_->lm_to_dit_proj(),
                             input,
                             base_lm_.config().hidden_size,
                             config_.loc_dit.hidden_size);
}

std::vector<float> VoxCPMRuntime::benchmark_run_lm_to_dit_projection_from_state(const VoxCPMDecodeState& state) {
    VOXCPM_ASSERT(state.persistent_state != nullptr);
    VOXCPM_ASSERT(state.persistent_state->is_initialized());
    return run_projection_1d_from_tensor(*components_->lm_to_dit_proj(),
                                         state.persistent_state->lm_hidden(),
                                         base_lm_.config().hidden_size,
                                         config_.loc_dit.hidden_size);
}

std::vector<float> VoxCPMRuntime::benchmark_run_res_to_dit_projection(const std::vector<float>& input) {
    return run_projection_1d(*components_->res_to_dit_proj(),
                             input,
                             residual_lm_.config().hidden_size,
                             config_.loc_dit.hidden_size);
}

std::vector<float> VoxCPMRuntime::benchmark_run_res_to_dit_projection_from_state(const VoxCPMDecodeState& state) {
    VOXCPM_ASSERT(state.persistent_state != nullptr);
    VOXCPM_ASSERT(state.persistent_state->is_initialized());
    return run_projection_1d_from_tensor(*components_->res_to_dit_proj(),
                                         state.persistent_state->residual_hidden(),
                                         residual_lm_.config().hidden_size,
                                         config_.loc_dit.hidden_size);
}

std::vector<float> VoxCPMRuntime::benchmark_run_fsq_2d(const std::vector<float>& input, int seq_len) {
    return run_fsq_2d(input, seq_len);
}

std::vector<float> VoxCPMRuntime::benchmark_run_fsq_from_state(const VoxCPMDecodeState& state) {
    VOXCPM_ASSERT(state.persistent_state != nullptr);
    VOXCPM_ASSERT(state.persistent_state->is_initialized());
    return run_fsq_1d_from_tensor(state.persistent_state->lm_hidden());
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

std::vector<float> VoxCPMRuntime::benchmark_run_residual_lm_forward_last_hidden(const std::vector<float>& input,
                                                                                 int seq_len,
                                                                                 MiniCPMKVCache& kv_cache,
                                                                                 bool is_causal) {
    return run_minicpm_forward_last_hidden(residual_lm_, input, seq_len, kv_cache, is_causal);
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

std::array<float, 2> VoxCPMRuntime::benchmark_run_stop_predictor_from_state(const VoxCPMDecodeState& state,
                                                                            bool publish_to_output_pool) {
    return run_stop_predictor_from_state(state, publish_to_output_pool);
}

std::vector<float> VoxCPMRuntime::benchmark_run_locenc_patch(const std::vector<float>& patch) {
    VOXCPM_ASSERT(static_cast<int>(patch.size()) == config_.feat_dim * config_.patch_size);
    return run_locenc_patch(patch.data());
}

std::vector<float> VoxCPMRuntime::benchmark_run_locenc_patch_to_lm_embed(const std::vector<float>& patch) {
    return run_locenc_patch_to_lm_embed(patch);
}

std::vector<float> VoxCPMRuntime::benchmark_run_locenc_patch_from_output_pool(const VoxCPMDecodeState& state,
                                                                               int frame_index) {
    VOXCPM_ASSERT(state.output_pool != nullptr);
    VOXCPM_ASSERT(state.output_pool->is_initialized());
    ggml_tensor* patch_view = state.output_pool->latent_patch_view(frame_index);
    VOXCPM_ASSERT(patch_view != nullptr);
    return run_locenc_patch_from_tensor(patch_view);
}

std::vector<float> VoxCPMRuntime::benchmark_run_locenc_patch_to_lm_embed_from_output_pool(const VoxCPMDecodeState& state,
                                                                                           int frame_index) {
    VOXCPM_ASSERT(state.output_pool != nullptr);
    VOXCPM_ASSERT(state.output_pool->is_initialized());
    ggml_tensor* patch_view = state.output_pool->latent_patch_view(frame_index);
    VOXCPM_ASSERT(patch_view != nullptr);
    return run_locenc_patch_to_lm_embed_from_tensor(patch_view);
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

std::pair<std::vector<float>, std::vector<float>> VoxCPMRuntime::benchmark_run_decode_front_half_with_curr_embed(
    const std::vector<float>& z,
    const std::vector<float>& lm_hidden,
    const std::vector<float>& residual_hidden,
    const std::vector<float>& prefix_feat_cond,
    int inference_timesteps,
    float cfg_value) {
    std::vector<float> output;
    std::vector<float> curr_embed;
    run_decode_front_half(z,
                          lm_hidden,
                          residual_hidden,
                          prefix_feat_cond,
                          inference_timesteps,
                          cfg_value,
                          output,
                          &curr_embed);
    return {std::move(output), std::move(curr_embed)};
}

std::pair<std::vector<float>, std::vector<float>> VoxCPMRuntime::benchmark_run_decode_front_half_from_state(
    const std::vector<float>& z,
    const VoxCPMDecodeState& state,
    int inference_timesteps,
    float cfg_value) {
    VoxCPMCachedGraph& cached = run_decode_front_half_graph(z, state, inference_timesteps, cfg_value);
    std::vector<float> output(static_cast<size_t>(config_.feat_dim * config_.patch_size));
    std::vector<float> curr_embed(static_cast<size_t>(base_lm_.config().hidden_size));
    backend_->tensor_get(cached.output, output.data(), 0, output.size() * sizeof(float));
    VOXCPM_ASSERT(cached.output_aux0 != nullptr);
    backend_->tensor_get(cached.output_aux0, curr_embed.data(), 0, curr_embed.size() * sizeof(float));
    return {std::move(output), std::move(curr_embed)};
}

VoxCPMDecodeState VoxCPMRuntime::benchmark_clone_state(const VoxCPMDecodeState& state) const {
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(state.base_lm_cache != nullptr);
    VOXCPM_ASSERT(state.residual_lm_cache != nullptr);

    const bool use_persistent_state = state.persistent_state != nullptr && state.persistent_state->is_initialized();
    const bool host_state_complete = has_complete_host_state(state,
                                                             base_lm_.config().hidden_size,
                                                             residual_lm_.config().hidden_size,
                                                             config_.patch_size * config_.feat_dim);

    VoxCPMDecodeState copy = create_decode_state_internal(false);
    copy.base_lm_cache->copy_from(*state.base_lm_cache, *backend_);
    copy.residual_lm_cache->copy_from(*state.residual_lm_cache, *backend_);
    copy.current_position = state.current_position;
    copy.audio_frame_count = state.audio_frame_count;
    copy.streaming_prefix_len = state.streaming_prefix_len;
    copy.base_lm_step_graph_position = -1;
    copy.residual_lm_step_graph_position = -1;

    if (use_persistent_state) {
        backend_->tensor_copy(state.persistent_state->lm_hidden(), copy.persistent_state->lm_hidden());
        backend_->tensor_copy(state.persistent_state->residual_hidden(), copy.persistent_state->residual_hidden());
        backend_->tensor_copy(state.persistent_state->prefix_patch(), copy.persistent_state->prefix_patch());
        if (host_state_complete) {
            copy.lm_hidden = state.lm_hidden;
            copy.residual_hidden = state.residual_hidden;
            copy.prefix_feat_cond = state.prefix_feat_cond;
        }
    } else {
        copy.lm_hidden = state.lm_hidden;
        copy.residual_hidden = state.residual_hidden;
        copy.prefix_feat_cond = state.prefix_feat_cond;
        sync_host_state_to_persistent(copy);
    }

    if (copy.output_pool != nullptr && state.output_pool != nullptr && state.output_pool->is_initialized()) {
        if (state.output_pool->has_patch_output()) {
            copy.output_pool->publish_patch_output(*backend_, state.output_pool->patch_output());
        }
        if (state.output_pool->has_stop_logits()) {
            copy.output_pool->publish_stop_logits(*backend_, state.output_pool->stop_logits());
        }
        if (state.audio_frame_count > 0) {
            copy.output_pool->publish_latent_seq_prefix(*backend_, *state.output_pool, state.audio_frame_count);
        }
    }
    return copy;
}

}  // namespace voxcpm
