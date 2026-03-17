/**
 * @file locdit.cpp
 * @brief VoxCPM Local Diffusion Transformer implementation
 */

#include "voxcpm/locdit.h"

#include "voxcpm/backend.h"
#include "voxcpm/weight-store.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace voxcpm {

namespace {

constexpr float kCfgPairMaskNeg = -1.0e9f;

}

LocDiTModel::~LocDiTModel() {
    scratch_kv_cache_.reset();

    if (cfg_pair_buffer_) {
        if (backend_) {
            backend_->free_buffer(cfg_pair_buffer_);
        } else {
            ggml_backend_buffer_free(cfg_pair_buffer_);
        }
        cfg_pair_buffer_ = nullptr;
    }
    if (cfg_pair_ctx_) {
        ggml_free(cfg_pair_ctx_);
        cfg_pair_ctx_ = nullptr;
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

bool LocDiTModel::init_scratch_cache(VoxCPMBackend& backend) {
    if (scratch_kv_cache_) {
        return true;
    }

    scratch_kv_cache_ = std::make_unique<MiniCPMKVCache>(
        config().n_layer,
        config().n_kv_heads,
        config().max_length,
        config().head_dim());
    scratch_kv_cache_->init(backend);
    return true;
}

bool LocDiTModel::ensure_cfg_pair_constants(int branch_len) {
    VOXCPM_ASSERT(branch_len > 0);
    VOXCPM_ASSERT(backend_ != nullptr);

    if (cfg_pair_branch_len_ == branch_len &&
        cfg_pair_positions_ != nullptr &&
        cfg_pair_attention_mask_ != nullptr) {
        return true;
    }

    if (cfg_pair_buffer_) {
        backend_->free_buffer(cfg_pair_buffer_);
        cfg_pair_buffer_ = nullptr;
    }
    if (cfg_pair_ctx_) {
        ggml_free(cfg_pair_ctx_);
        cfg_pair_ctx_ = nullptr;
    }
    cfg_pair_positions_ = nullptr;
    cfg_pair_attention_mask_ = nullptr;
    cfg_pair_branch_len_ = 0;

    ggml_init_params params = {
        .mem_size = ggml_tensor_overhead() * 2 + 1024,
        .mem_buffer = nullptr,
        .no_alloc = true,
    };
    cfg_pair_ctx_ = ggml_init(params);
    if (!cfg_pair_ctx_) {
        return false;
    }

    const int total_len = branch_len * 2;
    cfg_pair_positions_ = ggml_new_tensor_1d(cfg_pair_ctx_, GGML_TYPE_I32, total_len);
    cfg_pair_attention_mask_ = ggml_new_tensor_2d(cfg_pair_ctx_, GGML_TYPE_F16, total_len, total_len);
    if (!cfg_pair_positions_ || !cfg_pair_attention_mask_) {
        return false;
    }

    cfg_pair_buffer_ = backend_->alloc_buffer(cfg_pair_ctx_, BufferUsage::Weights);

    std::vector<int32_t> positions(static_cast<size_t>(total_len));
    for (int i = 0; i < total_len; ++i) {
        positions[static_cast<size_t>(i)] = i % branch_len;
    }
    backend_->tensor_set(cfg_pair_positions_,
                         positions.data(),
                         0,
                         positions.size() * sizeof(int32_t));

    std::vector<ggml_fp16_t> mask(static_cast<size_t>(total_len) * static_cast<size_t>(total_len));
    for (int row = 0; row < total_len; ++row) {
        const int row_branch = row / branch_len;
        for (int col = 0; col < total_len; ++col) {
            const int col_branch = col / branch_len;
            const float value = row_branch == col_branch ? 0.0f : kCfgPairMaskNeg;
            mask[static_cast<size_t>(row) * static_cast<size_t>(total_len) + static_cast<size_t>(col)] =
                ggml_fp32_to_fp16(value);
        }
    }
    backend_->tensor_set(cfg_pair_attention_mask_,
                         mask.data(),
                         0,
                         mask.size() * sizeof(ggml_fp16_t));

    cfg_pair_branch_len_ = branch_len;
    return true;
}

bool LocDiTModel::load_from_gguf(const std::string& gguf_path,
                                 VoxCPMContext& weight_ctx,
                                 VoxCPMContext& graph_ctx,
                                 VoxCPMBackend& backend) {
    VOXCPM_UNUSED(weight_ctx);
    VOXCPM_UNUSED(graph_ctx);

    auto store = std::make_shared<VoxCPMWeightStore>();
    if (!store->load_from_file(gguf_path, backend)) {
        return false;
    }
    return load_from_store(store, backend);
}

bool LocDiTModel::load_from_store(const std::shared_ptr<VoxCPMWeightStore>& store,
                                  VoxCPMBackend& backend) {
    if (!store || !store->owns_storage()) {
        return false;
    }

    shared_store_ = store;
    weights_.in_proj_weight = store->get_tensor("locdit.in_proj.weight");
    weights_.in_proj_bias = store->get_tensor("locdit.in_proj.bias");
    weights_.cond_proj_weight = store->get_tensor("locdit.cond_proj.weight");
    weights_.cond_proj_bias = store->get_tensor("locdit.cond_proj.bias");
    weights_.out_proj_weight = store->get_tensor("locdit.out_proj.weight");
    weights_.out_proj_bias = store->get_tensor("locdit.out_proj.bias");
    weights_.time_mlp_linear1_weight = store->get_tensor("locdit.time_mlp.linear_1.weight");
    weights_.time_mlp_linear1_bias = store->get_tensor("locdit.time_mlp.linear_1.bias");
    weights_.time_mlp_linear2_weight = store->get_tensor("locdit.time_mlp.linear_2.weight");
    weights_.time_mlp_linear2_bias = store->get_tensor("locdit.time_mlp.linear_2.bias");
    weights_.delta_time_mlp_linear1_weight = store->get_tensor("locdit.delta_time_mlp.linear_1.weight");
    weights_.delta_time_mlp_linear1_bias = store->get_tensor("locdit.delta_time_mlp.linear_1.bias");
    weights_.delta_time_mlp_linear2_weight = store->get_tensor("locdit.delta_time_mlp.linear_2.weight");
    weights_.delta_time_mlp_linear2_bias = store->get_tensor("locdit.delta_time_mlp.linear_2.bias");
    if (!weights_.in_proj_weight || !weights_.in_proj_bias ||
        !weights_.cond_proj_weight || !weights_.cond_proj_bias ||
        !weights_.out_proj_weight || !weights_.out_proj_bias ||
        !weights_.time_mlp_linear1_weight || !weights_.time_mlp_linear1_bias ||
        !weights_.time_mlp_linear2_weight || !weights_.time_mlp_linear2_bias ||
        !weights_.delta_time_mlp_linear1_weight || !weights_.delta_time_mlp_linear1_bias ||
        !weights_.delta_time_mlp_linear2_weight || !weights_.delta_time_mlp_linear2_bias) {
        return false;
    }

    feat_dim_ = static_cast<int>(weights_.in_proj_weight->ne[0]);

    if (!decoder_.load_from_store(store, "locdit", backend)) {
        return false;
    }

    backend_ = &backend;
    return init_scratch_cache(backend);
}

ggml_tensor* LocDiTModel::sinusoidal_embedding(VoxCPMContext& ctx,
                                               ggml_tensor* scalar,
                                               int dim,
                                               float scale) const {
    VOXCPM_ASSERT(scalar != nullptr);
    VOXCPM_ASSERT(dim % 2 == 0);

    ggml_context* raw = ctx.raw_context();
    const int half_dim = dim / 2;
    const float emb_val = std::log(10000.0f) / static_cast<float>(half_dim - 1);

    ggml_tensor* arange = ggml_arange(raw, 0.0f, static_cast<float>(half_dim), 1.0f);
    ggml_tensor* emb = ggml_scale(raw, arange, -emb_val);
    emb = ggml_exp(raw, emb);
    emb = ggml_scale(raw, emb, scale);

    ggml_tensor* scalar_view = ggml_reshape_1d(raw, scalar, 1);
    ggml_tensor* scalar_broadcast = ggml_repeat(raw, scalar_view, emb);
    emb = ggml_mul(raw, emb, scalar_broadcast);

    ggml_tensor* sin_emb = ggml_sin(raw, emb);
    ggml_tensor* cos_emb = ggml_cos(raw, emb);
    return ggml_concat(raw, sin_emb, cos_emb, 0);
}

ggml_tensor* LocDiTModel::timestep_mlp(VoxCPMContext& ctx,
                                       ggml_tensor* input,
                                       ggml_tensor* linear1_w,
                                       ggml_tensor* linear1_b,
                                       ggml_tensor* linear2_w,
                                       ggml_tensor* linear2_b) const {
    ggml_context* raw = ctx.raw_context();

    ggml_tensor* x = ggml_mul_mat(raw, linear1_w, input);
    x = ggml_add(raw, x, linear1_b);
    x = ggml_silu(raw, x);
    x = ggml_mul_mat(raw, linear2_w, x);
    x = ggml_add(raw, x, linear2_b);
    return x;
}

ggml_tensor* LocDiTModel::compute_time_embedding(VoxCPMContext& ctx, ggml_tensor* t_scalar) const {
    ggml_tensor* sinusoidal = sinusoidal_embedding(ctx, t_scalar, config().hidden_size, 1000.0f);
    return timestep_mlp(ctx,
                        sinusoidal,
                        weights_.time_mlp_linear1_weight,
                        weights_.time_mlp_linear1_bias,
                        weights_.time_mlp_linear2_weight,
                        weights_.time_mlp_linear2_bias);
}

ggml_tensor* LocDiTModel::compute_delta_time_embedding(VoxCPMContext& ctx, ggml_tensor* dt_scalar) const {
    ggml_tensor* sinusoidal = sinusoidal_embedding(ctx, dt_scalar, config().hidden_size, 1000.0f);
    return timestep_mlp(ctx,
                        sinusoidal,
                        weights_.delta_time_mlp_linear1_weight,
                        weights_.delta_time_mlp_linear1_bias,
                        weights_.delta_time_mlp_linear2_weight,
                        weights_.delta_time_mlp_linear2_bias);
}

ggml_tensor* LocDiTModel::project_input(VoxCPMContext& ctx, ggml_tensor* x) const {
    ggml_context* raw = ctx.raw_context();
    ggml_tensor* x_proj = ggml_mul_mat(raw, weights_.in_proj_weight, x);
    return ggml_add(raw, x_proj, weights_.in_proj_bias);
}

ggml_tensor* LocDiTModel::project_condition(VoxCPMContext& ctx, ggml_tensor* cond) const {
    ggml_context* raw = ctx.raw_context();
    ggml_tensor* cond_proj = ggml_mul_mat(raw, weights_.cond_proj_weight, cond);
    return ggml_add(raw, cond_proj, weights_.cond_proj_bias);
}

ggml_tensor* LocDiTModel::build_combined_token(VoxCPMContext& ctx,
                                               ggml_tensor* mu,
                                               ggml_tensor* t_scalar,
                                               ggml_tensor* dt_scalar) const {
    ggml_context* raw = ctx.raw_context();
    ggml_tensor* combined = compute_time_embedding(ctx, t_scalar);
    combined = ggml_add(raw, combined, compute_delta_time_embedding(ctx, dt_scalar));
    if (mu) {
        combined = ggml_add(raw, mu, combined);
    }
    return ggml_reshape_2d(raw, combined, config().hidden_size, 1);
}

ggml_tensor* LocDiTModel::build_cfg_pair_positions(VoxCPMContext& ctx, int branch_len) const {
    VOXCPM_ASSERT(branch_len > 0);
    VOXCPM_ASSERT(branch_len <= config().max_length);

    ggml_context* raw = ctx.raw_context();
    ggml_tensor* base_positions = ggml_view_1d(raw, decoder_.get_pos_tensor(), branch_len, 0);
    return ggml_cont(raw, ggml_concat(raw, base_positions, base_positions, 0));
}

ggml_tensor* LocDiTModel::build_cfg_pair_attention_mask(VoxCPMContext& ctx, int branch_len) const {
    VOXCPM_ASSERT(branch_len > 0);

    ggml_context* raw = ctx.raw_context();
    const int total_len = branch_len * 2;
    const float branch_boundary = 0.5f - static_cast<float>(branch_len);

    ggml_tensor* token_ids = ggml_arange(raw, 0.0f, static_cast<float>(total_len), 1.0f);
    ggml_tensor* branch_ids = ggml_add1(raw,
                                        token_ids,
                                        ggml_arange(raw, branch_boundary, branch_boundary + 1.0f, 1.0f));
    branch_ids = ggml_step(raw, branch_ids);

    ggml_tensor* key_ids = ggml_reshape_2d(raw, branch_ids, total_len, 1);
    ggml_tensor* query_ids = ggml_reshape_2d(raw, branch_ids, 1, total_len);
    ggml_tensor* target = ggml_new_tensor_2d(raw, GGML_TYPE_F32, total_len, total_len);

    ggml_tensor* key_grid = ggml_repeat(raw, key_ids, target);
    ggml_tensor* query_grid = ggml_repeat(raw, query_ids, target);
    ggml_tensor* mask = ggml_abs(raw, ggml_sub(raw, key_grid, query_grid));
    mask = ggml_scale(raw, mask, kCfgPairMaskNeg);

    return ggml_cont(raw, ggml_cast(raw, mask, GGML_TYPE_F16));
}

ggml_tensor* LocDiTModel::forward_projected(VoxCPMContext& ctx,
                                            ggml_tensor* x_proj,
                                            ggml_tensor* combined_token,
                                            ggml_tensor* cond_proj,
                                            int prefix_len,
                                            int seq_len) {
    VOXCPM_ASSERT(x_proj != nullptr);
    VOXCPM_ASSERT(combined_token != nullptr);
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(scratch_kv_cache_ != nullptr);

    ggml_context* raw = ctx.raw_context();
    const int hidden_size = config().hidden_size;

    ggml_tensor* seq = combined_token;
    if (cond_proj && prefix_len > 0) {
        seq = ggml_concat(raw, seq, cond_proj, 1);
    }
    seq = ggml_concat(raw, seq, x_proj, 1);

    ggml_tensor* hidden = decoder_.forward(ctx, seq, nullptr, *scratch_kv_cache_, false, false);
    ggml_tensor* hidden_out = ggml_view_2d(raw,
                                           hidden,
                                           hidden_size,
                                           seq_len,
                                           hidden->nb[1],
                                           static_cast<size_t>(prefix_len + 1) * hidden->nb[1]);
    ggml_tensor* output = ggml_mul_mat(raw, weights_.out_proj_weight, hidden_out);
    return ggml_add(raw, output, weights_.out_proj_bias);
}

ggml_tensor* LocDiTModel::forward_single(VoxCPMContext& ctx,
                                         ggml_tensor* x,
                                         ggml_tensor* mu,
                                         ggml_tensor* t_scalar,
                                         ggml_tensor* cond,
                                         ggml_tensor* dt_scalar) {
    VOXCPM_ASSERT(x != nullptr);
    VOXCPM_ASSERT(mu != nullptr);
    VOXCPM_ASSERT(t_scalar != nullptr);
    VOXCPM_ASSERT(dt_scalar != nullptr);
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(scratch_kv_cache_ != nullptr);

    const int64_t seq_len = x->ne[1];
    const int64_t prefix_len = cond ? cond->ne[1] : 0;
    const int hidden_size = config().hidden_size;

    VOXCPM_ASSERT(x->ne[0] == feat_dim_);
    VOXCPM_ASSERT(mu->ne[0] == hidden_size);
    VOXCPM_ASSERT(prefix_len + seq_len + 1 <= config().max_length);

    ggml_tensor* x_proj = project_input(ctx, x);
    ggml_tensor* cond_proj = (cond && prefix_len > 0) ? project_condition(ctx, cond) : nullptr;
    ggml_tensor* combined = build_combined_token(ctx, mu, t_scalar, dt_scalar);
    return forward_projected(ctx, x_proj, combined, cond_proj, static_cast<int>(prefix_len), static_cast<int>(seq_len));
}

std::vector<float> LocDiTModel::precompute_cfg_time_table(const std::vector<float>& t_values) const {
    VOXCPM_ASSERT(backend_ != nullptr);

    if (t_values.empty()) {
        return {};
    }

    VoxCPMContext graph_ctx(ContextType::Graph, 8192, 65536);
    ggml_tensor* t_scalar = graph_ctx.new_tensor_1d(GGML_TYPE_F32, 1);
    ggml_set_input(t_scalar);

    ggml_tensor* zero_scalar = ggml_arange(graph_ctx.raw_context(), 0.0f, 1.0f, 1.0f);
    ggml_tensor* combined = compute_time_embedding(graph_ctx, t_scalar);
    combined = ggml_add(graph_ctx.raw_context(), combined, compute_delta_time_embedding(graph_ctx, zero_scalar));
    ggml_set_output(combined);

    ggml_cgraph* graph = graph_ctx.new_graph();
    graph_ctx.build_forward(graph, combined);
    backend_->reserve_compute_memory(graph, "locdit.cfg_time_table.precompute");
    backend_->alloc_graph(graph, "locdit.cfg_time_table.precompute");

    const int hidden_size = config().hidden_size;
    std::vector<float> table(static_cast<size_t>(hidden_size) * t_values.size(), 0.0f);
    std::vector<float> scratch(static_cast<size_t>(hidden_size), 0.0f);

    for (size_t i = 0; i < t_values.size(); ++i) {
        const float t_value = t_values[i];
        backend_->tensor_set(t_scalar, &t_value, 0, sizeof(t_value));
        VOXCPM_ASSERT(backend_->compute(graph) == GGML_STATUS_SUCCESS);
        backend_->tensor_get(combined, scratch.data(), 0, scratch.size() * sizeof(float));
        std::memcpy(table.data() + i * scratch.size(), scratch.data(), scratch.size() * sizeof(float));
    }

    return table;
}

void LocDiTModel::forward_cfg_pair_projected(VoxCPMContext& ctx,
                                             ggml_tensor* x_proj,
                                             ggml_tensor* mu,
                                             ggml_tensor* combined_base,
                                             ggml_tensor* cond_proj,
                                             int prefix_len,
                                             ggml_tensor** conditioned,
                                             ggml_tensor** unconditioned) {
    VOXCPM_ASSERT(conditioned != nullptr);
    VOXCPM_ASSERT(unconditioned != nullptr);
    VOXCPM_ASSERT(x_proj != nullptr);
    VOXCPM_ASSERT(mu != nullptr);
    VOXCPM_ASSERT(combined_base != nullptr);

    const int seq_len = static_cast<int>(x_proj->ne[1]);
    VOXCPM_ASSERT(x_proj->ne[0] == config().hidden_size);
    VOXCPM_ASSERT(mu->ne[0] == config().hidden_size);
    VOXCPM_ASSERT(prefix_len + seq_len + 1 <= config().max_length);

    ggml_context* raw = ctx.raw_context();
    ggml_tensor* combined_cond = ggml_add(raw, ggml_reshape_1d(raw, combined_base, combined_base->ne[0]), mu);
    combined_cond = ggml_reshape_2d(raw, combined_cond, config().hidden_size, 1);
    ggml_tensor* combined_base_2d = ggml_reshape_2d(raw, combined_base, config().hidden_size, 1);

    const int branch_len = prefix_len + seq_len + 1;
    const int total_len = branch_len * 2;
    if (total_len > config().max_length) {
        *conditioned = forward_projected(ctx, x_proj, combined_cond, cond_proj, prefix_len, seq_len);
        *unconditioned = forward_projected(ctx, x_proj, combined_base_2d, cond_proj, prefix_len, seq_len);
        return;
    }

    ggml_tensor* conditioned_seq = combined_cond;
    ggml_tensor* unconditioned_seq = combined_base_2d;
    if (cond_proj && prefix_len > 0) {
        conditioned_seq = ggml_concat(raw, conditioned_seq, cond_proj, 1);
        unconditioned_seq = ggml_concat(raw, unconditioned_seq, cond_proj, 1);
    }
    conditioned_seq = ggml_concat(raw, conditioned_seq, x_proj, 1);
    unconditioned_seq = ggml_concat(raw, unconditioned_seq, x_proj, 1);

    VOXCPM_ASSERT(ensure_cfg_pair_constants(branch_len));
    ggml_tensor* paired_seq = ggml_concat(raw, conditioned_seq, unconditioned_seq, 1);
    ggml_tensor* paired_hidden = decoder_.forward(
        ctx, paired_seq, cfg_pair_positions_, *scratch_kv_cache_, false, false, cfg_pair_attention_mask_);

    ggml_tensor* conditioned_hidden = ggml_view_2d(raw,
                                                   paired_hidden,
                                                   config().hidden_size,
                                                   seq_len,
                                                   paired_hidden->nb[1],
                                                   static_cast<size_t>(prefix_len + 1) * paired_hidden->nb[1]);
    ggml_tensor* unconditioned_hidden = ggml_view_2d(raw,
                                                     paired_hidden,
                                                     config().hidden_size,
                                                     seq_len,
                                                     paired_hidden->nb[1],
                                                     static_cast<size_t>(branch_len + prefix_len + 1) * paired_hidden->nb[1]);

    ggml_tensor* conditioned_out = ggml_mul_mat(raw, weights_.out_proj_weight, conditioned_hidden);
    ggml_tensor* unconditioned_out = ggml_mul_mat(raw, weights_.out_proj_weight, unconditioned_hidden);
    *conditioned = ggml_add(raw, conditioned_out, weights_.out_proj_bias);
    *unconditioned = ggml_add(raw, unconditioned_out, weights_.out_proj_bias);
}

ggml_tensor* LocDiTModel::forward(VoxCPMContext& ctx,
                                  ggml_tensor* x,
                                  ggml_tensor* mu,
                                  ggml_tensor* t,
                                  ggml_tensor* cond,
                                  ggml_tensor* dt) {
    VOXCPM_ASSERT(x != nullptr);
    VOXCPM_ASSERT(mu != nullptr);
    VOXCPM_ASSERT(t != nullptr);
    VOXCPM_ASSERT(cond != nullptr);
    VOXCPM_ASSERT(dt != nullptr);

    ggml_context* raw = ctx.raw_context();

    VOXCPM_ASSERT(ggml_n_dims(x) >= 2 && ggml_n_dims(x) <= 3);
    VOXCPM_ASSERT(ggml_n_dims(cond) >= 2 && ggml_n_dims(cond) <= 3);
    VOXCPM_ASSERT(ggml_n_dims(mu) >= 1 && ggml_n_dims(mu) <= 2);
    VOXCPM_ASSERT(ggml_n_dims(t) == 1);
    VOXCPM_ASSERT(ggml_n_dims(dt) == 1);
    VOXCPM_ASSERT(x->ne[0] == feat_dim_);
    VOXCPM_ASSERT(cond->ne[0] == feat_dim_);
    VOXCPM_ASSERT(mu->ne[0] == config().hidden_size);

    const int64_t batch = std::max<int64_t>(1, std::max<int64_t>(x->ne[2], std::max<int64_t>(cond->ne[2], mu->ne[1])));
    VOXCPM_ASSERT(cond->ne[2] == 1 || cond->ne[2] == batch);
    VOXCPM_ASSERT(mu->ne[1] == 1 || mu->ne[1] == batch);
    VOXCPM_ASSERT(t->ne[0] == batch);
    VOXCPM_ASSERT(dt->ne[0] == batch);

    ggml_tensor* output = ggml_new_tensor_3d(raw, GGML_TYPE_F32, feat_dim_, x->ne[1], batch);
    ggml_tensor* sync = nullptr;

    for (int64_t b = 0; b < batch; ++b) {
        ggml_tensor* x_view = batch == 1
            ? ggml_reshape_2d(raw, x, x->ne[0], x->ne[1])
            : ggml_view_2d(raw, x, x->ne[0], x->ne[1], x->nb[1], static_cast<size_t>(b) * x->nb[2]);
        ggml_tensor* cond_view = batch == 1
            ? ggml_reshape_2d(raw, cond, cond->ne[0], cond->ne[1])
            : ggml_view_2d(raw, cond, cond->ne[0], cond->ne[1], cond->nb[1], static_cast<size_t>(b) * cond->nb[2]);
        ggml_tensor* mu_view = batch == 1
            ? ggml_reshape_1d(raw, mu, mu->ne[0])
            : ggml_view_1d(raw, mu, mu->ne[0], static_cast<size_t>(b) * mu->nb[1]);
        ggml_tensor* t_view = ggml_view_1d(raw, t, 1, static_cast<size_t>(b) * t->nb[0]);
        ggml_tensor* dt_view = ggml_view_1d(raw, dt, 1, static_cast<size_t>(b) * dt->nb[0]);

        ggml_tensor* sample_out = forward_single(ctx, x_view, mu_view, t_view, cond_view, dt_view);
        ggml_tensor* out_view = ggml_view_2d(raw,
                                             output,
                                             output->ne[0],
                                             output->ne[1],
                                             output->nb[1],
                                             static_cast<size_t>(b) * output->nb[2]);
        ggml_tensor* copied = ggml_cpy(raw, sample_out, out_view);
        ggml_tensor* copied_sum = ggml_sum(raw, copied);
        sync = sync ? ggml_add(raw, sync, copied_sum) : copied_sum;
    }

    if (!sync) {
        return output;
    }

    sync = ggml_scale(raw, sync, 0.0f);
    return ggml_add1(raw, output, sync);
}

void LocDiTModel::forward_cfg_pair(VoxCPMContext& ctx,
                                   ggml_tensor* x,
                                   ggml_tensor* mu,
                                   ggml_tensor* t_scalar,
                                   ggml_tensor* cond,
                                   ggml_tensor* dt_scalar,
                                   ggml_tensor** conditioned,
                                   ggml_tensor** unconditioned) {
    VOXCPM_ASSERT(conditioned != nullptr);
    VOXCPM_ASSERT(unconditioned != nullptr);
    VOXCPM_ASSERT(x != nullptr);
    VOXCPM_ASSERT(mu != nullptr);
    VOXCPM_ASSERT(t_scalar != nullptr);
    VOXCPM_ASSERT(cond != nullptr);
    VOXCPM_ASSERT(dt_scalar != nullptr);

    const int64_t seq_len = x->ne[1];
    const int64_t prefix_len = cond ? cond->ne[1] : 0;
    VOXCPM_ASSERT(x->ne[0] == feat_dim_);
    VOXCPM_ASSERT(mu->ne[0] == config().hidden_size);
    VOXCPM_ASSERT(prefix_len + seq_len + 1 <= config().max_length);

    ggml_tensor* x_proj = project_input(ctx, x);
    ggml_tensor* cond_proj = prefix_len > 0 ? project_condition(ctx, cond) : nullptr;
    ggml_tensor* delta_time_embedding = compute_delta_time_embedding(ctx, dt_scalar);
    ggml_tensor* combined_base = compute_time_embedding(ctx, t_scalar);
    combined_base = ggml_add(ctx.raw_context(), combined_base, delta_time_embedding);
    forward_cfg_pair_projected(ctx,
                               x_proj,
                               mu,
                               combined_base,
                               cond_proj,
                               static_cast<int>(prefix_len),
                               conditioned,
                               unconditioned);
}

}  // namespace voxcpm
