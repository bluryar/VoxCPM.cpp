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

LocDiTModel::~LocDiTModel() {
    scratch_kv_cache_.reset();

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

    ggml_context* raw = ctx.raw_context();
    const int64_t seq_len = x->ne[1];
    const int64_t prefix_len = cond ? cond->ne[1] : 0;
    const int hidden_size = config().hidden_size;

    VOXCPM_ASSERT(x->ne[0] == feat_dim_);
    VOXCPM_ASSERT(mu->ne[0] == hidden_size);
    VOXCPM_ASSERT(prefix_len + seq_len + 1 <= config().max_length);

    scratch_kv_cache_->clear();

    ggml_tensor* x_proj = ggml_mul_mat(raw, weights_.in_proj_weight, x);
    x_proj = ggml_add(raw, x_proj, weights_.in_proj_bias);

    ggml_tensor* combined = ggml_add(raw, mu, compute_time_embedding(ctx, t_scalar));
    combined = ggml_add(raw, combined, compute_delta_time_embedding(ctx, dt_scalar));
    combined = ggml_reshape_2d(raw, combined, hidden_size, 1);

    ggml_tensor* seq = combined;
    if (cond && prefix_len > 0) {
        ggml_tensor* cond_proj = ggml_mul_mat(raw, weights_.cond_proj_weight, cond);
        cond_proj = ggml_add(raw, cond_proj, weights_.cond_proj_bias);
        seq = ggml_concat(raw, seq, cond_proj, 1);
    }
    seq = ggml_concat(raw, seq, x_proj, 1);

    ggml_tensor* hidden = decoder_.forward(ctx, seq, nullptr, *scratch_kv_cache_, false);
    ggml_tensor* hidden_out = ggml_view_2d(raw,
                                           hidden,
                                           hidden_size,
                                           seq_len,
                                           hidden->nb[1],
                                           static_cast<size_t>(prefix_len + 1) * hidden->nb[1]);
    ggml_tensor* output = ggml_mul_mat(raw, weights_.out_proj_weight, hidden_out);
    output = ggml_add(raw, output, weights_.out_proj_bias);
    return output;
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

}  // namespace voxcpm
