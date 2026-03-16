/**
 * @file unified_cfm.cpp
 * @brief VoxCPM Unified Conditional Flow Matching solver implementation
 */

#include "voxcpm/unified_cfm.h"

#include <algorithm>
#include <cmath>

namespace voxcpm {

UnifiedCFM::UnifiedCFM(LocDiTModel& estimator, const CFMConfig& config)
    : estimator_(estimator),
      config_(config) {
}

std::vector<float> UnifiedCFM::compute_t_span(int n_timesteps, float sway_sampling_coef) {
    VOXCPM_ASSERT(n_timesteps > 0);

    std::vector<float> t_span(static_cast<size_t>(n_timesteps) + 1);
    for (int i = 0; i <= n_timesteps; ++i) {
        const float base = 1.0f - static_cast<float>(i) / static_cast<float>(n_timesteps);
        t_span[static_cast<size_t>(i)] =
            base + sway_sampling_coef * (std::cos(static_cast<float>(M_PI_2) * base) - 1.0f + base);
    }
    return t_span;
}

ggml_tensor* UnifiedCFM::optimized_scale(VoxCPMContext& ctx,
                                         ggml_tensor* positive,
                                         ggml_tensor* negative,
                                         float eps) const {
    ggml_context* raw = ctx.raw_context();

    ggml_tensor* dot_product = ggml_sum(raw, ggml_mul(raw, positive, negative));
    ggml_tensor* squared_norm = ggml_sum(raw, ggml_mul(raw, negative, negative));
    ggml_tensor* eps_tensor = ggml_arange(raw, eps, eps + 1.0f, 1.0f);
    ggml_tensor* denom = ggml_add(raw, squared_norm, eps_tensor);
    return ggml_div(raw, dot_product, denom);
}

ggml_tensor* UnifiedCFM::compute_velocity_with_cfg(VoxCPMContext& ctx,
                                                   ggml_tensor* x,
                                                   ggml_tensor* mu,
                                                   ggml_tensor* cond,
                                                   float t,
                                                   float dt,
                                                   float cfg_value,
                                                   bool use_cfg_zero_star) {
    VOXCPM_UNUSED(dt);

    ggml_context* raw = ctx.raw_context();
    const int64_t feat_dim = x->ne[0];
    const int64_t seq_len = x->ne[1];
    const int64_t cond_len = cond->ne[1];
    const int64_t hidden_size = mu->ne[0];

    ggml_tensor* x_target = ggml_new_tensor_3d(raw, GGML_TYPE_F32, feat_dim, seq_len, 2);
    ggml_tensor* x_in = ggml_repeat(raw, x, x_target);

    ggml_tensor* cond_target = ggml_new_tensor_3d(raw, GGML_TYPE_F32, feat_dim, cond_len, 2);
    ggml_tensor* cond_in = ggml_repeat(raw, cond, cond_target);

    ggml_tensor* mu_2d = ggml_reshape_2d(raw, mu, hidden_size, 1);
    ggml_tensor* mu_target = ggml_new_tensor_2d(raw, GGML_TYPE_F32, hidden_size, 2);
    ggml_tensor* mu_repeat = ggml_repeat(raw, mu_2d, mu_target);
    ggml_tensor* mu_mask = ggml_arange(raw, 0.0f, 2.0f, 1.0f);
    mu_mask = ggml_scale(raw, mu_mask, -1.0f);
    mu_mask = ggml_add1(raw, mu_mask, ggml_arange(raw, 1.0f, 2.0f, 1.0f));
    mu_mask = ggml_reshape_2d(raw, mu_mask, 1, 2);
    ggml_tensor* mu_mask_broadcast = ggml_repeat(raw, mu_mask, mu_repeat);
    ggml_tensor* mu_in = ggml_mul(raw, mu_repeat, mu_mask_broadcast);

    ggml_tensor* t_scalar = ggml_arange(raw, t, t + 1.0f, 1.0f);
    ggml_tensor* t_target = ggml_new_tensor_1d(raw, GGML_TYPE_F32, 2);
    ggml_tensor* t_in = ggml_repeat(raw, t_scalar, t_target);

    ggml_tensor* zero_scalar = ggml_arange(raw, 0.0f, 1.0f, 1.0f);
    ggml_tensor* dt_target = ggml_new_tensor_1d(raw, GGML_TYPE_F32, 2);
    ggml_tensor* dt_in = ggml_repeat(raw, zero_scalar, dt_target);

    ggml_tensor* velocity = estimator_.forward(ctx, x_in, mu_in, t_in, cond_in, dt_in);
    ggml_tensor* dphi_dt_cond = ggml_view_2d(raw, velocity, feat_dim, seq_len, velocity->nb[1], 0);
    ggml_tensor* dphi_dt_uncond = ggml_view_2d(raw,
                                               velocity,
                                               feat_dim,
                                               seq_len,
                                               velocity->nb[1],
                                               velocity->nb[2]);

    if (use_cfg_zero_star) {
        ggml_tensor* st_star = optimized_scale(ctx, dphi_dt_cond, dphi_dt_uncond);
        ggml_tensor* uncond_scaled = ggml_mul(raw, dphi_dt_uncond, ggml_repeat(raw, st_star, dphi_dt_uncond));
        ggml_tensor* diff = ggml_sub(raw, dphi_dt_cond, uncond_scaled);
        ggml_tensor* cfg_scaled = ggml_scale(raw, diff, cfg_value);
        return ggml_add(raw, uncond_scaled, cfg_scaled);
    }

    ggml_tensor* diff = ggml_sub(raw, dphi_dt_cond, dphi_dt_uncond);
    ggml_tensor* cfg_scaled = ggml_scale(raw, diff, cfg_value);
    return ggml_add(raw, dphi_dt_uncond, cfg_scaled);
}

ggml_tensor* UnifiedCFM::solve_euler(VoxCPMContext& ctx,
                                     ggml_tensor* x,
                                     const std::vector<float>& t_span,
                                     ggml_tensor* mu,
                                     ggml_tensor* cond,
                                     float cfg_value,
                                     bool use_cfg_zero_star) {
    VOXCPM_ASSERT(x != nullptr);
    VOXCPM_ASSERT(mu != nullptr);
    VOXCPM_ASSERT(cond != nullptr);
    VOXCPM_ASSERT(t_span.size() >= 2);

    ggml_context* raw = ctx.raw_context();
    float t = t_span[0];
    float dt = t_span[0] - t_span[1];

    const int n_steps = static_cast<int>(t_span.size()) - 1;
    const int zero_init_steps = n_steps > 1
        ? std::max(1, static_cast<int>(t_span.size() * 0.04f))
        : 0;

    for (int step = 1; step <= n_steps; ++step) {
        ggml_tensor* dphi_dt = nullptr;

        if (use_cfg_zero_star && step <= zero_init_steps) {
            dphi_dt = ggml_scale(raw, x, 0.0f);
        } else {
            dphi_dt = compute_velocity_with_cfg(ctx, x, mu, cond, t, dt, cfg_value, use_cfg_zero_star);
        }

        x = ggml_sub(raw, x, ggml_scale(raw, dphi_dt, dt));
        t -= dt;

        if (step < n_steps) {
            dt = t - t_span[static_cast<size_t>(step + 1)];
        }
    }

    return x;
}

ggml_tensor* UnifiedCFM::forward(VoxCPMContext& ctx,
                                 ggml_tensor* z,
                                 ggml_tensor* mu,
                                 int patch_size,
                                 ggml_tensor* cond,
                                 int n_timesteps,
                                 float cfg_value,
                                 float temperature,
                                 float sway_sampling_coef,
                                 bool use_cfg_zero_star) {
    VOXCPM_UNUSED(patch_size);

    VOXCPM_ASSERT(z != nullptr);
    VOXCPM_ASSERT(mu != nullptr);
    VOXCPM_ASSERT(cond != nullptr);
    VOXCPM_ASSERT(ggml_n_dims(z) == 2);
    VOXCPM_ASSERT(ggml_n_dims(mu) == 1);
    VOXCPM_ASSERT(ggml_n_dims(cond) == 2);

    ggml_tensor* x = (temperature == 1.0f) ? z : ggml_scale(ctx.raw_context(), z, temperature);
    const std::vector<float> t_span = compute_t_span(n_timesteps, sway_sampling_coef);

    return solve_euler(ctx, x, t_span, mu, cond, cfg_value, use_cfg_zero_star);
}

}  // namespace voxcpm
