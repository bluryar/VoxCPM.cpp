/**
 * @file unified_cfm.h
 * @brief VoxCPM Unified Conditional Flow Matching solver
 */

#ifndef VOXCPM_UNIFIED_CFM_H
#define VOXCPM_UNIFIED_CFM_H

#include "voxcpm/context.h"
#include "voxcpm/locdit.h"

#include <vector>

namespace voxcpm {

struct CFMConfig {
    float sigma_min = 1.0e-6f;
    float inference_cfg_rate = 1.0f;
    float temperature = 1.0f;
    float sway_sampling_coef = 1.0f;
    bool use_cfg_zero_star = true;
};

class UnifiedCFM {
public:
    explicit UnifiedCFM(LocDiTModel& estimator, const CFMConfig& config = CFMConfig());

    ggml_tensor* forward(VoxCPMContext& ctx,
                         ggml_tensor* z,
                         ggml_tensor* mu,
                         int patch_size,
                         ggml_tensor* cond,
                         int n_timesteps,
                         float cfg_value,
                         float temperature = 1.0f,
                         float sway_sampling_coef = 1.0f,
                         bool use_cfg_zero_star = true);

    static std::vector<float> compute_t_span(int n_timesteps, float sway_sampling_coef);

    const CFMConfig& config() const { return config_; }

private:
    ggml_tensor* optimized_scale(VoxCPMContext& ctx,
                                 ggml_tensor* positive,
                                 ggml_tensor* negative,
                                 float eps = 1.0e-8f) const;

    ggml_tensor* compute_velocity_with_cfg(VoxCPMContext& ctx,
                                           ggml_tensor* x,
                                           ggml_tensor* mu,
                                           ggml_tensor* cond,
                                           float t,
                                           float dt,
                                           float cfg_value,
                                           bool use_cfg_zero_star);

    ggml_tensor* solve_euler(VoxCPMContext& ctx,
                             ggml_tensor* x,
                             const std::vector<float>& t_span,
                             ggml_tensor* mu,
                             ggml_tensor* cond,
                             float cfg_value,
                             bool use_cfg_zero_star);

    LocDiTModel& estimator_;
    CFMConfig config_;
};

}  // namespace voxcpm

#endif  // VOXCPM_UNIFIED_CFM_H
