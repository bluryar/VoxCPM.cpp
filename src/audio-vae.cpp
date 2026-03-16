/**
 * @file audio-vae.cpp
 * @brief VoxCPM AudioVAE implementation
 */

#include "voxcpm/audio-vae.h"

#include "voxcpm/backend.h"
#include "voxcpm/weight-store.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace voxcpm {

struct AudioVAEDepthwiseConvOpData {
    int stride = 1;
    int dilation = 1;
    int padding = 0;
};

struct AudioVAEConv1DSpec {
    int64_t kernel = 0;
    int64_t in_channels = 0;
    int64_t out_channels = 0;
};

namespace {

static float load_f32_scalar(const uint8_t* ptr, ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return *reinterpret_cast<const float*>(ptr);
        case GGML_TYPE_F16:
            return ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t*>(ptr));
        default:
            VOXCPM_ASSERT(false && "unsupported tensor type");
            return 0.0f;
    }
}

static AudioVAEConv1DSpec resolve_conv1d_spec(const ggml_tensor* weight, int expected_kernel) {
    VOXCPM_ASSERT(weight != nullptr);

    AudioVAEConv1DSpec spec;
    if (ggml_n_dims(weight) == 3) {
        spec.kernel = weight->ne[0];
        spec.in_channels = weight->ne[1];
        spec.out_channels = weight->ne[2];
        return spec;
    }

    VOXCPM_ASSERT(ggml_n_dims(weight) == 2);
    VOXCPM_ASSERT(expected_kernel > 0);

    if (weight->ne[0] == expected_kernel) {
        spec.kernel = weight->ne[0];
        spec.in_channels = weight->ne[1];
        spec.out_channels = 1;
        return spec;
    }

    VOXCPM_ASSERT(weight->ne[0] % expected_kernel == 0);

    spec.kernel = expected_kernel;
    spec.in_channels = weight->ne[0] / expected_kernel;
    spec.out_channels = weight->ne[1];
    return spec;
}

static ggml_tensor* reshape_conv1d_weight_2d(ggml_context* ctx,
                                             ggml_tensor* weight,
                                             const AudioVAEConv1DSpec& spec) {
    if (ggml_n_dims(weight) == 2 && !(weight->ne[0] == spec.kernel && spec.out_channels == 1)) {
        return weight;
    }
    ggml_tensor* reshaped = ggml_reshape_2d(ctx, weight, spec.kernel * spec.in_channels, spec.out_channels);
    ggml_set_name(reshaped, weight->name);
    return reshaped;
}

static void depthwise_conv_custom(ggml_tensor* dst,
                                  const ggml_tensor* x,
                                  const ggml_tensor* weight,
                                  const ggml_tensor* bias,
                                  int ith,
                                  int nth,
                                  void* userdata) {
    const auto* op = static_cast<const AudioVAEDepthwiseConvOpData*>(userdata);
    const int64_t t_len = x->ne[0];
    const int64_t channels = x->ne[1];
    const int64_t batch = x->ne[2];
    const int64_t kernel = weight->ne[0];

    VOXCPM_ASSERT(batch >= 1);
    VOXCPM_ASSERT(weight->ne[1] == 1);
    VOXCPM_ASSERT(weight->ne[2] == channels);
    VOXCPM_ASSERT(dst->ne[0] == x->ne[0]);
    VOXCPM_ASSERT(dst->ne[1] == x->ne[1]);
    VOXCPM_ASSERT(dst->ne[2] == x->ne[2]);
    VOXCPM_ASSERT(x->type == GGML_TYPE_F32);
    VOXCPM_ASSERT(dst->type == GGML_TYPE_F32);
    VOXCPM_ASSERT(weight->type == GGML_TYPE_F32 || weight->type == GGML_TYPE_F16);
    VOXCPM_ASSERT(bias == nullptr || bias->type == GGML_TYPE_F32 || bias->type == GGML_TYPE_F16);

    const uint8_t* x_data = static_cast<const uint8_t*>(x->data);
    const uint8_t* w_data = static_cast<const uint8_t*>(weight->data);
    const uint8_t* b_data = bias ? static_cast<const uint8_t*>(bias->data) : nullptr;
    uint8_t* dst_data = static_cast<uint8_t*>(dst->data);

    const int64_t work_items = channels * batch;
    const int64_t items_per_thread = (work_items + nth - 1) / nth;
    const int64_t item_begin = ith * items_per_thread;
    const int64_t item_end = std::min<int64_t>(item_begin + items_per_thread, work_items);

    for (int64_t item = item_begin; item < item_end; ++item) {
        const int64_t b = item / channels;
        const int64_t c = item % channels;
        const float bias_val = b_data ? load_f32_scalar(b_data + c * bias->nb[0], bias->type) : 0.0f;

        const uint8_t* x_channel = x_data + b * x->nb[2] + c * x->nb[1];
        const uint8_t* w_channel = w_data + c * weight->nb[2];
        uint8_t* dst_channel = dst_data + b * dst->nb[2] + c * dst->nb[1];

        for (int64_t t = 0; t < t_len; ++t) {
            float sum = bias_val;
            for (int64_t k = 0; k < kernel; ++k) {
                const int64_t src_t = t * op->stride + k * op->dilation - op->padding * 2;
                if (src_t < 0 || src_t >= t_len) {
                    continue;
                }
                const float x_val = load_f32_scalar(x_channel + src_t * x->nb[0], x->type);
                const float w_val = load_f32_scalar(w_channel + k * weight->nb[0], weight->type);
                sum += x_val * w_val;
            }
            *reinterpret_cast<float*>(dst_channel + t * dst->nb[0]) = sum;
        }
    }
}

static ggml_tensor* conv1d_mul_mat_impl(ggml_context* ctx,
                                        ggml_tensor* weight,
                                        ggml_tensor* input,
                                        int expected_kernel,
                                        int stride,
                                        int dilation) {
    const AudioVAEConv1DSpec spec = resolve_conv1d_spec(weight, expected_kernel);
    ggml_tensor* weight_2d = reshape_conv1d_weight_2d(ctx, weight, spec);

    ggml_tensor* kernel_shape = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, spec.kernel, spec.in_channels, spec.out_channels);
    ggml_tensor* im2col =
        ggml_im2col(ctx, kernel_shape, input, stride, 0, 0, 0, dilation, 0, false, GGML_TYPE_F32);
    ggml_tensor* activations = ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[1] * im2col->ne[2]);
    ggml_tensor* result = ggml_mul_mat(ctx, weight_2d, activations);
    result = ggml_reshape_3d(ctx, result, spec.out_channels, im2col->ne[1], im2col->ne[2]);
    return ggml_cont(ctx, ggml_permute(ctx, result, 1, 0, 2, 3));
}

static std::string encoder_res_prefix(int block_idx, int res_idx) {
    return "audio_vae.encoder.block." + std::to_string(block_idx) + ".block." + std::to_string(res_idx) + ".block.";
}

static std::string decoder_res_prefix(int model_idx, int res_idx) {
    return "audio_vae.decoder.model." + std::to_string(model_idx) + ".block." + std::to_string(res_idx) + ".block.";
}

static int decoder_final_snake_model_idx(const AudioVAEConfig& config) {
    return config.num_decoder_blocks() + 2;
}

static int decoder_final_conv_model_idx(const AudioVAEConfig& config) {
    return config.num_decoder_blocks() + 3;
}

static bool get_required_tensor(ggml_context* ctx, const std::string& name, ggml_tensor** dst) {
    *dst = ggml_get_tensor(ctx, name.c_str());
    return *dst != nullptr;
}

static ggml_tensor* reshape_bias_3d(ggml_context* ctx, ggml_tensor* bias) {
    const int64_t channels = bias->ne[0];
    return ggml_reshape_3d(ctx, bias, 1, channels, 1);
}

}  // namespace

ggml_tensor* snake_activation(ggml_context* ctx, ggml_tensor* x, ggml_tensor* alpha, float eps) {
    const int64_t channels = alpha->ne[1] > 1 ? alpha->ne[1] : alpha->ne[0];
    ggml_tensor* alpha_view = ggml_reshape_3d(ctx, alpha, 1, channels, 1);
    ggml_tensor* alpha_broadcast = ggml_repeat(ctx, alpha_view, x);
    ggml_tensor* alpha_eps = ggml_add1(ctx, alpha_broadcast, ggml_arange(ctx, eps, eps + 1.0f, 1.0f));
    ggml_tensor* ax = ggml_mul(ctx, x, alpha_broadcast);
    ggml_tensor* sin_sq = ggml_sqr(ctx, ggml_sin(ctx, ax));
    ggml_tensor* one = ggml_arange(ctx, 1.0f, 2.0f, 1.0f);
    return ggml_add(ctx, x, ggml_mul(ctx, sin_sq, ggml_div(ctx, ggml_repeat(ctx, one, alpha_eps), alpha_eps)));
}

AudioVAE::AudioVAE(const AudioVAEConfig& config)
    : config_(config) {
}

AudioVAE::~AudioVAE() {
    if (weight_buffer_) {
        ggml_backend_buffer_free(weight_buffer_);
        weight_buffer_ = nullptr;
    }
    if (weight_ctx_) {
        ggml_free(weight_ctx_);
        weight_ctx_ = nullptr;
    }
}

bool AudioVAE::load_tensor_data(FILE* file,
                                gguf_context* gguf_ctx,
                                int tensor_idx,
                                ggml_tensor* tensor,
                                ggml_backend_buffer_t buffer) const {
    if (!file || !gguf_ctx || !tensor || !buffer) {
        return false;
    }

    const size_t offset = gguf_get_data_offset(gguf_ctx) + gguf_get_tensor_offset(gguf_ctx, tensor_idx);
    const size_t nbytes = ggml_nbytes(tensor);

    if (fseek(file, static_cast<long>(offset), SEEK_SET) != 0) {
        return false;
    }

    if (ggml_backend_buffer_is_host(buffer)) {
        return fread(tensor->data, 1, nbytes, file) == nbytes;
    }

    std::vector<uint8_t> temp(nbytes);
    if (fread(temp.data(), 1, nbytes, file) != nbytes) {
        return false;
    }
    ggml_backend_tensor_set(tensor, temp.data(), 0, nbytes);
    return true;
}

bool AudioVAE::load_encoder_weights(ggml_context* ggml_ctx_ptr) {
    bool ok = true;
    ok &= get_required_tensor(ggml_ctx_ptr, "audio_vae.encoder.block.0.weight", &weights_.encoder_block_0_weight);
    ok &= get_required_tensor(ggml_ctx_ptr, "audio_vae.encoder.block.0.bias", &weights_.encoder_block_0_bias);
    ok &= get_required_tensor(ggml_ctx_ptr, "audio_vae.encoder.fc_mu.weight", &weights_.encoder_fc_mu_weight);
    ok &= get_required_tensor(ggml_ctx_ptr, "audio_vae.encoder.fc_mu.bias", &weights_.encoder_fc_mu_bias);

    weights_.encoder_blocks.resize(static_cast<size_t>(config_.num_encoder_blocks()));
    for (int i = 0; i < config_.num_encoder_blocks(); ++i) {
        EncoderBlockWeights& block = weights_.encoder_blocks[static_cast<size_t>(i)];
        const int block_idx = i + 1;
        const std::string block_prefix = "audio_vae.encoder.block." + std::to_string(block_idx) + ".block.";

        auto load_res = [&](ResidualUnitWeights& res, int res_idx) {
            const std::string prefix = encoder_res_prefix(block_idx, res_idx);
            ok &= get_required_tensor(ggml_ctx_ptr, prefix + "0.alpha", &res.snake1_alpha);
            ok &= get_required_tensor(ggml_ctx_ptr, prefix + "1.weight", &res.conv1_weight);
            ok &= get_required_tensor(ggml_ctx_ptr, prefix + "1.bias", &res.conv1_bias);
            ok &= get_required_tensor(ggml_ctx_ptr, prefix + "2.alpha", &res.snake2_alpha);
            ok &= get_required_tensor(ggml_ctx_ptr, prefix + "3.weight", &res.conv2_weight);
            ok &= get_required_tensor(ggml_ctx_ptr, prefix + "3.bias", &res.conv2_bias);
        };

        load_res(block.res0, 0);
        load_res(block.res1, 1);
        load_res(block.res2, 2);
        ok &= get_required_tensor(ggml_ctx_ptr, block_prefix + "3.alpha", &block.snake_alpha);
        ok &= get_required_tensor(ggml_ctx_ptr, block_prefix + "4.weight", &block.conv_weight);
        ok &= get_required_tensor(ggml_ctx_ptr, block_prefix + "4.bias", &block.conv_bias);
    }

    return ok;
}

bool AudioVAE::load_decoder_weights(ggml_context* ggml_ctx_ptr) {
    bool ok = true;
    ok &= get_required_tensor(ggml_ctx_ptr, "audio_vae.decoder.model.0.weight", &weights_.decoder_model_0_weight);
    ok &= get_required_tensor(ggml_ctx_ptr, "audio_vae.decoder.model.0.bias", &weights_.decoder_model_0_bias);
    ok &= get_required_tensor(ggml_ctx_ptr, "audio_vae.decoder.model.1.weight", &weights_.decoder_model_1_weight);
    ok &= get_required_tensor(ggml_ctx_ptr, "audio_vae.decoder.model.1.bias", &weights_.decoder_model_1_bias);

    const int final_snake_idx = decoder_final_snake_model_idx(config_);
    const int final_conv_idx = decoder_final_conv_model_idx(config_);

    ok &= get_required_tensor(ggml_ctx_ptr,
                              "audio_vae.decoder.model." + std::to_string(final_snake_idx) + ".alpha",
                              &weights_.decoder_final_snake_alpha);
    ok &= get_required_tensor(ggml_ctx_ptr,
                              "audio_vae.decoder.model." + std::to_string(final_conv_idx) + ".weight",
                              &weights_.decoder_final_conv_weight);
    ok &= get_required_tensor(ggml_ctx_ptr,
                              "audio_vae.decoder.model." + std::to_string(final_conv_idx) + ".bias",
                              &weights_.decoder_final_conv_bias);

    weights_.decoder_blocks.resize(static_cast<size_t>(config_.num_decoder_blocks()));
    for (int i = 0; i < config_.num_decoder_blocks(); ++i) {
        DecoderBlockWeights& block = weights_.decoder_blocks[static_cast<size_t>(i)];
        const int model_idx = i + 2;
        const std::string block_prefix = "audio_vae.decoder.model." + std::to_string(model_idx) + ".block.";

        ok &= get_required_tensor(ggml_ctx_ptr, block_prefix + "0.alpha", &block.snake_alpha);
        ok &= get_required_tensor(ggml_ctx_ptr, block_prefix + "1.weight", &block.conv_weight);
        ok &= get_required_tensor(ggml_ctx_ptr, block_prefix + "1.bias", &block.conv_bias);

        auto load_res = [&](ResidualUnitWeights& res, int res_idx) {
            const std::string prefix = decoder_res_prefix(model_idx, res_idx + 2);
            ok &= get_required_tensor(ggml_ctx_ptr, prefix + "0.alpha", &res.snake1_alpha);
            ok &= get_required_tensor(ggml_ctx_ptr, prefix + "1.weight", &res.conv1_weight);
            ok &= get_required_tensor(ggml_ctx_ptr, prefix + "1.bias", &res.conv1_bias);
            ok &= get_required_tensor(ggml_ctx_ptr, prefix + "2.alpha", &res.snake2_alpha);
            ok &= get_required_tensor(ggml_ctx_ptr, prefix + "3.weight", &res.conv2_weight);
            ok &= get_required_tensor(ggml_ctx_ptr, prefix + "3.bias", &res.conv2_bias);
        };

        load_res(block.res0, 0);
        load_res(block.res1, 1);
        load_res(block.res2, 2);
    }

    return ok;
}

bool AudioVAE::load_from_gguf(const std::string& gguf_path,
                              VoxCPMContext& weight_ctx,
                              VoxCPMContext& graph_ctx,
                              VoxCPMBackend& backend) {
    VOXCPM_UNUSED(weight_ctx);
    VOXCPM_UNUSED(graph_ctx);

    auto store = std::make_shared<VoxCPMWeightStore>();
    if (!store->load_from_file(gguf_path, backend)) {
        return false;
    }
    return load_from_store(store);
}

bool AudioVAE::load_from_store(const std::shared_ptr<VoxCPMWeightStore>& store) {
    if (!store || !store->owns_storage()) {
        return false;
    }

    shared_store_ = store;

    uint32_t u32 = 0;
    const bool has_encoder_dim = store->get_u32("voxcpm_audio_vae_config_encoder_dim", u32);
    if (has_encoder_dim) config_.encoder_dim = static_cast<int>(u32);
    const bool has_decoder_dim = store->get_u32("voxcpm_audio_vae_config_decoder_dim", u32);
    if (has_decoder_dim) config_.decoder_dim = static_cast<int>(u32);
    const bool has_latent_dim = store->get_u32("voxcpm_audio_vae_config_latent_dim", u32);
    if (has_latent_dim) config_.latent_dim = static_cast<int>(u32);
    const bool has_sample_rate = store->get_u32("voxcpm_audio_vae_config_sample_rate", u32);
    if (has_sample_rate) config_.sample_rate = static_cast<int>(u32);
    const bool has_depthwise = store->get_bool("voxcpm_audio_vae_config_depthwise", config_.depthwise);
    const bool has_use_noise_block = store->get_bool("voxcpm_audio_vae_config_use_noise_block", config_.use_noise_block);
    const bool has_encoder_rates = store->get_i32_array("voxcpm_audio_vae_config_encoder_rates", config_.encoder_rates);
    const bool has_decoder_rates = store->get_i32_array("voxcpm_audio_vae_config_decoder_rates", config_.decoder_rates);

    if (!has_encoder_dim || !has_decoder_dim || !has_latent_dim || !has_sample_rate ||
        !has_encoder_rates || !has_decoder_rates) {
        return false;
    }
    if (config_.encoder_rates.empty() || config_.decoder_rates.empty()) {
        return false;
    }
    if (!has_depthwise) {
        config_.depthwise = true;
    }
    if (!has_use_noise_block) {
        config_.use_noise_block = false;
    }

    return load_encoder_weights(store->ggml_ctx()) && load_decoder_weights(store->ggml_ctx());
}

std::vector<float> AudioVAE::preprocess(std::vector<float> audio_data, int sample_rate) const {
    const int actual_sample_rate = sample_rate < 0 ? config_.sample_rate : sample_rate;
    VOXCPM_ASSERT(actual_sample_rate == config_.sample_rate);

    const int hop = config_.hop_length();
    const size_t length = audio_data.size();
    const size_t aligned = ((length + static_cast<size_t>(hop) - 1) / static_cast<size_t>(hop)) * static_cast<size_t>(hop);
    audio_data.resize(aligned, 0.0f);
    return audio_data;
}

ggml_tensor* AudioVAE::causal_conv1d(ggml_context* ctx,
                                     ggml_tensor* x,
                                     ggml_tensor* weight,
                                     ggml_tensor* bias,
                                     int kernel_size,
                                     int stride,
                                     int dilation,
                                     int padding) const {
    ggml_tensor* padded = x;
    if (padding > 0) {
        padded = ggml_pad_ext(ctx, x, padding * 2, 0, 0, 0, 0, 0, 0, 0);
    }
    ggml_tensor* result = conv1d_mul_mat_impl(ctx, weight, padded, kernel_size, stride, dilation);
    if (bias) {
        result = ggml_add(ctx, result, reshape_bias_3d(ctx, bias));
    }
    return result;
}

ggml_tensor* AudioVAE::causal_conv1d_dw(ggml_context* ctx,
                                        ggml_tensor* x,
                                        ggml_tensor* weight,
                                        ggml_tensor* bias,
                                        int stride,
                                        int dilation,
                                        int padding) const {
    auto op = std::make_unique<AudioVAEDepthwiseConvOpData>();
    op->stride = stride;
    op->dilation = dilation;
    op->padding = padding;

    AudioVAEDepthwiseConvOpData* op_ptr = op.get();
    depthwise_ops_.push_back(std::move(op));
    return ggml_map_custom3(ctx, x, weight, bias, depthwise_conv_custom, GGML_N_TASKS_MAX, op_ptr);
}

ggml_tensor* AudioVAE::causal_transpose_conv1d(ggml_context* ctx,
                                               ggml_tensor* x,
                                               ggml_tensor* weight,
                                               ggml_tensor* bias,
                                               int stride,
                                               int padding,
                                               int output_padding) const {
    ggml_tensor* result = ggml_conv_transpose_1d(ctx, weight, x, stride, 0, 1);
    if (result->ne[3] == 1) {
        result = ggml_reshape_3d(ctx, result, result->ne[0], result->ne[1], result->ne[2]);
    }

    const int crop = padding * 2 - output_padding;
    if (crop > 0) {
        result = ggml_view_3d(ctx, result, result->ne[0] - crop, result->ne[1], result->ne[2], result->nb[1], result->nb[2], 0);
    }
    if (bias) {
        result = ggml_add(ctx, result, reshape_bias_3d(ctx, bias));
    }
    return result;
}

ggml_tensor* AudioVAE::residual_unit_forward(ggml_context* ctx,
                                             ggml_tensor* x,
                                             const ResidualUnitWeights& weights,
                                             int dilation) const {
    ggml_tensor* h = snake_activation(ctx, x, weights.snake1_alpha);
    h = causal_conv1d_dw(ctx, h, weights.conv1_weight, weights.conv1_bias, 1, dilation, ((7 - 1) * dilation) / 2);
    h = snake_activation(ctx, h, weights.snake2_alpha);
    h = causal_conv1d(ctx, h, weights.conv2_weight, weights.conv2_bias, 1, 1, 1, 0);

    if (x->ne[0] != h->ne[0]) {
        const int64_t target = std::min<int64_t>(x->ne[0], h->ne[0]);
        x = ggml_view_3d(ctx, x, target, x->ne[1], x->ne[2], x->nb[1], x->nb[2], 0);
        h = ggml_view_3d(ctx, h, target, h->ne[1], h->ne[2], h->nb[1], h->nb[2], 0);
    }
    return ggml_add(ctx, x, h);
}

ggml_tensor* AudioVAE::encoder_block_forward(ggml_context* ctx,
                                             ggml_tensor* x,
                                             const EncoderBlockWeights& weights,
                                             int stride) const {
    x = residual_unit_forward(ctx, x, weights.res0, 1);
    x = residual_unit_forward(ctx, x, weights.res1, 3);
    x = residual_unit_forward(ctx, x, weights.res2, 9);
    x = snake_activation(ctx, x, weights.snake_alpha);
    return causal_conv1d(
        ctx,
        x,
        weights.conv_weight,
        weights.conv_bias,
        stride * 2,
        stride,
        1,
        static_cast<int>(std::ceil(stride / 2.0f)));
}

ggml_tensor* AudioVAE::decoder_block_forward(ggml_context* ctx,
                                             ggml_tensor* x,
                                             const DecoderBlockWeights& weights,
                                             int stride) const {
    x = snake_activation(ctx, x, weights.snake_alpha);
    x = causal_transpose_conv1d(ctx,
                                x,
                                weights.conv_weight,
                                weights.conv_bias,
                                stride,
                                static_cast<int>(std::ceil(stride / 2.0f)),
                                stride % 2);
    x = residual_unit_forward(ctx, x, weights.res0, 1);
    x = residual_unit_forward(ctx, x, weights.res1, 3);
    x = residual_unit_forward(ctx, x, weights.res2, 9);
    return x;
}

ggml_tensor* AudioVAE::encode_tensor(VoxCPMContext& ctx, ggml_tensor* audio) const {
    ggml_context* raw = ctx.raw_context();
    ggml_tensor* x = causal_conv1d(raw, audio, weights_.encoder_block_0_weight, weights_.encoder_block_0_bias, 7, 1, 1, 3);

    for (int i = 0; i < config_.num_encoder_blocks(); ++i) {
        x = encoder_block_forward(raw, x, weights_.encoder_blocks[static_cast<size_t>(i)], config_.encoder_rates[static_cast<size_t>(i)]);
    }

    return causal_conv1d(raw, x, weights_.encoder_fc_mu_weight, weights_.encoder_fc_mu_bias, 3, 1, 1, 1);
}

ggml_tensor* AudioVAE::encode(VoxCPMContext& ctx, std::vector<float>& audio_data, int sample_rate) {
    depthwise_ops_.clear();
    last_preprocessed_audio_ = preprocess(audio_data, sample_rate);
    last_input_tensor_ = ctx.new_tensor_3d(GGML_TYPE_F32, static_cast<int64_t>(last_preprocessed_audio_.size()), 1, 1);
    VOXCPM_ASSERT(last_input_tensor_ != nullptr);
    ggml_set_input(last_input_tensor_);

    ggml_tensor* latent = encode_tensor(ctx, last_input_tensor_);
    ggml_set_output(latent);
    return latent;
}

ggml_tensor* AudioVAE::decode(VoxCPMContext& ctx, ggml_tensor* z) {
    depthwise_ops_.clear();
    VOXCPM_ASSERT(z != nullptr);
    ggml_context* raw = ctx.raw_context();

    ggml_tensor* x = z;
    if (ggml_n_dims(x) == 2) {
        x = ggml_reshape_3d(raw, x, x->ne[0], x->ne[1], 1);
    }
    VOXCPM_ASSERT(x->ne[1] == config_.latent_dim);
    VOXCPM_ASSERT(x->ne[2] == 1);

    x = causal_conv1d_dw(raw, x, weights_.decoder_model_0_weight, weights_.decoder_model_0_bias, 1, 1, 3);
    x = causal_conv1d(raw, x, weights_.decoder_model_1_weight, weights_.decoder_model_1_bias, 1, 1, 1, 0);

    for (int i = 0; i < config_.num_decoder_blocks(); ++i) {
        x = decoder_block_forward(raw, x, weights_.decoder_blocks[static_cast<size_t>(i)], config_.decoder_rates[static_cast<size_t>(i)]);
    }

    x = snake_activation(raw, x, weights_.decoder_final_snake_alpha);
    x = causal_conv1d(raw, x, weights_.decoder_final_conv_weight, weights_.decoder_final_conv_bias, 7, 1, 1, 3);
    x = ggml_tanh(raw, x);
    ggml_set_output(x);
    return x;
}

}  // namespace voxcpm
