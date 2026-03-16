/**
 * @file fsq.cpp
 * @brief Finite scalar quantization implementation
 */

#include "voxcpm/fsq.h"

#include "voxcpm/backend.h"
#include "voxcpm/weight-store.h"

#include <cstdio>
#include <vector>

namespace voxcpm {

namespace {

static bool load_tensor_data(FILE* file,
                             gguf_context* gguf_ctx,
                             int tensor_idx,
                             ggml_tensor* tensor,
                             ggml_backend_buffer_t buffer) {
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

static bool get_u32_kv(gguf_context* gguf_ctx, const char* key, uint32_t& value) {
    const int idx = gguf_find_key(gguf_ctx, key);
    if (idx < 0) {
        return false;
    }
    value = gguf_get_val_u32(gguf_ctx, idx);
    return true;
}

static ggml_tensor* reshape_bias_for_output(ggml_context* ctx, ggml_tensor* bias, ggml_tensor* output) {
    VOXCPM_ASSERT(ctx != nullptr);
    VOXCPM_ASSERT(bias != nullptr);
    VOXCPM_ASSERT(output != nullptr);

    const int n_dims = ggml_n_dims(output);
    if (n_dims <= 1) {
        return bias;
    }
    if (n_dims == 2) {
        return ggml_reshape_2d(ctx, bias, bias->ne[0], 1);
    }
    return ggml_reshape_3d(ctx, bias, bias->ne[0], 1, 1);
}

static ggml_tensor* add_bias(ggml_context* ctx, ggml_tensor* output, ggml_tensor* bias) {
    if (!bias) {
        return output;
    }
    return ggml_add(ctx, output, reshape_bias_for_output(ctx, bias, output));
}

}  // namespace

ggml_tensor* fsq_quantize(ggml_context* ctx, ggml_tensor* x, int scale) {
    VOXCPM_ASSERT(ctx != nullptr);
    VOXCPM_ASSERT(x != nullptr);
    VOXCPM_ASSERT(scale > 0);

    ggml_tensor* scaled = ggml_scale(ctx, x, static_cast<float>(scale));
    ggml_tensor* rounded = ggml_round(ctx, scaled);
    return ggml_scale(ctx, rounded, 1.0f / static_cast<float>(scale));
}

FSQ::FSQ(const FSQConfig& config)
    : config_(config) {
}

FSQ::~FSQ() {
    if (weight_buffer_) {
        ggml_backend_buffer_free(weight_buffer_);
        weight_buffer_ = nullptr;
    }
    if (weight_ctx_) {
        ggml_free(weight_ctx_);
        weight_ctx_ = nullptr;
    }
}

bool FSQ::update_config_from_gguf(gguf_context* gguf_ctx) {
    uint32_t value = 0;

    if (get_u32_kv(gguf_ctx, "voxcpm_scalar_quantization_latent_dim", value)) {
        config_.latent_dim = static_cast<int>(value);
    }
    if (get_u32_kv(gguf_ctx, "voxcpm_scalar_quantization_scale", value)) {
        config_.scale = static_cast<int>(value);
    }
    if (get_u32_kv(gguf_ctx, "voxcpm_lm_config_hidden_size", value)) {
        config_.hidden_size = static_cast<int>(value);
    }

    return true;
}

bool FSQ::load_weight_data(FILE* file, gguf_context* gguf_ctx) {
    const int n_tensors = gguf_get_n_tensors(gguf_ctx);
    for (int i = 0; i < n_tensors; ++i) {
        const char* name = gguf_get_tensor_name(gguf_ctx, i);
        if (strncmp(name, "fsq.", 4) != 0) {
            continue;
        }

        ggml_tensor* tensor = ggml_get_tensor(weight_ctx_, name);
        if (tensor && !load_tensor_data(file, gguf_ctx, i, tensor, weight_buffer_)) {
            return false;
        }
    }

    return true;
}

bool FSQ::load_from_gguf(const std::string& gguf_path,
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

bool FSQ::load_from_store(const std::shared_ptr<VoxCPMWeightStore>& store) {
    if (!store || !store->owns_storage()) {
        return false;
    }

    shared_store_ = store;
    update_config_from_gguf(store->gguf());

    weights_.in_proj_weight = store->get_tensor("fsq.in_proj.weight");
    weights_.in_proj_bias = store->get_tensor("fsq.in_proj.bias");
    weights_.out_proj_weight = store->get_tensor("fsq.out_proj.weight");
    weights_.out_proj_bias = store->get_tensor("fsq.out_proj.bias");
    if (!weights_.in_proj_weight || !weights_.in_proj_bias ||
        !weights_.out_proj_weight || !weights_.out_proj_bias) {
        return false;
    }

    config_.hidden_size = static_cast<int>(weights_.in_proj_weight->ne[0]);
    config_.latent_dim = static_cast<int>(weights_.in_proj_weight->ne[1]);
    return true;
}

ggml_tensor* FSQ::forward(VoxCPMContext& ctx, ggml_tensor* hidden) {
    VOXCPM_ASSERT(hidden != nullptr);
    VOXCPM_ASSERT(weights_.in_proj_weight != nullptr);
    VOXCPM_ASSERT(weights_.out_proj_weight != nullptr);
    VOXCPM_ASSERT(hidden->ne[0] == weights_.in_proj_weight->ne[0]);

    ggml_context* raw = ctx.raw_context();

    ggml_tensor* latent = ggml_mul_mat(raw, weights_.in_proj_weight, hidden);
    latent = add_bias(raw, latent, weights_.in_proj_bias);
    latent = ggml_tanh(raw, latent);
    latent = quantize(raw, latent);

    ggml_tensor* output = ggml_mul_mat(raw, weights_.out_proj_weight, latent);
    return add_bias(raw, output, weights_.out_proj_bias);
}

ggml_tensor* FSQ::quantize(ggml_context* ctx, ggml_tensor* x) {
    return fsq_quantize(ctx, x, config_.scale);
}

}  // namespace voxcpm
