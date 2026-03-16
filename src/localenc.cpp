/**
 * @file localenc.cpp
 * @brief VoxCPM Local Encoder implementation
 */

#include "voxcpm/localenc.h"

#include "voxcpm/backend.h"
#include "voxcpm/weight-store.h"

#include <cstdio>

namespace voxcpm {

LocEncModel::~LocEncModel() {
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

bool LocEncModel::init_scratch_cache(VoxCPMBackend& backend) {
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

bool LocEncModel::load_from_gguf(const std::string& gguf_path,
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

bool LocEncModel::load_from_store(const std::shared_ptr<VoxCPMWeightStore>& store,
                                  VoxCPMBackend& backend) {
    if (!store || !store->owns_storage()) {
        return false;
    }

    shared_store_ = store;
    weights_.in_proj_weight = store->get_tensor("locenc.in_proj.weight");
    weights_.in_proj_bias = store->get_tensor("locenc.in_proj.bias");
    weights_.special_token = store->get_tensor("locenc.special_token");
    if (!weights_.in_proj_weight || !weights_.in_proj_bias || !weights_.special_token) {
        return false;
    }

    feat_dim_ = static_cast<int>(weights_.in_proj_weight->ne[0]);

    if (!encoder_.load_from_store(store, "locenc", backend)) {
        return false;
    }
    if (config().hidden_size != static_cast<int>(weights_.special_token->ne[0])) {
        return false;
    }

    backend_ = &backend;
    return init_scratch_cache(backend);
}

ggml_tensor* LocEncModel::forward_patch(VoxCPMContext& ctx, ggml_tensor* input) {
    VOXCPM_ASSERT(input != nullptr);
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(scratch_kv_cache_ != nullptr);
    VOXCPM_ASSERT(input->ne[1] > 0);

    ggml_context* raw = ctx.raw_context();
    const int64_t n_patches = input->ne[1];
    const int hidden_size = config().hidden_size;

    VOXCPM_ASSERT(n_patches + 1 <= config().max_length);
    VOXCPM_ASSERT(input->ne[0] == feat_dim_ || input->ne[0] == hidden_size);

    scratch_kv_cache_->clear();

    ggml_tensor* projected = input;
    if (input->ne[0] != hidden_size) {
        projected = ggml_mul_mat(raw, weights_.in_proj_weight, input);
        projected = ggml_add(raw, projected, weights_.in_proj_bias);
    }

    ggml_tensor* cls = ggml_reshape_2d(raw, weights_.special_token, hidden_size, 1);
    ggml_tensor* full_input = ggml_concat(raw, cls, projected, 1);
    ggml_tensor* output = encoder_.forward(ctx, full_input, nullptr, *scratch_kv_cache_, false);

    return ggml_view_1d(raw, output, hidden_size, 0);
}

}  // namespace voxcpm
