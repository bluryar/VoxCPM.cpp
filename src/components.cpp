/**
 * @file components.cpp
 * @brief VoxCPM auxiliary components implementation
 */

#include "voxcpm/components.h"

#include "voxcpm/backend.h"
#include "voxcpm/weight-store.h"

#include <cstdio>

namespace voxcpm {

namespace {

static std::string normalize_prefix(const std::string& prefix) {
    if (prefix.empty() || prefix.back() == '.') {
        return prefix;
    }
    return prefix + ".";
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

LinearProjection::LinearProjection(const ProjectionConfig& config)
    : config_(config) {
}

LinearProjection::~LinearProjection() {
    if (weight_buffer_) {
        ggml_backend_buffer_free(weight_buffer_);
        weight_buffer_ = nullptr;
    }
    if (weight_ctx_) {
        ggml_free(weight_ctx_);
        weight_ctx_ = nullptr;
    }
}

bool LinearProjection::load_from_gguf(const std::string& gguf_path,
                                      const std::string& weight_prefix,
                                      VoxCPMContext& weight_ctx,
                                      VoxCPMContext& graph_ctx,
                                      VoxCPMBackend& backend) {
    VOXCPM_UNUSED(weight_ctx);
    VOXCPM_UNUSED(graph_ctx);

    auto store = std::make_shared<VoxCPMWeightStore>();
    if (!store->load_from_file(gguf_path, backend)) {
        return false;
    }
    return load_from_store(store, weight_prefix);
}

bool LinearProjection::load_from_store(const std::shared_ptr<VoxCPMWeightStore>& store,
                                       const std::string& weight_prefix) {
    if (!store || !store->owns_storage()) {
        return false;
    }

    shared_store_ = store;
    const std::string prefix = normalize_prefix(weight_prefix);
    const std::string weight_name = prefix + "weight";
    const std::string bias_name = prefix + "bias";

    weights_.weight = store->get_tensor(weight_name.c_str());
    weights_.bias = store->get_tensor(bias_name.c_str());
    if (!weights_.weight) {
        return false;
    }

    config_.in_dim = static_cast<int>(weights_.weight->ne[0]);
    config_.out_dim = static_cast<int>(weights_.weight->ne[1]);
    return true;
}

ggml_tensor* LinearProjection::forward(VoxCPMContext& ctx, ggml_tensor* input) {
    VOXCPM_ASSERT(input != nullptr);
    VOXCPM_ASSERT(weights_.weight != nullptr);
    VOXCPM_ASSERT(input->ne[0] == weights_.weight->ne[0]);

    ggml_context* raw = ctx.raw_context();
    ggml_tensor* output = ggml_mul_mat(raw, weights_.weight, input);
    return add_bias(raw, output, weights_.bias);
}

StopTokenPredictor::StopTokenPredictor(const StopTokenConfig& config)
    : config_(config) {
}

StopTokenPredictor::~StopTokenPredictor() {
    if (weight_buffer_) {
        ggml_backend_buffer_free(weight_buffer_);
        weight_buffer_ = nullptr;
    }
    if (weight_ctx_) {
        ggml_free(weight_ctx_);
        weight_ctx_ = nullptr;
    }
}

bool StopTokenPredictor::load_from_gguf(const std::string& gguf_path,
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

bool StopTokenPredictor::load_from_store(const std::shared_ptr<VoxCPMWeightStore>& store) {
    if (!store || !store->owns_storage()) {
        return false;
    }

    shared_store_ = store;
    weights_.stop_proj_weight = store->get_tensor("stop.stop_proj.weight");
    weights_.stop_proj_bias = store->get_tensor("stop.stop_proj.bias");
    weights_.stop_head_weight = store->get_tensor("stop.stop_head.weight");
    if (!weights_.stop_proj_weight || !weights_.stop_head_weight) {
        return false;
    }

    config_.hidden_dim = static_cast<int>(weights_.stop_proj_weight->ne[0]);
    config_.num_classes = static_cast<int>(weights_.stop_head_weight->ne[1]);
    return true;
}

ggml_tensor* StopTokenPredictor::forward(VoxCPMContext& ctx, ggml_tensor* input) {
    VOXCPM_ASSERT(input != nullptr);
    VOXCPM_ASSERT(weights_.stop_proj_weight != nullptr);
    VOXCPM_ASSERT(weights_.stop_head_weight != nullptr);
    VOXCPM_ASSERT(input->ne[0] == weights_.stop_proj_weight->ne[0]);

    ggml_context* raw = ctx.raw_context();

    ggml_tensor* hidden = ggml_mul_mat(raw, weights_.stop_proj_weight, input);
    hidden = add_bias(raw, hidden, weights_.stop_proj_bias);
    hidden = ggml_silu(raw, hidden);

    return ggml_mul_mat(raw, weights_.stop_head_weight, hidden);
}

Embedding::Embedding(const EmbeddingConfig& config)
    : config_(config) {
}

Embedding::~Embedding() {
    if (weight_buffer_) {
        ggml_backend_buffer_free(weight_buffer_);
        weight_buffer_ = nullptr;
    }
    if (weight_ctx_) {
        ggml_free(weight_ctx_);
        weight_ctx_ = nullptr;
    }
}

bool Embedding::load_from_gguf(const std::string& gguf_path,
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

bool Embedding::load_from_store(const std::shared_ptr<VoxCPMWeightStore>& store) {
    if (!store || !store->owns_storage()) {
        return false;
    }

    shared_store_ = store;
    weights_.weight = store->get_tensor("token_embd.weight");
    if (!weights_.weight) {
        return false;
    }

    config_.hidden_dim = static_cast<int>(weights_.weight->ne[0]);
    config_.vocab_size = static_cast<int>(weights_.weight->ne[1]);
    return true;
}

ggml_tensor* Embedding::forward(VoxCPMContext& ctx, ggml_tensor* token_ids) {
    VOXCPM_ASSERT(token_ids != nullptr);
    VOXCPM_ASSERT(weights_.weight != nullptr);
    VOXCPM_ASSERT(token_ids->type == GGML_TYPE_I32);

    ggml_context* raw = ctx.raw_context();
    ggml_tensor* embeddings = ggml_get_rows(raw, weights_.weight, token_ids);
    if (config_.scale != 1.0f) {
        embeddings = ggml_scale(raw, embeddings, config_.scale);
    }
    return embeddings;
}

ggml_tensor* Embedding::forward(VoxCPMContext& ctx, const std::vector<int32_t>& token_ids) {
    ggml_context* raw = ctx.raw_context();

    last_input_tensor_ = ggml_new_tensor_1d(raw, GGML_TYPE_I32, static_cast<int64_t>(token_ids.size()));
    ggml_set_input(last_input_tensor_);
    return forward(ctx, last_input_tensor_);
}

std::unique_ptr<VoxCPMComponents> VoxCPMComponents::from_gguf(const std::string& gguf_path,
                                                              int hidden_dim,
                                                              int vocab_size,
                                                              float scale_emb,
                                                              VoxCPMContext& weight_ctx,
                                                              VoxCPMContext& graph_ctx,
                                                              VoxCPMBackend& backend) {
    VOXCPM_UNUSED(weight_ctx);
    VOXCPM_UNUSED(graph_ctx);

    auto store = std::make_shared<VoxCPMWeightStore>();
    if (!store->load_from_file(gguf_path, backend)) {
        return nullptr;
    }
    return from_store(store, hidden_dim, vocab_size, scale_emb);
}

std::unique_ptr<VoxCPMComponents> VoxCPMComponents::from_store(const std::shared_ptr<VoxCPMWeightStore>& store,
                                                               int hidden_dim,
                                                               int vocab_size,
                                                               float scale_emb) {
    auto components = std::make_unique<VoxCPMComponents>();
    components->shared_store_ = store;

    auto load_projection = [&](std::unique_ptr<LinearProjection>& slot, const char* prefix) -> bool {
        auto projection = std::make_unique<LinearProjection>(ProjectionConfig{hidden_dim, hidden_dim});
        if (!projection->load_from_store(store, prefix)) {
            return false;
        }
        slot = std::move(projection);
        return true;
    };

    if (!load_projection(components->enc_to_lm_proj_, "proj.enc_to_lm")) {
        return nullptr;
    }
    if (!load_projection(components->lm_to_dit_proj_, "proj.lm_to_dit")) {
        return nullptr;
    }
    if (!load_projection(components->res_to_dit_proj_, "proj.res_to_dit")) {
        return nullptr;
    }

    auto stop_token = std::make_unique<StopTokenPredictor>(StopTokenConfig{hidden_dim, 2});
    if (!stop_token->load_from_store(store)) {
        return nullptr;
    }
    components->stop_token_ = std::move(stop_token);

    auto embedding = std::make_unique<Embedding>(EmbeddingConfig{vocab_size, hidden_dim, scale_emb});
    if (!embedding->load_from_store(store)) {
        return nullptr;
    }
    components->embed_tokens_ = std::move(embedding);

    return components;
}

}  // namespace voxcpm
