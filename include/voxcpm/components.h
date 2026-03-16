/**
 * @file components.h
 * @brief VoxCPM Auxiliary Components
 *
 * Implements projection layers, stop token prediction, and embedding layer
 * using pure GGML operations.
 *
 * Components:
 * - LinearProjection: Simple linear layer with optional bias
 *   - enc_to_lm_proj: LocEnc output -> LM input
 *   - lm_to_dit_proj: LM output -> DiT input
 *   - res_to_dit_proj: ResidualLM output -> DiT input
 * - StopTokenPredictor: Binary classification for stop/continue
 * - Embedding: Token embedding lookup with optional scale
 *
 * Reference: examples/voxcpm_components_ggml.py
 */

#ifndef VOXCPM_COMPONENTS_H
#define VOXCPM_COMPONENTS_H

#include "common.h"
#include "config.h"
#include "context.h"
#include <memory>
#include <string>
#include <vector>

namespace voxcpm {

// Forward declaration
class VoxCPMBackend;
class VoxCPMWeightStore;

// =============================================================================
// Linear Projection Layer
// =============================================================================

/**
 * @brief Weights for Linear Projection
 *
 * GGUF tensor names:
 * - proj.enc_to_lm.weight / proj.enc_to_lm.bias
 * - proj.lm_to_dit.weight / proj.lm_to_dit.bias
 * - proj.res_to_dit.weight / proj.res_to_dit.bias
 *
 * Weight shape in GGML: [in_dim, out_dim]
 */
struct ProjectionWeights {
    ggml_tensor* weight = nullptr;  // [in_dim, out_dim]
    ggml_tensor* bias = nullptr;    // [out_dim], optional

    ProjectionWeights() = default;
};

/**
 * @brief Linear Projection Layer
 *
 * Implements: output = input @ weight + bias
 *
 * Tensor layout (GGML convention: ne[0] is contiguous dimension):
 * - Input: [in_dim, T, B] or [in_dim, B]
 * - Output: [out_dim, T, B] or [out_dim, B]
 */
class LinearProjection {
public:
    explicit LinearProjection(const ProjectionConfig& config = ProjectionConfig());
    ~LinearProjection();

    LinearProjection(const LinearProjection&) = delete;
    LinearProjection& operator=(const LinearProjection&) = delete;
    LinearProjection(LinearProjection&&) = delete;
    LinearProjection& operator=(LinearProjection&&) = delete;

    // =========================================================================
    // Weight Management
    // =========================================================================

    /**
     * @brief Load weights from GGUF file
     * @param gguf_path Path to GGUF file
     * @param weight_prefix GGUF tensor prefix (e.g., "proj.enc_to_lm")
     * @param weight_ctx Context for weight tensors
     * @param graph_ctx Context for intermediate tensors
     * @param backend Backend for buffer allocation
     * @return true on success
     */
    bool load_from_gguf(const std::string& gguf_path,
                         const std::string& weight_prefix,
                         VoxCPMContext& weight_ctx,
                         VoxCPMContext& graph_ctx,
                         VoxCPMBackend& backend);
    bool load_from_store(const std::shared_ptr<VoxCPMWeightStore>& store,
                         const std::string& weight_prefix);

    const ProjectionWeights& weights() const { return weights_; }

    // =========================================================================
    // Inference
    // =========================================================================

    /**
     * @brief Forward pass
     * @param ctx Graph context
     * @param input Input tensor [in_dim, ...]
     * @return Output tensor [out_dim, ...]
     */
    ggml_tensor* forward(VoxCPMContext& ctx, ggml_tensor* input);

    // =========================================================================
    // Configuration
    // =========================================================================

    const ProjectionConfig& config() const { return config_; }
    const void* shared_store_token() const { return shared_store_.get(); }
    bool uses_shared_weights() const { return shared_store_ != nullptr; }

private:
    ProjectionConfig config_;
    ProjectionWeights weights_;
    ggml_context* weight_ctx_ = nullptr;
    ggml_backend_buffer_t weight_buffer_ = nullptr;
    std::shared_ptr<VoxCPMWeightStore> shared_store_;
};

// =============================================================================
// Stop Token Predictor
// =============================================================================

/**
 * @brief Weights for Stop Token Prediction
 *
 * GGUF tensor names:
 * - stop.stop_proj.weight / stop.stop_proj.bias
 * - stop.stop_head.weight (NO bias!)
 *
 * Weight shapes in GGML:
 * - stop_proj: [hidden_dim, hidden_dim]
 * - stop_head: [hidden_dim, num_classes]
 */
struct StopTokenWeights {
    ggml_tensor* stop_proj_weight = nullptr;  // [hidden_dim, hidden_dim]
    ggml_tensor* stop_proj_bias = nullptr;    // [hidden_dim], optional
    ggml_tensor* stop_head_weight = nullptr;  // [hidden_dim, num_classes]
    // Note: stop_head has NO bias!

    StopTokenWeights() = default;
};

/**
 * @brief Stop Token Prediction Module
 *
 * Predicts when generation should stop (binary classification).
 *
 * Architecture:
 *   h = silu(input @ stop_proj + bias)
 *   logits = h @ stop_head  (no bias!)
 *
 * Tensor layout:
 * - Input: [hidden_dim, B]
 * - Output: [num_classes, B] = [2, B]
 */
class StopTokenPredictor {
public:
    explicit StopTokenPredictor(const StopTokenConfig& config = StopTokenConfig());
    ~StopTokenPredictor();

    StopTokenPredictor(const StopTokenPredictor&) = delete;
    StopTokenPredictor& operator=(const StopTokenPredictor&) = delete;
    StopTokenPredictor(StopTokenPredictor&&) = delete;
    StopTokenPredictor& operator=(StopTokenPredictor&&) = delete;

    // =========================================================================
    // Weight Management
    // =========================================================================

    /**
     * @brief Load weights from GGUF file
     * @param gguf_path Path to GGUF file
     * @param weight_ctx Context for weight tensors
     * @param graph_ctx Context for intermediate tensors
     * @param backend Backend for buffer allocation
     * @return true on success
     */
    bool load_from_gguf(const std::string& gguf_path,
                         VoxCPMContext& weight_ctx,
                         VoxCPMContext& graph_ctx,
                         VoxCPMBackend& backend);
    bool load_from_store(const std::shared_ptr<VoxCPMWeightStore>& store);

    const StopTokenWeights& weights() const { return weights_; }

    // =========================================================================
    // Inference
    // =========================================================================

    /**
     * @brief Forward pass
     * @param ctx Graph context
     * @param input Input tensor [hidden_dim, B]
     * @return Logits tensor [num_classes, B]
     */
    ggml_tensor* forward(VoxCPMContext& ctx, ggml_tensor* input);

    // =========================================================================
    // Configuration
    // =========================================================================

    const StopTokenConfig& config() const { return config_; }
    const void* shared_store_token() const { return shared_store_.get(); }
    bool uses_shared_weights() const { return shared_store_ != nullptr; }

private:
    StopTokenConfig config_;
    StopTokenWeights weights_;
    ggml_context* weight_ctx_ = nullptr;
    ggml_backend_buffer_t weight_buffer_ = nullptr;
    std::shared_ptr<VoxCPMWeightStore> shared_store_;
};

// =============================================================================
// Embedding Layer
// =============================================================================

/**
 * @brief Weights for Token Embedding
 *
 * GGUF tensor name: token_embd.weight
 *
 * Weight shape in GGML: [hidden_dim, vocab_size]
 * (Transposed from PyTorch [vocab_size, hidden_dim])
 */
struct EmbeddingWeights {
    ggml_tensor* weight = nullptr;  // [hidden_dim, vocab_size]

    EmbeddingWeights() = default;
};

/**
 * @brief Token Embedding Layer
 *
 * Looks up token embeddings from a vocabulary table.
 *
 * Forward:
 *   embeddings = weight[token_ids] * scale
 *
 * Tensor layout:
 * - token_ids: I32 tensor [seq_len] or [seq_len, batch]
 * - Output: [hidden_dim, seq_len] or [hidden_dim, seq_len, batch]
 *
 * Note: scale (scale_emb) is applied when use_mup=true (MiniCPM specific).
 */
class Embedding {
public:
    explicit Embedding(const EmbeddingConfig& config = EmbeddingConfig());
    ~Embedding();

    Embedding(const Embedding&) = delete;
    Embedding& operator=(const Embedding&) = delete;
    Embedding(Embedding&&) = delete;
    Embedding& operator=(Embedding&&) = delete;

    // =========================================================================
    // Weight Management
    // =========================================================================

    /**
     * @brief Load weights from GGUF file
     * @param gguf_path Path to GGUF file
     * @param weight_ctx Context for weight tensors
     * @param graph_ctx Context for intermediate tensors
     * @param backend Backend for buffer allocation
     * @return true on success
     */
    bool load_from_gguf(const std::string& gguf_path,
                         VoxCPMContext& weight_ctx,
                         VoxCPMContext& graph_ctx,
                         VoxCPMBackend& backend);
    bool load_from_store(const std::shared_ptr<VoxCPMWeightStore>& store);

    const EmbeddingWeights& weights() const { return weights_; }

    // =========================================================================
    // Inference
    // =========================================================================

    /**
     * @brief Forward pass with token ID tensor
     * @param ctx Graph context
     * @param token_ids Token ID tensor (I32 type) [seq_len] or [seq_len, batch]
     * @return Embeddings tensor [hidden_dim, seq_len] or [hidden_dim, seq_len, batch]
     */
    ggml_tensor* forward(VoxCPMContext& ctx, ggml_tensor* token_ids);

    /**
     * @brief Forward pass with token ID vector (convenience method)
     * @param ctx Graph context
     * @param token_ids Vector of token IDs
     * @return Embeddings tensor [hidden_dim, seq_len]
     */
    ggml_tensor* forward(VoxCPMContext& ctx, const std::vector<int32_t>& token_ids);

    // =========================================================================
    // Configuration
    // =========================================================================

    const EmbeddingConfig& config() const { return config_; }

    /**
     * @brief Get the input tensor created by forward(vector) method
     * @return Token IDs tensor, or nullptr if not created
     */
    ggml_tensor* last_input_tensor() const { return last_input_tensor_; }
    const void* shared_store_token() const { return shared_store_.get(); }
    bool uses_shared_weights() const { return shared_store_ != nullptr; }

private:
    EmbeddingConfig config_;
    EmbeddingWeights weights_;
    ggml_context* weight_ctx_ = nullptr;
    ggml_backend_buffer_t weight_buffer_ = nullptr;
    std::shared_ptr<VoxCPMWeightStore> shared_store_;

    // For vector input convenience method
    ggml_tensor* last_input_tensor_ = nullptr;
};

// =============================================================================
// VoxCPM Components Container
// =============================================================================

/**
 * @brief Container for all VoxCPM auxiliary components
 *
 * Groups all simple components together for easy access and loading.
 */
class VoxCPMComponents {
public:
    VoxCPMComponents() = default;
    ~VoxCPMComponents() = default;

    // Non-copyable, movable
    VoxCPMComponents(const VoxCPMComponents&) = delete;
    VoxCPMComponents& operator=(const VoxCPMComponents&) = delete;
    VoxCPMComponents(VoxCPMComponents&&) = default;
    VoxCPMComponents& operator=(VoxCPMComponents&&) = default;

    // =========================================================================
    // Component Access
    // =========================================================================

    LinearProjection* enc_to_lm_proj() { return enc_to_lm_proj_.get(); }
    LinearProjection* lm_to_dit_proj() { return lm_to_dit_proj_.get(); }
    LinearProjection* res_to_dit_proj() { return res_to_dit_proj_.get(); }
    StopTokenPredictor* stop_token() { return stop_token_.get(); }
    Embedding* embed_tokens() { return embed_tokens_.get(); }
    const LinearProjection* enc_to_lm_proj() const { return enc_to_lm_proj_.get(); }
    const LinearProjection* lm_to_dit_proj() const { return lm_to_dit_proj_.get(); }
    const LinearProjection* res_to_dit_proj() const { return res_to_dit_proj_.get(); }
    const StopTokenPredictor* stop_token() const { return stop_token_.get(); }
    const Embedding* embed_tokens() const { return embed_tokens_.get(); }

    // =========================================================================
    // Factory Method
    // =========================================================================

    /**
     * @brief Load all components from GGUF file
     * @param gguf_path Path to GGUF file
     * @param hidden_dim Hidden dimension (default 1024)
     * @param vocab_size Vocabulary size (default 73448)
     * @param scale_emb Embedding scale factor (default 12 for MiniCPM)
     * @param weight_ctx Context for weight tensors
     * @param graph_ctx Context for intermediate tensors
     * @param backend Backend for buffer allocation
     * @return Initialized components, or nullptr on failure
     */
    static std::unique_ptr<VoxCPMComponents> from_gguf(
        const std::string& gguf_path,
        int hidden_dim,
        int vocab_size,
        float scale_emb,
        VoxCPMContext& weight_ctx,
        VoxCPMContext& graph_ctx,
        VoxCPMBackend& backend);
    static std::unique_ptr<VoxCPMComponents> from_store(
        const std::shared_ptr<VoxCPMWeightStore>& store,
        int hidden_dim,
        int vocab_size,
        float scale_emb);

    const void* shared_store_token() const { return shared_store_.get(); }
    bool uses_shared_weights() const { return shared_store_ != nullptr; }

private:
    std::unique_ptr<LinearProjection> enc_to_lm_proj_;
    std::unique_ptr<LinearProjection> lm_to_dit_proj_;
    std::unique_ptr<LinearProjection> res_to_dit_proj_;
    std::unique_ptr<StopTokenPredictor> stop_token_;
    std::unique_ptr<Embedding> embed_tokens_;
    std::shared_ptr<VoxCPMWeightStore> shared_store_;
};

}  // namespace voxcpm

#endif  // VOXCPM_COMPONENTS_H
