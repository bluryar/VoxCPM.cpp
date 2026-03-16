/**
 * @file fsq.h
 * @brief FSQ (Finite Scalar Quantization) Module
 *
 * Implements the FSQ component of VoxCPM using pure GGML operations.
 * Architecture is dynamically configured from GGUF metadata.
 *
 * Architecture (following PyTorch model):
 * - Input: [T, hidden_size, B] (hidden features from BaseLM)
 * - in_proj: Linear(hidden_size, latent_dim)
 * - tanh activation
 * - quantization: round(x * scale) / scale
 * - out_proj: Linear(latent_dim, hidden_size)
 * - Output: [T, hidden_size, B]
 *
 * FSQ is a simple quantization method that maps continuous values to discrete
 * levels through rounding, enabling discrete latent representations.
 */

#ifndef VOXCPM_FSQ_H
#define VOXCPM_FSQ_H

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
// FSQ Weights (matching GGUF structure)
// =============================================================================

/**
 * @brief Weights for FSQ layer
 *
 * Weight tensor shapes in GGUF:
 * - in_proj.weight: [1024, 256] (hidden_size, latent_dim)
 * - in_proj.bias: [256]
 * - out_proj.weight: [256, 1024] (latent_dim, hidden_size)
 * - out_proj.bias: [1024]
 *
 * Note: GGML stores linear weights as [in_features, out_features]
 * for efficient matmul computation (transposed from PyTorch).
 */
struct FSQWeights {
    // in_proj: Linear(hidden_size, latent_dim)
    ggml_tensor* in_proj_weight = nullptr;   // [hidden_size, latent_dim]
    ggml_tensor* in_proj_bias = nullptr;     // [latent_dim]

    // out_proj: Linear(latent_dim, hidden_size)
    ggml_tensor* out_proj_weight = nullptr;  // [latent_dim, hidden_size]
    ggml_tensor* out_proj_bias = nullptr;    // [hidden_size]

    FSQWeights() = default;
};

// =============================================================================
// FSQ Class
// =============================================================================

/**
 * @brief FSQ (Finite Scalar Quantization) Module
 *
 * Implements finite scalar quantization using GGML operations.
 *
 * Tensor layout (GGML convention: ne[0] is contiguous dimension):
 * - Input: [hidden_size, T, B] where ne[0]=hidden_size
 * - Output: [hidden_size, T, B]
 *
 * Forward pass:
 * 1. in_proj: Linear(hidden_size, latent_dim) -> [latent_dim, T, B]
 * 2. tanh activation
 * 3. quantize: round(x * scale) / scale
 * 4. out_proj: Linear(latent_dim, hidden_size) -> [hidden_size, T, B]
 */
class FSQ {
public:
    explicit FSQ(const FSQConfig& config = FSQConfig());
    ~FSQ();

    FSQ(const FSQ&) = delete;
    FSQ& operator=(const FSQ&) = delete;
    FSQ(FSQ&&) = delete;
    FSQ& operator=(FSQ&&) = delete;

    // =========================================================================
    // Weight Management
    // =========================================================================

    /**
     * @brief Load weights from GGUF file
     * @param gguf_path Path to GGUF file
     * @param weight_ctx Context for weight tensors (no_alloc=true)
     * @param graph_ctx Context for intermediate tensors (no_alloc=true)
     * @param backend Backend for buffer allocation
     * @return true on success
     */
    bool load_from_gguf(const std::string& gguf_path,
                         VoxCPMContext& weight_ctx,
                         VoxCPMContext& graph_ctx,
                         VoxCPMBackend& backend);
    bool load_from_store(const std::shared_ptr<VoxCPMWeightStore>& store);

    /**
     * @brief Get weights structure
     */
    const FSQWeights& weights() const { return weights_; }

    // =========================================================================
    // Inference
    // =========================================================================

    /**
     * @brief Forward pass through FSQ
     * @param ctx Graph context
     * @param hidden Input tensor with shape [hidden_size, T, B]
     *               (ne[0]=hidden_size, ne[1]=T, ne[2]=B)
     * @return Quantized tensor with shape [hidden_size, T, B]
     *
     * Implements:
     *   h = in_proj(hidden)        # [hidden_size, T, B] -> [latent_dim, T, B]
     *   h = tanh(h)
     *   h = round(h * scale) / scale  # Quantization
     *   h = out_proj(h)            # [latent_dim, T, B] -> [hidden_size, T, B]
     *   return h
     */
    ggml_tensor* forward(VoxCPMContext& ctx, ggml_tensor* hidden);

    // =========================================================================
    // Configuration
    // =========================================================================

    const FSQConfig& config() const { return config_; }
    int latent_dim() const { return config_.latent_dim; }
    int scale() const { return config_.scale; }
    int hidden_size() const { return config_.hidden_size; }
    const void* shared_store_token() const { return shared_store_.get(); }
    bool uses_shared_weights() const { return shared_store_ != nullptr; }

private:
    bool update_config_from_gguf(gguf_context* gguf_ctx);
    bool load_weight_data(FILE* file, gguf_context* gguf_ctx);

    // =========================================================================
    // Internal Operations
    // =========================================================================

    /**
     * @brief Apply quantization: round(x * scale) / scale
     * @param ctx GGML context
     * @param x Input tensor
     * @return Quantized tensor
     */
    ggml_tensor* quantize(ggml_context* ctx, ggml_tensor* x);

private:
    FSQConfig config_;
    FSQWeights weights_;
    ggml_context* weight_ctx_ = nullptr;
    ggml_backend_buffer_t weight_buffer_ = nullptr;  // Owns weight memory
    std::shared_ptr<VoxCPMWeightStore> shared_store_;
};

// =============================================================================
// Standalone Quantization Function
// =============================================================================

/**
 * @brief Apply FSQ quantization to a tensor
 *
 * Implements: round(x * scale) / scale
 *
 * @param ctx GGML context
 * @param x Input tensor (any shape)
 * @param scale Quantization scale (number of discrete levels: [-scale, scale])
 * @return Quantized tensor
 */
ggml_tensor* fsq_quantize(ggml_context* ctx, ggml_tensor* x, int scale);

}  // namespace voxcpm

#endif  // VOXCPM_FSQ_H
