/**
 * @file context.h
 * @brief VoxCPM Context Management
 *
 * Encapsulates GGML context creation and management following best practices:
 * - Uses no_alloc=true for all contexts
 * - Context stores only tensor metadata, not data
 * - Provides tensor creation and lookup utilities
 */

#ifndef VOXCPM_CONTEXT_H
#define VOXCPM_CONTEXT_H

#include "common.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace voxcpm {

/**
 * @brief Context type enumeration
 */
enum class ContextType {
    Weights,    // Model weights (persistent)
    KVCache,    // KV cache (persistent)
    Graph,      // Compute graph (temporary)
};

/**
 * @brief GGML Context Wrapper
 *
 * This class encapsulates GGML context operations following best practices:
 * - Uses no_alloc=true: context stores only metadata
 * - Memory size calculation includes only overhead, not tensor data
 * - Tensor data is allocated separately via Backend
 *
 * Thread Safety: Each instance manages its own memory pool independently.
 */
class VoxCPMContext {
public:
    // =========================================================================
    // Construction / Destruction
    // =========================================================================

    /**
     * @brief Construct a context
     * @param type Context type
     * @param n_tensors Expected number of tensors
     * @param max_nodes Maximum graph nodes (for Graph type)
     */
    VoxCPMContext(ContextType type, int n_tensors, int max_nodes = 0);

    ~VoxCPMContext();

    // Non-copyable
    VoxCPMContext(const VoxCPMContext&) = delete;
    VoxCPMContext& operator=(const VoxCPMContext&) = delete;

    // Movable
    VoxCPMContext(VoxCPMContext&& other) noexcept;
    VoxCPMContext& operator=(VoxCPMContext&& other) noexcept;

    // =========================================================================
    // Tensor Creation
    // =========================================================================

    /**
     * @brief Create a 1D tensor
     */
    ggml_tensor* new_tensor_1d(ggml_type type, int64_t ne0);

    /**
     * @brief Create a 2D tensor
     */
    ggml_tensor* new_tensor_2d(ggml_type type, int64_t ne0, int64_t ne1);

    /**
     * @brief Create a 3D tensor
     */
    ggml_tensor* new_tensor_3d(ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2);

    /**
     * @brief Create a 4D tensor
     */
    ggml_tensor* new_tensor_4d(ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3);

    /**
     * @brief Create a new tensor with given dimensions
     */
    ggml_tensor* new_tensor(ggml_type type, int n_dims, const int64_t* ne);

    // =========================================================================
    // Tensor Lookup
    // =========================================================================

    /**
     * @brief Get tensor by name
     * @param name Tensor name
     * @return Tensor pointer or nullptr if not found
     */
    ggml_tensor* get_tensor(const std::string& name);

    /**
     * @brief Get tensor by name (const version)
     */
    const ggml_tensor* get_tensor(const std::string& name) const;

    /**
     * @brief Check if tensor exists
     */
    bool has_tensor(const std::string& name) const;

    /**
     * @brief Register tensor with name
     */
    void register_tensor(ggml_tensor* tensor, const std::string& name);

    // =========================================================================
    // Compute Graph
    // =========================================================================

    /**
     * @brief Create a new compute graph
     *
     * Uses the max_nodes_ value passed at construction time, or GGML_DEFAULT_GRAPH_SIZE
     */
    ggml_cgraph* new_graph();

    /**
     * @brief Create a new compute graph with custom size
     * @param size Maximum number of nodes in the graph
     */
    ggml_cgraph* new_graph(size_t size);

    /**
     * @brief Build forward computation graph
     * @param graph Graph to build
     * @param output Output tensor
     */
    void build_forward(ggml_cgraph* graph, ggml_tensor* output);

    // =========================================================================
    // Memory Size Calculation
    // =========================================================================

    /**
     * @brief Calculate memory size for weights context
     * @param n_layer_base BaseLM layers
     * @param n_layer_res ResidualLM layers
     * @param n_layer_enc LocEnc layers
     * @param n_layer_dit LocDiT layers
     */
    static size_t calc_weights_ctx_size(
        int n_layer_base, int n_layer_res,
        int n_layer_enc, int n_layer_dit);

    /**
     * @brief Calculate memory size for KV cache context
     * @param n_layer Number of layers
     */
    static size_t calc_kv_ctx_size(int n_layer);

    /**
     * @brief Calculate memory size for graph context
     * @param max_nodes Maximum number of graph nodes
     */
    static size_t calc_graph_ctx_size(int max_nodes);

    // =========================================================================
    // Utilities
    // =========================================================================

    /**
     * @brief Get raw context handle
     */
    ggml_context* raw_context() const { return ctx_; }

    /**
     * @brief Get context type
     */
    ContextType type() const { return type_; }

    /**
     * @brief Get memory size
     */
    size_t mem_size() const { return mem_size_; }

    /**
     * @brief Get maximum graph nodes
     */
    size_t max_nodes() const { return max_nodes_; }

    /**
     * @brief Get number of tensors created
     */
    int tensor_count() const;

    /**
     * @brief Check if context is valid
     */
    bool is_valid() const { return ctx_ != nullptr; }

    // =========================================================================
    // Iteration
    // =========================================================================

    /**
     * @brief Get first tensor in context
     */
    ggml_tensor* get_first_tensor();

    /**
     * @brief Get next tensor in context
     */
    ggml_tensor* get_next_tensor(ggml_tensor* tensor);

private:
    ContextType type_;
    ggml_context* ctx_;
    size_t mem_size_;
    size_t max_nodes_;  // Maximum graph nodes for graph contexts

    // Graph context memory pool (avoids frequent malloc/free)
    std::vector<uint8_t> graph_buffer_;

    // Tensor name lookup
    std::unordered_map<std::string, ggml_tensor*> tensor_map_;
};

// =============================================================================
// Graph Context Builder
// =============================================================================

/**
 * @brief Helper class for building compute graphs
 *
 * Provides a simple interface for building graphs with proper
 * input/output marking following GGML best practices.
 */
class GraphBuilder {
public:
    explicit GraphBuilder(VoxCPMContext& ctx);

    /**
     * @brief Create input tensor
     */
    ggml_tensor* create_input(const std::string& name, ggml_type type, int n_dims, const int64_t* ne);

    /**
     * @brief Create input tensor (1D)
     */
    ggml_tensor* create_input_1d(const std::string& name, ggml_type type, int64_t ne0);

    /**
     * @brief Create input tensor (2D)
     */
    ggml_tensor* create_input_2d(const std::string& name, ggml_type type, int64_t ne0, int64_t ne1);

    /**
     * @brief Create input tensor (3D)
     */
    ggml_tensor* create_input_3d(const std::string& name, ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2);

    /**
     * @brief Mark tensor as output
     */
    void mark_output(ggml_tensor* tensor);

    /**
     * @brief Build and return the graph
     */
    ggml_cgraph* build();

    /**
     * @brief Get the graph
     */
    ggml_cgraph* graph() const { return graph_; }

private:
    VoxCPMContext& ctx_;
    ggml_cgraph* graph_;
    ggml_tensor* last_output_;
};

}  // namespace voxcpm

#endif  // VOXCPM_CONTEXT_H
