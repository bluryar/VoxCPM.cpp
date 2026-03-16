/**
 * @file backend.h
 * @brief VoxCPM Backend Abstraction
 *
 * Encapsulates GGML backend and buffer management following best practices:
 * - no_alloc=true mode for all contexts
 * - Separate buffers for weights, KV cache, and compute
 * - Graph allocator management
 */

#ifndef VOXCPM_BACKEND_H
#define VOXCPM_BACKEND_H

#include "common.h"
#include <memory>
#include <vector>

namespace voxcpm {

/**
 * @brief Backend type enumeration
 */
enum class BackendType {
    CPU,
    CUDA,
    Metal,
    Vulkan,
    Auto,  // Auto-detect best available
};

/**
 * @brief Buffer usage type
 */
enum class BufferUsage {
    Weights,    // Read-only, persistent (model weights)
    KVCache,    // Read-write, persistent (KV cache)
    Compute,    // Read-write, dynamic (intermediate results)
};

/**
 * @brief GGML Backend Wrapper
 *
 * This class encapsulates GGML backend operations following best practices:
 * - Uses no_alloc=true for all context creation
 * - Manages separate buffers for weights, KV cache, and compute
 * - Provides graph allocator management
 * - Handles data transfer between host and device
 *
 * Thread Safety: Each instance manages its own resources independently.
 */
class VoxCPMBackend {
public:
    // =========================================================================
    // Construction / Destruction
    // =========================================================================

    /**
     * @brief Construct a backend
     * @param type Backend type (CPU, CUDA, Metal, etc.)
     * @param n_threads Number of threads for CPU backend
     */
    explicit VoxCPMBackend(BackendType type = BackendType::CPU, int n_threads = 4);

    ~VoxCPMBackend();

    // Non-copyable
    VoxCPMBackend(const VoxCPMBackend&) = delete;
    VoxCPMBackend& operator=(const VoxCPMBackend&) = delete;

    // Movable
    VoxCPMBackend(VoxCPMBackend&& other) noexcept;
    VoxCPMBackend& operator=(VoxCPMBackend&& other) noexcept;

    // =========================================================================
    // Buffer Management
    // =========================================================================

    /**
     * @brief Allocate buffer for tensors in a context
     * @param ctx Context with tensor metadata (must be no_alloc=true)
     * @param usage Buffer usage type
     * @return Buffer handle
     *
     * This implements the "two-phase model" from GGML best practices:
     * 1. Define tensor metadata in context (no_alloc=true)
     * 2. Allocate actual memory via backend
     */
    ggml_backend_buffer_t alloc_buffer(ggml_context* ctx, BufferUsage usage = BufferUsage::Weights);

    /**
     * @brief Free a buffer
     */
    void free_buffer(ggml_backend_buffer_t buffer);

    // =========================================================================
    // Graph Allocator
    // =========================================================================

    /**
     * @brief Initialize graph allocator
     */
    void init_allocator();

    /**
     * @brief Reserve compute memory for worst-case graph
     * @param graph Maximum size graph
     * @param stage Optional stage label for allocator debug logs
     *
     * Pre-allocates memory to avoid runtime allocations during inference.
     */
    void reserve_compute_memory(ggml_cgraph* graph, const char* stage = nullptr);

    /**
     * @brief Allocate memory for a compute graph
     * @param graph Compute graph
     * @param stage Optional stage label for allocator debug logs
     */
    void alloc_graph(ggml_cgraph* graph, const char* stage = nullptr);

    /**
     * @brief Check if allocator is initialized
     */
    bool has_allocator() const { return gallocr_ != nullptr; }

    // =========================================================================
    // Graph Execution
    // =========================================================================

    /**
     * @brief Execute a compute graph
     * @param graph Compute graph
     * @return Status code
     */
    ggml_status compute(ggml_cgraph* graph);

    // =========================================================================
    // Data Transfer
    // =========================================================================

    /**
     * @brief Set tensor data from host memory
     * @param tensor Destination tensor
     * @param data Source data
     * @param offset Byte offset in tensor
     * @param size Number of bytes to copy
     */
    void tensor_set(ggml_tensor* tensor, const void* data, size_t offset, size_t size);

    /**
     * @brief Set full tensor data from host memory
     * @param tensor Destination tensor
     * @param data Source data
     */
    void tensor_set(ggml_tensor* tensor, const void* data) {
        tensor_set(tensor, data, 0, ggml_nbytes(tensor));
    }

    /**
     * @brief Get tensor data to host memory
     * @param tensor Source tensor
     * @param data Destination buffer
     * @param offset Byte offset in tensor
     * @param size Number of bytes to copy
     */
    void tensor_get(const ggml_tensor* tensor, void* data, size_t offset, size_t size);

    /**
     * @brief Get full tensor data to host memory
     * @param tensor Source tensor
     * @param data Destination buffer
     */
    void tensor_get(const ggml_tensor* tensor, void* data) {
        tensor_get(tensor, data, 0, ggml_nbytes(tensor));
    }

    // =========================================================================
    // Utilities
    // =========================================================================

    /**
     * @brief Check if buffer is host-accessible
     * @param buffer Buffer to check
     * @return true if buffer can be accessed directly via pointer
     */
    bool is_host_buffer(ggml_backend_buffer_t buffer) const;

    /**
     * @brief Get number of threads
     */
    int n_threads() const { return n_threads_; }

    /**
     * @brief Get raw backend handle
     */
    ggml_backend_t raw_backend() const { return backend_; }

    /**
     * @brief Get graph allocator
     */
    ggml_gallocr_t allocator() const { return gallocr_; }

    /**
     * @brief Get buffer type for this backend
     */
    ggml_backend_buffer_type_t buffer_type() const;

    /**
     * @brief Get current compute arena size in bytes
     */
    size_t compute_buffer_size() const;

    /**
     * @brief Get backend type
     */
    BackendType type() const { return type_; }

    /**
     * @brief Check if the active backend is a GPU backend
     */
    bool is_gpu() const { return is_gpu_; }

    /**
     * @brief Get active backend name
     */
    const char* backend_name() const { return backend_name_.c_str(); }

    /**
     * @brief Get active backend description
     */
    const char* backend_description() const { return backend_description_.c_str(); }

    /**
     * @brief Check if backend is valid
     */
    bool is_valid() const { return backend_ != nullptr; }

private:
    BackendType type_;
    int n_threads_;
    ggml_backend_t backend_;
    ggml_gallocr_t gallocr_;
    bool is_gpu_ = false;
    bool allocator_logging_enabled_ = false;
    std::string backend_name_;
    std::string backend_description_;

    // Track allocated buffers for cleanup
    std::vector<ggml_backend_buffer_t> buffers_;
};

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * @brief Create a CPU backend
 * @param n_threads Number of threads
 */
inline std::unique_ptr<VoxCPMBackend> create_cpu_backend(int n_threads = 4) {
    return std::make_unique<VoxCPMBackend>(BackendType::CPU, n_threads);
}

/**
 * @brief Auto-detect and create best available backend
 * @param n_threads Number of threads for CPU fallback
 */
std::unique_ptr<VoxCPMBackend> create_best_backend(int n_threads = 4);

}  // namespace voxcpm

#endif  // VOXCPM_BACKEND_H
