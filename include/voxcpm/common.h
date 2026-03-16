/**
 * @file common.h
 * @brief VoxCPM Common Definitions
 *
 * Common macros, types, and utilities for VoxCPM GGML implementation.
 */

#ifndef VOXCPM_COMMON_H
#define VOXCPM_COMMON_H

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include <stdexcept>

// GGML headers
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

namespace voxcpm {

// =============================================================================
// Version
// =============================================================================

constexpr int VOXCPM_VERSION_MAJOR = 0;
constexpr int VOXCPM_VERSION_MINOR = 1;
constexpr int VOXCPM_VERSION_PATCH = 0;

// =============================================================================
// Macros
// =============================================================================

#define VOXCPM_UNUSED(x) (void)(x)

#define VOXCPM_ASSERT(x) \
    do { \
        if (!(x)) { \
            fprintf(stderr, "VOXCPM_ASSERT: %s:%d: %s\n", __FILE__, __LINE__, #x); \
            abort(); \
        } \
    } while (0)

#define VOXCPM_ABORT(msg) \
    do { \
        fprintf(stderr, "VOXCPM_ABORT: %s:%d: %s\n", __FILE__, __LINE__, msg); \
        abort(); \
    } while (0)

// =============================================================================
// Error Handling
// =============================================================================

enum class ErrorCode {
    Success = 0,
    InvalidArgument,
    OutOfMemory,
    FileNotFound,
    InvalidFormat,
    BackendError,
    NotImplemented,
};

class Error : public std::runtime_error {
public:
    explicit Error(ErrorCode code, const std::string& message)
        : std::runtime_error(message), code_(code) {}

    ErrorCode code() const { return code_; }

private:
    ErrorCode code_;
};

// =============================================================================
// Memory Utilities
// =============================================================================

/**
 * @brief Calculate context memory size for given tensor count
 *
 * Following GGML best practice: context only stores metadata, not tensor data.
 * Size = n_tensors * ggml_tensor_overhead() + ggml_graph_overhead() + margin
 */
inline size_t calc_context_size(int n_tensors, int max_nodes = 0) {
    size_t size = 0;
    size += n_tensors * ggml_tensor_overhead();
    if (max_nodes > 0) {
        size += ggml_graph_overhead_custom(max_nodes, false);
    } else {
        // Default to GGML_DEFAULT_GRAPH_SIZE for graph contexts
        size += ggml_graph_overhead();  // Uses GGML_DEFAULT_GRAPH_SIZE
    }
    size += 1024;  // Safety margin
    return size;
}

/**
 * @brief Align size to GGML alignment
 */
inline size_t align_size(size_t size) {
    const size_t alignment = 64;  // GGML_MEM_ALIGN
    return ((size + alignment - 1) / alignment) * alignment;
}

// =============================================================================
// Tensor Utilities
// =============================================================================

/**
 * @brief Get element count of a tensor
 */
inline int64_t tensor_nelements(const ggml_tensor* tensor) {
    if (!tensor) return 0;
    return ggml_nelements(tensor);
}

/**
 * @brief Get byte size of a tensor
 */
inline size_t tensor_nbytes(const ggml_tensor* tensor) {
    if (!tensor) return 0;
    return ggml_nbytes(tensor);
}

/**
 * @brief Get tensor shape as string
 */
inline std::string tensor_shape_str(const ggml_tensor* tensor) {
    if (!tensor) return "null";
    char buf[256];
    snprintf(buf, sizeof(buf), "[%ld, %ld, %ld, %ld]",
             (long)tensor->ne[0], (long)tensor->ne[1],
             (long)tensor->ne[2], (long)tensor->ne[3]);
    return std::string(buf);
}

// =============================================================================
// GGML Best Practice Helpers
// =============================================================================

/**
 * @brief Mark tensor as input (required for scheduler)
 */
inline void mark_input(ggml_tensor* tensor) {
    ggml_set_input(tensor);
}

/**
 * @brief Mark tensor as output (required for allocator)
 */
inline void mark_output(ggml_tensor* tensor) {
    ggml_set_output(tensor);
}

/**
 * @brief Set tensor name and optionally mark as input
 */
inline void set_tensor_name(ggml_tensor* tensor, const char* name, bool is_input = false) {
    ggml_set_name(tensor, name);
    if (is_input) {
        ggml_set_input(tensor);
    }
}

// =============================================================================
// Deleters for smart pointers
// =============================================================================

struct GGMLContextDeleter {
    void operator()(ggml_context* ctx) const {
        if (ctx) ggml_free(ctx);
    }
};

struct GGMLBackendDeleter {
    void operator()(ggml_backend_t backend) const {
        if (backend) ggml_backend_free(backend);
    }
};

struct GGMLBufferDeleter {
    void operator()(ggml_backend_buffer_t buffer) const {
        if (buffer) ggml_backend_buffer_free(buffer);
    }
};

struct GGMLGallocrDeleter {
    void operator()(ggml_gallocr_t gallocr) const {
        if (gallocr) ggml_gallocr_free(gallocr);
    }
};

struct GGUFContextDeleter {
    void operator()(gguf_context* ctx) const {
        if (ctx) gguf_free(ctx);
    }
};

// Smart pointer aliases
using UniqueContext = std::unique_ptr<ggml_context, GGMLContextDeleter>;
using UniqueBackend = std::unique_ptr<ggml_backend, GGMLBackendDeleter>;
using UniqueBuffer = std::unique_ptr<ggml_backend_buffer, GGMLBufferDeleter>;
using UniqueGallocr = std::unique_ptr<ggml_gallocr, GGMLGallocrDeleter>;
using UniqueGGUFContext = std::unique_ptr<gguf_context, GGUFContextDeleter>;

}  // namespace voxcpm

#endif  // VOXCPM_COMMON_H
