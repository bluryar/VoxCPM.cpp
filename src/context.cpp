/**
 * @file context.cpp
 * @brief VoxCPM Context Implementation
 */

#include "voxcpm/context.h"
#include <cstring>

namespace voxcpm {

// =============================================================================
// Construction / Destruction
// =============================================================================

VoxCPMContext::VoxCPMContext(ContextType type, int n_tensors, int max_nodes)
    : type_(type), ctx_(nullptr), mem_size_(0), max_nodes_(max_nodes) {

    // Calculate memory size (metadata only, no tensor data)
    mem_size_ = calc_context_size(n_tensors, max_nodes);

    // Allocate graph buffer if needed
    if (type == ContextType::Graph) {
        graph_buffer_.resize(mem_size_);
    }

    // Initialize context with no_alloc=true
    struct ggml_init_params params = {
        .mem_size = mem_size_,
        .mem_buffer = (type == ContextType::Graph) ? graph_buffer_.data() : nullptr,
        .no_alloc = true,  // Critical: only store metadata
    };

    ctx_ = ggml_init(params);
    if (!ctx_) {
        throw Error(ErrorCode::OutOfMemory, "Failed to create context");
    }
}

VoxCPMContext::~VoxCPMContext() {
    if (ctx_) {
        ggml_free(ctx_);
    }
}

VoxCPMContext::VoxCPMContext(VoxCPMContext&& other) noexcept
    : type_(other.type_),
      ctx_(other.ctx_),
      mem_size_(other.mem_size_),
      max_nodes_(other.max_nodes_),
      graph_buffer_(std::move(other.graph_buffer_)),
      tensor_map_(std::move(other.tensor_map_)) {
    other.ctx_ = nullptr;
    other.mem_size_ = 0;
    other.max_nodes_ = 0;
}

VoxCPMContext& VoxCPMContext::operator=(VoxCPMContext&& other) noexcept {
    if (this != &other) {
        if (ctx_) {
            ggml_free(ctx_);
        }

        type_ = other.type_;
        ctx_ = other.ctx_;
        mem_size_ = other.mem_size_;
        max_nodes_ = other.max_nodes_;
        graph_buffer_ = std::move(other.graph_buffer_);
        tensor_map_ = std::move(other.tensor_map_);

        other.ctx_ = nullptr;
        other.mem_size_ = 0;
        other.max_nodes_ = 0;
    }
    return *this;
}

// =============================================================================
// Tensor Creation
// =============================================================================

ggml_tensor* VoxCPMContext::new_tensor_1d(ggml_type type, int64_t ne0) {
    return ggml_new_tensor_1d(ctx_, type, ne0);
}

ggml_tensor* VoxCPMContext::new_tensor_2d(ggml_type type, int64_t ne0, int64_t ne1) {
    return ggml_new_tensor_2d(ctx_, type, ne0, ne1);
}

ggml_tensor* VoxCPMContext::new_tensor_3d(ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2) {
    return ggml_new_tensor_3d(ctx_, type, ne0, ne1, ne2);
}

ggml_tensor* VoxCPMContext::new_tensor_4d(ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    return ggml_new_tensor_4d(ctx_, type, ne0, ne1, ne2, ne3);
}

ggml_tensor* VoxCPMContext::new_tensor(ggml_type type, int n_dims, const int64_t* ne) {
    return ggml_new_tensor(ctx_, type, n_dims, ne);
}

// =============================================================================
// Tensor Lookup
// =============================================================================

ggml_tensor* VoxCPMContext::get_tensor(const std::string& name) {
    auto it = tensor_map_.find(name);
    if (it != tensor_map_.end()) {
        return it->second;
    }
    return nullptr;
}

const ggml_tensor* VoxCPMContext::get_tensor(const std::string& name) const {
    auto it = tensor_map_.find(name);
    if (it != tensor_map_.end()) {
        return it->second;
    }
    return nullptr;
}

bool VoxCPMContext::has_tensor(const std::string& name) const {
    return tensor_map_.find(name) != tensor_map_.end();
}

void VoxCPMContext::register_tensor(ggml_tensor* tensor, const std::string& name) {
    if (tensor) {
        ggml_set_name(tensor, name.c_str());
        tensor_map_[name] = tensor;
    }
}

// =============================================================================
// Compute Graph
// =============================================================================

ggml_cgraph* VoxCPMContext::new_graph() {
    // Use stored max_nodes_ or fall back to GGML_DEFAULT_GRAPH_SIZE
    size_t size = max_nodes_ > 0 ? max_nodes_ : GGML_DEFAULT_GRAPH_SIZE;
    return ggml_new_graph_custom(ctx_, size, false);
}

ggml_cgraph* VoxCPMContext::new_graph(size_t size) {
    return ggml_new_graph_custom(ctx_, size, false);
}

void VoxCPMContext::build_forward(ggml_cgraph* graph, ggml_tensor* output) {
    ggml_build_forward_expand(graph, output);
}

// =============================================================================
// Memory Size Calculation
// =============================================================================

size_t VoxCPMContext::calc_weights_ctx_size(
    int n_layer_base, int n_layer_res, int n_layer_enc, int n_layer_dit) {

    int n_tensors = 0;

    // Token embedding + output norm: 2
    n_tensors += 2;

    // BaseLM: 24 layers * 9 tensors/layer
    n_tensors += n_layer_base * 9;

    // ResidualLM: 8 layers * 9 + output_norm
    n_tensors += n_layer_res * 9 + 1;

    // LocEnc: layers + in_proj weight/bias + special_token + output_norm
    n_tensors += n_layer_enc * 9 + 4;

    // LocDiT: 8 layers * 9 + 5 projections + 4 time embeddings + output_norm
    n_tensors += n_layer_dit * 9 + 10;

    // FSQ: 4 tensors
    n_tensors += 4;

    // Projections: 6 tensors
    n_tensors += 6;

    // Stop predictor: 3 tensors
    n_tensors += 3;

    // AudioVAE: legacy rough estimate only.
    // Real tensor count is model-dependent and much higher for current VoxCPM exports,
    // so keep this intentionally conservative to avoid underestimating context size.
    n_tensors += 256;

    return calc_context_size(n_tensors, 0);
}

size_t VoxCPMContext::calc_kv_ctx_size(int n_layer) {
    // Each layer has K and V tensors
    int n_tensors = n_layer * 2;
    return calc_context_size(n_tensors, 0);
}

size_t VoxCPMContext::calc_graph_ctx_size(int max_nodes) {
    // Graph context needs space for tensor metadata + graph structure
    int n_tensors = max_nodes * 4;  // Rough estimate
    return calc_context_size(n_tensors, max_nodes);
}

// =============================================================================
// Utilities
// =============================================================================

int VoxCPMContext::tensor_count() const {
    if (!ctx_) return 0;

    int count = 0;
    for (ggml_tensor* t = ggml_get_first_tensor(ctx_); t != nullptr;
         t = ggml_get_next_tensor(ctx_, t)) {
        count++;
    }
    return count;
}

// =============================================================================
// Iteration
// =============================================================================

ggml_tensor* VoxCPMContext::get_first_tensor() {
    return ctx_ ? ggml_get_first_tensor(ctx_) : nullptr;
}

ggml_tensor* VoxCPMContext::get_next_tensor(ggml_tensor* tensor) {
    return ctx_ ? ggml_get_next_tensor(ctx_, tensor) : nullptr;
}

// =============================================================================
// GraphBuilder Implementation
// =============================================================================

GraphBuilder::GraphBuilder(VoxCPMContext& ctx)
    : ctx_(ctx), graph_(nullptr), last_output_(nullptr) {
    graph_ = ctx_.new_graph();
}

ggml_tensor* GraphBuilder::create_input(const std::string& name, ggml_type type, int n_dims, const int64_t* ne) {
    ggml_tensor* t = ctx_.new_tensor(type, n_dims, ne);
    if (t) {
        ggml_set_input(t);
        ctx_.register_tensor(t, name);
    }
    return t;
}

ggml_tensor* GraphBuilder::create_input_1d(const std::string& name, ggml_type type, int64_t ne0) {
    ggml_tensor* t = ctx_.new_tensor_1d(type, ne0);
    if (t) {
        ggml_set_input(t);
        ctx_.register_tensor(t, name);
    }
    return t;
}

ggml_tensor* GraphBuilder::create_input_2d(const std::string& name, ggml_type type, int64_t ne0, int64_t ne1) {
    ggml_tensor* t = ctx_.new_tensor_2d(type, ne0, ne1);
    if (t) {
        ggml_set_input(t);
        ctx_.register_tensor(t, name);
    }
    return t;
}

ggml_tensor* GraphBuilder::create_input_3d(const std::string& name, ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2) {
    ggml_tensor* t = ctx_.new_tensor_3d(type, ne0, ne1, ne2);
    if (t) {
        ggml_set_input(t);
        ctx_.register_tensor(t, name);
    }
    return t;
}

void GraphBuilder::mark_output(ggml_tensor* tensor) {
    if (tensor) {
        ggml_set_output(tensor);
        last_output_ = tensor;
    }
}

ggml_cgraph* GraphBuilder::build() {
    if (last_output_) {
        ctx_.build_forward(graph_, last_output_);
    }
    return graph_;
}

}  // namespace voxcpm
