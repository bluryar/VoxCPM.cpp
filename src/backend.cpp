/**
 * @file backend.cpp
 * @brief VoxCPM Backend Implementation
 */

#include "voxcpm/backend.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <utility>

namespace voxcpm {

namespace {

struct BackendInitResult {
    ggml_backend_t backend = nullptr;
    BackendType type = BackendType::CPU;
    bool is_gpu = false;
    std::string name;
    std::string description;
};

ggml_backend_t init_aux_cpu_backend_handle(int n_threads) {
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        throw Error(ErrorCode::BackendError, "Failed to initialize auxiliary CPU backend");
    }
    ggml_backend_cpu_set_n_threads(backend, n_threads);
    return backend;
}

std::string to_lower_copy(const char* value) {
    std::string result = value ? value : "";
    std::transform(result.begin(), result.end(), result.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return result;
}

bool env_flag_enabled(const char* name) {
    const char* raw = std::getenv(name);
    if (!raw || raw[0] == '\0') {
        return false;
    }

    const std::string value = to_lower_copy(raw);
    return value == "1" || value == "true" || value == "yes" || value == "on";
}

bool stage_should_use_scheduler(const char* stage) {
    if (!stage) {
        return false;
    }

    const std::string value = to_lower_copy(stage);
    return value == "runtime.base_lm.decode_step.state_cached" ||
           value == "runtime.residual_lm.decode_step.state_cached" ||
           value == "runtime.base_lm.decode_step";
}

std::string format_mib(size_t bytes) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2)
        << (static_cast<double>(bytes) / (1024.0 * 1024.0)) << " MiB";
    return oss.str();
}

bool is_vulkan_device(ggml_backend_dev_t dev) {
    if (!dev) {
        return false;
    }

    const enum ggml_backend_dev_type dev_type = ggml_backend_dev_type(dev);
    if (dev_type != GGML_BACKEND_DEVICE_TYPE_GPU &&
        dev_type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
        return false;
    }

    const ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    const std::string reg_name = to_lower_copy(reg ? ggml_backend_reg_name(reg) : nullptr);
    return reg_name.find("vulkan") != std::string::npos;
}

bool is_cuda_device(ggml_backend_dev_t dev) {
    if (!dev) {
        return false;
    }

    const enum ggml_backend_dev_type dev_type = ggml_backend_dev_type(dev);
    if (dev_type != GGML_BACKEND_DEVICE_TYPE_GPU &&
        dev_type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
        return false;
    }

    const ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    const std::string reg_name = to_lower_copy(reg ? ggml_backend_reg_name(reg) : nullptr);
    return reg_name.find("cuda") != std::string::npos;
}

BackendInitResult init_cpu_backend(int n_threads) {
    BackendInitResult result;
    result.backend = ggml_backend_cpu_init();
    if (!result.backend) {
        throw Error(ErrorCode::BackendError, "Failed to initialize CPU backend");
    }

    ggml_backend_cpu_set_n_threads(result.backend, n_threads);
    result.type = BackendType::CPU;
    result.is_gpu = false;
    result.name = ggml_backend_name(result.backend);
    result.description = "CPU backend";
    return result;
}

BackendInitResult init_vulkan_backend() {
    BackendInitResult result;

    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!is_vulkan_device(dev)) {
            continue;
        }

        result.backend = ggml_backend_dev_init(dev, nullptr);
        if (!result.backend) {
            continue;
        }

        result.type = BackendType::Vulkan;
        result.is_gpu = true;
        result.name = ggml_backend_dev_name(dev);
        result.description = ggml_backend_dev_description(dev);
        return result;
    }

    return result;
}

BackendInitResult init_cuda_backend() {
    BackendInitResult result;

    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!is_cuda_device(dev)) {
            continue;
        }

        result.backend = ggml_backend_dev_init(dev, nullptr);
        if (!result.backend) {
            continue;
        }

        result.type = BackendType::CUDA;
        result.is_gpu = true;
        result.name = ggml_backend_dev_name(dev);
        result.description = ggml_backend_dev_description(dev);
        return result;
    }

    return result;
}

BackendInitResult init_requested_backend(BackendType type, int n_threads) {
    switch (type) {
        case BackendType::CPU:
            return init_cpu_backend(n_threads);

        case BackendType::CUDA: {
            BackendInitResult result = init_cuda_backend();
            if (!result.backend) {
                throw Error(ErrorCode::BackendError,
                            "Failed to initialize CUDA backend. Check that VoxCPM was built with "
                            "VOXCPM_CUDA=ON and that CUDA drivers are working.");
            }
            return result;
        }

        case BackendType::Vulkan: {
            BackendInitResult result = init_vulkan_backend();
            if (!result.backend) {
                throw Error(ErrorCode::BackendError,
                            "Failed to initialize Vulkan backend. Check that VoxCPM was built with "
                            "VOXCPM_VULKAN=ON and that Vulkan drivers are working.");
            }
            return result;
        }

        case BackendType::Auto: {
            BackendInitResult result = init_cuda_backend();
            if (result.backend) {
                return result;
            }
            result = init_vulkan_backend();
            if (result.backend) {
                return result;
            }
            return init_cpu_backend(n_threads);
        }

        default:
            throw Error(ErrorCode::BackendError, "Requested backend is not implemented in VoxCPM yet");
    }
}

void free_tracked_buffers(std::vector<ggml_backend_buffer_t>& buffers) {
    for (auto& buf : buffers) {
        if (buf) {
            ggml_backend_buffer_free(buf);
        }
    }
    buffers.clear();
}

size_t scheduler_total_buffer_size(ggml_backend_sched_t sched,
                                   ggml_backend_t primary,
                                   ggml_backend_t cpu_backend) {
    if (!sched) {
        return 0;
    }

    size_t total = 0;
    if (primary) {
        total += ggml_backend_sched_get_buffer_size(sched, primary);
    }
    if (cpu_backend) {
        total += ggml_backend_sched_get_buffer_size(sched, cpu_backend);
    }
    return total;
}

size_t scheduler_graph_requirement(const ggml_cgraph* graph) {
    if (!graph) {
        return 0;
    }
    const int n_nodes = ggml_graph_n_nodes(const_cast<ggml_cgraph*>(graph));
    return static_cast<size_t>(std::max(1, n_nodes)) * 2;
}

}  // namespace

// =============================================================================
// Construction / Destruction
// =============================================================================

VoxCPMBackend::VoxCPMBackend(BackendType type, int n_threads)
    : type_(type), n_threads_(n_threads), backend_(nullptr), gallocr_(nullptr) {
    BackendInitResult result = init_requested_backend(type, n_threads);
    backend_ = result.backend;
    type_ = result.type;
    is_gpu_ = result.is_gpu;
    allocator_logging_enabled_ = env_flag_enabled("VOXCPM_LOG_ALLOCATOR");
    scheduler_logging_enabled_ = env_flag_enabled("VOXCPM_LOG_SCHEDULER");
    backend_name_ = std::move(result.name);
    backend_description_ = std::move(result.description);

    const bool enable_scheduler = env_flag_enabled("VOXCPM_ENABLE_SCHEDULER");
    const bool disable_scheduler = env_flag_enabled("VOXCPM_DISABLE_SCHEDULER");
    if (is_gpu_ && enable_scheduler && !disable_scheduler) {
        cpu_backend_ = init_aux_cpu_backend_handle(n_threads_);
        sched_graph_size_ = GGML_DEFAULT_GRAPH_SIZE;

        ggml_backend_t backends[] = { backend_, cpu_backend_ };
        ggml_backend_buffer_type_t bufts[] = {
            ggml_backend_get_default_buffer_type(backend_),
            ggml_backend_get_default_buffer_type(cpu_backend_),
        };

        ggml_backend_dev_t primary_dev = ggml_backend_get_device(backend_);
        if (primary_dev) {
            ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(primary_dev);
            if (host_buft) {
                bufts[1] = host_buft;
            }
        }

        sched_ = ggml_backend_sched_new(backends, bufts, 2, sched_graph_size_, false, true);
        if (scheduler_logging_enabled_) {
            std::cerr << "[scheduler] enabled=1"
                      << " primary=" << ggml_backend_name(backend_)
                      << " cpu=" << ggml_backend_name(cpu_backend_)
                      << " graph_size=" << sched_graph_size_
                      << " cpu_buft=" << ggml_backend_buft_name(bufts[1])
                      << "\n";
        }
    } else if (scheduler_logging_enabled_) {
        std::cerr << "[scheduler] enabled=0"
                  << " reason=";
        if (!is_gpu_) {
            std::cerr << "non_gpu_backend";
        } else if (disable_scheduler) {
            std::cerr << "disabled_by_env";
        } else if (!enable_scheduler) {
            std::cerr << "not_enabled";
        } else {
            std::cerr << "unknown";
        }
        std::cerr
                  << "\n";
    }
}

VoxCPMBackend::~VoxCPMBackend() {
    free_tracked_buffers(buffers_);

    // Free allocator
    if (sched_) {
        ggml_backend_sched_free(sched_);
    }
    if (gallocr_) {
        ggml_gallocr_free(gallocr_);
    }

    if (cpu_backend_) {
        ggml_backend_free(cpu_backend_);
    }

    // Free backend
    if (backend_) {
        ggml_backend_free(backend_);
    }
}

VoxCPMBackend::VoxCPMBackend(VoxCPMBackend&& other) noexcept
    : type_(other.type_),
      n_threads_(other.n_threads_),
      backend_(other.backend_),
      cpu_backend_(other.cpu_backend_),
      gallocr_(other.gallocr_),
      sched_(other.sched_),
      sched_graph_size_(other.sched_graph_size_),
      is_gpu_(other.is_gpu_),
      allocator_logging_enabled_(other.allocator_logging_enabled_),
      scheduler_logging_enabled_(other.scheduler_logging_enabled_),
      backend_name_(std::move(other.backend_name_)),
      backend_description_(std::move(other.backend_description_)),
      graph_scheduler_modes_(std::move(other.graph_scheduler_modes_)),
      buffers_(std::move(other.buffers_)) {
    other.backend_ = nullptr;
    other.cpu_backend_ = nullptr;
    other.gallocr_ = nullptr;
    other.sched_ = nullptr;
    other.sched_graph_size_ = 0;
    other.is_gpu_ = false;
    other.buffers_.clear();
}

VoxCPMBackend& VoxCPMBackend::operator=(VoxCPMBackend&& other) noexcept {
    if (this != &other) {
        // Free current resources
        free_tracked_buffers(buffers_);
        if (sched_) ggml_backend_sched_free(sched_);
        if (gallocr_) ggml_gallocr_free(gallocr_);
        if (cpu_backend_) ggml_backend_free(cpu_backend_);
        if (backend_) ggml_backend_free(backend_);

        // Move from other
        type_ = other.type_;
        n_threads_ = other.n_threads_;
        backend_ = other.backend_;
        cpu_backend_ = other.cpu_backend_;
        gallocr_ = other.gallocr_;
        sched_ = other.sched_;
        sched_graph_size_ = other.sched_graph_size_;
        is_gpu_ = other.is_gpu_;
        allocator_logging_enabled_ = other.allocator_logging_enabled_;
        scheduler_logging_enabled_ = other.scheduler_logging_enabled_;
        backend_name_ = std::move(other.backend_name_);
        backend_description_ = std::move(other.backend_description_);
        graph_scheduler_modes_ = std::move(other.graph_scheduler_modes_);
        buffers_ = std::move(other.buffers_);

        other.backend_ = nullptr;
        other.cpu_backend_ = nullptr;
        other.gallocr_ = nullptr;
        other.sched_ = nullptr;
        other.sched_graph_size_ = 0;
        other.is_gpu_ = false;
        other.buffers_.clear();
    }
    return *this;
}

// =============================================================================
// Buffer Management
// =============================================================================

ggml_backend_buffer_t VoxCPMBackend::alloc_buffer(ggml_context* ctx, BufferUsage usage) {
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend_);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);

    if (!buffer) {
        throw Error(ErrorCode::OutOfMemory, "Failed to allocate buffer");
    }

    // Set usage for weights
    if (usage == BufferUsage::Weights) {
        ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }

    // Track buffer
    buffers_.push_back(buffer);

    return buffer;
}

void VoxCPMBackend::free_buffer(ggml_backend_buffer_t buffer) {
    if (buffer) {
        // Remove from tracking
        auto it = std::find(buffers_.begin(), buffers_.end(), buffer);
        if (it != buffers_.end()) {
            buffers_.erase(it);
        }
        ggml_backend_buffer_free(buffer);
    }
}

// =============================================================================
// Graph Allocator
// =============================================================================

void VoxCPMBackend::init_allocator() {
    if (gallocr_) {
        ggml_gallocr_free(gallocr_);
    }
    gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    if (!gallocr_) {
        throw Error(ErrorCode::OutOfMemory, "Failed to create graph allocator");
    }
}

void VoxCPMBackend::reset_request_state() {
    graph_scheduler_modes_.clear();

    if (sched_) {
        ggml_backend_sched_reset(sched_);
    }

    if (gallocr_) {
        ggml_gallocr_free(gallocr_);
        gallocr_ = nullptr;
    }
}

void VoxCPMBackend::reserve_compute_memory(ggml_cgraph* graph, const char* stage) {
    const bool use_scheduler = sched_ != nullptr && stage_should_use_scheduler(stage);
    graph_scheduler_modes_[graph] = use_scheduler;

    if (use_scheduler) {
        const size_t required = scheduler_graph_requirement(graph);
        if (required > sched_graph_size_) {
            ggml_backend_t backends[] = { backend_, cpu_backend_ };
            ggml_backend_buffer_type_t bufts[] = {
                ggml_backend_get_default_buffer_type(backend_),
                ggml_backend_get_default_buffer_type(cpu_backend_),
            };

            ggml_backend_dev_t primary_dev = ggml_backend_get_device(backend_);
            if (primary_dev) {
                ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(primary_dev);
                if (host_buft) {
                    bufts[1] = host_buft;
                }
            }

            ggml_backend_sched_free(sched_);
            sched_graph_size_ = required + 1024;
            sched_ = ggml_backend_sched_new(backends, bufts, 2, sched_graph_size_, false, true);
            if (scheduler_logging_enabled_) {
                std::cerr << "[scheduler] resize=1"
                          << " stage=" << (stage ? stage : "(unnamed)")
                          << " required=" << required
                          << " new_graph_size=" << sched_graph_size_
                          << "\n";
            }
        }

        const size_t before = compute_buffer_size();
        const bool ok = ggml_backend_sched_reserve(sched_, graph);
        if (!ok) {
            throw Error(ErrorCode::OutOfMemory, "Failed to reserve scheduler compute buffers");
        }
        if (allocator_logging_enabled_ || scheduler_logging_enabled_) {
            const size_t after = compute_buffer_size();
            std::cerr << "[allocator] action=reserve"
                      << " stage=" << (stage ? stage : "(unnamed)")
                      << " mode=scheduler"
                      << " before=" << format_mib(before)
                      << " after=" << format_mib(after)
                      << " delta=" << format_mib(after >= before ? after - before : 0);
            if (backend_) {
                std::cerr << " primary=" << format_mib(ggml_backend_sched_get_buffer_size(sched_, backend_));
            }
            if (cpu_backend_) {
                std::cerr << " cpu=" << format_mib(ggml_backend_sched_get_buffer_size(sched_, cpu_backend_));
            }
            std::cerr << "\n";
        }
        return;
    }

    if (!gallocr_) {
        init_allocator();
    }
    const size_t before = compute_buffer_size();
    ggml_gallocr_reserve(gallocr_, graph);
    if (allocator_logging_enabled_) {
        const size_t after = compute_buffer_size();
        std::cerr << "[allocator] action=reserve"
                  << " stage=" << (stage ? stage : "(unnamed)")
                  << " before=" << format_mib(before)
                  << " after=" << format_mib(after)
                  << " delta=" << format_mib(after >= before ? after - before : 0)
                  << "\n";
    }
}

void VoxCPMBackend::alloc_graph(ggml_cgraph* graph, const char* stage) {
    const auto it_mode = graph_scheduler_modes_.find(graph);
    const bool use_scheduler = it_mode != graph_scheduler_modes_.end()
        ? it_mode->second
        : (sched_ != nullptr && stage_should_use_scheduler(stage));
    graph_scheduler_modes_[graph] = use_scheduler;

    if (use_scheduler) {
        const size_t before = compute_buffer_size();
        ggml_backend_sched_reset(sched_);
        const bool ok = ggml_backend_sched_alloc_graph(sched_, graph);
        if (!ok) {
            throw Error(ErrorCode::OutOfMemory, "Failed to allocate graph with backend scheduler");
        }
        if (allocator_logging_enabled_ || scheduler_logging_enabled_) {
            const size_t after = compute_buffer_size();
            std::cerr << "[allocator] action=alloc"
                      << " stage=" << (stage ? stage : "(unnamed)")
                      << " mode=scheduler"
                      << " before=" << format_mib(before)
                      << " after=" << format_mib(after)
                      << " delta=" << format_mib(after >= before ? after - before : 0);
            if (backend_) {
                std::cerr << " primary=" << format_mib(ggml_backend_sched_get_buffer_size(sched_, backend_));
            }
            if (cpu_backend_) {
                std::cerr << " cpu=" << format_mib(ggml_backend_sched_get_buffer_size(sched_, cpu_backend_));
            }
            std::cerr << "\n";
        }
        return;
    }

    if (!gallocr_) {
        init_allocator();
    }
    const size_t before = compute_buffer_size();
    ggml_gallocr_alloc_graph(gallocr_, graph);
    if (allocator_logging_enabled_) {
        const size_t after = compute_buffer_size();
        std::cerr << "[allocator] action=alloc"
                  << " stage=" << (stage ? stage : "(unnamed)")
                  << " before=" << format_mib(before)
                  << " after=" << format_mib(after)
                  << " delta=" << format_mib(after >= before ? after - before : 0)
                  << "\n";
    }
}

// =============================================================================
// Graph Execution
// =============================================================================

ggml_status VoxCPMBackend::compute(ggml_cgraph* graph) {
    const auto it_mode = graph_scheduler_modes_.find(graph);
    if (it_mode != graph_scheduler_modes_.end() && it_mode->second) {
        return ggml_backend_sched_graph_compute(sched_, graph);
    }
    return ggml_backend_graph_compute(backend_, graph);
}

// =============================================================================
// Data Transfer
// =============================================================================

void VoxCPMBackend::tensor_set(ggml_tensor* tensor, const void* data, size_t offset, size_t size) {
    if (!tensor) {
        throw Error(ErrorCode::BackendError, "tensor_set received a null tensor");
    }
    if (!data && size > 0) {
        throw Error(ErrorCode::BackendError, "tensor_set received a null data pointer");
    }
    const size_t tensor_bytes = ggml_nbytes(tensor);
    if (offset > tensor_bytes || size > tensor_bytes - offset) {
        std::ostringstream oss;
        oss << "tensor_set overflow for tensor '" << tensor->name
            << "': tensor_bytes=" << tensor_bytes
            << ", offset=" << offset
            << ", size=" << size;
        throw Error(ErrorCode::BackendError, oss.str());
    }

    const auto start = std::chrono::steady_clock::now();
    ggml_backend_tensor_set(tensor, data, offset, size);
    const auto end = std::chrono::steady_clock::now();
    transfer_stats_.host_to_device_bytes += size;
    transfer_stats_.host_to_device_ms +=
        std::chrono::duration<double, std::milli>(end - start).count();
}

void VoxCPMBackend::tensor_get(const ggml_tensor* tensor, void* data, size_t offset, size_t size) {
    const auto start = std::chrono::steady_clock::now();
    ggml_backend_tensor_get(tensor, data, offset, size);
    const auto end = std::chrono::steady_clock::now();
    transfer_stats_.device_to_host_bytes += size;
    transfer_stats_.device_to_host_ms +=
        std::chrono::duration<double, std::milli>(end - start).count();
}

void VoxCPMBackend::tensor_copy(ggml_tensor* src, ggml_tensor* dst) {
    const auto start = std::chrono::steady_clock::now();
    ggml_backend_tensor_copy(src, dst);
    const auto end = std::chrono::steady_clock::now();
    transfer_stats_.device_to_device_bytes += std::min(ggml_nbytes(src), ggml_nbytes(dst));
    transfer_stats_.device_to_device_ms +=
        std::chrono::duration<double, std::milli>(end - start).count();
}

// =============================================================================
// Utilities
// =============================================================================

bool VoxCPMBackend::is_host_buffer(ggml_backend_buffer_t buffer) const {
    return ggml_backend_buffer_is_host(buffer);
}

ggml_backend_buffer_type_t VoxCPMBackend::buffer_type() const {
    return ggml_backend_get_default_buffer_type(backend_);
}

size_t VoxCPMBackend::compute_buffer_size() const {
    size_t total = 0;
    if (sched_) {
        total += scheduler_total_buffer_size(sched_, backend_, cpu_backend_);
    }
    if (gallocr_) {
        total += ggml_gallocr_get_buffer_size(gallocr_, 0);
    }
    return total;
}

// =============================================================================
// Helper Functions
// =============================================================================

std::unique_ptr<VoxCPMBackend> create_best_backend(int n_threads) {
    return std::make_unique<VoxCPMBackend>(BackendType::Auto, n_threads);
}

}  // namespace voxcpm
