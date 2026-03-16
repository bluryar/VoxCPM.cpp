#ifndef VOXCPM_WEIGHT_STORE_H
#define VOXCPM_WEIGHT_STORE_H

#include "voxcpm/common.h"

#include <memory>
#include <string>
#include <vector>

namespace voxcpm {

class VoxCPMBackend;

class VoxCPMWeightStore {
public:
    VoxCPMWeightStore() = default;
    ~VoxCPMWeightStore();

    VoxCPMWeightStore(const VoxCPMWeightStore&) = delete;
    VoxCPMWeightStore& operator=(const VoxCPMWeightStore&) = delete;

    bool load_from_file(const std::string& gguf_path, VoxCPMBackend& backend);

    ggml_tensor* get_tensor(const char* name) const;
    bool has_tensor(const char* name) const;

    bool get_u32(const char* key, uint32_t& value) const;
    bool get_f32(const char* key, float& value) const;
    bool get_bool(const char* key, bool& value) const;
    bool get_string(const char* key, std::string& value) const;
    bool get_i32_array(const char* key, std::vector<int>& values) const;
    bool get_f32_array(const char* key, std::vector<float>& values) const;
    bool get_string_array(const char* key, std::vector<std::string>& values) const;

    gguf_context* gguf() const { return gguf_ctx_; }
    ggml_context* ggml_ctx() const { return ggml_ctx_; }
    ggml_backend_buffer_t buffer() const { return buffer_; }
    const std::string& path() const { return path_; }

    size_t buffer_size() const;
    int tensor_count() const;
    bool owns_storage() const { return gguf_ctx_ != nullptr && ggml_ctx_ != nullptr && buffer_ != nullptr; }

private:
    gguf_context* gguf_ctx_ = nullptr;
    ggml_context* ggml_ctx_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    std::string path_;
};

using SharedWeightStore = std::shared_ptr<VoxCPMWeightStore>;

}  // namespace voxcpm

#endif  // VOXCPM_WEIGHT_STORE_H
