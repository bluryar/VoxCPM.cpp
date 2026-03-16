#ifndef VOXCPM_QUANTIZE_H
#define VOXCPM_QUANTIZE_H

#include "voxcpm/common.h"

#include <string>
#include <vector>

namespace voxcpm {

enum class AudioVAEQuantizationMode {
    Mixed,
    F16,
};

struct QuantizeOptions {
    std::string input_path;
    std::string output_path;
    std::string imatrix_path;
    ggml_ftype file_type = GGML_FTYPE_MOSTLY_Q4_K;
    AudioVAEQuantizationMode audio_vae_mode = AudioVAEQuantizationMode::Mixed;
    int n_threads = 4;
    bool dry_run = false;
};

struct QuantizeStats {
    size_t input_bytes = 0;
    size_t output_bytes = 0;
    int total_tensors = 0;
    int quantized_tensors = 0;
    int preserved_tensors = 0;
    int audio_vae_tensors = 0;
    int audio_vae_quantized_tensors = 0;
    int audio_vae_f16_tensors = 0;
    int audio_vae_preserved_tensors = 0;
    int skipped_for_shape = 0;
    int skipped_for_policy = 0;
    std::vector<size_t> input_type_counts = std::vector<size_t>(GGML_TYPE_COUNT, 0);
    std::vector<size_t> output_type_counts = std::vector<size_t>(GGML_TYPE_COUNT, 0);
};

bool quantize_gguf(const QuantizeOptions& options, QuantizeStats* stats = nullptr);

}  // namespace voxcpm

#endif  // VOXCPM_QUANTIZE_H
