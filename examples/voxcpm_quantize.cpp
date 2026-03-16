/**
 * @file voxcpm_quantize.cpp
 * @brief VoxCPM GGUF offline quantization CLI
 */

#include "voxcpm/quantize.h"

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace voxcpm {
namespace {

struct Options {
    std::string input_path;
    std::string output_path;
    std::string imatrix_path;
    ggml_ftype file_type = GGML_FTYPE_MOSTLY_Q4_K;
    std::string file_type_label = "Q4_K";
    AudioVAEQuantizationMode audio_vae_mode = AudioVAEQuantizationMode::Mixed;
    int threads = 4;
    bool dry_run = false;
};

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void print_usage(const char* argv0) {
    std::cerr << "Usage:\n"
              << "  " << argv0
              << " --input MODEL.gguf --output MODEL-quant.gguf"
              << " --type {Q2_K|Q3_K|Q4_K|Q5_K|Q8_0|F16|IQ2_XXS|IQ2_XS|IQ2_S|IQ3_XXS|IQ3_S|IQ1_S|IQ1_M|IQ4_NL|IQ4_XS}"
              << " [options]\n\n"
              << "Options:\n"
              << "  --input PATH\n"
              << "  --output PATH\n"
              << "  --type {Q2_K|Q3_K|Q4_K|Q5_K|Q8_0|F16|IQ2_XXS|IQ2_XS|IQ2_S|IQ3_XXS|IQ3_S|IQ1_S|IQ1_M|IQ4_NL|IQ4_XS} (default: Q4_K)\n"
              << "  --audio-vae-mode {mixed|f16} (default: mixed)\n"
              << "  --imatrix PATH\n"
              << "  --threads INT (default: 4)\n"
              << "  --dry-run\n";
}

bool parse_audio_vae_mode(const std::string& raw, AudioVAEQuantizationMode* mode) {
    std::string value = raw;
    for (char& ch : value) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    if (value == "mixed") {
        *mode = AudioVAEQuantizationMode::Mixed;
        return true;
    }
    if (value == "f16") {
        *mode = AudioVAEQuantizationMode::F16;
        return true;
    }
    return false;
}

bool parse_type(const std::string& raw, ggml_ftype* type, std::string* normalized) {
    const std::string upper = [&]() {
        std::string out = raw;
        for (char& ch : out) {
            ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
        }
        return out;
    }();

    if (upper == "Q4_K" || upper == "Q4_K_M") {
        *type = GGML_FTYPE_MOSTLY_Q4_K;
        *normalized = "Q4_K";
        return true;
    }
    if (upper == "Q3_K" || upper == "Q3_K_M" || upper == "Q3_K_L" || upper == "Q3_K_S") {
        *type = GGML_FTYPE_MOSTLY_Q3_K;
        *normalized = "Q3_K";
        return true;
    }
    if (upper == "Q2_K" || upper == "Q2_K_S") {
        *type = GGML_FTYPE_MOSTLY_Q2_K;
        *normalized = "Q2_K";
        return true;
    }
    if (upper == "Q5_K" || upper == "Q5_K_M") {
        *type = GGML_FTYPE_MOSTLY_Q5_K;
        *normalized = "Q5_K";
        return true;
    }
    if (upper == "Q8_0") {
        *type = GGML_FTYPE_MOSTLY_Q8_0;
        *normalized = "Q8_0";
        return true;
    }
    if (upper == "F16") {
        *type = GGML_FTYPE_MOSTLY_F16;
        *normalized = "F16";
        return true;
    }
    if (upper == "IQ2_XXS") {
        *type = GGML_FTYPE_MOSTLY_IQ2_XXS;
        *normalized = "IQ2_XXS";
        return true;
    }
    if (upper == "IQ2_XS") {
        *type = GGML_FTYPE_MOSTLY_IQ2_XS;
        *normalized = "IQ2_XS";
        return true;
    }
    if (upper == "IQ2_S") {
        *type = GGML_FTYPE_MOSTLY_IQ2_S;
        *normalized = "IQ2_S";
        return true;
    }
    if (upper == "IQ3_XXS") {
        *type = GGML_FTYPE_MOSTLY_IQ3_XXS;
        *normalized = "IQ3_XXS";
        return true;
    }
    if (upper == "IQ3_S") {
        *type = GGML_FTYPE_MOSTLY_IQ3_S;
        *normalized = "IQ3_S";
        return true;
    }
    if (upper == "IQ1_S") {
        *type = GGML_FTYPE_MOSTLY_IQ1_S;
        *normalized = "IQ1_S";
        return true;
    }
    if (upper == "IQ1_M") {
        *type = GGML_FTYPE_MOSTLY_IQ1_M;
        *normalized = "IQ1_M";
        return true;
    }
    if (upper == "IQ4_NL") {
        *type = GGML_FTYPE_MOSTLY_IQ4_NL;
        *normalized = "IQ4_NL";
        return true;
    }
    if (upper == "IQ4_XS") {
        *type = GGML_FTYPE_MOSTLY_IQ4_XS;
        *normalized = "IQ4_XS";
        return true;
    }
    return false;
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto require_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                fail(std::string("Missing value for ") + name);
            }
            return argv[++i];
        };

        if (arg == "--input") {
            options.input_path = require_value("--input");
        } else if (arg == "--output") {
            options.output_path = require_value("--output");
        } else if (arg == "--type") {
            if (!parse_type(require_value("--type"), &options.file_type, &options.file_type_label)) {
                fail("--type must be one of: Q2_K, Q3_K, Q4_K, Q5_K, Q8_0, F16, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ1_S, IQ1_M, IQ4_NL, IQ4_XS");
            }
        } else if (arg == "--audio-vae-mode") {
            if (!parse_audio_vae_mode(require_value("--audio-vae-mode"), &options.audio_vae_mode)) {
                fail("--audio-vae-mode must be one of: mixed, f16");
            }
        } else if (arg == "--imatrix") {
            options.imatrix_path = require_value("--imatrix");
        } else if (arg == "--threads") {
            options.threads = std::stoi(require_value("--threads"));
        } else if (arg == "--dry-run") {
            options.dry_run = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            fail("Unknown argument: " + arg);
        }
    }

    if (options.input_path.empty()) {
        fail("--input is required");
    }
    if (options.output_path.empty() && !options.dry_run) {
        fail("--output is required unless --dry-run is used");
    }
    if (options.threads < 1) {
        fail("--threads must be >= 1");
    }

    const std::filesystem::path input_path(options.input_path);
    if (!std::filesystem::exists(input_path) || !std::filesystem::is_regular_file(input_path)) {
        fail("--input must point to an existing GGUF file");
    }
    if (!options.imatrix_path.empty()) {
        const std::filesystem::path imatrix_path(options.imatrix_path);
        if (!std::filesystem::exists(imatrix_path) || !std::filesystem::is_regular_file(imatrix_path)) {
            fail("--imatrix must point to an existing file");
        }
    }

    return options;
}

void print_stats(const QuantizeStats& stats) {
    const auto print_size_mb = [](size_t bytes) {
        return static_cast<double>(bytes) / (1024.0 * 1024.0);
    };

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "tensors: total=" << stats.total_tensors
              << ", quantized=" << stats.quantized_tensors
              << ", preserved=" << stats.preserved_tensors << "\n";
    std::cout << "audio_vae detail: total=" << stats.audio_vae_tensors
              << ", quantized=" << stats.audio_vae_quantized_tensors
              << ", f16=" << stats.audio_vae_f16_tensors
              << ", preserved=" << stats.audio_vae_preserved_tensors << "\n";
    std::cout << "preserved detail: audio_vae=" << stats.audio_vae_preserved_tensors
              << ", policy=" << stats.skipped_for_policy
              << ", shape=" << stats.skipped_for_shape << "\n";
    std::cout << "size: input=" << print_size_mb(stats.input_bytes)
              << " MiB, output=" << print_size_mb(stats.output_bytes)
              << " MiB\n";

    std::cout << "output tensor types:\n";
    for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
        const size_t count = stats.output_type_counts[static_cast<size_t>(i)];
        if (count == 0) {
            continue;
        }
        std::cout << "  " << ggml_type_name(static_cast<ggml_type>(i)) << ": " << count << "\n";
    }
}

}  // namespace
}  // namespace voxcpm

int main(int argc, char** argv) {
    try {
        const voxcpm::Options options = voxcpm::parse_args(argc, argv);

        voxcpm::QuantizeOptions quantize_options;
        quantize_options.input_path = options.input_path;
        quantize_options.output_path = options.output_path;
        quantize_options.imatrix_path = options.imatrix_path;
        quantize_options.file_type = options.file_type;
        quantize_options.audio_vae_mode = options.audio_vae_mode;
        quantize_options.n_threads = options.threads;
        quantize_options.dry_run = options.dry_run;

        voxcpm::QuantizeStats stats;
        voxcpm::quantize_gguf(quantize_options, &stats);

        std::cout << (options.dry_run ? "Dry run completed" : "Quantization completed")
                  << " for type " << options.file_type_label << "\n";
        voxcpm::print_stats(stats);

        if (!options.dry_run) {
            std::cout << "output: " << options.output_path << "\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "voxcpm_quantize: " << e.what() << "\n";
        return 1;
    }
}
