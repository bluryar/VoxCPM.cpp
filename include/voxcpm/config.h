/**
 * @file config.h
 * @brief VoxCPM Configuration Structures
 *
 * Configuration structures for all VoxCPM components.
 */

#ifndef VOXCPM_CONFIG_H
#define VOXCPM_CONFIG_H

#include <array>
#include <cstdint>
#include <string>
#include <vector>
#include <algorithm>

namespace voxcpm {


static inline int ascii_tolower_char(unsigned char c) {
    return std::tolower(c);
}
inline std::string ascii_tolower(const std::string& str) {
    std::string lower = str;
    std::transform(lower.begin(), lower.end(), lower.begin(), ascii_tolower_char);
    return lower;
}


// =============================================================================
// AudioVAE Configuration
// =============================================================================

// ModelVersion labels must be ordered in the same direction as the enum values (presumably ascending).
enum class ModelVersion {
    Unknown,
    v0_5,
    v1_5,
    v2_0
};
// Parse in descending order, since newer versions are more likely to be used.
inline ModelVersion parse_model_version_from_str(const std::string& str) {
    std::string lower = ascii_tolower(str);
    // This format was used in the 2.0 model file:
    if (lower == "voxcpm2")     return ModelVersion::v2_0;
    // This format was used by the `--model-name` parameter:
    if (lower == "voxcpm-2")    return ModelVersion::v2_0;
    if (lower == "voxcpm-1.5")  return ModelVersion::v1_5;
    if (lower == "voxcpm-0.5b") return ModelVersion::v0_5;

    return ModelVersion::Unknown;
}

/**
 * @brief AudioVAE Configuration
 *
 * AudioVAE is a variational autoencoder for audio processing:
 * - Encoder: Audio waveform -> Latent representation (64-dim)
 * - Decoder: Latent representation -> Audio waveform
 *
 * Key parameters:
 * - encoder_rates: Downsampling factors for each encoder block
 *   hop_length = product of encoder_rates (e.g., 2*3*6*7*7 = 1764)
 * - decoder_rates: Upsampling factors for each decoder block (reverse order)
 */
struct AudioVAEConfig {
    // Model version
    ModelVersion model_version = ModelVersion::Unknown;

    // Dimensions
    int encoder_dim = 128;      // Upstream torch default encoder channel dimension
    int latent_dim = 64;        // Latent space dimension
    int decoder_dim = 1536;     // Upstream torch default decoder channel dimension

    // Sampling
    int sample_rate = 16000;    // Upstream torch default sample rate (Hz)
    int out_sample_rate = 0;    // Optional decode/output sample rate (Hz)
    std::vector<int> sr_bin_boundaries = {20000, 30000, 40000};

    // Encoder/Decoder rates (downsampling/upsampling factors)
    // Supports variable number of blocks via std::vector
    std::vector<int> encoder_rates = {2, 5, 8, 8};
    std::vector<int> decoder_rates = {8, 8, 5, 2};

    // Convolution settings
    bool depthwise = true;      // Use depthwise convolution

    // Noise injection (optional)
    bool use_noise_block = false;

    // Decoder-side sample-rate conditioning (AudioVAE v2)
    std::string cond_type = "scale_bias";
    int cond_dim = 128;
    bool cond_out_layer = false;

    /**
     * @brief Get hop length (total downsampling factor)
     * hop_length = product of all encoder_rates
     */
    int hop_length() const {
        int hop = 1;
        for (int r : encoder_rates) hop *= r;
        return hop;
    }

    /**
     * @brief Get decoder output step length (total upsampling factor)
     * decode_hop_length = product of all decoder_rates
     */
    int decode_hop_length() const {
        int hop = 1;
        for (int r : decoder_rates) hop *= r;
        return hop;
    }

    /**
     * @brief Get number of encoder blocks
     */
    int num_encoder_blocks() const { return static_cast<int>(encoder_rates.size()); }

    /**
     * @brief Get number of decoder blocks
     */
    int num_decoder_blocks() const { return static_cast<int>(decoder_rates.size()); }

    int output_sample_rate() const { return out_sample_rate > 0 ? out_sample_rate : sample_rate; }

    int sr_bin_bucket_count() const { return static_cast<int>(sr_bin_boundaries.size()) + 1; }

    int sample_rate_bucket(int sample_rate_hz) const {
        int bucket = 0;
        while (bucket < static_cast<int>(sr_bin_boundaries.size()) &&
               sample_rate_hz > sr_bin_boundaries[static_cast<size_t>(bucket)]) {
            ++bucket;
        }
        return bucket;
    }

    /**
     * @brief Calculate encoder channel progression
     * e.g., 64 -> 128 -> 256 -> 512 -> 1024 -> 2048 for 5 blocks
     */
    std::vector<int> encoder_channels() const {
        std::vector<int> channels;
        channels.push_back(encoder_dim);
        int ch = encoder_dim;
        for (size_t i = 0; i < encoder_rates.size(); i++) {
            ch *= 2;
            channels.push_back(ch);
        }
        return channels;
    }

    /**
     * @brief Calculate decoder channel progression
     * e.g., 2048 -> 1024 -> 512 -> 256 -> 128 -> 64 for 5 blocks
     */
    std::vector<int> decoder_channels() const {
        std::vector<int> channels;
        channels.push_back(decoder_dim);
        int ch = decoder_dim;
        for (size_t i = 0; i < decoder_rates.size(); i++) {
            ch /= 2;
            channels.push_back(ch);
        }
        return channels;
    }
};

// =============================================================================
// MiniCPM Configuration (for future use)
// =============================================================================

/**
 * @brief MiniCPM Transformer Configuration
 *
 * Used by BaseLM (24 layers), ResidualLM (8 layers),
 * LocEnc (8 layers), LocDiT (8 layers).
 */
struct MiniCPMConfig {
    int hidden_size = 1024;
    int intermediate_size = 4096;
    int n_layer = 8;
    int n_heads = 16;
    int n_kv_heads = 2;          // GQA: 16 query heads, 2 KV heads
    int kv_channels = 64;        // Per-head Q/K/V channel width
    int vocab_size = 73448;
    int max_length = 32768;

    float rms_norm_eps = 1e-5f;
    float rope_freq_base = 10000.0f;

    // Scale factors (MiniCPM specific)
    int scale_emb = 12;
    int dim_model_base = 256;
    float scale_depth = 1.4f;
    bool use_mup = false;
    bool no_rope = false;

    // LongRoPE configuration
    int rope_original_max = 32768;
    std::vector<float> rope_long_factor;
    std::vector<float> rope_short_factor;

    int head_dim() const { return kv_channels > 0 ? kv_channels : hidden_size / n_heads; }
};

// =============================================================================
// LocEnc Configuration (for future use)
// =============================================================================

/**
 * @brief LocEnc (Local Encoder) Configuration
 *
 * Non-causal Transformer encoder for audio feature encoding.
 * Processes patch features with special CLS token.
 */
struct LocEncConfig {
    int hidden_size = 1024;
    int n_layer = 4;
    int n_heads = 16;
    int n_kv_heads = 2;
    int intermediate_size = 4096;
    int patch_size = 2;
    int feat_dim = 64;
    float rms_norm_eps = 1e-5f;
};

// =============================================================================
// LocDiT Configuration (for future use)
// =============================================================================

/**
 * @brief LocDiT (Local Diffusion Transformer) Configuration
 *
 * DiT-based diffusion model for audio generation with CFM.
 */
struct LocDiTConfig {
    int hidden_size = 1024;
    int n_layer = 4;
    int n_heads = 16;
    int n_kv_heads = 2;
    int intermediate_size = 4096;
    int patch_size = 2;
    int feat_dim = 64;
    float rms_norm_eps = 1e-5f;

    // CFM settings
    float sigma_min = 1e-6f;
    float cfg_rate = 2.0f;
    int cfm_steps = 10;
};

// =============================================================================
// FSQ Configuration (for future use)
// =============================================================================

/**
 * @brief FSQ (Finite Scalar Quantization) Configuration
 */
struct FSQConfig {
    int latent_dim = 256;
    int scale = 9;               // Quantization levels: [-scale, scale]
    int hidden_size = 1024;      // Input/output dimension
};

// =============================================================================
// Projection Configuration
// =============================================================================

/**
 * @brief Linear Projection Layer Configuration
 *
 * Used for:
 * - enc_to_lm_proj: Projects LocEnc output to LM input
 * - lm_to_dit_proj: Projects LM output to DiT input
 * - res_to_dit_proj: Projects ResidualLM output to DiT input
 */
struct ProjectionConfig {
    int in_dim = 1024;
    int out_dim = 1024;
};

// =============================================================================
// Stop Token Configuration
// =============================================================================

/**
 * @brief Stop Token Prediction Configuration
 *
 * Predicts when generation should stop (binary classification).
 * Architecture: Linear -> SiLU -> Linear (no bias on last layer)
 */
struct StopTokenConfig {
    int hidden_dim = 1024;
    int num_classes = 2;         // stop / continue
};

// =============================================================================
// Embedding Configuration
// =============================================================================

/**
 * @brief Token Embedding Configuration
 *
 * Token embedding lookup table.
 * scale is applied when use_mup=true (MiniCPM specific).
 */
struct EmbeddingConfig {
    int vocab_size = 73448;
    int hidden_dim = 1024;
    float scale = 1.0f;          // scale_emb (12 for MiniCPM)
};

// =============================================================================
// VoxCPM Full Model Configuration
// =============================================================================

/**
 * @brief Complete VoxCPM Model Configuration
 */
struct VoxCPMConfig {
    AudioVAEConfig audio_vae;
    MiniCPMConfig base_lm;
    MiniCPMConfig residual_lm;
    LocEncConfig loc_enc;
    LocDiTConfig loc_dit;
    FSQConfig fsq;

    // Global settings
    int patch_size = 2;
    int feat_dim = 64;
    int max_length = 4096;
};

}  // namespace voxcpm

#endif  // VOXCPM_CONFIG_H
