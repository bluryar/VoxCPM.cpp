/**
 * @file voxcpm_tts.cpp
 * @brief Non-streaming VoxCPM TTS CLI
 */

#include "voxcpm/audio-vae.h"
#include "voxcpm/backend.h"
#include "voxcpm/context.h"
#include "voxcpm/tokenizer.h"
#include "voxcpm/voxcpm.h"
#include "voxcpm/weight-store.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace voxcpm {
namespace {

struct Options {
    std::string text;
    std::string output_path;
    std::string stream_dir;
    std::string prompt_audio_path;
    std::string prompt_text;
    std::string model_path;
    BackendType backend = BackendType::CPU;
    float cfg_value = 2.0f;
    int inference_timesteps = 10;
    int threads = 4;
    bool normalize = false;
    bool stream = false;
};

struct WavData {
    int sample_rate = 0;
    int channels = 0;
    std::vector<float> samples;
};

struct PreparedInputs {
    std::vector<int32_t> full_text_tokens;
    std::vector<int32_t> text_mask;
    std::vector<int32_t> feat_mask;
    std::vector<float> feat;
    std::vector<float> prompt_feat;
    int prompt_audio_length = 0;
    bool has_prompt_audio = false;
};

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

const char* backend_type_name(BackendType type) {
    switch (type) {
        case BackendType::CPU:
            return "cpu";
        case BackendType::Vulkan:
            return "vulkan";
        case BackendType::Auto:
            return "auto";
        case BackendType::CUDA:
            return "cuda";
        case BackendType::Metal:
            return "metal";
        default:
            return "unknown";
    }
}

bool env_flag_enabled(const char* name) {
    const char* raw = std::getenv(name);
    if (!raw || raw[0] == '\0') {
        return false;
    }

    std::string value(raw);
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value == "1" || value == "true" || value == "yes" || value == "on";
}

double bytes_to_mib(size_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

size_t decode_state_kv_bytes(const VoxCPMDecodeState& state) {
    size_t total = 0;
    if (state.base_lm_cache) {
        total += state.base_lm_cache->buffer_size();
    }
    if (state.residual_lm_cache) {
        total += state.residual_lm_cache->buffer_size();
    }
    return total;
}

void log_memory_breakdown(bool enabled,
                          const char* stage,
                          const VoxCPMWeightStore& store,
                          const VoxCPMBackend& backend,
                          const VoxCPMDecodeState* state) {
    if (!enabled) {
        return;
    }

    const size_t weights = store.buffer_size();
    const size_t compute = backend.compute_buffer_size();
    const size_t kv = state ? decode_state_kv_bytes(*state) : 0;
    const size_t tracked_total = weights + compute + kv;

    std::cerr << std::fixed << std::setprecision(2)
              << "[memory] stage=" << stage
              << " weights=" << bytes_to_mib(weights) << " MiB"
              << " compute_arena=" << bytes_to_mib(compute) << " MiB"
              << " kv_cache=" << bytes_to_mib(kv) << " MiB"
              << " tracked_total=" << bytes_to_mib(tracked_total) << " MiB"
              << "\n";
}

BackendType parse_backend_type(const std::string& value) {
    if (value == "cpu") {
        return BackendType::CPU;
    }
    if (value == "vulkan") {
        return BackendType::Vulkan;
    }
    if (value == "auto") {
        return BackendType::Auto;
    }
    fail("Unsupported backend: " + value + " (expected cpu, vulkan, or auto)");
}

void print_usage(const char* argv0) {
    std::cerr << "Usage:\n"
              << "  " << argv0 << " --text TEXT --output OUTPUT --model-path MODEL.gguf [options]\n\n"
              << "Options:\n"
              << "  --text, -t TEXT\n"
              << "  --output, -o OUTPUT\n"
              << "  --prompt-audio, -pa PROMPT_AUDIO\n"
              << "  --prompt-text, -pt PROMPT_TEXT\n"
              << "  --cfg-value FLOAT (default: 2.0)\n"
              << "  --inference-timesteps INT (default: 10)\n"
              << "  --backend {cpu|vulkan|auto} (default: cpu)\n"
              << "  --threads INT (default: 4)\n"
              << "  --stream\n"
              << "  --stream-dir DIR\n"
              << "  --normalize\n"
              << "  --model-path GGUF\n";
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

        if (arg == "--text" || arg == "-t") {
            options.text = require_value("--text");
        } else if (arg == "--output" || arg == "-o") {
            options.output_path = require_value("--output");
        } else if (arg == "--prompt-audio" || arg == "-pa") {
            options.prompt_audio_path = require_value("--prompt-audio");
        } else if (arg == "--prompt-text" || arg == "-pt") {
            options.prompt_text = require_value("--prompt-text");
        } else if (arg == "--cfg-value") {
            options.cfg_value = std::stof(require_value("--cfg-value"));
        } else if (arg == "--inference-timesteps") {
            options.inference_timesteps = std::stoi(require_value("--inference-timesteps"));
        } else if (arg == "--backend") {
            options.backend = parse_backend_type(require_value("--backend"));
        } else if (arg == "--threads") {
            options.threads = std::stoi(require_value("--threads"));
        } else if (arg == "--stream") {
            options.stream = true;
        } else if (arg == "--stream-dir") {
            options.stream_dir = require_value("--stream-dir");
        } else if (arg == "--normalize") {
            options.normalize = true;
        } else if (arg == "--model-path") {
            options.model_path = require_value("--model-path");
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            fail("Unknown argument: " + arg);
        }
    }

    if (options.text.empty()) {
        fail("--text is required");
    }
    if (options.output_path.empty()) {
        fail("--output is required");
    }
    if (options.model_path.empty()) {
        fail("--model-path is required");
    }
    if ((options.prompt_audio_path.empty()) != (options.prompt_text.empty())) {
        fail("--prompt-audio and --prompt-text must be provided together");
    }
    if (!(0.1f <= options.cfg_value && options.cfg_value <= 10.0f)) {
        fail("--cfg-value must be between 0.1 and 10.0");
    }
    if (!(1 <= options.inference_timesteps && options.inference_timesteps <= 100)) {
        fail("--inference-timesteps must be between 1 and 100");
    }
    if (options.threads < 1) {
        fail("--threads must be >= 1");
    }
    if (options.normalize) {
        fail("C++ text normalization not implemented");
    }
    if (options.stream && options.stream_dir.empty()) {
        fail("--stream requires --stream-dir");
    }

    const std::filesystem::path model_path(options.model_path);
    if (!std::filesystem::exists(model_path) || !std::filesystem::is_regular_file(model_path)) {
        fail("--model-path must point to an existing GGUF file");
    }
    if (!options.prompt_audio_path.empty() && !std::filesystem::exists(options.prompt_audio_path)) {
        fail("Prompt audio file does not exist: " + options.prompt_audio_path);
    }

    return options;
}

uint16_t read_le_u16(std::istream& in) {
    uint8_t bytes[2] = {0, 0};
    in.read(reinterpret_cast<char*>(bytes), 2);
    return static_cast<uint16_t>(bytes[0] | (bytes[1] << 8));
}

uint32_t read_le_u32(std::istream& in) {
    uint8_t bytes[4] = {0, 0, 0, 0};
    in.read(reinterpret_cast<char*>(bytes), 4);
    return static_cast<uint32_t>(bytes[0] |
                                 (bytes[1] << 8) |
                                 (bytes[2] << 16) |
                                 (bytes[3] << 24));
}

WavData read_wav_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        fail("Failed to open WAV file: " + path);
    }

    char riff[4] = {0};
    char wave[4] = {0};
    in.read(riff, 4);
    const uint32_t riff_size = read_le_u32(in);
    (void)riff_size;
    in.read(wave, 4);
    if (std::string(riff, 4) != "RIFF" || std::string(wave, 4) != "WAVE") {
        fail("Invalid WAV header: " + path);
    }

    uint16_t audio_format = 0;
    uint16_t num_channels = 0;
    uint32_t sample_rate = 0;
    uint16_t bits_per_sample = 0;
    std::vector<uint8_t> data_chunk;

    while (in && (!sample_rate || data_chunk.empty())) {
        char chunk_id[4] = {0};
        in.read(chunk_id, 4);
        if (in.gcount() != 4) {
            break;
        }
        const uint32_t chunk_size = read_le_u32(in);
        const std::string id(chunk_id, 4);

        if (id == "fmt ") {
            audio_format = read_le_u16(in);
            num_channels = read_le_u16(in);
            sample_rate = read_le_u32(in);
            const uint32_t byte_rate = read_le_u32(in);
            const uint16_t block_align = read_le_u16(in);
            (void)byte_rate;
            (void)block_align;
            bits_per_sample = read_le_u16(in);
            if (chunk_size > 16) {
                in.seekg(static_cast<std::streamoff>(chunk_size - 16), std::ios::cur);
            }
        } else if (id == "data") {
            data_chunk.resize(chunk_size);
            in.read(reinterpret_cast<char*>(data_chunk.data()), static_cast<std::streamsize>(chunk_size));
        } else {
            in.seekg(static_cast<std::streamoff>(chunk_size), std::ios::cur);
        }
        if (chunk_size % 2 != 0) {
            in.seekg(1, std::ios::cur);
        }
    }

    if (sample_rate == 0 || num_channels == 0 || data_chunk.empty()) {
        fail("Incomplete WAV file: " + path);
    }
    if (audio_format != 1 && audio_format != 3) {
        fail("Unsupported WAV format in " + path + " (only PCM/float supported)");
    }

    const size_t bytes_per_sample = static_cast<size_t>(bits_per_sample) / 8;
    if (bytes_per_sample == 0) {
        fail("Invalid bits-per-sample in WAV file: " + path);
    }

    const size_t frame_count = data_chunk.size() / (bytes_per_sample * num_channels);
    std::vector<float> samples(frame_count * num_channels, 0.0f);

    size_t offset = 0;
    for (size_t i = 0; i < frame_count * num_channels; ++i) {
        if (audio_format == 3 && bits_per_sample == 32) {
            float value = 0.0f;
            std::memcpy(&value, data_chunk.data() + offset, sizeof(float));
            samples[i] = value;
        } else if (audio_format == 1 && bits_per_sample == 16) {
            const int16_t value = static_cast<int16_t>(data_chunk[offset] | (data_chunk[offset + 1] << 8));
            samples[i] = static_cast<float>(value) / 32768.0f;
        } else if (audio_format == 1 && bits_per_sample == 24) {
            int32_t value = (static_cast<int32_t>(data_chunk[offset]) |
                             (static_cast<int32_t>(data_chunk[offset + 1]) << 8) |
                             (static_cast<int32_t>(data_chunk[offset + 2]) << 16));
            if (value & 0x800000) {
                value |= ~0xFFFFFF;
            }
            samples[i] = static_cast<float>(value) / 8388608.0f;
        } else if (audio_format == 1 && bits_per_sample == 32) {
            int32_t value = 0;
            std::memcpy(&value, data_chunk.data() + offset, sizeof(int32_t));
            samples[i] = static_cast<float>(value) / 2147483648.0f;
        } else {
            fail("Unsupported WAV bit depth in " + path);
        }
        offset += bytes_per_sample;
    }

    return WavData{
        static_cast<int>(sample_rate),
        static_cast<int>(num_channels),
        std::move(samples),
    };
}

std::vector<float> convert_to_mono(const WavData& wav) {
    if (wav.channels == 1) {
        return wav.samples;
    }

    const size_t frame_count = wav.samples.size() / static_cast<size_t>(wav.channels);
    std::vector<float> mono(frame_count, 0.0f);
    for (size_t frame = 0; frame < frame_count; ++frame) {
        float sum = 0.0f;
        for (int channel = 0; channel < wav.channels; ++channel) {
            sum += wav.samples[frame * static_cast<size_t>(wav.channels) + static_cast<size_t>(channel)];
        }
        mono[frame] = sum / static_cast<float>(wav.channels);
    }
    return mono;
}

std::vector<float> linear_resample(const std::vector<float>& input, int src_rate, int dst_rate) {
    if (src_rate == dst_rate || input.empty()) {
        return input;
    }

    const double scale = static_cast<double>(dst_rate) / static_cast<double>(src_rate);
    const size_t out_size = std::max<size_t>(1, static_cast<size_t>(std::llround(input.size() * scale)));
    std::vector<float> out(out_size, 0.0f);

    for (size_t i = 0; i < out_size; ++i) {
        const double src_pos = static_cast<double>(i) / scale;
        const size_t left = static_cast<size_t>(std::floor(src_pos));
        const size_t right = std::min(left + 1, input.size() - 1);
        const double frac = src_pos - static_cast<double>(left);
        out[i] = static_cast<float>((1.0 - frac) * input[left] + frac * input[right]);
    }

    return out;
}

void write_wav_pcm16(const std::string& path, const std::vector<float>& audio, int sample_rate) {
    const std::filesystem::path output_path(path);
    if (output_path.has_parent_path()) {
        std::filesystem::create_directories(output_path.parent_path());
    }

    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        fail("Failed to open output WAV file: " + path);
    }

    const uint16_t channels = 1;
    const uint16_t bits_per_sample = 16;
    const uint32_t byte_rate = sample_rate * channels * (bits_per_sample / 8);
    const uint16_t block_align = channels * (bits_per_sample / 8);
    const uint32_t data_size = static_cast<uint32_t>(audio.size() * sizeof(int16_t));
    const uint32_t riff_size = 36 + data_size;

    out.write("RIFF", 4);
    out.write(reinterpret_cast<const char*>(&riff_size), 4);
    out.write("WAVE", 4);

    const uint32_t fmt_size = 16;
    const uint16_t audio_format = 1;
    out.write("fmt ", 4);
    out.write(reinterpret_cast<const char*>(&fmt_size), 4);
    out.write(reinterpret_cast<const char*>(&audio_format), 2);
    out.write(reinterpret_cast<const char*>(&channels), 2);
    out.write(reinterpret_cast<const char*>(&sample_rate), 4);
    out.write(reinterpret_cast<const char*>(&byte_rate), 4);
    out.write(reinterpret_cast<const char*>(&block_align), 2);
    out.write(reinterpret_cast<const char*>(&bits_per_sample), 2);

    out.write("data", 4);
    out.write(reinterpret_cast<const char*>(&data_size), 4);
    for (float sample : audio) {
        const float clamped = std::max(-1.0f, std::min(1.0f, sample));
        const int16_t pcm = static_cast<int16_t>(std::lrint(clamped * 32767.0f));
        out.write(reinterpret_cast<const char*>(&pcm), sizeof(int16_t));
    }
}

std::string chunk_output_path(const std::string& stream_dir, int index) {
    std::ostringstream oss;
    oss << "chunk_" << std::setw(4) << std::setfill('0') << index << ".wav";
    return (std::filesystem::path(stream_dir) / oss.str()).string();
}

std::vector<float> extract_prompt_features(AudioVAE& audio_vae,
                                           VoxCPMBackend& backend,
                                           std::vector<float> audio,
                                           int sample_rate,
                                           int patch_size,
                                           int feat_dim) {
    std::cerr << "Encoding prompt audio...\n";
    VoxCPMContext graph_ctx(ContextType::Graph, 32768, 262144);
    ggml_tensor* latent = audio_vae.encode(graph_ctx, audio, sample_rate);
    if (!latent) {
        fail("Failed to build AudioVAE encode graph");
    }

    ggml_cgraph* graph = graph_ctx.new_graph();
    graph_ctx.build_forward(graph, latent);
    backend.reserve_compute_memory(graph, "tts.audio_vae.encode");
    backend.alloc_graph(graph, "tts.audio_vae.encode");
    const auto& preprocessed = audio_vae.last_preprocessed_audio();
    backend.tensor_set(audio_vae.last_input_tensor(), preprocessed.data(), 0, preprocessed.size() * sizeof(float));
    if (backend.compute(graph) != GGML_STATUS_SUCCESS) {
        fail("AudioVAE encode failed");
    }

    const int total_patches = static_cast<int>(latent->ne[0]);
    const int latent_dim = static_cast<int>(latent->ne[1]);
    if (latent_dim != feat_dim) {
        fail("Prompt latent dim mismatch");
    }
    if (total_patches % patch_size != 0) {
        fail("Prompt latent patches are not divisible by patch size");
    }

    std::vector<float> encoded(static_cast<size_t>(total_patches) * latent_dim);
    backend.tensor_get(latent, encoded.data(), 0, encoded.size() * sizeof(float));

    const int audio_length = total_patches / patch_size;
    std::vector<float> features(static_cast<size_t>(audio_length) * patch_size * feat_dim, 0.0f);
    for (int t = 0; t < audio_length; ++t) {
        for (int p = 0; p < patch_size; ++p) {
            const int patch_index = t * patch_size + p;
            for (int d = 0; d < feat_dim; ++d) {
                const size_t src = static_cast<size_t>(d) * total_patches + patch_index;
                const size_t dst = (static_cast<size_t>(t) * patch_size + p) * feat_dim + d;
                features[dst] = encoded[src];
            }
        }
    }
    return features;
}

std::vector<float> decode_audio(AudioVAE& audio_vae,
                                VoxCPMBackend& backend,
                                const std::vector<float>& features,
                                int total_patches,
                                int feat_dim) {
    std::cerr << "Decoding waveform from " << total_patches << " latent patches...\n";
    VoxCPMContext graph_ctx(ContextType::Graph, 32768, 262144);
    ggml_tensor* latent = graph_ctx.new_tensor_2d(GGML_TYPE_F32, total_patches, feat_dim);
    ggml_set_input(latent);
    ggml_tensor* audio = audio_vae.decode(graph_ctx, latent);
    if (!audio) {
        fail("Failed to build AudioVAE decode graph");
    }

    ggml_cgraph* graph = graph_ctx.new_graph();
    graph_ctx.build_forward(graph, audio);
    backend.reserve_compute_memory(graph, "tts.audio_vae.decode");
    backend.alloc_graph(graph, "tts.audio_vae.decode");
    backend.tensor_set(latent, features.data(), 0, features.size() * sizeof(float));
    if (backend.compute(graph) != GGML_STATUS_SUCCESS) {
        fail("AudioVAE decode failed");
    }

    std::vector<float> waveform(static_cast<size_t>(ggml_nelements(audio)));
    backend.tensor_get(audio, waveform.data(), 0, waveform.size() * sizeof(float));
    return waveform;
}

void fill_noise(std::vector<float>& noise, int patch_size, int feat_dim, std::mt19937& rng) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    noise.resize(static_cast<size_t>(patch_size) * feat_dim);
    for (float& value : noise) {
        value = dist(rng);
    }
}

std::vector<float> build_decode_feature_sequence(const std::vector<float>& prompt_feat,
                                                 int prompt_audio_length,
                                                 const std::vector<float>& generated_steps,
                                                 int streaming_prefix_len,
                                                 int patch_size,
                                                 int feat_dim,
                                                 int* prepended_context_frames) {
    const size_t frame_stride = static_cast<size_t>(patch_size) * feat_dim;

    int context_frames = 0;
    if (!prompt_feat.empty() && prompt_audio_length > 0 && streaming_prefix_len > 1) {
        context_frames = std::min(streaming_prefix_len - 1, prompt_audio_length);
    }

    std::vector<float> decode_frames;
    decode_frames.reserve(static_cast<size_t>(context_frames) * frame_stride + generated_steps.size());
    if (context_frames > 0) {
        const size_t context_offset = static_cast<size_t>(prompt_audio_length - context_frames) * frame_stride;
        decode_frames.insert(decode_frames.end(),
                             prompt_feat.begin() + static_cast<std::ptrdiff_t>(context_offset),
                             prompt_feat.end());
    }
    decode_frames.insert(decode_frames.end(), generated_steps.begin(), generated_steps.end());

    if (prepended_context_frames != nullptr) {
        *prepended_context_frames = context_frames;
    }
    return decode_frames;
}

void patch_major_to_latent(const std::vector<float>& frames,
                           int patch_size,
                           int feat_dim,
                           std::vector<float>& latent) {
    const size_t frame_stride = static_cast<size_t>(patch_size) * feat_dim;
    const int total_frames = static_cast<int>(frames.size() / frame_stride);
    const int total_patches = total_frames * patch_size;
    latent.assign(static_cast<size_t>(total_patches) * feat_dim, 0.0f);
    for (int frame = 0; frame < total_frames; ++frame) {
        for (int patch = 0; patch < patch_size; ++patch) {
            const int time_index = frame * patch_size + patch;
            for (int d = 0; d < feat_dim; ++d) {
                const size_t src = (static_cast<size_t>(frame) * patch_size + patch) * feat_dim + d;
                const size_t dst = static_cast<size_t>(d) * total_patches + time_index;
                latent[dst] = frames[src];
            }
        }
    }
}

std::vector<float> patch_major_to_latent(const std::vector<float>& frames,
                                         int patch_size,
                                         int feat_dim) {
    std::vector<float> latent;
    patch_major_to_latent(frames, patch_size, feat_dim, latent);
    return latent;
}

void append_stream_frame(std::vector<float>& recent_frames,
                         const std::vector<float>& patch,
                         int max_frames,
                         int patch_size,
                         int feat_dim) {
    const size_t frame_stride = static_cast<size_t>(patch_size) * feat_dim;
    recent_frames.insert(recent_frames.end(), patch.begin(), patch.end());
    const size_t max_elems = static_cast<size_t>(max_frames) * frame_stride;
    if (recent_frames.size() > max_elems) {
        recent_frames.erase(recent_frames.begin(),
                            recent_frames.begin() + static_cast<std::ptrdiff_t>(recent_frames.size() - max_elems));
    }
}

PreparedInputs prepare_inputs(const Options& options,
                              ChineseCharSplitTokenizer& split_tokenizer,
                              AudioVAE& audio_vae,
                              VoxCPMBackend& backend,
                              int patch_size,
                              int feat_dim,
                              int patch_len) {
    PreparedInputs prepared;

    std::vector<int32_t> text_tokens = split_tokenizer.encode(
        options.prompt_audio_path.empty() ? options.text : options.prompt_text + options.text,
        false);
    text_tokens.push_back(101);

    prepared.full_text_tokens = text_tokens;
    if (options.prompt_audio_path.empty()) {
        const int seq_len = static_cast<int>(text_tokens.size());
        prepared.feat.assign(static_cast<size_t>(seq_len) * patch_size * feat_dim, 0.0f);
        prepared.text_mask.assign(static_cast<size_t>(seq_len), 1);
        prepared.feat_mask.assign(static_cast<size_t>(seq_len), 0);
        return prepared;
    }

    prepared.has_prompt_audio = true;
    const WavData wav = read_wav_file(options.prompt_audio_path);
    std::vector<float> mono = convert_to_mono(wav);
    mono = linear_resample(mono, wav.sample_rate, audio_vae.config().sample_rate);
    if (mono.size() % static_cast<size_t>(patch_len) != 0) {
        const size_t padding = static_cast<size_t>(patch_len) - (mono.size() % static_cast<size_t>(patch_len));
        mono.insert(mono.begin(), padding, 0.0f);
    }

    prepared.prompt_feat = extract_prompt_features(
        audio_vae, backend, mono, audio_vae.config().sample_rate, patch_size, feat_dim);
    prepared.prompt_audio_length =
        static_cast<int>(prepared.prompt_feat.size() / static_cast<size_t>(patch_size * feat_dim));
    prepared.full_text_tokens.resize(text_tokens.size() + static_cast<size_t>(prepared.prompt_audio_length), 0);

    const int seq_len = static_cast<int>(prepared.full_text_tokens.size());
    prepared.feat.assign(static_cast<size_t>(seq_len) * patch_size * feat_dim, 0.0f);
    std::copy(prepared.prompt_feat.begin(),
              prepared.prompt_feat.end(),
              prepared.feat.begin() + static_cast<std::ptrdiff_t>(text_tokens.size()) * patch_size * feat_dim);

    prepared.text_mask.assign(text_tokens.size(), 1);
    prepared.text_mask.resize(seq_len, 0);
    prepared.feat_mask.assign(text_tokens.size(), 0);
    prepared.feat_mask.resize(seq_len, 1);
    return prepared;
}

}  // namespace
}  // namespace voxcpm

int main(int argc, char** argv) {
    using namespace voxcpm;

    try {
        const Options options = parse_args(argc, argv);
        constexpr int kStreamingPrefixLen = 3;
        const bool log_memory = env_flag_enabled("VOXCPM_LOG_MEMORY_BREAKDOWN");

        VoxCPMBackend backend(options.backend, options.threads);
        std::cerr << "Using backend: " << backend_type_name(backend.type())
                  << " (" << backend.backend_name();
        if (std::strlen(backend.backend_description()) > 0) {
            std::cerr << " | " << backend.backend_description();
        }
        std::cerr << ")\n";
        std::cerr << "Loading GGUF from " << options.model_path << " with " << options.threads
                  << " threads...\n";
        auto store = std::make_shared<VoxCPMWeightStore>();
        if (!store->load_from_file(options.model_path, backend)) {
            fail("Failed to load GGUF: " + options.model_path);
        }

        VoxCPMRuntime runtime;
        if (!runtime.load_from_store(store, backend)) {
            fail("Failed to initialize VoxCPM runtime from GGUF");
        }

        AudioVAE audio_vae;
        if (!audio_vae.load_from_store(store)) {
            fail("Failed to initialize AudioVAE from GGUF");
        }

        VoxCPMTokenizer tokenizer;
        if (!tokenizer.load_from_store(*store)) {
            fail("Failed to load tokenizer metadata from GGUF");
        }
        ChineseCharSplitTokenizer split_tokenizer(tokenizer);
        std::cerr << "Tokenizer loaded from GGUF metadata.\n";
        log_memory_breakdown(log_memory, "post_load", *store, backend, nullptr);

        const int patch_size = runtime.config().patch_size;
        const int feat_dim = runtime.config().feat_dim;
        const int patch_len = patch_size * audio_vae.config().hop_length();
        const PreparedInputs prepared = prepare_inputs(
            options, split_tokenizer, audio_vae, backend, patch_size, feat_dim, patch_len);
        log_memory_breakdown(log_memory, "post_prompt_encode", *store, backend, nullptr);
        const int seq_len = static_cast<int>(prepared.full_text_tokens.size());
        const auto synth_start = std::chrono::steady_clock::now();

        std::cerr << "Running prefill, seq_len=" << seq_len << "...\n";
        VoxCPMDecodeState state = runtime.prefill(prepared.full_text_tokens,
                                                 prepared.text_mask,
                                                 prepared.feat,
                                                 prepared.feat_mask,
                                                 seq_len,
                                                 kStreamingPrefixLen);
        log_memory_breakdown(log_memory, "post_prefill", *store, backend, &state);

        const int target_text_token_count =
            std::max<int>(1, static_cast<int>(split_tokenizer.tokenize(options.text).size()));
        const int max_len = std::min(target_text_token_count * 6 + 10, 2000);
        constexpr int kMinLen = 2;

        std::mt19937 rng(std::random_device{}());
        std::vector<float> generated_steps;
        generated_steps.reserve(static_cast<size_t>(max_len) * patch_size * feat_dim);
        std::vector<float> noise;
        std::vector<float> stream_recent_frames;
        std::vector<float> stream_latent;
        if (options.stream) {
            std::filesystem::create_directories(options.stream_dir);
            const size_t frame_stride = static_cast<size_t>(patch_size) * feat_dim;
            const int context_frames = (!prepared.prompt_feat.empty() && prepared.prompt_audio_length > 0 && kStreamingPrefixLen > 1)
                ? std::min(kStreamingPrefixLen - 1, prepared.prompt_audio_length)
                : 0;
            if (context_frames > 0) {
                const size_t context_offset = static_cast<size_t>(prepared.prompt_audio_length - context_frames) * frame_stride;
                stream_recent_frames.insert(stream_recent_frames.end(),
                                            prepared.prompt_feat.begin() + static_cast<std::ptrdiff_t>(context_offset),
                                            prepared.prompt_feat.end());
            }
        }

        std::cerr << "Running decode loop, max_len=" << max_len << "...\n";
        for (int step = 0; step < max_len; ++step) {
            std::cerr << "Decode step " << step << "...\n";
            fill_noise(noise, patch_size, feat_dim, rng);
            VoxCPMDecodeResult result = runtime.decode(std::move(state),
                                                       noise,
                                                       options.inference_timesteps,
                                                       options.cfg_value);
            generated_steps.insert(generated_steps.end(), result.output_0.begin(), result.output_0.end());
            state = std::move(result.output_1);

            if (options.stream) {
                append_stream_frame(stream_recent_frames,
                                    result.output_0,
                                    kStreamingPrefixLen,
                                    patch_size,
                                    feat_dim);
                const int recent_frame_count =
                    static_cast<int>(stream_recent_frames.size() / static_cast<size_t>(patch_size * feat_dim));
                const int recent_patches = recent_frame_count * patch_size;
                if (recent_patches > 0) {
                    patch_major_to_latent(stream_recent_frames, patch_size, feat_dim, stream_latent);
                    std::vector<float> chunk_waveform = decode_audio(audio_vae, backend, stream_latent, recent_patches, feat_dim);
                    if (chunk_waveform.size() > static_cast<size_t>(patch_len)) {
                        chunk_waveform.erase(chunk_waveform.begin(),
                                             chunk_waveform.end() - static_cast<std::ptrdiff_t>(patch_len));
                    }
                    write_wav_pcm16(chunk_output_path(options.stream_dir, step),
                                    chunk_waveform,
                                    audio_vae.config().sample_rate);
                }
            }

            if (step > kMinLen && result.output_2) {
                std::cerr << "Stop token triggered at step " << step << ".\n";
                break;
            }
        }

        const int generated_frames = static_cast<int>(generated_steps.size() / static_cast<size_t>(patch_size * feat_dim));
        int prepended_context_frames = 0;
        const std::vector<float> decode_frames = build_decode_feature_sequence(prepared.prompt_feat,
                                                                               prepared.prompt_audio_length,
                                                                               generated_steps,
                                                                               kStreamingPrefixLen,
                                                                               patch_size,
                                                                               feat_dim,
                                                                               &prepended_context_frames);
        const int total_frames = static_cast<int>(decode_frames.size() / static_cast<size_t>(patch_size * feat_dim));
        const int total_patches = total_frames * patch_size;
        if (generated_frames == 0 || total_patches == 0) {
            fail("Model generated no audio patches");
        }

        const std::vector<float> latent = patch_major_to_latent(decode_frames, patch_size, feat_dim);
        std::vector<float> waveform = decode_audio(audio_vae, backend, latent, total_patches, feat_dim);
        log_memory_breakdown(log_memory, "post_waveform_decode", *store, backend, &state);
        if (prepared.has_prompt_audio) {
            const size_t trim = static_cast<size_t>(patch_len) * static_cast<size_t>(prepended_context_frames);
            if (waveform.size() > trim) {
                waveform.erase(waveform.begin(), waveform.begin() + static_cast<std::ptrdiff_t>(trim));
            }
        }

        write_wav_pcm16(options.output_path, waveform, audio_vae.config().sample_rate);
        const auto synth_end = std::chrono::steady_clock::now();
        const double audio_seconds =
            static_cast<double>(waveform.size()) / static_cast<double>(audio_vae.config().sample_rate);
        const double synth_seconds =
            std::chrono::duration<double>(synth_end - synth_start).count();
        const double rtf = audio_seconds > 0.0 ? (synth_seconds / audio_seconds) : 0.0;
        std::cerr << "Saved audio to " << options.output_path
                  << " (" << audio_seconds << "s)\n";
        std::cerr << std::fixed << std::setprecision(3)
                  << "Synthesis stats: synth_time=" << synth_seconds
                  << "s, audio_time=" << audio_seconds
                  << "s, RTF=" << rtf << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
