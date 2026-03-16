#include "voxcpm/audio-vae.h"
#include "voxcpm/backend.h"
#include "voxcpm/context.h"
#include "voxcpm/tokenizer.h"
#include "voxcpm/voxcpm.h"
#include "voxcpm/weight-store.h"

#include <nlohmann/json.hpp>

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
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <unistd.h>
#include <vector>

namespace voxcpm {
namespace benchmark {

using json = nlohmann::json;
using Clock = std::chrono::steady_clock;

namespace {

constexpr int kStreamingPrefixLen = 3;
constexpr int kShortTextCodepoints = 12;
constexpr int kMediumTextCodepoints = 36;
constexpr int kLongTextCodepoints = 96;
constexpr int kShortPromptFrames = 4;
constexpr int kMediumPromptFrames = 16;
constexpr int kLongPromptFrames = 48;
constexpr uint32_t kBaseSeed = 1337;
constexpr const char* kRunnerVersion = "1.0.0";

struct Options {
    std::string text;
    std::string prompt_audio_path;
    std::string prompt_text;
    std::string model_path;
    std::string output_json;
    BackendType backend = BackendType::CPU;
    int threads = 4;
    int repeat = 8;
    int warmup = 2;
    int inference_timesteps = 10;
    float cfg_value = 2.0f;
    std::string level = "all";
    std::string scenario = "all";
    std::unordered_set<std::string> benchmark_filters;
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

struct ScenarioSpec {
    std::string id;
    std::string text;
    std::vector<float> mono_audio;
};

struct ScenarioData {
    ScenarioSpec spec;
    PreparedInputs prepared;
    int seq_len = 0;
    int prompt_frames = 0;
    int patch_size = 0;
    int feat_dim = 0;
    int patch_len = 0;
    int prompt_patches = 0;
    int prompt_samples = 0;
    std::vector<float> prompt_latent;
};

struct StatsSummary {
    double total_ms = 0.0;
    double mean_ms = 0.0;
    double median_ms = 0.0;
    double p90_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    double stddev_ms = 0.0;
};

struct BenchmarkRecord {
    std::string id;
    std::string level;
    std::string module;
    std::string submodule;
    std::string position_bucket;
    json input_shape;
    std::vector<double> samples_ms;
    StatsSummary stats;
    json derived_metrics;
    json timing_breakdown;
};

struct RuntimeBundle {
    VoxCPMBackend backend;
    std::shared_ptr<VoxCPMWeightStore> store;
    VoxCPMRuntime runtime;
    AudioVAE audio_vae;
    VoxCPMTokenizer tokenizer;
    std::unique_ptr<ChineseCharSplitTokenizer> split_tokenizer;
    double load_model_ms = 0.0;

    RuntimeBundle(BackendType backend_type, int threads)
        : backend(backend_type, threads) {
    }

    RuntimeBundle(const RuntimeBundle&) = delete;
    RuntimeBundle& operator=(const RuntimeBundle&) = delete;
};

struct ProgressTracker {
    int current_case = 0;
    int total_cases = 0;
};

struct DecodePrepared {
    VoxCPMDecodeState target_state;
    std::vector<float> z;
};

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

BackendType parse_backend_type(const std::string& value) {
    if (value == "cpu") return BackendType::CPU;
    if (value == "vulkan") return BackendType::Vulkan;
    if (value == "cuda") return BackendType::CUDA;
    if (value == "metal") return BackendType::Metal;
    if (value == "auto") return BackendType::Auto;
    fail("Unsupported backend: " + value);
}

const char* backend_type_name(BackendType type) {
    switch (type) {
        case BackendType::CPU: return "cpu";
        case BackendType::Vulkan: return "vulkan";
        case BackendType::CUDA: return "cuda";
        case BackendType::Metal: return "metal";
        case BackendType::Auto: return "auto";
        default: return "unknown";
    }
}

void print_usage(const char* argv0) {
    std::cerr << "Usage:\n"
              << "  " << argv0 << " --model-path MODEL.gguf --prompt-audio PROMPT.wav --prompt-text TEXT --text TEXT [options]\n\n"
              << "Options:\n"
              << "  --model-path GGUF\n"
              << "  --prompt-audio WAV\n"
              << "  --prompt-text TEXT\n"
              << "  --text TEXT\n"
              << "  --backend {cpu|vulkan|cuda|metal|auto} (default: cpu)\n"
              << "  --threads INT (default: 4)\n"
              << "  --repeat INT (default: 8)\n"
              << "  --warmup INT (default: 2)\n"
              << "  --level {l1|l2|all} (default: all)\n"
              << "  --scenario {short|medium|long|all} (default: all)\n"
              << "  --benchmarks ID[,ID...] (optional)\n"
              << "  --cfg-value FLOAT (default: 2.0)\n"
              << "  --inference-timesteps INT (default: 10)\n"
              << "  --output-json PATH\n";
}

void add_benchmark_filters(std::unordered_set<std::string>& filters, const std::string& csv) {
    size_t start = 0;
    while (start <= csv.size()) {
        const size_t comma = csv.find(',', start);
        const std::string token = csv.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if (!token.empty()) {
            filters.insert(token);
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
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

        if (arg == "--model-path") {
            options.model_path = require_value("--model-path");
        } else if (arg == "--prompt-audio") {
            options.prompt_audio_path = require_value("--prompt-audio");
        } else if (arg == "--prompt-text") {
            options.prompt_text = require_value("--prompt-text");
        } else if (arg == "--text") {
            options.text = require_value("--text");
        } else if (arg == "--backend") {
            options.backend = parse_backend_type(require_value("--backend"));
        } else if (arg == "--threads") {
            options.threads = std::stoi(require_value("--threads"));
        } else if (arg == "--repeat") {
            options.repeat = std::stoi(require_value("--repeat"));
        } else if (arg == "--warmup") {
            options.warmup = std::stoi(require_value("--warmup"));
        } else if (arg == "--level") {
            options.level = require_value("--level");
        } else if (arg == "--scenario") {
            options.scenario = require_value("--scenario");
        } else if (arg == "--benchmarks") {
            add_benchmark_filters(options.benchmark_filters, require_value("--benchmarks"));
        } else if (arg == "--cfg-value") {
            options.cfg_value = std::stof(require_value("--cfg-value"));
        } else if (arg == "--inference-timesteps") {
            options.inference_timesteps = std::stoi(require_value("--inference-timesteps"));
        } else if (arg == "--output-json") {
            options.output_json = require_value("--output-json");
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            fail("Unknown argument: " + arg);
        }
    }

    if (options.text.empty()) fail("--text is required");
    if (options.prompt_text.empty()) fail("--prompt-text is required");
    if (options.prompt_audio_path.empty()) fail("--prompt-audio is required");
    if (options.model_path.empty()) fail("--model-path is required");
    if (options.threads < 1) fail("--threads must be >= 1");
    if (options.repeat < 1) fail("--repeat must be >= 1");
    if (options.warmup < 0) fail("--warmup must be >= 0");
    if (options.level != "l1" && options.level != "l2" && options.level != "all") {
        fail("--level must be one of l1, l2, all");
    }
    if (options.scenario != "short" && options.scenario != "medium" &&
        options.scenario != "long" && options.scenario != "all") {
        fail("--scenario must be one of short, medium, long, all");
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
    (void)read_le_u32(in);
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
            (void)read_le_u32(in);
            (void)read_le_u16(in);
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
    }

    if (audio_format != 1 || (bits_per_sample != 16 && bits_per_sample != 32) || sample_rate == 0 || num_channels == 0 ||
        data_chunk.empty()) {
        fail("Unsupported WAV format: " + path);
    }

    WavData wav;
    wav.sample_rate = static_cast<int>(sample_rate);
    wav.channels = static_cast<int>(num_channels);

    if (bits_per_sample == 16) {
        const size_t sample_count = data_chunk.size() / sizeof(int16_t);
        wav.samples.resize(sample_count);
        const int16_t* pcm = reinterpret_cast<const int16_t*>(data_chunk.data());
        for (size_t i = 0; i < sample_count; ++i) {
            wav.samples[i] = static_cast<float>(pcm[i]) / 32768.0f;
        }
    } else {
        const size_t sample_count = data_chunk.size() / sizeof(int32_t);
        wav.samples.resize(sample_count);
        const int32_t* pcm = reinterpret_cast<const int32_t*>(data_chunk.data());
        for (size_t i = 0; i < sample_count; ++i) {
            wav.samples[i] = static_cast<float>(pcm[i]) / 2147483648.0f;
        }
    }

    return wav;
}

std::vector<float> convert_to_mono(const WavData& wav) {
    if (wav.channels == 1) {
        return wav.samples;
    }

    const size_t frame_count = wav.samples.size() / static_cast<size_t>(wav.channels);
    std::vector<float> mono(frame_count, 0.0f);
    for (size_t i = 0; i < frame_count; ++i) {
        float sum = 0.0f;
        for (int c = 0; c < wav.channels; ++c) {
            sum += wav.samples[i * static_cast<size_t>(wav.channels) + static_cast<size_t>(c)];
        }
        mono[i] = sum / static_cast<float>(wav.channels);
    }
    return mono;
}

std::vector<float> linear_resample(const std::vector<float>& input, int src_rate, int dst_rate) {
    if (src_rate == dst_rate || input.empty()) {
        return input;
    }

    const double ratio = static_cast<double>(dst_rate) / static_cast<double>(src_rate);
    const size_t output_size = static_cast<size_t>(std::llround(static_cast<double>(input.size()) * ratio));
    std::vector<float> output(output_size, 0.0f);

    for (size_t i = 0; i < output_size; ++i) {
        const double src_pos = static_cast<double>(i) / ratio;
        const size_t left = static_cast<size_t>(std::floor(src_pos));
        const size_t right = std::min(left + 1, input.size() - 1);
        const double alpha = src_pos - static_cast<double>(left);
        output[i] = static_cast<float>((1.0 - alpha) * input[left] + alpha * input[right]);
    }
    return output;
}

std::string utf8_prefix_codepoints(const std::string& text, int max_codepoints) {
    if (max_codepoints <= 0) {
        return "";
    }

    size_t i = 0;
    int count = 0;
    while (i < text.size() && count < max_codepoints) {
        const unsigned char ch = static_cast<unsigned char>(text[i]);
        size_t char_len = 1;
        if ((ch & 0x80u) == 0) {
            char_len = 1;
        } else if ((ch & 0xE0u) == 0xC0u) {
            char_len = 2;
        } else if ((ch & 0xF0u) == 0xE0u) {
            char_len = 3;
        } else if ((ch & 0xF8u) == 0xF0u) {
            char_len = 4;
        }
        i += char_len;
        ++count;
    }
    return text.substr(0, i);
}

int count_utf8_codepoints(const std::string& text) {
    int count = 0;
    for (unsigned char ch : text) {
        if ((ch & 0xC0u) != 0x80u) {
            ++count;
        }
    }
    return count;
}

std::string make_text_variant(const std::string& base, int target_codepoints) {
    if (base.empty()) {
        return "benchmark";
    }

    if (count_utf8_codepoints(base) >= target_codepoints) {
        return utf8_prefix_codepoints(base, target_codepoints);
    }

    std::string output;
    while (count_utf8_codepoints(output) < target_codepoints) {
        if (!output.empty()) {
            output.push_back(' ');
        }
        output += base;
    }
    return utf8_prefix_codepoints(output, target_codepoints);
}

std::vector<float> make_prompt_audio_variant(const std::vector<float>& mono, int patch_len, int target_frames) {
    const size_t target_samples = static_cast<size_t>(patch_len) * static_cast<size_t>(target_frames);
    std::vector<float> output;
    output.reserve(target_samples);
    if (mono.size() >= target_samples) {
        output.insert(output.end(), mono.begin(), mono.begin() + static_cast<std::ptrdiff_t>(target_samples));
        return output;
    }

    output = mono;
    output.resize(target_samples, 0.0f);
    return output;
}

std::vector<float> extract_prompt_features(AudioVAE& audio_vae,
                                           VoxCPMBackend& backend,
                                           std::vector<float> audio,
                                           int sample_rate,
                                           int patch_size,
                                           int feat_dim) {
    VoxCPMContext graph_ctx(ContextType::Graph, 32768, 262144);
    ggml_tensor* latent = audio_vae.encode(graph_ctx, audio, sample_rate);
    if (!latent) {
        fail("Failed to build AudioVAE encode graph");
    }

    ggml_cgraph* graph = graph_ctx.new_graph();
    graph_ctx.build_forward(graph, latent);
    backend.reserve_compute_memory(graph, "benchmark.audio_vae.encode.prompt");
    backend.alloc_graph(graph, "benchmark.audio_vae.encode.prompt");
    const auto& preprocessed = audio_vae.last_preprocessed_audio();
    backend.tensor_set(audio_vae.last_input_tensor(), preprocessed.data(), 0, preprocessed.size() * sizeof(float));
    if (backend.compute(graph) != GGML_STATUS_SUCCESS) {
        fail("AudioVAE encode failed");
    }

    const int total_patches = static_cast<int>(latent->ne[0]);
    const int latent_dim = static_cast<int>(latent->ne[1]);
    if (latent_dim != feat_dim || total_patches % patch_size != 0) {
        fail("Prompt latent shape mismatch");
    }

    std::vector<float> encoded(static_cast<size_t>(total_patches) * static_cast<size_t>(latent_dim));
    backend.tensor_get(latent, encoded.data(), 0, encoded.size() * sizeof(float));

    const int audio_length = total_patches / patch_size;
    std::vector<float> features(static_cast<size_t>(audio_length) * static_cast<size_t>(patch_size) * static_cast<size_t>(feat_dim), 0.0f);
    for (int t = 0; t < audio_length; ++t) {
        for (int p = 0; p < patch_size; ++p) {
            const int patch_index = t * patch_size + p;
            for (int d = 0; d < feat_dim; ++d) {
                const size_t src = static_cast<size_t>(d) * static_cast<size_t>(total_patches) + static_cast<size_t>(patch_index);
                const size_t dst = (static_cast<size_t>(t) * static_cast<size_t>(patch_size) + static_cast<size_t>(p)) *
                                   static_cast<size_t>(feat_dim) + static_cast<size_t>(d);
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
    VoxCPMContext graph_ctx(ContextType::Graph, 32768, 262144);
    ggml_tensor* latent = graph_ctx.new_tensor_2d(GGML_TYPE_F32, total_patches, feat_dim);
    ggml_set_input(latent);
    ggml_tensor* audio = audio_vae.decode(graph_ctx, latent);
    if (!audio) {
        fail("Failed to build AudioVAE decode graph");
    }

    ggml_cgraph* graph = graph_ctx.new_graph();
    graph_ctx.build_forward(graph, audio);
    backend.reserve_compute_memory(graph, "benchmark.audio_vae.decode");
    backend.alloc_graph(graph, "benchmark.audio_vae.decode");
    backend.tensor_set(latent, features.data(), 0, features.size() * sizeof(float));
    if (backend.compute(graph) != GGML_STATUS_SUCCESS) {
        fail("AudioVAE decode failed");
    }

    std::vector<float> waveform(static_cast<size_t>(ggml_nelements(audio)));
    backend.tensor_get(audio, waveform.data(), 0, waveform.size() * sizeof(float));
    return waveform;
}

void patch_major_to_latent(const std::vector<float>& frames,
                           int patch_size,
                           int feat_dim,
                           std::vector<float>& latent) {
    const size_t frame_stride = static_cast<size_t>(patch_size) * static_cast<size_t>(feat_dim);
    const int total_frames = static_cast<int>(frames.size() / frame_stride);
    const int total_patches = total_frames * patch_size;
    latent.assign(static_cast<size_t>(total_patches) * static_cast<size_t>(feat_dim), 0.0f);
    for (int frame = 0; frame < total_frames; ++frame) {
        for (int patch = 0; patch < patch_size; ++patch) {
            const int time_index = frame * patch_size + patch;
            for (int d = 0; d < feat_dim; ++d) {
                const size_t src = (static_cast<size_t>(frame) * static_cast<size_t>(patch_size) + static_cast<size_t>(patch)) *
                                   static_cast<size_t>(feat_dim) + static_cast<size_t>(d);
                const size_t dst = static_cast<size_t>(d) * static_cast<size_t>(total_patches) + static_cast<size_t>(time_index);
                latent[dst] = frames[src];
            }
        }
    }
}

std::vector<float> patch_major_to_latent(const std::vector<float>& frames, int patch_size, int feat_dim) {
    std::vector<float> latent;
    patch_major_to_latent(frames, patch_size, feat_dim, latent);
    return latent;
}

void fill_noise(std::vector<float>& noise, int patch_size, int feat_dim, std::mt19937& rng) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    noise.resize(static_cast<size_t>(patch_size) * static_cast<size_t>(feat_dim));
    for (float& value : noise) {
        value = dist(rng);
    }
}

PreparedInputs prepare_inputs(const ScenarioSpec& scenario,
                              ChineseCharSplitTokenizer& split_tokenizer,
                              AudioVAE& audio_vae,
                              VoxCPMBackend& backend,
                              int patch_size,
                              int feat_dim,
                              int patch_len,
                              const std::string& prompt_text) {
    PreparedInputs prepared;
    std::vector<int32_t> text_tokens = split_tokenizer.encode(prompt_text + scenario.text, false);
    text_tokens.push_back(101);

    prepared.has_prompt_audio = true;
    std::vector<float> mono = scenario.mono_audio;
    if (mono.size() % static_cast<size_t>(patch_len) != 0) {
        const size_t padding = static_cast<size_t>(patch_len) - (mono.size() % static_cast<size_t>(patch_len));
        mono.insert(mono.begin(), padding, 0.0f);
    }

    prepared.prompt_feat = extract_prompt_features(audio_vae,
                                                   backend,
                                                   mono,
                                                   audio_vae.config().sample_rate,
                                                   patch_size,
                                                   feat_dim);
    prepared.prompt_audio_length =
        static_cast<int>(prepared.prompt_feat.size() / static_cast<size_t>(patch_size * feat_dim));
    prepared.full_text_tokens = text_tokens;
    prepared.full_text_tokens.resize(text_tokens.size() + static_cast<size_t>(prepared.prompt_audio_length), 0);

    const int seq_len = static_cast<int>(prepared.full_text_tokens.size());
    prepared.feat.assign(static_cast<size_t>(seq_len) * static_cast<size_t>(patch_size) * static_cast<size_t>(feat_dim), 0.0f);
    std::copy(prepared.prompt_feat.begin(),
              prepared.prompt_feat.end(),
              prepared.feat.begin() + static_cast<std::ptrdiff_t>(text_tokens.size()) * patch_size * feat_dim);

    prepared.text_mask.assign(text_tokens.size(), 1);
    prepared.text_mask.resize(seq_len, 0);
    prepared.feat_mask.assign(text_tokens.size(), 0);
    prepared.feat_mask.resize(seq_len, 1);
    return prepared;
}

template <typename Fn>
std::vector<double> measure_samples(int warmup, int repeat, Fn&& fn) {
    for (int i = 0; i < warmup; ++i) {
        fn();
    }

    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
        const auto start = Clock::now();
        fn();
        const auto end = Clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }
    return samples;
}

double compute_quantile(std::vector<double> values, double q) {
    if (values.empty()) {
        return 0.0;
    }
    const size_t index = static_cast<size_t>(std::floor(q * static_cast<double>(values.size() - 1)));
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(index), values.end());
    return values[index];
}

StatsSummary summarize(const std::vector<double>& samples) {
    StatsSummary out;
    if (samples.empty()) {
        return out;
    }

    out.total_ms = std::accumulate(samples.begin(), samples.end(), 0.0);
    out.mean_ms = out.total_ms / static_cast<double>(samples.size());
    out.median_ms = compute_quantile(samples, 0.5);
    out.p90_ms = compute_quantile(samples, 0.9);
    const auto [min_it, max_it] = std::minmax_element(samples.begin(), samples.end());
    out.min_ms = *min_it;
    out.max_ms = *max_it;
    double sq_sum = 0.0;
    for (double sample : samples) {
        const double diff = sample - out.mean_ms;
        sq_sum += diff * diff;
    }
    out.stddev_ms = std::sqrt(sq_sum / static_cast<double>(samples.size()));
    return out;
}

json stats_to_json(const StatsSummary& stats) {
    return json{
        {"total_ms", stats.total_ms},
        {"mean_ms", stats.mean_ms},
        {"median_ms", stats.median_ms},
        {"p90_ms", stats.p90_ms},
        {"min_ms", stats.min_ms},
        {"max_ms", stats.max_ms},
        {"stddev_ms", stats.stddev_ms},
    };
}

std::string current_timestamp_string() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&now_c);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
    return oss.str();
}

std::string format_double(double value, int width = 10, int precision = 3) {
    std::ostringstream oss;
    oss << std::fixed << std::setw(width) << std::setprecision(precision) << value;
    return oss.str();
}

void log_progress(const std::string& message) {
    std::cout << "[progress] " << message << std::endl;
}

std::string format_case_progress(const ProgressTracker& progress) {
    std::ostringstream oss;
    if (progress.total_cases <= 0) {
        oss << "[case ?/? |   0.0%] ";
        return oss.str();
    }

    const int current_display = std::min(progress.current_case + 1, progress.total_cases);
    const double percent = (100.0 * static_cast<double>(progress.current_case)) /
                           static_cast<double>(progress.total_cases);
    oss << "[case " << current_display << "/" << progress.total_cases
        << " | " << std::fixed << std::setw(5) << std::setprecision(1) << percent << "%] ";
    return oss.str();
}

std::string describe_samples(int warmup, int repeat) {
    std::ostringstream oss;
    oss << "warmup=" << warmup << ", repeat=" << repeat;
    return oss.str();
}

bool benchmark_selected(const Options& options, const std::string& case_id) {
    if (options.benchmark_filters.empty()) {
        return true;
    }

    for (const std::string& filter : options.benchmark_filters) {
        if (case_id == filter) {
            return true;
        }
        if (case_id.size() > filter.size() &&
            case_id.compare(0, filter.size(), filter) == 0 &&
            case_id[filter.size()] == '.') {
            return true;
        }
    }
    return false;
}

template <typename Fn>
std::vector<double> run_benchmark_case(const std::string& case_id,
                                       int warmup,
                                       int repeat,
                                       ProgressTracker& progress,
                                       Fn&& fn,
                                       const std::string& detail = std::string()) {
    std::ostringstream oss;
    oss << format_case_progress(progress) << "Running " << case_id << " (" << describe_samples(warmup, repeat);
    if (!detail.empty()) {
        oss << ", " << detail;
    }
    oss << ")";
    log_progress(oss.str());
    std::vector<double> samples = measure_samples(warmup, repeat, std::forward<Fn>(fn));
    const StatsSummary stats = summarize(samples);
    std::ostringstream done;
    progress.current_case += 1;
    const double percent = progress.total_cases > 0
        ? (100.0 * static_cast<double>(progress.current_case)) / static_cast<double>(progress.total_cases)
        : 100.0;
    done << "[case " << progress.current_case << "/" << progress.total_cases
         << " | " << std::fixed << std::setw(5) << std::setprecision(1) << percent << "%] "
         << "Finished " << case_id << " mean_ms=" << std::fixed << std::setprecision(3) << stats.mean_ms;
    log_progress(done.str());
    return samples;
}

int cases_per_scenario_for_level(const std::string& level) {
    int total = 0;
    if (level == "l1" || level == "all") total += 8;
    if (level == "l2" || level == "all") total += 30;
    return total;
}

int selected_cases_per_scenario(const Options& options) {
    const std::vector<std::string> case_ids = {
        "audio_vae.encode",
        "audio_vae.decode",
        "voxcpm.prefill",
        "voxcpm.decode_step.early",
        "voxcpm.decode_step.mid",
        "voxcpm.decode_step.late",
        "voxcpm.decode_loop_total",
        "tts.e2e_total",
        "prefill.locenc_all",
        "prefill.enc_to_lm_proj",
        "prefill.text_embedding",
        "prefill.base_lm",
        "prefill.fsq",
        "prefill.residual_lm",
        "decode.front_half_total.early",
        "decode.front_half_total.mid",
        "decode.front_half_total.late",
        "decode.lm_to_dit_proj.early",
        "decode.lm_to_dit_proj.mid",
        "decode.lm_to_dit_proj.late",
        "decode.res_to_dit_proj.early",
        "decode.res_to_dit_proj.mid",
        "decode.res_to_dit_proj.late",
        "decode.unified_cfm.early",
        "decode.unified_cfm.mid",
        "decode.unified_cfm.late",
        "decode.stop_predictor.early",
        "decode.stop_predictor.mid",
        "decode.stop_predictor.late",
        "decode.locenc_patch_to_lm.early",
        "decode.locenc_patch_to_lm.mid",
        "decode.locenc_patch_to_lm.late",
        "decode.base_lm_step_fsq.early",
        "decode.base_lm_step_fsq.mid",
        "decode.base_lm_step_fsq.late",
        "decode.residual_lm_step.early",
        "decode.residual_lm_step.mid",
        "decode.residual_lm_step.late",
    };

    int total = 0;
    for (const std::string& case_id : case_ids) {
        const bool is_l1 = case_id.rfind("audio_vae.", 0) == 0 ||
                           case_id.rfind("voxcpm.", 0) == 0 ||
                           case_id == "tts.e2e_total";
        const bool level_ok = (options.level == "all") ||
                              (options.level == "l1" && is_l1) ||
                              (options.level == "l2" && !is_l1);
        if (level_ok && benchmark_selected(options, case_id)) {
            total += 1;
        }
    }
    return total;
}

std::string discover_git_commit() {
    namespace fs = std::filesystem;

    fs::path path = fs::path(__FILE__).parent_path();
    for (int i = 0; i < 5; ++i) {
        const fs::path git_dir = path / ".git";
        if (fs::exists(git_dir)) {
            std::ifstream head(git_dir / "HEAD");
            std::string head_ref;
            std::getline(head, head_ref);
            if (head_ref.rfind("ref: ", 0) == 0) {
                const fs::path ref_path = git_dir / head_ref.substr(5);
                std::ifstream ref_file(ref_path);
                std::string commit;
                std::getline(ref_file, commit);
                return commit.empty() ? "unknown" : commit;
            }
            return head_ref.empty() ? "unknown" : head_ref;
        }
        path = path.parent_path();
    }
    return "unknown";
}

std::string discover_hostname() {
    char hostname[256] = {0};
    if (gethostname(hostname, sizeof(hostname) - 1) == 0) {
        return hostname;
    }
    return "unknown";
}

std::string default_output_json_path() {
    namespace fs = std::filesystem;
    const fs::path path = fs::path(__FILE__).parent_path() / "results" /
                          ("voxcpm_benchmark_" + current_timestamp_string() + ".json");
    return path.string();
}

void load_runtime_bundle(const Options& options, RuntimeBundle& bundle) {
    log_progress("Loading model and runtime");
    const auto start = Clock::now();
    bundle.store = std::make_shared<VoxCPMWeightStore>();
    if (!bundle.store->load_from_file(options.model_path, bundle.backend)) {
        fail("Failed to load GGUF: " + options.model_path);
    }
    if (!bundle.runtime.load_from_store(bundle.store, bundle.backend)) {
        fail("Failed to initialize VoxCPM runtime");
    }
    if (!bundle.audio_vae.load_from_store(bundle.store)) {
        fail("Failed to initialize AudioVAE");
    }
    if (!bundle.tokenizer.load_from_store(*bundle.store)) {
        fail("Failed to load tokenizer metadata");
    }
    bundle.split_tokenizer = std::make_unique<ChineseCharSplitTokenizer>(bundle.tokenizer);
    const auto end = Clock::now();
    bundle.load_model_ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::ostringstream oss;
    oss << "Loaded model in " << std::fixed << std::setprecision(3) << bundle.load_model_ms
        << " ms using backend=" << backend_type_name(options.backend)
        << " threads=" << options.threads;
    log_progress(oss.str());
}

std::vector<ScenarioSpec> build_scenarios(const Options& options, int patch_len, int sample_rate) {
    const WavData wav = read_wav_file(options.prompt_audio_path);
    std::vector<float> mono = convert_to_mono(wav);
    mono = linear_resample(mono, wav.sample_rate, sample_rate);

    const std::vector<ScenarioSpec> all_scenarios = {
        {"short", make_text_variant(options.text, kShortTextCodepoints), make_prompt_audio_variant(mono, patch_len, kShortPromptFrames)},
        {"medium", make_text_variant(options.text, kMediumTextCodepoints), make_prompt_audio_variant(mono, patch_len, kMediumPromptFrames)},
        {"long", make_text_variant(options.text, kLongTextCodepoints), make_prompt_audio_variant(mono, patch_len, kLongPromptFrames)},
    };

    if (options.scenario == "all") {
        return all_scenarios;
    }

    for (const auto& scenario : all_scenarios) {
        if (scenario.id == options.scenario) {
            return {scenario};
        }
    }
    fail("Unsupported scenario selection");
}

ScenarioData build_scenario_data(const ScenarioSpec& spec,
                                 RuntimeBundle& bundle,
                                 const Options& options) {
    log_progress("Preparing scenario " + spec.id);
    ScenarioData data;
    data.spec = spec;
    data.patch_size = bundle.runtime.config().patch_size;
    data.feat_dim = bundle.runtime.config().feat_dim;
    data.patch_len = data.patch_size * bundle.audio_vae.config().hop_length();
    data.prompt_samples = static_cast<int>(spec.mono_audio.size());
    data.prepared = prepare_inputs(spec,
                                   *bundle.split_tokenizer,
                                   bundle.audio_vae,
                                   bundle.backend,
                                   data.patch_size,
                                   data.feat_dim,
                                   data.patch_len,
                                   options.prompt_text);
    data.seq_len = static_cast<int>(data.prepared.full_text_tokens.size());
    data.prompt_frames = data.prepared.prompt_audio_length;
    data.prompt_patches = data.prompt_frames * data.patch_size;
    data.prompt_latent = patch_major_to_latent(data.prepared.prompt_feat, data.patch_size, data.feat_dim);
    std::ostringstream oss;
    oss << "Prepared scenario " << spec.id
        << " seq_len=" << data.seq_len
        << " prompt_frames=" << data.prompt_frames
        << " prompt_samples=" << data.prompt_samples;
    log_progress(oss.str());
    return data;
}

std::vector<float> combine_embeds(const std::vector<float>& text_embed,
                                  const std::vector<float>& feat_embed,
                                  const std::vector<int32_t>& text_mask,
                                  const std::vector<int32_t>& feat_mask,
                                  int seq_len,
                                  int hidden_size) {
    std::vector<float> combined(static_cast<size_t>(hidden_size) * static_cast<size_t>(seq_len), 0.0f);
    for (int t = 0; t < seq_len; ++t) {
        const float text_scale = text_mask[static_cast<size_t>(t)] != 0 ? 1.0f : 0.0f;
        const float feat_scale = feat_mask[static_cast<size_t>(t)] != 0 ? 1.0f : 0.0f;
        for (int h = 0; h < hidden_size; ++h) {
            const size_t idx = static_cast<size_t>(t) * static_cast<size_t>(hidden_size) + static_cast<size_t>(h);
            combined[idx] = text_scale * text_embed[idx] + feat_scale * feat_embed[idx];
        }
    }
    return combined;
}

std::vector<float> make_residual_inputs(std::vector<float> enc_outputs,
                                        const std::vector<float>& feat_embed,
                                        const std::vector<int32_t>& feat_mask,
                                        int seq_len,
                                        int hidden_size) {
    for (int t = 0; t < seq_len; ++t) {
        const float feat_scale = feat_mask[static_cast<size_t>(t)] != 0 ? 1.0f : 0.0f;
        for (int h = 0; h < hidden_size; ++h) {
            const size_t idx = static_cast<size_t>(t) * static_cast<size_t>(hidden_size) + static_cast<size_t>(h);
            enc_outputs[idx] += feat_scale * feat_embed[idx];
        }
    }
    return enc_outputs;
}

std::vector<float> add_vectors(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        fail("Vector size mismatch");
    }
    std::vector<float> out(a.size(), 0.0f);
    for (size_t i = 0; i < a.size(); ++i) {
        out[i] = a[i] + b[i];
    }
    return out;
}

DecodePrepared prepare_decode_bucket(RuntimeBundle& bundle,
                                     const ScenarioData& scenario,
                                     const Options& options,
                                     int warm_steps,
                                     uint32_t seed_offset) {
    DecodePrepared prepared;
    prepared.target_state = bundle.runtime.prefill(scenario.prepared.full_text_tokens,
                                                   scenario.prepared.text_mask,
                                                   scenario.prepared.feat,
                                                   scenario.prepared.feat_mask,
                                                   scenario.seq_len,
                                                   kStreamingPrefixLen);

    std::mt19937 rng(kBaseSeed + seed_offset);
    std::vector<float> noise;
    for (int step = 0; step < warm_steps; ++step) {
        fill_noise(noise, scenario.patch_size, scenario.feat_dim, rng);
        VoxCPMDecodeResult result = bundle.runtime.decode(std::move(prepared.target_state),
                                                          noise,
                                                          options.inference_timesteps,
                                                          options.cfg_value);
        prepared.target_state = std::move(result.output_1);
    }

    fill_noise(prepared.z, scenario.patch_size, scenario.feat_dim, rng);
    return prepared;
}

BenchmarkRecord make_record(std::string id,
                            std::string level,
                            std::string module,
                            std::string submodule,
                            std::string position_bucket,
                            json input_shape,
                            std::vector<double> samples,
                            json derived_metrics = json::object()) {
    BenchmarkRecord record;
    record.id = std::move(id);
    record.level = std::move(level);
    record.module = std::move(module);
    record.submodule = std::move(submodule);
    record.position_bucket = std::move(position_bucket);
    record.input_shape = std::move(input_shape);
    record.samples_ms = std::move(samples);
    record.stats = summarize(record.samples_ms);
    record.derived_metrics = std::move(derived_metrics);
    record.timing_breakdown = json{
        {"build_graph_ms", nullptr},
        {"alloc_ms", nullptr},
        {"input_copy_ms", nullptr},
        {"compute_ms", nullptr},
        {"output_copy_ms", nullptr},
    };
    return record;
}

json record_to_json(const BenchmarkRecord& record) {
    return json{
        {"id", record.id},
        {"level", record.level},
        {"module", record.module},
        {"submodule", record.submodule},
        {"position_bucket", record.position_bucket},
        {"input_shape", record.input_shape},
        {"samples", record.samples_ms},
        {"stats", stats_to_json(record.stats)},
        {"derived_metrics", record.derived_metrics},
        {"timing_breakdown", record.timing_breakdown},
    };
}

void print_summary_table(const std::string& scenario_id, const std::vector<BenchmarkRecord>& records) {
    std::cout << "\nScenario: " << scenario_id << "\n";
    std::cout << std::left << std::setw(34) << "Benchmark"
              << std::right << std::setw(12) << "mean_ms"
              << std::setw(12) << "median_ms"
              << std::setw(12) << "p90_ms"
              << "  notes\n";
    std::cout << std::string(82, '-') << "\n";

    for (const auto& record : records) {
        std::ostringstream notes;
        if (record.derived_metrics.contains("steps_per_s")) {
            notes << "steps/s=" << std::fixed << std::setprecision(2) << record.derived_metrics["steps_per_s"].get<double>();
        } else if (record.derived_metrics.contains("patches_per_s")) {
            notes << "patches/s=" << std::fixed << std::setprecision(2) << record.derived_metrics["patches_per_s"].get<double>();
        } else if (record.derived_metrics.contains("seq_len_per_s")) {
            notes << "seq/s=" << std::fixed << std::setprecision(2) << record.derived_metrics["seq_len_per_s"].get<double>();
        } else if (record.derived_metrics.contains("samples_per_s")) {
            notes << "samples/s=" << std::fixed << std::setprecision(2) << record.derived_metrics["samples_per_s"].get<double>();
        }

        std::cout << std::left << std::setw(34) << record.id
                  << std::right << format_double(record.stats.mean_ms)
                  << format_double(record.stats.median_ms)
                  << format_double(record.stats.p90_ms)
                  << "  " << notes.str() << "\n";
    }
}

void append_l1_records(std::vector<BenchmarkRecord>& records,
                       RuntimeBundle& bundle,
                       const ScenarioData& scenario,
                       const Options& options,
                       ProgressTracker& progress) {
    const int hidden_size = bundle.runtime.base_lm().config().hidden_size;

    if (options.benchmark_filters.empty() || benchmark_selected(options, "setup.load_model_ms")) {
        records.push_back(make_record(
            "setup.load_model_ms",
            "overview",
            "setup",
            "load_model",
            "",
            json::object(),
            {bundle.load_model_ms}));
    }

    if (benchmark_selected(options, "audio_vae.encode")) {
    const auto encode_samples = run_benchmark_case("audio_vae.encode",
                                                   options.warmup,
                                                   options.repeat,
                                                   progress,
                                                   [&]() {
        std::vector<float> audio = scenario.spec.mono_audio;
        VoxCPMContext graph_ctx(ContextType::Graph, 32768, 262144);
        ggml_tensor* latent = bundle.audio_vae.encode(graph_ctx, audio, bundle.audio_vae.config().sample_rate);
        ggml_cgraph* graph = graph_ctx.new_graph();
        graph_ctx.build_forward(graph, latent);
        bundle.backend.reserve_compute_memory(graph, "benchmark.audio_vae.encode");
        bundle.backend.alloc_graph(graph, "benchmark.audio_vae.encode");
        const auto& preprocessed = bundle.audio_vae.last_preprocessed_audio();
        bundle.backend.tensor_set(bundle.audio_vae.last_input_tensor(), preprocessed.data(), 0, preprocessed.size() * sizeof(float));
        if (bundle.backend.compute(graph) != GGML_STATUS_SUCCESS) {
            fail("AudioVAE encode failed");
        }
        std::vector<float> scratch(static_cast<size_t>(ggml_nelements(latent)));
        bundle.backend.tensor_get(latent, scratch.data(), 0, scratch.size() * sizeof(float));
    },
                                                   "scenario=" + scenario.spec.id);
    records.push_back(make_record(
        "audio_vae.encode",
        "l1",
        "audio_vae",
        "encode",
        "",
        json{{"prompt_samples", scenario.prompt_samples}, {"prompt_patches", scenario.prompt_patches}},
        encode_samples,
        json{{"patches_per_s", scenario.prompt_patches / (summarize(encode_samples).mean_ms / 1000.0)}}));
    }

    if (benchmark_selected(options, "audio_vae.decode")) {
    const auto decode_samples = run_benchmark_case("audio_vae.decode",
                                                   options.warmup,
                                                   options.repeat,
                                                   progress,
                                                   [&]() {
        std::vector<float> waveform = decode_audio(bundle.audio_vae,
                                                   bundle.backend,
                                                   scenario.prompt_latent,
                                                   scenario.prompt_patches,
                                                   scenario.feat_dim);
        (void)waveform;
    },
                                                   "scenario=" + scenario.spec.id);
    const double decode_mean = summarize(decode_samples).mean_ms;
    records.push_back(make_record(
        "audio_vae.decode",
        "l1",
        "audio_vae",
        "decode",
        "",
        json{{"prompt_patches", scenario.prompt_patches}, {"prompt_samples", scenario.prompt_samples}},
        decode_samples,
        json{
            {"samples_per_s", scenario.prompt_samples / (decode_mean / 1000.0)},
            {"rtf", (decode_mean / 1000.0) /
                        (static_cast<double>(scenario.prompt_samples) / static_cast<double>(bundle.audio_vae.config().sample_rate))}
        }));
    }

    if (benchmark_selected(options, "voxcpm.prefill")) {
    const auto prefill_samples = run_benchmark_case("voxcpm.prefill",
                                                    options.warmup,
                                                    options.repeat,
                                                    progress,
                                                    [&]() {
        VoxCPMDecodeState state = bundle.runtime.prefill(scenario.prepared.full_text_tokens,
                                                        scenario.prepared.text_mask,
                                                        scenario.prepared.feat,
                                                        scenario.prepared.feat_mask,
                                                        scenario.seq_len,
                                                        kStreamingPrefixLen);
        (void)state;
    },
                                                    "scenario=" + scenario.spec.id);
    const double prefill_mean = summarize(prefill_samples).mean_ms;
    records.push_back(make_record(
        "voxcpm.prefill",
        "l1",
        "voxcpm",
        "prefill",
        "",
        json{{"seq_len", scenario.seq_len}, {"hidden_size", hidden_size}},
        prefill_samples,
        json{{"seq_len_per_s", scenario.seq_len / (prefill_mean / 1000.0)}}));
    }

    struct BucketSpec { const char* name; int warm_steps; uint32_t seed_offset; };
    const std::vector<BucketSpec> buckets = {
        {"early", 0, 10},
        {"mid", 64, 20},
        {"late", 256, 30},
    };

    for (const auto& bucket : buckets) {
        const std::string case_id = "voxcpm.decode_step." + std::string(bucket.name);
        if (!benchmark_selected(options, case_id)) {
            continue;
        }
        const DecodePrepared decode_prepared = prepare_decode_bucket(bundle, scenario, options, bucket.warm_steps, bucket.seed_offset);
        const auto decode_step_samples = run_benchmark_case(case_id,
                                                            options.warmup,
                                                            options.repeat,
                                                            progress,
                                                            [&]() {
            VoxCPMDecodeState state = bundle.runtime.benchmark_clone_state(decode_prepared.target_state);
            VoxCPMDecodeResult result = bundle.runtime.decode(std::move(state),
                                                              decode_prepared.z,
                                                              options.inference_timesteps,
                                                              options.cfg_value);
            (void)result;
        },
                                                            "scenario=" + scenario.spec.id);
        const double decode_step_mean = summarize(decode_step_samples).mean_ms;
        records.push_back(make_record(
            "voxcpm.decode_step",
            "l1",
            "voxcpm",
            "decode_step",
            bucket.name,
            json{{"position", scenario.seq_len + bucket.warm_steps + 1}, {"patch_size", scenario.patch_size}, {"feat_dim", scenario.feat_dim}},
            decode_step_samples,
            json{
                {"steps_per_s", 1000.0 / decode_step_mean},
                {"patches_per_s", scenario.patch_size / (decode_step_mean / 1000.0)}
            }));
    }

    if (benchmark_selected(options, "voxcpm.decode_loop_total")) {
    const auto decode_loop_samples = run_benchmark_case("voxcpm.decode_loop_total",
                                                        options.warmup,
                                                        options.repeat,
                                                        progress,
                                                        [&]() {
        VoxCPMDecodeState state = bundle.runtime.prefill(scenario.prepared.full_text_tokens,
                                                        scenario.prepared.text_mask,
                                                        scenario.prepared.feat,
                                                        scenario.prepared.feat_mask,
                                                        scenario.seq_len,
                                                        kStreamingPrefixLen);
        const int target_text_token_count =
            std::max<int>(1, static_cast<int>(bundle.split_tokenizer->tokenize(scenario.spec.text).size()));
        const int max_len = std::min(target_text_token_count * 6 + 10, 2000);
        std::mt19937 rng(kBaseSeed + 500);
        std::vector<float> noise;
        for (int step = 0; step < max_len; ++step) {
            fill_noise(noise, scenario.patch_size, scenario.feat_dim, rng);
            VoxCPMDecodeResult result = bundle.runtime.decode(std::move(state),
                                                              noise,
                                                              options.inference_timesteps,
                                                              options.cfg_value);
            state = std::move(result.output_1);
            if (step > 2 && result.output_2) {
                break;
            }
        }
    },
                                                        "scenario=" + scenario.spec.id);
    records.push_back(make_record(
        "voxcpm.decode_loop_total",
        "overview",
        "voxcpm",
        "decode_loop_total",
        "",
        json{{"seq_len", scenario.seq_len}},
        decode_loop_samples));
    }

    if (benchmark_selected(options, "tts.e2e_total")) {
    const auto e2e_samples = run_benchmark_case("tts.e2e_total",
                                                options.warmup,
                                                options.repeat,
                                                progress,
                                                [&]() {
        std::vector<float> audio = scenario.spec.mono_audio;
        const std::vector<float> prompt_feat = extract_prompt_features(bundle.audio_vae,
                                                                       bundle.backend,
                                                                       audio,
                                                                       bundle.audio_vae.config().sample_rate,
                                                                       scenario.patch_size,
                                                                       scenario.feat_dim);
        PreparedInputs prepared = scenario.prepared;
        prepared.prompt_feat = prompt_feat;
        prepared.feat.assign(scenario.prepared.feat.size(), 0.0f);
        std::copy(prompt_feat.begin(),
                  prompt_feat.end(),
                  prepared.feat.begin() + static_cast<std::ptrdiff_t>(prepared.full_text_tokens.size() - prepared.prompt_audio_length) *
                                            scenario.patch_size * scenario.feat_dim);

        VoxCPMDecodeState state = bundle.runtime.prefill(prepared.full_text_tokens,
                                                        prepared.text_mask,
                                                        prepared.feat,
                                                        prepared.feat_mask,
                                                        scenario.seq_len,
                                                        kStreamingPrefixLen);
        const int target_text_token_count =
            std::max<int>(1, static_cast<int>(bundle.split_tokenizer->tokenize(scenario.spec.text).size()));
        const int max_len = std::min(target_text_token_count * 6 + 10, 2000);

        std::mt19937 rng(kBaseSeed + 600);
        std::vector<float> generated_steps;
        std::vector<float> noise;
        generated_steps.reserve(static_cast<size_t>(max_len) * static_cast<size_t>(scenario.patch_size) * static_cast<size_t>(scenario.feat_dim));
        for (int step = 0; step < max_len; ++step) {
            fill_noise(noise, scenario.patch_size, scenario.feat_dim, rng);
            VoxCPMDecodeResult result = bundle.runtime.decode(std::move(state),
                                                              noise,
                                                              options.inference_timesteps,
                                                              options.cfg_value);
            generated_steps.insert(generated_steps.end(), result.output_0.begin(), result.output_0.end());
            state = std::move(result.output_1);
            if (step > 2 && result.output_2) {
                break;
            }
        }

        const std::vector<float> latent = patch_major_to_latent(generated_steps, scenario.patch_size, scenario.feat_dim);
        if (!latent.empty()) {
            std::vector<float> waveform = decode_audio(bundle.audio_vae,
                                                       bundle.backend,
                                                       latent,
                                                       static_cast<int>(generated_steps.size() / static_cast<size_t>(scenario.feat_dim)),
                                                       scenario.feat_dim);
                (void)waveform;
        }
    },
                                                "scenario=" + scenario.spec.id);
    records.push_back(make_record(
        "tts.e2e_total",
        "overview",
        "tts",
        "e2e_total",
        "",
        json{{"scenario", scenario.spec.id}},
        e2e_samples));
    }
}

void append_l2_records(std::vector<BenchmarkRecord>& records,
                       RuntimeBundle& bundle,
                       const ScenarioData& scenario,
                       const Options& options,
                       ProgressTracker& progress) {
    const int hidden_size = bundle.runtime.base_lm().config().hidden_size;

    if (benchmark_selected(options, "prefill.locenc_all")) {
    const auto locenc_samples = run_benchmark_case("prefill.locenc_all",
                                                   options.warmup,
                                                   options.repeat,
                                                   progress,
                                                   [&]() {
        std::vector<float> out = bundle.runtime.benchmark_encode_feature_sequence(scenario.prepared.feat, scenario.seq_len);
        (void)out;
    },
                                                   "scenario=" + scenario.spec.id);
    records.push_back(make_record(
        "prefill.locenc_all",
        "l2",
        "prefill",
        "locenc_all",
        "",
        json{{"seq_len", scenario.seq_len}, {"patch_size", scenario.patch_size}, {"feat_dim", scenario.feat_dim}, {"hidden_size", hidden_size}},
        locenc_samples));
    }

    const std::vector<float> feat_encoded = bundle.runtime.benchmark_encode_feature_sequence(scenario.prepared.feat, scenario.seq_len);
    if (benchmark_selected(options, "prefill.enc_to_lm_proj")) {
    const auto enc_to_lm_samples = run_benchmark_case("prefill.enc_to_lm_proj",
                                                      options.warmup,
                                                      options.repeat,
                                                      progress,
                                                      [&]() {
        std::vector<float> out = bundle.runtime.benchmark_run_enc_to_lm_projection(feat_encoded, scenario.seq_len);
        (void)out;
    },
                                                      "scenario=" + scenario.spec.id);
    records.push_back(make_record(
        "prefill.enc_to_lm_proj",
        "l2",
        "prefill",
        "enc_to_lm_proj",
        "",
        json{{"seq_len", scenario.seq_len}, {"hidden_size", hidden_size}},
        enc_to_lm_samples));
    }

    if (benchmark_selected(options, "prefill.text_embedding")) {
    const auto text_embedding_samples = run_benchmark_case("prefill.text_embedding",
                                                           options.warmup,
                                                           options.repeat,
                                                           progress,
                                                           [&]() {
        std::vector<float> out = bundle.runtime.benchmark_run_embedding(scenario.prepared.full_text_tokens);
        (void)out;
    },
                                                           "scenario=" + scenario.spec.id);
    records.push_back(make_record(
        "prefill.text_embedding",
        "l2",
        "prefill",
        "text_embedding",
        "",
        json{{"seq_len", scenario.seq_len}},
        text_embedding_samples));
    }

    const std::vector<float> feat_embed = bundle.runtime.benchmark_run_enc_to_lm_projection(feat_encoded, scenario.seq_len);
    const std::vector<float> text_embed = bundle.runtime.benchmark_run_embedding(scenario.prepared.full_text_tokens);
    const std::vector<float> combined_embed = combine_embeds(text_embed,
                                                             feat_embed,
                                                             scenario.prepared.text_mask,
                                                             scenario.prepared.feat_mask,
                                                             scenario.seq_len,
                                                             hidden_size);

    if (benchmark_selected(options, "prefill.base_lm")) {
    const auto base_lm_samples = run_benchmark_case("prefill.base_lm",
                                                    options.warmup,
                                                    options.repeat,
                                                    progress,
                                                    [&]() {
        VoxCPMDecodeState state = bundle.runtime.create_decode_state();
        std::vector<float> out = bundle.runtime.benchmark_run_base_lm_forward(combined_embed,
                                                                              scenario.seq_len,
                                                                              *state.base_lm_cache,
                                                                              true);
        (void)out;
    },
                                                    "scenario=" + scenario.spec.id);
    records.push_back(make_record(
        "prefill.base_lm",
        "l2",
        "prefill",
        "base_lm",
        "",
        json{{"seq_len", scenario.seq_len}, {"hidden_size", hidden_size}},
        base_lm_samples));
    }

    VoxCPMDecodeState fsq_state = bundle.runtime.create_decode_state();
    const std::vector<float> enc_outputs = bundle.runtime.benchmark_run_base_lm_forward(combined_embed,
                                                                                        scenario.seq_len,
                                                                                        *fsq_state.base_lm_cache,
                                                                                        true);
    if (benchmark_selected(options, "prefill.fsq")) {
    const auto fsq_samples = run_benchmark_case("prefill.fsq",
                                                options.warmup,
                                                options.repeat,
                                                progress,
                                                [&]() {
        std::vector<float> out = bundle.runtime.benchmark_run_fsq_2d(enc_outputs, scenario.seq_len);
        (void)out;
    },
                                                "scenario=" + scenario.spec.id);
    records.push_back(make_record(
        "prefill.fsq",
        "l2",
        "prefill",
        "fsq",
        "",
        json{{"seq_len", scenario.seq_len}, {"hidden_size", hidden_size}},
        fsq_samples));
    }

    const std::vector<float> residual_inputs = make_residual_inputs(enc_outputs,
                                                                    feat_embed,
                                                                    scenario.prepared.feat_mask,
                                                                    scenario.seq_len,
                                                                    hidden_size);
    if (benchmark_selected(options, "prefill.residual_lm")) {
    const auto residual_lm_samples = run_benchmark_case("prefill.residual_lm",
                                                        options.warmup,
                                                        options.repeat,
                                                        progress,
                                                        [&]() {
        VoxCPMDecodeState state = bundle.runtime.create_decode_state();
        std::vector<float> out = bundle.runtime.benchmark_run_residual_lm_forward(residual_inputs,
                                                                                  scenario.seq_len,
                                                                                  *state.residual_lm_cache,
                                                                                  true);
        (void)out;
    },
                                                        "scenario=" + scenario.spec.id);
    records.push_back(make_record(
        "prefill.residual_lm",
        "l2",
        "prefill",
        "residual_lm",
        "",
        json{{"seq_len", scenario.seq_len}, {"hidden_size", hidden_size}, {"prompt_audio_frames", scenario.prompt_frames}},
        residual_lm_samples));
    }

    struct BucketSpec { const char* name; int warm_steps; uint32_t seed_offset; };
    const std::vector<BucketSpec> buckets = {
        {"early", 0, 101},
        {"mid", 64, 201},
        {"late", 256, 301},
    };

    for (const auto& bucket : buckets) {
        const std::string front_half_case = "decode.front_half_total." + std::string(bucket.name);
        const std::string lm_to_dit_case = "decode.lm_to_dit_proj." + std::string(bucket.name);
        const std::string res_to_dit_case = "decode.res_to_dit_proj." + std::string(bucket.name);
        const std::string unified_case = "decode.unified_cfm." + std::string(bucket.name);
        const std::string stop_case = "decode.stop_predictor." + std::string(bucket.name);
        const std::string locenc_case = "decode.locenc_patch_to_lm." + std::string(bucket.name);
        const std::string base_step_case = "decode.base_lm_step_fsq." + std::string(bucket.name);
        const std::string residual_step_case = "decode.residual_lm_step." + std::string(bucket.name);

        if (benchmark_selected(options, front_half_case)) {
        const DecodePrepared front_half_prepared = prepare_decode_bucket(bundle, scenario, options, bucket.warm_steps, bucket.seed_offset);
        const auto front_half_samples = run_benchmark_case(front_half_case,
                                                           options.warmup,
                                                           options.repeat,
                                                           progress,
                                                           [&]() {
            std::vector<float> out = bundle.runtime.benchmark_run_decode_front_half(front_half_prepared.z,
                                                                                    front_half_prepared.target_state.lm_hidden,
                                                                                    front_half_prepared.target_state.residual_hidden,
                                                                                    front_half_prepared.target_state.prefix_feat_cond,
                                                                                    options.inference_timesteps,
                                                                                    options.cfg_value);
            (void)out;
        },
                                                           "scenario=" + scenario.spec.id);
        records.push_back(make_record(
            "decode.front_half_total",
            "l2",
            "decode",
            "front_half_total",
            bucket.name,
            json{{"position", scenario.seq_len + bucket.warm_steps + 1}},
            front_half_samples));
        }

        if (benchmark_selected(options, lm_to_dit_case)) {
        const DecodePrepared lm_to_dit_prepared = prepare_decode_bucket(bundle, scenario, options, bucket.warm_steps, bucket.seed_offset + 1);
        const auto lm_to_dit_samples = run_benchmark_case(lm_to_dit_case,
                                                          options.warmup,
                                                          options.repeat,
                                                          progress,
                                                          [&]() {
            std::vector<float> out = bundle.runtime.benchmark_run_lm_to_dit_projection(lm_to_dit_prepared.target_state.lm_hidden);
            (void)out;
        },
                                                          "scenario=" + scenario.spec.id);
        records.push_back(make_record(
            "decode.lm_to_dit_proj",
            "l2",
            "decode",
            "lm_to_dit_proj",
            bucket.name,
            json{{"position", scenario.seq_len + bucket.warm_steps + 1}},
            lm_to_dit_samples));
        }

        if (benchmark_selected(options, res_to_dit_case)) {
        const DecodePrepared res_to_dit_prepared = prepare_decode_bucket(bundle, scenario, options, bucket.warm_steps, bucket.seed_offset + 2);
        const auto res_to_dit_samples = run_benchmark_case(res_to_dit_case,
                                                           options.warmup,
                                                           options.repeat,
                                                           progress,
                                                           [&]() {
            std::vector<float> out = bundle.runtime.benchmark_run_res_to_dit_projection(res_to_dit_prepared.target_state.residual_hidden);
            (void)out;
        },
                                                           "scenario=" + scenario.spec.id);
        records.push_back(make_record(
            "decode.res_to_dit_proj",
            "l2",
            "decode",
            "res_to_dit_proj",
            bucket.name,
            json{{"position", scenario.seq_len + bucket.warm_steps + 1}},
            res_to_dit_samples));
        }

        if (benchmark_selected(options, unified_case)) {
        const DecodePrepared unified_setup = prepare_decode_bucket(bundle, scenario, options, bucket.warm_steps, bucket.seed_offset + 3);
        const std::vector<float> lm_dit = bundle.runtime.benchmark_run_lm_to_dit_projection(unified_setup.target_state.lm_hidden);
        const std::vector<float> res_dit = bundle.runtime.benchmark_run_res_to_dit_projection(unified_setup.target_state.residual_hidden);
        const std::vector<float> dit_hidden = add_vectors(lm_dit, res_dit);
        const auto unified_cfm_samples = run_benchmark_case(unified_case,
                                                            options.warmup,
                                                            options.repeat,
                                                            progress,
                                                            [&]() {
            std::vector<float> out = bundle.runtime.benchmark_run_unified_cfm(unified_setup.z,
                                                                              dit_hidden,
                                                                              unified_setup.target_state.prefix_feat_cond,
                                                                              options.inference_timesteps,
                                                                              options.cfg_value);
            (void)out;
        },
                                                            "scenario=" + scenario.spec.id);
        const double unified_mean = summarize(unified_cfm_samples).mean_ms;
        records.push_back(make_record(
            "decode.unified_cfm",
            "l2",
            "decode",
            "unified_cfm",
            bucket.name,
            json{{"position", scenario.seq_len + bucket.warm_steps + 1}, {"timesteps", options.inference_timesteps}},
            unified_cfm_samples,
            json{
                {"total_ms", unified_mean},
                {"ms_per_timestep", unified_mean / static_cast<double>(options.inference_timesteps)}
            }));
        }

        if (benchmark_selected(options, stop_case)) {
        const DecodePrepared stop_prepared = prepare_decode_bucket(bundle, scenario, options, bucket.warm_steps, bucket.seed_offset + 4);
        const auto stop_predictor_samples = run_benchmark_case(stop_case,
                                                               options.warmup,
                                                               options.repeat,
                                                               progress,
                                                               [&]() {
            std::vector<float> out = bundle.runtime.benchmark_run_stop_predictor(stop_prepared.target_state.lm_hidden);
            (void)out;
        },
                                                               "scenario=" + scenario.spec.id);
        records.push_back(make_record(
            "decode.stop_predictor",
            "l2",
            "decode",
            "stop_predictor",
            bucket.name,
            json{{"position", scenario.seq_len + bucket.warm_steps + 1}},
            stop_predictor_samples));
        }

        if (benchmark_selected(options, locenc_case) ||
            benchmark_selected(options, base_step_case) ||
            benchmark_selected(options, residual_step_case)) {
        const DecodePrepared patch_setup = prepare_decode_bucket(bundle, scenario, options, bucket.warm_steps, bucket.seed_offset + 5);
        const std::vector<float> patch = bundle.runtime.benchmark_run_decode_front_half(patch_setup.z,
                                                                                         patch_setup.target_state.lm_hidden,
                                                                                         patch_setup.target_state.residual_hidden,
                                                                                         patch_setup.target_state.prefix_feat_cond,
                                                                                         options.inference_timesteps,
                                                                                         options.cfg_value);
        if (benchmark_selected(options, locenc_case)) {
        const auto locenc_patch_samples = run_benchmark_case("decode.locenc_patch_to_lm." + std::string(bucket.name),
                                                             options.warmup,
                                                             options.repeat,
                                                             progress,
                                                             [&]() {
            std::vector<float> out = bundle.runtime.benchmark_run_locenc_patch_to_lm_embed(patch);
            (void)out;
        },
                                                             "scenario=" + scenario.spec.id);
        records.push_back(make_record(
            "decode.locenc_patch_to_lm",
            "l2",
            "decode",
            "locenc_patch_to_lm",
            bucket.name,
            json{{"position", scenario.seq_len + bucket.warm_steps + 1}},
            locenc_patch_samples));
        }

        const std::vector<float> curr_embed = bundle.runtime.benchmark_run_locenc_patch_to_lm_embed(patch);
        if (benchmark_selected(options, base_step_case)) {
        const DecodePrepared base_step_prepared = prepare_decode_bucket(bundle, scenario, options, bucket.warm_steps, bucket.seed_offset + 6);
        const auto base_lm_step_samples = run_benchmark_case(base_step_case,
                                                             options.warmup,
                                                             options.repeat,
                                                             progress,
                                                             [&]() {
            VoxCPMDecodeState state = bundle.runtime.benchmark_clone_state(base_step_prepared.target_state);
            std::vector<float> out = bundle.runtime.benchmark_run_base_lm_decode_step(curr_embed,
                                                                                      state.current_position + 1,
                                                                                      *state.base_lm_cache);
            (void)out;
        },
                                                             "scenario=" + scenario.spec.id);
        records.push_back(make_record(
            "decode.base_lm_step_fsq",
            "l2",
            "decode",
            "base_lm_step_fsq",
            bucket.name,
            json{{"position", scenario.seq_len + bucket.warm_steps + 1}},
            base_lm_step_samples));
        }

        if (benchmark_selected(options, residual_step_case)) {
        const DecodePrepared residual_step_prepared = prepare_decode_bucket(bundle, scenario, options, bucket.warm_steps, bucket.seed_offset + 7);
        const auto residual_step_samples = run_benchmark_case(residual_step_case,
                                                              options.warmup,
                                                              options.repeat,
                                                              progress,
                                                              [&]() {
            VoxCPMDecodeState state = bundle.runtime.benchmark_clone_state(residual_step_prepared.target_state);
            std::vector<float> lm_hidden = bundle.runtime.benchmark_run_base_lm_decode_step(curr_embed,
                                                                                            state.current_position + 1,
                                                                                            *state.base_lm_cache);
            std::vector<float> residual_input = add_vectors(lm_hidden, curr_embed);
            std::vector<float> out = bundle.runtime.benchmark_run_residual_lm_decode_step(residual_input,
                                                                                          state.current_position + 1,
                                                                                          *state.residual_lm_cache,
                                                                                          true);
            (void)out;
        },
                                                              "scenario=" + scenario.spec.id);
        records.push_back(make_record(
            "decode.residual_lm_step",
            "l2",
            "decode",
            "residual_lm_step",
            bucket.name,
            json{{"position", scenario.seq_len + bucket.warm_steps + 1}},
            residual_step_samples));
        }
        }
    }
}

json make_meta_json(const Options& options) {
    return json{
        {"timestamp", current_timestamp_string()},
        {"git_commit", discover_git_commit()},
        {"hostname", discover_hostname()},
        {"backend", backend_type_name(options.backend)},
        {"threads", options.threads},
        {"model_path", options.model_path},
        {"runner_version", kRunnerVersion},
    };
}

}  // namespace

int run(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        RuntimeBundle bundle(options.backend, options.threads);
        load_runtime_bundle(options, bundle);
        const int patch_len = bundle.runtime.config().patch_size * bundle.audio_vae.config().hop_length();
        const std::vector<ScenarioSpec> scenario_specs =
            build_scenarios(options, patch_len, bundle.audio_vae.config().sample_rate);
        ProgressTracker progress;
        progress.total_cases = static_cast<int>(scenario_specs.size()) * selected_cases_per_scenario(options);

        {
            std::ostringstream oss;
            oss << "Planned benchmark cases: " << progress.total_cases
                << " across " << scenario_specs.size() << " scenario(s)";
            log_progress(oss.str());
        }

        json output;
        output["meta"] = make_meta_json(options);
        output["scenarios"] = json::array();

        for (const auto& spec : scenario_specs) {
            log_progress("Starting scenario " + spec.id);
            ScenarioData scenario = build_scenario_data(spec, bundle, options);
            std::vector<BenchmarkRecord> records;

            if (options.level == "l1" || options.level == "all") {
                append_l1_records(records, bundle, scenario, options, progress);
            }
            if (options.level == "l2" || options.level == "all") {
                append_l2_records(records, bundle, scenario, options, progress);
            }

            print_summary_table(spec.id, records);

            json scenario_json;
            scenario_json["scenario"] = {
                {"id", spec.id},
                {"text", spec.text},
                {"prompt_text", options.prompt_text},
                {"prompt_audio_path", options.prompt_audio_path},
                {"inference_timesteps", options.inference_timesteps},
                {"seed", kBaseSeed},
            };
            scenario_json["benchmarks"] = json::array();
            for (const auto& record : records) {
                scenario_json["benchmarks"].push_back(record_to_json(record));
            }
            output["scenarios"].push_back(std::move(scenario_json));
            log_progress("Completed scenario " + spec.id);
        }

        const std::string output_path = options.output_json.empty() ? default_output_json_path() : options.output_json;
        std::filesystem::create_directories(std::filesystem::path(output_path).parent_path());
        std::ofstream out(output_path);
        out << std::setw(2) << output << "\n";
        std::cout << "\nWrote benchmark JSON to " << output_path << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Benchmark failed: " << e.what() << "\n";
        return 1;
    }
}

}  // namespace benchmark
}  // namespace voxcpm

int main(int argc, char** argv) {
    return voxcpm::benchmark::run(argc, argv);
}
