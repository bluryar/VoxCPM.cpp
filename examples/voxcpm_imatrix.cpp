/**
 * @file voxcpm_imatrix.cpp
 * @brief VoxCPM calibration collector for imatrix.gguf generation
 */

#include "voxcpm/audio-vae.h"
#include "voxcpm/backend.h"
#include "voxcpm/imatrix.h"
#include "voxcpm/tokenizer.h"
#include "voxcpm/voxcpm.h"
#include "voxcpm/weight-store.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace voxcpm {
namespace {

struct Options {
    std::string text_file;
    std::string dataset_file;
    std::string output_path;
    std::string input_path;
    std::string prompt_audio_path;
    std::string prompt_text;
    std::string model_path;
    float cfg_value = 2.0f;
    int inference_timesteps = 10;
    int threads = 4;
    int max_samples = 0;
    int max_decode_steps = 32;
    int save_frequency = 0;
    bool show_statistics = false;
    uint32_t seed = 1234;
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
};

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void print_usage(const char* argv0) {
    std::cerr << "Usage:\n"
              << "  " << argv0 << " (--text-file TEXTS.txt | --dataset-file DATA.tsv) "
              << "--output imatrix.gguf --model-path MODEL.gguf [options]\n\n"
              << "  " << argv0 << " --show-statistics --in-file imatrix.gguf\n\n"
              << "Options:\n"
              << "  --text-file PATH\n"
              << "  --dataset-file PATH (TSV: text or text<TAB>prompt_text<TAB>prompt_audio)\n"
              << "  --output PATH\n"
              << "  --in-file PATH\n"
              << "  --prompt-audio PATH\n"
              << "  --prompt-text TEXT\n"
              << "  --model-path GGUF\n"
              << "  --cfg-value FLOAT (default: 2.0)\n"
              << "  --inference-timesteps INT (default: 10)\n"
              << "  --threads INT (default: 4)\n"
              << "  --max-samples INT (default: all)\n"
              << "  --max-decode-steps INT (default: 32)\n"
              << "  --save-frequency INT (default: 0, disabled)\n"
              << "  --show-statistics\n"
              << "  --seed INT (default: 1234)\n";
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

        if (arg == "--text-file") {
            options.text_file = require_value("--text-file");
        } else if (arg == "--dataset-file") {
            options.dataset_file = require_value("--dataset-file");
        } else if (arg == "--output" || arg == "-o") {
            options.output_path = require_value("--output");
        } else if (arg == "--in-file") {
            options.input_path = require_value("--in-file");
        } else if (arg == "--prompt-audio") {
            options.prompt_audio_path = require_value("--prompt-audio");
        } else if (arg == "--prompt-text") {
            options.prompt_text = require_value("--prompt-text");
        } else if (arg == "--cfg-value") {
            options.cfg_value = std::stof(require_value("--cfg-value"));
        } else if (arg == "--inference-timesteps") {
            options.inference_timesteps = std::stoi(require_value("--inference-timesteps"));
        } else if (arg == "--threads") {
            options.threads = std::stoi(require_value("--threads"));
        } else if (arg == "--max-samples") {
            options.max_samples = std::stoi(require_value("--max-samples"));
        } else if (arg == "--max-decode-steps") {
            options.max_decode_steps = std::stoi(require_value("--max-decode-steps"));
        } else if (arg == "--save-frequency") {
            options.save_frequency = std::stoi(require_value("--save-frequency"));
        } else if (arg == "--show-statistics") {
            options.show_statistics = true;
        } else if (arg == "--seed") {
            options.seed = static_cast<uint32_t>(std::stoul(require_value("--seed")));
        } else if (arg == "--model-path") {
            options.model_path = require_value("--model-path");
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            fail("Unknown argument: " + arg);
        }
    }

    const bool statistics_only =
        options.show_statistics && !options.input_path.empty() && options.text_file.empty() && options.model_path.empty();

    if (statistics_only) {
        if (!std::filesystem::exists(options.input_path)) {
            fail("--in-file must point to an existing imatrix file");
        }
        return options;
    }

    const bool has_text_file = !options.text_file.empty();
    const bool has_dataset_file = !options.dataset_file.empty();
    if (has_text_file == has_dataset_file) {
        fail("Exactly one of --text-file or --dataset-file is required");
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
    if (options.threads < 1) {
        fail("--threads must be >= 1");
    }
    if (options.max_samples < 0) {
        fail("--max-samples must be >= 0");
    }
    if (options.max_decode_steps < 1) {
        fail("--max-decode-steps must be >= 1");
    }
    if (options.save_frequency < 0) {
        fail("--save-frequency must be >= 0");
    }

    if (!std::filesystem::exists(options.model_path)) {
        fail("--model-path must point to an existing GGUF file");
    }
    if (has_text_file && !std::filesystem::exists(options.text_file)) {
        fail("--text-file must point to an existing text file");
    }
    if (has_dataset_file && !std::filesystem::exists(options.dataset_file)) {
        fail("--dataset-file must point to an existing TSV file");
    }
    if (!options.prompt_audio_path.empty() && !std::filesystem::exists(options.prompt_audio_path)) {
        fail("Prompt audio file does not exist: " + options.prompt_audio_path);
    }

    return options;
}

std::string make_snapshot_path(const std::string& output_path, int chunk_index) {
    if (chunk_index <= 0) {
        return output_path;
    }
    return output_path + ".at_" + std::to_string(chunk_index);
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
    (void) read_le_u32(in);
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
            (void) read_le_u32(in);
            (void) read_le_u16(in);
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

std::vector<float> extract_prompt_features(AudioVAE& audio_vae,
                                           VoxCPMBackend& backend,
                                           VoxCPMImatrixCollector* collector,
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
    backend.reserve_compute_memory(graph, "imatrix.audio_vae.encode");
    backend.alloc_graph(graph, "imatrix.audio_vae.encode");
    const auto& preprocessed = audio_vae.last_preprocessed_audio();
    backend.tensor_set(audio_vae.last_input_tensor(), preprocessed.data(), 0, preprocessed.size() * sizeof(float));
    if (backend.compute(graph) != GGML_STATUS_SUCCESS) {
        fail("AudioVAE encode failed");
    }
    if (collector) {
        collector->observe_graph(graph, backend);
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

std::vector<float> build_decode_feature_sequence(const std::vector<float>& prompt_feat,
                                                 int prompt_audio_length,
                                                 const std::vector<float>& generated_steps,
                                                 int streaming_prefix_len,
                                                 int patch_size,
                                                 int feat_dim) {
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
    return decode_frames;
}

std::vector<float> patch_major_to_latent(const std::vector<float>& frames,
                                         int patch_size,
                                         int feat_dim) {
    const size_t frame_stride = static_cast<size_t>(patch_size) * feat_dim;
    const int total_frames = static_cast<int>(frames.size() / frame_stride);
    const int total_patches = total_frames * patch_size;

    std::vector<float> latent(static_cast<size_t>(total_patches) * feat_dim, 0.0f);
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
    return latent;
}

void collect_decode_audio_imatrix(AudioVAE& audio_vae,
                                  VoxCPMBackend& backend,
                                  VoxCPMImatrixCollector* collector,
                                  const std::vector<float>& latent,
                                  int total_patches,
                                  int feat_dim) {
    if (!collector || latent.empty() || total_patches <= 0) {
        return;
    }

    VoxCPMContext graph_ctx(ContextType::Graph, 32768, 262144);
    ggml_tensor* latent_tensor = graph_ctx.new_tensor_2d(GGML_TYPE_F32, total_patches, feat_dim);
    ggml_set_input(latent_tensor);
    ggml_tensor* audio = audio_vae.decode(graph_ctx, latent_tensor);
    if (!audio) {
        fail("Failed to build AudioVAE decode graph");
    }

    ggml_cgraph* graph = graph_ctx.new_graph();
    graph_ctx.build_forward(graph, audio);
    backend.reserve_compute_memory(graph, "imatrix.audio_vae.decode");
    backend.alloc_graph(graph, "imatrix.audio_vae.decode");
    backend.tensor_set(latent_tensor, latent.data(), 0, latent.size() * sizeof(float));
    if (backend.compute(graph) != GGML_STATUS_SUCCESS) {
        fail("AudioVAE decode failed during imatrix collection");
    }
    collector->observe_graph(graph, backend);
}

PreparedInputs prepare_inputs(const std::string& text,
                              const std::string& prompt_audio_path,
                              const std::string& prompt_text,
                              ChineseCharSplitTokenizer& split_tokenizer,
                              AudioVAE& audio_vae,
                              VoxCPMBackend& backend,
                              VoxCPMImatrixCollector* collector,
                              int patch_size,
                              int feat_dim,
                              int patch_len) {
    PreparedInputs prepared;

    std::vector<int32_t> text_tokens = split_tokenizer.encode(
        prompt_audio_path.empty() ? text : prompt_text + text,
        false);
    text_tokens.push_back(101);

    prepared.full_text_tokens = text_tokens;
    if (prompt_audio_path.empty()) {
        const int seq_len = static_cast<int>(text_tokens.size());
        prepared.feat.assign(static_cast<size_t>(seq_len) * patch_size * feat_dim, 0.0f);
        prepared.text_mask.assign(static_cast<size_t>(seq_len), 1);
        prepared.feat_mask.assign(static_cast<size_t>(seq_len), 0);
        return prepared;
    }

    const WavData wav = read_wav_file(prompt_audio_path);
    std::vector<float> mono = convert_to_mono(wav);
    mono = linear_resample(mono, wav.sample_rate, audio_vae.config().sample_rate);
    if (mono.size() % static_cast<size_t>(patch_len) != 0) {
        const size_t padding = static_cast<size_t>(patch_len) - (mono.size() % static_cast<size_t>(patch_len));
        mono.insert(mono.begin(), padding, 0.0f);
    }

    prepared.prompt_feat = extract_prompt_features(
        audio_vae, backend, collector, mono, audio_vae.config().sample_rate, patch_size, feat_dim);
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

std::vector<float> generate_noise(int patch_size, int feat_dim, std::mt19937& rng) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> noise(static_cast<size_t>(patch_size) * feat_dim, 0.0f);
    for (float& value : noise) {
        value = dist(rng);
    }
    return noise;
}

std::vector<VoxCPMCalibrationSample> load_calibration_samples(const Options& options) {
    if (!options.dataset_file.empty()) {
        return load_calibration_dataset_file(options.dataset_file, options.max_samples);
    }
    return load_text_calibration_file(options.text_file, options.max_samples);
}

}  // namespace
}  // namespace voxcpm

int main(int argc, char** argv) {
    using namespace voxcpm;

    try {
        const Options options = parse_args(argc, argv);
        const bool statistics_only =
            options.show_statistics && !options.input_path.empty() && options.text_file.empty() && options.model_path.empty();
        if (statistics_only) {
            VoxCPMImatrixCollector collector;
            collector.load_from_file(options.input_path);
            collector.print_statistics(std::cout);
            return 0;
        }

        constexpr int kStreamingPrefixLen = 3;
        constexpr int kMinStopStep = 2;

        VoxCPMBackend backend(BackendType::CPU, options.threads);
        std::cerr << "Loading GGUF from " << options.model_path << " with " << options.threads << " threads...\n";
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

        VoxCPMImatrixCollector collector;
        collector.set_chunk_size(1);
        if (!options.dataset_file.empty()) {
            collector.add_dataset(options.dataset_file);
        } else {
            collector.add_dataset(options.text_file);
        }
        if (!options.prompt_audio_path.empty()) {
            collector.add_dataset(options.prompt_audio_path);
        }
        runtime.set_imatrix_collector(&collector);

        const int patch_size = runtime.config().patch_size;
        const int feat_dim = runtime.config().feat_dim;
        const int patch_len = patch_size * audio_vae.config().hop_length();
        const std::vector<VoxCPMCalibrationSample> samples = load_calibration_samples(options);
        for (const VoxCPMCalibrationSample& sample : samples) {
            if (!sample.prompt_audio_path.empty()) {
                collector.add_dataset(sample.prompt_audio_path);
            }
        }
        std::mt19937 rng(options.seed);

        std::cerr << "Collecting imatrix from " << samples.size() << " samples...\n";
        for (size_t i = 0; i < samples.size(); ++i) {
            const VoxCPMCalibrationSample& sample = samples[i];
            const std::string& active_prompt_audio =
                sample.prompt_audio_path.empty() ? options.prompt_audio_path : sample.prompt_audio_path;
            const std::string& active_prompt_text =
                sample.prompt_audio_path.empty() ? options.prompt_text : sample.prompt_text;
            const PreparedInputs prepared = prepare_inputs(
                sample.text,
                active_prompt_audio,
                active_prompt_text,
                split_tokenizer,
                audio_vae,
                backend,
                &collector,
                patch_size,
                feat_dim,
                patch_len);
            const int seq_len = static_cast<int>(prepared.full_text_tokens.size());
            std::cerr << "Sample " << (i + 1) << "/" << samples.size() << ": prefill seq_len=" << seq_len << "\n";

            VoxCPMDecodeState state = runtime.prefill(prepared.full_text_tokens,
                                                     prepared.text_mask,
                                                     prepared.feat,
                                                     prepared.feat_mask,
                                                     seq_len,
                                                     kStreamingPrefixLen);

            const int target_text_token_count =
                std::max<int>(1, static_cast<int>(split_tokenizer.tokenize(sample.text).size()));
            const int natural_max_len = std::min(target_text_token_count * 6 + 10, 2000);
            const int max_len = std::min(natural_max_len, options.max_decode_steps);
            std::vector<float> generated_steps;
            generated_steps.reserve(static_cast<size_t>(max_len) * patch_size * feat_dim);

            for (int step = 0; step < max_len; ++step) {
                const std::vector<float> z = generate_noise(patch_size, feat_dim, rng);
                VoxCPMDecodeResult result = runtime.decode(std::move(state),
                                                           z,
                                                           options.inference_timesteps,
                                                           options.cfg_value);
                generated_steps.insert(generated_steps.end(), result.output_0.begin(), result.output_0.end());
                state = std::move(result.output_1);
                if (step > kMinStopStep && result.output_2) {
                    break;
                }
            }

            const std::vector<float> decode_frames = build_decode_feature_sequence(prepared.prompt_feat,
                                                                                   prepared.prompt_audio_length,
                                                                                   generated_steps,
                                                                                   kStreamingPrefixLen,
                                                                                   patch_size,
                                                                                   feat_dim);
            const std::vector<float> latent = patch_major_to_latent(decode_frames, patch_size, feat_dim);
            const int total_patches = static_cast<int>(latent.size() / static_cast<size_t>(feat_dim));
            collect_decode_audio_imatrix(audio_vae, backend, &collector, latent, total_patches, feat_dim);

            collector.mark_chunk_processed();
            if (options.save_frequency > 0 &&
                collector.chunks_count() > 0 &&
                collector.chunks_count() % options.save_frequency == 0) {
                const std::string snapshot_path = make_snapshot_path(options.output_path, collector.chunks_count());
                collector.save_to_file(snapshot_path);
                std::cerr << "Saved imatrix snapshot to " << snapshot_path
                          << " after " << collector.chunks_count() << " chunks.\n";
            }
        }

        collector.save_to_file(options.output_path);
        std::cerr << "Saved imatrix to " << options.output_path
                  << " with " << collector.entry_count()
                  << " entries across " << collector.chunks_count() << " chunks.\n";
        if (options.show_statistics) {
            collector.print_statistics(std::cerr);
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
