#include "voxcpm/server_common.h"

#include "voxcpm/audio_io.h"
#include "voxcpm/context.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace voxcpm {

namespace {

using json = nlohmann::json;

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

std::filesystem::path manifest_path_for(const std::string& root, const std::string& id) {
    return std::filesystem::path(root) / id / "manifest.json";
}

std::filesystem::path prompt_path_for(const std::string& root, const std::string& id) {
    return std::filesystem::path(root) / id / "prompt_feat.bin";
}

std::filesystem::path reference_path_for(const std::string& root, const std::string& id) {
    return std::filesystem::path(root) / id / "reference_feat.bin";
}

constexpr int32_t kAudioStartToken = 101;
constexpr int32_t kRefAudioStartToken = 103;
constexpr int32_t kRefAudioEndToken = 104;

enum class PaddingMode {
    Left,
    Right,
};

struct PreparedConditioning {
    std::vector<int32_t> full_text_tokens;
    std::vector<int32_t> text_mask;
    std::vector<int32_t> feat_mask;
    std::vector<float> feat;
};

size_t skip_ascii_whitespace(const std::string& text, size_t pos) {
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])) != 0) {
        ++pos;
    }
    return pos;
}

std::pair<std::string, bool> strip_hifi_control_prefix(const std::string& text) {
    const size_t start = skip_ascii_whitespace(text, 0);
    if (start >= text.size()) {
        return {text, false};
    }

    size_t content_start = std::string::npos;
    size_t close_pos = std::string::npos;
    size_t close_len = 0;
    if (text.compare(start, 1, "(") == 0) {
        content_start = start + 1;
        close_pos = text.find(')', content_start);
        close_len = 1;
    } else if (text.compare(start, 3, "（") == 0) {
        content_start = start + 3;
        close_pos = text.find("）", content_start);
        close_len = 3;
    }
    if (close_pos == std::string::npos) {
        return {text, false};
    }

    const size_t next = skip_ascii_whitespace(text, close_pos + close_len);
    return {text.substr(next), true};
}

void write_binary_file(const std::filesystem::path& path, const std::vector<float>& values) {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        fail("Failed to open file for writing: " + path.string());
    }
    out.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
}

std::vector<float> read_binary_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in.is_open()) {
        fail("Failed to open file for reading: " + path.string());
    }
    const std::streamsize size = in.tellg();
    if (size < 0 || (size % static_cast<std::streamsize>(sizeof(float))) != 0) {
        fail("Invalid prompt feature blob: " + path.string());
    }
    in.seekg(0, std::ios::beg);
    std::vector<float> values(static_cast<size_t>(size) / sizeof(float), 0.0f);
    in.read(reinterpret_cast<char*>(values.data()), size);
    return values;
}

void pad_audio_for_patch_alignment(std::vector<float>& audio, size_t patch_len, PaddingMode mode) {
    if (patch_len == 0 || audio.empty() || (audio.size() % patch_len) == 0) {
        return;
    }
    const size_t padding = patch_len - (audio.size() % patch_len);
    if (mode == PaddingMode::Left) {
        audio.insert(audio.begin(), padding, 0.0f);
    } else {
        audio.insert(audio.end(), padding, 0.0f);
    }
}

std::vector<float> extract_prompt_features(AudioVAE& audio_vae,
                                           VoxCPMBackend& backend,
                                           std::vector<float> audio,
                                           int sample_rate,
                                           int patch_size,
                                           int feat_dim) {
    VoxCPMContext graph_ctx(ContextType::Graph, 32768, 262144);
    ggml_tensor* latent = audio_vae.encode(graph_ctx, backend, audio, sample_rate);
    if (!latent) {
        fail("Failed to build AudioVAE encode graph");
    }

    ggml_cgraph* graph = graph_ctx.new_graph();
    ggml_tensor* patch_major = ggml_cont(graph_ctx.raw_context(), ggml_transpose(graph_ctx.raw_context(), latent));
    graph_ctx.build_forward(graph, patch_major);
    backend.reserve_compute_memory(graph, "server.audio_vae.encode");
    backend.alloc_graph(graph, "server.audio_vae.encode");
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

    const int audio_length = total_patches / patch_size;
    std::vector<float> features(static_cast<size_t>(audio_length) * patch_size * feat_dim, 0.0f);
    backend.tensor_get(patch_major, features.data(), 0, features.size() * sizeof(float));
    return features;
}

PreparedConditioning build_conditioning(ChineseCharSplitTokenizer& split_tokenizer,
                                        const std::string& effective_target_text,
                                        const PromptFeatures& prompt,
                                        int patch_size,
                                        int feat_dim) {
    const int prompt_audio_length = prompt.prompt_audio_length;
    const int reference_audio_length = prompt.reference_audio_length;
    const size_t frame_stride = static_cast<size_t>(patch_size) * feat_dim;
    const std::string main_text =
        prompt_audio_length > 0 ? prompt.prompt_text + effective_target_text : effective_target_text;

    std::vector<int32_t> text_tokens = split_tokenizer.encode(main_text, false);
    text_tokens.push_back(kAudioStartToken);

    const size_t total_frames = static_cast<size_t>(reference_audio_length) + 2 +
                                static_cast<size_t>(text_tokens.size()) +
                                static_cast<size_t>(prompt_audio_length);
    PreparedConditioning prepared;
    prepared.full_text_tokens.reserve(total_frames);
    prepared.text_mask.reserve(total_frames);
    prepared.feat_mask.reserve(total_frames);
    prepared.feat.reserve(total_frames * frame_stride);

    const auto append_zero_frame = [&]() {
        prepared.feat.insert(prepared.feat.end(), frame_stride, 0.0f);
    };
    const auto append_feat_frames = [&](const std::vector<float>& frames, int frame_count) {
        prepared.feat.insert(prepared.feat.end(),
                             frames.begin(),
                             frames.begin() + static_cast<std::ptrdiff_t>(static_cast<size_t>(frame_count) * frame_stride));
    };

    if (reference_audio_length > 0) {
        prepared.full_text_tokens.push_back(kRefAudioStartToken);
        prepared.text_mask.push_back(1);
        prepared.feat_mask.push_back(0);
        append_zero_frame();

        for (int i = 0; i < reference_audio_length; ++i) {
            prepared.full_text_tokens.push_back(0);
            prepared.text_mask.push_back(0);
            prepared.feat_mask.push_back(1);
        }
        append_feat_frames(prompt.reference_feat, reference_audio_length);

        prepared.full_text_tokens.push_back(kRefAudioEndToken);
        prepared.text_mask.push_back(1);
        prepared.feat_mask.push_back(0);
        append_zero_frame();
    }

    for (int32_t token : text_tokens) {
        prepared.full_text_tokens.push_back(token);
        prepared.text_mask.push_back(1);
        prepared.feat_mask.push_back(0);
        append_zero_frame();
    }

    if (prompt_audio_length > 0) {
        for (int i = 0; i < prompt_audio_length; ++i) {
            prepared.full_text_tokens.push_back(0);
            prepared.text_mask.push_back(0);
            prepared.feat_mask.push_back(1);
        }
        append_feat_frames(prompt.prompt_feat, prompt_audio_length);
    }

    return prepared;
}

std::vector<float> decode_audio(AudioVAE& audio_vae,
                                VoxCPMBackend& backend,
                                const std::vector<float>& features,
                                int total_patches,
                                int feat_dim) {
    VoxCPMContext graph_ctx(ContextType::Graph, 32768, 262144);
    ggml_tensor* latent = graph_ctx.new_tensor_2d(GGML_TYPE_F32, total_patches, feat_dim);
    ggml_set_input(latent);
    ggml_tensor* audio = audio_vae.decode(graph_ctx, backend, latent);
    if (!audio) {
        fail("Failed to build AudioVAE decode graph");
    }

    ggml_cgraph* graph = graph_ctx.new_graph();
    graph_ctx.build_forward(graph, audio);
    backend.reserve_compute_memory(graph, "server.audio_vae.decode");
    backend.alloc_graph(graph, "server.audio_vae.decode");
    backend.tensor_set(latent, features.data(), 0, features.size() * sizeof(float));
    audio_vae.prepare_decode_inputs(backend);
    if (backend.compute(graph) != GGML_STATUS_SUCCESS) {
        fail("AudioVAE decode failed");
    }

    std::vector<float> waveform(static_cast<size_t>(ggml_nelements(audio)));
    backend.tensor_get(audio, waveform.data(), 0, waveform.size() * sizeof(float));
    return waveform;
}

std::vector<float> decode_audio_from_patch_major_frames(AudioVAE& audio_vae,
                                                        VoxCPMBackend& backend,
                                                        const std::vector<float>& frames,
                                                        int patch_size,
                                                        int feat_dim) {
    const size_t frame_stride = static_cast<size_t>(patch_size) * feat_dim;
    if (frames.empty() || (frames.size() % frame_stride) != 0) {
        return {};
    }

    const int total_frames = static_cast<int>(frames.size() / frame_stride);
    const int total_patches = total_frames * patch_size;

    VoxCPMContext graph_ctx(ContextType::Graph, 32768, 262144);
    ggml_tensor* patch_major = graph_ctx.new_tensor_2d(GGML_TYPE_F32, feat_dim, total_patches);
    ggml_set_input(patch_major);
    ggml_tensor* latent = ggml_cont(graph_ctx.raw_context(), ggml_transpose(graph_ctx.raw_context(), patch_major));
    ggml_tensor* audio = audio_vae.decode(graph_ctx, backend, latent);
    if (!audio) {
        fail("Failed to build AudioVAE decode graph from patch-major frames");
    }

    ggml_cgraph* graph = graph_ctx.new_graph();
    graph_ctx.build_forward(graph, audio);
    backend.reserve_compute_memory(graph, "server.audio_vae.decode.patch_major");
    backend.alloc_graph(graph, "server.audio_vae.decode.patch_major");
    backend.tensor_set(patch_major, frames.data(), 0, frames.size() * sizeof(float));
    audio_vae.prepare_decode_inputs(backend);
    if (backend.compute(graph) != GGML_STATUS_SUCCESS) {
        fail("AudioVAE decode from patch-major frames failed");
    }

    std::vector<float> waveform(static_cast<size_t>(ggml_nelements(audio)));
    backend.tensor_get(audio, waveform.data(), 0, waveform.size() * sizeof(float));
    return waveform;
}

std::vector<float> decode_audio_from_output_pool(AudioVAE& audio_vae,
                                                 VoxCPMBackend& backend,
                                                 const VoxCPMOutputPool& output_pool,
                                                 int frame_offset,
                                                 int frame_count,
                                                 int patch_size,
                                                 int feat_dim) {
    if (frame_count <= 0) {
        return {};
    }
    ggml_tensor* latent_seq = output_pool.latent_seq();
    if (latent_seq == nullptr) {
        return {};
    }

    const int total_patches = frame_count * patch_size;
    const int patch_offset = frame_offset * patch_size;
    if (patch_offset < 0 || patch_offset + total_patches > output_pool.shape().max_latent_patches * patch_size) {
        return {};
    }
    if (output_pool.shape().feat_dim != feat_dim || output_pool.shape().patch_size != patch_size) {
        fail("Output pool shape does not match AudioVAE decode request");
    }

    VoxCPMContext graph_ctx(ContextType::Graph, 32768, 262144);
    ggml_tensor* latent = output_pool.make_audio_vae_latent_view(graph_ctx.raw_context(), frame_offset, frame_count);
    if (latent == nullptr) {
        return {};
    }
    ggml_tensor* audio = audio_vae.decode(graph_ctx, backend, latent);
    if (!audio) {
        fail("Failed to build AudioVAE decode graph");
    }

    ggml_cgraph* graph = graph_ctx.new_graph();
    graph_ctx.build_forward(graph, audio);
    backend.reserve_compute_memory(graph, "server.audio_vae.decode.output_pool");
    backend.alloc_graph(graph, "server.audio_vae.decode.output_pool");
    audio_vae.prepare_decode_inputs(backend);
    if (backend.compute(graph) != GGML_STATUS_SUCCESS) {
        fail("AudioVAE decode from output pool failed");
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

int decode_step_cap_for_service(BackendType backend_type, int seq_len) {
    constexpr int kShortSeqThreshold = 256;
    constexpr int kLongSeqThreshold = 512;

    if (backend_type == BackendType::CPU) {
        return seq_len > kLongSeqThreshold ? 32 : (seq_len > kShortSeqThreshold ? 48 : 64);
    }
    return seq_len > kLongSeqThreshold ? 64 : (seq_len > kShortSeqThreshold ? 96 : 128);
}

bool should_use_output_pool_timeline(const VoxCPMDecodeState& state, bool has_reference_audio, int seq_len) {
    constexpr int kOutputPoolSeqLimit = 256;
    return state.output_pool != nullptr && state.output_pool->is_initialized() && !has_reference_audio &&
           seq_len <= kOutputPoolSeqLimit;
}

std::vector<float> build_decode_latent_sequence(const std::vector<float>& prompt_feat,
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

    if (prepended_context_frames != nullptr) {
        *prepended_context_frames = context_frames;
    }
    const int generated_frames = static_cast<int>(generated_steps.size() / frame_stride);
    const int total_frames = context_frames + generated_frames;
    const int total_patches = total_frames * patch_size;
    std::vector<float> latent(static_cast<size_t>(total_patches) * feat_dim, 0.0f);

    auto write_patch_major_frames = [&](const float* frames, int frame_count, int frame_base) {
        if (frames == nullptr || frame_count <= 0) {
            return;
        }
        for (int frame = 0; frame < frame_count; ++frame) {
            for (int patch = 0; patch < patch_size; ++patch) {
                const int time_index = (frame_base + frame) * patch_size + patch;
                for (int d = 0; d < feat_dim; ++d) {
                    const size_t src = (static_cast<size_t>(frame) * patch_size + patch) * feat_dim + d;
                    const size_t dst = static_cast<size_t>(d) * total_patches + time_index;
                    latent[dst] = frames[src];
                }
            }
        }
    };

    if (context_frames > 0) {
        const size_t context_offset = static_cast<size_t>(prompt_audio_length - context_frames) * frame_stride;
        write_patch_major_frames(prompt_feat.data() + static_cast<std::ptrdiff_t>(context_offset), context_frames, 0);
    }
    write_patch_major_frames(generated_steps.data(), generated_frames, context_frames);
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

void clear_decode_graph_caches_after_streaming_audio_decode(VoxCPMRuntime& runtime,
                                                            VoxCPMDecodeState& state) {
    // Streaming AudioVAE chunk decode can resize the shared compute arena.
    // Drop cached graph handles before the next decode step can reuse stale tensor data pointers.
    runtime.reset_request_state();
    state.base_lm_step_graph.clear();
    state.residual_lm_step_graph.clear();
    state.base_lm_step_graph_position = -1;
    state.residual_lm_step_graph_position = -1;
}

}  // namespace

VoiceStore::VoiceStore(std::string root_dir)
    : root_dir_(std::move(root_dir)) {
    std::filesystem::create_directories(root_dir_);
}

bool VoiceStore::has_voice(const std::string& id) const {
    return std::filesystem::exists(manifest_path_for(root_dir_, id)) &&
           std::filesystem::exists(prompt_path_for(root_dir_, id));
}

void VoiceStore::save_voice(const PromptFeatures& features) {
    if (!is_valid_voice_id(features.id)) {
        fail("Invalid voice id");
    }
    const auto dir = std::filesystem::path(root_dir_) / features.id;
    std::filesystem::create_directories(dir);

    json manifest = {
        {"id", features.id},
        {"prompt_text", features.prompt_text},
        {"prompt_audio_length", features.prompt_audio_length},
        {"reference_audio_length", features.reference_audio_length},
        {"sample_rate", features.sample_rate},
        {"patch_size", features.patch_size},
        {"feat_dim", features.feat_dim},
        {"created_at", features.created_at},
        {"updated_at", features.updated_at},
    };

    std::ofstream out(manifest_path_for(root_dir_, features.id));
    if (!out.is_open()) {
        fail("Failed to write voice manifest");
    }
    out << manifest.dump(2);
    write_binary_file(prompt_path_for(root_dir_, features.id), features.prompt_feat);
    if (!features.reference_feat.empty()) {
        write_binary_file(reference_path_for(root_dir_, features.id), features.reference_feat);
    } else {
        std::error_code ec;
        std::filesystem::remove(reference_path_for(root_dir_, features.id), ec);
    }
}

PromptFeatures VoiceStore::load_voice(const std::string& id) const {
    const auto manifest_path = manifest_path_for(root_dir_, id);
    const auto prompt_path = prompt_path_for(root_dir_, id);
    if (!std::filesystem::exists(manifest_path) || !std::filesystem::exists(prompt_path)) {
        fail("Voice not found: " + id);
    }

    std::ifstream in(manifest_path);
    if (!in.is_open()) {
        fail("Failed to read voice manifest");
    }
    json manifest = json::parse(in);
    PromptFeatures features;
    features.id = manifest.at("id").get<std::string>();
    features.prompt_text = manifest.at("prompt_text").get<std::string>();
    features.prompt_audio_length = manifest.at("prompt_audio_length").get<int>();
    features.reference_audio_length = manifest.value("reference_audio_length", 0);
    features.sample_rate = manifest.at("sample_rate").get<int>();
    features.patch_size = manifest.at("patch_size").get<int>();
    features.feat_dim = manifest.at("feat_dim").get<int>();
    features.created_at = manifest.value("created_at", "");
    features.updated_at = manifest.value("updated_at", "");
    features.prompt_feat = read_binary_file(prompt_path);
    if (features.reference_audio_length > 0) {
        const auto ref_path = reference_path_for(root_dir_, id);
        if (!std::filesystem::exists(ref_path)) {
            fail("Reference feature blob missing for voice: " + id);
        }
        features.reference_feat = read_binary_file(ref_path);
    }
    return features;
}

VoiceMetadata VoiceStore::load_metadata(const std::string& id) const {
    const PromptFeatures features = load_voice(id);
    return VoiceMetadata{
        features.id,
        features.prompt_text,
        features.prompt_audio_length,
        features.reference_audio_length,
        features.sample_rate,
        features.patch_size,
        features.feat_dim,
        features.created_at,
        features.updated_at,
    };
}

void VoiceStore::delete_voice(const std::string& id) {
    const auto dir = std::filesystem::path(root_dir_) / id;
    if (!std::filesystem::exists(dir)) {
        fail("Voice not found: " + id);
    }
    std::filesystem::remove_all(dir);
}

VoxCPMServiceCore::VoxCPMServiceCore(std::string model_path, BackendType backend_type, int threads)
    : model_path_(std::move(model_path)),
      backend_type_(backend_type),
      threads_(threads) {}

void VoxCPMServiceCore::load() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (loaded_) {
        return;
    }

    if (!std::filesystem::exists(model_path_) || !std::filesystem::is_regular_file(model_path_)) {
        fail("Model path must point to an existing GGUF file");
    }

    backend_ = std::make_unique<VoxCPMBackend>(backend_type_, threads_);
    store_ = std::make_shared<VoxCPMWeightStore>();
    if (!store_->load_from_file(model_path_, *backend_)) {
        fail("Failed to load GGUF: " + model_path_);
    }
    if (!runtime_.load_from_store(store_, *backend_)) {
        fail("Failed to initialize VoxCPM runtime from GGUF");
    }
    if (!audio_vae_.load_from_store(store_)) {
        fail("Failed to initialize AudioVAE from GGUF");
    }

    tokenizer_ = std::make_unique<VoxCPMTokenizer>();
    if (!tokenizer_->load_from_store(*store_)) {
        fail("Failed to load tokenizer metadata from GGUF");
    }
    split_tokenizer_ = std::make_unique<ChineseCharSplitTokenizer>(*tokenizer_);
    loaded_ = true;
}

PromptFeatures VoxCPMServiceCore::encode_prompt_audio(const std::string& id,
                                                      const std::string& prompt_text,
                                                      const std::vector<float>& mono_audio,
                                                      int sample_rate) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!loaded_) {
        fail("Model core is not loaded");
    }
    return encode_prompt_audio_locked(id, prompt_text, mono_audio, sample_rate);
}

PromptFeatures VoxCPMServiceCore::encode_reference_audio(const std::string& id,
                                                         const std::vector<float>& mono_audio,
                                                         int sample_rate) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!loaded_) {
        fail("Model core is not loaded");
    }
    return encode_reference_audio_locked(id, mono_audio, sample_rate);
}

PromptFeatures VoxCPMServiceCore::encode_prompt_audio_locked(const std::string& id,
                                                             const std::string& prompt_text,
                                                             const std::vector<float>& mono_audio,
                                                             int sample_rate) {
    const int patch_size_value = runtime_.config().patch_size;
    const int feat_dim_value = runtime_.config().feat_dim;
    const int patch_len = patch_size_value * audio_vae_.config().hop_length();
    std::vector<float> resampled = resample_audio_to_rate(mono_audio, sample_rate, audio_vae_.config().sample_rate);
    resampled = trim_audio_silence_vad(resampled, audio_vae_.config().sample_rate);
    pad_audio_for_patch_alignment(resampled, static_cast<size_t>(patch_len), PaddingMode::Left);

    PromptFeatures features;
    features.id = id;
    features.prompt_text = prompt_text;
    features.prompt_feat = extract_prompt_features(audio_vae_,
                                                   *backend_,
                                                   resampled,
                                                   audio_vae_.config().sample_rate,
                                                   patch_size_value,
                                                   feat_dim_value);
    features.prompt_audio_length =
        static_cast<int>(features.prompt_feat.size() / static_cast<size_t>(patch_size_value * feat_dim_value));
    features.sample_rate = audio_vae_.config().sample_rate;
    features.patch_size = patch_size_value;
    features.feat_dim = feat_dim_value;
    const std::string now = make_timestamp_utc();
    features.created_at = now;
    features.updated_at = now;
    std::cerr << "[voice] encoded id=" << id
              << " prompt_audio_length=" << features.prompt_audio_length
              << " patch_size=" << features.patch_size
              << " feat_dim=" << features.feat_dim
              << " sample_rate=" << features.sample_rate
              << "\n";
    return features;
}

PromptFeatures VoxCPMServiceCore::encode_reference_audio_locked(const std::string& id,
                                                                const std::vector<float>& mono_audio,
                                                                int sample_rate) {
    const int patch_size_value = runtime_.config().patch_size;
    const int feat_dim_value = runtime_.config().feat_dim;
    const int patch_len = patch_size_value * audio_vae_.config().hop_length();
    std::vector<float> resampled = resample_audio_to_rate(mono_audio, sample_rate, audio_vae_.config().sample_rate);
    resampled = trim_audio_silence_vad(resampled, audio_vae_.config().sample_rate);
    pad_audio_for_patch_alignment(resampled, static_cast<size_t>(patch_len), PaddingMode::Right);

    PromptFeatures features;
    features.id = id;
    features.reference_feat = extract_prompt_features(audio_vae_,
                                                      *backend_,
                                                      resampled,
                                                      audio_vae_.config().sample_rate,
                                                      patch_size_value,
                                                      feat_dim_value);
    features.reference_audio_length =
        static_cast<int>(features.reference_feat.size() / static_cast<size_t>(patch_size_value * feat_dim_value));
    features.sample_rate = audio_vae_.config().sample_rate;
    features.patch_size = patch_size_value;
    features.feat_dim = feat_dim_value;
    const std::string now = make_timestamp_utc();
    features.created_at = now;
    features.updated_at = now;
    std::cerr << "[voice] encoded reference id=" << id
              << " reference_audio_length=" << features.reference_audio_length
              << " patch_size=" << features.patch_size
              << " feat_dim=" << features.feat_dim
              << " sample_rate=" << features.sample_rate
              << "\n";
    return features;
}

SynthesisResult VoxCPMServiceCore::synthesize(const SynthesisRequest& request) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!loaded_) {
        fail("Model core is not loaded");
    }
    return synthesize_locked(request);
}

SynthesisResult VoxCPMServiceCore::synthesize_locked(const SynthesisRequest& request) {
    if (request.text.empty()) {
        fail("Text input must not be empty");
    }
    const int patch_size_value = runtime_.config().patch_size;
    const int feat_dim_value = runtime_.config().feat_dim;
    const int patch_len = patch_size_value * audio_vae_.config().hop_length();
    const size_t expected_prompt_feat_size =
        static_cast<size_t>(request.prompt.prompt_audio_length) *
        static_cast<size_t>(patch_size_value) *
        static_cast<size_t>(feat_dim_value);
    const size_t expected_reference_feat_size =
        static_cast<size_t>(request.prompt.reference_audio_length) *
        static_cast<size_t>(patch_size_value) *
        static_cast<size_t>(feat_dim_value);

    if (request.prompt.prompt_audio_length < 0) {
        fail("Voice metadata is invalid: prompt_audio_length must be >= 0");
    }
    if (request.prompt.reference_audio_length < 0) {
        fail("Voice metadata is invalid: reference_audio_length must be >= 0");
    }
    if (request.prompt.patch_size != patch_size_value) {
        fail("Voice metadata patch_size does not match the loaded model");
    }
    if (request.prompt.feat_dim != feat_dim_value) {
        fail("Voice metadata feat_dim does not match the loaded model");
    }
    if (request.prompt.prompt_feat.size() != expected_prompt_feat_size) {
        fail("Voice metadata is inconsistent with stored prompt features");
    }
    if (request.prompt.reference_feat.size() != expected_reference_feat_size) {
        fail("Voice metadata is inconsistent with stored reference features");
    }
    if ((request.prompt.prompt_audio_length == 0) != request.prompt.prompt_text.empty()) {
        fail("Voice metadata is invalid: prompt_text must be provided iff prompt audio is present");
    }
    if (request.retry_badcase_max_times < 1) {
        fail("retry_badcase_max_times must be >= 1");
    }
    if (request.retry_badcase_ratio_threshold <= 0.0f) {
        fail("retry_badcase_ratio_threshold must be > 0");
    }

    // Request boundary reset: cached runtime graphs and backend graph bookkeeping
    // are rebuilt per synthesis request so reused service instances do not carry
    // stale graph state across calls.
    runtime_.reset_request_state();
    backend_->reset_request_state();

    const bool has_prompt_audio = request.prompt.prompt_audio_length > 0;
    const bool has_reference_audio = request.prompt.reference_audio_length > 0;
    std::string effective_text = request.text;
    if (has_prompt_audio) {
        const auto [stripped_text, stripped] = strip_hifi_control_prefix(request.text);
        if (stripped) {
            std::cerr << "[tts] Hi-Fi mode ignores control instructions; stripping the leading parenthesized prefix.\n";
            effective_text = stripped_text;
        }
    }
    bool retry_badcase = request.retry_badcase;
    if (retry_badcase && request.chunk_callback) {
        std::cerr << "[tts] retry_badcase is not supported with streaming chunks; disabling retries.\n";
        retry_badcase = false;
    }
    const PreparedConditioning prepared =
        build_conditioning(*split_tokenizer_, effective_text, request.prompt, patch_size_value, feat_dim_value);
    const int seq_len = static_cast<int>(prepared.full_text_tokens.size());
    std::cerr << "[tts] synth start seq_len=" << prepared.full_text_tokens.size()
              << " prompt_audio_length=" << request.prompt.prompt_audio_length
              << " reference_audio_length=" << request.prompt.reference_audio_length
              << " prompt_feat_size=" << request.prompt.prompt_feat.size()
              << "\n";
    const int prompt_audio_length = request.prompt.prompt_audio_length;
    const int target_text_token_count =
        std::max<int>(1, static_cast<int>(split_tokenizer_->tokenize(effective_text).size()));
    const int hard_max_len = decode_step_cap_for_service(backend_type_, seq_len);
    const int max_len = std::min({static_cast<int>(target_text_token_count * request.retry_badcase_ratio_threshold + 10.0f),
                                  hard_max_len,
                                  2000});
    constexpr int kMinLen = 2;
    const int max_attempts = retry_badcase ? std::max(1, request.retry_badcase_max_times) : 1;

    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        VoxCPMDecodeState state = runtime_.prefill(prepared.full_text_tokens,
                                                   prepared.text_mask,
                                                   prepared.feat,
                                                   prepared.feat_mask,
                                                   seq_len,
                                                   request.streaming_prefix_len);
        std::cerr << "[tts] prefill done seq_len=" << seq_len << " attempt=" << (attempt + 1) << "\n";
        const bool use_output_pool_timeline = should_use_output_pool_timeline(state, has_reference_audio, seq_len);

        std::mt19937 rng(std::random_device{}());
        std::vector<float> generated_steps;
        if (!use_output_pool_timeline) {
            generated_steps.reserve(static_cast<size_t>(max_len) * patch_size_value * feat_dim_value);
        }
        std::vector<float> noise;
        std::vector<float> stream_recent_frames;
        const size_t frame_stride = static_cast<size_t>(patch_size_value) * feat_dim_value;
        const int context_frames =
            (has_prompt_audio && request.streaming_prefix_len > 1)
                ? std::min(request.streaming_prefix_len - 1, prompt_audio_length)
                : 0;
        const bool use_fallback_streaming_window = request.chunk_callback && !use_output_pool_timeline;
        if (use_fallback_streaming_window) {
            stream_recent_frames.reserve(static_cast<size_t>(request.streaming_prefix_len) * frame_stride);
        }
        if (use_fallback_streaming_window && context_frames > 0) {
            const size_t context_offset = static_cast<size_t>(prompt_audio_length - context_frames) * frame_stride;
            stream_recent_frames.insert(stream_recent_frames.end(),
                                        request.prompt.prompt_feat.begin() + static_cast<std::ptrdiff_t>(context_offset),
                                        request.prompt.prompt_feat.end());
        }

        for (int step = 0; step < max_len; ++step) {
            fill_noise(noise, patch_size_value, feat_dim_value, rng);
            VoxCPMDecodeOptions decode_options;
            decode_options.export_patch_to_host = !use_output_pool_timeline;
            decode_options.publish_stop_logits_to_output = !use_output_pool_timeline;
            decode_options.publish_patch_to_output = !use_output_pool_timeline;
            decode_options.trust_persistent_state = use_output_pool_timeline;
            VoxCPMDecodeResult result = runtime_.decode(std::move(state),
                                                        noise,
                                                        request.inference_timesteps,
                                                        request.cfg_value,
                                                        decode_options);
            if (!use_output_pool_timeline) {
                generated_steps.insert(generated_steps.end(), result.output_0.begin(), result.output_0.end());
            }
            state = std::move(result.output_1);

            if (request.chunk_callback) {
                int recent_frame_count = 0;

                if (use_output_pool_timeline && state.audio_frame_count > 0) {
                    recent_frame_count = std::min(request.streaming_prefix_len, state.audio_frame_count);
                    const int frame_offset = state.audio_frame_count - recent_frame_count;
                    if (recent_frame_count > 0) {
                        std::vector<float> chunk_waveform =
                            decode_audio_from_output_pool(audio_vae_,
                                                          *backend_,
                                                          *state.output_pool,
                                                          frame_offset,
                                                          recent_frame_count,
                                                          patch_size_value,
                                                          feat_dim_value);
                        if (chunk_waveform.size() > static_cast<size_t>(patch_len)) {
                            chunk_waveform.erase(chunk_waveform.begin(),
                                                 chunk_waveform.end() - static_cast<std::ptrdiff_t>(patch_len));
                        }
                        clear_decode_graph_caches_after_streaming_audio_decode(runtime_, state);
                        request.chunk_callback(chunk_waveform);
                    }
                } else {
                    append_stream_frame(stream_recent_frames,
                                        result.output_0,
                                        request.streaming_prefix_len,
                                        patch_size_value,
                                        feat_dim_value);
                    recent_frame_count = static_cast<int>(stream_recent_frames.size() / frame_stride);
                    if (recent_frame_count > 0) {
                        std::vector<float> chunk_waveform = decode_audio_from_patch_major_frames(audio_vae_,
                                                                                                 *backend_,
                                                                                                 stream_recent_frames,
                                                                                                 patch_size_value,
                                                                                                 feat_dim_value);
                        if (chunk_waveform.size() > static_cast<size_t>(patch_len)) {
                            chunk_waveform.erase(chunk_waveform.begin(),
                                                 chunk_waveform.end() - static_cast<std::ptrdiff_t>(patch_len));
                        }
                        clear_decode_graph_caches_after_streaming_audio_decode(runtime_, state);
                        request.chunk_callback(chunk_waveform);
                    }
                }
            }

            if (step > kMinLen && result.output_2) {
                break;
            }
        }
        std::cerr << "[tts] decode loop done generated_frames="
                  << (use_output_pool_timeline
                          ? std::max(0, state.audio_frame_count - prompt_audio_length)
                          : static_cast<int>(generated_steps.size() / frame_stride))
                  << " attempt=" << (attempt + 1)
                  << "\n";

        const int generated_frames = use_output_pool_timeline
                                         ? std::max(0, state.audio_frame_count - prompt_audio_length)
                                         : static_cast<int>(generated_steps.size() / frame_stride);
        if (retry_badcase &&
            generated_frames >= static_cast<int>(target_text_token_count * request.retry_badcase_ratio_threshold) &&
            attempt + 1 < max_attempts) {
            std::cerr << "[tts] badcase detected: audio_text_ratio="
                      << (static_cast<float>(generated_frames) / static_cast<float>(target_text_token_count))
                      << ", retrying attempt " << (attempt + 2) << "/" << max_attempts << "\n";
            continue;
        }

        int prepended_context_frames = 0;
        const int total_frames = (has_prompt_audio && request.streaming_prefix_len > 1
                                      ? std::min(request.streaming_prefix_len - 1, prompt_audio_length)
                                      : 0) +
                                 generated_frames;
        const int total_patches = total_frames * patch_size_value;
        if (generated_frames == 0 || total_patches == 0) {
            fail("Model generated no audio patches");
        }

        std::vector<float> waveform;
        std::vector<float> latent;
        if (use_output_pool_timeline &&
            state.audio_frame_count >= prompt_audio_length + generated_frames) {
            const int frame_offset =
                std::max(0, prompt_audio_length - std::min(request.streaming_prefix_len - 1, prompt_audio_length));
            waveform = decode_audio_from_output_pool(audio_vae_,
                                                     *backend_,
                                                     *state.output_pool,
                                                     frame_offset,
                                                     total_frames,
                                                     patch_size_value,
                                                     feat_dim_value);
            prepended_context_frames = has_prompt_audio && request.streaming_prefix_len > 1
                                           ? std::min(request.streaming_prefix_len - 1, prompt_audio_length)
                                           : 0;
        } else {
            latent = build_decode_latent_sequence(request.prompt.prompt_feat,
                                                  prompt_audio_length,
                                                  generated_steps,
                                                  request.streaming_prefix_len,
                                                  patch_size_value,
                                                  feat_dim_value,
                                                  &prepended_context_frames);
            waveform = decode_audio(audio_vae_, *backend_, latent, total_patches, feat_dim_value);
        }
        if (has_prompt_audio) {
            const size_t trim = static_cast<size_t>(patch_len) * static_cast<size_t>(prepended_context_frames);
            if (waveform.size() > trim) {
                waveform.erase(waveform.begin(), waveform.begin() + static_cast<std::ptrdiff_t>(trim));
            }
        }

        return SynthesisResult{
            std::move(waveform),
            audio_vae_.config().output_sample_rate(),
            generated_frames,
        };
    }

    fail("Retry loop exhausted without producing an accepted sample");
}

int VoxCPMServiceCore::sample_rate() const {
    return audio_vae_.config().output_sample_rate();
}

int VoxCPMServiceCore::patch_size() const {
    return runtime_.config().patch_size;
}

int VoxCPMServiceCore::feat_dim() const {
    return runtime_.config().feat_dim;
}

std::string make_timestamp_utc() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &now_time);
#else
    gmtime_r(&now_time, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

bool is_valid_voice_id(const std::string& id) {
    if (id.empty()) {
        return false;
    }
    return std::all_of(id.begin(), id.end(), [](unsigned char c) {
        return std::isalnum(c) || c == '-' || c == '_' || c == '.';
    });
}

}  // namespace voxcpm
