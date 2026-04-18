#include <catch2/catch_test_macros.hpp>

#include "voxcpm/audio_io.h"
#include "voxcpm/server_common.h"
#include "test_config.h"

#include <array>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <vector>

namespace voxcpm {
namespace test {

TEST_CASE("Audio response format parser supports the documented formats", "[server]") {
    REQUIRE(parse_audio_response_format("mp3") == AudioResponseFormat::Mp3);
    REQUIRE(parse_audio_response_format("opus") == AudioResponseFormat::Opus);
    REQUIRE(parse_audio_response_format("flac") == AudioResponseFormat::Flac);
    REQUIRE(parse_audio_response_format("wav") == AudioResponseFormat::Wav);
    REQUIRE(parse_audio_response_format("pcm") == AudioResponseFormat::Pcm);
    REQUIRE_THROWS(parse_audio_response_format("aac"));
    REQUIRE_THROWS(parse_audio_response_format("bogus"));
}

TEST_CASE("Audio response format metadata matches the runtime encoders", "[server]") {
    REQUIRE(audio_response_format_name(AudioResponseFormat::Mp3) == std::string("mp3"));
    REQUIRE(audio_response_format_name(AudioResponseFormat::Opus) == std::string("opus"));
    REQUIRE(audio_response_format_name(AudioResponseFormat::Flac) == std::string("flac"));
    REQUIRE(audio_response_format_name(AudioResponseFormat::Wav) == std::string("wav"));
    REQUIRE(audio_response_format_name(AudioResponseFormat::Pcm) == std::string("pcm"));

    REQUIRE(audio_content_type(AudioResponseFormat::Mp3) == std::string("audio/mpeg"));
    REQUIRE(audio_content_type(AudioResponseFormat::Opus) == std::string("audio/ogg; codecs=opus"));
    REQUIRE(audio_content_type(AudioResponseFormat::Flac) == std::string("audio/flac"));
    REQUIRE(audio_content_type(AudioResponseFormat::Wav) == std::string("audio/wav"));
    REQUIRE(audio_content_type(AudioResponseFormat::Pcm) == std::string("application/octet-stream"));

    REQUIRE(audio_response_format_supported(AudioResponseFormat::Wav));
    REQUIRE(audio_response_format_supported(AudioResponseFormat::Pcm));
    REQUIRE(audio_response_format_supported(AudioResponseFormat::Flac));

#if VOXCPM_ENABLE_MP3
    REQUIRE(audio_response_format_supported(AudioResponseFormat::Mp3));
#else
    REQUIRE_FALSE(audio_response_format_supported(AudioResponseFormat::Mp3));
#endif

#if VOXCPM_ENABLE_OPUS
    REQUIRE(audio_response_format_supported(AudioResponseFormat::Opus));
#else
    REQUIRE_FALSE(audio_response_format_supported(AudioResponseFormat::Opus));
#endif
}

TEST_CASE("Audio encoder produces playable payloads for mp3, opus, wav, and pcm", "[server]") {
    constexpr int kSampleRate = 24000;
    constexpr float kPi = 3.14159265358979323846f;
    std::vector<float> waveform(480, 0.0f);
    for (size_t i = 0; i < waveform.size(); ++i) {
        waveform[i] = 0.1f * std::sin(2.0f * kPi * 220.0f * static_cast<float>(i) / static_cast<float>(kSampleRate));
    }

    const std::vector<uint8_t> wav = encode_audio(AudioResponseFormat::Wav, waveform, kSampleRate);
    REQUIRE(wav.size() > 44);
    REQUIRE(std::equal(wav.begin(), wav.begin() + 4, "RIFF"));

    const std::vector<uint8_t> pcm = encode_audio(AudioResponseFormat::Pcm, waveform, kSampleRate);
    REQUIRE(pcm.size() == waveform.size() * sizeof(int16_t));

#if VOXCPM_ENABLE_MP3
    const std::vector<uint8_t> mp3 = encode_audio(AudioResponseFormat::Mp3, waveform, kSampleRate);
    REQUIRE(mp3.size() > 0);
    REQUIRE(mp3.size() >= 3);
    REQUIRE((std::equal(mp3.begin(), mp3.begin() + 3, "ID3") || (mp3[0] == 0xFF && (mp3[1] & 0xE0) == 0xE0)));
#endif

#if VOXCPM_ENABLE_OPUS
    const std::vector<uint8_t> opus = encode_audio(AudioResponseFormat::Opus, waveform, kSampleRate);
    REQUIRE(opus.size() > 0);
    REQUIRE(opus.size() >= 8);
    REQUIRE(std::equal(opus.begin(), opus.begin() + 4, "OggS"));
    REQUIRE(std::search(opus.begin(), opus.end(), "OpusHead", "OpusHead" + 8) != opus.end());
#endif
}

TEST_CASE("Voice ids are restricted to filesystem-safe characters", "[server]") {
    REQUIRE(is_valid_voice_id("voice_123"));
    REQUIRE(is_valid_voice_id("voice.demo-1"));
    REQUIRE_FALSE(is_valid_voice_id(""));
    REQUIRE_FALSE(is_valid_voice_id("../escape"));
    REQUIRE_FALSE(is_valid_voice_id("voice with space"));
}

TEST_CASE("VoiceStore persists manifest and prompt features round-trip", "[server]") {
    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / "voxcpm_server_voice_store_test";
    std::filesystem::remove_all(root);

    VoiceStore store(root.string());
    PromptFeatures features;
    features.id = "voice_abc";
    features.prompt_text = "hello";
    features.prompt_feat = {1.0f, 2.5f, -3.0f, 4.0f};
    features.prompt_audio_length = 2;
    features.reference_feat = {9.0f, 8.0f, 7.0f, 6.0f};
    features.reference_audio_length = 2;
    features.sample_rate = 16000;
    features.patch_size = 2;
    features.feat_dim = 2;
    features.created_at = "2026-03-18T00:00:00Z";
    features.updated_at = "2026-03-18T00:00:01Z";

    store.save_voice(features);
    REQUIRE(store.has_voice(features.id));

    const PromptFeatures loaded = store.load_voice(features.id);
    REQUIRE(loaded.id == features.id);
    REQUIRE(loaded.prompt_text == features.prompt_text);
    REQUIRE(loaded.prompt_feat == features.prompt_feat);
    REQUIRE(loaded.prompt_audio_length == features.prompt_audio_length);
    REQUIRE(loaded.reference_feat == features.reference_feat);
    REQUIRE(loaded.reference_audio_length == features.reference_audio_length);
    REQUIRE(loaded.sample_rate == features.sample_rate);
    REQUIRE(loaded.patch_size == features.patch_size);
    REQUIRE(loaded.feat_dim == features.feat_dim);

    const VoiceMetadata metadata = store.load_metadata(features.id);
    REQUIRE(metadata.id == features.id);
    REQUIRE(metadata.prompt_text == features.prompt_text);
    REQUIRE(metadata.reference_audio_length == features.reference_audio_length);

    store.delete_voice(features.id);
    REQUIRE_FALSE(store.has_voice(features.id));
    std::filesystem::remove_all(root);
}

TEST_CASE("Service synthesize runs end-to-end with encoded prompt audio", "[server][integration]") {
    const std::string model_path = get_model_path();
    REQUIRE(std::filesystem::exists(model_path));

    VoxCPMServiceCore service(model_path, BackendType::CPU, 2);
    service.load();
    REQUIRE(service.loaded());

    constexpr int kInputSampleRate = 16000;
    constexpr float kPi = 3.14159265358979323846f;
    std::vector<float> mono_audio(1600, 0.0f);
    for (size_t i = 0; i < mono_audio.size(); ++i) {
        const float phase = 2.0f * kPi * 220.0f * static_cast<float>(i) / static_cast<float>(kInputSampleRate);
        mono_audio[i] = 0.05f * std::sin(phase);
    }

    PromptFeatures prompt = service.encode_prompt_audio("voice_test", "你好", mono_audio, kInputSampleRate);
    REQUIRE(prompt.prompt_audio_length > 0);
    REQUIRE(prompt.sample_rate > 0);
    REQUIRE(prompt.patch_size == service.patch_size());
    REQUIRE(prompt.feat_dim == service.feat_dim());
    REQUIRE(prompt.prompt_feat.size() ==
            static_cast<size_t>(prompt.prompt_audio_length * prompt.patch_size * prompt.feat_dim));

    PromptFeatures reference = service.encode_reference_audio("voice_ref", mono_audio, kInputSampleRate);
    REQUIRE(reference.reference_audio_length > 0);
    REQUIRE(reference.prompt_audio_length == 0);
    REQUIRE(reference.prompt_text.empty());
    REQUIRE(reference.sample_rate == prompt.sample_rate);
    REQUIRE(reference.patch_size == service.patch_size());
    REQUIRE(reference.feat_dim == service.feat_dim());
    REQUIRE(reference.reference_feat.size() ==
            static_cast<size_t>(reference.reference_audio_length * reference.patch_size * reference.feat_dim));

    prompt.reference_feat = reference.reference_feat;
    prompt.reference_audio_length = reference.reference_audio_length;

    int chunk_count = 0;
    size_t last_chunk_size = 0;
    SynthesisRequest request;
    request.text = "测试";
    request.prompt = prompt;
    request.cfg_value = 1.5f;
    request.inference_timesteps = 4;
    request.streaming_prefix_len = 2;
    request.chunk_callback = [&](const std::vector<float>& chunk) {
        ++chunk_count;
        last_chunk_size = chunk.size();
    };

    const SynthesisResult result = service.synthesize(request);
    REQUIRE(result.sample_rate == service.sample_rate());
    REQUIRE(result.generated_frames > 0);
    REQUIRE_FALSE(result.waveform.empty());
    REQUIRE(std::all_of(result.waveform.begin(), result.waveform.end(), [](float value) {
        return std::isfinite(value);
    }));
    REQUIRE(chunk_count > 0);
    REQUIRE(last_chunk_size > 0);
}

TEST_CASE("Service synthesize resets request-scoped runtime state between calls", "[server][integration]") {
    const std::string model_path = get_model_path();
    REQUIRE(std::filesystem::exists(model_path));

    VoxCPMServiceCore service(model_path, BackendType::CPU, 2);
    service.load();
    REQUIRE(service.loaded());

    constexpr int kInputSampleRate = 16000;
    constexpr float kPi = 3.14159265358979323846f;
    std::vector<float> mono_audio(1600, 0.0f);
    for (size_t i = 0; i < mono_audio.size(); ++i) {
        const float phase = 2.0f * kPi * 220.0f * static_cast<float>(i) / static_cast<float>(kInputSampleRate);
        mono_audio[i] = 0.05f * std::sin(phase);
    }

    PromptFeatures prompt = service.encode_prompt_audio("voice_repeat", "你好", mono_audio, kInputSampleRate);
    REQUIRE(prompt.prompt_audio_length > 0);
    REQUIRE(prompt.sample_rate > 0);

    SynthesisRequest request;
    request.text = "测试";
    request.prompt = prompt;
    request.cfg_value = 1.5f;
    request.inference_timesteps = 4;
    request.streaming_prefix_len = 2;

    const SynthesisResult first = service.synthesize(request);
    const SynthesisResult second = service.synthesize(request);

    REQUIRE(first.sample_rate == service.sample_rate());
    REQUIRE(second.sample_rate == service.sample_rate());
    REQUIRE(first.generated_frames > 0);
    REQUIRE(second.generated_frames > 0);
    REQUIRE_FALSE(first.waveform.empty());
    REQUIRE_FALSE(second.waveform.empty());
    REQUIRE(std::all_of(first.waveform.begin(), first.waveform.end(), [](float value) {
        return std::isfinite(value);
    }));
    REQUIRE(std::all_of(second.waveform.begin(), second.waveform.end(), [](float value) {
        return std::isfinite(value);
    }));
}

TEST_CASE("Service synthesize handles longer text inputs without graph context exhaustion", "[server][integration]") {
    const std::string model_path = get_model_path();
    REQUIRE(std::filesystem::exists(model_path));

    VoxCPMServiceCore service(model_path, BackendType::CPU, 2);
    service.load();
    REQUIRE(service.loaded());

    constexpr int kInputSampleRate = 16000;
    constexpr float kPi = 3.14159265358979323846f;
    std::vector<float> mono_audio(1600, 0.0f);
    for (size_t i = 0; i < mono_audio.size(); ++i) {
        const float phase = 2.0f * kPi * 220.0f * static_cast<float>(i) / static_cast<float>(kInputSampleRate);
        mono_audio[i] = 0.05f * std::sin(phase);
    }

    PromptFeatures prompt = service.encode_prompt_audio("voice_long", "你好", mono_audio, kInputSampleRate);
    REQUIRE(prompt.prompt_audio_length > 0);

    const std::array<int, 2> repeat_counts = {16, 32};
    for (int repeat_count : repeat_counts) {
        std::string long_text;
        for (int i = 0; i < repeat_count; ++i) {
            long_text += "这是一个用于回归测试的较长句子。";
        }
        REQUIRE(long_text.size() <= 4096);

        SynthesisRequest request;
        request.text = long_text;
        request.prompt = prompt;
        request.cfg_value = 1.5f;
        request.inference_timesteps = 1;
        request.streaming_prefix_len = 1;

        const SynthesisResult result = service.synthesize(request);
        REQUIRE(result.sample_rate == service.sample_rate());
        REQUIRE(result.generated_frames > 0);
        REQUIRE_FALSE(result.waveform.empty());
        REQUIRE(std::all_of(result.waveform.begin(), result.waveform.end(), [](float value) {
            return std::isfinite(value);
        }));
    }
}

}  // namespace test
}  // namespace voxcpm
