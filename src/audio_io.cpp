#include "voxcpm/audio_io.h"

#include "miniaudio.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>

#ifndef VOXCPM_ENABLE_MP3
#define VOXCPM_ENABLE_MP3 1
#endif

#ifndef VOXCPM_ENABLE_OPUS
#define VOXCPM_ENABLE_OPUS 1
#endif

namespace voxcpm {

namespace {

struct MemoryWriter {
    std::vector<uint8_t> bytes;
};

void fail_audio(const std::string& message) {
    throw std::runtime_error(message);
}

std::string shell_quote(const std::filesystem::path& path) {
    std::string quoted = "'";
    const std::string raw = path.string();
    for (char ch : raw) {
        if (ch == '\'') {
            quoted += "'\"'\"'";
        } else {
            quoted.push_back(ch);
        }
    }
    quoted.push_back('\'');
    return quoted;
}

std::filesystem::path make_temp_codec_path(const char* suffix) {
    const auto stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::random_device rd;
    std::mt19937_64 rng((static_cast<uint64_t>(rd()) << 32) ^ static_cast<uint64_t>(stamp));
    std::uniform_int_distribution<uint64_t> dist;
    const std::filesystem::path base = std::filesystem::temp_directory_path();
    return base / ("voxcpm-audio-" + std::to_string(stamp) + "-" + std::to_string(dist(rng)) + suffix);
}

std::vector<uint8_t> read_binary_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in.is_open()) {
        fail_audio("Failed to read encoded audio payload: " + path.string());
    }

    const std::streamsize size = in.tellg();
    if (size < 0) {
        fail_audio("Failed to read encoded audio payload: " + path.string());
    }
    std::vector<uint8_t> bytes(static_cast<size_t>(size), 0);
    in.seekg(0, std::ios::beg);
    if (size > 0) {
        in.read(reinterpret_cast<char*>(bytes.data()), size);
    }
    return bytes;
}

std::vector<uint8_t> write_wav_or_pcm_bytes(const std::vector<float>& waveform) {
    std::vector<uint8_t> bytes(waveform.size() * sizeof(int16_t), 0);
    for (size_t i = 0; i < waveform.size(); ++i) {
        const float clamped = std::max(-1.0f, std::min(1.0f, waveform[i]));
        const int16_t pcm = static_cast<int16_t>(std::lrint(clamped * 32767.0f));
        std::memcpy(bytes.data() + i * sizeof(int16_t), &pcm, sizeof(pcm));
    }
    return bytes;
}

std::vector<uint8_t> encode_ogg_opus_via_ffmpeg(const std::vector<float>& waveform, int sample_rate) {
    if (sample_rate <= 0) {
        fail_audio("Invalid sample rate for opus encoding");
    }

    const std::filesystem::path input_path = make_temp_codec_path(".pcm");
    const std::filesystem::path output_path = make_temp_codec_path(".opus");

    struct Cleanup {
        std::filesystem::path input;
        std::filesystem::path output;
        ~Cleanup() {
            std::error_code ec;
            if (!input.empty()) {
                std::filesystem::remove(input, ec);
            }
            if (!output.empty()) {
                std::filesystem::remove(output, ec);
            }
        }
    } cleanup{input_path, output_path};

    {
        std::ofstream out(input_path, std::ios::binary);
        if (!out.is_open()) {
            fail_audio("Failed to stage pcm input for opus encoding");
        }
        const std::vector<uint8_t> pcm_bytes = write_wav_or_pcm_bytes(waveform);
        if (!pcm_bytes.empty()) {
            out.write(reinterpret_cast<const char*>(pcm_bytes.data()), static_cast<std::streamsize>(pcm_bytes.size()));
        }
    }

    std::ostringstream command;
    command << "ffmpeg -hide_banner -loglevel error -y"
            << " -f s16le -ar " << sample_rate << " -ac 1"
            << " -i " << shell_quote(input_path)
            << " -c:a libopus -application audio -vbr on -b:a 96k -f ogg "
            << shell_quote(output_path);
    const int rc = std::system(command.str().c_str());
    if (rc != 0) {
        fail_audio("Failed to encode audio as opus");
    }

    return read_binary_file(output_path);
}

std::vector<uint8_t> encode_mp3_via_ffmpeg(const std::vector<float>& waveform, int sample_rate) {
    if (sample_rate <= 0) {
        fail_audio("Invalid sample rate for mp3 encoding");
    }

    const std::filesystem::path input_path = make_temp_codec_path(".pcm");
    const std::filesystem::path output_path = make_temp_codec_path(".mp3");

    struct Cleanup {
        std::filesystem::path input;
        std::filesystem::path output;
        ~Cleanup() {
            std::error_code ec;
            if (!input.empty()) {
                std::filesystem::remove(input, ec);
            }
            if (!output.empty()) {
                std::filesystem::remove(output, ec);
            }
        }
    } cleanup{input_path, output_path};

    {
        std::ofstream out(input_path, std::ios::binary);
        if (!out.is_open()) {
            fail_audio("Failed to stage pcm input for mp3 encoding");
        }
        const std::vector<uint8_t> pcm_bytes = write_wav_or_pcm_bytes(waveform);
        if (!pcm_bytes.empty()) {
            out.write(reinterpret_cast<const char*>(pcm_bytes.data()), static_cast<std::streamsize>(pcm_bytes.size()));
        }
    }

    std::ostringstream command;
    command << "ffmpeg -hide_banner -loglevel error -y"
            << " -f s16le -ar " << sample_rate << " -ac 1"
            << " -i " << shell_quote(input_path)
            << " -c:a libmp3lame -b:a 128k -f mp3 "
            << shell_quote(output_path);
    const int rc = std::system(command.str().c_str());
    if (rc != 0) {
        fail_audio("Failed to encode audio as mp3");
    }

    return read_binary_file(output_path);
}

ma_result encoder_write(ma_encoder* encoder, const void* pBufferIn, size_t bytesToWrite, size_t* pBytesWritten) {
    auto* writer = static_cast<MemoryWriter*>(encoder->pUserData);
    const auto* bytes = static_cast<const uint8_t*>(pBufferIn);
    writer->bytes.insert(writer->bytes.end(), bytes, bytes + bytesToWrite);
    if (pBytesWritten != nullptr) {
        *pBytesWritten = bytesToWrite;
    }
    return MA_SUCCESS;
}

ma_result encoder_seek(ma_encoder* encoder, ma_int64 byteOffset, ma_seek_origin origin) {
    auto* writer = static_cast<MemoryWriter*>(encoder->pUserData);
    size_t target = 0;
    if (origin == ma_seek_origin_start) {
        if (byteOffset < 0) {
            return MA_INVALID_ARGS;
        }
        target = static_cast<size_t>(byteOffset);
    } else if (origin == ma_seek_origin_current) {
        if (byteOffset < 0 && static_cast<size_t>(-byteOffset) > writer->bytes.size()) {
            return MA_INVALID_ARGS;
        }
        target = byteOffset >= 0 ? writer->bytes.size() + static_cast<size_t>(byteOffset)
                                 : writer->bytes.size() - static_cast<size_t>(-byteOffset);
    } else {
        return MA_INVALID_OPERATION;
    }

    writer->bytes.resize(target);
    return MA_SUCCESS;
}

ma_encoding_format to_ma_encoding_format(AudioResponseFormat format) {
    switch (format) {
        case AudioResponseFormat::Mp3:
            return ma_encoding_format_mp3;
        case AudioResponseFormat::Flac:
            return ma_encoding_format_flac;
        case AudioResponseFormat::Wav:
            return ma_encoding_format_wav;
        default:
            return ma_encoding_format_unknown;
    }
}

std::vector<uint8_t> encode_pcm16_wav(const std::vector<float>& waveform, int sample_rate) {
    const std::vector<uint8_t> pcm_bytes = write_wav_or_pcm_bytes(waveform);
    const uint16_t channels = 1;
    const uint16_t bits_per_sample = 16;
    const uint32_t byte_rate = static_cast<uint32_t>(sample_rate) * channels * (bits_per_sample / 8);
    const uint16_t block_align = channels * (bits_per_sample / 8);
    const uint32_t data_size = static_cast<uint32_t>(pcm_bytes.size());
    const uint32_t riff_size = 36 + data_size;

    std::vector<uint8_t> bytes;
    bytes.reserve(44 + data_size);

    const auto append_bytes = [&](const void* ptr, size_t len) {
        const auto* begin = static_cast<const uint8_t*>(ptr);
        bytes.insert(bytes.end(), begin, begin + len);
    };

    const uint32_t fmt_size = 16;
    const uint16_t audio_format = 1;

    append_bytes("RIFF", 4);
    append_bytes(&riff_size, sizeof(riff_size));
    append_bytes("WAVE", 4);
    append_bytes("fmt ", 4);
    append_bytes(&fmt_size, sizeof(fmt_size));
    append_bytes(&audio_format, sizeof(audio_format));
    append_bytes(&channels, sizeof(channels));
    append_bytes(&sample_rate, sizeof(sample_rate));
    append_bytes(&byte_rate, sizeof(byte_rate));
    append_bytes(&block_align, sizeof(block_align));
    append_bytes(&bits_per_sample, sizeof(bits_per_sample));
    append_bytes("data", 4);
    append_bytes(&data_size, sizeof(data_size));
    bytes.insert(bytes.end(), pcm_bytes.begin(), pcm_bytes.end());

    return bytes;
}

}  // namespace

DecodedAudio decode_audio_from_memory(const void* data, size_t size) {
    ma_decoder_config config = ma_decoder_config_init(ma_format_f32, 0, 0);
    ma_decoder decoder;
    if (ma_decoder_init_memory(data, size, &config, &decoder) != MA_SUCCESS) {
        fail_audio("Failed to decode uploaded audio");
    }

    ma_uint64 frame_count = 0;
    if (ma_decoder_get_length_in_pcm_frames(&decoder, &frame_count) != MA_SUCCESS) {
        ma_decoder_uninit(&decoder);
        fail_audio("Failed to inspect uploaded audio length");
    }

    std::vector<float> samples(static_cast<size_t>(frame_count) * decoder.outputChannels, 0.0f);
    ma_uint64 frames_read = 0;
    if (ma_decoder_read_pcm_frames(&decoder, samples.data(), frame_count, &frames_read) != MA_SUCCESS) {
        ma_decoder_uninit(&decoder);
        fail_audio("Failed to read uploaded audio samples");
    }
    const int output_sample_rate = static_cast<int>(decoder.outputSampleRate);
    const int output_channels = static_cast<int>(decoder.outputChannels);
    ma_decoder_uninit(&decoder);
    samples.resize(static_cast<size_t>(frames_read) * static_cast<size_t>(output_channels));

    return DecodedAudio{
        output_sample_rate,
        output_channels,
        std::move(samples),
    };
}

std::vector<float> convert_to_mono(const DecodedAudio& audio) {
    if (audio.channels <= 1) {
        return audio.samples;
    }

    const size_t frame_count = audio.samples.size() / static_cast<size_t>(audio.channels);
    std::vector<float> mono(frame_count, 0.0f);
    for (size_t frame = 0; frame < frame_count; ++frame) {
        float sum = 0.0f;
        for (int channel = 0; channel < audio.channels; ++channel) {
            sum += audio.samples[frame * static_cast<size_t>(audio.channels) + static_cast<size_t>(channel)];
        }
        mono[frame] = sum / static_cast<float>(audio.channels);
    }
    return mono;
}

std::vector<float> resample_audio_to_rate(const std::vector<float>& input, int src_rate, int dst_rate) {
    if (src_rate <= 0 || dst_rate <= 0 || input.empty() || src_rate == dst_rate) {
        return input;
    }

    const double scale = static_cast<double>(dst_rate) / static_cast<double>(src_rate);
    const size_t out_size = std::max<size_t>(1, static_cast<size_t>(std::llround(static_cast<double>(input.size()) * scale)));
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

std::vector<float> trim_audio_silence_vad(const std::vector<float>& input,
                                          int sample_rate,
                                          float max_silence_ms,
                                          float top_db) {
    if (input.empty() || sample_rate <= 0) {
        return input;
    }

    constexpr int kFrameLength = 2048;
    constexpr int kHopLength = 512;
    const float ref = *std::max_element(input.begin(), input.end(), [](float a, float b) {
        return std::fabs(a) < std::fabs(b);
    });
    if (std::fabs(ref) <= 0.0f) {
        return input;
    }

    const float threshold = std::fabs(ref) * std::pow(10.0f, -top_db / 20.0f);
    const size_t n = input.size();
    int first_voice_frame = -1;
    int last_voice_frame = -1;

    for (size_t idx = 0, frame = 0; idx < n; idx += kHopLength, ++frame) {
        const size_t frame_end = std::min(idx + static_cast<size_t>(kFrameLength), n);
        const size_t frame_size = frame_end - idx;
        if (frame_size == 0) {
            break;
        }
        double energy = 0.0;
        for (size_t i = idx; i < frame_end; ++i) {
            energy += static_cast<double>(input[i]) * static_cast<double>(input[i]);
        }
        const float rms = static_cast<float>(std::sqrt(energy / static_cast<double>(frame_size)));
        if (rms >= threshold) {
            if (first_voice_frame < 0) {
                first_voice_frame = static_cast<int>(frame);
            }
            last_voice_frame = static_cast<int>(frame);
        }
        if (frame_end == n) {
            break;
        }
    }

    if (first_voice_frame < 0 || last_voice_frame < 0) {
        return input;
    }

    const int max_silence_samples = std::max(0, static_cast<int>(std::lround(max_silence_ms * sample_rate / 1000.0f)));
    const int start = std::max(0, first_voice_frame * kHopLength - max_silence_samples);
    const int end = std::min(static_cast<int>(n),
                             (last_voice_frame + 1) * kHopLength + (kFrameLength - kHopLength) + max_silence_samples);
    if (start >= end) {
        return input;
    }
    return std::vector<float>(input.begin() + start, input.begin() + end);
}

std::vector<float> resample_audio_linear(const std::vector<float>& input, double speed) {
    if (input.empty() || speed <= 0.0 || std::abs(speed - 1.0) < 1e-6) {
        return input;
    }

    const double scale = 1.0 / speed;
    const size_t out_size = std::max<size_t>(1, static_cast<size_t>(std::llround(static_cast<double>(input.size()) * scale)));
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

AudioResponseFormat parse_audio_response_format(const std::string& format) {
    if (format == "mp3" || format.empty()) return AudioResponseFormat::Mp3;
    if (format == "opus") return AudioResponseFormat::Opus;
    if (format == "flac") return AudioResponseFormat::Flac;
    if (format == "wav") return AudioResponseFormat::Wav;
    if (format == "pcm") return AudioResponseFormat::Pcm;
    fail_audio("Unsupported response_format: " + format + " (supported: mp3, opus, flac, wav, pcm)");
    return AudioResponseFormat::Mp3;
}

const char* audio_response_format_name(AudioResponseFormat format) {
    switch (format) {
        case AudioResponseFormat::Mp3: return "mp3";
        case AudioResponseFormat::Opus: return "opus";
        case AudioResponseFormat::Flac: return "flac";
        case AudioResponseFormat::Wav: return "wav";
        case AudioResponseFormat::Pcm: return "pcm";
        default: return "unknown";
    }
}

const char* audio_content_type(AudioResponseFormat format) {
    switch (format) {
        case AudioResponseFormat::Mp3: return "audio/mpeg";
        case AudioResponseFormat::Opus: return "audio/ogg; codecs=opus";
        case AudioResponseFormat::Flac: return "audio/flac";
        case AudioResponseFormat::Wav: return "audio/wav";
        case AudioResponseFormat::Pcm: return "application/octet-stream";
        default: return "application/octet-stream";
    }
}

bool audio_response_format_supported(AudioResponseFormat format) {
    switch (format) {
        case AudioResponseFormat::Mp3:
            return VOXCPM_ENABLE_MP3 != 0;
        case AudioResponseFormat::Opus:
            return VOXCPM_ENABLE_OPUS != 0;
        case AudioResponseFormat::Flac:
        case AudioResponseFormat::Wav:
        case AudioResponseFormat::Pcm:
            return true;
        default:
            return false;
    }
}

std::vector<uint8_t> encode_audio(AudioResponseFormat format,
                                  const std::vector<float>& waveform,
                                  int sample_rate) {
    if (format == AudioResponseFormat::Wav) {
        return encode_pcm16_wav(waveform, sample_rate);
    }

    if (format == AudioResponseFormat::Pcm) {
        return write_wav_or_pcm_bytes(waveform);
    }

    if (!audio_response_format_supported(format)) {
        fail_audio(std::string("this build does not include ") + audio_response_format_name(format) +
                   " encoder support");
    }

    if (format == AudioResponseFormat::Opus) {
        return encode_ogg_opus_via_ffmpeg(waveform, sample_rate);
    }

    const ma_encoding_format ma_format = to_ma_encoding_format(format);
    if (ma_format == ma_encoding_format_unknown) {
        fail_audio(std::string("Requested response_format is not supported by this build: ") +
                   audio_response_format_name(format));
    }

    MemoryWriter writer;
    ma_encoder_config config = ma_encoder_config_init(ma_format, ma_format_f32, 1, static_cast<ma_uint32>(sample_rate));
    ma_encoder encoder;
    if (ma_encoder_init(encoder_write, encoder_seek, &writer, &config, &encoder) != MA_SUCCESS) {
        // Some miniaudio builds expose MP3 decoding without a working MP3 encoder, so keep a fallback.
        if (format == AudioResponseFormat::Mp3) {
            return encode_mp3_via_ffmpeg(waveform, sample_rate);
        }
        fail_audio(std::string("Failed to initialize encoder for format ") + audio_response_format_name(format));
    }

    if (ma_encoder_write_pcm_frames(&encoder, waveform.data(), waveform.size(), nullptr) != MA_SUCCESS) {
        ma_encoder_uninit(&encoder);
        if (format == AudioResponseFormat::Mp3) {
            return encode_mp3_via_ffmpeg(waveform, sample_rate);
        }
        fail_audio(std::string("Failed to encode audio as ") + audio_response_format_name(format));
    }
    ma_encoder_uninit(&encoder);
    return writer.bytes;
}

std::string base64_encode(const uint8_t* data, size_t size) {
    static constexpr char kAlphabet[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    std::string out;
    out.reserve(((size + 2) / 3) * 4);
    size_t i = 0;
    while (i + 3 <= size) {
        const uint32_t chunk = (static_cast<uint32_t>(data[i]) << 16) |
                               (static_cast<uint32_t>(data[i + 1]) << 8) |
                               static_cast<uint32_t>(data[i + 2]);
        out.push_back(kAlphabet[(chunk >> 18) & 0x3F]);
        out.push_back(kAlphabet[(chunk >> 12) & 0x3F]);
        out.push_back(kAlphabet[(chunk >> 6) & 0x3F]);
        out.push_back(kAlphabet[chunk & 0x3F]);
        i += 3;
    }

    const size_t remaining = size - i;
    if (remaining > 0) {
        uint32_t chunk = static_cast<uint32_t>(data[i]) << 16;
        if (remaining == 2) {
            chunk |= static_cast<uint32_t>(data[i + 1]) << 8;
        }
        out.push_back(kAlphabet[(chunk >> 18) & 0x3F]);
        out.push_back(kAlphabet[(chunk >> 12) & 0x3F]);
        out.push_back(remaining == 2 ? kAlphabet[(chunk >> 6) & 0x3F] : '=');
        out.push_back('=');
    }

    return out;
}

}  // namespace voxcpm
