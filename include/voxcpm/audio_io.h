#ifndef VOXCPM_AUDIO_IO_H
#define VOXCPM_AUDIO_IO_H

#include <cstdint>
#include <string>
#include <vector>

namespace voxcpm {

enum class AudioResponseFormat {
    Mp3,
    Opus,
    Flac,
    Wav,
    Pcm,
};

struct DecodedAudio {
    int sample_rate = 0;
    int channels = 0;
    std::vector<float> samples;
};

DecodedAudio decode_audio_from_memory(const void* data, size_t size);
std::vector<float> convert_to_mono(const DecodedAudio& audio);
std::vector<float> resample_audio_linear(const std::vector<float>& input, double speed);
std::vector<float> resample_audio_to_rate(const std::vector<float>& input, int src_rate, int dst_rate);
std::vector<float> trim_audio_silence_vad(const std::vector<float>& input,
                                          int sample_rate,
                                          float max_silence_ms = 200.0f,
                                          float top_db = 35.0f);

AudioResponseFormat parse_audio_response_format(const std::string& format);
const char* audio_response_format_name(AudioResponseFormat format);
const char* audio_content_type(AudioResponseFormat format);
bool audio_response_format_supported(AudioResponseFormat format);

std::vector<uint8_t> encode_audio(AudioResponseFormat format,
                                  const std::vector<float>& waveform,
                                  int sample_rate);
std::string base64_encode(const uint8_t* data, size_t size);

}  // namespace voxcpm

#endif  // VOXCPM_AUDIO_IO_H
