#ifndef VOXCPM_SERVER_COMMON_H
#define VOXCPM_SERVER_COMMON_H

#include "voxcpm/audio-vae.h"
#include "voxcpm/backend.h"
#include "voxcpm/tokenizer.h"
#include "voxcpm/voxcpm.h"
#include "voxcpm/weight-store.h"

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace voxcpm {

struct PromptFeatures {
    std::string id;
    std::string prompt_text;
    std::vector<float> prompt_feat;    // Left-padded.
    int prompt_audio_length = 0;
    std::vector<float> reference_feat; // Right-padded.
    int reference_audio_length = 0;
    int sample_rate = 0;
    int patch_size = 0;
    int feat_dim = 0;
    std::string created_at;
    std::string updated_at;
};

struct VoiceMetadata {
    std::string id;
    std::string prompt_text;
    int prompt_audio_length = 0;
    int reference_audio_length = 0;
    int sample_rate = 0;
    int patch_size = 0;
    int feat_dim = 0;
    std::string created_at;
    std::string updated_at;
};

struct SynthesisRequest {
    std::string text;
    PromptFeatures prompt;
    float cfg_value = 2.0f;
    int inference_timesteps = 10;
    bool has_uploaded_prompt_audio = false;
    int streaming_prefix_len = 4;
    bool retry_badcase = false;
    int retry_badcase_max_times = 3;
    float retry_badcase_ratio_threshold = 6.0f;
    int max_decode_steps = 0;
    std::function<void(const std::vector<float>&)> chunk_callback;
};

struct SynthesisResult {
    std::vector<float> waveform;
    int sample_rate = 0;
    int generated_frames = 0;
};

class VoiceStore {
public:
    explicit VoiceStore(std::string root_dir);

    bool has_voice(const std::string& id) const;
    void save_voice(const PromptFeatures& features);
    PromptFeatures load_voice(const std::string& id) const;
    VoiceMetadata load_metadata(const std::string& id) const;
    void delete_voice(const std::string& id);

    const std::string& root_dir() const { return root_dir_; }

private:
    std::string root_dir_;
};

class VoxCPMServiceCore {
public:
    VoxCPMServiceCore(std::string model_path, BackendType backend_type, int threads);

    void load();
    PromptFeatures encode_prompt_audio(const std::string& id,
                                      const std::string& prompt_text,
                                      const std::vector<float>& mono_audio,
                                      int sample_rate);
    PromptFeatures encode_reference_audio(const std::string& id,
                                         const std::vector<float>& mono_audio,
                                         int sample_rate);
    SynthesisResult synthesize(const SynthesisRequest& request);
    std::vector<float> convert_prompt_feat_to_reference_feat(
                                      const std::vector<float>& prompt_feat);

    int sample_rate() const;
    int patch_size() const;
    int feat_dim() const;
    bool loaded() const { return loaded_; }

private:
    PromptFeatures encode_prompt_audio_locked(const std::string& id,
                                             const std::string& prompt_text,
                                             const std::vector<float>& mono_audio,
                                             int sample_rate);
    PromptFeatures encode_reference_audio_locked(const std::string& id,
                                                const std::vector<float>& mono_audio,
                                                int sample_rate);
    SynthesisResult synthesize_locked(const SynthesisRequest& request);

    void prepare_combined_features(std::vector<float>& combined_features,
                                  const int patch_size_value,
                                  const int feat_dim_value,
                                  PromptFeatures& mode_prompt,
                                  const std::string& effective_text,
                                  int& effective_prompt_audio_length);

    std::string model_path_;
    BackendType backend_type_;
    int threads_ = 4;
    bool loaded_ = false;

    std::shared_ptr<VoxCPMWeightStore> store_;
    std::unique_ptr<VoxCPMBackend> backend_;
    VoxCPMRuntime runtime_;
    AudioVAE audio_vae_;
    std::unique_ptr<VoxCPMTokenizer> tokenizer_;
    std::unique_ptr<ChineseCharSplitTokenizer> split_tokenizer_;

    mutable std::mutex mutex_;
};

std::string make_timestamp_utc();
bool is_valid_voice_id(const std::string& id);

}  // namespace voxcpm

#endif  // VOXCPM_SERVER_COMMON_H
