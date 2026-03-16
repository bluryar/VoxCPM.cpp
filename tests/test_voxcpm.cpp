/**
 * @file test_voxcpm.cpp
 * @brief Trace-driven tests for VoxCPM Prefill/Decode runtime
 */

#include <catch2/catch_test_macros.hpp>

#include "voxcpm/backend.h"
#include "voxcpm/context.h"
#include "voxcpm/voxcpm.h"
#include "voxcpm/weight-store.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace voxcpm {
namespace test {

namespace {

constexpr const char* kModelPath = "/home/orangepi/Codes/ggbond/VoxCPM.cpp/models/voxcpm1.5.gguf";
constexpr const char* kPrefillTracePath = "/home/orangepi/Codes/ggbond/VoxCPM.cpp/tests/fixtures/trace/trace_Prefill_one.jsonl";
constexpr const char* kDecodeTracePath = "/home/orangepi/Codes/ggbond/VoxCPM.cpp/tests/fixtures/trace/trace_Decode_one.jsonl";
constexpr int kPrefillSeqLen = 100;
constexpr int kDecodeCacheSeqLen = 102;
constexpr float kHiddenTolerance = 0.10f;
constexpr float kCacheTolerance = 0.10f;
constexpr float kPredTolerance = 0.12f;
constexpr float kMaxMismatchRate = 0.05f;

struct RuntimeMetadataExpectations {
    int patch_size = 0;
    int feat_dim = 0;
    int max_length = 0;
    int residual_lm_layers = 0;
};

struct TensorStats {
    float min_val = 0.0f;
    float max_val = 0.0f;
    float mean = 0.0f;
};

struct PrefillTraceData {
    std::vector<int32_t> text;
    std::vector<int32_t> text_mask;
    std::vector<float> feat;
    std::vector<int32_t> feat_mask;

    std::vector<float> out_lm_hidden;
    std::vector<float> out_residual_hidden;
    std::vector<float> out_prefix_feat_cond;
    int out_current_position = -1;
    int out_streaming_prefix_len = -1;

    std::vector<std::vector<float>> out_base_keys;
    std::vector<std::vector<float>> out_base_values;
    std::vector<std::vector<float>> out_res_keys;
    std::vector<std::vector<float>> out_res_values;
};

struct DecodeTraceData {
    std::vector<float> in_lm_hidden;
    std::vector<float> in_residual_hidden;
    std::vector<float> in_prefix_feat_cond;
    int in_current_position = -1;
    int in_streaming_prefix_len = -1;

    std::vector<std::vector<float>> in_base_keys;
    std::vector<std::vector<float>> in_base_values;
    std::vector<std::vector<float>> in_res_keys;
    std::vector<std::vector<float>> in_res_values;

    std::vector<float> z_json_order;
    int inference_timesteps = 0;
    float cfg_value = 0.0f;

    std::vector<float> out_pred_feat;
    std::vector<float> out_lm_hidden;
    std::vector<float> out_residual_hidden;
    std::vector<float> out_prefix_feat_cond;
    int out_current_position = -1;
    int out_streaming_prefix_len = -1;
    bool out_stop = false;

    std::vector<std::vector<float>> out_base_keys;
    std::vector<std::vector<float>> out_base_values;
    std::vector<std::vector<float>> out_res_keys;
    std::vector<std::vector<float>> out_res_values;
};

bool file_exists(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    return file.good();
}

RuntimeMetadataExpectations load_runtime_metadata_expectations(const std::string& path, VoxCPMBackend& backend) {
    auto store = std::make_shared<VoxCPMWeightStore>();
    if (!store->load_from_file(path, backend)) {
        throw std::runtime_error("Failed to load runtime metadata from GGUF");
    }

    RuntimeMetadataExpectations out;
    uint32_t value = 0;
    if (!store->get_u32("voxcpm_patch_size", value)) {
        throw std::runtime_error("Missing voxcpm_patch_size");
    }
    out.patch_size = static_cast<int>(value);
    if (!store->get_u32("voxcpm_feat_dim", value)) {
        throw std::runtime_error("Missing voxcpm_feat_dim");
    }
    out.feat_dim = static_cast<int>(value);
    if (!store->get_u32("voxcpm_max_length", value)) {
        throw std::runtime_error("Missing voxcpm_max_length");
    }
    out.max_length = static_cast<int>(value);
    if (!store->get_u32("voxcpm_residual_lm_num_layers", value)) {
        throw std::runtime_error("Missing voxcpm_residual_lm_num_layers");
    }
    out.residual_lm_layers = static_cast<int>(value);
    return out;
}

TensorStats compute_stats(const std::vector<float>& data) {
    REQUIRE_FALSE(data.empty());

    TensorStats stats;
    stats.min_val = std::numeric_limits<float>::max();
    stats.max_val = std::numeric_limits<float>::lowest();
    for (float v : data) {
        stats.min_val = std::min(stats.min_val, v);
        stats.max_val = std::max(stats.max_val, v);
        stats.mean += v;
    }
    stats.mean /= static_cast<float>(data.size());
    return stats;
}

bool all_finite(const std::vector<float>& data) {
    return std::all_of(data.begin(), data.end(), [](float v) {
        return std::isfinite(v);
    });
}

std::vector<float> make_deterministic_noise(int feat_dim, int patch_size, int step) {
    const size_t size = static_cast<size_t>(feat_dim) * patch_size;
    std::vector<float> noise(size, 0.0f);
    for (size_t i = 0; i < size; ++i) {
        const float phase = 0.173f * static_cast<float>(i + 1) + 0.619f * static_cast<float>(step + 1);
        noise[i] = 0.75f * std::sin(phase);
    }
    return noise;
}

void validate_with_tolerance(const std::vector<float>& actual,
                             const std::vector<float>& expected,
                             const std::vector<float>& reference,
                             float tolerance,
                             float max_mismatch_rate,
                             const char* label) {
    REQUIRE(actual.size() == expected.size());

    size_t mismatch_count = 0;
    float max_abs_diff = 0.0f;
    float sum_abs_diff = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        const float diff = std::fabs(actual[i] - expected[i]);
        max_abs_diff = std::max(max_abs_diff, diff);
        sum_abs_diff += diff;
        if (diff > tolerance) {
            ++mismatch_count;
        }
    }

    const float mismatch_rate = static_cast<float>(mismatch_count) / static_cast<float>(actual.size());
    const float avg_abs_diff = sum_abs_diff / static_cast<float>(actual.size());

    const TensorStats input_stats = compute_stats(reference);
    const TensorStats expected_stats = compute_stats(expected);
    const TensorStats actual_stats = compute_stats(actual);

    std::cout << "\n=== " << label << " ===\n";
    std::cout << "elements: " << actual.size() << "\n";
    std::cout << "input range: [" << input_stats.min_val << ", " << input_stats.max_val
              << "], mean=" << input_stats.mean << "\n";
    std::cout << "expected range: [" << expected_stats.min_val << ", " << expected_stats.max_val
              << "], mean=" << expected_stats.mean << "\n";
    std::cout << "actual range: [" << actual_stats.min_val << ", " << actual_stats.max_val
              << "], mean=" << actual_stats.mean << "\n";
    std::cout << "max abs error: " << max_abs_diff << "\n";
    std::cout << "avg abs error: " << avg_abs_diff << "\n";
    std::cout << "mismatch rate (> " << tolerance << "): " << (mismatch_rate * 100.0f) << "%\n";

    INFO(label << " max abs diff = " << max_abs_diff);
    INFO(label << " avg abs diff = " << avg_abs_diff);
    INFO(label << " mismatch rate = " << mismatch_rate);
    REQUIRE(mismatch_rate <= max_mismatch_rate);
}

std::vector<float> concat_cache_layers(const std::vector<std::vector<float>>& per_layer) {
    size_t total = 0;
    for (const auto& layer : per_layer) {
        total += layer.size();
    }

    std::vector<float> out;
    out.reserve(total);
    for (const auto& layer : per_layer) {
        out.insert(out.end(), layer.begin(), layer.end());
    }
    return out;
}

std::vector<float> read_cache_prefix(VoxCPMBackend& backend,
                                     const ggml_tensor* raw_cache,
                                     int head_dim,
                                     int n_kv_heads,
                                     int max_length,
                                     int seq_len) {
    REQUIRE(raw_cache != nullptr);
    REQUIRE(seq_len <= max_length);

    std::vector<float> prefix(static_cast<size_t>(head_dim) * seq_len * n_kv_heads, 0.0f);
    for (int h = 0; h < n_kv_heads; ++h) {
        const size_t dst_offset = static_cast<size_t>(h) * seq_len * head_dim;
        const size_t src_offset = static_cast<size_t>(h) * max_length * head_dim * sizeof(float);
        const size_t bytes = static_cast<size_t>(seq_len) * head_dim * sizeof(float);
        backend.tensor_get(raw_cache, prefix.data() + dst_offset, src_offset, bytes);
    }
    return prefix;
}

void write_cache_prefix(VoxCPMBackend& backend,
                        ggml_tensor* raw_cache,
                        const std::vector<float>& prefix,
                        int head_dim,
                        int n_kv_heads,
                        int max_length,
                        int seq_len) {
    REQUIRE(raw_cache != nullptr);
    REQUIRE(seq_len <= max_length);
    REQUIRE(prefix.size() == static_cast<size_t>(head_dim) * seq_len * n_kv_heads);

    for (int h = 0; h < n_kv_heads; ++h) {
        const size_t src_offset = static_cast<size_t>(h) * seq_len * head_dim;
        const size_t dst_offset = static_cast<size_t>(h) * max_length * head_dim * sizeof(float);
        const size_t bytes = static_cast<size_t>(seq_len) * head_dim * sizeof(float);
        backend.tensor_set(raw_cache, prefix.data() + src_offset, dst_offset, bytes);
    }
}

std::vector<float> convert_single_bct_json_order_to_ct(const std::vector<float>& input,
                                                       int feat_dim,
                                                       int seq_len) {
    REQUIRE(input.size() == static_cast<size_t>(feat_dim) * seq_len);
    std::vector<float> out(input.size());
    for (int c = 0; c < feat_dim; ++c) {
        for (int t = 0; t < seq_len; ++t) {
            out[static_cast<size_t>(t) * feat_dim + c] = input[static_cast<size_t>(c) * seq_len + t];
        }
    }
    return out;
}

class TraceSaxBase : public json::json_sax_t {
public:
    struct ContainerState {
        bool pushed_path = false;
        bool is_capture_array = false;
    };

    bool null() override {
        pending_key_.clear();
        return true;
    }

    bool binary(json::binary_t&) override {
        pending_key_.clear();
        return true;
    }

    bool string(string_t&) override {
        pending_key_.clear();
        return true;
    }

    bool start_object(std::size_t) override {
        const bool pushed = push_pending_key();
        containers_.push_back({pushed, false});
        return true;
    }

    bool end_object() override {
        if (containers_.empty()) {
            return false;
        }
        const ContainerState state = containers_.back();
        containers_.pop_back();
        if (state.pushed_path && !path_.empty()) {
            path_.pop_back();
        }
        pending_key_.clear();
        return true;
    }

    bool start_array(std::size_t) override {
        const bool pushed = push_pending_key();
        const bool is_capture = begin_capture_for_current_path();
        containers_.push_back({pushed, is_capture});
        return true;
    }

    bool end_array() override {
        if (containers_.empty()) {
            return false;
        }
        const ContainerState state = containers_.back();
        containers_.pop_back();
        if (state.is_capture_array) {
            end_capture_array();
        }
        if (state.pushed_path && !path_.empty()) {
            path_.pop_back();
        }
        pending_key_.clear();
        return true;
    }

    bool key(string_t& val) override {
        pending_key_ = val;
        return true;
    }

    bool parse_error(std::size_t, const std::string&, const json::exception&) override {
        std::cerr << "JSON SAX parse error near path:";
        for (const std::string& token : path_) {
            std::cerr << "/" << token;
        }
        std::cerr << "\n";
        return false;
    }

protected:
    bool push_pending_key() {
        if (pending_key_.empty()) {
            return false;
        }
        path_.push_back(pending_key_);
        pending_key_.clear();
        return true;
    }

    bool path_equals(std::initializer_list<const char*> expected) const {
        if (path_.size() != expected.size()) {
            return false;
        }
        size_t i = 0;
        for (const char* token : expected) {
            if (path_[i] != token) {
                return false;
            }
            ++i;
        }
        return true;
    }

    virtual bool begin_capture_for_current_path() = 0;
    virtual void end_capture_array() = 0;

    std::vector<std::string> path_;
    std::vector<ContainerState> containers_;
    std::string pending_key_;
};

class PrefillTraceHandler : public TraceSaxBase {
public:
    enum class Capture {
        None,
        Text,
        TextMask,
        Feat,
        FeatMask,
        OutLmHidden,
        OutResidualHidden,
        OutPrefixFeatCond,
        OutBaseKey,
        OutBaseValue,
        OutResKey,
        OutResValue,
    };

    PrefillTraceHandler(PrefillTraceData& trace,
                        int base_layers,
                        int residual_layers,
                        int n_kv_heads,
                        int max_length,
                        int head_dim)
        : trace_(trace),
          n_kv_heads_(n_kv_heads),
          max_length_(max_length),
          head_dim_(head_dim) {
        trace_.out_base_keys.assign(base_layers, {});
        trace_.out_base_values.assign(base_layers, {});
        trace_.out_res_keys.assign(residual_layers, {});
        trace_.out_res_values.assign(residual_layers, {});
        const size_t prefix_elems = static_cast<size_t>(n_kv_heads_) * kPrefillSeqLen * head_dim_;
        for (auto& v : trace_.out_base_keys) v.reserve(prefix_elems);
        for (auto& v : trace_.out_base_values) v.reserve(prefix_elems);
        for (auto& v : trace_.out_res_keys) v.reserve(prefix_elems);
        for (auto& v : trace_.out_res_values) v.reserve(prefix_elems);
        trace_.text.reserve(kPrefillSeqLen);
        trace_.text_mask.reserve(kPrefillSeqLen);
        trace_.feat_mask.reserve(kPrefillSeqLen);
        trace_.feat.reserve(static_cast<size_t>(kPrefillSeqLen) * 4 * 64);
        trace_.out_lm_hidden.reserve(1024);
        trace_.out_residual_hidden.reserve(1024);
        trace_.out_prefix_feat_cond.reserve(4 * 64);
    }

    bool boolean(bool) override {
        pending_key_.clear();
        return true;
    }

    bool number_integer(number_integer_t val) override {
        const bool ok = handle_integer(static_cast<int64_t>(val));
        pending_key_.clear();
        return ok;
    }

    bool number_unsigned(number_unsigned_t val) override {
        const bool ok = handle_integer(static_cast<int64_t>(val));
        pending_key_.clear();
        return ok;
    }

    bool number_float(number_float_t val, const string_t&) override {
        const bool ok = handle_float(static_cast<float>(val));
        pending_key_.clear();
        return ok;
    }

protected:
    bool begin_capture_for_current_path() override {
        if (active_capture_ != Capture::None) {
            return false;
        }
        active_capture_ = capture_for_current_path();
        if (active_capture_ == Capture::OutBaseKey || active_capture_ == Capture::OutBaseValue ||
            active_capture_ == Capture::OutResKey || active_capture_ == Capture::OutResValue) {
            cache_flat_index_ = 0;
        }
        return active_capture_ != Capture::None;
    }

    void end_capture_array() override {
        active_capture_ = Capture::None;
        cache_flat_index_ = 0;
    }

private:
    Capture capture_for_current_path() const {
        if (path_equals({"inputs", "text"})) return Capture::Text;
        if (path_equals({"inputs", "text_mask"})) return Capture::TextMask;
        if (path_equals({"inputs", "feat"})) return Capture::Feat;
        if (path_equals({"inputs", "feat_mask"})) return Capture::FeatMask;
        if (path_equals({"outputs", "output", "lm_hidden"})) return Capture::OutLmHidden;
        if (path_equals({"outputs", "output", "residual_hidden"})) return Capture::OutResidualHidden;
        if (path_equals({"outputs", "output", "prefix_feat_cond"})) return Capture::OutPrefixFeatCond;
        if (path_equals({"outputs", "output", "base_lm_cache", "layers", "key"})) return Capture::OutBaseKey;
        if (path_equals({"outputs", "output", "base_lm_cache", "layers", "value"})) return Capture::OutBaseValue;
        if (path_equals({"outputs", "output", "residual_lm_cache", "layers", "key"})) return Capture::OutResKey;
        if (path_equals({"outputs", "output", "residual_lm_cache", "layers", "value"})) return Capture::OutResValue;
        return Capture::None;
    }

    bool handle_integer(int64_t val) {
        if (pending_key_ == "layer") {
            if (path_equals({"outputs", "output", "base_lm_cache", "layers"})) {
                current_base_layer_ = static_cast<int>(val);
                return true;
            }
            if (path_equals({"outputs", "output", "residual_lm_cache", "layers"})) {
                current_res_layer_ = static_cast<int>(val);
                return true;
            }
        }
        if (pending_key_ == "current_position" && path_equals({"outputs", "output"})) {
            trace_.out_current_position = static_cast<int>(val);
            return true;
        }
        if (pending_key_ == "streaming_prefix_len" && path_equals({"outputs", "output"})) {
            trace_.out_streaming_prefix_len = static_cast<int>(val);
            return true;
        }
        return handle_float(static_cast<float>(val));
    }

    void append_cache_value(std::vector<std::vector<float>>& layers, int layer, float val) {
        if (layer < 0 || layer >= static_cast<int>(layers.size())) {
            return;
        }
        const size_t head_block = static_cast<size_t>(max_length_) * head_dim_;
        const size_t keep_per_head = static_cast<size_t>(kPrefillSeqLen) * head_dim_;
        const size_t head = cache_flat_index_ / head_block;
        const size_t offset_in_head = cache_flat_index_ % head_block;
        if (head < static_cast<size_t>(n_kv_heads_) && offset_in_head < keep_per_head) {
            layers[static_cast<size_t>(layer)].push_back(val);
        }
        ++cache_flat_index_;
    }

    bool handle_float(float val) {
        switch (active_capture_) {
        case Capture::Text:
            trace_.text.push_back(static_cast<int32_t>(val));
            return true;
        case Capture::TextMask:
            trace_.text_mask.push_back(static_cast<int32_t>(val));
            return true;
        case Capture::Feat:
            trace_.feat.push_back(val);
            return true;
        case Capture::FeatMask:
            trace_.feat_mask.push_back(static_cast<int32_t>(val));
            return true;
        case Capture::OutLmHidden:
            trace_.out_lm_hidden.push_back(val);
            return true;
        case Capture::OutResidualHidden:
            trace_.out_residual_hidden.push_back(val);
            return true;
        case Capture::OutPrefixFeatCond:
            trace_.out_prefix_feat_cond.push_back(val);
            return true;
        case Capture::OutBaseKey:
            append_cache_value(trace_.out_base_keys, current_base_layer_, val);
            return true;
        case Capture::OutBaseValue:
            append_cache_value(trace_.out_base_values, current_base_layer_, val);
            return true;
        case Capture::OutResKey:
            append_cache_value(trace_.out_res_keys, current_res_layer_, val);
            return true;
        case Capture::OutResValue:
            append_cache_value(trace_.out_res_values, current_res_layer_, val);
            return true;
        case Capture::None:
            return true;
        }
        return true;
    }

    PrefillTraceData& trace_;
    int n_kv_heads_;
    int max_length_;
    int head_dim_;
    Capture active_capture_ = Capture::None;
    int current_base_layer_ = -1;
    int current_res_layer_ = -1;
    size_t cache_flat_index_ = 0;
};

class DecodeTraceHandler : public TraceSaxBase {
public:
    enum class Capture {
        None,
        InLmHidden,
        InResidualHidden,
        InPrefixFeatCond,
        InBaseKey,
        InBaseValue,
        InResKey,
        InResValue,
        Z,
        OutPredFeat,
        OutLmHidden,
        OutResidualHidden,
        OutPrefixFeatCond,
        OutBaseKey,
        OutBaseValue,
        OutResKey,
        OutResValue,
    };

    DecodeTraceHandler(DecodeTraceData& trace,
                       int base_layers,
                       int residual_layers,
                       int n_kv_heads,
                       int max_length,
                       int head_dim)
        : trace_(trace),
          n_kv_heads_(n_kv_heads),
          max_length_(max_length),
          head_dim_(head_dim) {
        trace_.in_base_keys.assign(base_layers, {});
        trace_.in_base_values.assign(base_layers, {});
        trace_.in_res_keys.assign(residual_layers, {});
        trace_.in_res_values.assign(residual_layers, {});
        trace_.out_base_keys.assign(base_layers, {});
        trace_.out_base_values.assign(base_layers, {});
        trace_.out_res_keys.assign(residual_layers, {});
        trace_.out_res_values.assign(residual_layers, {});

        const size_t prefix_elems = static_cast<size_t>(n_kv_heads_) * kDecodeCacheSeqLen * head_dim_;
        for (auto& v : trace_.in_base_keys) v.reserve(prefix_elems);
        for (auto& v : trace_.in_base_values) v.reserve(prefix_elems);
        for (auto& v : trace_.in_res_keys) v.reserve(prefix_elems);
        for (auto& v : trace_.in_res_values) v.reserve(prefix_elems);
        for (auto& v : trace_.out_base_keys) v.reserve(prefix_elems);
        for (auto& v : trace_.out_base_values) v.reserve(prefix_elems);
        for (auto& v : trace_.out_res_keys) v.reserve(prefix_elems);
        for (auto& v : trace_.out_res_values) v.reserve(prefix_elems);
        trace_.in_lm_hidden.reserve(1024);
        trace_.in_residual_hidden.reserve(1024);
        trace_.in_prefix_feat_cond.reserve(4 * 64);
        trace_.z_json_order.reserve(64 * 4);
        trace_.out_pred_feat.reserve(4 * 64);
        trace_.out_lm_hidden.reserve(1024);
        trace_.out_residual_hidden.reserve(1024);
        trace_.out_prefix_feat_cond.reserve(4 * 64);
    }

    bool boolean(bool val) override {
        if (pending_key_ == "output_2" && path_equals({"outputs"})) {
            trace_.out_stop = val;
        }
        pending_key_.clear();
        return true;
    }

    bool number_integer(number_integer_t val) override {
        const bool ok = handle_integer(static_cast<int64_t>(val));
        pending_key_.clear();
        return ok;
    }

    bool number_unsigned(number_unsigned_t val) override {
        const bool ok = handle_integer(static_cast<int64_t>(val));
        pending_key_.clear();
        return ok;
    }

    bool number_float(number_float_t val, const string_t&) override {
        const bool ok = handle_float(static_cast<float>(val));
        pending_key_.clear();
        return ok;
    }

protected:
    bool begin_capture_for_current_path() override {
        if (active_capture_ != Capture::None) {
            return false;
        }
        active_capture_ = capture_for_current_path();
        if (active_capture_ == Capture::InBaseKey || active_capture_ == Capture::InBaseValue ||
            active_capture_ == Capture::InResKey || active_capture_ == Capture::InResValue ||
            active_capture_ == Capture::OutBaseKey || active_capture_ == Capture::OutBaseValue ||
            active_capture_ == Capture::OutResKey || active_capture_ == Capture::OutResValue) {
            cache_flat_index_ = 0;
        }
        return active_capture_ != Capture::None;
    }

    void end_capture_array() override {
        active_capture_ = Capture::None;
        cache_flat_index_ = 0;
    }

private:
    Capture capture_for_current_path() const {
        if (path_equals({"inputs", "state", "lm_hidden"})) return Capture::InLmHidden;
        if (path_equals({"inputs", "state", "residual_hidden"})) return Capture::InResidualHidden;
        if (path_equals({"inputs", "state", "prefix_feat_cond"})) return Capture::InPrefixFeatCond;
        if (path_equals({"inputs", "state", "base_lm_cache", "layers", "key"})) return Capture::InBaseKey;
        if (path_equals({"inputs", "state", "base_lm_cache", "layers", "value"})) return Capture::InBaseValue;
        if (path_equals({"inputs", "state", "residual_lm_cache", "layers", "key"})) return Capture::InResKey;
        if (path_equals({"inputs", "state", "residual_lm_cache", "layers", "value"})) return Capture::InResValue;
        if (path_equals({"inputs", "z"})) return Capture::Z;
        if (path_equals({"outputs", "output_0"})) return Capture::OutPredFeat;
        if (path_equals({"outputs", "output_1", "lm_hidden"})) return Capture::OutLmHidden;
        if (path_equals({"outputs", "output_1", "residual_hidden"})) return Capture::OutResidualHidden;
        if (path_equals({"outputs", "output_1", "prefix_feat_cond"})) return Capture::OutPrefixFeatCond;
        if (path_equals({"outputs", "output_1", "base_lm_cache", "layers", "key"})) return Capture::OutBaseKey;
        if (path_equals({"outputs", "output_1", "base_lm_cache", "layers", "value"})) return Capture::OutBaseValue;
        if (path_equals({"outputs", "output_1", "residual_lm_cache", "layers", "key"})) return Capture::OutResKey;
        if (path_equals({"outputs", "output_1", "residual_lm_cache", "layers", "value"})) return Capture::OutResValue;
        return Capture::None;
    }

    bool handle_integer(int64_t val) {
        if (pending_key_ == "layer") {
            if (path_equals({"inputs", "state", "base_lm_cache", "layers"})) {
                current_in_base_layer_ = static_cast<int>(val);
                return true;
            }
            if (path_equals({"inputs", "state", "residual_lm_cache", "layers"})) {
                current_in_res_layer_ = static_cast<int>(val);
                return true;
            }
            if (path_equals({"outputs", "output_1", "base_lm_cache", "layers"})) {
                current_out_base_layer_ = static_cast<int>(val);
                return true;
            }
            if (path_equals({"outputs", "output_1", "residual_lm_cache", "layers"})) {
                current_out_res_layer_ = static_cast<int>(val);
                return true;
            }
        }
        if (pending_key_ == "current_position" && path_equals({"inputs", "state"})) {
            trace_.in_current_position = static_cast<int>(val);
            return true;
        }
        if (pending_key_ == "streaming_prefix_len" && path_equals({"inputs", "state"})) {
            trace_.in_streaming_prefix_len = static_cast<int>(val);
            return true;
        }
        if (pending_key_ == "inference_timesteps" && path_equals({"inputs"})) {
            trace_.inference_timesteps = static_cast<int>(val);
            return true;
        }
        if (pending_key_ == "current_position" && path_equals({"outputs", "output_1"})) {
            trace_.out_current_position = static_cast<int>(val);
            return true;
        }
        if (pending_key_ == "streaming_prefix_len" && path_equals({"outputs", "output_1"})) {
            trace_.out_streaming_prefix_len = static_cast<int>(val);
            return true;
        }
        return handle_float(static_cast<float>(val));
    }

    void append_cache_value(std::vector<std::vector<float>>& layers, int layer, float val) {
        if (layer < 0 || layer >= static_cast<int>(layers.size())) {
            return;
        }
        const size_t head_block = static_cast<size_t>(max_length_) * head_dim_;
        const size_t keep_per_head = static_cast<size_t>(kDecodeCacheSeqLen) * head_dim_;
        const size_t head = cache_flat_index_ / head_block;
        const size_t offset_in_head = cache_flat_index_ % head_block;
        if (head < static_cast<size_t>(n_kv_heads_) && offset_in_head < keep_per_head) {
            layers[static_cast<size_t>(layer)].push_back(val);
        }
        ++cache_flat_index_;
    }

    bool handle_float(float val) {
        switch (active_capture_) {
        case Capture::InLmHidden:
            trace_.in_lm_hidden.push_back(val);
            return true;
        case Capture::InResidualHidden:
            trace_.in_residual_hidden.push_back(val);
            return true;
        case Capture::InPrefixFeatCond:
            trace_.in_prefix_feat_cond.push_back(val);
            return true;
        case Capture::InBaseKey:
            append_cache_value(trace_.in_base_keys, current_in_base_layer_, val);
            return true;
        case Capture::InBaseValue:
            append_cache_value(trace_.in_base_values, current_in_base_layer_, val);
            return true;
        case Capture::InResKey:
            append_cache_value(trace_.in_res_keys, current_in_res_layer_, val);
            return true;
        case Capture::InResValue:
            append_cache_value(trace_.in_res_values, current_in_res_layer_, val);
            return true;
        case Capture::Z:
            trace_.z_json_order.push_back(val);
            return true;
        case Capture::OutPredFeat:
            trace_.out_pred_feat.push_back(val);
            return true;
        case Capture::OutLmHidden:
            trace_.out_lm_hidden.push_back(val);
            return true;
        case Capture::OutResidualHidden:
            trace_.out_residual_hidden.push_back(val);
            return true;
        case Capture::OutPrefixFeatCond:
            trace_.out_prefix_feat_cond.push_back(val);
            return true;
        case Capture::OutBaseKey:
            append_cache_value(trace_.out_base_keys, current_out_base_layer_, val);
            return true;
        case Capture::OutBaseValue:
            append_cache_value(trace_.out_base_values, current_out_base_layer_, val);
            return true;
        case Capture::OutResKey:
            append_cache_value(trace_.out_res_keys, current_out_res_layer_, val);
            return true;
        case Capture::OutResValue:
            append_cache_value(trace_.out_res_values, current_out_res_layer_, val);
            return true;
        case Capture::None:
            if (pending_key_ == "cfg_value" && path_equals({"inputs"})) {
                trace_.cfg_value = val;
            }
            return true;
        }
        return true;
    }

    DecodeTraceData& trace_;
    int n_kv_heads_;
    int max_length_;
    int head_dim_;
    Capture active_capture_ = Capture::None;
    int current_in_base_layer_ = -1;
    int current_in_res_layer_ = -1;
    int current_out_base_layer_ = -1;
    int current_out_res_layer_ = -1;
    size_t cache_flat_index_ = 0;
};

PrefillTraceData load_prefill_trace_streaming(const std::string& trace_path,
                                              const VoxCPMRuntime& runtime) {
    std::ifstream file(trace_path);
    REQUIRE(file.is_open());

    PrefillTraceData trace;
    PrefillTraceHandler handler(trace,
                                runtime.base_lm().config().n_layer,
                                runtime.residual_lm().config().n_layer,
                                runtime.base_lm().config().n_kv_heads,
                                runtime.config().max_length,
                                runtime.base_lm().config().head_dim());
    REQUIRE(json::sax_parse(file, &handler));
    return trace;
}

DecodeTraceData load_decode_trace_streaming(const std::string& trace_path,
                                            const VoxCPMRuntime& runtime) {
    std::ifstream file(trace_path);
    REQUIRE(file.is_open());

    DecodeTraceData trace;
    DecodeTraceHandler handler(trace,
                               runtime.base_lm().config().n_layer,
                               runtime.residual_lm().config().n_layer,
                               runtime.base_lm().config().n_kv_heads,
                               runtime.config().max_length,
                               runtime.base_lm().config().head_dim());
    REQUIRE(json::sax_parse(file, &handler));
    return trace;
}

}  // namespace

TEST_CASE("VoxCPM runtime loads from GGUF", "[voxcpm][integration]") {
    if (!file_exists(kModelPath)) {
        WARN("Model file not found, skipping test: " << kModelPath);
        return;
    }

    VoxCPMBackend backend(BackendType::CPU, 2);
    VoxCPMContext weight_ctx(ContextType::Weights, 1024);
    VoxCPMContext graph_ctx(ContextType::Graph, 4096, 32768);

    VoxCPMRuntime runtime;
    REQUIRE(runtime.load_from_gguf(kModelPath, weight_ctx, graph_ctx, backend));
    const RuntimeMetadataExpectations expected = load_runtime_metadata_expectations(kModelPath, backend);
    REQUIRE(runtime.config().patch_size == expected.patch_size);
    REQUIRE(runtime.config().feat_dim == expected.feat_dim);
    REQUIRE(runtime.config().max_length == expected.max_length);
    REQUIRE(runtime.base_lm().config().n_layer == 24);
    REQUIRE(runtime.residual_lm().config().n_layer == expected.residual_lm_layers);
}

TEST_CASE("VoxCPM runtime shares one GGUF weight store", "[voxcpm][weights][shared]") {
    if (!file_exists(kModelPath)) {
        WARN("Model file not found, skipping test: " << kModelPath);
        return;
    }

    VoxCPMBackend backend(BackendType::CPU, 2);
    VoxCPMContext weight_ctx(ContextType::Weights, 1024);
    VoxCPMContext graph_ctx(ContextType::Graph, 4096, 32768);

    VoxCPMRuntime runtime;
    REQUIRE(runtime.load_from_gguf(kModelPath, weight_ctx, graph_ctx, backend));
    REQUIRE(runtime.uses_shared_weights());
    REQUIRE(runtime.shared_store_token() != nullptr);

    const void* store = runtime.shared_store_token();
    REQUIRE(runtime.base_lm().shared_store_token() == store);
    REQUIRE(runtime.residual_lm().shared_store_token() == store);
    REQUIRE(runtime.feat_encoder().shared_store_token() == store);
    REQUIRE(runtime.feat_encoder().encoder_model().shared_store_token() == store);
    REQUIRE(runtime.feat_decoder_estimator().shared_store_token() == store);
    REQUIRE(runtime.feat_decoder_estimator().decoder_model().shared_store_token() == store);
    REQUIRE(runtime.fsq_layer().shared_store_token() == store);
    REQUIRE(runtime.components() != nullptr);
    REQUIRE(runtime.components()->shared_store_token() == store);
    REQUIRE(runtime.components()->enc_to_lm_proj()->shared_store_token() == store);
    REQUIRE(runtime.components()->lm_to_dit_proj()->shared_store_token() == store);
    REQUIRE(runtime.components()->res_to_dit_proj()->shared_store_token() == store);
    REQUIRE(runtime.components()->stop_token()->shared_store_token() == store);
    REQUIRE(runtime.components()->embed_tokens()->shared_store_token() == store);
}

TEST_CASE("VoxCPM Prefill matches trace", "[voxcpm][prefill][trace]") {
    if (!file_exists(kModelPath)) {
        WARN("Model file not found, skipping test: " << kModelPath);
        return;
    }
    if (!file_exists(kPrefillTracePath)) {
        WARN("Trace file not found, skipping test: " << kPrefillTracePath);
        return;
    }

    VoxCPMBackend backend(BackendType::CPU, 2);
    VoxCPMContext weight_ctx(ContextType::Weights, 1024);
    VoxCPMContext graph_ctx(ContextType::Graph, 4096, 32768);

    VoxCPMRuntime runtime;
    REQUIRE(runtime.load_from_gguf(kModelPath, weight_ctx, graph_ctx, backend));

    const PrefillTraceData trace = load_prefill_trace_streaming(kPrefillTracePath, runtime);
    REQUIRE(trace.text.size() == kPrefillSeqLen);
    REQUIRE(trace.text_mask.size() == kPrefillSeqLen);
    REQUIRE(trace.feat_mask.size() == kPrefillSeqLen);
    REQUIRE(trace.feat.size() == static_cast<size_t>(kPrefillSeqLen) * runtime.config().patch_size * runtime.config().feat_dim);

    VoxCPMDecodeState state = runtime.prefill(trace.text,
                                             trace.text_mask,
                                             trace.feat,
                                             trace.feat_mask,
                                             kPrefillSeqLen,
                                             trace.out_streaming_prefix_len);

    REQUIRE(state.current_position == trace.out_current_position);
    REQUIRE(state.streaming_prefix_len == trace.out_streaming_prefix_len);

    validate_with_tolerance(state.lm_hidden,
                            trace.out_lm_hidden,
                            trace.feat,
                            kHiddenTolerance,
                            kMaxMismatchRate,
                            "VoxCPM Prefill lm_hidden");
    validate_with_tolerance(state.residual_hidden,
                            trace.out_residual_hidden,
                            trace.feat,
                            kHiddenTolerance,
                            kMaxMismatchRate,
                            "VoxCPM Prefill residual_hidden");
    validate_with_tolerance(state.prefix_feat_cond,
                            trace.out_prefix_feat_cond,
                            trace.out_prefix_feat_cond,
                            kHiddenTolerance,
                            kMaxMismatchRate,
                            "VoxCPM Prefill prefix_feat_cond");

    std::vector<std::vector<float>> actual_base_keys(runtime.base_lm().config().n_layer);
    std::vector<std::vector<float>> actual_base_values(runtime.base_lm().config().n_layer);
    std::vector<std::vector<float>> actual_res_keys(runtime.residual_lm().config().n_layer);
    std::vector<std::vector<float>> actual_res_values(runtime.residual_lm().config().n_layer);

    for (int layer = 0; layer < runtime.base_lm().config().n_layer; ++layer) {
        actual_base_keys[layer] = read_cache_prefix(backend,
                                                    state.base_lm_cache->raw_k_cache(layer),
                                                    runtime.base_lm().config().head_dim(),
                                                    runtime.base_lm().config().n_kv_heads,
                                                    runtime.config().max_length,
                                                    kPrefillSeqLen);
        actual_base_values[layer] = read_cache_prefix(backend,
                                                      state.base_lm_cache->raw_v_cache(layer),
                                                      runtime.base_lm().config().head_dim(),
                                                      runtime.base_lm().config().n_kv_heads,
                                                      runtime.config().max_length,
                                                      kPrefillSeqLen);
    }
    for (int layer = 0; layer < runtime.residual_lm().config().n_layer; ++layer) {
        actual_res_keys[layer] = read_cache_prefix(backend,
                                                   state.residual_lm_cache->raw_k_cache(layer),
                                                   runtime.residual_lm().config().head_dim(),
                                                   runtime.residual_lm().config().n_kv_heads,
                                                   runtime.config().max_length,
                                                   kPrefillSeqLen);
        actual_res_values[layer] = read_cache_prefix(backend,
                                                     state.residual_lm_cache->raw_v_cache(layer),
                                                     runtime.residual_lm().config().head_dim(),
                                                     runtime.residual_lm().config().n_kv_heads,
                                                     runtime.config().max_length,
                                                     kPrefillSeqLen);
    }

    validate_with_tolerance(concat_cache_layers(actual_base_keys),
                            concat_cache_layers(trace.out_base_keys),
                            trace.feat,
                            kCacheTolerance,
                            kMaxMismatchRate,
                            "VoxCPM Prefill base_lm_cache.keys");
    validate_with_tolerance(concat_cache_layers(actual_base_values),
                            concat_cache_layers(trace.out_base_values),
                            trace.feat,
                            kCacheTolerance,
                            kMaxMismatchRate,
                            "VoxCPM Prefill base_lm_cache.values");
    validate_with_tolerance(concat_cache_layers(actual_res_keys),
                            concat_cache_layers(trace.out_res_keys),
                            trace.feat,
                            kCacheTolerance,
                            kMaxMismatchRate,
                            "VoxCPM Prefill residual_lm_cache.keys");
    validate_with_tolerance(concat_cache_layers(actual_res_values),
                            concat_cache_layers(trace.out_res_values),
                            trace.feat,
                            kCacheTolerance,
                            kMaxMismatchRate,
                            "VoxCPM Prefill residual_lm_cache.values");
}

TEST_CASE("VoxCPM Decode matches trace", "[voxcpm][decode][trace]") {
    if (!file_exists(kModelPath)) {
        WARN("Model file not found, skipping test: " << kModelPath);
        return;
    }
    if (!file_exists(kDecodeTracePath)) {
        WARN("Trace file not found, skipping test: " << kDecodeTracePath);
        return;
    }

    VoxCPMBackend backend(BackendType::CPU, 2);
    VoxCPMContext weight_ctx(ContextType::Weights, 1024);
    VoxCPMContext graph_ctx(ContextType::Graph, 4096, 32768);

    VoxCPMRuntime runtime;
    REQUIRE(runtime.load_from_gguf(kModelPath, weight_ctx, graph_ctx, backend));

    const DecodeTraceData trace = load_decode_trace_streaming(kDecodeTracePath, runtime);
    const std::vector<float> z = convert_single_bct_json_order_to_ct(trace.z_json_order,
                                                                     runtime.config().feat_dim,
                                                                     runtime.config().patch_size);

    VoxCPMDecodeState state = runtime.create_decode_state();
    state.lm_hidden = trace.in_lm_hidden;
    state.residual_hidden = trace.in_residual_hidden;
    state.current_position = trace.in_current_position;
    state.prefix_feat_cond = trace.in_prefix_feat_cond;
    state.streaming_prefix_len = trace.in_streaming_prefix_len;

    for (int layer = 0; layer < runtime.base_lm().config().n_layer; ++layer) {
        write_cache_prefix(backend,
                           state.base_lm_cache->raw_k_cache(layer),
                           trace.in_base_keys[layer],
                           runtime.base_lm().config().head_dim(),
                           runtime.base_lm().config().n_kv_heads,
                           runtime.config().max_length,
                           kDecodeCacheSeqLen);
        write_cache_prefix(backend,
                           state.base_lm_cache->raw_v_cache(layer),
                           trace.in_base_values[layer],
                           runtime.base_lm().config().head_dim(),
                           runtime.base_lm().config().n_kv_heads,
                           runtime.config().max_length,
                           kDecodeCacheSeqLen);
    }
    for (int layer = 0; layer < runtime.residual_lm().config().n_layer; ++layer) {
        write_cache_prefix(backend,
                           state.residual_lm_cache->raw_k_cache(layer),
                           trace.in_res_keys[layer],
                           runtime.residual_lm().config().head_dim(),
                           runtime.residual_lm().config().n_kv_heads,
                           runtime.config().max_length,
                           kDecodeCacheSeqLen);
        write_cache_prefix(backend,
                           state.residual_lm_cache->raw_v_cache(layer),
                           trace.in_res_values[layer],
                           runtime.residual_lm().config().head_dim(),
                           runtime.residual_lm().config().n_kv_heads,
                           runtime.config().max_length,
                           kDecodeCacheSeqLen);
    }

    VoxCPMDecodeResult result = runtime.decode(std::move(state), z, trace.inference_timesteps, trace.cfg_value);

    REQUIRE(result.output_1.current_position == trace.out_current_position);
    REQUIRE(result.output_1.streaming_prefix_len == trace.out_streaming_prefix_len);
    REQUIRE(result.output_2 == trace.out_stop);

    validate_with_tolerance(result.output_0,
                            trace.out_pred_feat,
                            z,
                            kPredTolerance,
                            kMaxMismatchRate,
                            "VoxCPM Decode output_0");
    validate_with_tolerance(result.output_1.lm_hidden,
                            trace.out_lm_hidden,
                            trace.in_lm_hidden,
                            kHiddenTolerance,
                            kMaxMismatchRate,
                            "VoxCPM Decode lm_hidden");
    validate_with_tolerance(result.output_1.residual_hidden,
                            trace.out_residual_hidden,
                            trace.in_residual_hidden,
                            kHiddenTolerance,
                            kMaxMismatchRate,
                            "VoxCPM Decode residual_hidden");
    validate_with_tolerance(result.output_1.prefix_feat_cond,
                            trace.out_prefix_feat_cond,
                            trace.in_prefix_feat_cond,
                            kHiddenTolerance,
                            kMaxMismatchRate,
                            "VoxCPM Decode prefix_feat_cond");

    std::vector<std::vector<float>> actual_base_keys(runtime.base_lm().config().n_layer);
    std::vector<std::vector<float>> actual_base_values(runtime.base_lm().config().n_layer);
    std::vector<std::vector<float>> actual_res_keys(runtime.residual_lm().config().n_layer);
    std::vector<std::vector<float>> actual_res_values(runtime.residual_lm().config().n_layer);

    for (int layer = 0; layer < runtime.base_lm().config().n_layer; ++layer) {
        actual_base_keys[layer] = read_cache_prefix(backend,
                                                    result.output_1.base_lm_cache->raw_k_cache(layer),
                                                    runtime.base_lm().config().head_dim(),
                                                    runtime.base_lm().config().n_kv_heads,
                                                    runtime.config().max_length,
                                                    kDecodeCacheSeqLen);
        actual_base_values[layer] = read_cache_prefix(backend,
                                                      result.output_1.base_lm_cache->raw_v_cache(layer),
                                                      runtime.base_lm().config().head_dim(),
                                                      runtime.base_lm().config().n_kv_heads,
                                                      runtime.config().max_length,
                                                      kDecodeCacheSeqLen);
    }
    for (int layer = 0; layer < runtime.residual_lm().config().n_layer; ++layer) {
        actual_res_keys[layer] = read_cache_prefix(backend,
                                                   result.output_1.residual_lm_cache->raw_k_cache(layer),
                                                   runtime.residual_lm().config().head_dim(),
                                                   runtime.residual_lm().config().n_kv_heads,
                                                   runtime.config().max_length,
                                                   kDecodeCacheSeqLen);
        actual_res_values[layer] = read_cache_prefix(backend,
                                                     result.output_1.residual_lm_cache->raw_v_cache(layer),
                                                     runtime.residual_lm().config().head_dim(),
                                                     runtime.residual_lm().config().n_kv_heads,
                                                     runtime.config().max_length,
                                                     kDecodeCacheSeqLen);
    }

    const std::vector<float> input_base_keys = concat_cache_layers(trace.in_base_keys);
    const std::vector<float> input_res_keys = concat_cache_layers(trace.in_res_keys);

    validate_with_tolerance(concat_cache_layers(actual_base_keys),
                            concat_cache_layers(trace.out_base_keys),
                            input_base_keys,
                            kCacheTolerance,
                            kMaxMismatchRate,
                            "VoxCPM Decode base_lm_cache.keys");
    validate_with_tolerance(concat_cache_layers(actual_base_values),
                            concat_cache_layers(trace.out_base_values),
                            concat_cache_layers(trace.in_base_values),
                            kCacheTolerance,
                            kMaxMismatchRate,
                            "VoxCPM Decode base_lm_cache.values");
    validate_with_tolerance(concat_cache_layers(actual_res_keys),
                            concat_cache_layers(trace.out_res_keys),
                            input_res_keys,
                            kCacheTolerance,
                            kMaxMismatchRate,
                            "VoxCPM Decode residual_lm_cache.keys");
    validate_with_tolerance(concat_cache_layers(actual_res_values),
                            concat_cache_layers(trace.out_res_values),
                            concat_cache_layers(trace.in_res_values),
                            kCacheTolerance,
                            kMaxMismatchRate,
                            "VoxCPM Decode residual_lm_cache.values");
}

TEST_CASE("VoxCPM multi-step decode remains stable", "[voxcpm][decode][stability]") {
    if (!file_exists(kModelPath)) {
        WARN("Model file not found, skipping test: " << kModelPath);
        return;
    }
    if (!file_exists(kPrefillTracePath)) {
        WARN("Trace file not found, skipping test: " << kPrefillTracePath);
        return;
    }

    VoxCPMBackend backend(BackendType::CPU, 2);
    VoxCPMContext weight_ctx(ContextType::Weights, 1024);
    VoxCPMContext graph_ctx(ContextType::Graph, 4096, 32768);

    VoxCPMRuntime runtime;
    REQUIRE(runtime.load_from_gguf(kModelPath, weight_ctx, graph_ctx, backend));

    const PrefillTraceData trace = load_prefill_trace_streaming(kPrefillTracePath, runtime);
    VoxCPMDecodeState state = runtime.prefill(trace.text,
                                             trace.text_mask,
                                             trace.feat,
                                             trace.feat_mask,
                                             kPrefillSeqLen,
                                             trace.out_streaming_prefix_len);

    const int decode_steps = 6;
    const size_t patch_elements = static_cast<size_t>(runtime.config().patch_size) * runtime.config().feat_dim;

    for (int step = 0; step < decode_steps; ++step) {
        const std::vector<float> z = make_deterministic_noise(runtime.config().feat_dim,
                                                              runtime.config().patch_size,
                                                              step);
        const int expected_position = state.current_position + 1;
        VoxCPMDecodeResult result = runtime.decode(std::move(state), z, 10, 2.0f);

        REQUIRE(result.output_0.size() == patch_elements);
        REQUIRE(all_finite(result.output_0));
        REQUIRE(all_finite(result.output_1.lm_hidden));
        REQUIRE(all_finite(result.output_1.residual_hidden));
        REQUIRE(all_finite(result.output_1.prefix_feat_cond));
        REQUIRE(result.output_1.current_position == expected_position);
        REQUIRE(result.output_1.streaming_prefix_len == trace.out_streaming_prefix_len);

        state = std::move(result.output_1);
    }
}

}  // namespace test
}  // namespace voxcpm
