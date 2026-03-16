/**
 * @file test_minicpm.cpp
 * @brief Unit Tests for MiniCPM transformer module
 */

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "voxcpm/backend.h"
#include "voxcpm/context.h"
#include "voxcpm/minicpm.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

using Catch::Approx;
using json = nlohmann::json;

namespace voxcpm {
namespace test {

namespace {

constexpr const char* kModelPath = "/home/orangepi/Codes/ggbond/VoxCPM.cpp/models/voxcpm1.5.gguf";
constexpr const char* kTraceDir = "/home/orangepi/Codes/ggbond/VoxCPM.cpp/tests/fixtures/trace/";
constexpr float kTraceTolerance = 0.05f;
constexpr float kMaxMismatchRate = 0.05f;

struct TensorStats {
    float min_val = 0.0f;
    float max_val = 0.0f;
    float mean = 0.0f;
    float range = 0.0f;
};

struct ForwardStepTraceMeta {
    int hidden_size = 0;
    int num_layers = 0;
    int n_kv_heads = 0;
    int kv_seq_len = 0;
    int head_dim = 0;
};

struct ForwardStepTraceData {
    std::vector<float> input_embeds;
    int position = -1;
    std::vector<float> output;
    std::vector<std::vector<float>> input_keys;
    std::vector<std::vector<float>> input_values;
    std::vector<std::vector<float>> output_keys;
    std::vector<std::vector<float>> output_values;
};

enum class ForwardStepCapture {
    None,
    InputEmbeds,
    PositionId,
    InputKey,
    InputValue,
    Output0,
    OutputKey,
    OutputValue,
};

bool file_exists(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    return file.good();
}

json load_first_jsonl(const std::string& path) {
    std::ifstream file(path);
    REQUIRE(file.is_open());

    std::string line;
    REQUIRE(std::getline(file, line));
    REQUIRE_FALSE(line.empty());
    return json::parse(line);
}

ForwardStepTraceMeta load_forward_step_trace_meta(const std::string& path) {
    const json meta = load_first_jsonl(path);

    ForwardStepTraceMeta result;
    const json& input_shape = meta.at("input_shapes").at("inputs_embeds").at("shape");
    const json& kv_shape = meta.at("input_shapes").at("past_key_values").at("key_shape");

    REQUIRE(input_shape.size() == 2);
    REQUIRE(kv_shape.size() == 4);

    result.hidden_size = input_shape[1].get<int>();
    result.num_layers = meta.at("input_shapes").at("past_key_values").at("num_layers").get<int>();
    result.n_kv_heads = kv_shape[1].get<int>();
    result.kv_seq_len = kv_shape[2].get<int>();
    result.head_dim = kv_shape[3].get<int>();
    return result;
}

// For batch size 1, PyTorch [B, T, C] and GGML [C, T] share the same flat order.
std::vector<float> flatten_btc_tensor(const json& tensor) {
    REQUIRE(tensor.is_array());
    const size_t batch = tensor.size();
    REQUIRE(batch == 1);

    const size_t seq_len = tensor[0].size();
    const size_t hidden = seq_len > 0 ? tensor[0][0].size() : 0;

    std::vector<float> flat;
    flat.reserve(batch * seq_len * hidden);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t t = 0; t < seq_len; ++t) {
            REQUIRE(tensor[b][t].size() == hidden);
            for (size_t c = 0; c < hidden; ++c) {
                flat.push_back(tensor[b][t][c].get<float>());
            }
        }
    }
    return flat;
}

// For batch size 1, PyTorch [B, H, T, D] and GGML [D, T, H] also share flat order.
std::vector<float> flatten_bhtd_tensor(const json& tensor) {
    REQUIRE(tensor.is_array());
    const size_t batch = tensor.size();
    REQUIRE(batch == 1);

    const size_t n_heads = tensor[0].size();
    const size_t seq_len = n_heads > 0 ? tensor[0][0].size() : 0;
    const size_t head_dim = seq_len > 0 ? tensor[0][0][0].size() : 0;

    std::vector<float> flat;
    flat.reserve(batch * n_heads * seq_len * head_dim);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t h = 0; h < n_heads; ++h) {
            REQUIRE(tensor[b][h].size() == seq_len);
            for (size_t t = 0; t < seq_len; ++t) {
                REQUIRE(tensor[b][h][t].size() == head_dim);
                for (size_t d = 0; d < head_dim; ++d) {
                    flat.push_back(tensor[b][h][t][d].get<float>());
                }
            }
        }
    }
    return flat;
}

const json& find_kv_layer(const json& kv_cache, int layer_idx) {
    REQUIRE(kv_cache.contains("layers"));
    for (const auto& layer : kv_cache.at("layers")) {
        if (layer.contains("layer") && layer.at("layer").get<int>() == layer_idx) {
            return layer;
        }
    }
    FAIL("Missing KV cache layer " << layer_idx);
    throw std::runtime_error("unreachable");
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

struct ForwardStepTraceSaxHandler : public json::json_sax_t {
    struct ContainerState {
        bool pushed_path = false;
        bool is_capture_array = false;
    };

    explicit ForwardStepTraceSaxHandler(ForwardStepTraceData& trace,
                                        const ForwardStepTraceMeta& meta)
        : trace_(trace), meta_(meta) {
        const size_t cache_elems =
            static_cast<size_t>(meta_.n_kv_heads) * meta_.kv_seq_len * meta_.head_dim;

        trace_.input_embeds.reserve(meta_.hidden_size);
        trace_.output.reserve(meta_.hidden_size);
        trace_.input_keys.assign(meta_.num_layers, {});
        trace_.input_values.assign(meta_.num_layers, {});
        trace_.output_keys.assign(meta_.num_layers, {});
        trace_.output_values.assign(meta_.num_layers, {});

        for (int i = 0; i < meta_.num_layers; ++i) {
            trace_.input_keys[i].reserve(cache_elems);
            trace_.input_values[i].reserve(cache_elems);
            trace_.output_keys[i].reserve(cache_elems);
            trace_.output_values[i].reserve(cache_elems);
        }
    }

    bool null() override {
        pending_key_.clear();
        return true;
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

    bool string(string_t&) override {
        pending_key_.clear();
        return true;
    }

    bool binary(json::binary_t&) override {
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
        if (state.pushed_path) {
            if (path_.empty()) {
                return false;
            }
            path_.pop_back();
        }
        pending_key_.clear();
        return true;
    }

    bool start_array(std::size_t) override {
        const bool pushed = push_pending_key();
        const ForwardStepCapture capture = capture_for_current_path();
        if (capture != ForwardStepCapture::None) {
            active_capture_ = capture;
        }
        containers_.push_back({pushed, capture != ForwardStepCapture::None});
        return true;
    }

    bool end_array() override {
        if (containers_.empty()) {
            return false;
        }
        const ContainerState state = containers_.back();
        containers_.pop_back();
        if (state.is_capture_array) {
            active_capture_ = ForwardStepCapture::None;
        }
        if (state.pushed_path) {
            if (path_.empty()) {
                return false;
            }
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
        return false;
    }

private:
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

        size_t idx = 0;
        for (const char* token : expected) {
            if (path_[idx] != token) {
                return false;
            }
            ++idx;
        }
        return true;
    }

    ForwardStepCapture capture_for_current_path() const {
        if (path_equals({"inputs", "inputs_embeds"})) {
            return ForwardStepCapture::InputEmbeds;
        }
        if (path_equals({"inputs", "position_id"})) {
            return ForwardStepCapture::PositionId;
        }
        if (path_equals({"inputs", "past_key_values", "layers", "key"})) {
            return ForwardStepCapture::InputKey;
        }
        if (path_equals({"inputs", "past_key_values", "layers", "value"})) {
            return ForwardStepCapture::InputValue;
        }
        if (path_equals({"outputs", "output_0"})) {
            return ForwardStepCapture::Output0;
        }
        if (path_equals({"outputs", "output_1", "layers", "key"})) {
            return ForwardStepCapture::OutputKey;
        }
        if (path_equals({"outputs", "output_1", "layers", "value"})) {
            return ForwardStepCapture::OutputValue;
        }
        return ForwardStepCapture::None;
    }

    bool handle_integer(int64_t val) {
        if (active_capture_ == ForwardStepCapture::PositionId) {
            trace_.position = static_cast<int>(val);
            return true;
        }

        if (pending_key_ == "layer") {
            if (path_equals({"inputs", "past_key_values", "layers"})) {
                current_input_layer_ = static_cast<int>(val);
                return true;
            }
            if (path_equals({"outputs", "output_1", "layers"})) {
                current_output_layer_ = static_cast<int>(val);
                return true;
            }
        }

        return handle_float(static_cast<float>(val));
    }

    bool handle_float(float val) {
        switch (active_capture_) {
        case ForwardStepCapture::InputEmbeds:
            trace_.input_embeds.push_back(val);
            return true;
        case ForwardStepCapture::Output0:
            trace_.output.push_back(val);
            return true;
        case ForwardStepCapture::InputKey:
            if (current_input_layer_ < 0) {
                return false;
            }
            trace_.input_keys[current_input_layer_].push_back(val);
            return true;
        case ForwardStepCapture::InputValue:
            if (current_input_layer_ < 0) {
                return false;
            }
            trace_.input_values[current_input_layer_].push_back(val);
            return true;
        case ForwardStepCapture::OutputKey:
            if (current_output_layer_ < 0) {
                return false;
            }
            trace_.output_keys[current_output_layer_].push_back(val);
            return true;
        case ForwardStepCapture::OutputValue:
            if (current_output_layer_ < 0) {
                return false;
            }
            trace_.output_values[current_output_layer_].push_back(val);
            return true;
        case ForwardStepCapture::PositionId:
            trace_.position = static_cast<int>(val);
            return true;
        case ForwardStepCapture::None:
            return true;
        }

        return true;
    }

    ForwardStepTraceData& trace_;
    const ForwardStepTraceMeta& meta_;
    std::vector<std::string> path_;
    std::vector<ContainerState> containers_;
    std::string pending_key_;
    ForwardStepCapture active_capture_ = ForwardStepCapture::None;
    int current_input_layer_ = -1;
    int current_output_layer_ = -1;
};

ForwardStepTraceData load_forward_step_trace_streaming(const std::string& trace_path,
                                                      const ForwardStepTraceMeta& meta) {
    std::ifstream file(trace_path);
    REQUIRE(file.is_open());

    ForwardStepTraceData trace;
    ForwardStepTraceSaxHandler handler(trace, meta);
    REQUIRE(json::sax_parse(file, &handler));

    const size_t cache_elems =
        static_cast<size_t>(meta.n_kv_heads) * meta.kv_seq_len * meta.head_dim;
    REQUIRE(trace.input_embeds.size() == static_cast<size_t>(meta.hidden_size));
    REQUIRE(trace.output.size() == static_cast<size_t>(meta.hidden_size));
    REQUIRE(trace.position >= 0);
    REQUIRE(trace.input_keys.size() == static_cast<size_t>(meta.num_layers));
    REQUIRE(trace.input_values.size() == static_cast<size_t>(meta.num_layers));
    REQUIRE(trace.output_keys.size() == static_cast<size_t>(meta.num_layers));
    REQUIRE(trace.output_values.size() == static_cast<size_t>(meta.num_layers));

    for (int i = 0; i < meta.num_layers; ++i) {
        REQUIRE(trace.input_keys[i].size() == cache_elems);
        REQUIRE(trace.input_values[i].size() == cache_elems);
        REQUIRE(trace.output_keys[i].size() == cache_elems);
        REQUIRE(trace.output_values[i].size() == cache_elems);
    }

    return trace;
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
    stats.range = stats.max_val - stats.min_val;
    return stats;
}

void validate_with_tolerance(const std::vector<float>& actual,
                             const std::vector<float>& expected,
                             const std::vector<float>& reference,
                             float tolerance,
                             float max_mismatch_rate,
                             const std::string& test_name,
                             bool verbose = true) {
    REQUIRE(actual.size() == expected.size());

    int mismatches = 0;
    float max_error = 0.0f;
    float sum_error = 0.0f;

    for (size_t i = 0; i < actual.size(); ++i) {
        const float error = std::abs(actual[i] - expected[i]);
        if (error > tolerance) {
            ++mismatches;
        }
        max_error = std::max(max_error, error);
        sum_error += error;
    }

    const float mismatch_rate = static_cast<float>(mismatches) / static_cast<float>(actual.size());
    const float avg_error = sum_error / static_cast<float>(actual.size());

    const TensorStats ref_stats = compute_stats(reference);
    const TensorStats expected_stats = compute_stats(expected);
    const TensorStats actual_stats = compute_stats(actual);

    if (verbose) {
        std::cout << "\n=== " << test_name << " ===\n";
        std::cout << "elements: " << actual.size() << "\n";
        std::cout << "reference range: [" << ref_stats.min_val << ", " << ref_stats.max_val
                  << "], mean=" << ref_stats.mean << "\n";
        std::cout << "expected range: [" << expected_stats.min_val << ", " << expected_stats.max_val
                  << "], mean=" << expected_stats.mean << "\n";
        std::cout << "actual range: [" << actual_stats.min_val << ", " << actual_stats.max_val
                  << "], mean=" << actual_stats.mean << "\n";
        std::cout << "max abs error: " << max_error << "\n";
        std::cout << "avg abs error: " << avg_error << "\n";
        std::cout << "mismatch rate (> " << tolerance << "): " << (mismatch_rate * 100.0f) << "%\n";
    }

    INFO(test_name);
    INFO("max abs error: " << max_error);
    INFO("avg abs error: " << avg_error);
    INFO("mismatch rate: " << mismatch_rate);
    REQUIRE(mismatch_rate <= max_mismatch_rate);
}

void run_forward_trace_validation(const std::string& model_prefix,
                                  const std::string& trace_name,
                                  const std::string& label) {
    const std::string trace_path = std::string(kTraceDir) + trace_name;
    if (!file_exists(kModelPath)) {
        WARN("Model file not found, skipping test: " << kModelPath);
        return;
    }
    if (!file_exists(trace_path)) {
        WARN("Trace file not found, skipping test: " << trace_path);
        return;
    }

    const json trace = load_first_jsonl(trace_path);
    const json& input_json = trace.at("inputs").at("inputs_embeds");
    const json& output_json = trace.at("outputs").at("output_0");

    REQUIRE(input_json.size() == 1);
    REQUIRE(output_json.size() == 1);

    const int seq_len = static_cast<int>(input_json[0].size());
    const int hidden = seq_len > 0 ? static_cast<int>(input_json[0][0].size()) : 0;
    REQUIRE(seq_len > 0);
    REQUIRE(hidden > 0);

    VoxCPMBackend backend(BackendType::CPU, 2);
    VoxCPMContext weight_ctx(ContextType::Weights, 256);
    VoxCPMContext graph_ctx(ContextType::Graph, 8192, 65536);

    MiniCPMModel model;
    REQUIRE(model.load_from_gguf(kModelPath, model_prefix, weight_ctx, graph_ctx, backend));
    REQUIRE(model.config().hidden_size == hidden);

    MiniCPMKVCache kv_cache(model.config().n_layer,
                            model.config().n_kv_heads,
                            model.config().max_length,
                            model.config().head_dim());
    kv_cache.init(backend);
    kv_cache.clear();

    ggml_tensor* input = graph_ctx.new_tensor_2d(GGML_TYPE_F32, hidden, seq_len);
    REQUIRE(input != nullptr);
    ggml_set_input(input);

    ggml_tensor* output = model.forward(graph_ctx, input, nullptr, kv_cache, true);
    REQUIRE(output != nullptr);
    ggml_set_output(output);

    ggml_cgraph* graph = graph_ctx.new_graph();
    REQUIRE(graph != nullptr);
    graph_ctx.build_forward(graph, output);
    backend.alloc_graph(graph);

    const std::vector<float> input_data = flatten_btc_tensor(input_json);
    const std::vector<float> expected_output = flatten_btc_tensor(output_json);
    std::vector<float> actual_output(expected_output.size(), 0.0f);

    backend.tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));
    REQUIRE(backend.compute(graph) == GGML_STATUS_SUCCESS);
    backend.tensor_get(output, actual_output.data(), 0, actual_output.size() * sizeof(float));

    validate_with_tolerance(actual_output, expected_output, input_data,
                            kTraceTolerance, kMaxMismatchRate,
                            label + " output");

    const json& kv_output = trace.at("outputs").at("output_1");
    std::array<int, 2> layers_to_check = {0, model.config().n_layer - 1};
    if (layers_to_check[0] == layers_to_check[1]) {
        layers_to_check[1] = -1;
    }

    for (int layer_idx : layers_to_check) {
        if (layer_idx < 0) {
            continue;
        }

        const json& layer = find_kv_layer(kv_output, layer_idx);
        const std::vector<float> expected_k = flatten_bhtd_tensor(layer.at("key"));
        const std::vector<float> expected_v = flatten_bhtd_tensor(layer.at("value"));
        const std::vector<float> actual_k = read_cache_prefix(
            backend, kv_cache.raw_k_cache(layer_idx),
            kv_cache.head_dim(), kv_cache.n_kv_heads(),
            kv_cache.max_length(), seq_len);
        const std::vector<float> actual_v = read_cache_prefix(
            backend, kv_cache.raw_v_cache(layer_idx),
            kv_cache.head_dim(), kv_cache.n_kv_heads(),
            kv_cache.max_length(), seq_len);

        validate_with_tolerance(actual_k, expected_k, expected_k,
                                kTraceTolerance, kMaxMismatchRate,
                                label + " key layer " + std::to_string(layer_idx),
                                false);
        validate_with_tolerance(actual_v, expected_v, expected_v,
                                kTraceTolerance, kMaxMismatchRate,
                                label + " value layer " + std::to_string(layer_idx),
                                false);
    }
}

void run_forward_step_trace_validation(const std::string& model_prefix,
                                       const std::string& trace_name,
                                       const std::string& meta_name,
                                       const std::string& label) {
    const std::string trace_path = std::string(kTraceDir) + trace_name;
    const std::string meta_path = std::string(kTraceDir) + meta_name;

    if (!file_exists(kModelPath)) {
        WARN("Model file not found, skipping test: " << kModelPath);
        return;
    }
    if (!file_exists(trace_path)) {
        WARN("Trace file not found, skipping test: " << trace_path);
        return;
    }
    if (!file_exists(meta_path)) {
        WARN("Trace meta file not found, skipping test: " << meta_path);
        return;
    }

    const ForwardStepTraceMeta meta = load_forward_step_trace_meta(meta_path);
    const ForwardStepTraceData trace = load_forward_step_trace_streaming(trace_path, meta);

    VoxCPMBackend backend(BackendType::CPU, 2);
    VoxCPMContext weight_ctx(ContextType::Weights, 256);
    VoxCPMContext graph_ctx(ContextType::Graph, 8192, 65536);

    MiniCPMModel model;
    REQUIRE(model.load_from_gguf(kModelPath, model_prefix, weight_ctx, graph_ctx, backend));
    REQUIRE(model.config().hidden_size == meta.hidden_size);
    REQUIRE(model.config().n_layer == meta.num_layers);
    REQUIRE(model.config().n_kv_heads == meta.n_kv_heads);
    REQUIRE(model.config().head_dim() == meta.head_dim);

    MiniCPMKVCache kv_cache(model.config().n_layer,
                            model.config().n_kv_heads,
                            model.config().max_length,
                            model.config().head_dim());
    kv_cache.init(backend);
    kv_cache.clear();

    for (int layer = 0; layer < model.config().n_layer; ++layer) {
        write_cache_prefix(backend, kv_cache.raw_k_cache(layer), trace.input_keys[layer],
                           model.config().head_dim(), model.config().n_kv_heads,
                           model.config().max_length, meta.kv_seq_len);
        write_cache_prefix(backend, kv_cache.raw_v_cache(layer), trace.input_values[layer],
                           model.config().head_dim(), model.config().n_kv_heads,
                           model.config().max_length, meta.kv_seq_len);
    }

    ggml_tensor* input = graph_ctx.new_tensor_1d(GGML_TYPE_F32, meta.hidden_size);
    REQUIRE(input != nullptr);
    ggml_set_input(input);

    ggml_tensor* output = model.forward_step(graph_ctx, input, trace.position, kv_cache, true);
    REQUIRE(output != nullptr);
    ggml_set_output(output);

    ggml_cgraph* graph = graph_ctx.new_graph();
    REQUIRE(graph != nullptr);
    graph_ctx.build_forward(graph, output);
    backend.alloc_graph(graph);

    std::vector<float> actual_output(trace.output.size(), 0.0f);
    backend.tensor_set(input, trace.input_embeds.data(), 0, trace.input_embeds.size() * sizeof(float));
    REQUIRE(backend.compute(graph) == GGML_STATUS_SUCCESS);
    backend.tensor_get(output, actual_output.data(), 0, actual_output.size() * sizeof(float));

    validate_with_tolerance(actual_output, trace.output, trace.input_embeds,
                            kTraceTolerance, kMaxMismatchRate,
                            label + " output");

    std::vector<std::vector<float>> actual_keys(model.config().n_layer);
    std::vector<std::vector<float>> actual_values(model.config().n_layer);
    for (int layer = 0; layer < model.config().n_layer; ++layer) {
        actual_keys[layer] = read_cache_prefix(
            backend, kv_cache.raw_k_cache(layer),
            model.config().head_dim(), model.config().n_kv_heads,
            model.config().max_length, meta.kv_seq_len);
        actual_values[layer] = read_cache_prefix(
            backend, kv_cache.raw_v_cache(layer),
            model.config().head_dim(), model.config().n_kv_heads,
            model.config().max_length, meta.kv_seq_len);
    }

    const std::vector<float> expected_keys = concat_cache_layers(trace.output_keys);
    const std::vector<float> expected_values = concat_cache_layers(trace.output_values);
    const std::vector<float> actual_keys_flat = concat_cache_layers(actual_keys);
    const std::vector<float> actual_values_flat = concat_cache_layers(actual_values);

    validate_with_tolerance(actual_keys_flat, expected_keys, expected_keys,
                            kTraceTolerance, kMaxMismatchRate,
                            label + " full key cache");
    validate_with_tolerance(actual_values_flat, expected_values, expected_values,
                            kTraceTolerance, kMaxMismatchRate,
                            label + " full value cache");
}

}  // namespace

TEST_CASE("MiniCPMConfig defaults", "[minicpm][config]") {
    MiniCPMConfig config;

    REQUIRE(config.hidden_size == 1024);
    REQUIRE(config.intermediate_size == 4096);
    REQUIRE(config.n_layer == 8);
    REQUIRE(config.n_heads == 16);
    REQUIRE(config.n_kv_heads == 2);
    REQUIRE(config.head_dim() == 64);
    REQUIRE(config.use_mup == false);
    REQUIRE(config.scale_depth == Approx(1.4f));
}

TEST_CASE("MiniCPMKVCache init and views", "[minicpm][kv]") {
    VoxCPMBackend backend(BackendType::CPU, 2);
    MiniCPMKVCache kv_cache(2, 2, 16, 8);
    kv_cache.init(backend);

    REQUIRE(kv_cache.n_layer() == 2);
    REQUIRE(kv_cache.max_length() == 16);

    VoxCPMContext graph_ctx(ContextType::Graph, 64, 512);
    ggml_tensor* k = kv_cache.get_k(graph_ctx.raw_context(), 0, 4);
    ggml_tensor* v = kv_cache.get_v(graph_ctx.raw_context(), 1, 7);
    ggml_tensor* slot = kv_cache.get_k_slot(graph_ctx.raw_context(), 0, 3);
    ggml_tensor* batch = kv_cache.get_v_batch(graph_ctx.raw_context(), 1, 5, 2);

    REQUIRE(k != nullptr);
    REQUIRE(v != nullptr);
    REQUIRE(slot != nullptr);
    REQUIRE(batch != nullptr);

    REQUIRE(k->ne[0] == 8);
    REQUIRE(k->ne[1] == 4);
    REQUIRE(k->ne[2] == 2);

    REQUIRE(v->ne[0] == 8);
    REQUIRE(v->ne[1] == 7);
    REQUIRE(v->ne[2] == 2);

    REQUIRE(slot->ne[0] == 8);
    REQUIRE(slot->ne[1] == 2);
    REQUIRE(slot->ne[2] == 1);

    REQUIRE(batch->ne[0] == 8);
    REQUIRE(batch->ne[1] == 2);
    REQUIRE(batch->ne[2] == 2);
}

TEST_CASE("MiniCPM model loads BaseLM weights from GGUF", "[minicpm][integration]") {
    const char* model_path = "/home/orangepi/Codes/ggbond/VoxCPM.cpp/models/voxcpm1.5.gguf";
    std::ifstream file(model_path, std::ios::binary);
    if (!file.is_open()) {
        WARN("Model file not found, skipping test: " << model_path);
        return;
    }
    file.close();

    VoxCPMBackend backend(BackendType::CPU, 2);
    VoxCPMContext weight_ctx(ContextType::Weights, 256);
    VoxCPMContext graph_ctx(ContextType::Graph, 512, 4096);

    MiniCPMModel model;
    REQUIRE(model.load_from_gguf(model_path, "", weight_ctx, graph_ctx, backend));

    REQUIRE(model.config().n_layer == 24);
    REQUIRE(model.config().hidden_size == 1024);
    REQUIRE(model.config().n_heads == 16);
    REQUIRE(model.config().n_kv_heads == 2);
    REQUIRE(model.weights().embed_tokens != nullptr);
    REQUIRE(model.weights().norm != nullptr);
    REQUIRE(model.weights().layers.size() == 24);
    REQUIRE(model.get_pos_tensor() != nullptr);
}

TEST_CASE("MiniCPM model loads variant-specific config from GGUF", "[minicpm][integration]") {
    const char* model_path = "/home/orangepi/Codes/ggbond/VoxCPM.cpp/models/voxcpm1.5.gguf";
    std::ifstream file(model_path, std::ios::binary);
    if (!file.is_open()) {
        WARN("Model file not found, skipping test: " << model_path);
        return;
    }
    file.close();

    VoxCPMBackend backend(BackendType::CPU, 2);
    VoxCPMContext weight_ctx(ContextType::Weights, 256);
    VoxCPMContext graph_ctx(ContextType::Graph, 512, 4096);

    {
        MiniCPMModel residual;
        REQUIRE(residual.load_from_gguf(model_path, "residual_lm", weight_ctx, graph_ctx, backend));
        REQUIRE(residual.config().n_layer == 8);
        REQUIRE(residual.config().hidden_size == 1024);
        REQUIRE(residual.config().intermediate_size == 4096);
        REQUIRE(residual.config().n_heads == 16);
    }

    {
        MiniCPMModel locenc;
        REQUIRE(locenc.load_from_gguf(model_path, "locenc", weight_ctx, graph_ctx, backend));
        REQUIRE(locenc.config().n_layer == 8);
        REQUIRE(locenc.config().hidden_size == 1024);
        REQUIRE(locenc.config().intermediate_size == 4096);
        REQUIRE(locenc.config().n_heads == 16);
    }

    {
        MiniCPMModel locdit;
        REQUIRE(locdit.load_from_gguf(model_path, "locdit", weight_ctx, graph_ctx, backend));
        REQUIRE(locdit.config().n_layer == 8);
        REQUIRE(locdit.config().hidden_size == 1024);
        REQUIRE(locdit.config().intermediate_size == 4096);
        REQUIRE(locdit.config().n_heads == 16);
    }
}

TEST_CASE("MiniCPM forward graph builds", "[minicpm][integration]") {
    const char* model_path = "/home/orangepi/Codes/ggbond/VoxCPM.cpp/models/voxcpm1.5.gguf";
    std::ifstream file(model_path, std::ios::binary);
    if (!file.is_open()) {
        WARN("Model file not found, skipping test: " << model_path);
        return;
    }
    file.close();

    VoxCPMBackend backend(BackendType::CPU, 2);
    VoxCPMContext weight_ctx(ContextType::Weights, 256);
    VoxCPMContext graph_ctx(ContextType::Graph, 4096, 32768);

    MiniCPMModel model;
    REQUIRE(model.load_from_gguf(model_path, "", weight_ctx, graph_ctx, backend));

    MiniCPMKVCache kv_cache(model.config().n_layer,
                            model.config().n_kv_heads,
                            model.config().max_length,
                            model.config().head_dim());
    kv_cache.init(backend);

    ggml_tensor* input = graph_ctx.new_tensor_2d(GGML_TYPE_F32, model.config().hidden_size, 4);
    ggml_set_input(input);

    ggml_tensor* positions = ggml_view_1d(graph_ctx.raw_context(), model.get_pos_tensor(), 4, 0);
    ggml_tensor* output = model.forward(graph_ctx, input, positions, kv_cache, true);
    REQUIRE(output != nullptr);
    REQUIRE(output->ne[0] == model.config().hidden_size);
    REQUIRE(output->ne[1] == 4);

    ggml_set_output(output);
    ggml_cgraph* graph = graph_ctx.new_graph();
    graph_ctx.build_forward(graph, output);

    REQUIRE(graph != nullptr);
    REQUIRE(ggml_graph_node(graph, 0) != nullptr);
}

TEST_CASE("MiniCPM decode graph builds", "[minicpm][integration]") {
    const char* model_path = "/home/orangepi/Codes/ggbond/VoxCPM.cpp/models/voxcpm1.5.gguf";
    std::ifstream file(model_path, std::ios::binary);
    if (!file.is_open()) {
        WARN("Model file not found, skipping test: " << model_path);
        return;
    }
    file.close();

    VoxCPMBackend backend(BackendType::CPU, 2);
    VoxCPMContext weight_ctx(ContextType::Weights, 256);
    VoxCPMContext graph_ctx(ContextType::Graph, 4096, 32768);

    MiniCPMModel model;
    REQUIRE(model.load_from_gguf(model_path, "residual_lm", weight_ctx, graph_ctx, backend));

    MiniCPMKVCache kv_cache(model.config().n_layer,
                            model.config().n_kv_heads,
                            model.config().max_length,
                            model.config().head_dim());
    kv_cache.init(backend);

    ggml_tensor* input = graph_ctx.new_tensor_1d(GGML_TYPE_F32, model.config().hidden_size);
    ggml_set_input(input);

    ggml_tensor* output = model.forward_step(graph_ctx, input, 3, kv_cache, true);
    REQUIRE(output != nullptr);
    REQUIRE(output->ne[0] == model.config().hidden_size);

    ggml_set_output(output);
    ggml_cgraph* graph = graph_ctx.new_graph();
    graph_ctx.build_forward(graph, output);

    REQUIRE(graph != nullptr);
    REQUIRE(ggml_graph_node(graph, 0) != nullptr);
}

TEST_CASE("MiniCPM base_lm forward trace aligns with torch", "[minicpm][trace][prefill]") {
    run_forward_trace_validation("", "trace_MiniCPM_base_lm_forward.jsonl", "MiniCPM base_lm forward");
}

TEST_CASE("MiniCPM residual_lm forward trace aligns with torch", "[minicpm][trace][prefill]") {
    run_forward_trace_validation("residual_lm", "trace_MiniCPM_residual_lm_forward.jsonl", "MiniCPM residual_lm forward");
}

TEST_CASE("MiniCPM base_lm forward_step trace aligns with torch", "[minicpm][trace][decode]") {
    run_forward_step_trace_validation("",
                                      "trace_MiniCPM_base_lm_forward_step_one.jsonl",
                                      "meta_MiniCPM_base_lm_forward_step_one.jsonl",
                                      "MiniCPM base_lm forward_step");
}

TEST_CASE("MiniCPM residual_lm forward_step trace aligns with torch", "[minicpm][trace][decode]") {
    run_forward_step_trace_validation("residual_lm",
                                      "trace_MiniCPM_residual_lm_forward_step_one.jsonl",
                                      "meta_MiniCPM_residual_lm_forward_step_one.jsonl",
                                      "MiniCPM residual_lm forward_step");
}

}  // namespace test
}  // namespace voxcpm
