/**
 * @file test_localenc.cpp
 * @brief Unit tests for VoxCPM Local Encoder
 */

#include <catch2/catch_test_macros.hpp>

#include "voxcpm/backend.h"
#include "voxcpm/context.h"
#include "voxcpm/localenc.h"
#include "voxcpm/minicpm.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace voxcpm {
namespace test {

namespace {

constexpr const char* kModelPath = "/home/orangepi/Codes/ggbond/VoxCPM.cpp/models/voxcpm1.5.gguf";
constexpr const char* kTracePath = "/home/orangepi/Codes/ggbond/vendor/VoxCPM/trace/trace_LocalEnc.jsonl";
constexpr float kTraceTolerance = 0.08f;
constexpr float kMaxMismatchRate = 0.05f;

struct TensorStats {
    float min_val = 0.0f;
    float max_val = 0.0f;
    float mean = 0.0f;
    float range = 0.0f;
};

bool file_exists(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    return file.good();
}

json load_jsonl_line(const std::string& path, int line_index) {
    std::ifstream file(path);
    REQUIRE(file.is_open());

    std::string line;
    for (int i = 0; i <= line_index; ++i) {
        REQUIRE(std::getline(file, line));
    }
    REQUIRE_FALSE(line.empty());
    return json::parse(line);
}

std::vector<float> flatten_single_tpd_to_dp(const json& tensor) {
    REQUIRE(tensor.is_array());
    REQUIRE(tensor.size() == 1);
    REQUIRE(tensor[0].is_array());
    REQUIRE(tensor[0].size() == 1);

    const json& time_slice = tensor[0][0];
    const size_t n_patches = time_slice.size();
    const size_t feat_dim = n_patches > 0 ? time_slice[0].size() : 0;

    std::vector<float> flat;
    flat.reserve(n_patches * feat_dim);
    for (size_t p = 0; p < n_patches; ++p) {
        REQUIRE(time_slice[p].size() == feat_dim);
        for (size_t d = 0; d < feat_dim; ++d) {
            flat.push_back(time_slice[p][d].get<float>());
        }
    }
    return flat;
}

std::vector<float> flatten_single_tc_to_c(const json& tensor) {
    REQUIRE(tensor.is_array());
    REQUIRE(tensor.size() == 1);
    REQUIRE(tensor[0].is_array());
    REQUIRE(tensor[0].size() == 1);

    const json& time_slice = tensor[0][0];
    std::vector<float> flat;
    flat.reserve(time_slice.size());
    for (const auto& value : time_slice) {
        flat.push_back(value.get<float>());
    }
    return flat;
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
                             const std::vector<float>& input_reference,
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

    const TensorStats input_stats = compute_stats(input_reference);
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
    INFO(label << " mismatch count = " << mismatch_count << " / " << actual.size());
    INFO(label << " avg abs diff = " << avg_abs_diff);
    INFO(label << " mismatch rate = " << mismatch_rate);
    REQUIRE(mismatch_rate <= max_mismatch_rate);
}

}  // namespace

TEST_CASE("LocEnc model loads from GGUF", "[localenc][integration]") {
    if (!file_exists(kModelPath)) {
        WARN("Model file not found, skipping test: " << kModelPath);
        return;
    }

    VoxCPMBackend backend(BackendType::CPU, 2);
    VoxCPMContext weight_ctx(ContextType::Weights, 256);
    VoxCPMContext graph_ctx(ContextType::Graph, 512, 4096);

    LocEncModel locenc;
    REQUIRE(locenc.load_from_gguf(kModelPath, weight_ctx, graph_ctx, backend));

    REQUIRE(locenc.weights().in_proj_weight != nullptr);
    REQUIRE(locenc.weights().in_proj_bias != nullptr);
    REQUIRE(locenc.weights().special_token != nullptr);

    REQUIRE(locenc.feat_dim() == 64);
    REQUIRE(locenc.config().hidden_size == 1024);
    REQUIRE(locenc.config().intermediate_size == 4096);
    REQUIRE(locenc.config().n_layer == 8);
    REQUIRE(locenc.config().n_heads == 16);
}

TEST_CASE("LocEnc forward_patch executes projected path", "[localenc][integration]") {
    if (!file_exists(kModelPath)) {
        WARN("Model file not found, skipping test: " << kModelPath);
        return;
    }

    VoxCPMBackend backend(BackendType::CPU, 2);
    VoxCPMContext weight_ctx(ContextType::Weights, 256);
    VoxCPMContext graph_ctx(ContextType::Graph, 8192, 65536);

    LocEncModel locenc;
    REQUIRE(locenc.load_from_gguf(kModelPath, weight_ctx, graph_ctx, backend));

    ggml_tensor* input = graph_ctx.new_tensor_2d(GGML_TYPE_F32, locenc.feat_dim(), 4);
    REQUIRE(input != nullptr);
    ggml_set_input(input);

    ggml_tensor* output = locenc.forward_patch(graph_ctx, input);
    REQUIRE(output != nullptr);
    REQUIRE(output->ne[0] == locenc.config().hidden_size);
    REQUIRE(output->ne[1] == 1);
    ggml_set_output(output);

    ggml_cgraph* graph = graph_ctx.new_graph();
    REQUIRE(graph != nullptr);
    graph_ctx.build_forward(graph, output);
    backend.alloc_graph(graph);

    std::vector<float> zeros(static_cast<size_t>(locenc.feat_dim()) * 4, 0.0f);
    std::vector<float> actual(locenc.config().hidden_size, 0.0f);
    backend.tensor_set(input, zeros.data(), 0, zeros.size() * sizeof(float));
    REQUIRE(backend.compute(graph) == GGML_STATUS_SUCCESS);
    backend.tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
}

TEST_CASE("LocEnc forward_patch accepts hidden-size input", "[localenc][integration]") {
    if (!file_exists(kModelPath)) {
        WARN("Model file not found, skipping test: " << kModelPath);
        return;
    }

    VoxCPMBackend backend(BackendType::CPU, 2);
    VoxCPMContext weight_ctx(ContextType::Weights, 256);
    VoxCPMContext graph_ctx(ContextType::Graph, 8192, 65536);

    LocEncModel locenc;
    REQUIRE(locenc.load_from_gguf(kModelPath, weight_ctx, graph_ctx, backend));

    ggml_tensor* input = graph_ctx.new_tensor_2d(GGML_TYPE_F32, locenc.config().hidden_size, 4);
    REQUIRE(input != nullptr);
    ggml_set_input(input);

    ggml_tensor* output = locenc.forward_patch(graph_ctx, input);
    REQUIRE(output != nullptr);
    REQUIRE(output->ne[0] == locenc.config().hidden_size);
    ggml_set_output(output);

    ggml_cgraph* graph = graph_ctx.new_graph();
    REQUIRE(graph != nullptr);
    graph_ctx.build_forward(graph, output);
    backend.alloc_graph(graph);

    std::vector<float> zeros(static_cast<size_t>(locenc.config().hidden_size) * 4, 0.0f);
    std::vector<float> actual(locenc.config().hidden_size, 0.0f);
    backend.tensor_set(input, zeros.data(), 0, zeros.size() * sizeof(float));
    REQUIRE(backend.compute(graph) == GGML_STATUS_SUCCESS);
    backend.tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
}

TEST_CASE("LocEnc matches trace for single-step sample", "[localenc][trace][integration]") {
    if (!file_exists(kModelPath)) {
        WARN("Model file not found, skipping test: " << kModelPath);
        return;
    }
    if (!file_exists(kTracePath)) {
        WARN("Trace file not found, skipping test: " << kTracePath);
        return;
    }

    const json trace = load_jsonl_line(kTracePath, 1);
    const std::vector<float> input_data = flatten_single_tpd_to_dp(trace.at("inputs").at("arg_0"));
    const std::vector<float> expected = flatten_single_tc_to_c(trace.at("outputs").at("output"));

    VoxCPMBackend backend(BackendType::CPU, 2);
    VoxCPMContext weight_ctx(ContextType::Weights, 256);
    VoxCPMContext graph_ctx(ContextType::Graph, 8192, 65536);

    LocEncModel locenc;
    REQUIRE(locenc.load_from_gguf(kModelPath, weight_ctx, graph_ctx, backend));

    ggml_tensor* input = graph_ctx.new_tensor_2d(GGML_TYPE_F32, locenc.feat_dim(), 4);
    REQUIRE(input != nullptr);
    ggml_set_input(input);

    ggml_tensor* output = locenc.forward_patch(graph_ctx, input);
    REQUIRE(output != nullptr);
    ggml_set_output(output);

    ggml_cgraph* graph = graph_ctx.new_graph();
    REQUIRE(graph != nullptr);
    graph_ctx.build_forward(graph, output);
    backend.alloc_graph(graph);

    std::vector<float> actual(expected.size(), 0.0f);
    backend.tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));
    REQUIRE(backend.compute(graph) == GGML_STATUS_SUCCESS);
    backend.tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));

    validate_with_tolerance(actual, expected, input_data,
                            kTraceTolerance, kMaxMismatchRate,
                            "LocalEnc trace output");
}

TEST_CASE("MiniCPM locenc prefix supports non-causal forward", "[localenc][minicpm][integration]") {
    if (!file_exists(kModelPath)) {
        WARN("Model file not found, skipping test: " << kModelPath);
        return;
    }

    VoxCPMBackend backend(BackendType::CPU, 2);
    VoxCPMContext weight_ctx(ContextType::Weights, 256);
    VoxCPMContext graph_ctx(ContextType::Graph, 4096, 32768);

    MiniCPMModel encoder;
    REQUIRE(encoder.load_from_gguf(kModelPath, "locenc", weight_ctx, graph_ctx, backend));

    MiniCPMKVCache kv_cache(encoder.config().n_layer,
                            encoder.config().n_kv_heads,
                            encoder.config().max_length,
                            encoder.config().head_dim());
    kv_cache.init(backend);

    ggml_tensor* input = graph_ctx.new_tensor_2d(GGML_TYPE_F32, encoder.config().hidden_size, 5);
    REQUIRE(input != nullptr);
    ggml_set_input(input);

    ggml_tensor* output = encoder.forward(graph_ctx, input, nullptr, kv_cache, false);
    REQUIRE(output != nullptr);
    REQUIRE(output->ne[0] == encoder.config().hidden_size);
    REQUIRE(output->ne[1] == 5);
}

}  // namespace test
}  // namespace voxcpm
