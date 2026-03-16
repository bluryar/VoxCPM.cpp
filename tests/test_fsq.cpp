/**
 * @file test_fsq.cpp
 * @brief Unit tests for VoxCPM FSQ
 */

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "voxcpm/backend.h"
#include "voxcpm/context.h"
#include "voxcpm/fsq.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

using Catch::Approx;
using json = nlohmann::json;

namespace voxcpm {
namespace test {

namespace {

constexpr const char* kModelPath = "/home/orangepi/Codes/ggbond/VoxCPM.cpp/models/voxcpm1.5.gguf";
constexpr const char* kTracePath = "/home/orangepi/Codes/ggbond/VoxCPM.cpp/tests/fixtures/trace/trace_FSQ.jsonl";
constexpr float kTraceTolerance = 0.05f;
constexpr float kMaxMismatchRate = 0.0f;

struct TensorStats {
    float min_val = 0.0f;
    float max_val = 0.0f;
    float mean = 0.0f;
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

std::vector<float> flatten_btc_to_ctb(const json& tensor) {
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

std::vector<float> flatten_bc_to_cb(const json& tensor) {
    REQUIRE(tensor.is_array());
    const size_t batch = tensor.size();
    REQUIRE(batch == 1);

    const size_t hidden = tensor[0].size();
    std::vector<float> flat;
    flat.reserve(batch * hidden);
    for (size_t b = 0; b < batch; ++b) {
        REQUIRE(tensor[b].size() == hidden);
        for (size_t c = 0; c < hidden; ++c) {
            flat.push_back(tensor[b][c].get<float>());
        }
    }
    return flat;
}

TensorStats compute_stats(const std::vector<float>& data) {
    REQUIRE_FALSE(data.empty());

    TensorStats stats;
    stats.min_val = std::numeric_limits<float>::max();
    stats.max_val = std::numeric_limits<float>::lowest();
    for (float value : data) {
        stats.min_val = std::min(stats.min_val, value);
        stats.max_val = std::max(stats.max_val, value);
        stats.mean += value;
    }
    stats.mean /= static_cast<float>(data.size());
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

TEST_CASE("FSQ config defaults", "[fsq]") {
    FSQConfig config;
    REQUIRE(config.latent_dim == 256);
    REQUIRE(config.scale == 9);
    REQUIRE(config.hidden_size == 1024);
}

TEST_CASE("FSQ quantize matches scalar formula", "[fsq][unit]") {
    VoxCPMContext graph_ctx(ContextType::Graph, 128, 4096);
    VoxCPMBackend backend(BackendType::CPU, 1);

    ggml_tensor* input = graph_ctx.new_tensor_1d(GGML_TYPE_F32, 10);
    REQUIRE(input != nullptr);
    ggml_set_input(input);

    ggml_tensor* output = fsq_quantize(graph_ctx.raw_context(), input, 9);
    REQUIRE(output != nullptr);
    ggml_set_output(output);

    ggml_cgraph* graph = graph_ctx.new_graph();
    REQUIRE(graph != nullptr);
    graph_ctx.build_forward(graph, output);
    backend.alloc_graph(graph);

    const std::vector<float> input_data = {-0.9f, -0.5f, -0.1f, 0.0f, 0.1f, 0.3f, 0.5f, 0.7f, 0.9f, 1.0f};
    backend.tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));
    REQUIRE(backend.compute(graph) == GGML_STATUS_SUCCESS);

    std::vector<float> actual(input_data.size(), 0.0f);
    backend.tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));

    for (size_t i = 0; i < input_data.size(); ++i) {
        const float expected = std::round(input_data[i] * 9.0f) / 9.0f;
        REQUIRE(actual[i] == Approx(expected).margin(1.0e-6f));
    }
}

TEST_CASE("FSQ loads from GGUF", "[fsq][integration]") {
    if (!file_exists(kModelPath)) {
        WARN("Model file not found, skipping test: " << kModelPath);
        return;
    }

    VoxCPMBackend backend(BackendType::CPU, 2);
    VoxCPMContext weight_ctx(ContextType::Weights, 128);
    VoxCPMContext graph_ctx(ContextType::Graph, 256, 4096);

    FSQ fsq;
    REQUIRE(fsq.load_from_gguf(kModelPath, weight_ctx, graph_ctx, backend));

    REQUIRE(fsq.weights().in_proj_weight != nullptr);
    REQUIRE(fsq.weights().in_proj_bias != nullptr);
    REQUIRE(fsq.weights().out_proj_weight != nullptr);
    REQUIRE(fsq.weights().out_proj_bias != nullptr);

    REQUIRE(fsq.hidden_size() == 1024);
    REQUIRE(fsq.latent_dim() == 256);
    REQUIRE(fsq.scale() == 9);

    REQUIRE(fsq.weights().in_proj_weight->ne[0] == 1024);
    REQUIRE(fsq.weights().in_proj_weight->ne[1] == 256);
    REQUIRE(fsq.weights().out_proj_weight->ne[0] == 256);
    REQUIRE(fsq.weights().out_proj_weight->ne[1] == 1024);
}

TEST_CASE("FSQ forward executes for prefill and decode shapes", "[fsq][integration]") {
    if (!file_exists(kModelPath)) {
        WARN("Model file not found, skipping test: " << kModelPath);
        return;
    }

    VoxCPMBackend backend(BackendType::CPU, 2);
    VoxCPMContext weight_ctx(ContextType::Weights, 128);
    VoxCPMContext graph_ctx(ContextType::Graph, 2048, 32768);

    FSQ fsq;
    REQUIRE(fsq.load_from_gguf(kModelPath, weight_ctx, graph_ctx, backend));

    SECTION("prefill path") {
        ggml_tensor* input = graph_ctx.new_tensor_2d(GGML_TYPE_F32, fsq.hidden_size(), 4);
        REQUIRE(input != nullptr);
        ggml_set_input(input);

        ggml_tensor* output = fsq.forward(graph_ctx, input);
        REQUIRE(output != nullptr);
        REQUIRE(output->ne[0] == fsq.hidden_size());
        REQUIRE(output->ne[1] == 4);
        ggml_set_output(output);

        ggml_cgraph* graph = graph_ctx.new_graph();
        REQUIRE(graph != nullptr);
        graph_ctx.build_forward(graph, output);
        backend.alloc_graph(graph);

        std::vector<float> input_data(static_cast<size_t>(fsq.hidden_size()) * 4, 0.0f);
        for (size_t i = 0; i < input_data.size(); ++i) {
            input_data[i] = static_cast<float>((static_cast<int>(i) % 17) - 8) / 16.0f;
        }
        std::vector<float> actual(input_data.size(), 0.0f);

        backend.tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));
        REQUIRE(backend.compute(graph) == GGML_STATUS_SUCCESS);
        backend.tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));

        REQUIRE(std::any_of(actual.begin(), actual.end(), [](float v) { return std::fabs(v) > 1.0e-6f; }));
    }

    SECTION("decode path") {
        ggml_tensor* input = graph_ctx.new_tensor_1d(GGML_TYPE_F32, fsq.hidden_size());
        REQUIRE(input != nullptr);
        ggml_set_input(input);

        ggml_tensor* output = fsq.forward(graph_ctx, input);
        REQUIRE(output != nullptr);
        REQUIRE(output->ne[0] == fsq.hidden_size());
        ggml_set_output(output);

        ggml_cgraph* graph = graph_ctx.new_graph();
        REQUIRE(graph != nullptr);
        graph_ctx.build_forward(graph, output);
        backend.alloc_graph(graph);

        std::vector<float> input_data(static_cast<size_t>(fsq.hidden_size()), 0.0f);
        for (size_t i = 0; i < input_data.size(); ++i) {
            input_data[i] = static_cast<float>((static_cast<int>(i) % 11) - 5) / 8.0f;
        }
        std::vector<float> actual(input_data.size(), 0.0f);

        backend.tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));
        REQUIRE(backend.compute(graph) == GGML_STATUS_SUCCESS);
        backend.tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));

        REQUIRE(std::any_of(actual.begin(), actual.end(), [](float v) { return std::fabs(v) > 1.0e-6f; }));
    }
}

TEST_CASE("FSQ matches prefill trace", "[fsq][trace][prefill]") {
    if (!file_exists(kModelPath) || !file_exists(kTracePath)) {
        WARN("FSQ prefill trace dependencies missing, skipping test");
        return;
    }

    const json trace = load_jsonl_line(kTracePath, 0);
    REQUIRE(trace.at("module").get<std::string>() == "FSQ");

    const std::vector<float> input_data = flatten_btc_to_ctb(trace.at("inputs").at("arg_0"));
    const std::vector<float> expected = flatten_btc_to_ctb(trace.at("outputs").at("output"));

    VoxCPMBackend backend(BackendType::CPU, 2);
    VoxCPMContext weight_ctx(ContextType::Weights, 128);
    VoxCPMContext graph_ctx(ContextType::Graph, 2048, 65536);

    FSQ fsq;
    REQUIRE(fsq.load_from_gguf(kModelPath, weight_ctx, graph_ctx, backend));

    ggml_tensor* input = graph_ctx.new_tensor_3d(GGML_TYPE_F32, fsq.hidden_size(), 100, 1);
    REQUIRE(input != nullptr);
    ggml_set_input(input);

    ggml_tensor* output = fsq.forward(graph_ctx, input);
    REQUIRE(output != nullptr);
    ggml_set_output(output);

    ggml_cgraph* graph = graph_ctx.new_graph();
    REQUIRE(graph != nullptr);
    graph_ctx.build_forward(graph, output);
    backend.alloc_graph(graph);

    backend.tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));
    REQUIRE(backend.compute(graph) == GGML_STATUS_SUCCESS);

    std::vector<float> actual(expected.size(), 0.0f);
    backend.tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));

    validate_with_tolerance(actual, expected, input_data, kTraceTolerance, kMaxMismatchRate, "FSQ prefill trace");
}

TEST_CASE("FSQ matches decode trace", "[fsq][trace][decode]") {
    if (!file_exists(kModelPath) || !file_exists(kTracePath)) {
        WARN("FSQ decode trace dependencies missing, skipping test");
        return;
    }

    const json trace = load_jsonl_line(kTracePath, 1);
    REQUIRE(trace.at("module").get<std::string>() == "FSQ");

    const std::vector<float> input_data = flatten_bc_to_cb(trace.at("inputs").at("arg_0"));
    const std::vector<float> expected = flatten_bc_to_cb(trace.at("outputs").at("output"));

    VoxCPMBackend backend(BackendType::CPU, 2);
    VoxCPMContext weight_ctx(ContextType::Weights, 128);
    VoxCPMContext graph_ctx(ContextType::Graph, 1024, 16384);

    FSQ fsq;
    REQUIRE(fsq.load_from_gguf(kModelPath, weight_ctx, graph_ctx, backend));

    ggml_tensor* input = graph_ctx.new_tensor_1d(GGML_TYPE_F32, fsq.hidden_size());
    REQUIRE(input != nullptr);
    ggml_set_input(input);

    ggml_tensor* output = fsq.forward(graph_ctx, input);
    REQUIRE(output != nullptr);
    ggml_set_output(output);

    ggml_cgraph* graph = graph_ctx.new_graph();
    REQUIRE(graph != nullptr);
    graph_ctx.build_forward(graph, output);
    backend.alloc_graph(graph);

    backend.tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));
    REQUIRE(backend.compute(graph) == GGML_STATUS_SUCCESS);

    std::vector<float> actual(expected.size(), 0.0f);
    backend.tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));

    validate_with_tolerance(actual, expected, input_data, kTraceTolerance, kMaxMismatchRate, "FSQ decode trace");
}

}  // namespace test
}  // namespace voxcpm
