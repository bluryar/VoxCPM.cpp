/**
 * @file test_locdit.cpp
 * @brief Unit tests for VoxCPM Local DiT
 */

#include <catch2/catch_test_macros.hpp>

#include "voxcpm/backend.h"
#include "voxcpm/context.h"
#include "voxcpm/locdit.h"

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
constexpr const char* kTracePath = "/home/orangepi/Codes/ggbond/VoxCPM.cpp/tests/fixtures/trace/trace_LocalDit.jsonl";
constexpr float kTraceTolerance = 0.08f;
constexpr float kMaxMismatchRate = 0.05f;

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

std::vector<float> flatten_bct_to_ctb(const json& tensor) {
    REQUIRE(tensor.is_array());
    const size_t batch = tensor.size();
    const size_t feat_dim = batch > 0 ? tensor[0].size() : 0;
    const size_t seq_len = feat_dim > 0 ? tensor[0][0].size() : 0;

    std::vector<float> flat;
    flat.reserve(batch * feat_dim * seq_len);
    for (size_t b = 0; b < batch; ++b) {
        REQUIRE(tensor[b].size() == feat_dim);
        for (size_t t = 0; t < seq_len; ++t) {
            for (size_t c = 0; c < feat_dim; ++c) {
                REQUIRE(tensor[b][c].size() == seq_len);
                flat.push_back(tensor[b][c][t].get<float>());
            }
        }
    }
    return flat;
}

std::vector<float> flatten_bh_to_hb(const json& tensor) {
    REQUIRE(tensor.is_array());
    const size_t batch = tensor.size();
    const size_t hidden_size = batch > 0 ? tensor[0].size() : 0;

    std::vector<float> flat;
    flat.reserve(batch * hidden_size);
    for (size_t b = 0; b < batch; ++b) {
        REQUIRE(tensor[b].size() == hidden_size);
        for (size_t h = 0; h < hidden_size; ++h) {
            flat.push_back(tensor[b][h].get<float>());
        }
    }
    return flat;
}

std::vector<float> flatten_1d(const json& tensor) {
    REQUIRE(tensor.is_array());
    std::vector<float> flat;
    flat.reserve(tensor.size());
    for (const auto& value : tensor) {
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
    return stats;
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

    const TensorStats ref_stats = compute_stats(reference);
    const TensorStats expected_stats = compute_stats(expected);
    const TensorStats actual_stats = compute_stats(actual);

    std::cout << "\n=== " << label << " ===\n";
    std::cout << "elements: " << actual.size() << "\n";
    std::cout << "input range: [" << ref_stats.min_val << ", " << ref_stats.max_val
              << "], mean=" << ref_stats.mean << "\n";
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

TEST_CASE("LocDiT model loads from GGUF", "[locdit][integration]") {
    if (!file_exists(kModelPath)) {
        WARN("Model file not found, skipping test: " << kModelPath);
        return;
    }

    VoxCPMBackend backend(BackendType::CPU, 2);
    VoxCPMContext weight_ctx(ContextType::Weights, 256);
    VoxCPMContext graph_ctx(ContextType::Graph, 1024, 4096);

    LocDiTModel locdit;
    REQUIRE(locdit.load_from_gguf(kModelPath, weight_ctx, graph_ctx, backend));

    REQUIRE(locdit.weights().in_proj_weight != nullptr);
    REQUIRE(locdit.weights().cond_proj_weight != nullptr);
    REQUIRE(locdit.weights().out_proj_weight != nullptr);
    REQUIRE(locdit.weights().time_mlp_linear1_weight != nullptr);
    REQUIRE(locdit.weights().delta_time_mlp_linear2_bias != nullptr);

    REQUIRE(locdit.feat_dim() == 64);
    REQUIRE(locdit.config().hidden_size == 1024);
    REQUIRE(locdit.config().intermediate_size == 4096);
    REQUIRE(locdit.config().n_layer == 8);
    REQUIRE(locdit.config().n_heads == 16);
}

TEST_CASE("LocDiT forward builds for batch one and two", "[locdit][integration]") {
    if (!file_exists(kModelPath)) {
        WARN("Model file not found, skipping test: " << kModelPath);
        return;
    }

    VoxCPMBackend backend(BackendType::CPU, 2);
    VoxCPMContext weight_ctx(ContextType::Weights, 256);
    VoxCPMContext graph_ctx(ContextType::Graph, 32768, 262144);

    LocDiTModel locdit;
    REQUIRE(locdit.load_from_gguf(kModelPath, weight_ctx, graph_ctx, backend));

    for (int batch : {1, 2}) {
        ggml_tensor* x = graph_ctx.new_tensor_3d(GGML_TYPE_F32, locdit.feat_dim(), 4, batch);
        ggml_tensor* mu = graph_ctx.new_tensor_2d(GGML_TYPE_F32, locdit.config().hidden_size, batch);
        ggml_tensor* t = graph_ctx.new_tensor_1d(GGML_TYPE_F32, batch);
        ggml_tensor* cond = graph_ctx.new_tensor_3d(GGML_TYPE_F32, locdit.feat_dim(), 4, batch);
        ggml_tensor* dt = graph_ctx.new_tensor_1d(GGML_TYPE_F32, batch);

        ggml_set_input(x);
        ggml_set_input(mu);
        ggml_set_input(t);
        ggml_set_input(cond);
        ggml_set_input(dt);

        ggml_tensor* output = locdit.forward(graph_ctx, x, mu, t, cond, dt);
        REQUIRE(output != nullptr);
        REQUIRE(output->ne[0] == locdit.feat_dim());
        REQUIRE(output->ne[1] == 4);
        REQUIRE(output->ne[2] == batch);
        ggml_set_output(output);

        ggml_cgraph* graph = graph_ctx.new_graph();
        REQUIRE(graph != nullptr);
        graph_ctx.build_forward(graph, output);
        backend.alloc_graph(graph);

        std::vector<float> x_data(static_cast<size_t>(locdit.feat_dim()) * 4 * batch, 0.0f);
        std::vector<float> mu_data(static_cast<size_t>(locdit.config().hidden_size) * batch, 0.0f);
        std::vector<float> t_data(static_cast<size_t>(batch), 0.0f);
        std::vector<float> cond_data(static_cast<size_t>(locdit.feat_dim()) * 4 * batch, 0.0f);
        std::vector<float> dt_data(static_cast<size_t>(batch), 0.0f);
        std::vector<float> actual(static_cast<size_t>(locdit.feat_dim()) * 4 * batch, 0.0f);

        backend.tensor_set(x, x_data.data(), 0, x_data.size() * sizeof(float));
        backend.tensor_set(mu, mu_data.data(), 0, mu_data.size() * sizeof(float));
        backend.tensor_set(t, t_data.data(), 0, t_data.size() * sizeof(float));
        backend.tensor_set(cond, cond_data.data(), 0, cond_data.size() * sizeof(float));
        backend.tensor_set(dt, dt_data.data(), 0, dt_data.size() * sizeof(float));

        REQUIRE(backend.compute(graph) == GGML_STATUS_SUCCESS);
        backend.tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
    }
}

TEST_CASE("LocDiT matches trace sample", "[locdit][trace][integration]") {
    if (!file_exists(kModelPath)) {
        WARN("Model file not found, skipping test: " << kModelPath);
        return;
    }
    if (!file_exists(kTracePath)) {
        WARN("Trace file not found, skipping test: " << kTracePath);
        return;
    }

    const json trace = load_jsonl_line(kTracePath, 0);
    const std::vector<float> x_data = flatten_bct_to_ctb(trace.at("inputs").at("arg_0"));
    const std::vector<float> mu_data = flatten_bh_to_hb(trace.at("inputs").at("arg_1"));
    const std::vector<float> t_data = flatten_1d(trace.at("inputs").at("arg_2"));
    const std::vector<float> cond_data = flatten_bct_to_ctb(trace.at("inputs").at("arg_3"));
    const std::vector<float> dt_data = flatten_1d(trace.at("inputs").at("arg_4"));
    const std::vector<float> expected = flatten_bct_to_ctb(trace.at("outputs").at("output"));

    VoxCPMBackend backend(BackendType::CPU, 2);
    VoxCPMContext weight_ctx(ContextType::Weights, 256);
    VoxCPMContext graph_ctx(ContextType::Graph, 32768, 262144);

    LocDiTModel locdit;
    REQUIRE(locdit.load_from_gguf(kModelPath, weight_ctx, graph_ctx, backend));

    ggml_tensor* x = graph_ctx.new_tensor_3d(GGML_TYPE_F32, locdit.feat_dim(), 4, 2);
    ggml_tensor* mu = graph_ctx.new_tensor_2d(GGML_TYPE_F32, locdit.config().hidden_size, 2);
    ggml_tensor* t = graph_ctx.new_tensor_1d(GGML_TYPE_F32, 2);
    ggml_tensor* cond = graph_ctx.new_tensor_3d(GGML_TYPE_F32, locdit.feat_dim(), 4, 2);
    ggml_tensor* dt = graph_ctx.new_tensor_1d(GGML_TYPE_F32, 2);

    ggml_set_input(x);
    ggml_set_input(mu);
    ggml_set_input(t);
    ggml_set_input(cond);
    ggml_set_input(dt);

    ggml_tensor* output = locdit.forward(graph_ctx, x, mu, t, cond, dt);
    REQUIRE(output != nullptr);
    ggml_set_output(output);

    ggml_cgraph* graph = graph_ctx.new_graph();
    REQUIRE(graph != nullptr);
    graph_ctx.build_forward(graph, output);
    backend.alloc_graph(graph);

    std::vector<float> actual(expected.size(), 0.0f);
    backend.tensor_set(x, x_data.data(), 0, x_data.size() * sizeof(float));
    backend.tensor_set(mu, mu_data.data(), 0, mu_data.size() * sizeof(float));
    backend.tensor_set(t, t_data.data(), 0, t_data.size() * sizeof(float));
    backend.tensor_set(cond, cond_data.data(), 0, cond_data.size() * sizeof(float));
    backend.tensor_set(dt, dt_data.data(), 0, dt_data.size() * sizeof(float));

    REQUIRE(backend.compute(graph) == GGML_STATUS_SUCCESS);
    backend.tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));

    validate_with_tolerance(actual, expected, x_data,
                            kTraceTolerance, kMaxMismatchRate,
                            "LocDiT trace output");
}

}  // namespace test
}  // namespace voxcpm
