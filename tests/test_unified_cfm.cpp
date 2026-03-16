/**
 * @file test_unified_cfm.cpp
 * @brief Unit tests for VoxCPM Unified CFM
 */

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "voxcpm/backend.h"
#include "voxcpm/context.h"
#include "voxcpm/locdit.h"
#include "voxcpm/unified_cfm.h"

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
constexpr const char* kTracePath = "/home/orangepi/Codes/ggbond/VoxCPM.cpp/tests/fixtures/trace/trace_UnifiedCFM.jsonl";
constexpr float kTraceTolerance = 0.12f;
constexpr float kMaxMismatchRate = 0.10f;

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

std::vector<float> flatten_single_bct_to_ct(const json& tensor) {
    REQUIRE(tensor.is_array());
    REQUIRE(tensor.size() == 1);

    const size_t feat_dim = tensor[0].size();
    const size_t seq_len = feat_dim > 0 ? tensor[0][0].size() : 0;

    std::vector<float> flat;
    flat.reserve(feat_dim * seq_len);
    for (size_t t = 0; t < seq_len; ++t) {
        for (size_t c = 0; c < feat_dim; ++c) {
            REQUIRE(tensor[0][c].size() == seq_len);
            flat.push_back(tensor[0][c][t].get<float>());
        }
    }
    return flat;
}

std::vector<float> flatten_single_bh_to_h(const json& tensor) {
    REQUIRE(tensor.is_array());
    REQUIRE(tensor.size() == 1);

    std::vector<float> flat;
    flat.reserve(tensor[0].size());
    for (const auto& value : tensor[0]) {
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

TEST_CASE("UnifiedCFM compute_t_span matches formula", "[unified_cfm][unit]") {
    const std::vector<float> t_span = UnifiedCFM::compute_t_span(10, 1.0f);
    REQUIRE(t_span.size() == 11);

    for (int i = 0; i <= 10; ++i) {
        const float base = 1.0f - static_cast<float>(i) / 10.0f;
        const float expected = base + 1.0f * (std::cos(static_cast<float>(M_PI_2) * base) - 1.0f + base);
        REQUIRE(t_span[static_cast<size_t>(i)] == Approx(expected).margin(1.0e-6f));
    }
}

TEST_CASE("UnifiedCFM forward graph builds", "[unified_cfm][integration]") {
    if (!file_exists(kModelPath)) {
        WARN("Model file not found, skipping test: " << kModelPath);
        return;
    }

    VoxCPMBackend backend(BackendType::CPU, 2);
    VoxCPMContext weight_ctx(ContextType::Weights, 256);
    VoxCPMContext graph_ctx(ContextType::Graph, 65536, 524288);

    LocDiTModel locdit;
    REQUIRE(locdit.load_from_gguf(kModelPath, weight_ctx, graph_ctx, backend));

    UnifiedCFM cfm(locdit);

    ggml_tensor* z = graph_ctx.new_tensor_2d(GGML_TYPE_F32, locdit.feat_dim(), 4);
    ggml_tensor* mu = graph_ctx.new_tensor_1d(GGML_TYPE_F32, locdit.config().hidden_size);
    ggml_tensor* cond = graph_ctx.new_tensor_2d(GGML_TYPE_F32, locdit.feat_dim(), 4);
    ggml_set_input(z);
    ggml_set_input(mu);
    ggml_set_input(cond);

    ggml_tensor* output = cfm.forward(graph_ctx, z, mu, 4, cond, 10, 2.0f);
    REQUIRE(output != nullptr);
    REQUIRE(output->ne[0] == locdit.feat_dim());
    REQUIRE(output->ne[1] == 4);
    ggml_set_output(output);

    ggml_cgraph* graph = graph_ctx.new_graph();
    REQUIRE(graph != nullptr);
    graph_ctx.build_forward(graph, output);
    backend.alloc_graph(graph);

    std::vector<float> z_data(static_cast<size_t>(locdit.feat_dim()) * 4, 0.0f);
    std::vector<float> mu_data(static_cast<size_t>(locdit.config().hidden_size), 0.0f);
    std::vector<float> cond_data(static_cast<size_t>(locdit.feat_dim()) * 4, 0.0f);
    std::vector<float> actual(static_cast<size_t>(locdit.feat_dim()) * 4, 0.0f);

    backend.tensor_set(z, z_data.data(), 0, z_data.size() * sizeof(float));
    backend.tensor_set(mu, mu_data.data(), 0, mu_data.size() * sizeof(float));
    backend.tensor_set(cond, cond_data.data(), 0, cond_data.size() * sizeof(float));

    REQUIRE(backend.compute(graph) == GGML_STATUS_SUCCESS);
    backend.tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
}

TEST_CASE("UnifiedCFM matches trace sample", "[unified_cfm][trace][integration]") {
    if (!file_exists(kModelPath)) {
        WARN("Model file not found, skipping test: " << kModelPath);
        return;
    }
    if (!file_exists(kTracePath)) {
        WARN("Trace file not found, skipping test: " << kTracePath);
        return;
    }

    const json trace = load_jsonl_line(kTracePath, 0);
    const std::vector<float> z_data = flatten_single_bct_to_ct(trace.at("inputs").at("z"));
    const std::vector<float> mu_data = flatten_single_bh_to_h(trace.at("inputs").at("mu"));
    const int patch_size = trace.at("inputs").at("patch_size").get<int>();
    const std::vector<float> cond_data = flatten_single_bct_to_ct(trace.at("inputs").at("cond"));
    const int n_timesteps = trace.at("inputs").at("n_timesteps").get<int>();
    const float cfg_value = trace.at("inputs").at("cfg_value").get<float>();
    const std::vector<float> expected = flatten_single_bct_to_ct(trace.at("outputs").at("output"));

    VoxCPMBackend backend(BackendType::CPU, 2);
    VoxCPMContext weight_ctx(ContextType::Weights, 256);
    VoxCPMContext graph_ctx(ContextType::Graph, 65536, 524288);

    LocDiTModel locdit;
    REQUIRE(locdit.load_from_gguf(kModelPath, weight_ctx, graph_ctx, backend));

    UnifiedCFM cfm(locdit);

    ggml_tensor* z = graph_ctx.new_tensor_2d(GGML_TYPE_F32, locdit.feat_dim(), patch_size);
    ggml_tensor* mu = graph_ctx.new_tensor_1d(GGML_TYPE_F32, locdit.config().hidden_size);
    ggml_tensor* cond = graph_ctx.new_tensor_2d(GGML_TYPE_F32, locdit.feat_dim(), patch_size);
    ggml_set_input(z);
    ggml_set_input(mu);
    ggml_set_input(cond);

    ggml_tensor* output = cfm.forward(graph_ctx, z, mu, patch_size, cond, n_timesteps, cfg_value);
    REQUIRE(output != nullptr);
    ggml_set_output(output);

    ggml_cgraph* graph = graph_ctx.new_graph();
    REQUIRE(graph != nullptr);
    graph_ctx.build_forward(graph, output);
    backend.alloc_graph(graph);

    std::vector<float> actual(expected.size(), 0.0f);
    backend.tensor_set(z, z_data.data(), 0, z_data.size() * sizeof(float));
    backend.tensor_set(mu, mu_data.data(), 0, mu_data.size() * sizeof(float));
    backend.tensor_set(cond, cond_data.data(), 0, cond_data.size() * sizeof(float));

    REQUIRE(backend.compute(graph) == GGML_STATUS_SUCCESS);
    backend.tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));

    validate_with_tolerance(actual, expected, z_data,
                            kTraceTolerance, kMaxMismatchRate,
                            "UnifiedCFM trace output");
}

}  // namespace test
}  // namespace voxcpm
