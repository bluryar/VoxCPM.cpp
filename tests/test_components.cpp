/**
 * @file test_components.cpp
 * @brief Unit tests for VoxCPM auxiliary components
 */

#include <catch2/catch_test_macros.hpp>

#include "voxcpm/backend.h"
#include "voxcpm/components.h"
#include "voxcpm/context.h"

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
constexpr const char* kTraceDir = "/home/orangepi/Codes/ggbond/vendor/VoxCPM/trace/";
constexpr float kProjectionTolerance = 0.02f;
constexpr float kProjectionMaxMismatchRate = 0.05f;
constexpr float kStopTolerance = 0.02f;
constexpr float kStopMaxMismatchRate = 0.05f;
constexpr float kEmbeddingTolerance = 1.0e-6f;
constexpr float kEmbeddingMaxMismatchRate = 0.0f;

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

std::vector<float> flatten_btc_to_ctb(const json& tensor) {
    REQUIRE(tensor.is_array());
    const size_t batch = tensor.size();
    REQUIRE(batch == 1);

    const size_t seq_len = tensor[0].size();
    const size_t channels = seq_len > 0 ? tensor[0][0].size() : 0;

    std::vector<float> out;
    out.reserve(batch * seq_len * channels);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t t = 0; t < seq_len; ++t) {
            REQUIRE(tensor[b][t].size() == channels);
            for (size_t c = 0; c < channels; ++c) {
                out.push_back(tensor[b][t][c].get<float>());
            }
        }
    }
    return out;
}

std::vector<float> flatten_bc_to_cb(const json& tensor) {
    REQUIRE(tensor.is_array());
    const size_t batch = tensor.size();
    REQUIRE(batch == 1);

    const size_t channels = tensor[0].size();
    std::vector<float> out;
    out.reserve(batch * channels);
    for (size_t b = 0; b < batch; ++b) {
        REQUIRE(tensor[b].size() == channels);
        for (size_t c = 0; c < channels; ++c) {
            out.push_back(tensor[b][c].get<float>());
        }
    }
    return out;
}

std::vector<int32_t> flatten_token_ids(const json& tensor) {
    REQUIRE(tensor.is_array());
    REQUIRE(tensor.size() == 1);

    std::vector<int32_t> out;
    out.reserve(tensor[0].size());
    for (const auto& value : tensor[0]) {
        out.push_back(value.get<int32_t>());
    }
    return out;
}

std::vector<float> flatten_embed_output_to_ct(const json& tensor) {
    REQUIRE(tensor.is_array());
    REQUIRE(tensor.size() == 1);

    const size_t seq_len = tensor[0].size();
    const size_t hidden = seq_len > 0 ? tensor[0][0].size() : 0;

    std::vector<float> out;
    out.reserve(seq_len * hidden);
    for (size_t t = 0; t < seq_len; ++t) {
        REQUIRE(tensor[0][t].size() == hidden);
        for (size_t c = 0; c < hidden; ++c) {
            out.push_back(tensor[0][t][c].get<float>());
        }
    }
    return out;
}

std::vector<float> ints_to_floats(const std::vector<int32_t>& input) {
    return std::vector<float>(input.begin(), input.end());
}

}  // namespace

TEST_CASE("Component configs expose expected defaults", "[components]") {
    ProjectionConfig projection;
    StopTokenConfig stop_token;
    EmbeddingConfig embedding;

    REQUIRE(projection.in_dim == 1024);
    REQUIRE(projection.out_dim == 1024);
    REQUIRE(stop_token.hidden_dim == 1024);
    REQUIRE(stop_token.num_classes == 2);
    REQUIRE(embedding.vocab_size == 73448);
    REQUIRE(embedding.hidden_dim == 1024);
    REQUIRE(embedding.scale == 1.0f);
}

TEST_CASE("VoxCPMComponents loads all submodules", "[components][container][integration]") {
    if (!file_exists(kModelPath)) {
        WARN("Model file not found, skipping test: " << kModelPath);
        return;
    }

    VoxCPMBackend backend(BackendType::CPU, 2);
    VoxCPMContext weight_ctx(ContextType::Weights, 512);
    VoxCPMContext graph_ctx(ContextType::Graph, 1024, 16384);

    std::unique_ptr<VoxCPMComponents> components = VoxCPMComponents::from_gguf(
        kModelPath,
        1024,
        73448,
        12.0f,
        weight_ctx,
        graph_ctx,
        backend);

    REQUIRE(components != nullptr);
    REQUIRE(components->enc_to_lm_proj() != nullptr);
    REQUIRE(components->lm_to_dit_proj() != nullptr);
    REQUIRE(components->res_to_dit_proj() != nullptr);
    REQUIRE(components->stop_token() != nullptr);
    REQUIRE(components->embed_tokens() != nullptr);
}

TEST_CASE("LinearProjection enc_to_lm matches trace", "[components][projection][trace]") {
    const std::string trace_path = std::string(kTraceDir) + "trace_proj_enc_to_lm.jsonl";
    if (!file_exists(kModelPath) || !file_exists(trace_path)) {
        WARN("enc_to_lm trace dependencies missing, skipping test");
        return;
    }

    const json trace = load_jsonl_line(trace_path, 0);
    REQUIRE(trace.at("module").get<std::string>() == "proj.enc_to_lm");

    const std::vector<float> input_data = flatten_btc_to_ctb(trace.at("inputs").at("arg_0"));
    const std::vector<float> expected = flatten_btc_to_ctb(trace.at("outputs").at("output"));

    VoxCPMBackend backend(BackendType::CPU, 2);
    VoxCPMContext weight_ctx(ContextType::Weights, 128);
    VoxCPMContext graph_ctx(ContextType::Graph, 2048, 65536);

    LinearProjection projection;
    REQUIRE(projection.load_from_gguf(kModelPath, "proj.enc_to_lm", weight_ctx, graph_ctx, backend));

    ggml_tensor* input = graph_ctx.new_tensor_3d(GGML_TYPE_F32, 1024, 100, 1);
    REQUIRE(input != nullptr);
    ggml_set_input(input);

    ggml_tensor* output = projection.forward(graph_ctx, input);
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

    validate_with_tolerance(actual, expected, input_data, kProjectionTolerance, kProjectionMaxMismatchRate, "LinearProjection enc_to_lm trace");
}

TEST_CASE("LinearProjection lm_to_dit matches trace", "[components][projection][trace]") {
    const std::string trace_path = std::string(kTraceDir) + "trace_proj_lm_to_dit.jsonl";
    if (!file_exists(kModelPath) || !file_exists(trace_path)) {
        WARN("lm_to_dit trace dependencies missing, skipping test");
        return;
    }

    const json trace = load_jsonl_line(trace_path, 0);
    REQUIRE(trace.at("module").get<std::string>() == "proj.lm_to_dit");

    const std::vector<float> input_data = flatten_bc_to_cb(trace.at("inputs").at("arg_0"));
    const std::vector<float> expected = flatten_bc_to_cb(trace.at("outputs").at("output"));

    VoxCPMBackend backend(BackendType::CPU, 2);
    VoxCPMContext weight_ctx(ContextType::Weights, 128);
    VoxCPMContext graph_ctx(ContextType::Graph, 1024, 16384);

    LinearProjection projection;
    REQUIRE(projection.load_from_gguf(kModelPath, "proj.lm_to_dit", weight_ctx, graph_ctx, backend));

    ggml_tensor* input = graph_ctx.new_tensor_1d(GGML_TYPE_F32, 1024);
    REQUIRE(input != nullptr);
    ggml_set_input(input);

    ggml_tensor* output = projection.forward(graph_ctx, input);
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

    validate_with_tolerance(actual, expected, input_data, kProjectionTolerance, kProjectionMaxMismatchRate, "LinearProjection lm_to_dit trace");
}

TEST_CASE("LinearProjection res_to_dit matches trace", "[components][projection][trace]") {
    const std::string trace_path = std::string(kTraceDir) + "trace_proj_res_to_dit.jsonl";
    if (!file_exists(kModelPath) || !file_exists(trace_path)) {
        WARN("res_to_dit trace dependencies missing, skipping test");
        return;
    }

    const json trace = load_jsonl_line(trace_path, 0);
    REQUIRE(trace.at("module").get<std::string>() == "proj.res_to_dit");

    const std::vector<float> input_data = flatten_bc_to_cb(trace.at("inputs").at("arg_0"));
    const std::vector<float> expected = flatten_bc_to_cb(trace.at("outputs").at("output"));

    VoxCPMBackend backend(BackendType::CPU, 2);
    VoxCPMContext weight_ctx(ContextType::Weights, 128);
    VoxCPMContext graph_ctx(ContextType::Graph, 1024, 16384);

    LinearProjection projection;
    REQUIRE(projection.load_from_gguf(kModelPath, "proj.res_to_dit", weight_ctx, graph_ctx, backend));

    ggml_tensor* input = graph_ctx.new_tensor_1d(GGML_TYPE_F32, 1024);
    REQUIRE(input != nullptr);
    ggml_set_input(input);

    ggml_tensor* output = projection.forward(graph_ctx, input);
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

    validate_with_tolerance(actual, expected, input_data, kProjectionTolerance, kProjectionMaxMismatchRate, "LinearProjection res_to_dit trace");
}

TEST_CASE("StopTokenPredictor matches trace", "[components][stop_token][trace]") {
    const std::string proj_trace_path = std::string(kTraceDir) + "trace_proj_stop_proj.jsonl";
    const std::string head_trace_path = std::string(kTraceDir) + "trace_proj_stop_head.jsonl";
    if (!file_exists(kModelPath) || !file_exists(proj_trace_path) || !file_exists(head_trace_path)) {
        WARN("stop token trace dependencies missing, skipping test");
        return;
    }

    const json proj_trace = load_jsonl_line(proj_trace_path, 0);
    const json head_trace = load_jsonl_line(head_trace_path, 0);
    REQUIRE(proj_trace.at("module").get<std::string>() == "proj.stop_proj");
    REQUIRE(head_trace.at("module").get<std::string>() == "proj.stop_head");

    const std::vector<float> input_data = flatten_bc_to_cb(proj_trace.at("inputs").at("arg_0"));
    const std::vector<float> expected = flatten_bc_to_cb(head_trace.at("outputs").at("output"));

    VoxCPMBackend backend(BackendType::CPU, 2);
    VoxCPMContext weight_ctx(ContextType::Weights, 128);
    VoxCPMContext graph_ctx(ContextType::Graph, 1024, 16384);

    StopTokenPredictor predictor;
    REQUIRE(predictor.load_from_gguf(kModelPath, weight_ctx, graph_ctx, backend));

    ggml_tensor* input = graph_ctx.new_tensor_1d(GGML_TYPE_F32, 1024);
    REQUIRE(input != nullptr);
    ggml_set_input(input);

    ggml_tensor* output = predictor.forward(graph_ctx, input);
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

    validate_with_tolerance(actual, expected, input_data, kStopTolerance, kStopMaxMismatchRate, "StopTokenPredictor trace");
}

TEST_CASE("Embedding matches trace", "[components][embedding][trace]") {
    const std::string trace_path = std::string(kTraceDir) + "trace_embed_tokens.jsonl";
    if (!file_exists(kModelPath) || !file_exists(trace_path)) {
        WARN("embedding trace dependencies missing, skipping test");
        return;
    }

    const json trace = load_jsonl_line(trace_path, 0);
    REQUIRE(trace.at("module").get<std::string>() == "embed_tokens");

    const std::vector<int32_t> token_ids = flatten_token_ids(trace.at("inputs").at("arg_0"));
    const std::vector<float> expected = flatten_embed_output_to_ct(trace.at("outputs").at("output"));
    const std::vector<float> input_reference = ints_to_floats(token_ids);

    VoxCPMBackend backend(BackendType::CPU, 2);
    VoxCPMContext weight_ctx(ContextType::Weights, 128);
    VoxCPMContext graph_ctx(ContextType::Graph, 1024, 32768);

    Embedding embedding(EmbeddingConfig{73448, 1024, 1.0f});
    REQUIRE(embedding.load_from_gguf(kModelPath, weight_ctx, graph_ctx, backend));

    ggml_tensor* token_tensor = graph_ctx.new_tensor_1d(GGML_TYPE_I32, static_cast<int64_t>(token_ids.size()));
    REQUIRE(token_tensor != nullptr);
    ggml_set_input(token_tensor);

    ggml_tensor* output = embedding.forward(graph_ctx, token_tensor);
    REQUIRE(output != nullptr);
    ggml_set_output(output);

    ggml_cgraph* graph = graph_ctx.new_graph();
    REQUIRE(graph != nullptr);
    graph_ctx.build_forward(graph, output);
    backend.alloc_graph(graph);

    backend.tensor_set(token_tensor, token_ids.data(), 0, token_ids.size() * sizeof(int32_t));
    REQUIRE(backend.compute(graph) == GGML_STATUS_SUCCESS);

    std::vector<float> actual(expected.size(), 0.0f);
    backend.tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));

    validate_with_tolerance(actual, expected, input_reference, kEmbeddingTolerance, kEmbeddingMaxMismatchRate, "Embedding trace");
}

}  // namespace test
}  // namespace voxcpm
