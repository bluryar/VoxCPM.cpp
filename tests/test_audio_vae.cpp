#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "voxcpm/audio-vae.h"
#include "voxcpm/backend.h"
#include "voxcpm/context.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

using json = nlohmann::json;
using Catch::Approx;

namespace voxcpm {
namespace test {

namespace {

constexpr const char* kModelPath = "/home/orangepi/Codes/ggbond/VoxCPM.cpp/models/voxcpm1.5.gguf";
constexpr const char* kTraceEncodePath = "/home/orangepi/Codes/ggbond/VoxCPM.cpp/tests/fixtures/trace/trace_AudioVAE_encode.jsonl";
constexpr const char* kTraceDecodePath = "/home/orangepi/Codes/ggbond/VoxCPM.cpp/tests/fixtures/trace/trace_AudioVAE_decode.jsonl";
constexpr float kTargetMaxDiff = 2e-4f;

int cpu_thread_count() {
    if (const char* env = std::getenv("VOXCPM_TEST_THREADS")) {
        const int value = std::atoi(env);
        if (value > 0) {
            return value;
        }
    }
    return 2;
}

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

std::vector<float> flatten_bct_to_tc(const json& tensor) {
    REQUIRE(tensor.is_array());
    REQUIRE(tensor.size() == 1);
    const json& channels = tensor[0];
    const size_t c = channels.size();
    const size_t t = c > 0 ? channels[0].size() : 0;

    std::vector<float> out;
    out.reserve(t * c);
    for (size_t ci = 0; ci < c; ++ci) {
        REQUIRE(channels[ci].size() == t);
        for (size_t ti = 0; ti < t; ++ti) {
            out.push_back(channels[ci][ti].get<float>());
        }
    }
    return out;
}

std::vector<float> flatten_bt_to_t(const json& tensor) {
    REQUIRE(tensor.is_array());
    REQUIRE(tensor.size() == 1);
    const json& values = tensor[0];
    std::vector<float> out;
    out.reserve(values.size());
    for (const auto& v : values) {
        out.push_back(v.get<float>());
    }
    return out;
}

struct ErrorStats {
    float max_abs_diff = 0.0f;
    float mean_abs_diff = 0.0f;
    float rmse = 0.0f;
};

struct TensorStats {
    float min_val = 0.0f;
    float max_val = 0.0f;
    float mean = 0.0f;
};

struct GraphTimingStats {
    double build_ms = 0.0;
    double alloc_ms = 0.0;
    double compute_ms = 0.0;
};

TensorStats compute_tensor_stats(const std::vector<float>& data) {
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

ErrorStats compute_error_stats(const std::vector<float>& actual, const std::vector<float>& expected) {
    REQUIRE(actual.size() == expected.size());

    ErrorStats stats;
    double sum_abs = 0.0;
    double sum_sq = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double diff = static_cast<double>(actual[i]) - static_cast<double>(expected[i]);
        stats.max_abs_diff = std::max(stats.max_abs_diff, static_cast<float>(std::fabs(diff)));
        sum_abs += std::fabs(diff);
        sum_sq += diff * diff;
    }

    const double n = static_cast<double>(actual.size());
    stats.mean_abs_diff = static_cast<float>(sum_abs / n);
    stats.rmse = static_cast<float>(std::sqrt(sum_sq / n));
    return stats;
}

void print_error_stats(const char* label,
                       const std::vector<float>& input,
                       const std::vector<float>& expected,
                       const std::vector<float>& actual,
                       const ErrorStats& stats,
                       float tolerance) {
    const TensorStats input_stats = compute_tensor_stats(input);
    const TensorStats expected_stats = compute_tensor_stats(expected);
    const TensorStats actual_stats = compute_tensor_stats(actual);

    size_t mismatch_count = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (std::fabs(actual[i] - expected[i]) > tolerance) {
            ++mismatch_count;
        }
    }
    const float mismatch_rate = static_cast<float>(mismatch_count) / static_cast<float>(actual.size());

    std::cout << "\n=== " << label << " ===\n";
    std::cout << "elements: " << actual.size() << "\n";
    std::cout << "input range: [" << input_stats.min_val << ", " << input_stats.max_val
              << "], mean=" << input_stats.mean << "\n";
    std::cout << "expected range: [" << expected_stats.min_val << ", " << expected_stats.max_val
              << "], mean=" << expected_stats.mean << "\n";
    std::cout << "actual range: [" << actual_stats.min_val << ", " << actual_stats.max_val
              << "], mean=" << actual_stats.mean << "\n";
    std::cout << "max abs error: " << stats.max_abs_diff << "\n";
    std::cout << "avg abs error: " << stats.mean_abs_diff << "\n";
    std::cout << "rmse: " << stats.rmse << "\n";
    std::cout << "mismatch rate (> " << tolerance << "): " << (mismatch_rate * 100.0f) << "%\n";
}

void print_graph_timing(const char* label, const GraphTimingStats& stats) {
    std::cout << label
              << " graph timing: build=" << stats.build_ms << " ms"
              << ", alloc=" << stats.alloc_ms << " ms"
              << ", compute=" << stats.compute_ms << " ms\n";
}

template <typename PrepareInputFn>
GraphTimingStats run_graph_with_timing(VoxCPMContext& graph_ctx,
                                       VoxCPMBackend& backend,
                                       ggml_tensor* output,
                                       PrepareInputFn&& prepare_input) {
    GraphTimingStats stats;
    ggml_cgraph* graph = graph_ctx.new_graph();
    REQUIRE(graph != nullptr);

    const auto build_start = std::chrono::steady_clock::now();
    graph_ctx.build_forward(graph, output);
    const auto build_end = std::chrono::steady_clock::now();

    const auto alloc_start = build_end;
    backend.alloc_graph(graph);
    const auto alloc_end = std::chrono::steady_clock::now();

    prepare_input();

    const auto compute_start = alloc_end;
    REQUIRE(backend.compute(graph) == GGML_STATUS_SUCCESS);
    const auto compute_end = std::chrono::steady_clock::now();

    stats.build_ms = std::chrono::duration<double, std::milli>(build_end - build_start).count();
    stats.alloc_ms = std::chrono::duration<double, std::milli>(alloc_end - alloc_start).count();
    stats.compute_ms = std::chrono::duration<double, std::milli>(compute_end - compute_start).count();
    return stats;
}

}  // namespace

TEST_CASE("AudioVAE preprocess aligns to hop length", "[audio_vae][preprocess]") {
    AudioVAE vae;
    const int sample_rate = vae.config().sample_rate;
    const size_t hop_length = static_cast<size_t>(vae.config().hop_length());

    SECTION("aligned input remains unchanged") {
        std::vector<float> input(hop_length * 5, 0.25f);
        std::vector<float> processed = vae.preprocess(input, sample_rate);
        REQUIRE(processed.size() == hop_length * 5);
        REQUIRE(processed.front() == Approx(0.25f));
        REQUIRE(processed.back() == Approx(0.25f));
    }

    SECTION("unaligned input is right padded") {
        std::vector<float> input(hop_length * 5 + 1, 0.5f);
        std::vector<float> processed = vae.preprocess(input, sample_rate);
        REQUIRE(processed.size() == hop_length * 6);
        REQUIRE(processed[hop_length * 5] == Approx(0.5f));
        REQUIRE(processed.back() == Approx(0.0f));
    }
}

TEST_CASE("AudioVAE loads config and weights from GGUF", "[audio_vae][weights]") {
    if (!file_exists(kModelPath)) {
        WARN("Model file not found, skipping test");
        return;
    }

    VoxCPMBackend backend(BackendType::CPU, cpu_thread_count());
    VoxCPMContext weight_ctx(ContextType::Weights, 512);
    VoxCPMContext graph_ctx(ContextType::Graph, 4096, 65536);

    AudioVAE vae;
    REQUIRE(vae.load_from_gguf(kModelPath, weight_ctx, graph_ctx, backend));

    REQUIRE(vae.config().encoder_dim == 64);
    REQUIRE(vae.config().decoder_dim == 2048);
    REQUIRE(vae.config().latent_dim == 64);
    REQUIRE(vae.config().sample_rate == 44100);
    REQUIRE(vae.config().encoder_rates == std::vector<int>({2, 3, 6, 7, 7}));
    REQUIRE(vae.config().decoder_rates == std::vector<int>({7, 7, 6, 3, 2}));

    REQUIRE(vae.weights().encoder_block_0_weight != nullptr);
    REQUIRE(vae.weights().encoder_fc_mu_weight != nullptr);
    REQUIRE(vae.weights().decoder_model_0_weight != nullptr);
    REQUIRE(vae.weights().decoder_model_1_weight != nullptr);
    REQUIRE(vae.weights().decoder_final_conv_weight != nullptr);
    REQUIRE(vae.weights().encoder_blocks.size() == 5);
    REQUIRE(vae.weights().decoder_blocks.size() == 5);
}

TEST_CASE("AudioVAE encode matches trace", "[audio_vae][encode][trace]") {
    if (!file_exists(kModelPath) || !file_exists(kTraceEncodePath)) {
        WARN("Encode test dependencies missing, skipping test");
        return;
    }

    const json trace = load_jsonl_line(kTraceEncodePath, 0);
    std::vector<float> input_audio = flatten_bt_to_t(trace.at("inputs").at("audio_data"));
    const std::vector<float> expected = flatten_bct_to_tc(trace.at("outputs").at("output"));

    VoxCPMBackend backend(BackendType::CPU, cpu_thread_count());
    VoxCPMContext weight_ctx(ContextType::Weights, 512);
    VoxCPMContext graph_ctx(ContextType::Graph, 65536, 262144);

    AudioVAE vae;
    REQUIRE(vae.load_from_gguf(kModelPath, weight_ctx, graph_ctx, backend));

    std::vector<float> audio_copy = input_audio;
    ggml_tensor* latent = vae.encode(graph_ctx, audio_copy, 44100);
    REQUIRE(latent != nullptr);
    REQUIRE(latent->ne[0] == 216);
    REQUIRE(latent->ne[1] == 64);
    REQUIRE(latent->ne[2] == 1);

    const GraphTimingStats timing = run_graph_with_timing(graph_ctx, backend, latent, [&]() {
        backend.tensor_set(vae.last_input_tensor(),
                           vae.last_preprocessed_audio().data(),
                           0,
                           vae.last_preprocessed_audio().size() * sizeof(float));
    });

    std::vector<float> actual(expected.size(), 0.0f);
    backend.tensor_get(latent, actual.data(), 0, actual.size() * sizeof(float));

    const ErrorStats stats = compute_error_stats(actual, expected);
    print_error_stats("AudioVAE encode trace output", input_audio, expected, actual, stats, kTargetMaxDiff);
    print_graph_timing("AudioVAE encode", timing);
    INFO("encode max_abs_diff = " << stats.max_abs_diff);
    INFO("encode mean_abs_diff = " << stats.mean_abs_diff);
    INFO("encode rmse = " << stats.rmse);
    REQUIRE(stats.max_abs_diff <= kTargetMaxDiff);
}

TEST_CASE("AudioVAE decode matches trace", "[audio_vae][decode][trace]") {
    if (!file_exists(kModelPath) || !file_exists(kTraceDecodePath)) {
        WARN("Decode test dependencies missing, skipping test");
        return;
    }

    const json trace = load_jsonl_line(kTraceDecodePath, 0);
    const std::vector<float> latent_input = flatten_bct_to_tc(trace.at("inputs").at("z"));
    const std::vector<float> expected = flatten_bct_to_tc(trace.at("outputs").at("output"));

    VoxCPMBackend backend(BackendType::CPU, cpu_thread_count());
    VoxCPMContext weight_ctx(ContextType::Weights, 512);
    VoxCPMContext graph_ctx(ContextType::Graph, 65536, 262144);

    AudioVAE vae;
    REQUIRE(vae.load_from_gguf(kModelPath, weight_ctx, graph_ctx, backend));

    ggml_tensor* latent = graph_ctx.new_tensor_3d(GGML_TYPE_F32, 36, 64, 1);
    REQUIRE(latent != nullptr);
    ggml_set_input(latent);
    ggml_tensor* audio = vae.decode(graph_ctx, latent);
    REQUIRE(audio != nullptr);
    REQUIRE(audio->ne[0] == 63504);
    REQUIRE(audio->ne[1] == 1);
    REQUIRE(audio->ne[2] == 1);

    const GraphTimingStats timing = run_graph_with_timing(graph_ctx, backend, audio, [&]() {
        backend.tensor_set(latent, latent_input.data(), 0, latent_input.size() * sizeof(float));
    });

    std::vector<float> actual(expected.size(), 0.0f);
    backend.tensor_get(audio, actual.data(), 0, actual.size() * sizeof(float));

    const auto [min_it, max_it] = std::minmax_element(actual.begin(), actual.end());
    REQUIRE(*min_it >= -1.001f);
    REQUIRE(*max_it <= 1.001f);

    const ErrorStats stats = compute_error_stats(actual, expected);
    print_error_stats("AudioVAE decode trace output", latent_input, expected, actual, stats, kTargetMaxDiff);
    print_graph_timing("AudioVAE decode", timing);
    INFO("decode max_abs_diff = " << stats.max_abs_diff);
    INFO("decode mean_abs_diff = " << stats.mean_abs_diff);
    INFO("decode rmse = " << stats.rmse);
    REQUIRE(stats.max_abs_diff <= kTargetMaxDiff);
}

}  // namespace test
}  // namespace voxcpm
