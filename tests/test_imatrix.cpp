#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "voxcpm/backend.h"
#include "voxcpm/imatrix.h"
#include "ggml-cpu.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace voxcpm {
namespace test {

namespace {

std::filesystem::path make_temp_path(const char* stem) {
    const auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() /
           (std::string(stem) + "_" + std::to_string(static_cast<long long>(ticks)) + ".gguf");
}

std::filesystem::path make_temp_text_path(const char* stem, const char* ext) {
    const auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() /
           (std::string(stem) + "_" + std::to_string(static_cast<long long>(ticks)) + ext);
}

}  // namespace

TEST_CASE("imatrix collector observes mul_mat graphs and saves gguf", "[imatrix]") {
    VoxCPMBackend backend(BackendType::CPU, 2);
    VoxCPMImatrixCollector collector;
    collector.set_chunk_size(1);
    collector.add_dataset("unit-test.txt");

    ggml_init_params params = {
        .mem_size = 1 << 20,
        .mem_buffer = nullptr,
        .no_alloc = false,
    };
    ggml_context* ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor* w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 3);
    ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2);
    ggml_set_name(w, "blk.0.attn_q.weight");
    ggml_set_name(x, "input");

    float* w_data = static_cast<float*>(w->data);
    float* x_data = static_cast<float*>(x->data);
    for (int i = 0; i < 12; ++i) {
        w_data[i] = 0.1f * static_cast<float>(i + 1);
    }
    for (int i = 0; i < 8; ++i) {
        x_data[i] = static_cast<float>(i + 1);
    }

    ggml_tensor* y = ggml_mul_mat(ctx, w, x);
    ggml_set_output(y);

    ggml_cgraph* graph = ggml_new_graph_custom(ctx, 16, false);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, y);
    REQUIRE(ggml_graph_compute_with_ctx(ctx, graph, 2) == GGML_STATUS_SUCCESS);
    collector.observe_graph(graph, backend);
    collector.mark_chunk_processed();

    REQUIRE(collector.entry_count() == 1);
    const auto it = collector.stats().find("blk.0.attn_q.weight");
    REQUIRE(it != collector.stats().end());
    REQUIRE(it->second.counts.size() == 1);
    REQUIRE(it->second.counts[0] == 2);
    REQUIRE(it->second.values.size() == 4);
    REQUIRE(it->second.values[0] == Catch::Approx(1.0f * 1.0f + 5.0f * 5.0f));

    const std::filesystem::path output_path = make_temp_path("voxcpm_imatrix_test");
    REQUIRE(collector.save_to_file(output_path.string()));

    ggml_context* load_ctx_raw = nullptr;
    gguf_init_params load_params = {
        .no_alloc = false,
        .ctx = &load_ctx_raw,
    };
    gguf_context* gguf_ctx = gguf_init_from_file(output_path.c_str(), load_params);
    REQUIRE(gguf_ctx != nullptr);
    REQUIRE(load_ctx_raw != nullptr);

    const int chunk_count_idx = gguf_find_key(gguf_ctx, "imatrix.chunk_count");
    REQUIRE(chunk_count_idx >= 0);
    REQUIRE(gguf_get_val_u32(gguf_ctx, chunk_count_idx) == 1);

    REQUIRE(gguf_find_tensor(gguf_ctx, "blk.0.attn_q.weight.in_sum2") >= 0);
    REQUIRE(gguf_find_tensor(gguf_ctx, "blk.0.attn_q.weight.counts") >= 0);

    gguf_free(gguf_ctx);
    ggml_free(load_ctx_raw);

    VoxCPMImatrixCollector loaded;
    REQUIRE(loaded.load_from_file(output_path.string()));
    REQUIRE(loaded.entry_count() == 1);
    REQUIRE(loaded.chunk_size() == 1);
    REQUIRE(loaded.chunks_count() == 1);
    REQUIRE(loaded.datasets().size() == 1);
    REQUIRE(loaded.datasets().front() == "unit-test.txt");
    const auto loaded_it = loaded.stats().find("blk.0.attn_q.weight");
    REQUIRE(loaded_it != loaded.stats().end());
    REQUIRE(loaded_it->second.counts.size() == 1);
    REQUIRE(loaded_it->second.counts[0] == 2);
    REQUIRE(loaded_it->second.values.size() == 4);
    REQUIRE(loaded_it->second.values[0] == Catch::Approx(it->second.values[0]));

    std::ostringstream statistics;
    loaded.print_statistics(statistics, 4);
    REQUIRE(statistics.str().find("entries=1") != std::string::npos);
    REQUIRE(statistics.str().find("blk.0.attn_q.weight") != std::string::npos);

    ggml_free(ctx);
    std::filesystem::remove(output_path);
}

TEST_CASE("imatrix calibration sample loaders support text and dataset formats", "[imatrix]") {
    const std::filesystem::path text_path = make_temp_text_path("voxcpm_imatrix_text", ".txt");
    const std::filesystem::path dataset_path = make_temp_text_path("voxcpm_imatrix_dataset", ".tsv");
    const std::filesystem::path invalid_dataset_path = make_temp_text_path("voxcpm_imatrix_invalid_dataset", ".tsv");
    const std::filesystem::path audio_path = make_temp_text_path("voxcpm_imatrix_prompt", ".wav");

    {
        std::ofstream out(text_path);
        REQUIRE(out.is_open());
        out << "第一条测试文本\n";
        out << "\n";
        out << "第二条测试文本\n";
    }

    {
        std::ofstream out(dataset_path);
        REQUIRE(out.is_open());
        out << "# comment\n";
        out << "纯文本样本\n";
        out << "带参考的样本\t提示文本\t" << audio_path.string() << "\n";
    }

    {
        std::ofstream out(invalid_dataset_path);
        REQUIRE(out.is_open());
        out << "bad\ttoo\tmany\tfields\n";
    }

    {
        std::ofstream out(audio_path, std::ios::binary);
        REQUIRE(out.is_open());
        out << "RIFF";
    }

    const auto text_samples = load_text_calibration_file(text_path.string(), 0);
    REQUIRE(text_samples.size() == 2);
    REQUIRE(text_samples[0].text == "第一条测试文本");
    REQUIRE(text_samples[0].prompt_text.empty());
    REQUIRE(text_samples[0].prompt_audio_path.empty());
    REQUIRE(text_samples[1].text == "第二条测试文本");

    const auto dataset_samples = load_calibration_dataset_file(dataset_path.string(), 0);
    REQUIRE(dataset_samples.size() == 2);
    REQUIRE(dataset_samples[0].text == "纯文本样本");
    REQUIRE(dataset_samples[0].prompt_text.empty());
    REQUIRE(dataset_samples[0].prompt_audio_path.empty());
    REQUIRE(dataset_samples[1].text == "带参考的样本");
    REQUIRE(dataset_samples[1].prompt_text == "提示文本");
    REQUIRE(dataset_samples[1].prompt_audio_path == audio_path.string());

    REQUIRE_THROWS(load_calibration_dataset_file(invalid_dataset_path.string(), 0));

    std::filesystem::remove(text_path);
    std::filesystem::remove(dataset_path);
    std::filesystem::remove(invalid_dataset_path);
    std::filesystem::remove(audio_path);
}

}  // namespace test
}  // namespace voxcpm
