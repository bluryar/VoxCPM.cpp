#include <catch2/catch_test_macros.hpp>

#include "voxcpm/backend.h"
#include "voxcpm/quantize.h"
#include "voxcpm/weight-store.h"

#include <chrono>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace voxcpm {
namespace test {

namespace {

struct SampleModelData {
    std::vector<float> token_embd;
    std::vector<float> blk0_ffn_down;
    std::vector<float> blk5_attn_q;
    std::vector<float> residual_blk7_attn_v;
    std::vector<float> locdit_in_proj;
    std::vector<float> audio_vae_regular_conv;
    std::vector<float> audio_vae_fc_logvar;
    std::vector<float> audio_vae_small_conv;
    std::vector<float> audio_vae_depthwise;
    std::vector<float> audio_vae_transpose_conv;
    std::vector<float> audio_vae_bias;
    std::vector<float> audio_vae_alpha;
    std::vector<float> locenc_special_token;
    std::vector<float> blk0_attn_norm;
};

void fill_sequential(std::vector<float>* values, float start) {
    REQUIRE(values != nullptr);
    for (size_t i = 0; i < values->size(); ++i) {
        (*values)[i] = start + static_cast<float>(i) * 0.01f;
    }
}

void write_sample_imatrix(const std::filesystem::path& path,
                          const std::vector<std::pair<std::string, int64_t>>& entries,
                          float count_value = 4.0f) {
    REQUIRE_FALSE(entries.empty());

    ggml_init_params params = {
        .mem_size = 1 << 20,
        .mem_buffer = nullptr,
        .no_alloc = false,
    };
    ggml_context* tensor_ctx = ggml_init(params);
    REQUIRE(tensor_ctx != nullptr);

    gguf_context* gguf_ctx = gguf_init_empty();
    REQUIRE(gguf_ctx != nullptr);

    const char* datasets[] = { "quantize-test.txt" };
    gguf_set_arr_str(gguf_ctx, "imatrix.datasets", datasets, 1);
    gguf_set_val_u32(gguf_ctx, "imatrix.chunk_count", 3);
    gguf_set_val_u32(gguf_ctx, "imatrix.chunk_size", 16);

    for (size_t i = 0; i < entries.size(); ++i) {
        const std::string& tensor_name = entries[i].first;
        const int64_t n_per_row = entries[i].second;

        std::vector<float> sums(static_cast<size_t>(n_per_row));
        fill_sequential(&sums, 1.0f + static_cast<float>(i));
        std::vector<float> counts(1, count_value);

        ggml_tensor* sums_tensor = ggml_new_tensor_2d(tensor_ctx, GGML_TYPE_F32, n_per_row, 1);
        ggml_set_name(sums_tensor, (tensor_name + ".in_sum2").c_str());
        std::memcpy(sums_tensor->data, sums.data(), sums.size() * sizeof(float));

        ggml_tensor* counts_tensor = ggml_new_tensor_1d(tensor_ctx, GGML_TYPE_F32, 1);
        ggml_set_name(counts_tensor, (tensor_name + ".counts").c_str());
        std::memcpy(counts_tensor->data, counts.data(), counts.size() * sizeof(float));

        gguf_add_tensor(gguf_ctx, sums_tensor);
        gguf_add_tensor(gguf_ctx, counts_tensor);
    }

    REQUIRE(gguf_write_to_file(gguf_ctx, path.c_str(), false));

    gguf_free(gguf_ctx);
    ggml_free(tensor_ctx);
}

std::filesystem::path make_temp_path(const char* stem) {
    const auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() /
           (std::string(stem) + "_" + std::to_string(static_cast<long long>(ticks)) + ".gguf");
}

void write_sample_quantize_model(const std::filesystem::path& path, SampleModelData* sample) {
    REQUIRE(sample != nullptr);

    sample->token_embd.resize(256 * 2);
    sample->blk0_ffn_down.resize(256 * 4);
    sample->blk5_attn_q.resize(256 * 4);
    sample->residual_blk7_attn_v.resize(256 * 2);
    sample->locdit_in_proj.resize(256 * 4);
    sample->audio_vae_regular_conv.resize(1 * 256 * 2);
    sample->audio_vae_fc_logvar.resize(3 * 2048 * 2);
    sample->audio_vae_small_conv.resize(7 * 1 * 2);
    sample->audio_vae_depthwise.resize(7 * 1 * 2);
    sample->audio_vae_transpose_conv.resize(4 * 2 * 4);
    sample->audio_vae_bias.resize(2);
    sample->audio_vae_alpha.resize(2);
    sample->locenc_special_token.resize(256);
    sample->blk0_attn_norm.resize(256);

    fill_sequential(&sample->token_embd, 0.1f);
    fill_sequential(&sample->blk0_ffn_down, 1.0f);
    fill_sequential(&sample->blk5_attn_q, 2.0f);
    fill_sequential(&sample->residual_blk7_attn_v, 3.0f);
    fill_sequential(&sample->locdit_in_proj, 4.0f);
    fill_sequential(&sample->audio_vae_regular_conv, 5.0f);
    fill_sequential(&sample->audio_vae_fc_logvar, 5.5f);
    fill_sequential(&sample->audio_vae_small_conv, 6.0f);
    fill_sequential(&sample->audio_vae_depthwise, 7.0f);
    fill_sequential(&sample->audio_vae_transpose_conv, 8.0f);
    fill_sequential(&sample->audio_vae_bias, 9.0f);
    fill_sequential(&sample->audio_vae_alpha, 10.0f);
    fill_sequential(&sample->locenc_special_token, 6.0f);
    fill_sequential(&sample->blk0_attn_norm, 7.0f);

    ggml_init_params params = {
        .mem_size = 1 << 20,
        .mem_buffer = nullptr,
        .no_alloc = false,
    };
    ggml_context* tensor_ctx = ggml_init(params);
    REQUIRE(tensor_ctx != nullptr);

    ggml_tensor* token_embd = ggml_new_tensor_2d(tensor_ctx, GGML_TYPE_F32, 256, 2);
    ggml_set_name(token_embd, "token_embd.weight");
    std::memcpy(token_embd->data, sample->token_embd.data(), sample->token_embd.size() * sizeof(float));

    ggml_tensor* blk0_ffn_down = ggml_new_tensor_2d(tensor_ctx, GGML_TYPE_F32, 256, 4);
    ggml_set_name(blk0_ffn_down, "blk.0.ffn_down.weight");
    std::memcpy(blk0_ffn_down->data, sample->blk0_ffn_down.data(), sample->blk0_ffn_down.size() * sizeof(float));

    ggml_tensor* blk5_attn_q = ggml_new_tensor_2d(tensor_ctx, GGML_TYPE_F32, 256, 4);
    ggml_set_name(blk5_attn_q, "blk.5.attn_q.weight");
    std::memcpy(blk5_attn_q->data, sample->blk5_attn_q.data(), sample->blk5_attn_q.size() * sizeof(float));

    ggml_tensor* residual_blk7_attn_v = ggml_new_tensor_2d(tensor_ctx, GGML_TYPE_F32, 256, 2);
    ggml_set_name(residual_blk7_attn_v, "residual_lm.blk.7.attn_v.weight");
    std::memcpy(
        residual_blk7_attn_v->data,
        sample->residual_blk7_attn_v.data(),
        sample->residual_blk7_attn_v.size() * sizeof(float));

    ggml_tensor* locdit_in_proj = ggml_new_tensor_2d(tensor_ctx, GGML_TYPE_F32, 256, 4);
    ggml_set_name(locdit_in_proj, "locdit.in_proj.weight");
    std::memcpy(locdit_in_proj->data, sample->locdit_in_proj.data(), sample->locdit_in_proj.size() * sizeof(float));

    ggml_tensor* audio_vae_regular_conv = ggml_new_tensor_3d(tensor_ctx, GGML_TYPE_F32, 1, 256, 2);
    ggml_set_name(audio_vae_regular_conv, "audio_vae.encoder.block.1.block.0.block.3.weight");
    std::memcpy(
        audio_vae_regular_conv->data,
        sample->audio_vae_regular_conv.data(),
        sample->audio_vae_regular_conv.size() * sizeof(float));

    ggml_tensor* audio_vae_fc_logvar = ggml_new_tensor_3d(tensor_ctx, GGML_TYPE_F32, 3, 2048, 2);
    ggml_set_name(audio_vae_fc_logvar, "audio_vae.encoder.fc_logvar.weight");
    std::memcpy(
        audio_vae_fc_logvar->data,
        sample->audio_vae_fc_logvar.data(),
        sample->audio_vae_fc_logvar.size() * sizeof(float));

    ggml_tensor* audio_vae_small_conv = ggml_new_tensor_3d(tensor_ctx, GGML_TYPE_F32, 7, 1, 2);
    ggml_set_name(audio_vae_small_conv, "audio_vae.encoder.block.0.weight");
    std::memcpy(
        audio_vae_small_conv->data,
        sample->audio_vae_small_conv.data(),
        sample->audio_vae_small_conv.size() * sizeof(float));

    ggml_tensor* audio_vae_depthwise = ggml_new_tensor_3d(tensor_ctx, GGML_TYPE_F32, 7, 1, 2);
    ggml_set_name(audio_vae_depthwise, "audio_vae.decoder.model.0.weight");
    std::memcpy(
        audio_vae_depthwise->data,
        sample->audio_vae_depthwise.data(),
        sample->audio_vae_depthwise.size() * sizeof(float));

    ggml_tensor* audio_vae_transpose_conv = ggml_new_tensor_3d(tensor_ctx, GGML_TYPE_F32, 4, 2, 4);
    ggml_set_name(audio_vae_transpose_conv, "audio_vae.decoder.model.2.block.1.weight");
    std::memcpy(
        audio_vae_transpose_conv->data,
        sample->audio_vae_transpose_conv.data(),
        sample->audio_vae_transpose_conv.size() * sizeof(float));

    ggml_tensor* audio_vae_bias = ggml_new_tensor_1d(tensor_ctx, GGML_TYPE_F32, 2);
    ggml_set_name(audio_vae_bias, "audio_vae.encoder.block.0.bias");
    std::memcpy(audio_vae_bias->data, sample->audio_vae_bias.data(), sample->audio_vae_bias.size() * sizeof(float));

    ggml_tensor* audio_vae_alpha = ggml_new_tensor_1d(tensor_ctx, GGML_TYPE_F32, 2);
    ggml_set_name(audio_vae_alpha, "audio_vae.decoder.model.2.block.0.alpha");
    std::memcpy(audio_vae_alpha->data, sample->audio_vae_alpha.data(), sample->audio_vae_alpha.size() * sizeof(float));

    ggml_tensor* locenc_special_token = ggml_new_tensor_1d(tensor_ctx, GGML_TYPE_F32, 256);
    ggml_set_name(locenc_special_token, "locenc.special_token");
    std::memcpy(
        locenc_special_token->data,
        sample->locenc_special_token.data(),
        sample->locenc_special_token.size() * sizeof(float));

    ggml_tensor* blk0_attn_norm = ggml_new_tensor_1d(tensor_ctx, GGML_TYPE_F32, 256);
    ggml_set_name(blk0_attn_norm, "blk.0.attn_norm.weight");
    std::memcpy(
        blk0_attn_norm->data,
        sample->blk0_attn_norm.data(),
        sample->blk0_attn_norm.size() * sizeof(float));

    gguf_context* gguf_ctx = gguf_init_empty();
    REQUIRE(gguf_ctx != nullptr);
    gguf_set_val_str(gguf_ctx, "general.name", "quantize-test");
    gguf_set_val_str(gguf_ctx, "general.architecture", "llama");
    gguf_set_val_u32(gguf_ctx, "general.file_type", GGML_FTYPE_ALL_F32);

    gguf_add_tensor(gguf_ctx, token_embd);
    gguf_add_tensor(gguf_ctx, blk0_ffn_down);
    gguf_add_tensor(gguf_ctx, blk5_attn_q);
    gguf_add_tensor(gguf_ctx, residual_blk7_attn_v);
    gguf_add_tensor(gguf_ctx, locdit_in_proj);
    gguf_add_tensor(gguf_ctx, audio_vae_regular_conv);
    gguf_add_tensor(gguf_ctx, audio_vae_fc_logvar);
    gguf_add_tensor(gguf_ctx, audio_vae_small_conv);
    gguf_add_tensor(gguf_ctx, audio_vae_depthwise);
    gguf_add_tensor(gguf_ctx, audio_vae_transpose_conv);
    gguf_add_tensor(gguf_ctx, audio_vae_bias);
    gguf_add_tensor(gguf_ctx, audio_vae_alpha);
    gguf_add_tensor(gguf_ctx, locenc_special_token);
    gguf_add_tensor(gguf_ctx, blk0_attn_norm);

    REQUIRE(gguf_write_to_file(gguf_ctx, path.c_str(), false));

    gguf_free(gguf_ctx);
    ggml_free(tensor_ctx);
}

std::vector<uint8_t> tensor_bytes(const ggml_tensor* tensor) {
    REQUIRE(tensor != nullptr);
    const uint8_t* src = static_cast<const uint8_t*>(tensor->data);
    return std::vector<uint8_t>(src, src + ggml_nbytes(tensor));
}

}  // namespace

TEST_CASE("quantize_gguf applies VoxCPM tensor mapping policy", "[quantize]") {
    const std::filesystem::path input_path = make_temp_path("voxcpm_quantize_input");
    const std::filesystem::path output_path = make_temp_path("voxcpm_quantize_output");

    SampleModelData sample;
    write_sample_quantize_model(input_path, &sample);

    QuantizeOptions options;
    options.input_path = input_path.string();
    options.output_path = output_path.string();
    options.file_type = GGML_FTYPE_MOSTLY_Q4_K;
    options.n_threads = 2;

    QuantizeStats stats;
    REQUIRE(quantize_gguf(options, &stats));

    REQUIRE(stats.total_tensors == 14);
    REQUIRE(stats.quantized_tensors == 10);
    REQUIRE(stats.audio_vae_tensors == 7);
    REQUIRE(stats.audio_vae_quantized_tensors == 1);
    REQUIRE(stats.audio_vae_f16_tensors == 4);
    REQUIRE(stats.audio_vae_preserved_tensors == 2);

    VoxCPMBackend backend(BackendType::CPU, 2);
    VoxCPMWeightStore store;
    REQUIRE(store.load_from_file(output_path.string(), backend));

    uint32_t file_type = 0;
    REQUIRE(store.get_u32("general.file_type", file_type));
    REQUIRE(file_type == static_cast<uint32_t>(GGML_FTYPE_MOSTLY_Q4_K));

    uint32_t qnt_version = 0;
    REQUIRE(store.get_u32("general.quantization_version", qnt_version));
    REQUIRE(qnt_version == GGML_QNT_VERSION);

    REQUIRE(store.get_tensor("token_embd.weight")->type == GGML_TYPE_Q8_0);
    REQUIRE(store.get_tensor("blk.0.ffn_down.weight")->type == GGML_TYPE_Q5_K);
    REQUIRE(store.get_tensor("blk.5.attn_q.weight")->type == GGML_TYPE_Q4_K);
    REQUIRE(store.get_tensor("residual_lm.blk.7.attn_v.weight")->type == GGML_TYPE_Q5_K);
    REQUIRE(store.get_tensor("locdit.in_proj.weight")->type == GGML_TYPE_Q8_0);
    const ggml_tensor* audio_regular = store.get_tensor("audio_vae.encoder.block.1.block.0.block.3.weight");
    REQUIRE(audio_regular != nullptr);
    REQUIRE(audio_regular->type == GGML_TYPE_Q4_K);
    REQUIRE(ggml_n_dims(audio_regular) == 2);
    REQUIRE(audio_regular->ne[0] == 256);
    REQUIRE(audio_regular->ne[1] == 2);

    REQUIRE(store.get_tensor("audio_vae.encoder.fc_logvar.weight")->type == GGML_TYPE_F16);
    REQUIRE(store.get_tensor("audio_vae.encoder.block.0.weight")->type == GGML_TYPE_F16);
    REQUIRE(store.get_tensor("audio_vae.decoder.model.0.weight")->type == GGML_TYPE_F16);
    REQUIRE(store.get_tensor("audio_vae.decoder.model.2.block.1.weight")->type == GGML_TYPE_F16);
    REQUIRE(store.get_tensor("audio_vae.encoder.block.0.bias")->type == GGML_TYPE_F32);
    REQUIRE(store.get_tensor("audio_vae.decoder.model.2.block.0.alpha")->type == GGML_TYPE_F32);
    REQUIRE(store.get_tensor("locenc.special_token")->type == GGML_TYPE_F32);
    REQUIRE(store.get_tensor("blk.0.attn_norm.weight")->type == GGML_TYPE_F32);

    const std::vector<uint8_t> audio_small_conv_bytes = tensor_bytes(store.get_tensor("audio_vae.encoder.block.0.weight"));
    REQUIRE(audio_small_conv_bytes.size() == sample.audio_vae_small_conv.size() * sizeof(ggml_fp16_t));

    const std::vector<uint8_t> special_token_bytes = tensor_bytes(store.get_tensor("locenc.special_token"));
    REQUIRE(special_token_bytes.size() == sample.locenc_special_token.size() * sizeof(float));
    REQUIRE(std::memcmp(special_token_bytes.data(), sample.locenc_special_token.data(), special_token_bytes.size()) == 0);

    std::filesystem::remove(input_path);
    std::filesystem::remove(output_path);
}

TEST_CASE("quantize_gguf supports forcing AudioVAE weights to F16", "[quantize]") {
    const std::filesystem::path input_path = make_temp_path("voxcpm_quantize_input_audio_vae_f16");
    const std::filesystem::path output_path = make_temp_path("voxcpm_quantize_output_audio_vae_f16");

    SampleModelData sample;
    write_sample_quantize_model(input_path, &sample);

    QuantizeOptions options;
    options.input_path = input_path.string();
    options.output_path = output_path.string();
    options.file_type = GGML_FTYPE_MOSTLY_Q4_K;
    options.audio_vae_mode = AudioVAEQuantizationMode::F16;
    options.n_threads = 2;

    QuantizeStats stats;
    REQUIRE(quantize_gguf(options, &stats));

    REQUIRE(stats.audio_vae_tensors == 7);
    REQUIRE(stats.audio_vae_quantized_tensors == 0);
    REQUIRE(stats.audio_vae_f16_tensors == 5);
    REQUIRE(stats.audio_vae_preserved_tensors == 2);

    VoxCPMBackend backend(BackendType::CPU, 2);
    VoxCPMWeightStore store;
    REQUIRE(store.load_from_file(output_path.string(), backend));

    const ggml_tensor* encoder_regular = store.get_tensor("audio_vae.encoder.block.1.block.0.block.3.weight");
    REQUIRE(encoder_regular != nullptr);
    REQUIRE(encoder_regular->type == GGML_TYPE_F16);

    const ggml_tensor* encoder_fc_logvar = store.get_tensor("audio_vae.encoder.fc_logvar.weight");
    REQUIRE(encoder_fc_logvar != nullptr);
    REQUIRE(encoder_fc_logvar->type == GGML_TYPE_F16);

    const ggml_tensor* decoder_head = store.get_tensor("audio_vae.decoder.model.0.weight");
    REQUIRE(decoder_head != nullptr);
    REQUIRE(decoder_head->type == GGML_TYPE_F16);

    const ggml_tensor* decoder_transpose = store.get_tensor("audio_vae.decoder.model.2.block.1.weight");
    REQUIRE(decoder_transpose != nullptr);
    REQUIRE(decoder_transpose->type == GGML_TYPE_F16);

    const ggml_tensor* encoder_bias = store.get_tensor("audio_vae.encoder.block.0.bias");
    REQUIRE(encoder_bias != nullptr);
    REQUIRE(encoder_bias->type == GGML_TYPE_F32);

    const ggml_tensor* decoder_alpha = store.get_tensor("audio_vae.decoder.model.2.block.0.alpha");
    REQUIRE(decoder_alpha != nullptr);
    REQUIRE(decoder_alpha->type == GGML_TYPE_F32);

    std::filesystem::remove(input_path);
    std::filesystem::remove(output_path);
}

TEST_CASE("quantize_gguf supports lower-bit Q3_K and Q2_K presets", "[quantize]") {
    const std::filesystem::path input_path = make_temp_path("voxcpm_quantize_input_lowbit");

    SampleModelData sample;
    write_sample_quantize_model(input_path, &sample);

    const auto run_case = [&](ggml_ftype file_type,
                              ggml_type expected_blk_ffn_down,
                              ggml_type expected_blk_attn_q,
                              ggml_type expected_attn_v) {
        const std::filesystem::path output_path = make_temp_path("voxcpm_quantize_output_lowbit");

        QuantizeOptions options;
        options.input_path = input_path.string();
        options.output_path = output_path.string();
        options.file_type = file_type;
        options.n_threads = 2;

        REQUIRE(quantize_gguf(options, nullptr));

        VoxCPMBackend backend(BackendType::CPU, 2);
        VoxCPMWeightStore store;
        REQUIRE(store.load_from_file(output_path.string(), backend));

        REQUIRE(store.get_tensor("token_embd.weight")->type == GGML_TYPE_Q8_0);
        REQUIRE(store.get_tensor("locdit.in_proj.weight")->type == GGML_TYPE_Q8_0);
        REQUIRE(store.get_tensor("blk.0.ffn_down.weight")->type == expected_blk_ffn_down);
        REQUIRE(store.get_tensor("blk.5.attn_q.weight")->type == expected_blk_attn_q);
        REQUIRE(store.get_tensor("residual_lm.blk.7.attn_v.weight")->type == expected_attn_v);
        REQUIRE(store.get_tensor("audio_vae.encoder.block.1.block.0.block.3.weight")->type == expected_blk_attn_q);
        REQUIRE(store.get_tensor("audio_vae.encoder.block.0.weight")->type == GGML_TYPE_F16);

        std::filesystem::remove(output_path);
    };

    run_case(GGML_FTYPE_MOSTLY_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K);
    run_case(GGML_FTYPE_MOSTLY_Q2_K, GGML_TYPE_Q4_K, GGML_TYPE_Q2_K, GGML_TYPE_Q4_K);

    std::filesystem::remove(input_path);
}

TEST_CASE("quantize_gguf supports IQ presets and imatrix-gated types", "[quantize]") {
    const std::filesystem::path input_path = make_temp_path("voxcpm_quantize_input_iq");
    const std::filesystem::path imatrix_path = make_temp_path("voxcpm_quantize_imatrix");

    SampleModelData sample;
    write_sample_quantize_model(input_path, &sample);
    write_sample_imatrix(
        imatrix_path,
        {
            {"blk.5.attn_q.weight", 256},
            {"audio_vae.encoder.block.1.block.0.block.3.weight", 256},
        });

    const auto run_case = [&](ggml_ftype file_type,
                              const std::string& maybe_imatrix,
                              ggml_type expected_blk_ffn_down,
                              ggml_type expected_blk_attn_q,
                              ggml_type expected_attn_v) {
        const std::filesystem::path output_path = make_temp_path("voxcpm_quantize_output_iq");

        QuantizeOptions options;
        options.input_path = input_path.string();
        options.output_path = output_path.string();
        options.imatrix_path = maybe_imatrix;
        options.file_type = file_type;
        options.n_threads = 2;

        REQUIRE(quantize_gguf(options, nullptr));

        VoxCPMBackend backend(BackendType::CPU, 2);
        VoxCPMWeightStore store;
        REQUIRE(store.load_from_file(output_path.string(), backend));

        REQUIRE(store.get_tensor("token_embd.weight")->type == GGML_TYPE_Q8_0);
        REQUIRE(store.get_tensor("locdit.in_proj.weight")->type == GGML_TYPE_Q8_0);
        REQUIRE(store.get_tensor("blk.0.ffn_down.weight")->type == expected_blk_ffn_down);
        REQUIRE(store.get_tensor("blk.5.attn_q.weight")->type == expected_blk_attn_q);
        REQUIRE(store.get_tensor("residual_lm.blk.7.attn_v.weight")->type == expected_attn_v);
        REQUIRE(store.get_tensor("audio_vae.encoder.block.1.block.0.block.3.weight")->type == expected_blk_attn_q);
        REQUIRE(store.get_tensor("audio_vae.encoder.block.0.weight")->type == GGML_TYPE_F16);

        if (!maybe_imatrix.empty()) {
            std::string imatrix_file;
            REQUIRE(store.get_string("quantize.imatrix.file", imatrix_file));
            REQUIRE(imatrix_file == maybe_imatrix);

            uint32_t imatrix_entries = 0;
            REQUIRE(store.get_u32("quantize.imatrix.entries_count", imatrix_entries));
            REQUIRE(imatrix_entries == 2);
        }

        std::filesystem::remove(output_path);
    };

    run_case(GGML_FTYPE_MOSTLY_IQ4_NL, "", GGML_TYPE_Q5_K, GGML_TYPE_IQ4_NL, GGML_TYPE_Q5_K);
    run_case(GGML_FTYPE_MOSTLY_IQ2_XXS, imatrix_path.string(), GGML_TYPE_Q4_K, GGML_TYPE_IQ2_XXS, GGML_TYPE_Q4_K);

    QuantizeOptions missing_imatrix_options;
    missing_imatrix_options.input_path = input_path.string();
    missing_imatrix_options.output_path = make_temp_path("voxcpm_quantize_output_iq_missing").string();
    missing_imatrix_options.file_type = GGML_FTYPE_MOSTLY_IQ2_XXS;
    missing_imatrix_options.n_threads = 2;
    REQUIRE_THROWS(quantize_gguf(missing_imatrix_options, nullptr));

    std::filesystem::remove(input_path);
    std::filesystem::remove(imatrix_path);
    std::filesystem::remove(missing_imatrix_options.output_path);
}

TEST_CASE("quantize_gguf supports dry run and rejects requantizing low-bit inputs", "[quantize]") {
    const std::filesystem::path input_path = make_temp_path("voxcpm_quantize_input");
    const std::filesystem::path output_path = make_temp_path("voxcpm_quantize_output");

    SampleModelData sample;
    write_sample_quantize_model(input_path, &sample);

    QuantizeOptions dry_run_options;
    dry_run_options.input_path = input_path.string();
    dry_run_options.output_path = output_path.string();
    dry_run_options.file_type = GGML_FTYPE_MOSTLY_Q4_K;
    dry_run_options.n_threads = 1;
    dry_run_options.dry_run = true;

    QuantizeStats dry_run_stats;
    REQUIRE(quantize_gguf(dry_run_options, &dry_run_stats));
    REQUIRE_FALSE(std::filesystem::exists(output_path));

    QuantizeOptions quantize_options = dry_run_options;
    quantize_options.dry_run = false;
    REQUIRE(quantize_gguf(quantize_options, nullptr));

    QuantizeOptions requantize_options;
    requantize_options.input_path = output_path.string();
    requantize_options.output_path = make_temp_path("voxcpm_quantize_requantized").string();
    requantize_options.file_type = GGML_FTYPE_MOSTLY_Q5_K;
    requantize_options.n_threads = 1;

    REQUIRE_THROWS(quantize_gguf(requantize_options, nullptr));

    std::filesystem::remove(input_path);
    std::filesystem::remove(output_path);
    std::filesystem::remove(requantize_options.output_path);
}

}  // namespace test
}  // namespace voxcpm
