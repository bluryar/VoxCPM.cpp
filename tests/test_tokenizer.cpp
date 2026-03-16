/**
 * @file test_tokenizer.cpp
 * @brief Unit tests for VoxCPM tokenizer behavior
 */

#include <catch2/catch_test_macros.hpp>

#include "voxcpm/tokenizer.h"

#include "gguf.h"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace voxcpm {
namespace test {

namespace {

constexpr const char* kModelPath =
    "/home/orangepi/Codes/ggbond/VoxCPM.cpp/models/voxcpm1.5.gguf";

bool file_exists(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    return file.good();
}

VoxCPMTokenizer load_tokenizer() {
    VoxCPMTokenizer tokenizer;
    REQUIRE(tokenizer.load_from_gguf(kModelPath));
    return tokenizer;
}

std::string write_invalid_tokenizer_gguf() {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "voxcpm_tokenizer_missing_merges.gguf";
    if (std::filesystem::exists(path)) {
        std::filesystem::remove(path);
    }

    gguf_context* ctx = gguf_init_empty();
    REQUIRE(ctx != nullptr);

    const char* tokens[] = {"<unk>", "<s>", "▁", "测", "试"};
    const int32_t token_types[] = {2, 3, 1, 1, 1};

    gguf_set_val_str(ctx, "tokenizer.ggml.model", "gpt2");
    gguf_set_val_str(ctx, "tokenizer.ggml.pre", "default");
    gguf_set_arr_str(ctx, "tokenizer.ggml.tokens", tokens, 5);
    gguf_set_arr_data(ctx, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, token_types, 5);
    gguf_set_val_u32(ctx, "tokenizer.ggml.bos_token_id", 1);
    gguf_set_val_u32(ctx, "tokenizer.ggml.eos_token_id", 1);
    gguf_set_val_u32(ctx, "tokenizer.ggml.unknown_token_id", 0);

    REQUIRE(gguf_write_to_file(ctx, path.c_str(), false));
    gguf_free(ctx);
    return path.string();
}

}  // namespace

TEST_CASE("VoxCPMTokenizer matches english tokenization log", "[tokenizer]") {
    if (!file_exists(kModelPath)) {
        WARN("Model file not found, skipping test");
        return;
    }
    const VoxCPMTokenizer tokenizer = load_tokenizer();
    const std::string text = "Hello, this is the first test sentence.";

    const std::vector<std::string> expected_tokens = {
        "▁Hello", ",", "▁this", "▁is", "▁the", "▁first", "▁test", "▁sentence", "."
    };
    const std::vector<int32_t> expected_ids = {
        1, 21045, 59342, 1536, 1410, 1358, 1933, 2076, 16694, 72
    };

    REQUIRE(tokenizer.tokenize(text) == expected_tokens);
    REQUIRE(tokenizer.convert_tokens_to_ids(expected_tokens) ==
            std::vector<int32_t>{21045, 59342, 1536, 1410, 1358, 1933, 2076, 16694, 72});
    REQUIRE(tokenizer.encode(text) == expected_ids);
    REQUIRE(tokenizer.decode(expected_ids) == "<s> Hello, this is the first test sentence.");
}

TEST_CASE("VoxCPMTokenizer matches chinese tokenization log", "[tokenizer]") {
    if (!file_exists(kModelPath)) {
        WARN("Model file not found, skipping test");
        return;
    }
    const VoxCPMTokenizer tokenizer = load_tokenizer();
    const std::string text =
        "可哪怕位于堂堂超一品官职,在十二郡一言九鼎的大柱国口干舌燥了,这少年还是没什么反应测试一下";

    const std::vector<std::string> expected_tokens = {
        "▁可", "哪怕", "位于", "堂", "堂", "超", "一", "品", "官", "职", ",", "在", "十二", "郡",
        "一言", "九", "鼎", "的大", "柱", "国", "口", "干", "舌", "燥", "了", ",", "这", "少年",
        "还是", "没什么", "反应", "测试", "一下"
    };
    const std::vector<int32_t> expected_ids = {
        1, 36247, 18819, 4822, 60376, 60376, 59984, 59382, 59527, 60186, 59863, 59342,
        59403, 4363, 61925, 48711, 59999, 61482, 4874, 60881, 59417, 59718, 59901, 61765,
        61506, 59404, 59342, 59431, 7743, 2698, 16987, 7023, 6289, 3975
    };

    REQUIRE(tokenizer.tokenize(text) == expected_tokens);
    REQUIRE(tokenizer.encode(text) == expected_ids);
}

TEST_CASE("ChineseCharSplitTokenizer splits multi-char chinese vocab tokens", "[tokenizer]") {
    if (!file_exists(kModelPath)) {
        WARN("Model file not found, skipping test");
        return;
    }
    const VoxCPMTokenizer tokenizer = load_tokenizer();
    const ChineseCharSplitTokenizer split_tokenizer(tokenizer);
    const std::string text = "测试一下";

    const std::vector<std::string> expected_tokens = {"▁", "测", "试", "一", "下"};
    const std::vector<int32_t> expected_ids = {1, 59320, 59972, 59865, 59382, 59454};

    REQUIRE(split_tokenizer.tokenize(text) == expected_tokens);
    REQUIRE(split_tokenizer.encode(text) == expected_ids);
}

TEST_CASE("ChineseCharSplitTokenizer matches long chinese split ids from torch log", "[tokenizer]") {
    if (!file_exists(kModelPath)) {
        WARN("Model file not found, skipping test");
        return;
    }
    const VoxCPMTokenizer tokenizer = load_tokenizer();
    const ChineseCharSplitTokenizer split_tokenizer(tokenizer);
    const std::string text =
        "可哪怕位于堂堂超一品官职,在十二郡一言九鼎的大柱国口干舌燥了,这少年还是没什么反应测试一下";

    const std::vector<int32_t> expected_ids = {
        1, 36247, 60129, 60533, 59580, 59433, 60376, 60376, 59984, 59382, 59527, 60186,
        59863, 59342, 59403, 59482, 59475, 61925, 59382, 59971, 59999, 61482, 59350,
        59412, 60881, 59417, 59718, 59901, 61765, 61506, 59404, 59342, 59431, 59721,
        59406, 59526, 59390, 59590, 59747, 59551, 59877, 59552, 59972, 59865, 59382, 59454
    };

    REQUIRE(split_tokenizer.encode(text) == expected_ids);
}

TEST_CASE("VoxCPMTokenizer rejects GGUF files without merges", "[tokenizer]") {
    VoxCPMTokenizer tokenizer;
    const std::string invalid_path = write_invalid_tokenizer_gguf();
    REQUIRE_FALSE(tokenizer.load_from_gguf(invalid_path));
    std::filesystem::remove(invalid_path);
}

}  // namespace test
}  // namespace voxcpm
