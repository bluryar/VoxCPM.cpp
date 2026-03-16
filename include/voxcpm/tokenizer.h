/**
 * @file tokenizer.h
 * @brief GGUF-backed tokenizer for VoxCPM text tokenization
 */

#ifndef VOXCPM_TOKENIZER_H
#define VOXCPM_TOKENIZER_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace voxcpm {

class VoxCPMWeightStore;

class VoxCPMTokenizer {
public:
    VoxCPMTokenizer() = default;

    bool load_from_gguf(const std::string& gguf_path);
    bool load_from_store(const VoxCPMWeightStore& store);

    std::vector<std::string> tokenize(const std::string& text) const;
    std::vector<int32_t> encode(const std::string& text, bool add_bos = true) const;
    std::string decode(const std::vector<int32_t>& ids, bool skip_special_tokens = false) const;

    int32_t token_to_id(const std::string& token) const;
    std::string id_to_token(int32_t id) const;
    std::vector<int32_t> convert_tokens_to_ids(const std::vector<std::string>& tokens) const;
    std::vector<std::string> convert_ids_to_tokens(const std::vector<int32_t>& ids) const;

    bool is_loaded() const { return loaded_; }
    int32_t bos_token_id() const { return bos_token_id_; }
    int32_t eos_token_id() const { return eos_token_id_; }
    int32_t unk_token_id() const { return unk_token_id_; }
    bool is_multichar_chinese_token(const std::string& token) const;

private:
    std::vector<std::string> bpe_tokenize(const std::string& normalized_text) const;
    std::string normalize_text(const std::string& text) const;
    void clear();

    bool loaded_ = false;
    std::unordered_map<std::string, int32_t> vocab_;
    std::unordered_map<int32_t, std::string> id_to_token_;
    std::unordered_map<uint64_t, std::pair<int32_t, int32_t>> merge_rules_;
    std::unordered_set<int32_t> special_token_ids_;
    std::unordered_set<std::string> multichar_chinese_tokens_;

    std::string tokenizer_model_;
    std::string tokenizer_pre_;
    std::string normalizer_prefix_ = "▁";

    int32_t unk_token_id_ = 0;
    int32_t bos_token_id_ = -1;
    int32_t eos_token_id_ = -1;
};

class ChineseCharSplitTokenizer {
public:
    explicit ChineseCharSplitTokenizer(const VoxCPMTokenizer& base_tokenizer)
        : base_tokenizer_(base_tokenizer) {}

    std::vector<std::string> tokenize(const std::string& text) const;
    std::vector<int32_t> encode(const std::string& text, bool add_bos = true) const;

private:
    const VoxCPMTokenizer& base_tokenizer_;
};

}  // namespace voxcpm

#endif  // VOXCPM_TOKENIZER_H
