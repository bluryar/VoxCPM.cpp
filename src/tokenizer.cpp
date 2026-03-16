/**
 * @file tokenizer.cpp
 * @brief GGUF-backed tokenizer for VoxCPM text tokenization
 */

#include "voxcpm/tokenizer.h"

#include "voxcpm/weight-store.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <limits>
#include <memory>
#include <queue>
#include <stdexcept>

namespace voxcpm {

namespace {

constexpr int32_t kTokenTypeNormal = 1;
constexpr int32_t kTokenTypeByte = 6;

std::vector<std::string> split_utf8_chars(const std::string& text) {
    std::vector<std::string> out;
    for (size_t i = 0; i < text.size();) {
        const unsigned char c = static_cast<unsigned char>(text[i]);
        size_t len = 1;
        if ((c & 0x80U) == 0x00U) {
            len = 1;
        } else if ((c & 0xE0U) == 0xC0U) {
            len = 2;
        } else if ((c & 0xF0U) == 0xE0U) {
            len = 3;
        } else if ((c & 0xF8U) == 0xF0U) {
            len = 4;
        }

        if (i + len > text.size()) {
            len = 1;
        }
        out.emplace_back(text.substr(i, len));
        i += len;
    }
    return out;
}

bool decode_one_utf8_codepoint(const std::string& text, uint32_t& codepoint) {
    if (text.empty()) {
        return false;
    }

    const unsigned char c0 = static_cast<unsigned char>(text[0]);
    if ((c0 & 0x80U) == 0x00U) {
        codepoint = c0;
        return text.size() == 1;
    }
    if ((c0 & 0xE0U) == 0xC0U && text.size() == 2) {
        codepoint = ((c0 & 0x1FU) << 6) |
                    (static_cast<unsigned char>(text[1]) & 0x3FU);
        return true;
    }
    if ((c0 & 0xF0U) == 0xE0U && text.size() == 3) {
        codepoint = ((c0 & 0x0FU) << 12) |
                    ((static_cast<unsigned char>(text[1]) & 0x3FU) << 6) |
                    (static_cast<unsigned char>(text[2]) & 0x3FU);
        return true;
    }
    if ((c0 & 0xF8U) == 0xF0U && text.size() == 4) {
        codepoint = ((c0 & 0x07U) << 18) |
                    ((static_cast<unsigned char>(text[1]) & 0x3FU) << 12) |
                    ((static_cast<unsigned char>(text[2]) & 0x3FU) << 6) |
                    (static_cast<unsigned char>(text[3]) & 0x3FU);
        return true;
    }
    return false;
}

bool is_pure_chinese_token(const std::string& token) {
    const std::vector<std::string> chars = split_utf8_chars(token);
    if (chars.size() < 2) {
        return false;
    }

    for (const std::string& ch : chars) {
        uint32_t codepoint = 0;
        if (!decode_one_utf8_codepoint(ch, codepoint)) {
            return false;
        }
        if (codepoint < 0x4E00U || codepoint > 0x9FFFU) {
            return false;
        }
    }
    return true;
}

bool starts_with(const std::string& value, const std::string& prefix) {
    return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
}

bool parse_byte_token(const std::string& token, uint8_t& value) {
    if (token.size() != 6 || !starts_with(token, "<0x") || token.back() != '>') {
        return false;
    }

    auto hex_value = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
        return -1;
    };

    const int hi = hex_value(token[3]);
    const int lo = hex_value(token[4]);
    if (hi < 0 || lo < 0) {
        return false;
    }

    value = static_cast<uint8_t>((hi << 4) | lo);
    return true;
}

size_t utf8_char_len(unsigned char first) {
    if ((first & 0x80U) == 0x00U) return 1;
    if ((first & 0xE0U) == 0xC0U) return 2;
    if ((first & 0xF0U) == 0xE0U) return 3;
    if ((first & 0xF8U) == 0xF0U) return 4;
    return 1;
}

uint64_t merge_key(int32_t left, int32_t right) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(left)) << 32) |
           static_cast<uint32_t>(right);
}

struct Symbol {
    int prev = -1;
    int next = -1;
    int32_t id = -1;
};

struct MergeCandidate {
    int32_t rank = std::numeric_limits<int32_t>::max();
    int pos = -1;
    int32_t left_id = -1;
    int32_t right_id = -1;

    bool operator<(const MergeCandidate& other) const {
        if (rank != other.rank) {
            return rank > other.rank;
        }
        return pos > other.pos;
    }
};

}  // namespace

void VoxCPMTokenizer::clear() {
    loaded_ = false;
    vocab_.clear();
    id_to_token_.clear();
    merge_rules_.clear();
    special_token_ids_.clear();
    multichar_chinese_tokens_.clear();
    tokenizer_model_.clear();
    tokenizer_pre_.clear();
    unk_token_id_ = 0;
    bos_token_id_ = -1;
    eos_token_id_ = -1;
}

bool VoxCPMTokenizer::load_from_gguf(const std::string& gguf_path) {
    clear();

    ggml_context* ggml_ctx = nullptr;
    gguf_init_params params = {
        .no_alloc = true,
        .ctx = &ggml_ctx,
    };

    gguf_context* gguf_ctx = gguf_init_from_file(gguf_path.c_str(), params);
    if (!gguf_ctx) {
        if (ggml_ctx) {
            ggml_free(ggml_ctx);
        }
        return false;
    }
    std::unique_ptr<gguf_context, GGUFContextDeleter> gguf_holder(gguf_ctx);
    std::unique_ptr<ggml_context, GGMLContextDeleter> ggml_holder(ggml_ctx);

    const auto get_string = [&](const char* key, std::string& value) -> bool {
        const int idx = gguf_find_key(gguf_ctx, key);
        if (idx < 0 || gguf_get_kv_type(gguf_ctx, idx) != GGUF_TYPE_STRING) {
            return false;
        }
        const char* data = gguf_get_val_str(gguf_ctx, idx);
        if (!data) {
            return false;
        }
        value = data;
        return true;
    };

    const auto get_u32 = [&](const char* key, uint32_t& value) -> bool {
        const int idx = gguf_find_key(gguf_ctx, key);
        if (idx < 0) {
            return false;
        }
        value = gguf_get_val_u32(gguf_ctx, idx);
        return true;
    };

    const auto get_i32_array = [&](const char* key, std::vector<int>& values) -> bool {
        const int idx = gguf_find_key(gguf_ctx, key);
        if (idx < 0) {
            return false;
        }
        const int32_t* data = static_cast<const int32_t*>(gguf_get_arr_data(gguf_ctx, idx));
        const size_t n = gguf_get_arr_n(gguf_ctx, idx);
        if (!data && n != 0) {
            return false;
        }
        values.assign(data, data + n);
        return true;
    };

    const auto get_string_array = [&](const char* key, std::vector<std::string>& values) -> bool {
        const int idx = gguf_find_key(gguf_ctx, key);
        if (idx < 0 || gguf_get_arr_type(gguf_ctx, idx) != GGUF_TYPE_STRING) {
            return false;
        }
        const size_t n = gguf_get_arr_n(gguf_ctx, idx);
        values.clear();
        values.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            const char* v = gguf_get_arr_str(gguf_ctx, idx, i);
            values.emplace_back(v ? v : "");
        }
        return true;
    };

    std::vector<std::string> tokens;
    std::vector<int> token_types;
    std::vector<std::string> merges;
    uint32_t bos_id = 0;
    uint32_t eos_id = 0;
    uint32_t unk_id = 0;

    const bool ok = get_string("tokenizer.ggml.model", tokenizer_model_) &&
         get_string("tokenizer.ggml.pre", tokenizer_pre_) &&
         get_string_array("tokenizer.ggml.tokens", tokens) &&
         get_i32_array("tokenizer.ggml.token_type", token_types) &&
         get_string_array("tokenizer.ggml.merges", merges) &&
         get_u32("tokenizer.ggml.bos_token_id", bos_id) &&
         get_u32("tokenizer.ggml.eos_token_id", eos_id) &&
         get_u32("tokenizer.ggml.unknown_token_id", unk_id);

    if (!ok) {
        return false;
    }

    if (tokenizer_model_ != "gpt2" || tokens.empty() || merges.empty() || token_types.size() != tokens.size()) {
        return false;
    }

    for (size_t i = 0; i < tokens.size(); ++i) {
        const int32_t id = static_cast<int32_t>(i);
        vocab_[tokens[i]] = id;
        id_to_token_[id] = tokens[i];
        if (token_types[i] != kTokenTypeNormal && token_types[i] != kTokenTypeByte) {
            special_token_ids_.insert(id);
        }
        if (is_pure_chinese_token(tokens[i])) {
            multichar_chinese_tokens_.insert(tokens[i]);
        }
    }

    bos_token_id_ = static_cast<int32_t>(bos_id);
    eos_token_id_ = static_cast<int32_t>(eos_id);
    unk_token_id_ = static_cast<int32_t>(unk_id);

    int32_t rank = 0;
    for (const std::string& merge_text : merges) {
        const size_t split = merge_text.find(' ');
        if (split == std::string::npos) {
            ++rank;
            continue;
        }

        const std::string left = merge_text.substr(0, split);
        const std::string right = merge_text.substr(split + 1);
        const auto left_it = vocab_.find(left);
        const auto right_it = vocab_.find(right);
        const auto merged_it = vocab_.find(left + right);
        if (left_it != vocab_.end() && right_it != vocab_.end() && merged_it != vocab_.end()) {
            merge_rules_[merge_key(left_it->second, right_it->second)] = {rank, merged_it->second};
        }
        ++rank;
    }

    if (merge_rules_.empty()) {
        clear();
        return false;
    }

    loaded_ = true;
    return true;
}

bool VoxCPMTokenizer::load_from_store(const VoxCPMWeightStore& store) {
    std::vector<std::string> tokens;
    std::vector<int> token_types;
    std::vector<std::string> merges;
    uint32_t bos_id = 0;
    uint32_t eos_id = 0;
    uint32_t unk_id = 0;

    clear();
    if (!store.get_string("tokenizer.ggml.model", tokenizer_model_) ||
        !store.get_string("tokenizer.ggml.pre", tokenizer_pre_) ||
        !store.get_string_array("tokenizer.ggml.tokens", tokens) ||
        !store.get_i32_array("tokenizer.ggml.token_type", token_types) ||
        !store.get_string_array("tokenizer.ggml.merges", merges) ||
        !store.get_u32("tokenizer.ggml.bos_token_id", bos_id) ||
        !store.get_u32("tokenizer.ggml.eos_token_id", eos_id) ||
        !store.get_u32("tokenizer.ggml.unknown_token_id", unk_id)) {
        return false;
    }

    if (tokenizer_model_ != "gpt2" || tokens.empty() || merges.empty() || token_types.size() != tokens.size()) {
        return false;
    }

    for (size_t i = 0; i < tokens.size(); ++i) {
        const int32_t id = static_cast<int32_t>(i);
        vocab_[tokens[i]] = id;
        id_to_token_[id] = tokens[i];
        if (token_types[i] != kTokenTypeNormal && token_types[i] != kTokenTypeByte) {
            special_token_ids_.insert(id);
        }
        if (is_pure_chinese_token(tokens[i])) {
            multichar_chinese_tokens_.insert(tokens[i]);
        }
    }

    bos_token_id_ = static_cast<int32_t>(bos_id);
    eos_token_id_ = static_cast<int32_t>(eos_id);
    unk_token_id_ = static_cast<int32_t>(unk_id);

    int32_t rank = 0;
    for (const std::string& merge_text : merges) {
        const size_t split = merge_text.find(' ');
        if (split == std::string::npos) {
            ++rank;
            continue;
        }
        const std::string left = merge_text.substr(0, split);
        const std::string right = merge_text.substr(split + 1);
        const auto left_it = vocab_.find(left);
        const auto right_it = vocab_.find(right);
        const auto merged_it = vocab_.find(left + right);
        if (left_it != vocab_.end() && right_it != vocab_.end() && merged_it != vocab_.end()) {
            merge_rules_[merge_key(left_it->second, right_it->second)] = {rank, merged_it->second};
        }
        ++rank;
    }

    if (merge_rules_.empty()) {
        clear();
        return false;
    }

    loaded_ = true;
    return true;
}

std::string VoxCPMTokenizer::normalize_text(const std::string& text) const {
    if (!loaded_) {
        throw std::runtime_error("Tokenizer not loaded");
    }

    std::string out = normalizer_prefix_;
    out.reserve(text.size() + normalizer_prefix_.size());
    for (char c : text) {
        if (c == ' ') {
            out += normalizer_prefix_;
        } else {
            out.push_back(c);
        }
    }
    return out;
}

std::vector<std::string> VoxCPMTokenizer::bpe_tokenize(const std::string& normalized_text) const {
    if (normalized_text.empty()) {
        return {};
    }

    std::vector<int32_t> seed_ids;
    for (size_t i = 0; i < normalized_text.size();) {
        const size_t len = utf8_char_len(static_cast<unsigned char>(normalized_text[i]));
        const std::string piece = normalized_text.substr(i, len);
        const auto it = vocab_.find(piece);
        if (it != vocab_.end()) {
            seed_ids.push_back(it->second);
        } else {
            for (size_t j = 0; j < len; ++j) {
                char buf[7] = {0};
                std::snprintf(buf, sizeof(buf), "<0x%02X>",
                              static_cast<unsigned int>(static_cast<unsigned char>(piece[j])));
                const auto byte_it = vocab_.find(buf);
                seed_ids.push_back(byte_it != vocab_.end() ? byte_it->second : unk_token_id_);
            }
        }
        i += len;
    }

    if (seed_ids.empty()) {
        return {};
    }

    if (seed_ids.size() == 1) {
        return {id_to_token(seed_ids.front())};
    }

    std::vector<Symbol> symbols(seed_ids.size());
    for (size_t i = 0; i < seed_ids.size(); ++i) {
        symbols[i].id = seed_ids[i];
        symbols[i].prev = static_cast<int>(i) - 1;
        symbols[i].next = static_cast<int>(i) + 1;
    }
    symbols.back().next = -1;

    std::priority_queue<MergeCandidate> queue;
    const auto push_candidate = [&](int left, int right) {
        if (left < 0 || right < 0) {
            return;
        }
        const auto rule_it = merge_rules_.find(merge_key(symbols[left].id, symbols[right].id));
        if (rule_it == merge_rules_.end()) {
            return;
        }
        queue.push(MergeCandidate{rule_it->second.first, left, symbols[left].id, symbols[right].id});
    };

    for (size_t i = 0; i + 1 < symbols.size(); ++i) {
        push_candidate(static_cast<int>(i), static_cast<int>(i + 1));
    }

    while (!queue.empty()) {
        const MergeCandidate candidate = queue.top();
        queue.pop();

        if (candidate.pos < 0 || candidate.pos >= static_cast<int>(symbols.size())) {
            continue;
        }
        Symbol& left = symbols[static_cast<size_t>(candidate.pos)];
        if (left.id != candidate.left_id || left.next < 0) {
            continue;
        }

        Symbol& right = symbols[static_cast<size_t>(left.next)];
        if (right.id != candidate.right_id) {
            continue;
        }

        const auto rule_it = merge_rules_.find(merge_key(candidate.left_id, candidate.right_id));
        if (rule_it == merge_rules_.end()) {
            continue;
        }

        left.id = rule_it->second.second;
        const int next_next = right.next;
        left.next = next_next;
        if (next_next >= 0) {
            symbols[static_cast<size_t>(next_next)].prev = candidate.pos;
        }
        right.id = -1;

        if (left.prev >= 0) {
            push_candidate(left.prev, candidate.pos);
        }
        if (left.next >= 0) {
            push_candidate(candidate.pos, left.next);
        }
    }

    std::vector<std::string> tokens;
    int current = 0;
    while (current >= 0) {
        if (symbols[static_cast<size_t>(current)].id >= 0) {
            tokens.push_back(id_to_token(symbols[static_cast<size_t>(current)].id));
        }
        current = symbols[static_cast<size_t>(current)].next;
    }
    return tokens;
}

std::vector<std::string> VoxCPMTokenizer::tokenize(const std::string& text) const {
    return bpe_tokenize(normalize_text(text));
}

int32_t VoxCPMTokenizer::token_to_id(const std::string& token) const {
    const auto it = vocab_.find(token);
    return it != vocab_.end() ? it->second : unk_token_id_;
}

std::string VoxCPMTokenizer::id_to_token(int32_t id) const {
    const auto it = id_to_token_.find(id);
    return it != id_to_token_.end() ? it->second : "<unk>";
}

std::vector<int32_t> VoxCPMTokenizer::convert_tokens_to_ids(const std::vector<std::string>& tokens) const {
    std::vector<int32_t> ids;
    ids.reserve(tokens.size());
    for (const std::string& token : tokens) {
        ids.push_back(token_to_id(token));
    }
    return ids;
}

std::vector<std::string> VoxCPMTokenizer::convert_ids_to_tokens(const std::vector<int32_t>& ids) const {
    std::vector<std::string> tokens;
    tokens.reserve(ids.size());
    for (int32_t id : ids) {
        tokens.push_back(id_to_token(id));
    }
    return tokens;
}

std::vector<int32_t> VoxCPMTokenizer::encode(const std::string& text, bool add_bos) const {
    std::vector<int32_t> ids = convert_tokens_to_ids(tokenize(text));
    if (add_bos && bos_token_id_ >= 0) {
        ids.insert(ids.begin(), bos_token_id_);
    }
    return ids;
}

std::string VoxCPMTokenizer::decode(const std::vector<int32_t>& ids, bool skip_special_tokens) const {
    std::string decoded;
    std::string pending_bytes;

    const auto flush_bytes = [&decoded, &pending_bytes]() {
        if (!pending_bytes.empty()) {
            decoded += pending_bytes;
            pending_bytes.clear();
        }
    };

    for (int32_t id : ids) {
        if (skip_special_tokens && special_token_ids_.find(id) != special_token_ids_.end()) {
            continue;
        }

        const std::string token = id_to_token(id);
        uint8_t byte = 0;
        if (parse_byte_token(token, byte)) {
            pending_bytes.push_back(static_cast<char>(byte));
            continue;
        }

        flush_bytes();
        decoded += token;
    }
    flush_bytes();

    const std::string marker = "▁";
    size_t pos = 0;
    while ((pos = decoded.find(marker, pos)) != std::string::npos) {
        decoded.replace(pos, marker.size(), " ");
        pos += 1;
    }

    if (!decoded.empty() && decoded.front() == ' ') {
        decoded.erase(decoded.begin());
    }
    return decoded;
}

bool VoxCPMTokenizer::is_multichar_chinese_token(const std::string& token) const {
    return multichar_chinese_tokens_.find(token) != multichar_chinese_tokens_.end();
}

std::vector<std::string> ChineseCharSplitTokenizer::tokenize(const std::string& text) const {
    const std::vector<std::string> base_tokens = base_tokenizer_.tokenize(text);
    std::vector<std::string> processed;
    const std::string marker = "▁";

    for (const std::string& token : base_tokens) {
        std::string clean_token = token;
        size_t marker_pos = std::string::npos;
        while ((marker_pos = clean_token.find(marker)) != std::string::npos) {
            clean_token.erase(marker_pos, marker.size());
        }

        if (base_tokenizer_.is_multichar_chinese_token(clean_token)) {
            const std::vector<std::string> chars = split_utf8_chars(clean_token);
            processed.insert(processed.end(), chars.begin(), chars.end());
        } else {
            processed.push_back(token);
        }
    }

    return processed;
}

std::vector<int32_t> ChineseCharSplitTokenizer::encode(const std::string& text, bool add_bos) const {
    std::vector<int32_t> ids = base_tokenizer_.convert_tokens_to_ids(tokenize(text));
    if (add_bos && base_tokenizer_.bos_token_id() >= 0) {
        ids.insert(ids.begin(), base_tokenizer_.bos_token_id());
    }
    return ids;
}

}  // namespace voxcpm
