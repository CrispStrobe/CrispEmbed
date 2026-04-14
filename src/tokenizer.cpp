// tokenizer.cpp — WordPiece tokenizer for BERT-family models.

#include "tokenizer.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <sstream>

bool WordPieceTokenizer::load(const std::vector<std::string> & vocab,
                               int cls_id, int sep_id, int unk_id, int pad_id,
                               int max_length) {
    id_to_token_ = vocab;
    token_to_id_.clear();
    token_to_id_.reserve(vocab.size());
    for (int i = 0; i < (int)vocab.size(); i++) {
        token_to_id_[vocab[i]] = i;
    }
    cls_id_ = cls_id;
    sep_id_ = sep_id;
    unk_id_ = unk_id;
    pad_id_ = pad_id;
    max_length_ = max_length;
    return !vocab.empty();
}

std::vector<int> WordPieceTokenizer::wordpiece(const std::string & word) const {
    std::vector<int> ids;
    int start = 0;
    int len = (int)word.size();

    while (start < len) {
        int end = len;
        bool found = false;
        while (start < end) {
            std::string sub = word.substr(start, end - start);
            if (start > 0) sub = "##" + sub;

            auto it = token_to_id_.find(sub);
            if (it != token_to_id_.end()) {
                ids.push_back(it->second);
                found = true;
                break;
            }
            end--;
        }
        if (!found) {
            ids.push_back(unk_id_);
            break;
        }
        start = end;
    }
    return ids;
}

embed_tokens WordPieceTokenizer::encode(const std::string & text) const {
    // Basic preprocessing: lowercase + split on whitespace/punctuation
    std::vector<std::string> words;
    std::string current;
    for (size_t i = 0; i < text.size(); i++) {
        unsigned char c = text[i];
        if (std::isspace(c)) {
            if (!current.empty()) { words.push_back(current); current.clear(); }
        } else if (std::ispunct(c)) {
            if (!current.empty()) { words.push_back(current); current.clear(); }
            words.push_back(std::string(1, (char)c));
        } else {
            current += (char)std::tolower(c);
        }
    }
    if (!current.empty()) words.push_back(current);

    // Tokenize each word via WordPiece
    std::vector<int32_t> ids;
    ids.push_back(cls_id_);
    for (const auto & w : words) {
        auto wp = wordpiece(w);
        for (int id : wp) {
            if ((int)ids.size() >= max_length_ - 1) break;  // leave room for [SEP]
            ids.push_back(id);
        }
    }
    ids.push_back(sep_id_);

    // Build result with padding
    embed_tokens result;
    int seq_len = (int)ids.size();
    int pad_len = std::min(max_length_, std::max(seq_len, 1));

    result.ids.resize(pad_len, pad_id_);
    result.type_ids.resize(pad_len, 0);
    result.attn_mask.resize(pad_len, 0);

    for (int i = 0; i < seq_len && i < pad_len; i++) {
        result.ids[i] = ids[i];
        result.attn_mask[i] = 1;
    }

    return result;
}

// ---------------------------------------------------------------------------
// SentencePiece tokenizer (for XLM-RoBERTa / multilingual models)
// ---------------------------------------------------------------------------

bool SentencePieceTokenizer::load(const std::vector<std::string> & vocab,
                                   const std::vector<float> & scores,
                                   int bos_id, int eos_id, int unk_id, int pad_id,
                                   int max_length) {
    id_to_token_ = vocab;
    scores_ = scores;
    token_to_id_.clear();
    token_to_id_.reserve(vocab.size());
    for (int i = 0; i < (int)vocab.size(); i++) {
        token_to_id_[vocab[i]] = i;
    }
    bos_id_ = bos_id;
    eos_id_ = eos_id;
    unk_id_ = unk_id;
    pad_id_ = pad_id;
    max_length_ = max_length;
    return !vocab.empty();
}

std::vector<int> SentencePieceTokenizer::tokenize_text(const std::string & text) const {
    // SentencePiece unigram: greedy longest-match from left to right.
    // The ▁ (U+2581) character represents a word boundary (space).
    std::vector<int> ids;
    size_t i = 0;
    while (i < text.size()) {
        int best_id = unk_id_;
        size_t best_len = 1;  // at minimum consume 1 byte

        // Try progressively shorter substrings
        for (size_t end = std::min(text.size(), i + 64); end > i; end--) {
            std::string sub = text.substr(i, end - i);
            auto it = token_to_id_.find(sub);
            if (it != token_to_id_.end()) {
                best_id = it->second;
                best_len = end - i;
                break;
            }
        }

        // Also try with ▁ prefix (SentencePiece word boundary)
        if (i == 0 || (i > 0 && text[i-1] == ' ')) {
            std::string prefix = "\xe2\x96\x81";  // ▁ in UTF-8
            for (size_t end = std::min(text.size(), i + 64); end > i; end--) {
                std::string sub = prefix + text.substr(i, end - i);
                auto it = token_to_id_.find(sub);
                if (it != token_to_id_.end()) {
                    best_id = it->second;
                    best_len = end - i;
                    break;
                }
            }
        }

        ids.push_back(best_id);
        i += best_len;
    }
    return ids;
}

embed_tokens SentencePieceTokenizer::encode(const std::string & text) const {
    // Preprocess: lowercase for some models, add space prefix
    std::string processed = text;

    // Tokenize
    auto token_ids = tokenize_text(processed);

    // Build result: <s> + tokens + </s>
    std::vector<int32_t> ids;
    ids.push_back(bos_id_);
    for (int id : token_ids) {
        if ((int)ids.size() >= max_length_ - 1) break;
        ids.push_back(id);
    }
    ids.push_back(eos_id_);

    // Pad
    embed_tokens result;
    int seq_len = (int)ids.size();
    int pad_len = std::min(max_length_, std::max(seq_len, 1));

    result.ids.resize(pad_len, pad_id_);
    result.type_ids.resize(pad_len, 0);
    result.attn_mask.resize(pad_len, 0);

    for (int i = 0; i < seq_len && i < pad_len; i++) {
        result.ids[i] = ids[i];
        result.attn_mask[i] = 1;
    }

    return result;
}
