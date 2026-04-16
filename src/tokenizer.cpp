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

embed_tokens WordPieceTokenizer::encode_pair(const std::string & text_a,
                                              const std::string & text_b) const {
    // Tokenize a string to raw subword ids (no special tokens, no padding)
    auto tokenize_raw = [&](const std::string & text) -> std::vector<int32_t> {
        std::vector<std::string> words;
        std::string cur;
        for (size_t i = 0; i < text.size(); i++) {
            unsigned char c = text[i];
            if (std::isspace(c)) {
                if (!cur.empty()) { words.push_back(cur); cur.clear(); }
            } else if (std::ispunct(c)) {
                if (!cur.empty()) { words.push_back(cur); cur.clear(); }
                words.push_back(std::string(1, (char)c));
            } else {
                cur += (char)std::tolower(c);
            }
        }
        if (!cur.empty()) words.push_back(cur);

        std::vector<int32_t> ids;
        for (const auto & w : words)
            for (int id : wordpiece(w)) ids.push_back(id);
        return ids;
    };

    auto ids_a = tokenize_raw(text_a);
    auto ids_b = tokenize_raw(text_b);

    // Truncate longest-first to fit: [CLS] a [SEP] b [SEP] = n_a + n_b + 3 tokens
    int budget = max_length_ - 3;
    while ((int)(ids_a.size() + ids_b.size()) > budget) {
        if (ids_a.size() >= ids_b.size()) ids_a.pop_back();
        else ids_b.pop_back();
    }

    // Build combined sequence with type_ids
    std::vector<int32_t> ids, types;
    ids.push_back(cls_id_); types.push_back(0);
    for (int id : ids_a)  { ids.push_back(id);     types.push_back(0); }
    ids.push_back(sep_id_); types.push_back(0);
    for (int id : ids_b)  { ids.push_back(id);     types.push_back(1); }
    ids.push_back(sep_id_); types.push_back(1);

    embed_tokens result;
    int seq_len = (int)ids.size();
    result.ids.resize(max_length_, pad_id_);
    result.type_ids.resize(max_length_, 0);
    result.attn_mask.resize(max_length_, 0);
    for (int i = 0; i < seq_len; i++) {
        result.ids[i]      = ids[i];
        result.type_ids[i] = types[i];
        result.attn_mask[i] = 1;
    }
    return result;
}
