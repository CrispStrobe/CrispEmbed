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
