// tokenizer.cpp — WordPiece tokenizer for BERT-family models.

#include "tokenizer.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <sstream>

bool WordPieceTokenizer::load(const std::vector<std::string> & vocab,
                               int cls_id, int sep_id, int unk_id, int pad_id,
                               int max_length, bool do_lower_case) {
    do_lower_case_ = do_lower_case;
    // Ollama-format GGUFs store WordPiece vocab with SentencePiece-style
    // "▁" (U+2581, 3 bytes: 0xE2 0x96 0x81) prefix on whole-word tokens
    // and strip the "##" prefix from subword tokens. Undo this so the
    // standard WordPiece lookup works: "▁hello" → "hello", "ing" → "##ing".
    static const std::string SP_PREFIX = "\xe2\x96\x81"; // ▁ (U+2581)
    bool has_sp_prefix = false;
    for (size_t i = 0; i < std::min(vocab.size(), (size_t)1000); i++) {
        if (vocab[i].size() > 3 && vocab[i].compare(0, 3, SP_PREFIX) == 0
            && vocab[i][3] != '[') {
            has_sp_prefix = true;
            break;
        }
    }

    id_to_token_.resize(vocab.size());
    token_to_id_.clear();
    token_to_id_.reserve(vocab.size());
    for (int i = 0; i < (int)vocab.size(); i++) {
        std::string tok = vocab[i];
        if (has_sp_prefix) {
            if (tok.size() > 3 && tok.compare(0, 3, SP_PREFIX) == 0) {
                // "▁hello" → "hello" (whole-word token)
                tok = tok.substr(3);
            } else if (!tok.empty() && tok[0] != '[' && tok[0] != '<') {
                // "ing" → "##ing" (subword continuation)
                tok = "##" + tok;
            }
        }
        id_to_token_[i] = tok;
        token_to_id_[tok] = i;
    }
    cls_id_ = cls_id;
    sep_id_ = sep_id;
    unk_id_ = unk_id;
    pad_id_ = pad_id;
    max_length_ = max_length;
    build_trie();
    return !vocab.empty();
}

void WordPieceTokenizer::build_trie() {
    trie_nodes_.clear();
    // Create two roots: one for first-piece tokens, one for ## continuations
    trie_nodes_.push_back(TrieNode()); trie_root_ = 0;
    trie_nodes_.push_back(TrieNode()); trie_cont_ = 1;

    for (auto &[tok, id] : token_to_id_) {
        bool is_cont = (tok.size() >= 2 && tok[0] == '#' && tok[1] == '#');
        int root = is_cont ? trie_cont_ : trie_root_;
        const char *s = tok.c_str();
        int slen = (int)tok.size();
        if (is_cont) { s += 2; slen -= 2; }  // skip "##" prefix

        int node = root;
        for (int i = 0; i < slen; i++) {
            char c = s[i];
            auto it = trie_nodes_[node].children.find(c);
            if (it == trie_nodes_[node].children.end()) {
                int child = (int)trie_nodes_.size();
                trie_nodes_.push_back(TrieNode());
                trie_nodes_[node].children[c] = child;
                node = child;
            } else {
                node = it->second;
            }
        }
        trie_nodes_[node].token_id = id;
    }
    trie_built_ = true;
}

std::vector<int> WordPieceTokenizer::wordpiece(const std::string & word) const {
    std::vector<int> ids;
    int start = 0;
    int len = (int)word.size();

    if (!trie_built_) {
        // Fallback to original O(n²) if trie not built
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
            if (!found) { ids.push_back(unk_id_); break; }
            start = end;
        }
        return ids;
    }

    // Trie-based O(len) longest-match
    while (start < len) {
        int root = (start == 0) ? trie_root_ : trie_cont_;
        int node = root;
        int best_end = -1;
        int best_id = -1;

        for (int i = start; i < len; i++) {
            auto it = trie_nodes_[node].children.find(word[i]);
            if (it == trie_nodes_[node].children.end()) break;
            node = it->second;
            if (trie_nodes_[node].token_id >= 0) {
                best_end = i + 1;
                best_id = trie_nodes_[node].token_id;
            }
        }

        if (best_id >= 0) {
            ids.push_back(best_id);
            start = best_end;
        } else {
            ids.push_back(unk_id_);
            break;
        }
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
            current += do_lower_case_ ? (char)std::tolower(c) : (char)c;
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
                cur += do_lower_case_ ? (char)std::tolower(c) : (char)c;
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
