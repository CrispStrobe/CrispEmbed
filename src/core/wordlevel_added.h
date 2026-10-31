// wordlevel_added.h — HF `tokenizers` WordLevel model with a large added
// vocabulary and a per-code-point Split pre-tokenizer, as used by Surya OCR 2.5
// (datalab-to/surya-ocr-2.5 tokenizer.json, tokenizers 0.23.x semantics).
//
// Encode, exactly as tokenizers' AddedVocabulary + pipeline:
//   1. leftmost-longest match of the NON-normalized added tokens on raw text;
//   2. leftmost-longest match of the normalized added tokens on each remaining
//      piece (the source has no normalizer, so normalization is the identity);
//   3. each remaining Unicode scalar is one pre-token (Split `[\s\S]`,
//      Isolated) and is looked up in the WordLevel vocabulary, else <unk>.
// Leftmost-longest = earliest start wins, then the longest token there; the
// scan resumes after the match (Aho-Corasick MatchKind::LeftmostLongest).
// Decode (Fuse decoder): concatenation of token texts, skipping special ids.
//
// No environment gates here: this is a pure data structure. Callers gate it.
// Hermetic guard: tests/test_wordlevel_added.cpp; HF parity:
// tests/test_surya25_tokenizer_parity.py.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace core_wordlevel {

class byte_trie {
    std::unordered_map<uint64_t, int32_t> edges_; // (node << 8 | byte) -> child
    std::vector<int32_t> terminal_{ -1 };         // node -> token id or -1

public:
    bool insert(const std::string & key, int32_t id) {
        if (key.empty() || id < 0) return false;
        int32_t node = 0;
        for (unsigned char c : key) {
            const uint64_t edge = (uint64_t(uint32_t(node)) << 8) | c;
            auto found = edges_.find(edge);
            if (found == edges_.end()) {
                found = edges_.emplace(edge, int32_t(terminal_.size())).first;
                terminal_.push_back(-1);
            }
            node = found->second;
        }
        if (terminal_[node] >= 0) return false; // duplicate content
        terminal_[node] = id;
        return true;
    }
    // Longest token starting at `pos`; returns its byte length (0 if none).
    size_t longest(const std::string & text, size_t pos, int32_t & id) const {
        int32_t node = 0;
        size_t best = 0;
        for (size_t i = pos; i < text.size(); ++i) {
            const auto found = edges_.find((uint64_t(uint32_t(node)) << 8) | (unsigned char)text[i]);
            if (found == edges_.end()) break;
            node = found->second;
            if (terminal_[node] >= 0) {
                best = i + 1 - pos;
                id = terminal_[node];
            }
        }
        return best;
    }
    bool empty() const { return terminal_.size() == 1; }
};

// Strict UTF-8 scalar length at `i` (1..4), or 0 when invalid.
inline int utf8_scalar_length(const std::string & s, size_t i) {
    const unsigned char c = (unsigned char)s[i];
    const int n = c < 0x80 ? 1 : c >= 0xc2 && c <= 0xdf ? 2 : c >= 0xe0 && c <= 0xef ? 3 : c >= 0xf0 && c <= 0xf4 ? 4 : 0;
    if (!n || i + n > s.size()) return 0;
    for (int j = 1; j < n; ++j) {
        const unsigned char b = (unsigned char)s[i + j];
        if (b < 0x80 || b > 0xbf) return 0;
        if (j == 1 && ((c == 0xe0 && b < 0xa0) || (c == 0xed && b > 0x9f) || (c == 0xf0 && b < 0x90) ||
                       (c == 0xf4 && b > 0x8f)))
            return 0;
    }
    return n;
}

class tokenizer {
    std::vector<std::string> tokens_;                 // id -> text (full vocabulary)
    std::unordered_map<std::string, int32_t> model_;  // WordLevel model vocabulary
    byte_trie raw_, normalized_;
    std::unordered_set<int32_t> special_;
    int32_t unk_ = -1;

    void pass_model(const std::string & text, size_t begin, size_t end, std::vector<int32_t> & ids) const {
        for (size_t i = begin; i < end;) {
            const int n = utf8_scalar_length(text, i); // validated by encode()
            const auto found = model_.find(text.substr(i, size_t(n)));
            ids.push_back(found == model_.end() ? unk_ : found->second);
            i += size_t(n);
        }
    }
    void pass_added(const byte_trie & trie, const std::string & text, size_t begin, size_t end, bool second,
                    std::vector<int32_t> & ids) const {
        size_t plain = begin;
        for (size_t i = begin; i < end;) {
            int32_t id = -1;
            size_t length = trie.longest(text, i, id);
            if (length && i + length > end) length = 0; // must stay inside this piece
            if (!length) {
                ++i;
                continue;
            }
            if (plain < i) second ? pass_model(text, plain, i, ids) : pass_added(normalized_, text, plain, i, true, ids);
            ids.push_back(id);
            i += length;
            plain = i;
        }
        if (plain < end)
            second ? pass_model(text, plain, end, ids) : pass_added(normalized_, text, plain, end, true, ids);
    }

public:
    // `tokens`: id-indexed texts for the whole vocabulary. `model_ids`: ids of
    // the WordLevel model vocabulary. `raw_added`/`normalized_added`: added ids
    // by their `normalized` flag. Rejects duplicates, empties and bad ids.
    bool load(std::vector<std::string> tokens, const std::vector<int32_t> & model_ids,
              const std::vector<int32_t> & raw_added, const std::vector<int32_t> & normalized_added,
              const std::vector<int32_t> & special, int32_t unk) {
        *this = {};
        const auto valid = [&](int32_t id) { return id >= 0 && size_t(id) < tokens.size() && !tokens[id].empty(); };
        if (tokens.empty() || !valid(unk)) return false;
        for (int32_t id : model_ids)
            if (!valid(id) || !model_.emplace(tokens[id], id).second) return false;
        if (!model_.count(tokens[unk])) return false;
        for (int32_t id : raw_added)
            if (!valid(id) || !raw_.insert(tokens[id], id)) return false;
        for (int32_t id : normalized_added)
            if (!valid(id) || !normalized_.insert(tokens[id], id)) return false;
        for (int32_t id : special)
            if (!valid(id) || !special_.insert(id).second) return false;
        if (!special_.count(unk)) return false;
        tokens_ = std::move(tokens);
        unk_ = unk;
        return true;
    }
    bool loaded() const { return !tokens_.empty(); }
    size_t size() const { return tokens_.size(); }
    const std::string & text(int32_t id) const { return tokens_[size_t(id)]; }
    bool is_special(int32_t id) const { return special_.count(id) != 0; }
    // Invalid UTF-8 rejects (an HF Python str cannot carry it).
    bool encode(const std::string & text, std::vector<int32_t> & ids) const {
        ids.clear();
        if (!loaded()) return false;
        for (size_t i = 0; i < text.size();) {
            const int n = utf8_scalar_length(text, i);
            if (!n) return false;
            i += size_t(n);
        }
        pass_added(raw_, text, 0, text.size(), false, ids);
        return true;
    }
    bool decode(const std::vector<int32_t> & ids, std::string & text, bool skip_special = true) const {
        text.clear();
        for (int32_t id : ids) {
            if (id < 0 || size_t(id) >= tokens_.size()) {
                text.clear();
                return false;
            }
            if (!skip_special || !is_special(id)) text += tokens_[size_t(id)];
        }
        return true;
    }
};

} // namespace core_wordlevel
