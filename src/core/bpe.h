// src/core/bpe.h — shared GPT-2 byte-level BPE tokenizer.
//
// Replaces the per-model copies of the same byte_encoder + bytes_to_unicode
// + bpe_one + tokenize loop that qwen3_asr.cpp and granite_speech.cpp each
// have. Both models use the OpenAI GPT-2 byte-level BPE family
// (vocab.json + merges.txt loaded into the GGUF as
// `tokenizer.ggml.tokens` + `tokenizer.ggml.merges`), so the encode side
// is identical down to the byte-permutation table and the greedy
// lowest-rank merge loop.
//
// The decode side (id -> text) lives in each model already because the
// merging-space → utf-8 conversion can in principle differ between
// tokenizers; this header only covers encode for now.
//
// Header-only: each consumer compiles its own copy. The byte_encoder
// table and the per-call BPE merge work are tiny enough that the
// indirection cost of a function-pointer interface isn't worth it.

#pragma once

#include <climits>
#include <cstdint>
#include <cstring>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

namespace core_bpe {

// GPT-2 byte → unicode codepoint table. Built lazily on first call.
// Maps each of the 256 raw bytes to a printable unicode codepoint that
// can survive a roundtrip through json/utf-8 layers. Standard
// definition from `bytes_to_unicode()` in OpenAI's GPT-2 tokenizer.
inline const std::vector<int>& byte_encoder() {
    static std::vector<int> bs(256, 0);
    static bool initialized = false;
    if (initialized)
        return bs;
    std::vector<int> printable;
    for (int b = 0x21; b <= 0x7e; b++)
        printable.push_back(b);
    for (int b = 0xa1; b <= 0xac; b++)
        printable.push_back(b);
    for (int b = 0xae; b <= 0xff; b++)
        printable.push_back(b);
    int next_extra = 256;
    for (int b = 0; b < 256; b++) {
        bool is_printable = false;
        for (int p : printable)
            if (p == b) {
                is_printable = true;
                break;
            }
        if (is_printable)
            bs[b] = b;
        else
            bs[b] = next_extra++;
    }
    initialized = true;
    return bs;
}

// Encode a single Unicode codepoint as a UTF-8 byte sequence.
inline void utf8_encode(uint32_t cp, std::string& out) {
    if (cp < 0x80) {
        out.push_back((char)cp);
    } else if (cp < 0x800) {
        out.push_back((char)(0xC0 | (cp >> 6)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back((char)(0xE0 | (cp >> 12)));
        out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    } else {
        out.push_back((char)(0xF0 | (cp >> 18)));
        out.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    }
}

// Apply the byte→unicode encoder to a raw byte buffer. Each input byte
// becomes one Unicode codepoint via the GPT-2 byte_encoder() map, then
// is encoded as UTF-8.
inline std::string bytes_to_unicode(const char* bytes, size_t n) {
    auto& enc = byte_encoder();
    std::string out;
    out.reserve(n);
    for (size_t i = 0; i < n; i++) {
        utf8_encode((uint32_t)enc[(unsigned char)bytes[i]], out);
    }
    return out;
}

// Greedy lowest-rank BPE merge for a single byte-encoded pre-token.
// Appends the resulting vocab IDs to `out`. When merge_rank is empty
// (older converter that didn't write tokenizer.ggml.merges), only
// complete-token vocab lookups work — sub-words fall back to per-byte.
//
// Symbol identity check uses string concatenation with a literal space
// separator ("left right") to match the textual representation in the
// merges table.
inline void bpe_one(const std::unordered_map<std::string, int32_t>& token_to_id,
                    const std::unordered_map<std::string, int32_t>& merge_rank, const std::string& word,
                    std::vector<int32_t>& out) {
    if (word.empty())
        return;

    // Split into UTF-8 codepoint substrings — each codepoint is one symbol.
    std::vector<std::string> symbols;
    {
        size_t i = 0;
        while (i < word.size()) {
            unsigned char c = (unsigned char)word[i];
            size_t len;
            if (c < 0x80)
                len = 1;
            else if ((c & 0xE0) == 0xC0)
                len = 2;
            else if ((c & 0xF0) == 0xE0)
                len = 3;
            else if ((c & 0xF8) == 0xF0)
                len = 4;
            else
                len = 1;
            if (i + len > word.size())
                len = 1;
            symbols.emplace_back(word, i, len);
            i += len;
        }
    }
    if (symbols.empty())
        return;

    if (!merge_rank.empty() && symbols.size() >= 2) {
        // Priority-queue BPE: O(N log N) instead of O(N²).
        // Linked list of symbol nodes + min-heap of (rank, left_node_id) pairs.
        struct Node { std::string text; int prev, next; };
        int n = (int)symbols.size();
        std::vector<Node> nodes(n);
        for (int i = 0; i < n; i++) {
            nodes[i].text = std::move(symbols[i]);
            nodes[i].prev = i - 1;
            nodes[i].next = i < n - 1 ? i + 1 : -1;
        }

        // (rank, left_node_id) — lower rank = higher priority
        using PQEntry = std::pair<int32_t, int>;
        auto cmp = [](const PQEntry& a, const PQEntry& b) { return a.first > b.first; };
        std::priority_queue<PQEntry, std::vector<PQEntry>, decltype(cmp)> pq(cmp);

        // Helper: try to add pair (i, nodes[i].next) to the queue
        auto try_add = [&](int i) {
            int j = nodes[i].next;
            if (j < 0) return;
            std::string pair = nodes[i].text + " " + nodes[j].text;
            auto it = merge_rank.find(pair);
            if (it != merge_rank.end())
                pq.push({it->second, i});
        };

        // Seed queue with all initial adjacent pairs
        for (int i = 0; i < n; i++) try_add(i);

        while (!pq.empty()) {
            auto [rank, left] = pq.top(); pq.pop();
            int right = nodes[left].next;
            if (right < 0) continue; // stale entry

            // Validate: re-check that the merge is still the correct pair at this rank
            std::string pair = nodes[left].text + " " + nodes[right].text;
            auto it = merge_rank.find(pair);
            if (it == merge_rank.end() || it->second != rank) continue; // stale

            // Merge: left absorbs right
            nodes[left].text += nodes[right].text;
            nodes[left].next = nodes[right].next;
            if (nodes[right].next >= 0)
                nodes[nodes[right].next].prev = left;
            nodes[right].next = -1; nodes[right].prev = -1; // mark dead

            // Re-queue new adjacent pairs
            if (nodes[left].prev >= 0) try_add(nodes[left].prev);
            try_add(left);
        }

        // Collect surviving symbols (node 0 is always head — never absorbed)
        symbols.clear();
        for (int i = 0; i >= 0; i = nodes[i].next)
            symbols.push_back(nodes[i].text);
    }

    for (const auto& s : symbols) {
        auto it = token_to_id.find(s);
        if (it != token_to_id.end()) {
            out.push_back(it->second);
        } else {
            // Per-byte fallback: split into individual codepoints.
            size_t i = 0;
            while (i < s.size()) {
                unsigned char c = (unsigned char)s[i];
                size_t len;
                if (c < 0x80)
                    len = 1;
                else if ((c & 0xE0) == 0xC0)
                    len = 2;
                else if ((c & 0xF0) == 0xE0)
                    len = 3;
                else if ((c & 0xF8) == 0xF0)
                    len = 4;
                else
                    len = 1;
                std::string single(s, i, len);
                auto jt = token_to_id.find(single);
                if (jt != token_to_id.end())
                    out.push_back(jt->second);
                i += len;
            }
        }
    }
}

// Whitespace-split pre-tokenizer + BPE merge pass for arbitrary text.
// Pre-tokenization: collect runs of non-whitespace, prepend a leading
// space to all but the first run (matches GPT-2's "treat space as part
// of the token" convention), byte-encode each run, then BPE-merge it.
//
// This is the simple pre-tokenizer good for prompt fragments. Models
// that need full GPT-2 regex pre-tokenization (with letter / number /
// punctuation runs split separately) should call bpe_one directly.
inline std::vector<int32_t> tokenize_simple(const std::unordered_map<std::string, int32_t>& token_to_id,
                                            const std::unordered_map<std::string, int32_t>& merge_rank,
                                            const std::string& text) {
    std::vector<int32_t> result;
    size_t i = 0;
    bool first = true;
    while (i < text.size()) {
        while (i < text.size() && (text[i] == ' ' || text[i] == '\t' || text[i] == '\n'))
            i++;
        if (i >= text.size())
            break;
        size_t j = i;
        while (j < text.size() && text[j] != ' ' && text[j] != '\t' && text[j] != '\n')
            j++;
        std::string word = text.substr(i, j - i);
        if (!first)
            word = std::string(" ") + word;
        first = false;
        std::string encoded = bytes_to_unicode(word.data(), word.size());
        bpe_one(token_to_id, merge_rank, encoded, result);
        i = j;
    }
    return result;
}

} // namespace core_bpe
