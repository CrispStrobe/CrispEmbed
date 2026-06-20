// tokenizer_bpe.cpp — BPE tokenizer for decoder models.
//
// Two modes:
// - GPT-2 byte-level BPE (Qwen3): uses core_bpe from CrispASR
// - SentencePiece BPE (Gemma): ▁ space marker, standard BPE merges

#include "tokenizer.h"
#include "core/bpe.h"

#include <algorithm>
#include <climits>
#include <cstring>
#include <queue>
#include <string>
#include <vector>

bool BPETokenizer::load(const std::vector<std::string> & vocab,
                         const std::vector<std::string> & merges,
                         int eos_id, int pad_id, int suffix_id,
                         int bos_id, bool spm_style,
                         int max_length) {
    id_to_token_ = vocab;
    token_to_id_.clear();
    token_to_id_.reserve(vocab.size());
    for (int i = 0; i < (int)vocab.size(); i++) {
        token_to_id_[vocab[i]] = i;
    }

    merge_rank_.clear();
    merge_rank_.reserve(merges.size());
    for (int i = 0; i < (int)merges.size(); i++) {
        merge_rank_[merges[i]] = i;
    }

    eos_id_ = eos_id;
    pad_id_ = pad_id;
    suffix_id_ = suffix_id;
    bos_id_ = bos_id;
    spm_style_ = spm_style;
    max_length_ = max_length;
    return !vocab.empty();
}

// SentencePiece-style BPE: split into initial tokens, then merge by rank.
std::vector<int32_t> BPETokenizer::bpe_merge(const std::string & text) const {
    if (text.empty()) return {};

    // Split into individual UTF-8 characters as initial symbols
    std::vector<std::string> symbols;
    size_t i = 0;
    while (i < text.size()) {
        size_t len = 1;
        unsigned char c = (unsigned char)text[i];
        if (c >= 0xC0) {
            if (c < 0xE0) len = 2;
            else if (c < 0xF0) len = 3;
            else len = 4;
        }
        len = std::min(len, text.size() - i);
        symbols.push_back(text.substr(i, len));
        i += len;
    }

    // Priority-queue BPE: O(N log N) instead of O(N²)
    if (symbols.size() >= 2) {
        struct Node { std::string text; int prev, next; };
        int n = (int)symbols.size();
        std::vector<Node> nodes(n);
        for (int i = 0; i < n; i++) {
            nodes[i].text = std::move(symbols[i]);
            nodes[i].prev = i - 1;
            nodes[i].next = i < n - 1 ? i + 1 : -1;
        }
        using PQE = std::pair<int, int>;
        auto cmp = [](const PQE& a, const PQE& b) { return a.first > b.first; };
        std::priority_queue<PQE, std::vector<PQE>, decltype(cmp)> pq(cmp);
        auto try_add = [&](int i) {
            int j = nodes[i].next;
            if (j < 0) return;
            std::string pair = nodes[i].text + " " + nodes[j].text;
            auto it = merge_rank_.find(pair);
            if (it != merge_rank_.end()) pq.push({it->second, i});
        };
        for (int i = 0; i < n; i++) try_add(i);
        while (!pq.empty()) {
            auto [rank, left] = pq.top(); pq.pop();
            int right = nodes[left].next;
            if (right < 0) continue;
            std::string pair = nodes[left].text + " " + nodes[right].text;
            auto it = merge_rank_.find(pair);
            if (it == merge_rank_.end() || it->second != rank) continue;
            nodes[left].text += nodes[right].text;
            nodes[left].next = nodes[right].next;
            if (nodes[right].next >= 0) nodes[nodes[right].next].prev = left;
            nodes[right].next = -1; nodes[right].prev = -1;
            if (nodes[left].prev >= 0) try_add(nodes[left].prev);
            try_add(left);
        }
        symbols.clear();
        for (int i = 0; i >= 0; i = nodes[i].next)
            symbols.push_back(nodes[i].text);
    }

    // Convert symbols to token IDs
    std::vector<int32_t> ids;
    for (const auto & sym : symbols) {
        auto it = token_to_id_.find(sym);
        if (it != token_to_id_.end()) {
            ids.push_back(it->second);
        } else {
            // Byte fallback for unknown symbols
            for (unsigned char byte : sym) {
                char hex[16];
                snprintf(hex, sizeof(hex), "<0x%02X>", byte);
                auto bit = token_to_id_.find(hex);
                if (bit != token_to_id_.end()) {
                    ids.push_back(bit->second);
                }
                // else skip (shouldn't happen with proper vocab)
            }
        }
    }
    return ids;
}

embed_tokens BPETokenizer::encode(const std::string & text) const {
    std::vector<int32_t> ids;

    if (spm_style_) {
        // SentencePiece BPE (Gemma): prepend space, replace spaces with ▁
        std::string processed;
        // Note: Gemma prepends ▁ to the text (space → ▁)
        for (size_t i = 0; i < text.size(); i++) {
            if (text[i] == ' ' || i == 0) {
                if (i == 0 && text[i] != ' ') {
                    // No leading space in input — still don't prepend ▁ for Gemma
                    // Gemma's tokenizer does NOT prepend space like XLM-R
                    processed += text[i];
                } else if (text[i] == ' ') {
                    processed += "\xe2\x96\x81";  // ▁ (U+2581)
                }
            } else {
                processed += text[i];
            }
        }
        ids = bpe_merge(processed);

        // Add BOS/EOS
        if (bos_id_ >= 0) {
            ids.insert(ids.begin(), bos_id_);
        }
        // Append suffix (EOS)
        if (suffix_id_ >= 0) {
            ids.push_back(suffix_id_);
        }
    } else {
        // GPT-2 byte-level BPE (Qwen3, ModernBERT). Pre-split on `<|...|>`
        // special tokens — Qwen-style vocabs have these as added tokens
        // (e.g. <|im_start|>, <|image_pad|>, <|vision_start|>) that the
        // base BPE would otherwise split into individual sub-word tokens.
        // We scan for the exact string in the vocab; only known special
        // tokens are split out, unknown <|...|>-shaped substrings fall
        // through to the BPE byte-level path.
        size_t pos = 0;
        while (pos < text.size()) {
            // Find the next *valid* special token starting at or after pos.
            size_t scan = pos;
            size_t special_start = std::string::npos;
            int    special_id    = -1;
            size_t special_len   = 0;
            while (scan < text.size()) {
                const size_t s = text.find("<|", scan);
                if (s == std::string::npos) break;
                const size_t e = text.find("|>", s + 2);
                if (e == std::string::npos) break;
                const std::string cand = text.substr(s, e - s + 2);
                const auto it = token_to_id_.find(cand);
                if (it != token_to_id_.end()) {
                    special_start = s;
                    special_id    = it->second;
                    special_len   = e - s + 2;
                    break;
                }
                scan = s + 2;  // try the next `<|` occurrence
            }
            if (special_start == std::string::npos) {
                auto sub = core_bpe::tokenize_simple(token_to_id_, merge_rank_,
                                                     text.substr(pos));
                ids.insert(ids.end(), sub.begin(), sub.end());
                break;
            }
            if (special_start > pos) {
                auto sub = core_bpe::tokenize_simple(token_to_id_, merge_rank_,
                                                     text.substr(pos, special_start - pos));
                ids.insert(ids.end(), sub.begin(), sub.end());
            }
            ids.push_back(special_id);
            pos = special_start + special_len;
        }

        // For encoder models: wrap with BOS (CLS) and EOS (SEP)
        if (bos_id_ >= 0) {
            ids.insert(ids.begin(), bos_id_);
        }
        // Append suffix/EOS token
        if (suffix_id_ >= 0) {
            ids.push_back(suffix_id_);
        } else if (eos_id_ >= 0 && bos_id_ >= 0) {
            // Encoder BPE: add SEP at end (eos_id = sep_id)
            ids.push_back(eos_id_);
        }
    }

    // Build result (no padding for decoder models)
    int seq_len = std::min((int)ids.size(), max_length_);

    embed_tokens result;
    result.ids.resize(seq_len);
    result.type_ids.resize(seq_len, 0);
    result.attn_mask.resize(seq_len, 1);

    for (int i = 0; i < seq_len; i++) {
        result.ids[i] = ids[i];
    }

    return result;
}
