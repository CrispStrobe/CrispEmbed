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

    // Iteratively merge the highest-priority pair
    while (symbols.size() > 1) {
        int best_rank = INT_MAX;
        int best_idx = -1;
        for (int j = 0; j < (int)symbols.size() - 1; j++) {
            std::string pair = symbols[j] + " " + symbols[j + 1];
            auto it = merge_rank_.find(pair);
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_idx = j;
            }
        }
        if (best_idx < 0) break;  // no more merges possible

        // Merge symbols[best_idx] and symbols[best_idx+1]
        symbols[best_idx] = symbols[best_idx] + symbols[best_idx + 1];
        symbols.erase(symbols.begin() + best_idx + 1);
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
        // GPT-2 byte-level BPE (Qwen3)
        ids = core_bpe::tokenize_simple(token_to_id_, merge_rank_, text);

        // Append suffix token if model uses one
        if (suffix_id_ >= 0) {
            ids.push_back(suffix_id_);
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
