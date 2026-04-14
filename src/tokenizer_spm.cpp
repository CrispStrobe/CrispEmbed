// tokenizer_spm.cpp — SentencePiece unigram tokenizer.
//
// Adapted from llama.cpp's llm_tokenizer_spm_session (MIT license).
// Uses a priority-queue bigram merging approach with vocab scores.

#include "tokenizer.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <functional>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

// UTF-8 character length
static size_t utf8_len(unsigned char c) {
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

// Symbol in the merge chain
struct spm_symbol {
    const char * text = nullptr;
    size_t       n    = 0;
    int          prev = -1;
    int          next = -1;
};

// Bigram merge candidate
struct spm_bigram {
    int    left;
    int    right;
    float  score;
    size_t size;

    bool operator<(const spm_bigram & other) const {
        // Higher score = higher priority
        return score < other.score;
    }
};

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
    // Pad scores if shorter than vocab
    if (scores_.size() < vocab.size()) {
        scores_.resize(vocab.size(), 0.0f);
    }
    bos_id_ = bos_id;
    eos_id_ = eos_id;
    unk_id_ = unk_id;
    pad_id_ = pad_id;
    max_length_ = max_length;
    return !vocab.empty();
}

std::vector<int> SentencePieceTokenizer::tokenize_text(const std::string & text) const {
    if (text.empty()) return {};

    // Initialize: split into UTF-8 characters as initial symbols
    std::vector<spm_symbol> symbols;
    std::unordered_map<std::string, std::pair<int,int>> rev_merge;
    int index = 0;
    size_t offs = 0;

    while (offs < text.size()) {
        spm_symbol sym;
        size_t len = utf8_len((unsigned char)text[offs]);
        sym.text = text.c_str() + offs;
        sym.n = std::min(len, text.size() - offs);
        offs += sym.n;
        sym.prev = index - 1;
        sym.next = (offs == text.size()) ? -1 : index + 1;
        index++;
        symbols.push_back(sym);
    }

    // Try to add a bigram merge
    auto try_add_bigram = [&](int left, int right,
                               std::priority_queue<spm_bigram> & queue) {
        if (left < 0 || right < 0) return;
        std::string merged(symbols[left].text,
                           symbols[left].n + symbols[right].n);
        auto it = token_to_id_.find(merged);
        if (it == token_to_id_.end()) return;
        int tid = it->second;
        if (tid < 0 || tid >= (int)scores_.size()) return;

        spm_bigram bg;
        bg.left = left;
        bg.right = right;
        bg.score = scores_[tid];
        bg.size = merged.size();
        queue.push(bg);
        rev_merge[merged] = {left, right};
    };

    // Seed queue with all adjacent pairs
    std::priority_queue<spm_bigram> queue;
    for (int i = 1; i < (int)symbols.size(); i++) {
        try_add_bigram(i - 1, i, queue);
    }

    // Greedily merge highest-score bigrams
    while (!queue.empty()) {
        auto bg = queue.top();
        queue.pop();

        auto & left_sym = symbols[bg.left];
        auto & right_sym = symbols[bg.right];

        // Skip if already merged
        if (left_sym.n == 0 || right_sym.n == 0 ||
            left_sym.n + right_sym.n != bg.size) {
            continue;
        }

        // Merge right into left
        left_sym.n += right_sym.n;
        right_sym.n = 0;

        // Update linked list
        left_sym.next = right_sym.next;
        if (right_sym.next >= 0) {
            symbols[right_sym.next].prev = bg.left;
        }

        // Try new bigrams around the merged symbol
        try_add_bigram(left_sym.prev, bg.left, queue);
        try_add_bigram(bg.left, left_sym.next, queue);
    }

    // Collect result tokens
    std::vector<int> output;

    // Recursive resegment for unmatched pieces
    std::function<void(const std::string &)> resegment;
    resegment = [&](const std::string & piece) {
        auto it = token_to_id_.find(piece);
        if (it != token_to_id_.end()) {
            output.push_back(it->second);
            return;
        }
        auto rm = rev_merge.find(piece);
        if (rm != rev_merge.end()) {
            int l = rm->second.first;
            int r = rm->second.second;
            resegment(std::string(symbols[l].text, symbols[l].n));
            resegment(std::string(symbols[r].text, symbols[r].n));
            return;
        }
        // Byte fallback
        for (size_t i = 0; i < piece.size(); i++) {
            // Try <0xHH> format (common in SentencePiece byte fallback)
            char hex[8];
            snprintf(hex, sizeof(hex), "<0x%02X>", (unsigned char)piece[i]);
            auto bit = token_to_id_.find(hex);
            if (bit != token_to_id_.end()) {
                output.push_back(bit->second);
            } else {
                output.push_back(unk_id_);
            }
        }
    };

    for (int i = 0; i != -1; i = symbols[i].next) {
        if (symbols[i].n == 0) continue;
        std::string piece(symbols[i].text, symbols[i].n);
        resegment(piece);
    }

    return output;
}

embed_tokens SentencePieceTokenizer::encode(const std::string & text) const {
    // SentencePiece convention for XLM-R: prepend a space to the text,
    // then replace all spaces with ▁ (U+2581). The BPE merge algorithm
    // then operates on the ▁-prefixed text.
    std::string processed = " " + text;  // leading space (XLM-R convention)
    std::string with_marker;
    for (char c : processed) {
        if (c == ' ') {
            with_marker += "\xe2\x96\x81";  // ▁
        } else {
            with_marker += c;
        }
    }
    processed = with_marker;

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
