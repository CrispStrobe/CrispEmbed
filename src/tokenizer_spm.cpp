// tokenizer_spm.cpp — SentencePiece Unigram tokenizer.
//
// Uses Viterbi dynamic programming to find the optimal segmentation
// given unigram log-probability scores. This matches HuggingFace's
// tokenizers library behavior for XLM-RoBERTa and similar models.

#include "tokenizer.h"

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

// UTF-8 character byte length from lead byte
static size_t utf8_len(unsigned char c) {
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

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
    // Find max token length for Viterbi window
    max_token_len_ = 0;
    for (const auto & t : vocab) {
        if ((int)t.size() > max_token_len_)
            max_token_len_ = (int)t.size();
    }
    bos_id_ = bos_id;
    eos_id_ = eos_id;
    unk_id_ = unk_id;
    pad_id_ = pad_id;
    max_length_ = max_length;
    return !vocab.empty();
}

// Viterbi dynamic programming: find optimal segmentation of text
// into tokens that maximizes total score.
std::vector<int> SentencePieceTokenizer::tokenize_text(const std::string & text) const {
    if (text.empty()) return {};

    const int n = (int)text.size();

    // best[i] = best total score for text[0..i)
    // back[i] = (token_id, start_pos) for backtracking
    std::vector<float> best(n + 1, -1e30f);
    std::vector<std::pair<int, int>> back(n + 1, {-1, -1});
    best[0] = 0.0f;

    for (int i = 0; i < n; i++) {
        if (best[i] <= -1e29f) continue;  // unreachable position

        // Try all tokens starting at position i
        int max_len = std::min(max_token_len_, n - i);
        for (int len = 1; len <= max_len; len++) {
            // Only try lengths that end on UTF-8 character boundaries
            // (avoid splitting mid-character)
            int end = i + len;
            if (end < n) {
                unsigned char c = (unsigned char)text[end];
                if ((c & 0xC0) == 0x80) continue;  // mid-sequence byte
            }

            std::string piece = text.substr(i, len);
            auto it = token_to_id_.find(piece);
            if (it == token_to_id_.end()) continue;

            int tid = it->second;
            float score = (tid < (int)scores_.size()) ? scores_[tid] : 0.0f;
            float candidate = best[i] + score;

            if (candidate > best[end]) {
                best[end] = candidate;
                back[end] = {tid, i};
            }
        }

        // Byte fallback: if no token starts here, try single-byte fallback
        // (ensures we can always reach the next position)
        if (best[i + 1] <= -1e29f && i + 1 <= n) {
            unsigned char byte = (unsigned char)text[i];
            char hex[8];
            snprintf(hex, sizeof(hex), "<0x%02X>", byte);
            auto it = token_to_id_.find(hex);
            int tid = (it != token_to_id_.end()) ? it->second : unk_id_;
            float score = -100.0f;  // heavy penalty for byte fallback
            float candidate = best[i] + score;
            if (candidate > best[i + 1]) {
                best[i + 1] = candidate;
                back[i + 1] = {tid, i};
            }
        }
    }

    // Backtrack to recover token sequence
    std::vector<int> tokens;
    int pos = n;
    while (pos > 0) {
        auto [tid, start] = back[pos];
        if (tid < 0) {
            // Should not happen if byte fallback works
            pos--;
            continue;
        }
        tokens.push_back(tid);
        pos = start;
    }
    std::reverse(tokens.begin(), tokens.end());
    return tokens;
}

embed_tokens SentencePieceTokenizer::encode(const std::string & text) const {
    // SentencePiece convention for XLM-R: prepend a space to the text,
    // then replace all spaces with ▁ (U+2581). The Viterbi algorithm
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

embed_tokens SentencePieceTokenizer::encode_pair(const std::string & text_a,
                                                   const std::string & text_b) const {
    // XLM-R pair encoding: <s> a </s> b </s>  (type_ids all 0 — XLM-R doesn't use them)
    auto to_marked = [](const std::string & text) -> std::string {
        std::string out;
        for (char c : (" " + text)) {
            if (c == ' ') out += "\xe2\x96\x81";
            else          out += c;
        }
        return out;
    };

    auto ids_a = tokenize_text(to_marked(text_a));
    auto ids_b = tokenize_text(to_marked(text_b));

    // <s> a </s> b </s> = n_a + n_b + 3 tokens
    int budget = max_length_ - 3;
    while ((int)(ids_a.size() + ids_b.size()) > budget) {
        if (ids_a.size() >= ids_b.size()) ids_a.pop_back();
        else ids_b.pop_back();
    }

    std::vector<int32_t> ids;
    ids.push_back(bos_id_);
    for (int id : ids_a) ids.push_back(id);
    ids.push_back(eos_id_);
    for (int id : ids_b) ids.push_back(id);
    ids.push_back(eos_id_);

    embed_tokens result;
    int seq_len = (int)ids.size();
    result.ids.resize(max_length_, pad_id_);
    result.type_ids.resize(max_length_, 0);
    result.attn_mask.resize(max_length_, 0);
    for (int i = 0; i < seq_len; i++) {
        result.ids[i]       = ids[i];
        result.attn_mask[i] = 1;
    }
    return result;
}
