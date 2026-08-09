// tokenizer_spm.cpp — SentencePiece Unigram tokenizer.
//
// Uses Viterbi dynamic programming to find the optimal segmentation
// given unigram log-probability scores. This matches HuggingFace's
// tokenizers library behavior for XLM-RoBERTa and similar models.

#include "tokenizer.h"

#include "core/env_gate.h"
#include "core/spm_norm.h"

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cstdio>
#include <cstring>
#include <queue>
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

bool SentencePieceTokenizer::load(const std::vector<std::string> & vocab, const std::vector<float> & scores, int bos_id,
                                  int eos_id, int unk_id, int pad_id, int max_length) {
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
        if ((int)t.size() > max_token_len_) max_token_len_ = (int)t.size();
    }
    bos_id_ = bos_id;
    eos_id_ = eos_id;
    unk_id_ = unk_id;
    pad_id_ = pad_id;
    max_length_ = max_length;
    return !vocab.empty();
}

// HF's normalizer sequence for these models is `Precompiled + Replace`, in
// that order: the charsmap runs BEFORE " " -> "▁". Order matters — the
// charsmap maps U+3000 (ideographic space) to a plain space, which must then
// become a ▁ word boundary like any other space. Applying it after the
// replacement would leave U+3000 glued inside a word.
//
// Enabled per consumer via set_hf_normalize() (only the embedding path is
// measured); CRISPEMBED_SPM_HF_NORM=0 forces it off for a bit-exact
// comparison against pre-fix output.
std::string SentencePieceTokenizer::hf_normalize_text(const std::string & text) const {
    static const bool env_off = core_env::explicitly_off("CRISPEMBED_SPM_HF_NORM");
    if (!hf_normalize_ || env_off) return text;
    return core_spm::normalize(text);
}

// Viterbi dynamic programming: find optimal segmentation of text
// into tokens that maximizes total score.
std::vector<int> SentencePieceTokenizer::tokenize_text(const std::string & text) const {
    if (text.empty()) return {};

    const int n = (int)text.size();

    // best[i] = best total score for text[0..i)
    // back[i] = (token_id, start_pos) for backtracking
    std::vector<float> best(n + 1, -1e30f);
    std::vector<std::pair<int, int>> back(n + 1, { -1, -1 });
    best[0] = 0.0f;

    for (int i = 0; i < n; i++) {
        if (best[i] <= -1e29f) continue; // unreachable position

        // Try all tokens starting at position i
        int max_len = std::min(max_token_len_, n - i);
        for (int len = 1; len <= max_len; len++) {
            // Only try lengths that end on UTF-8 character boundaries
            // (avoid splitting mid-character)
            int end = i + len;
            if (end < n) {
                unsigned char c = (unsigned char)text[end];
                if ((c & 0xC0) == 0x80) continue; // mid-sequence byte
            }

            std::string piece = text.substr(i, len);
            auto it = token_to_id_.find(piece);
            if (it == token_to_id_.end()) continue;

            int tid = it->second;
            float score = (tid < (int)scores_.size()) ? scores_[tid] : 0.0f;
            float candidate = best[i] + score;

            if (candidate > best[end]) {
                best[end] = candidate;
                back[end] = { tid, i };
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
            float score = -100.0f; // heavy penalty for byte fallback
            float candidate = best[i] + score;
            if (candidate > best[i + 1]) {
                best[i + 1] = candidate;
                back[i + 1] = { tid, i };
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

// SentencePiece-BPE segmentation (llama.cpp SPM algorithm). Unlike Viterbi,
// this greedily merges the adjacent symbol pair whose concatenation exists in
// the vocab with the highest score, until no more merges apply. Correct for
// Gemma/Llama, whose `scores` are merge ranks (higher = merged earlier), for
// which Viterbi max-sum over ranks over-segments. Unmatched final symbols fall
// back to byte tokens (<0xXX>) or unk.
std::vector<int> SentencePieceTokenizer::tokenize_bpe(const std::string & text) const {
    if (text.empty()) return {};

    struct Symbol {
        int prev;
        int next;
        size_t pos;
        size_t n;
    };
    std::vector<Symbol> syms;
    for (size_t offs = 0; offs < text.size();) {
        size_t len = std::min(utf8_len((unsigned char)text[offs]), text.size() - offs);
        int idx = (int)syms.size();
        size_t nextoff = offs + len;
        syms.push_back({ idx - 1, nextoff >= text.size() ? -1 : idx + 1, offs, len });
        offs = nextoff;
    }

    struct Bigram {
        int left;
        int right;
        float score;
        size_t size;
    };
    // Max-heap by score, tie-broken by leftmost position (llama.cpp SPM order).
    auto cmp = [](const Bigram & a, const Bigram & b) {
        return a.score < b.score || (a.score == b.score && a.left > b.left);
    };
    std::priority_queue<Bigram, std::vector<Bigram>, decltype(cmp)> work(cmp);

    auto try_add = [&](int left, int right) {
        if (left == -1 || right == -1) return;
        std::string piece = text.substr(syms[left].pos, syms[left].n + syms[right].n);
        auto it = token_to_id_.find(piece);
        if (it == token_to_id_.end()) return;
        int tid = it->second;
        float sc = (tid < (int)scores_.size()) ? scores_[tid] : 0.0f;
        work.push({ left, right, sc, piece.size() });
    };

    for (int i = 1; i < (int)syms.size(); i++) try_add(i - 1, i);

    while (!work.empty()) {
        Bigram b = work.top();
        work.pop();
        Symbol & l = syms[b.left];
        Symbol & r = syms[b.right];
        // Skip if either symbol was already merged, or the pair grew stale.
        if (l.n == 0 || r.n == 0 || l.n + r.n != b.size) continue;
        l.n += r.n;
        r.n = 0;
        l.next = r.next;
        if (r.next >= 0) syms[r.next].prev = b.left;
        try_add(l.prev, b.left);
        try_add(b.left, l.next);
    }

    std::vector<int> out;
    for (int i = 0; i >= 0; i = syms[i].next) {
        if (syms[i].n == 0) continue;
        std::string piece = text.substr(syms[i].pos, syms[i].n);
        auto it = token_to_id_.find(piece);
        if (it != token_to_id_.end()) {
            out.push_back(it->second);
        } else {
            // Byte fallback: emit one <0xXX> token per byte (or unk).
            for (size_t k = 0; k < syms[i].n; k++) {
                unsigned char byte = (unsigned char)text[syms[i].pos + k];
                char hex[8];
                snprintf(hex, sizeof(hex), "<0x%02X>", byte);
                auto bit = token_to_id_.find(hex);
                out.push_back(bit != token_to_id_.end() ? bit->second : unk_id_);
            }
        }
    }
    return out;
}

embed_tokens SentencePieceTokenizer::encode(const std::string & text) const {
    const std::string src = hf_normalize_text(text);
    // SentencePiece: optionally prepend a dummy leading space (XLM-R / Llama
    // add_space_prefix=true), then replace all spaces with ▁ (U+2581). Gemma
    // sets add_space_prefix=false → no leading ▁ (its first word matches the
    // bare vocab token, e.g. "hello" not "▁hello").
    std::string processed = add_space_prefix_ ? (" " + src) : src;
    std::string with_marker;
    for (char c : processed) {
        if (c == ' ') {
            with_marker += "\xe2\x96\x81"; // ▁
        } else {
            with_marker += c;
        }
    }
    processed = with_marker;

    auto token_ids = bpe_merge_ ? tokenize_bpe(processed) : tokenize_text(processed);

    // Build result: <s> + tokens + </s> (each wrap gated by the C2
    // add_bos/add_eos behavior flags; both default true)
    std::vector<int32_t> ids;
    if (add_bos_) ids.push_back(bos_id_);
    for (int id : token_ids) {
        if ((int)ids.size() >= max_length_ - (add_eos_ ? 1 : 0)) break;
        ids.push_back(id);
    }
    if (add_eos_) ids.push_back(eos_id_);

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

embed_tokens SentencePieceTokenizer::encode_pair(const std::string & text_a, const std::string & text_b) const {
    // XLM-R pair encoding: <s> a </s> b </s>  (type_ids all 0 — XLM-R doesn't use them)
    // The rerankers take this path, so they get the charsmap too.
    auto to_marked = [this](const std::string & raw) -> std::string {
        const std::string text = hf_normalize_text(raw);
        std::string out;
        for (char c : (" " + text)) {
            if (c == ' ')
                out += "\xe2\x96\x81";
            else
                out += c;
        }
        return out;
    };

    auto ids_a = tokenize_text(to_marked(text_a));
    auto ids_b = tokenize_text(to_marked(text_b));

    // <s> a </s> b </s> = n_a + n_b + 3 tokens
    int budget = max_length_ - 3;
    while ((int)(ids_a.size() + ids_b.size()) > budget) {
        if (ids_a.size() >= ids_b.size())
            ids_a.pop_back();
        else
            ids_b.pop_back();
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
        result.ids[i] = ids[i];
        result.attn_mask[i] = 1;
    }
    return result;
}
