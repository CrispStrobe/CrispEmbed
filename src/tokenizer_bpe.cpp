// tokenizer_bpe.cpp — BPE tokenizer for decoder models.
//
// Three modes:
// - GPT-2 byte-level BPE (Qwen3): uses core_bpe from CrispASR
// - SentencePiece BPE (Gemma): ▁ space marker, standard BPE merges
// - CLIP text BPE (OpenAI CLIP): lowercase + whitespace-clean + regex
//   pre-tokenize + byte-level encode + </w> end-of-word suffix

#include "tokenizer.h"
#include "core/bpe.h"

#include <algorithm>
#include <climits>
#include <cstring>
#include <queue>
#include <string>
#include <vector>

bool BPETokenizer::load(const std::vector<std::string> & vocab, const std::vector<std::string> & merges, int eos_id,
                        int pad_id, int suffix_id, int bos_id, bool spm_style, int max_length, bool spm_dummy_prefix,
                        bool clip_style) {
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
    spm_dummy_prefix_ = spm_dummy_prefix;
    clip_style_ = clip_style;
    max_length_ = max_length;
    return !vocab.empty();
}

// Rank-merge a list of initial symbols in place. Priority-queue BPE:
// O(N log N) instead of O(N²). Symbol identity uses "left right" string
// concatenation to match the textual merge table.
void BPETokenizer::merge_symbols(std::vector<std::string> & symbols) const {
    if (symbols.size() < 2) return;

    struct Node {
        std::string text;
        int prev, next;
    };
    int n = (int)symbols.size();
    std::vector<Node> nodes(n);
    for (int i = 0; i < n; i++) {
        nodes[i].text = std::move(symbols[i]);
        nodes[i].prev = i - 1;
        nodes[i].next = i < n - 1 ? i + 1 : -1;
    }
    using PQE = std::pair<int, int>;
    auto cmp = [](const PQE & a, const PQE & b) { return a.first > b.first; };
    std::priority_queue<PQE, std::vector<PQE>, decltype(cmp)> pq(cmp);
    auto try_add = [&](int i) {
        int j = nodes[i].next;
        if (j < 0) return;
        std::string pair = nodes[i].text + " " + nodes[j].text;
        auto it = merge_rank_.find(pair);
        if (it != merge_rank_.end()) pq.push({ it->second, i });
    };
    for (int i = 0; i < n; i++) try_add(i);
    while (!pq.empty()) {
        auto [rank, left] = pq.top();
        pq.pop();
        int right = nodes[left].next;
        if (right < 0) continue;
        std::string pair = nodes[left].text + " " + nodes[right].text;
        auto it = merge_rank_.find(pair);
        if (it == merge_rank_.end() || it->second != rank) continue;
        nodes[left].text += nodes[right].text;
        nodes[left].next = nodes[right].next;
        if (nodes[right].next >= 0) nodes[nodes[right].next].prev = left;
        nodes[right].next = -1;
        nodes[right].prev = -1;
        if (nodes[left].prev >= 0) try_add(nodes[left].prev);
        try_add(left);
    }
    symbols.clear();
    for (int i = 0; i >= 0; i = nodes[i].next) symbols.push_back(nodes[i].text);
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
            if (c < 0xE0)
                len = 2;
            else if (c < 0xF0)
                len = 3;
            else
                len = 4;
        }
        len = std::min(len, text.size() - i);
        symbols.push_back(text.substr(i, len));
        i += len;
    }

    merge_symbols(symbols);

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

namespace {
// UTF-8 codepoint length from a leading byte (malformed → 1 for progress).
inline size_t clip_utf8_len(unsigned char c) {
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}
inline bool clip_is_space(unsigned char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}
inline bool clip_is_digit(unsigned char c) {
    return c >= '0' && c <= '9';
}
// CLIP's regex \p{L}: ASCII letters plus any non-ASCII byte (a multi-byte
// UTF-8 letter such as é/ï/CJK). Rare non-ASCII punctuation is approximated
// as a letter, harmless for CLIP's short-caption domain.
inline bool clip_is_letter(unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= 0x80);
}
} // namespace

// OpenAI CLIP text pre-tokenizer. Reproduces:
//   whitespace_clean(basic_clean(text)).lower() then re.findall(pat, text)
// with pat = 's|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+
// (single space collapse + strip are implicit — whitespace is just skipped).
std::vector<std::string> BPETokenizer::clip_pretokenize(const std::string & text) const {
    std::string s;
    s.reserve(text.size());
    for (unsigned char c : text) s.push_back((c >= 'A' && c <= 'Z') ? (char)(c - 'A' + 'a') : (char)c);

    std::vector<std::string> out;
    const size_t n = s.size();
    size_t i = 0;
    while (i < n) {
        unsigned char c = (unsigned char)s[i];
        if (clip_is_space(c)) {
            i++;
            continue;
        }
        // Contractions: 're 've 'll (two-letter) then 's 't 'm 'd (one-letter).
        if (c == '\'' && i + 1 < n) {
            if (i + 2 < n) {
                std::string two = s.substr(i + 1, 2);
                if (two == "re" || two == "ve" || two == "ll") {
                    out.push_back(s.substr(i, 3));
                    i += 3;
                    continue;
                }
            }
            unsigned char d = (unsigned char)s[i + 1];
            if (d == 's' || d == 't' || d == 'm' || d == 'd') {
                out.push_back(s.substr(i, 2));
                i += 2;
                continue;
            }
        }
        if (clip_is_letter(c)) { // [\p{L}]+
            size_t j = i;
            while (j < n && clip_is_letter((unsigned char)s[j])) j += clip_utf8_len((unsigned char)s[j]);
            out.push_back(s.substr(i, j - i));
            i = j;
            continue;
        }
        if (clip_is_digit(c)) { // [\p{N}] — a single digit per token
            out.push_back(s.substr(i, 1));
            i++;
            continue;
        }
        // [^\s\p{L}\p{N}]+ — a run of punctuation/symbols.
        size_t j = i;
        while (j < n) {
            unsigned char e = (unsigned char)s[j];
            if (clip_is_space(e) || clip_is_letter(e) || clip_is_digit(e)) break;
            j++;
        }
        out.push_back(s.substr(i, j - i));
        i = j;
    }
    return out;
}

// Byte-level encode one CLIP pre-token, append </w> to its final symbol,
// rank-merge, and append the resulting vocab IDs.
void BPETokenizer::clip_bpe_word(const std::string & pretoken, std::vector<int32_t> & out) const {
    if (pretoken.empty()) return;
    std::string enc = core_bpe::bytes_to_unicode(pretoken.data(), pretoken.size());

    std::vector<std::string> symbols;
    size_t i = 0;
    while (i < enc.size()) {
        size_t len = std::min(clip_utf8_len((unsigned char)enc[i]), enc.size() - i);
        symbols.emplace_back(enc, i, len);
        i += len;
    }
    if (symbols.empty()) return;
    symbols.back() += "</w>"; // end-of-word marker on the last character

    merge_symbols(symbols);

    for (const auto & sym : symbols) {
        auto it = token_to_id_.find(sym);
        if (it != token_to_id_.end()) out.push_back(it->second);
        // Unknown symbols cannot occur with a complete CLIP vocab (every
        // byte-unicode char exists both plain and with </w>); skip if so.
    }
}

// GPT-2 ByteLevel regex pre-tokenizer (HF ByteLevel, use_regex=true), as used
// by ModernBERT. Reproduces the reference alternation, tried in order at each
// position:
//   's|'t|'re|'ve|'m|'ll|'d | ?\p{L}+ | ?\p{N}+ | ?[^\s\p{L}\p{N}]+ | \s+(?!\S) | \s+
// where the leading ` ?` is a literal 0x20 space (consumed into the following
// run). \p{L} is approximated as ASCII letters + any non-ASCII byte (same as
// the CLIP pre-tokenizer); NFC normalization is not applied (inputs are assumed
// already-normalized — matches the pragmatic level of clip_pretokenize).
std::vector<std::string> BPETokenizer::gpt2_pretokenize(const std::string & text) const {
    std::vector<std::string> out;
    const std::string & s = text;
    const size_t n = s.size();
    size_t i = 0;
    while (i < n) {
        const unsigned char c = (unsigned char)s[i];

        // 1. Contractions (case-sensitive lowercase, no leading space).
        if (c == '\'' && i + 1 < n) {
            if (i + 2 < n) {
                const std::string two = s.substr(i + 1, 2);
                if (two == "re" || two == "ve" || two == "ll") {
                    out.push_back(s.substr(i, 3));
                    i += 3;
                    continue;
                }
            }
            const unsigned char d = (unsigned char)s[i + 1];
            if (d == 's' || d == 't' || d == 'm' || d == 'd') {
                out.push_back(s.substr(i, 2));
                i += 2;
                continue;
            }
        }

        // 2-4: optional single leading 0x20 space, then a run of one class.
        size_t k = i;
        if (c == ' ') k++; // ` ?` consumes at most one literal space
        if (k < n) {
            const unsigned char cc = (unsigned char)s[k];
            if (clip_is_letter(cc)) { // ` ?\p{L}+`
                size_t j = k;
                while (j < n && clip_is_letter((unsigned char)s[j])) j += clip_utf8_len((unsigned char)s[j]);
                out.push_back(s.substr(i, j - i));
                i = j;
                continue;
            }
            if (clip_is_digit(cc)) { // ` ?\p{N}+`
                size_t j = k;
                while (j < n && clip_is_digit((unsigned char)s[j])) j++;
                out.push_back(s.substr(i, j - i));
                i = j;
                continue;
            }
            if (!clip_is_space(cc)) { // ` ?[^\s\p{L}\p{N}]+`
                size_t j = k;
                while (j < n) {
                    const unsigned char e = (unsigned char)s[j];
                    if (clip_is_space(e) || clip_is_letter(e) || clip_is_digit(e)) break;
                    j += clip_utf8_len(e);
                }
                out.push_back(s.substr(i, j - i));
                i = j;
                continue;
            }
        }

        // 5-6: a whitespace run [i, j). `\s+(?!\S)` matches all-but-the-last
        // whitespace char when the run is followed by a non-space (the last is
        // left for the next word's ` ?`); the whole run at end-of-string or when
        // it is a single char (then plain `\s+`, alternative 6).
        size_t j = i;
        while (j < n && clip_is_space((unsigned char)s[j])) j++;
        if (j == n || (j - 1) == i) {
            out.push_back(s.substr(i, j - i));
            i = j;
        } else {
            out.push_back(s.substr(i, (j - 1) - i));
            i = j - 1;
        }
    }
    return out;
}

embed_tokens BPETokenizer::encode(const std::string & text) const {
    std::vector<int32_t> ids;

    if (clip_style_) {
        // OpenAI CLIP text: regex pre-tokenize + byte-level BPE with </w>.
        for (const auto & pt : clip_pretokenize(text)) clip_bpe_word(pt, ids);
        // Wrap with <|startoftext|> (BOS) and <|endoftext|> (EOS).
        if (bos_id_ >= 0) ids.insert(ids.begin(), bos_id_);
        if (eos_id_ >= 0) ids.push_back(eos_id_);
    } else if (spm_style_) {
        // SentencePiece normalization: optional add_dummy_prefix (a leading
        // space → ▁ at the very start, used by ERNIE-4.5/PaddleOCR-VL), then
        // every space → ▁ (U+2581). Newlines and other bytes fall through to
        // bpe_merge's byte fallback (e.g. \n → <0x0A>). With dummy_prefix=false
        // this reproduces the prior Gemma behavior (no leading ▁).
        std::string src = spm_dummy_prefix_ ? (" " + text) : text;
        std::string processed;
        for (char c : src) {
            if (c == ' ')
                processed += "\xe2\x96\x81"; // ▁ (U+2581)
            else
                processed += c;
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
    } else if (ministral_regex_pretok_) {
        ids = core_bpe::tokenize_ministral(token_to_id_, merge_rank_, text);

        if (bos_id_ >= 0) ids.insert(ids.begin(), bos_id_);
        if (suffix_id_ >= 0) ids.push_back(suffix_id_);
    } else if (o200k_regex_pretok_) {
        // o200k_base ByteLevel (granite-embedding-97m-multilingual-r2). Same
        // shape as the GPT-2 branch below, but with the o200k split (case-aware
        // letter runs, 3-digit groups, contraction suffixes) and the vocab's
        // `ignore_merges` flag.
        ids = core_bpe::tokenize_o200k(token_to_id_, merge_rank_, text, ignore_merges_);

        // For encoder models: wrap with BOS (CLS) and EOS (SEP)
        if (bos_id_ >= 0) ids.insert(ids.begin(), bos_id_);
        if (suffix_id_ >= 0)
            ids.push_back(suffix_id_);
        else if (eos_id_ >= 0 && bos_id_ >= 0)
            ids.push_back(eos_id_);
    } else if (gpt2_regex_pretok_) {
        // GPT-2 ByteLevel with the regex pre-tokenizer (ModernBERT). Split the
        // text with the GPT-2 regex, byte-encode each pre-token, then BPE-merge.
        // No `<|...|>` special handling — ModernBERT's specials ([CLS]/[SEP]/…)
        // are added via the bos/eos id wrap below, not embedded in text.
        for (const auto & pt : gpt2_pretokenize(text)) {
            std::string enc = core_bpe::bytes_to_unicode(pt.data(), pt.size());
            core_bpe::bpe_one(token_to_id_, merge_rank_, enc, ids);
        }

        // For encoder models: wrap with BOS (CLS) and EOS (SEP)
        if (bos_id_ >= 0) ids.insert(ids.begin(), bos_id_);
        if (suffix_id_ >= 0)
            ids.push_back(suffix_id_);
        else if (eos_id_ >= 0 && bos_id_ >= 0)
            ids.push_back(eos_id_);
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
            int special_id = -1;
            size_t special_len = 0;
            while (scan < text.size()) {
                const size_t s = text.find("<|", scan);
                if (s == std::string::npos) break;
                const size_t e = text.find("|>", s + 2);
                if (e == std::string::npos) break;
                const std::string cand = text.substr(s, e - s + 2);
                const auto it = token_to_id_.find(cand);
                if (it != token_to_id_.end()) {
                    special_start = s;
                    special_id = it->second;
                    special_len = e - s + 2;
                    break;
                }
                scan = s + 2; // try the next `<|` occurrence
            }
            // Qwen2/Qwen3 ByteLevel pre-tokenization. `tokenize_simple` splits
            // on whitespace and rejoins with a single space, which silently
            // deletes newlines and indentation — measured against the HF
            // reference on F2LLM-v2-160M that costs cosine 0.980 on a code
            // snippet and 0.991 on this family's own "Instruct: …\nQuery: "
            // prompt, while newline-free text is unaffected (1.000000).
            // CRISPEMBED_BPE_LEGACY_WHITESPACE=1 restores the old behavior.
            static const bool legacy_ws = (std::getenv("CRISPEMBED_BPE_LEGACY_WHITESPACE") != nullptr);
            auto tokenize_run = [&](const std::string & t) {
                return legacy_ws ? core_bpe::tokenize_simple(token_to_id_, merge_rank_, t)
                                 : core_bpe::tokenize_qwen(token_to_id_, merge_rank_, t);
            };
            if (special_start == std::string::npos) {
                auto sub = tokenize_run(text.substr(pos));
                ids.insert(ids.end(), sub.begin(), sub.end());
                break;
            }
            if (special_start > pos) {
                auto sub = tokenize_run(text.substr(pos, special_start - pos));
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
