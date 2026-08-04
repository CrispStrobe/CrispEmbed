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
// The decode side (piece -> raw bytes) is the mechanical inverse of the
// byte_encoder() permutation and is identical across every GPT-2 byte-level
// model, so it lives here too (byte_decoder() + unicode_to_bytes()). What
// stays in each model is only the *policy* of which special tokens to skip
// (<s>, <|...|>, [UNUSED_*], <0xXX> byte-fallbacks, …) — that genuinely
// varies per tokenizer and is not a byte-transform concern.
//
// Header-only: each consumer compiles its own copy. The byte_encoder
// table and the per-call BPE merge work are tiny enough that the
// indirection cost of a function-pointer interface isn't worth it.

#pragma once

#include "unicode_categ.h"

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
inline const std::vector<int> & byte_encoder() {
    static std::vector<int> bs(256, 0);
    static bool initialized = false;
    if (initialized) return bs;
    std::vector<int> printable;
    for (int b = 0x21; b <= 0x7e; b++) printable.push_back(b);
    for (int b = 0xa1; b <= 0xac; b++) printable.push_back(b);
    for (int b = 0xae; b <= 0xff; b++) printable.push_back(b);
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
inline void utf8_encode(uint32_t cp, std::string & out) {
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
inline std::string bytes_to_unicode(const char * bytes, size_t n) {
    auto & enc = byte_encoder();
    std::string out;
    out.reserve(n);
    for (size_t i = 0; i < n; i++) {
        utf8_encode((uint32_t)enc[(unsigned char)bytes[i]], out);
    }
    return out;
}

// Inverse of byte_encoder(): unicode codepoint -> raw byte. Built lazily
// once. This is the exact reverse permutation used by every GPT-2
// byte-level tokenizer, so decode is model-agnostic (see unicode_to_bytes).
inline const std::unordered_map<uint32_t, uint8_t> & byte_decoder() {
    static const std::unordered_map<uint32_t, uint8_t> dec = [] {
        std::unordered_map<uint32_t, uint8_t> m;
        const auto & enc = byte_encoder();
        for (int b = 0; b < 256; b++) m[(uint32_t)enc[b]] = (uint8_t)b;
        return m;
    }();
    return dec;
}

// Decode the utf-8 length of the byte at `s[i]` (1..4). Malformed leading
// bytes decode as length 1 so callers always make forward progress.
inline size_t utf8_len(unsigned char c) {
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

// Decode one byte-encoded BPE piece (a string of utf-8 codepoints, e.g.
// "Ġx" for " x") back to the raw bytes it represents. Each codepoint is
// mapped through byte_decoder(); codepoints not in the table (shouldn't
// happen for well-formed vocab pieces) are kept verbatim as their original
// utf-8 bytes. Appends to `out`.
inline void unicode_to_bytes(const std::string & piece, std::string & out) {
    const auto & dec = byte_decoder();
    size_t i = 0;
    while (i < piece.size()) {
        unsigned char c = (unsigned char)piece[i];
        size_t len = utf8_len(c);
        if (i + len > piece.size()) len = 1;
        uint32_t cp = 0;
        if (len == 1)
            cp = c;
        else if (len == 2)
            cp = ((c & 0x1F) << 6) | (piece[i + 1] & 0x3F);
        else if (len == 3)
            cp = ((c & 0x0F) << 12) | ((piece[i + 1] & 0x3F) << 6) | (piece[i + 2] & 0x3F);
        else
            cp = ((c & 0x07) << 18) | ((piece[i + 1] & 0x3F) << 12) | ((piece[i + 2] & 0x3F) << 6) |
                 (piece[i + 3] & 0x3F);
        auto it = dec.find(cp);
        if (it != dec.end())
            out.push_back((char)it->second);
        else
            out.append(piece, i, len);
        i += len;
    }
}

// Convenience overload returning the decoded bytes for a single piece.
inline std::string unicode_to_bytes(const std::string & piece) {
    std::string out;
    out.reserve(piece.size());
    unicode_to_bytes(piece, out);
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
inline void bpe_one(const std::unordered_map<std::string, int32_t> & token_to_id,
                    const std::unordered_map<std::string, int32_t> & merge_rank, const std::string & word,
                    std::vector<int32_t> & out) {
    if (word.empty()) return;

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
            if (i + len > word.size()) len = 1;
            symbols.emplace_back(word, i, len);
            i += len;
        }
    }
    if (symbols.empty()) return;

    if (!merge_rank.empty() && symbols.size() >= 2) {
        // Priority-queue BPE: O(N log N) instead of O(N²).
        // Linked list of symbol nodes + min-heap of (rank, left_node_id) pairs.
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

        // (rank, left_node_id) — lower rank = higher priority
        using PQEntry = std::pair<int32_t, int>;
        auto cmp = [](const PQEntry & a, const PQEntry & b) { return a.first > b.first; };
        std::priority_queue<PQEntry, std::vector<PQEntry>, decltype(cmp)> pq(cmp);

        // Helper: try to add pair (i, nodes[i].next) to the queue
        auto try_add = [&](int i) {
            int j = nodes[i].next;
            if (j < 0) return;
            std::string pair = nodes[i].text + " " + nodes[j].text;
            auto it = merge_rank.find(pair);
            if (it != merge_rank.end()) pq.push({ it->second, i });
        };

        // Seed queue with all initial adjacent pairs
        for (int i = 0; i < n; i++) try_add(i);

        while (!pq.empty()) {
            auto [rank, left] = pq.top();
            pq.pop();
            int right = nodes[left].next;
            if (right < 0) continue; // stale entry

            // Validate: re-check that the merge is still the correct pair at this rank
            std::string pair = nodes[left].text + " " + nodes[right].text;
            auto it = merge_rank.find(pair);
            if (it == merge_rank.end() || it->second != rank) continue; // stale

            // Merge: left absorbs right
            nodes[left].text += nodes[right].text;
            nodes[left].next = nodes[right].next;
            if (nodes[right].next >= 0) nodes[nodes[right].next].prev = left;
            nodes[right].next = -1;
            nodes[right].prev = -1; // mark dead

            // Re-queue new adjacent pairs
            if (nodes[left].prev >= 0) try_add(nodes[left].prev);
            try_add(left);
        }

        // Collect surviving symbols (node 0 is always head — never absorbed)
        symbols.clear();
        for (int i = 0; i >= 0; i = nodes[i].next) symbols.push_back(nodes[i].text);
    }

    for (const auto & s : symbols) {
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
                if (jt != token_to_id.end()) out.push_back(jt->second);
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
inline std::vector<int32_t> tokenize_simple(const std::unordered_map<std::string, int32_t> & token_to_id,
                                            const std::unordered_map<std::string, int32_t> & merge_rank,
                                            const std::string & text) {
    std::vector<int32_t> result;
    size_t i = 0;
    bool first = true;
    while (i < text.size()) {
        while (i < text.size() && (text[i] == ' ' || text[i] == '\t' || text[i] == '\n')) i++;
        if (i >= text.size()) break;
        size_t j = i;
        while (j < text.size() && text[j] != ' ' && text[j] != '\t' && text[j] != '\n') j++;
        std::string word = text.substr(i, j - i);
        if (!first) word = std::string(" ") + word;
        first = false;
        std::string encoded = bytes_to_unicode(word.data(), word.size());
        bpe_one(token_to_id, merge_rank, encoded, result);
        i = j;
    }
    return result;
}

// --- Qwen2/Qwen3 ByteLevel pre-tokenizer -----------------------------------
//
// `tokenize_simple` above throws whitespace away: it splits on space/tab/
// newline and rejoins the runs with a single space, so "a\n\n  b" and "a b"
// produce identical ids. For Qwen-family models that is a real defect — the
// newline is a meaningful token and the instruction prompts these embedding
// models ship contain one ("Instruct: ...\nQuery: ").
//
// The functions below implement the pre-tokenizer regex that Qwen2/Qwen3
// tokenizer.json actually declares, verbatim:
//
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)   -- 1: contraction, case-insensitive
//   |[^\r\n\p{L}\p{N}]?\p{L}+      -- 2: one optional non-CR/LF/alnum char, letters
//   |\p{N}                         -- 3: exactly ONE digit (Qwen splits digits)
//   | ?[^\s\p{L}\p{N}]+[\r\n]*     -- 4: optional space, punctuation, trailing CR/LF
//   |\s*[\r\n]+                    -- 5: whitespace run ending in newlines
//   |\s+(?!\S)                     -- 6: trailing whitespace
//   |\s+                           -- 7: any whitespace
//
// Alternatives are tried in order at each position, exactly as a regex engine
// would. \p{L} is approximated as ASCII letters plus any non-ASCII codepoint
// and \p{N} as ASCII digits — the same approximation the CLIP/GPT-2
// pre-tokenizers in this repo already use.
inline bool qwen_is_ascii_digit(unsigned char c) {
    return c >= '0' && c <= '9';
}

inline bool qwen_is_space(unsigned char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

// \p{L}: ASCII letters + any non-ASCII lead byte (approximation).
inline bool qwen_is_letter(unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 0x80;
}

inline bool qwen_is_newline(unsigned char c) {
    return c == '\r' || c == '\n';
}

inline std::vector<std::string> qwen_pretokenize(const std::string & s) {
    std::vector<std::string> out;
    const size_t n = s.size();
    size_t i = 0;
    while (i < n) {
        const unsigned char c = (unsigned char)s[i];

        // 1. (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if (c == '\'' && i + 1 < n) {
            auto lower = [](unsigned char x) -> unsigned char {
                return (x >= 'A' && x <= 'Z') ? (unsigned char)(x - 'A' + 'a') : x;
            };
            const unsigned char d1 = lower((unsigned char)s[i + 1]);
            const unsigned char d2 = (i + 2 < n) ? lower((unsigned char)s[i + 2]) : 0;
            if ((d1 == 'r' && d2 == 'e') || (d1 == 'v' && d2 == 'e') || (d1 == 'l' && d2 == 'l')) {
                out.push_back(s.substr(i, 3));
                i += 3;
                continue;
            }
            if (d1 == 's' || d1 == 't' || d1 == 'm' || d1 == 'd') {
                out.push_back(s.substr(i, 2));
                i += 2;
                continue;
            }
        }

        // 2. [^\r\n\p{L}\p{N}]?\p{L}+
        //    The optional first char is ANY single char that is not CR/LF and
        //    not a letter or digit — a space, but also '(' or '-'.
        {
            size_t k = i;
            if (!qwen_is_newline(c) && !qwen_is_letter(c) && !qwen_is_ascii_digit(c)) k += utf8_len(c);
            if (k < n && qwen_is_letter((unsigned char)s[k])) {
                size_t j = k;
                while (j < n && qwen_is_letter((unsigned char)s[j])) j += utf8_len((unsigned char)s[j]);
                out.push_back(s.substr(i, j - i));
                i = j;
                continue;
            }
        }

        // 3. \p{N} — a single digit, never a run.
        if (qwen_is_ascii_digit(c)) {
            out.push_back(s.substr(i, 1));
            i += 1;
            continue;
        }

        // 4. ` ?[^\s\p{L}\p{N}]+[\r\n]*`
        {
            size_t k = i;
            if (c == ' ') k++;
            if (k < n) {
                const unsigned char cc = (unsigned char)s[k];
                if (!qwen_is_space(cc) && !qwen_is_letter(cc) && !qwen_is_ascii_digit(cc)) {
                    size_t j = k;
                    while (j < n) {
                        const unsigned char e = (unsigned char)s[j];
                        if (qwen_is_space(e) || qwen_is_letter(e) || qwen_is_ascii_digit(e)) break;
                        j += utf8_len(e);
                    }
                    while (j < n && qwen_is_newline((unsigned char)s[j])) j++;
                    out.push_back(s.substr(i, j - i));
                    i = j;
                    continue;
                }
            }
        }

        // 5. `\s*[\r\n]+` — a whitespace run that contains a newline. Take the
        //    longest prefix of whitespace ending at the last newline of the run.
        {
            size_t j = i;
            while (j < n && qwen_is_space((unsigned char)s[j])) j++;
            size_t last_nl = std::string::npos;
            for (size_t t = i; t < j; t++)
                if (qwen_is_newline((unsigned char)s[t])) last_nl = t;
            if (last_nl != std::string::npos) {
                out.push_back(s.substr(i, last_nl + 1 - i));
                i = last_nl + 1;
                continue;
            }

            // 6/7. `\s+(?!\S)` then `\s+`: a whitespace run with no newline.
            //      When the run is followed by a non-space, the last space is
            //      left for the next token's leading ` ?`.
            if (j == n || (j - i) == 1) {
                out.push_back(s.substr(i, j - i));
                i = j;
            } else {
                out.push_back(s.substr(i, (j - 1) - i));
                i = j - 1;
            }
            continue;
        }
    }
    return out;
}

// Qwen2/Qwen3 pre-tokenizer + BPE merge pass. Drop-in replacement for
// tokenize_simple for Qwen-family vocabs; unlike that function it is
// whitespace- and newline-faithful.
inline std::vector<int32_t> tokenize_qwen(const std::unordered_map<std::string, int32_t> & token_to_id,
                                          const std::unordered_map<std::string, int32_t> & merge_rank,
                                          const std::string & text) {
    std::vector<int32_t> result;
    for (const auto & pt : qwen_pretokenize(text)) {
        std::string encoded = bytes_to_unicode(pt.data(), pt.size());
        bpe_one(token_to_id, merge_rank, encoded, result);
    }
    return result;
}

// --- o200k ByteLevel pre-tokenizer -----------------------------------------
//
// The split declared by tokenizer.json of the o200k_base family (GPT-4o,
// gpt-oss, and — the reason it is here — ibm-granite/granite-embedding-97m-
// multilingual-r2), verbatim:
//
//   1. [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*
//                        [\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?
//   2. [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+
//                        [\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?
//   3. \p{N}{1,3}
//   4.  ?[^\s\p{L}\p{N}]+[\r\n/]*
//   5. \s*[\r\n]+
//   6. \s+(?!\S)
//   7. \s+
//
// Alternatives are tried in order at each position, exactly as a backtracking
// regex engine would, and alternatives 1-2 are implemented with the real
// greedy/backtrack semantics of `U* L+` and `U+ L*`.
//
// Unlike the Qwen and GPT-2 pre-tokenizers above, this one BRANCHES ON LETTER
// CASE, so their shared "any byte >= 0x80 is a letter" shortcut is not usable:
// with every non-ASCII letter in both the uppercase and the lowercase class,
// alternative 1 matches a single leading umlaut of an all-caps German word
// ("ÄRGER" -> "Ä" + "RGER") where the reference takes the whole word via
// alternative 2. `core_unicode` therefore carries the real general-category
// table, and this function is the only consumer that needs it.
namespace o200k_detail {

// Longest `(?i:'s|'t|'re|'ve|'m|'ll|'d)` at codepoint index k, in codepoints
// (0 = no match). The apostrophe and the letters are ASCII in the reference
// pattern; a Unicode right-single-quote is NOT a contraction there.
inline size_t contraction_len(const std::string & s, const std::vector<size_t> & off, size_t k, size_t ncp) {
    if (k >= ncp || s[off[k]] != '\'') return 0;
    auto low = [&](size_t idx) -> unsigned char {
        if (idx >= ncp || off[idx + 1] - off[idx] != 1) return 0;
        const unsigned char x = (unsigned char)s[off[idx]];
        return (x >= 'A' && x <= 'Z') ? (unsigned char)(x - 'A' + 'a') : x;
    };
    const unsigned char d1 = low(k + 1);
    if (d1 == 's' || d1 == 't' || d1 == 'm' || d1 == 'd') return 2;
    const unsigned char d2 = low(k + 2);
    if ((d1 == 'r' && d2 == 'e') || (d1 == 'v' && d2 == 'e') || (d1 == 'l' && d2 == 'l')) return 3;
    return 0;
}

} // namespace o200k_detail

inline std::vector<std::string> o200k_pretokenize(const std::string & s) {
    std::vector<std::string> out;
    const size_t n = s.size();
    if (n == 0) return out;

    // Decode once into codepoint boundaries + categories; the alternation is
    // then pure index arithmetic.
    std::vector<size_t> off;
    std::vector<uint8_t> cat;
    off.reserve(n + 1);
    cat.reserve(n);
    for (size_t i = 0; i < n;) {
        off.push_back(i);
        cat.push_back(core_unicode::category(core_unicode::utf8_next(s.data(), n, i)));
    }
    off.push_back(n);
    const size_t ncp = cat.size();
    const size_t npos = (size_t)-1;

    // [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}] — the "uppercase side" class.
    auto in_u = [&](size_t k) {
        return k < ncp &&
               (cat[k] == core_unicode::CAT_LU || cat[k] == core_unicode::CAT_LO || cat[k] == core_unicode::CAT_M);
    };
    // [\p{Ll}\p{Lm}\p{Lo}\p{M}] — the "lowercase side" class.
    auto in_l = [&](size_t k) {
        return k < ncp &&
               (cat[k] == core_unicode::CAT_LL || cat[k] == core_unicode::CAT_LO || cat[k] == core_unicode::CAT_M);
    };
    auto is_letter = [&](size_t k) { return k < ncp && core_unicode::is_letter(cat[k]); };
    auto is_num = [&](size_t k) { return k < ncp && cat[k] == core_unicode::CAT_N; };
    auto is_ws = [&](size_t k) { return k < ncp && cat[k] == core_unicode::CAT_WS; };
    auto byte_at = [&](size_t k) -> char { return k < ncp ? s[off[k]] : '\0'; };
    auto is_nl = [&](size_t k) { return k < ncp && (byte_at(k) == '\r' || byte_at(k) == '\n'); };
    // [^\r\n\p{L}\p{N}] — the optional single leading codepoint of alt 1/2.
    auto is_lead = [&](size_t k) { return k < ncp && !is_nl(k) && !is_letter(k) && !is_num(k); };

    size_t k = 0;
    while (k < ncp) {
        size_t end = npos;

        // Alternatives 1 and 2. The leading `[^\r\n\p{L}\p{N}]?` is greedy, so
        // try it present first and fall back to absent (they can differ only
        // when s[k] is a combining mark, which is both a valid lead and a
        // member of both letter-side classes).
        for (int use_lead = 1; use_lead >= 0 && end == npos; --use_lead) {
            if (use_lead && !is_lead(k)) continue;
            const size_t st = k + (use_lead ? 1 : 0);

            size_t u = st;
            while (in_u(u)) u++; // greedy [Lu Lt Lm Lo M]*

            // Alt 1: `U* L+` — back off the U run to the largest split that
            // leaves at least one L-class codepoint.
            size_t m = npos;
            if (in_l(u)) {
                m = u; // the codepoint after the U run is an Ll
            } else {
                for (size_t t = u; t > st; --t) {
                    if (in_l(t - 1)) {
                        m = t - 1;
                        break;
                    }
                }
            }
            if (m != npos) {
                size_t e = m;
                while (in_l(e)) e++;
                end = e + o200k_detail::contraction_len(s, off, e, ncp);
                break;
            }
            // Alt 2: `U+ L*`.
            if (u > st) {
                size_t e = u;
                while (in_l(e)) e++;
                end = e + o200k_detail::contraction_len(s, off, e, ncp);
                break;
            }
        }

        // Alt 3: `\p{N}{1,3}` — digits in groups of at most three.
        if (end == npos && is_num(k)) {
            size_t e = k;
            while (e < ncp && e - k < 3 && is_num(e)) e++;
            end = e;
        }

        // Alt 4: ` ?[^\s\p{L}\p{N}]+[\r\n/]*`.
        if (end == npos) {
            const size_t st = (byte_at(k) == ' ' && off[k + 1] - off[k] == 1) ? k + 1 : k;
            if (st < ncp && !is_ws(st) && !is_letter(st) && !is_num(st)) {
                size_t e = st;
                while (e < ncp && !is_ws(e) && !is_letter(e) && !is_num(e)) e++;
                while (e < ncp && (byte_at(e) == '\r' || byte_at(e) == '\n' || byte_at(e) == '/')) e++;
                end = e;
            }
        }

        // Alts 5-7: whitespace. `\s*[\r\n]+` takes the run up to and including
        // its LAST newline; otherwise `\s+(?!\S)` leaves the final space of a
        // run for the next token's leading ` ?`, and `\s+` takes the rest.
        if (end == npos && is_ws(k)) {
            size_t j = k;
            while (is_ws(j)) j++;
            size_t last_nl = npos;
            for (size_t t = k; t < j; t++)
                if (is_nl(t)) last_nl = t;
            if (last_nl != npos)
                end = last_nl + 1;
            else if (j == ncp || (j - k) == 1)
                end = j;
            else
                end = j - 1;
        }

        // No alternative matched (only reachable for exotic input): emit one
        // codepoint so the partition stays lossless and the loop terminates.
        if (end == npos || end <= k) end = k + 1;

        out.push_back(s.substr(off[k], off[end] - off[k]));
        k = end;
    }
    return out;
}

// o200k pre-tokenizer + BPE merge pass.
//
// `ignore_merges` mirrors the tokenizer.json flag of the same name (true for
// o200k / llama-3 style vocabs): a pre-token that is itself a vocabulary entry
// is emitted directly, WITHOUT running the merge table over it. The flag exists
// because for a small number of pieces the greedy merge order does not
// reconstruct the vocabulary token, and the reference implementation short-
// circuits those.
inline std::vector<int32_t> tokenize_o200k(const std::unordered_map<std::string, int32_t> & token_to_id,
                                           const std::unordered_map<std::string, int32_t> & merge_rank,
                                           const std::string & text, bool ignore_merges = true) {
    std::vector<int32_t> result;
    for (const auto & pt : o200k_pretokenize(text)) {
        std::string encoded = bytes_to_unicode(pt.data(), pt.size());
        if (ignore_merges) {
            auto it = token_to_id.find(encoded);
            if (it != token_to_id.end()) {
                result.push_back(it->second);
                continue;
            }
        }
        bpe_one(token_to_id, merge_rank, encoded, result);
    }
    return result;
}

} // namespace core_bpe
