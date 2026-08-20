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
#include "unicode_class.h"

#include <climits>
#include <cstdint>
#include <cstdlib>
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

        // (rank, left_node_id) — lower rank = higher priority, and on a TIE the
        // leftmost pair wins. The tie-break is not cosmetic: HuggingFace's BPE
        // orders its heap by (rank, pos) both ascending, so "qqqc" merges as
        // "qq"+"q"+"c". Without the second key std::priority_queue leaves equal
        // ranks in an unspecified order and the same input came out "q"+"qq"+"c"
        // — a different token id, on 4 of 1508 random strings per vocab.
        using PQEntry = std::pair<int32_t, int>;
        auto cmp = [](const PQEntry & a, const PQEntry & b) {
            return a.first != b.first ? a.first > b.first : a.second > b.second;
        };
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

// CRISPEMBED_BPE_LEGACY_WHITESPACE=1 restores `tokenize_simple` at every call
// site this audit converted to a declared-regex pre-tokenizer, so the old
// token ids stay reachable for bisection without a rebuild. Read once.
//
// Covered sites: src/tokenizer_bpe.cpp (Qwen-family embedders, T19-E1),
// src/lfm2_embed.cpp, src/deepseek_ocr2.cpp (x2), src/unlimited_ocr.cpp.
inline bool legacy_whitespace() {
    static const bool v = (std::getenv("CRISPEMBED_BPE_LEGACY_WHITESPACE") != nullptr);
    return v;
}

// --- Declared-regex ByteLevel pre-tokenizers --------------------------------
//
// `tokenize_simple` above throws whitespace away: it splits on space/tab/
// newline and rejoins the runs with a single space, so "a\n\n  b" and "a b"
// produce identical ids. That is a real defect for every byte-level BPE model
// — the newline is a meaningful token, the instruction prompts these models
// ship contain one ("Instruct: ...\nQuery: ", "\nFree OCR."), and the damage
// is invisible on newline-free text, which is why it shipped.
//
// The functions below transcribe the pre_tokenizer regex each family's
// tokenizer.json actually declares. Three families are covered; they differ
// less than they look, so read `bytelevel_pretokenize` first and the DeepSeek
// section second.
//
// Callers, and what each one's checkpoint declares:
//
//   qwen_pretokenize      Qwen2/Qwen3 vocabs (src/tokenizer_bpe.cpp)
//   lfm2_pretokenize      LiquidAI/LFM2.5-Embedding-350M (src/lfm2_embed.cpp)
//   deepseek_pretokenize  deepseek-ai/DeepSeek-OCR-2 (src/deepseek_ocr2.cpp)
//                         == baidu/Unlimited-OCR (src/unlimited_ocr.cpp); the
//                         two pre_tokenizer sections are byte-identical.
//
// Every case is pinned against HuggingFace's own `pre_tokenize_str()` output
// in tests/test_qwen_pretokenize.cpp and tests/test_bpe_pretokenize.cpp.

// Decode the codepoint starting at `s[i]`; `len` receives its byte length
// (>= 1, so callers always make forward progress on malformed input).
inline uint32_t utf8_decode_at(const std::string & s, size_t i, size_t & len) {
    const unsigned char c = (unsigned char)s[i];
    len = utf8_len(c);
    if (i + len > s.size()) len = 1;
    if (len == 1) return c;
    if (len == 2) return ((uint32_t)(c & 0x1F) << 6) | (uint32_t)(s[i + 1] & 0x3F);
    if (len == 3)
        return ((uint32_t)(c & 0x0F) << 12) | ((uint32_t)(s[i + 1] & 0x3F) << 6) | (uint32_t)(s[i + 2] & 0x3F);
    return ((uint32_t)(c & 0x07) << 18) | ((uint32_t)(s[i + 1] & 0x3F) << 12) | ((uint32_t)(s[i + 2] & 0x3F) << 6) |
           (uint32_t)(s[i + 3] & 0x3F);
}

// Unicode general category of the codepoint at `s[i]` (see unicode_class.h).
inline core_uc_class uc_at(const std::string & s, size_t i, size_t & len) {
    return core_uc_classify(utf8_decode_at(s, i, len));
}

// The GPT-4-family regex, shared by Qwen2/Qwen3 and LFM2. They are the same
// pattern except for how many digits one token may hold:
//
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)   -- 1: contraction, case-insensitive
//   |[^\r\n\p{L}\p{N}]?\p{L}+      -- 2: one optional non-CR/LF/alnum char, letters
//   |\p{N}{1,MAX}                  -- 3: digits; MAX is 1 for Qwen, 3 for LFM2
//   | ?[^\s\p{L}\p{N}]+[\r\n]*     -- 4: optional space, punctuation, trailing CR/LF
//   |\s*[\r\n]+                    -- 5: whitespace run ending in newlines
//   |\s+(?!\S)                     -- 6: trailing whitespace
//   |\s+                           -- 7: any whitespace
//
// Alternatives are tried in order at each position, leftmost-first, exactly as
// the Rust regex engine behind HuggingFace `tokenizers` does.
inline std::vector<std::string> bytelevel_pretokenize(const std::string & s, int max_digit_run) {
    std::vector<std::string> out;
    const size_t n = s.size();
    size_t i = 0;
    size_t len = 0;
    while (i < n) {
        const unsigned char c = (unsigned char)s[i];
        const core_uc_class cls = uc_at(s, i, len);
        const size_t clen = len;

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
        //    The optional first char is ANY single codepoint that is not CR/LF
        //    and not a letter or digit — a space, but also '(' or '-' or '«'.
        {
            size_t k = i;
            if (c != '\r' && c != '\n' && cls != CORE_UC_L && cls != CORE_UC_N) k += clen;
            if (k < n && uc_at(s, k, len) == CORE_UC_L) {
                size_t j = k;
                while (j < n && uc_at(s, j, len) == CORE_UC_L) j += len;
                out.push_back(s.substr(i, j - i));
                i = j;
                continue;
            }
            // No backtracking arm: retrying without the optional codepoint
            // needs \p{L} at `i`, and `cls != CORE_UC_L` got us here.
        }

        // 3. \p{N}{1,max_digit_run}
        if (cls == CORE_UC_N) {
            size_t j = i;
            for (int taken = 0; taken < max_digit_run && j < n && uc_at(s, j, len) == CORE_UC_N; taken++) j += len;
            out.push_back(s.substr(i, j - i));
            i = j;
            continue;
        }

        // 4. ` ?[^\s\p{L}\p{N}]+[\r\n]*`
        {
            size_t k = i;
            if (c == ' ') k++;
            if (k < n) {
                const core_uc_class ck = uc_at(s, k, len);
                if (ck != CORE_UC_Z && ck != CORE_UC_L && ck != CORE_UC_N) {
                    size_t j = k;
                    while (j < n) {
                        const core_uc_class e = uc_at(s, j, len);
                        if (e == CORE_UC_Z || e == CORE_UC_L || e == CORE_UC_N) break;
                        j += len;
                    }
                    while (j < n && (s[j] == '\r' || s[j] == '\n')) j++;
                    out.push_back(s.substr(i, j - i));
                    i = j;
                    continue;
                }
            }
            // No backtracking arm either: dropping the optional ' ' needs a
            // non-\s codepoint at `i`, and ' ' is \s.
        }

        // 5. `\s*[\r\n]+` — a whitespace run that contains a newline. Take the
        //    longest prefix of whitespace ending at the last newline of the run.
        {
            size_t j = i;
            while (j < n && uc_at(s, j, len) == CORE_UC_Z) j += len;
            if (j == i) {
                // Unreachable for well-formed input: alternatives 2/3/4 cover
                // every class except \s. Emit one codepoint so a malformed
                // byte cannot spin here.
                out.push_back(s.substr(i, clen));
                i += clen;
                continue;
            }
            size_t last_nl = std::string::npos;
            for (size_t t = i; t < j; t++)
                if (s[t] == '\r' || s[t] == '\n') last_nl = t;
            if (last_nl != std::string::npos) {
                out.push_back(s.substr(i, last_nl + 1 - i));
                i = last_nl + 1;
                continue;
            }

            // 6/7. `\s+(?!\S)` then `\s+`: a whitespace run with no newline.
            //      When the run is followed by a non-space, `\s+(?!\S)`
            //      backtracks off the last whitespace codepoint, leaving it
            //      for the next token's leading ` ?`.
            size_t last = j;
            {
                size_t t = i, prev = i, l2 = 0;
                while (t < j) {
                    prev = t;
                    utf8_decode_at(s, t, l2);
                    t += l2;
                }
                last = prev;
            }
            if (j == n || last == i) {
                out.push_back(s.substr(i, j - i));
                i = j;
            } else {
                out.push_back(s.substr(i, last - i));
                i = last;
            }
            continue;
        }
    }
    return out;
}

// Qwen2/Qwen3: one token per digit.
inline std::vector<std::string> qwen_pretokenize(const std::string & s) {
    return bytelevel_pretokenize(s, 1);
}

// LFM2.5: identical regex, but digit runs of up to three.
inline std::vector<std::string> lfm2_pretokenize(const std::string & s) {
    return bytelevel_pretokenize(s, 3);
}

// --- DeepSeek-OCR-2 / Unlimited-OCR ByteLevel pre-tokenizer -----------------
//
// Both checkpoints declare the SAME pre_tokenizer, and it is not one regex but
// a Sequence of three `Split`s with behavior "Isolated", each applied to the
// pieces the previous one produced:
//
//   1. \p{N}{1,3}
//   2. [一-龥぀-ゟ゠-ヿ]+                      (CJK ideographs, hiragana, katakana)
//   3. [!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+
//      |[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+
//      | ?[\p{P}\p{S}]+[\r\n]*
//      |\s*[\r\n]+|\s+(?!\S)|\s+
//
// "Isolated" means the matches AND the text between them both become pieces.
// Stage 3 has no \p{N} alternative at all — stage 1 has already isolated every
// digit run, so a stage-3 piece is either all digits (and matches nothing,
// surviving whole) or contains none.
//
// Versus the Qwen/LFM2 regex the differences that actually move tokens are:
// CJK runs are cut away from adjacent Latin ("中文abc" -> "中文" + "abc"), the
// letter alternative admits combining marks, and the leading-character class is
// built on [\p{P}\p{S}] instead of [^\s\p{L}\p{N}] so "«quote»" splits three
// ways here and two ways under Qwen.

// The literal ranges of stage 2: U+4E00..U+9FA5, U+3040..U+309F, U+30A0..U+30FF.
inline bool ds_is_cjk(uint32_t cp) {
    return (cp >= 0x4E00 && cp <= 0x9FA5) || (cp >= 0x3040 && cp <= 0x309F) || (cp >= 0x30A0 && cp <= 0x30FF);
}

// Stage 1 + stage 2 share this shape: walk the piece, and whenever `match`
// starts at the cursor, flush the pending gap and emit the matched run.
template <typename MatchFn>
inline void ds_split_isolated(const std::string & s, MatchFn match, std::vector<std::string> & out) {
    const size_t n = s.size();
    size_t i = 0, gap = 0, len = 0;
    while (i < n) {
        const size_t j = match(s, i);
        if (j > i) {
            if (i > gap) out.push_back(s.substr(gap, i - gap));
            out.push_back(s.substr(i, j - i));
            i = j;
            gap = i;
        } else {
            utf8_decode_at(s, i, len);
            i += len;
        }
    }
    if (n > gap) out.push_back(s.substr(gap, n - gap));
}

// Stage 3's alternation. Returns the end offset of the match starting at `i`,
// or `i` when no alternative applies (the codepoint joins the gap).
inline size_t ds_match_main(const std::string & s, size_t i) {
    const size_t n = s.size();
    size_t len = 0;
    const unsigned char c = (unsigned char)s[i];
    const core_uc_class cls = uc_at(s, i, len);
    const size_t clen = len;

    // 1. `[<ascii punct/symbol>][A-Za-z]+` — the 32 ASCII characters the regex
    //    lists are exactly the ASCII members of \p{P} u \p{S}.
    if (c < 0x80 && cls == CORE_UC_P) {
        size_t j = i + 1;
        while (j < n && (((unsigned char)s[j] >= 'a' && (unsigned char)s[j] <= 'z') ||
                         ((unsigned char)s[j] >= 'A' && (unsigned char)s[j] <= 'Z')))
            j++;
        if (j > i + 1) return j;
    }

    // 2. `[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+`
    {
        auto is_lm = [](core_uc_class k) { return k == CORE_UC_L || k == CORE_UC_M; };
        size_t k = i;
        bool took_optional = false;
        if (c != '\r' && c != '\n' && cls != CORE_UC_L && cls != CORE_UC_P) {
            k += clen;
            took_optional = true;
        }
        size_t start = std::string::npos;
        if (k < n && is_lm(uc_at(s, k, len)))
            start = k;
        else if (took_optional && is_lm(cls))
            start = i; // backtrack: the optional class also admits \p{M}
        if (start != std::string::npos) {
            size_t j = start;
            while (j < n && is_lm(uc_at(s, j, len))) j += len;
            return j;
        }
    }

    // 3. ` ?[\p{P}\p{S}]+[\r\n]*`
    {
        size_t k = i;
        if (c == ' ') k++;
        if (k < n && uc_at(s, k, len) == CORE_UC_P) {
            size_t j = k;
            while (j < n && uc_at(s, j, len) == CORE_UC_P) j += len;
            while (j < n && (s[j] == '\r' || s[j] == '\n')) j++;
            return j;
        }
    }

    // 4/5/6. `\s*[\r\n]+` then `\s+(?!\S)` then `\s+` — identical semantics to
    //        alternatives 5/6/7 of bytelevel_pretokenize above.
    {
        size_t j = i;
        while (j < n && uc_at(s, j, len) == CORE_UC_Z) j += len;
        if (j == i) return i;
        size_t last_nl = std::string::npos;
        for (size_t t = i; t < j; t++)
            if (s[t] == '\r' || s[t] == '\n') last_nl = t;
        if (last_nl != std::string::npos) return last_nl + 1;
        size_t last = i, t = i, l2 = 0;
        while (t < j) {
            last = t;
            utf8_decode_at(s, t, l2);
            t += l2;
        }
        return (j == n || last == i) ? j : last;
    }
}

inline std::vector<std::string> deepseek_pretokenize(const std::string & s) {
    std::vector<std::string> stage, next;

    // 1. \p{N}{1,3}
    ds_split_isolated(
        s,
        [](const std::string & t, size_t i) {
            size_t j = i, len = 0;
            for (int taken = 0; taken < 3 && j < t.size() && uc_at(t, j, len) == CORE_UC_N; taken++) j += len;
            return j;
        },
        stage);

    // 2. [一-龥぀-ゟ゠-ヿ]+
    for (const auto & p : stage)
        ds_split_isolated(
            p,
            [](const std::string & t, size_t i) {
                size_t j = i, len = 0;
                while (j < t.size() && ds_is_cjk(utf8_decode_at(t, j, len))) j += len;
                return j;
            },
            next);
    stage.swap(next);
    next.clear();

    // 3. the main alternation
    for (const auto & p : stage) ds_split_isolated(p, ds_match_main, next);
    return next;
}

// --- pre-tokenizer + BPE merge pass ----------------------------------------
//
// Drop-in replacements for tokenize_simple; unlike that function they are
// whitespace- and newline-faithful.
inline std::vector<int32_t> tokenize_pretokenized(const std::unordered_map<std::string, int32_t> & token_to_id,
                                                  const std::unordered_map<std::string, int32_t> & merge_rank,
                                                  const std::vector<std::string> & pretokens) {
    std::vector<int32_t> result;
    for (const auto & pt : pretokens) {
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

inline std::vector<std::string> case_aware_pretokenize(const std::string & s, size_t max_digit_run,
                                                       bool contraction_suffixes) {
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
                end = e + (contraction_suffixes ? o200k_detail::contraction_len(s, off, e, ncp) : 0);
                break;
            }
            // Alt 2: `U+ L*`.
            if (u > st) {
                size_t e = u;
                while (in_l(e)) e++;
                end = e + (contraction_suffixes ? o200k_detail::contraction_len(s, off, e, ncp) : 0);
                break;
            }
        }

        // Alt 3: `\p{N}{1,N}` — N=3 for o200k and N=1 for Ministral3/Tekken.
        if (end == npos && is_num(k)) {
            size_t e = k;
            while (e < ncp && e - k < max_digit_run && is_num(e)) e++;
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

inline std::vector<std::string> o200k_pretokenize(const std::string & s) {
    return case_aware_pretokenize(s, 3, true);
}

// Ministral3's TokenizersBackend/Tekken split is the o200k case-aware pattern
// without contraction suffixes and with one digit per pre-token.  Keep this as
// a named path: using Qwen's simpler letter classes silently changes German
// umlaut/all-caps boundaries, while using o200k groups numeric identifiers.
inline std::vector<std::string> ministral_pretokenize(const std::string & s) {
    return case_aware_pretokenize(s, 1, false);
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

inline std::vector<int32_t> tokenize_ministral(const std::unordered_map<std::string, int32_t> & token_to_id,
                                               const std::unordered_map<std::string, int32_t> & merge_rank,
                                               const std::string & text) {
    return tokenize_pretokenized(token_to_id, merge_rank, ministral_pretokenize(text));
}

inline std::vector<int32_t> tokenize_qwen(const std::unordered_map<std::string, int32_t> & token_to_id,
                                          const std::unordered_map<std::string, int32_t> & merge_rank,
                                          const std::string & text) {
    return tokenize_pretokenized(token_to_id, merge_rank, qwen_pretokenize(text));
}

inline std::vector<int32_t> tokenize_lfm2(const std::unordered_map<std::string, int32_t> & token_to_id,
                                          const std::unordered_map<std::string, int32_t> & merge_rank,
                                          const std::string & text) {
    return tokenize_pretokenized(token_to_id, merge_rank, lfm2_pretokenize(text));
}

inline std::vector<int32_t> tokenize_deepseek(const std::unordered_map<std::string, int32_t> & token_to_id,
                                              const std::unordered_map<std::string, int32_t> & merge_rank,
                                              const std::string & text) {
    return tokenize_pretokenized(token_to_id, merge_rank, deepseek_pretokenize(text));
}

} // namespace core_bpe
