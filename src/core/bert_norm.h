// src/core/bert_norm.h — the lowercase + strip-accents half of HuggingFace's
// BertNormalizer, for uncased WordPiece models.
//
// WHAT WAS WRONG. `WordPieceTokenizer::split_words` lowercased with a per-BYTE
// `std::tolower`, which does nothing to a multi-byte UTF-8 sequence and strips
// no accents at all. HF's BertNormalizer, for a model whose tokenizer.json says
// `lowercase: true, strip_accents: null`, runs NFD and drops combining marks
// BEFORE the WordPiece lookup, so `café` is the single in-vocab token `cafe`.
// Ours produced `caf` + `[UNK]`. Measured on all-MiniLM-L6-v2 (docs/LANGUAGES.md):
//
//     café -> cafe        vs  caf + [UNK]
//     Müller -> muller    vs  m   + [UNK]
//     über -> uber        vs  [UNK]
//
// so ordinary German / French / Spanish / Portuguese words — text that IS in
// vocabulary — embedded as [UNK].
//
// WHY THE TABLE IS GENERATED (tools/gen_unicode_bert_norm.py). Two traps that a
// hand-written "strip the diacritic" table falls into, both silent:
//
//   * Ø, Ł, Đ, ß, ı, ﬁ have NO canonical decomposition. HF keeps them exactly
//     as they are; only a real NFD tells you that. `Łódź` -> `łodz`, NOT `lodz`.
//   * The RUST normalizer is the authority — every affected model ships a
//     tokenizer.json, so BertTokenizerFast runs it — and it disagrees with
//     Python's `unicodedata` on 441 late-Unicode combining marks. Generating
//     the table from `unicodedata` would have been wrong on all 441.
//
// THE ASCII INVARIANT. For every ASCII codepoint this function is exactly
// `std::tolower`. That is what makes enabling it by default safe: pure-ASCII
// input cannot change, so no shipped English embedding moves. It is asserted
// at generation time AND pinned in tests/test_bert_norm.cpp.
//
// Gated by CRISPEMBED_WORDPIECE_HF_NORM (default on; `=0` restores the
// historical per-byte lowercase for bit-exact comparison against old output).
#pragma once

#include "unicode_bert_norm.h"
#include "unicode_categ.h"

#include <string>

namespace core_bert {

inline void append_utf8_cp(std::string & out, uint32_t cp) {
    if (cp < 0x80) {
        out += (char)cp;
    } else if (cp < 0x800) {
        out += (char)(0xC0 | (cp >> 6));
        out += (char)(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += (char)(0xE0 | (cp >> 12));
        out += (char)(0x80 | ((cp >> 6) & 0x3F));
        out += (char)(0x80 | (cp & 0x3F));
    } else {
        out += (char)(0xF0 | (cp >> 18));
        out += (char)(0x80 | ((cp >> 12) & 0x3F));
        out += (char)(0x80 | ((cp >> 6) & 0x3F));
        out += (char)(0x80 | (cp & 0x3F));
    }
}

// Append the normalized form of one codepoint. Nothing is appended when the
// codepoint is a combining mark (HF's strip-accents filter drops it).
inline void lower_strip_cp(uint32_t cp, std::string & out) {
    using namespace core_unicode_norm;

    // ASCII: identical to std::tolower by construction (see the invariant
    // above), and the overwhelmingly common case — keep it off the table.
    if (cp < 0x80) {
        out += (char)((cp >= 'A' && cp <= 'Z') ? cp - 'A' + 'a' : cp);
        return;
    }

    // Precomposed Hangul decomposes arithmetically into L/V/T jamo, none of
    // which are combining marks. 11172 codepoints that never reach the table.
    if (cp >= HANGUL_SBASE && cp < HANGUL_SBASE + HANGUL_SCOUNT) {
        const uint32_t idx = cp - HANGUL_SBASE;
        const uint32_t t = idx % HANGUL_TCOUNT;
        append_utf8_cp(out, HANGUL_LBASE + idx / HANGUL_NCOUNT);
        append_utf8_cp(out, HANGUL_VBASE + (idx % HANGUL_NCOUNT) / HANGUL_TCOUNT);
        if (t != 0) append_utf8_cp(out, HANGUL_TBASE + t);
        return;
    }

    int lo = 0, hi = N_ROWS - 1;
    while (lo <= hi) {
        const int mid = lo + (hi - lo) / 2;
        if (ROWS[mid].cp == cp) {
            const uint32_t payload = ROWS[mid].payload;
            if (payload == 0) return; // combining mark: dropped
            if (payload & 0x80000000u) {
                const uint32_t * p = &MULTI[payload & 0x7FFFFFFFu];
                for (uint32_t k = 1; k <= p[0]; k++) append_utf8_cp(out, p[k]);
            } else {
                append_utf8_cp(out, payload);
            }
            return;
        }
        if (ROWS[mid].cp < cp) {
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    append_utf8_cp(out, cp); // not in the table: unchanged
}

// HF BertNormalizer's strip_accents + lowercase stage over a whole string.
// clean_text and handle_chinese_chars are pretokenize()'s job; HF runs them
// first, and this stage is per-codepoint so the two compose in either order.
inline std::string lower_strip_accents(const std::string & text) {
    std::string out;
    out.reserve(text.size());
    const size_t n = text.size();
    size_t i = 0;
    while (i < n) {
        lower_strip_cp(core_unicode::utf8_next(text.data(), n, i), out);
    }
    return out;
}

} // namespace core_bert
