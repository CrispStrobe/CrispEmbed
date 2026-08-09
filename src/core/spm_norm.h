// src/core/spm_norm.h — the SentencePiece `Precompiled` (nmt_nfkc charsmap)
// normalizer that every XLM-R-family Unigram tokenizer declares.
//
// WHAT WAS WRONG. CrispEmbed's SentencePiece path applied NO normalizer at
// all — `grep precompiled_charsmap` found nothing in the runtime, the
// converter, or any GGUF — while every multilingual embedder we ship declares
// one. Measured on multilingual-e5-small, `…` (U+2026) must normalize to
// `...`, a single in-vocab token; we emitted three `<unk>`.
//
// It is not an ellipsis edge case. 4837 codepoints are affected and the
// common ones are routine in CJK text:
//
//     …  ‥        -> ...  ..
//     Ａ ａ １     -> A a 1        (ALL fullwidth forms)
//     U+3000      -> ' '          (ideographic space)
//     ﬁ ﬂ         -> fi fl
//     ① Ⅳ ㎏ ㈱   -> 1 IV kg (株)
//
// Typographic quotes and dashes are NOT in this charsmap, which is why plain
// Latin text already matched HF and this went unnoticed.
//
// WHY ONE SHARED TABLE IS CORRECT. The charsmap is byte-identical (sha256
// ce10d747...) across every SentencePiece model this repo loads that has one:
// multilingual-e5-small/base, bge-m3, granite-embedding-107m/278m-
// multilingual, arctic-embed-m-v2, and google/siglip-base-patch16-{224,384}.
// All agree on every one of the 65536 BMP codepoints. The shipped GLiNER GGUF
// is an LFM2 **BPE** model with no normalizer, so it never reaches this path.
//
// THE PRINTABLE-ASCII INVARIANT. Codepoints 0x20-0x7E are untouched, asserted
// at table-generation time and in tests/test_spm_norm.cpp. It is *printable*
// ASCII, not all of it: the charsmap also folds \t \n \f \r to a space and
// deletes the remaining C0 controls and DEL, matching HF. Both are pinned by
// tests rather than glossed as "ASCII is unchanged", which would be false.
//
// Gated by CRISPEMBED_SPM_HF_NORM (default on; `=0` restores the historical
// no-normalization path for bit-exact comparison against old output).
#pragma once

#include "unicode_categ.h"
#include "unicode_lower.h"
#include "unicode_spm_norm.h"

#include <string>

namespace core_spm {

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

// Append the normalized form of one codepoint; appends nothing when the
// charsmap deletes it.
inline void normalize_cp(uint32_t cp, std::string & out) {
    using namespace core_unicode_spm;

    // Printable ASCII is untouched by this charsmap (asserted at generation
    // time), and is the overwhelmingly common case.
    if (cp >= 0x20 && cp < 0x7F) {
        out += (char)cp;
        return;
    }

    int lo = 0, hi = N_ROWS - 1;
    while (lo <= hi) {
        const int mid = lo + (hi - lo) / 2;
        if (ROWS[mid].cp == cp) {
            const uint32_t payload = ROWS[mid].payload;
            if (payload == 0) return; // deleted
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

// The Precompiled stage over a whole string. The trailing " " -> "▁" Replace
// that HF's normalizer sequence also carries is deliberately NOT here:
// SentencePieceTokenizer already performs that substitution itself, and
// doing it twice would corrupt every word boundary.
inline std::string normalize(const std::string & text) {
    std::string out;
    out.reserve(text.size());
    const size_t n = text.size();
    size_t i = 0;
    while (i < n) {
        normalize_cp(core_unicode::utf8_next(text.data(), n, i), out);
    }
    return out;
}

// ---------------------------------------------------------------------------
// SigLIP text normalizer
// ---------------------------------------------------------------------------
//
// SigLIP canonicalizes text before SentencePiece, and CrispEmbed implemented
// none of it. `transformers.models.siglip.tokenization_siglip`:
//
//     if self.do_lower_case: text = text.lower()
//     text = self.remove_punctuation(text)      # str.translate over
//                                               # string.punctuation
//     text = re.sub(r"\s+", " ", text)
//     text = text.strip()
//
// then SentencePiece applies its own nmt_nfkc charsmap.
//
// ⚠ WHICH IMPLEMENTATION IS THE AUTHORITY. tokenizer.json ALSO declares a
// normalizer Sequence, and its Replace regex keeps `/`, `<` and `>` — but
// **there is no fast SigLIP tokenizer**: `AutoTokenizer.from_pretrained(...,
// use_fast=True)` still returns the slow `SiglipTokenizer` (`is_fast=False`).
// So the Python path is what runs, and it strips ALL of `string.punctuation`,
// `/ < >` included. Reading the JSON regex instead of measuring the tokenizer
// produced exactly that wrong answer here (16/17 with `/ < >` kept). This is
// the mirror image of core/bert_norm.h, where the Rust side IS the authority
// because those models really do run the fast tokenizer — the lesson is to
// check which one executes, not to prefer one by habit.
//
// It LOWERCASES but does NOT strip accents — `café Müller` -> `café müller` —
// which is why this uses core/unicode_lower.h and not core/bert_norm.h.
//
// Measured on the real engine via crispembed-diff (tests/test_clip_text_diff.cpp
// vs an HF AutoModel reference): "A photo of a CAT, running fast!" scored
// cos 0.8110 without this and 0.9995 with it. The stock fixture
// "a photo of a fox" is all-lowercase and punctuation-free, which is exactly
// why the existing regression never saw the bug.
inline bool is_siglip_stripped_punct(uint32_t cp) {
    // Python's string.punctuation, verbatim: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    return (cp >= '!' && cp <= '/') || (cp >= ':' && cp <= '@') || (cp >= '[' && cp <= '`') || (cp >= '{' && cp <= '~');
}

inline void lower_cp(uint32_t cp, std::string & out) {
    using namespace core_unicode_lower;
    if (cp < 0x80) {
        out += (char)((cp >= 'A' && cp <= 'Z') ? cp - 'A' + 'a' : cp);
        return;
    }
    int lo = 0, hi = N_ROWS - 1;
    while (lo <= hi) {
        const int mid = lo + (hi - lo) / 2;
        if (ROWS[mid].cp == cp) {
            const uint32_t payload = ROWS[mid].payload;
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
    append_utf8_cp(out, cp);
}

inline std::string siglip_normalize(const std::string & text) {
    // [0] Lowercase, [1] drop the punctuation class, [2] collapse whitespace.
    std::string s;
    s.reserve(text.size());
    {
        const size_t n = text.size();
        size_t i = 0;
        bool pending_space = false;
        bool any = false;
        while (i < n) {
            const uint32_t cp = core_unicode::utf8_next(text.data(), n, i);
            if (core_unicode::category(cp) == core_unicode::CAT_WS) {
                pending_space = true;
                continue;
            }
            if (is_siglip_stripped_punct(cp)) continue; // deleted outright
            if (pending_space && any) s += ' ';         // [3] also strips the leading run
            pending_space = false;
            any = true;
            lower_cp(cp, s);
        }
        // A trailing whitespace run is dropped, which is [3]'s right strip.
    }

    // [4] the charsmap, then [5] collapse any doubled spaces it introduced
    // (U+3000 -> ' ' can sit next to an existing space).
    const std::string mapped = normalize(s);
    std::string out;
    out.reserve(mapped.size());
    for (size_t i = 0; i < mapped.size(); i++) {
        if (mapped[i] == ' ' && !out.empty() && out.back() == ' ') continue;
        out += mapped[i];
    }
    return out;
}

} // namespace core_spm
