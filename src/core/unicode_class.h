// src/core/unicode_class.h — coarse codepoint classes for the byte-level BPE
// pre-tokenizers in core/bpe.h (Qwen / LFM2 / DeepSeek regex transcriptions).
//
// This is a thin MAPPING over core/unicode_categ.h, the repo's single
// generated general-category table (which additionally carries the letter
// CASE distinction the o200k split needs — this coarse view does not).
// Two parallel generated tables for the same job briefly coexisted on
// `feat/tokenize-simple-audit` and `feat/granite-r2-tokenizers`; the categ
// table is a strict superset, so this header now derives from it and the
// second generator was dropped. Both default unlisted codepoints to letter,
// so the fallback semantics are identical.
//
// The regexes here use the Unicode general categories \p{L} \p{M} \p{N}
// \p{P} \p{S} and the \s (White_Space) class. Getting those wrong is not
// cosmetic: treating every non-ASCII byte as a letter (the approximation
// this replaces) merges German typographic quotes into the adjacent word —
// HuggingFace splits `sagte „Hallo“ heute` into 5 pre-tokens, the
// byte>=0x80 approximation produced 3.
//
// \p{P} and \p{S} are one class: every regex in core/bpe.h uses them only
// as the union `[\p{P}\p{S}]` (or its complement), never apart.

#pragma once

#include "unicode_categ.h"

#include <cstdint>

namespace core_bpe {

// Codepoint classes. Ordered so CORE_UC_L stays the zero/default answer.
enum core_uc_class : uint8_t {
    CORE_UC_L = 0, // \p{L}  letter (also the fallback for unlisted codepoints)
    CORE_UC_M = 1, // \p{M}  combining mark
    CORE_UC_N = 2, // \p{N}  number
    CORE_UC_P = 3, // \p{P} | \p{S}  punctuation or symbol
    CORE_UC_Z = 4, // \s     White_Space
    CORE_UC_O = 5, // none of the above (control / format characters)
};

// Class of one Unicode codepoint, derived from the general-category table.
inline core_uc_class core_uc_classify(uint32_t cp) {
    switch (core_unicode::category(cp)) {
    case core_unicode::CAT_LU:
    case core_unicode::CAT_LL:
    case core_unicode::CAT_LO:
        return CORE_UC_L;
    case core_unicode::CAT_M:
        return CORE_UC_M;
    case core_unicode::CAT_N:
        return CORE_UC_N;
    case core_unicode::CAT_P:
        return CORE_UC_P;
    case core_unicode::CAT_WS:
        return CORE_UC_Z;
    case core_unicode::CAT_C:
    default:
        return CORE_UC_O;
    }
}

} // namespace core_bpe
