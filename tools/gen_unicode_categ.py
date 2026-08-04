#!/usr/bin/env python
"""Generate src/core/unicode_categ.h — the Unicode general-category table the
regex pre-tokenizers in src/core/bpe.h branch on.

    python tools/gen_unicode_categ.py > src/core/unicode_categ.h
    tools/format.sh --fix src/core/unicode_categ.h
"""
import sys
import unicodedata

# CAT_LO is the DEFAULT answer, so caseless letters (CJK, Hangul, Arabic, ...)
# and unassigned codepoints cost no table rows.
CAT_LO, CAT_LU, CAT_LL, CAT_M, CAT_N, CAT_P, CAT_WS, CAT_C = 0, 1, 2, 3, 4, 5, 6, 7


def classify(cp):
    if 0xD800 <= cp < 0xE000:
        return CAT_C
    c = unicodedata.category(chr(cp))
    if c in ("Lu", "Lt"):
        return CAT_LU
    if c == "Ll":
        return CAT_LL
    if c in ("Lm", "Lo"):
        return CAT_LO
    if c[0] == "M":
        return CAT_M
    if c[0] == "N":
        return CAT_N
    if c[0] in ("P", "S"):
        return CAT_P
    if chr(cp).isspace() or cp == 0x85:
        return CAT_WS
    return CAT_C


def ranges():
    out, prev, start = [], CAT_LO, 0
    for cp in range(0x110000):
        c = classify(cp)
        if c != prev:
            if prev != CAT_LO:
                out.append((start, cp - 1, prev))
            start, prev = cp, c
    if prev != CAT_LO:
        out.append((start, 0x10FFFF, prev))
    return [r for r in out if r[1] >= 0x80]  # ASCII has a fast path


NAMES = {CAT_LU: "CAT_LU", CAT_LL: "CAT_LL", CAT_M: "CAT_M", CAT_N: "CAT_N",
         CAT_P: "CAT_P", CAT_WS: "CAT_WS", CAT_C: "CAT_C"}

HEADER = '''// src/core/unicode_categ.h -- GENERATED, do not edit by hand.
//
// Unicode general-category lookup for the regex pre-tokenizers in core/bpe.h.
// Regenerate with `python tools/gen_unicode_categ.py > src/core/unicode_categ.h`
// (Python unicodedata %s), then `tools/format.sh --fix`.
//
// WHY A REAL TABLE. The repo's historical shortcut -- "any byte >= 0x80 is a
// letter" -- is wrong in two independent ways, one of which no other table in
// the tree can fix:
//   * it swallows non-ASCII PUNCTUATION into the adjacent word, so German
//     typographic quotes and dashes mis-split;
//   * it erases LETTER CASE. The o200k_base split (granite-embedding-97m-
//     multilingual-r2) branches on `[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]` versus
//     `[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]` in two of its seven alternatives, and with
//     every non-ASCII letter in BOTH classes an all-caps German word splits
//     after its umlaut ("AERGER" with an umlaut -> "AE" + "RGER").
//
// ⚠ MERGE NOTE: branch `feat/tokenize-simple-audit` adds `core/unicode_class.h`,
// a second generated table for the same job. That one has no case information,
// so it cannot serve the o200k split; this one is a strict SUPERSET of it and
// maps 1:1 onto its enum:
//     CORE_UC_L <- CAT_LU | CAT_LL | CAT_LO      CORE_UC_P <- CAT_P
//     CORE_UC_M <- CAT_M   CORE_UC_N <- CAT_N    CORE_UC_Z <- CAT_WS
//     CORE_UC_O <- CAT_C
// On merge, keep THIS table, express `core_uc_class` as that mapping, and drop
// the other generator. Both default unlisted codepoints to "letter", so the
// approximation for unassigned codepoints is identical in either direction.

#pragma once

#include <cstddef>
#include <cstdint>

namespace core_unicode {

// General category, collapsed to what the pre-tokenizer regexes branch on.
// CAT_LO is the DEFAULT for any codepoint not in the table below.
enum categ : uint8_t {
    CAT_LO = 0, // Lm | Lo   -- caseless letters; also the unlisted fallback
    CAT_LU = 1, // Lu | Lt   -- uppercase / titlecase letters
    CAT_LL = 2, // Ll        -- lowercase letters
    CAT_M  = 3, // Mn|Mc|Me  -- combining marks
    CAT_N  = 4, // Nd|Nl|No  -- numbers
    CAT_P  = 5, // P* | S*   -- punctuation or symbol (never needed apart)
    CAT_WS = 6, // White_Space
    CAT_C  = 7, // controls, format characters, surrogates
};

struct range {
    uint32_t lo;
    uint32_t hi;
    uint8_t  cat;
};

// Sorted, disjoint, non-ASCII only. Gaps mean CAT_LO.
static const range k_ranges[] = {
'''

FOOTER = '''};
static const size_t k_nranges = sizeof(k_ranges) / sizeof(k_ranges[0]);

// Category of one codepoint. ASCII is answered without touching the table.
inline uint8_t category(uint32_t cp) {
    if (cp < 0x80) {
        if (cp >= 'a' && cp <= 'z') return CAT_LL;
        if (cp >= 'A' && cp <= 'Z') return CAT_LU;
        if (cp >= '0' && cp <= '9') return CAT_N;
        if (cp == ' ' || (cp >= 0x09 && cp <= 0x0D)) return CAT_WS;
        if (cp < 0x20 || cp == 0x7F) return CAT_C;
        return CAT_P;
    }
    size_t lo = 0, hi = k_nranges;
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (cp < k_ranges[mid].lo)
            hi = mid;
        else if (cp > k_ranges[mid].hi)
            lo = mid + 1;
        else
            return k_ranges[mid].cat;
    }
    return CAT_LO; // unlisted == caseless letter
}

// \\p{L} = Lu | Ll | Lt | Lm | Lo. Marks and numbers are NOT letters.
inline bool is_letter(uint8_t c) {
    return c == CAT_LU || c == CAT_LL || c == CAT_LO;
}
inline bool is_number(uint8_t c) {
    return c == CAT_N;
}
inline bool is_space(uint8_t c) {
    return c == CAT_WS;
}

// Decode one UTF-8 codepoint at s[i]; advances `i`. Malformed input yields the
// raw byte as a codepoint and advances by one so callers always progress.
inline uint32_t utf8_next(const char * s, size_t n, size_t & i) {
    const unsigned char c = (unsigned char) s[i];
    size_t len = 1;
    uint32_t cp = c;
    if (c >= 0xF0 && i + 3 < n) {
        len = 4;
        cp = ((uint32_t) (c & 0x07) << 18) | ((uint32_t) (s[i + 1] & 0x3F) << 12) |
             ((uint32_t) (s[i + 2] & 0x3F) << 6) | (uint32_t) (s[i + 3] & 0x3F);
    } else if (c >= 0xE0 && i + 2 < n) {
        len = 3;
        cp = ((uint32_t) (c & 0x0F) << 12) | ((uint32_t) (s[i + 1] & 0x3F) << 6) | (uint32_t) (s[i + 2] & 0x3F);
    } else if (c >= 0xC0 && i + 1 < n) {
        len = 2;
        cp = ((uint32_t) (c & 0x1F) << 6) | (uint32_t) (s[i + 1] & 0x3F);
    }
    i += len;
    return cp;
}

} // namespace core_unicode'''


def main():
    rs = ranges()
    sys.stderr.write("ranges: %d\n" % len(rs))
    body = "\n".join("    { 0x%04X, 0x%04X, %s }," % (lo, hi, NAMES[c]) for lo, hi, c in rs)
    sys.stdout.write(HEADER % unicodedata.unidata_version + body + "\n" + FOOTER + "\n")


if __name__ == "__main__":
    main()
