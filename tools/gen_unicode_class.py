"""Emit src/core/unicode_class.h — non-ASCII codepoint class ranges.

Classes: L letter, M mark, N number, P punctuation-or-symbol, Z whitespace.
Anything not listed defaults to L (letter), which is what every non-listed
non-ASCII codepoint is in practice (CJK, Cyrillic, Greek, Arabic, Hangul, ...).
"""
import sys
import unicodedata

# Rust regex `\s` == Unicode White_Space property. Non-ASCII members:
WS = [(0x85, 0x85), (0xA0, 0xA0), (0x1680, 0x1680), (0x2000, 0x200A),
      (0x2028, 0x2029), (0x202F, 0x202F), (0x205F, 0x205F), (0x3000, 0x3000)]


def cls_of(cp):
    for lo, hi in WS:
        if lo <= cp <= hi:
            return 'Z'
    c = unicodedata.category(chr(cp))
    if c[0] == 'M':
        return 'M'
    if c[0] == 'N':
        return 'N'
    if c[0] in ('P', 'S'):
        return 'P'
    return None  # defaults to L


rows = []
start = None
cur = None
for cp in range(0x80, 0x110000):
    if 0xD800 <= cp <= 0xDFFF:
        c = None
    else:
        c = cls_of(cp)
    if c != cur:
        if cur is not None:
            rows.append((start, cp - 1, cur))
        start, cur = cp, c
if cur is not None:
    rows.append((start, 0x10FFFF, cur))

CLS = {'M': 'CORE_UC_M', 'N': 'CORE_UC_N', 'P': 'CORE_UC_P', 'Z': 'CORE_UC_Z'}
out = []
out.append('''// src/core/unicode_class.h — non-ASCII codepoint class table.
//
// GENERATED. Regenerate with `python tools/gen_unicode_class.py
// src/core/unicode_class.h && tools/format.sh --fix` (built with Python %s,
// unicodedata %s). Do not hand-edit the table.
//
// The byte-level BPE pre-tokenizers in core/bpe.h transcribe regexes that
// use the Unicode general categories \\p{L} \\p{M} \\p{N} \\p{P} \\p{S} and
// the \\s (White_Space) class. Getting those wrong is not cosmetic: treating
// every non-ASCII byte as a letter (the approximation this table replaces)
// merges German typographic quotes into the adjacent word — HuggingFace
// splits `sagte \\u201eHallo\\u201c heute` into 5 pre-tokens, the byte>=0x80
// approximation produced 3.
//
// \\p{P} and \\p{S} are stored as one class: every regex in core/bpe.h uses
// them only as the union `[\\p{P}\\p{S}]` (or its complement), never apart.
// Anything not in the table is a letter, which is correct for the scripts
// (CJK, Cyrillic, Greek, Arabic, Hangul, Devanagari, ...) that dominate the
// unlisted space.

#pragma once

#include <cstddef>
#include <cstdint>

namespace core_bpe {

// Codepoint classes. Ordered so CORE_UC_L stays the zero/default answer.
enum core_uc_class : uint8_t {
    CORE_UC_L = 0, // \\p{L}  letter (also the fallback for unlisted codepoints)
    CORE_UC_M = 1, // \\p{M}  combining mark
    CORE_UC_N = 2, // \\p{N}  number
    CORE_UC_P = 3, // \\p{P} | \\p{S}  punctuation or symbol
    CORE_UC_Z = 4, // \\s     White_Space
    CORE_UC_O = 5, // none of the above (ASCII control characters)
};

struct core_uc_range {
    uint32_t lo;
    uint32_t hi;
    uint8_t cls;
};

// Sorted, non-overlapping, non-ASCII only (lo >= 0x80).
inline const core_uc_range * core_uc_table(size_t & n) {
    static const core_uc_range t[] = {''' % (sys.version.split()[0], unicodedata.unidata_version))

line = '        '
for lo, hi, c in rows:
    piece = '{ 0x%X, 0x%X, %s },' % (lo, hi, CLS[c])
    if len(line) + len(piece) > 118:
        out.append(line.rstrip())
        line = '        '
    line += piece + ' '
out.append(line.rstrip())
out.append('''    };
    n = sizeof(t) / sizeof(t[0]);
    return t;
}

// Class of one Unicode codepoint. ASCII is decided inline; everything else
// binary-searches the table above and defaults to letter.
inline core_uc_class core_uc_classify(uint32_t cp) {
    if (cp < 0x80) {
        if (cp == 0x20 || (cp >= 0x09 && cp <= 0x0D)) return CORE_UC_Z;
        if (cp >= '0' && cp <= '9') return CORE_UC_N;
        if ((cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z')) return CORE_UC_L;
        // The 32 ASCII punctuation/symbol characters: !"#$%&'()*+,-./ :;<=>?@
        // [\\]^_` {|}~ — all \\p{P} or \\p{S}.
        if ((cp >= 0x21 && cp <= 0x2F) || (cp >= 0x3A && cp <= 0x40) || (cp >= 0x5B && cp <= 0x60) ||
            (cp >= 0x7B && cp <= 0x7E))
            return CORE_UC_P;
        return CORE_UC_O; // control characters
    }
    size_t n = 0;
    const core_uc_range * t = core_uc_table(n);
    size_t lo = 0, hi = n;
    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        if (cp < t[mid].lo)
            hi = mid;
        else if (cp > t[mid].hi)
            lo = mid + 1;
        else
            return (core_uc_class)t[mid].cls;
    }
    return CORE_UC_L;
}

} // namespace core_bpe''')
open(sys.argv[1], 'w').write('\n'.join(out) + '\n')
print('ranges:', len(rows))
