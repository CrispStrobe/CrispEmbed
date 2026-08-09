#!/usr/bin/env python
"""Generate src/core/unicode_bert_norm.h from HuggingFace's OWN normalizer.

The table is the lowercase + strip-accents stage of
`tokenizers.normalizers.BertNormalizer(lowercase=True, strip_accents=None)`,
sampled per codepoint. It is generated, never hand-written, for two reasons:

  1. **The Rust (fast) normalizer is the authority, and it does NOT agree with
     Python's `BasicTokenizer`.** Every affected model ships a `tokenizer.json`,
     so `BertTokenizerFast` -> the Rust `BertNormalizer` is what actually runs.
     Rust's combining-mark predicate is built against an older Unicode than
     `unicodedata`, so 441 codepoints (NKO/Arabic/Bengali/Gujarati marks added
     later) are DROPPED by Python and KEPT by Rust. Generating from
     `unicodedata` would have silently shipped 441 divergences from the thing
     users actually run.
  2. A hand-written "strip the accent" table gets `Ø`/`Ł`/`Đ` wrong. Those have
     no canonical decomposition, so HF keeps them intact; only a real NFD tells
     you that.

Hangul is emitted as ARITHMETIC, not table rows: canonical decomposition of the
11172 precomposed syllables is the standard L/V/T formula, and none of the
resulting jamo are combining marks, so nothing is filtered. That removes 11172
of the 15889 non-identity codepoints from the table. The formula is asserted
against the oracle for every syllable at generation time.

Usage:  python tools/gen_unicode_bert_norm.py > src/core/unicode_bert_norm.h
"""
import sys

from tokenizers import NormalizedString
from tokenizers.normalizers import BertNormalizer

# clean_text / handle_chinese_chars are core_bert::pretokenize's job; this
# table is only the strip_accents + lowercase stage.
NORM = BertNormalizer(clean_text=False, handle_chinese_chars=False,
                      strip_accents=None, lowercase=True)

SBASE, LBASE, VBASE, TBASE = 0xAC00, 0x1100, 0x1161, 0x11A7
LCOUNT, VCOUNT, TCOUNT = 19, 21, 28
NCOUNT = VCOUNT * TCOUNT
SCOUNT = LCOUNT * NCOUNT


def norm(s: str) -> str:
    ns = NormalizedString(s)
    NORM.normalize(ns)
    return str(ns)


def hangul_decompose(cp: int) -> str:
    """Unicode canonical decomposition for a precomposed Hangul syllable."""
    idx = cp - SBASE
    l, v, t = LBASE + idx // NCOUNT, VBASE + (idx % NCOUNT) // TCOUNT, TBASE + idx % TCOUNT
    return chr(l) + chr(v) + (chr(t) if t != TBASE else "")


def main() -> int:
    # --- Build the full oracle map -------------------------------------
    mapping = {}
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        ch = chr(cp)
        out = norm(ch)
        if out != ch:
            mapping[cp] = out

    # --- Invariant: the Hangul formula reproduces the oracle exactly ----
    bad = [cp for cp in range(SBASE, SBASE + SCOUNT)
           if mapping.get(cp, chr(cp)) != hangul_decompose(cp)]
    if bad:
        print(f"FATAL: Hangul formula disagrees with the oracle on "
              f"{len(bad)} syllables, first U+{bad[0]:04X}", file=sys.stderr)
        return 1
    for cp in range(SBASE, SBASE + SCOUNT):
        mapping.pop(cp, None)

    # --- Invariant: ASCII is untouched ---------------------------------
    # This is what lets the runtime keep a fast ASCII path, and what makes the
    # default flip safe: pure-ASCII input cannot change.
    ascii_moved = {cp: mapping[cp] for cp in range(0x80) if cp in mapping}
    expected = {cp: chr(cp).lower() for cp in range(ord('A'), ord('Z') + 1)}
    if ascii_moved != expected:
        print(f"FATAL: ASCII mapping is not plain A-Z lowering: "
              f"{sorted(set(ascii_moved) ^ set(expected))[:8]}", file=sys.stderr)
        return 1
    for cp in range(0x80):
        mapping.pop(cp, None)

    # --- Pack ----------------------------------------------------------
    multi_pool: list[int] = []
    multi_index: dict[str, int] = {}
    rows: list[tuple[int, int]] = []
    for cp in sorted(mapping):
        out = mapping[cp]
        if len(out) == 0:
            payload = 0  # DELETE (codepoint 0 is never a replacement)
        elif len(out) == 1:
            payload = ord(out)
        else:
            if out not in multi_index:
                multi_index[out] = len(multi_pool)
                multi_pool.append(len(out))
                multi_pool.extend(ord(c) for c in out)
            payload = 0x80000000 | multi_index[out]
        rows.append((cp, payload))

    import tokenizers
    w = sys.stdout.write
    w("// src/core/unicode_bert_norm.h — GENERATED, DO NOT EDIT BY HAND.\n")
    w("//\n")
    w("// Regenerate with:\n")
    w("//     python tools/gen_unicode_bert_norm.py > src/core/unicode_bert_norm.h\n")
    w("//\n")
    w(f"// Source of truth: HuggingFace tokenizers {tokenizers.__version__}\n")
    w("// BertNormalizer(lowercase=true, strip_accents=null) — the RUST\n")
    w("// normalizer, which is what BertTokenizerFast actually runs. It does not\n")
    w("// agree with Python BasicTokenizer on 441 late-Unicode combining marks;\n")
    w("// see the generator's docstring for why Rust is the authority here.\n")
    w("//\n")
    w("// Hermetic goldens: tests/test_bert_pretokenize.cpp.\n")
    w("#pragma once\n\n")
    w("#include <cstdint>\n#include <string>\n\n")
    w("namespace core_unicode_norm {\n\n")
    w("// payload 0            -> delete the codepoint (it is a combining mark)\n")
    w("// payload & 0x80000000 -> index into MULTI (layout: len, cp, cp, ...)\n")
    w("// otherwise            -> single replacement codepoint\n")
    w("struct Row {\n    uint32_t cp;\n    uint32_t payload;\n};\n\n")
    w(f"inline constexpr uint32_t MULTI[] = {{\n")
    for i in range(0, len(multi_pool), 12):
        w("    " + " ".join(f"0x{v:X}," for v in multi_pool[i:i + 12]) + "\n")
    w("};\n\n")
    w(f"// {len(rows)} rows; the 11172 precomposed Hangul syllables are handled\n")
    w("// arithmetically below instead of being listed here.\n")
    w("inline constexpr Row ROWS[] = {\n")
    for i in range(0, len(rows), 4):
        w("    " + " ".join(f"{{0x{cp:X},0x{p:X}}}," for cp, p in rows[i:i + 4]) + "\n")
    w("};\n\n")
    w(f"inline constexpr int N_ROWS = {len(rows)};\n\n")
    w("""// Hangul canonical decomposition (Unicode 3.12 "Hangul Syllable
// Decomposition"). None of the produced jamo are combining marks, so the
// strip-accents filter never touches them.
inline constexpr uint32_t HANGUL_SBASE = 0xAC00, HANGUL_LBASE = 0x1100;
inline constexpr uint32_t HANGUL_VBASE = 0x1161, HANGUL_TBASE = 0x11A7;
inline constexpr uint32_t HANGUL_TCOUNT = 28, HANGUL_NCOUNT = 588, HANGUL_SCOUNT = 11172;

} // namespace core_unicode_norm
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
