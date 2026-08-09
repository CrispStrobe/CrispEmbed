#!/usr/bin/env python
"""Generate src/core/unicode_lower.h — plain Unicode lowercase, no accent strip.

SigLIP's normalizer sequence starts with a bare `Lowercase`, and it must NOT
strip accents: `café Müller` -> `café müller`, keeping é and ü. That rules out
reusing core/bert_norm.h's table, which folds lowercase AND NFD accent removal
into one step by design.

Sampled per codepoint from `tokenizers.normalizers.Lowercase`, the same Rust
implementation SigLIP actually runs.

Usage:  python tools/gen_unicode_lower.py > src/core/unicode_lower.h
"""
import sys

from tokenizers import NormalizedString
from tokenizers.normalizers import Lowercase


def main() -> int:
    norm = Lowercase()

    def apply(s: str) -> str:
        ns = NormalizedString(s)
        norm.normalize(ns)
        return str(ns)

    mapping = {}
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        ch = chr(cp)
        out = apply(ch)
        if out != ch:
            mapping[cp] = out

    # Invariant: over ASCII this is exactly A-Z -> a-z and nothing else, which
    # is what lets the runtime keep an ASCII fast path.
    ascii_moved = {cp: mapping[cp] for cp in range(0x80) if cp in mapping}
    expected = {cp: chr(cp).lower() for cp in range(ord("A"), ord("Z") + 1)}
    if ascii_moved != expected:
        print(f"FATAL: ASCII lowercase is not plain A-Z: "
              f"{sorted(set(ascii_moved) ^ set(expected))[:8]}", file=sys.stderr)
        return 1
    for cp in range(0x80):
        mapping.pop(cp, None)

    multi_pool: list[int] = []
    multi_index: dict[str, int] = {}
    rows: list[tuple[int, int]] = []
    for cp in sorted(mapping):
        out = mapping[cp]
        if len(out) == 1:
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
    w("// src/core/unicode_lower.h — GENERATED, DO NOT EDIT BY HAND.\n")
    w("//\n")
    w("// Regenerate with:\n")
    w("//     python tools/gen_unicode_lower.py > src/core/unicode_lower.h\n")
    w("//\n")
    w(f"// Source of truth: HuggingFace tokenizers {tokenizers.__version__}\n")
    w("// `Lowercase` normalizer. Plain case folding with NO accent stripping —\n")
    w("// distinct from core/bert_norm.h, which deliberately folds both.\n")
    w("#pragma once\n\n")
    w("#include <cstdint>\n\n")
    w("namespace core_unicode_lower {\n\n")
    w("// payload & 0x80000000 -> index into MULTI (layout: len, cp, cp, ...)\n")
    w("// otherwise            -> single replacement codepoint\n")
    w("struct Row {\n    uint32_t cp;\n    uint32_t payload;\n};\n\n")
    w("inline constexpr uint32_t MULTI[] = {\n")
    for i in range(0, len(multi_pool), 12):
        w("    " + " ".join(f"0x{v:X}," for v in multi_pool[i:i + 12]) + "\n")
    w("};\n\n")
    w("inline constexpr Row ROWS[] = {\n")
    for i in range(0, len(rows), 4):
        w("    " + " ".join(f"{{0x{cp:X},0x{p:X}}}," for cp, p in rows[i:i + 4]) + "\n")
    w("};\n\n")
    w(f"inline constexpr int N_ROWS = {len(rows)};\n\n")
    w("} // namespace core_unicode_lower\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
