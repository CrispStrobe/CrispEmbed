#!/usr/bin/env python
"""Generate src/core/unicode_spm_norm.h from HuggingFace's OWN normalizer.

The table is the `Precompiled` (sentencepiece `nmt_nfkc` charsmap) stage that
every XLM-R-family Unigram tokenizer declares, sampled per codepoint from the
Rust normalizer that actually runs.

WHY ONLY THE Precompiled COMPONENT. The embedders declare
`Precompiled + Replace`; the trailing Replace is the " " -> "▁" substitution,
which SentencePieceTokenizer already does itself. Baking it into this table
would double-apply it. SigLIP wraps the same charsmap in
`Lowercase + Replace + Replace + Strip + Precompiled + Replace` — again, only
the Precompiled part belongs here.

WHY IT IS SAFE TO SHARE ONE TABLE. The charsmap is byte-identical (sha256
ce10d747...) across every SentencePiece model this repo loads that has one:
multilingual-e5-small/base, bge-m3, granite-embedding-107m/278m-multilingual,
arctic-embed-m-v2, and google/siglip-base-patch16-{224,384}. They agree on all
65536 BMP codepoints. (The shipped GLiNER GGUF is an LFM2 BPE model with no
normalizer at all, so it never reaches this path.)

Usage:  python tools/gen_unicode_spm_norm.py > src/core/unicode_spm_norm.h
"""
import base64
import json
import sys

from huggingface_hub import hf_hub_download
from tokenizers import NormalizedString
from tokenizers.normalizers import Precompiled

# Any of the models above would do; they carry the identical blob.
SOURCE_REPO = "intfloat/multilingual-e5-small"


def load_precompiled():
    tj = json.load(open(hf_hub_download(SOURCE_REPO, "tokenizer.json")))
    n = tj["normalizer"]
    seq = n["normalizers"] if n.get("type") == "Sequence" else [n]
    for x in seq:
        if x.get("type") == "Precompiled":
            return Precompiled(base64.b64decode(x["precompiled_charsmap"]))
    raise SystemExit("FATAL: no Precompiled normalizer in " + SOURCE_REPO)


def main() -> int:
    norm = load_precompiled()

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

    # --- Invariant: PRINTABLE ASCII is untouched ---------------------------
    # This is what makes enabling the normalizer by default safe for English
    # text. It is *printable* ASCII, not all of ASCII: the charsmap also folds
    # \t \n \f \r to a plain space and DELETES the remaining C0 controls and
    # DEL. Both are harmless for a tokenizer that splits on whitespace anyway,
    # and both are what HF does — but assert the exact shape rather than
    # waving it through, because "ASCII is untouched" would be a false claim.
    printable_moved = {cp: mapping[cp] for cp in range(0x20, 0x7F) if cp in mapping}
    if printable_moved:
        print(f"FATAL: the charsmap alters PRINTABLE ASCII: "
              f"{ {hex(k): v for k, v in list(printable_moved.items())[:8]} }", file=sys.stderr)
        return 1
    for cp in (0x09, 0x0A, 0x0C, 0x0D):
        if mapping.get(cp) != " ":
            print(f"FATAL: expected U+{cp:04X} -> ' ', got {mapping.get(cp)!r}", file=sys.stderr)
            return 1
    for cp in (0x01, 0x08, 0x0B, 0x1F, 0x7F):
        if mapping.get(cp) != "":
            print(f"FATAL: expected U+{cp:04X} deleted, got {mapping.get(cp)!r}", file=sys.stderr)
            return 1

    multi_pool: list[int] = []
    multi_index: dict[str, int] = {}
    rows: list[tuple[int, int]] = []
    for cp in sorted(mapping):
        out = mapping[cp]
        if len(out) == 0:
            payload = 0
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
    w("// src/core/unicode_spm_norm.h — GENERATED, DO NOT EDIT BY HAND.\n")
    w("//\n")
    w("// Regenerate with:\n")
    w("//     python tools/gen_unicode_spm_norm.py > src/core/unicode_spm_norm.h\n")
    w("//\n")
    w(f"// Source of truth: HuggingFace tokenizers {tokenizers.__version__}, the\n")
    w(f"// `Precompiled` (sentencepiece nmt_nfkc charsmap) normalizer carried by\n")
    w(f"// {SOURCE_REPO} — byte-identical across every SentencePiece model this\n")
    w("// repo loads that has one (see the generator docstring).\n")
    w("//\n")
    w("// Hermetic goldens: tests/test_spm_norm.cpp.\n")
    w("#pragma once\n\n")
    w("#include <cstdint>\n\n")
    w("namespace core_unicode_spm {\n\n")
    w("// payload 0            -> delete the codepoint\n")
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
    w("} // namespace core_unicode_spm\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
