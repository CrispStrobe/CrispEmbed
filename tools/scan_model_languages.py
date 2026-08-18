#!/usr/bin/env python
"""Scan a GGUF's tokenizer vocabulary and report script-range coverage.

For OCR recognizers this IS the registry's `languages` field — the dictionary
genuinely gates which characters the model can emit. Zero coverage is
conclusive: PP-OCRv6's tiny recognizer has ZERO kana and silently fails on
Japanese.

For embedding/reranker models the scan is UNRELIABLE IN BOTH DIRECTIONS and
MUST NOT be surfaced as a --list-models column (E7 decision, 2026-08-17):
  - False positive: all-MiniLM-L6-v2 scans kana=188 yet bit-identical JA
  - False negative: jina-v5 scans kana=0 yet passes JA/AR/KO strongly
Root cause: BPE tokenizers encode non-Latin text as byte sequences (no
script code points in the tokens), while 30k WordPiece vocabs carry script
tokens the model can't functionally use. Use embed_language_eval.py instead.

Usage:
    python tools/scan_model_languages.py MODEL.gguf [MODEL.gguf ...]
    python tools/scan_model_languages.py --registry MODEL.gguf   # C literal
"""

import sys
import unicodedata

# (label, [(lo, hi), ...]) over Unicode code points. Ranges are deliberately
# coarse — this answers "can this model emit this script", not "which language".
SCRIPTS = [
    ("latin", [(0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F)]),
    ("digits", [(0x0030, 0x0039)]),
    ("cjk", [(0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0xF900, 0xFAFF)]),
    ("kana", [(0x3040, 0x309F), (0x30A0, 0x30FF)]),
    ("hangul", [(0xAC00, 0xD7AF), (0x1100, 0x11FF), (0x3130, 0x318F)]),
    ("cyrillic", [(0x0400, 0x04FF)]),
    ("greek", [(0x0370, 0x03FF)]),
    ("arabic", [(0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)]),
    ("hebrew", [(0x0590, 0x05FF)]),
    ("devanagari", [(0x0900, 0x097F)]),
    ("thai", [(0x0E00, 0x0E7F)]),
]


def read_tokens(path):
    from gguf import GGUFReader

    reader = GGUFReader(path)
    # OCR recognizers expose `tokenizer.tokens` (for Tesseract that is the
    # unicharset); embedding/reranker GGUFs use the llama.cpp-style
    # `tokenizer.ggml.tokens`. One scanner covers both lanes -- the script
    # question ("can this vocabulary even represent kana?") is identical.
    field = None
    embedding_lane = False
    for key in ("tokenizer.tokens", "tokenizer.ggml.tokens"):
        field = reader.fields.get(key)
        if field is not None:
            embedding_lane = key.endswith("ggml.tokens")
            break
    read_tokens.embedding_lane = embedding_lane
    if field is None:
        raise SystemExit(
            f"{path}: no tokenizer.tokens / tokenizer.ggml.tokens field "
            f"(not a recognizer or embedding GGUF?)")
    return [bytes(field.parts[i].tobytes()).decode("utf-8", "replace") for i in field.data]


EMBED_CAVEAT = (
    "  NOTE (embedding/reranker GGUF): for these models script coverage is\n"
    "  UNRELIABLE IN BOTH DIRECTIONS — do not use it to judge language support.\n"
    "  False positive: all-MiniLM-L6-v2 scans kana=188 yet returns BIT-IDENTICAL\n"
    "  embeddings for two different Japanese sentences (E1, 2026-08-08).\n"
    "  False negative: jina-v5-small scans kana=0 hangul=0 arabic=0 yet PASSES\n"
    "  all three language checks for JA/AR/KO with strong margins (E3, 2026-08-17).\n"
    "  Root cause: BPE tokenizers encode non-Latin scripts as byte sequences that\n"
    "  don't contain script-range code points, while 30k WordPiece vocabs contain\n"
    "  script tokens they can't use. Use tests/embed_language_eval.py instead;\n"
    "  see docs/LANGUAGES.md."
)


def scan(tokens):
    counts = {label: 0 for label, _ in SCRIPTS}
    for tok in tokens:
        for chunk in tok:
            cp = ord(chunk)
            for label, ranges in SCRIPTS:
                if any(lo <= cp <= hi for lo, hi in ranges):
                    counts[label] += 1
                    break
    return counts


def summary(counts):
    """The short, honest string the registry carries.

    Only scripts with real coverage are named. `digits` alone is not a script
    worth advertising, so it is folded into latin.
    """
    present = [label for label, _ in SCRIPTS if counts[label] > 0 and label != "digits"]
    return "+".join(present) if present else "none"


def main(argv):
    as_registry = "--registry" in argv
    paths = [a for a in argv if not a.startswith("--")]
    if not paths:
        raise SystemExit(__doc__)
    for path in paths:
        tokens = read_tokens(path)
        counts = scan(tokens)
        name = path.split("/")[-1]
        if as_registry:
            print(f'{name}: "{summary(counts)}"')
        else:
            detail = "  ".join(f"{k}={v}" for k, v in counts.items() if v)
            print(f"{name}\n  classes={len(tokens)}  {detail}\n  languages = {summary(counts)!r}")
            if getattr(read_tokens, "embedding_lane", False):
                print(EMBED_CAVEAT)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
