#!/usr/bin/env python
"""E5 — WordPiece CJK tokenizer parity: CrispEmbed historical path vs HF reference.

Demonstrates the three-way difference on 30k uncased WordPiece models
(all-MiniLM-L6-v2, all-mpnet-base-v2) fed Japanese text:

  1. HF BasicTokenizer (do_lower_case=True, strip_accents=True):
     NFD accent strip → CJK ideograph split → whitespace split → punct split
     → WordPiece subword lookup. Produces DIFFERENT token sequences for
     different Japanese sentences (10 vs 9 tokens on the fixture pair).

  2. CrispEmbed historical per-byte path (bert_pretok_=false):
     ASCII isspace/ispunct only → entire Japanese string is ONE word →
     WordPiece maps the whole thing to [UNK]. BIT-IDENTICAL for any two
     Japanese inputs. This is the shipped behavior on all-MiniLM-L6-v2.

  3. CrispEmbed core_bert::pretokenize path (bert_pretok_=true, not currently
     enabled for these models):
     CJK ideograph split + Unicode punct/whitespace — but NO NFD accent strip.
     Would produce different sequences from HF because dakuten stays: が→が
     (HF strips to か), で→で (HF strips to て).

Impact: confined to English-only vocabularies fed CJK (garbage either way).
Every multilingual embedder is SentencePiece/XLM-R, unaffected. This script
documents the gap; the C++ guard in test_bert_pretokenize.cpp pins it.

Usage:
    python tests/wordpiece_cjk_parity.py
"""
import sys

try:
    from transformers import AutoTokenizer
except ImportError:
    print("SKIP: transformers not installed", file=sys.stderr)
    sys.exit(0)

FIXTURE = [
    ("ja_cat_a",   "猫がソファの上で眠っている。"),
    ("ja_cat_b",   "ソファーで猫が寝ています。"),
    ("ja_weather", "明日の東京の天気は雨でしょう。"),
]

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def hf_breakdown(tok, text):
    """Return (normalized, pre_tokens, tokens, ids) from HF tokenizer."""
    ns = tok.backend_tokenizer.normalizer.normalize_str(text)
    pts = [w for w, _ in tok.backend_tokenizer.pre_tokenizer.pre_tokenize_str(ns)]
    tokens = tok.tokenize(text)
    ids = tok.encode(text, add_special_tokens=True)
    return ns, pts, tokens, ids


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    print(f"Model: {MODEL}")
    print(f"Vocab size: {tok.vocab_size}")
    print(f"do_lower_case: {tok.do_lower_case}")
    print()

    print("=" * 72)
    print("HF REFERENCE TOKENIZATION (NFD accent-strip + CJK split)")
    print("=" * 72)

    for label, text in FIXTURE:
        ns, pts, tokens, ids = hf_breakdown(tok, text)
        unk_n = sum(1 for t in tokens if t == "[UNK]")
        print(f"\n[{label}] {text}")
        print(f"  normalized:  {repr(ns)}")
        print(f"  pre-tokens:  {pts}")
        print(f"  tokens ({len(tokens)}): {tokens}")
        print(f"  ids:         {ids}")
        print(f"  UNK ratio:   {unk_n}/{len(tokens)} = {unk_n/len(tokens):.0%}")

    # Show the key finding: the two cat sentences produce DIFFERENT sequences
    _, _, tok_a, ids_a = hf_breakdown(tok, FIXTURE[0][1])
    _, _, tok_b, ids_b = hf_breakdown(tok, FIXTURE[1][1])
    print(f"\n--- Key finding ---")
    print(f"ja_cat_a tokens: {tok_a}")
    print(f"ja_cat_b tokens: {tok_b}")
    print(f"Sequences identical: {tok_a == tok_b}")
    print(f"IDs identical:       {ids_a == ids_b}")

    print()
    print("=" * 72)
    print("CRISPEMBED HISTORICAL PATH (per-byte ASCII isspace/ispunct)")
    print("=" * 72)
    print()
    print("The historical split_words() path treats bytes individually:")
    print("  - std::isspace(c): only ASCII whitespace (0x09-0x0D, 0x20)")
    print("  - std::ispunct(c): only ASCII punctuation (33-47, 58-64, etc.)")
    print("  - All multi-byte UTF-8 chars: treated as letters, stay glued")
    print()
    print("Result: entire Japanese string → one word → WordPiece → [UNK]")
    print("Both sentences → [CLS] [UNK] [SEP] → BIT-IDENTICAL embeddings")
    print()

    print("=" * 72)
    print("CORE_BERT::PRETOKENIZE PATH (CJK split, no NFD accent strip)")
    print("=" * 72)
    print()
    print("If bert_pretok_ were enabled for these models, the CJK ideographs")
    print("would split correctly (猫→猫, 上→上, 眠→眠) and Unicode punct (。)")
    print("would isolate. But kana dakuten would NOT be stripped:")
    print("  - が stays が (HF strips to か)")
    print("  - で stays で (HF strips to て)")
    print("  - ぞ stays ぞ (HF strips to そ)")
    print()
    print("This produces different pre-tokens from HF, but crucially the two")
    print("cat sentences would STILL produce different WordPiece sequences")
    print("(because the kana runs differ), just not HF-identical ones.")
    print()

    # Show the NFD accent-stripping effect on kana
    print("=" * 72)
    print("NFD ACCENT STRIPPING ON KANA (the dakuten difference)")
    print("=" * 72)
    import unicodedata
    kana_pairs = [
        ("が", "か + ゙"),
        ("で", "て + ゙"),
        ("ぞ", "そ + ゙"),
        ("ば", "は + ゙"),
        ("ぱ", "は + ゚"),
    ]
    print()
    for kana, desc in kana_pairs:
        nfd = unicodedata.normalize("NFD", kana)
        stripped = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
        nfd_repr = " + ".join(f"U+{ord(c):04X} ({unicodedata.name(c, '?')})" for c in nfd)
        print(f"  {kana} → NFD: {nfd_repr}")
        print(f"       → strip Mn: {stripped}")
    print()
    print("HF's BasicTokenizer with do_lower_case=True applies this NFD+strip,")
    print("which is why kana with dakuten become their base forms. CrispEmbed's")
    print("core_bert::pretokenize does NOT do this (it was built for LaBSE, which")
    print("is cased and does not strip accents).")


if __name__ == "__main__":
    main()
