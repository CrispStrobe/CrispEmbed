#!/usr/bin/env python3
"""Generate a HuggingFace CLIP tokenizer reference for test_clip_tokenizer_parity.

Dumps the CLIP text vocab + merges and the expected token IDs for a fixed set
of probe strings, so the C++ harness can replay them and assert exact parity
with `CLIPTokenizerFast` (the ground truth this fix was validated against).

    python tests/gen_clip_tokenizer_reference.py \
        --model openai/clip-vit-base-patch32 \
        --out-dir /tmp/clip-tok-ref

Writes vocab.tsv (one token per id, in id order), merges.tsv (one "a b" merge
per line, rank order) and expected.tsv ("<text>\\t<comma-separated ids>").
Any CLIP checkpoint works — they share the same 49408-entry BPE vocab.
"""
import argparse
from pathlib import Path


# Probe strings covering the tokenizer's tricky cases. Keep in sync with the
# table in test_clip_tokenizer_parity.cpp is NOT required — the harness reads
# expected.tsv — but do keep them representative.
PROBES = [
    "hello",
    "a photo of a fox",  # the handover's decisive example (49406 320 1125 539 320 3240 49407)
    "a photo of a cat",
    "Hello World!",  # lowercasing + punctuation run
    "don't stop",  # contraction split
    "CVPR2024 paper",  # letter-run merges + per-digit split
    "naïve café",  # byte-level non-ASCII
    "hello  world",  # whitespace collapse
    "  leading and trailing  ",  # strip
    "3.14 is pi",
    "e-mail: test@example.com",
    "",  # empty -> BOS/EOS only
    "the QUICK brown Fox",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    import json

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    tj_path = Path(tok.name_or_path) / "tokenizer.json"
    if tj_path.exists():
        model = json.load(open(tj_path))["model"]
        vocab, merges = model["vocab"], model["merges"]
    else:
        vocab = tok.get_vocab()
        merges = [f"{a} {b}" for (a, b) in sorted(tok.bpe_ranks, key=tok.bpe_ranks.get)]

    vsize = max(vocab.values()) + 1
    tokens = [""] * vsize
    for t, i in vocab.items():
        if 0 <= i < vsize:
            tokens[i] = t
    mlist = [m if isinstance(m, str) else f"{m[0]} {m[1]}" for m in merges]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "vocab.tsv", "w") as f:
        for t in tokens:
            f.write(t.replace("\n", "\\n") + "\n")
    with open(out / "merges.tsv", "w") as f:
        for m in mlist:
            f.write(m + "\n")
    with open(out / "expected.tsv", "w") as f:
        for t in PROBES:
            ids = tok(t, add_special_tokens=True)["input_ids"]
            esc = t.replace("\t", "\\t").replace("\n", "\\n")
            f.write(esc + "\t" + ",".join(map(str, ids)) + "\n")

    print(f"bos={tok.bos_token_id} eos={tok.eos_token_id} "
          f"vocab={len(tokens)} merges={len(mlist)} probes={len(PROBES)} -> {out}")


if __name__ == "__main__":
    main()
