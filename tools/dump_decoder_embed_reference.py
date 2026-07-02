#!/usr/bin/env python3
"""Dump decoder-embedding (Qwen3/Gemma3-Embedding) reference for crispembed_diff.

Independent HF reference for the decoder_embed engine (crispembed_encode on a decoder
GGUF). decoder_embed does last-token pooling + L2 normalize; we replicate that on the
raw text so the comparison matches what crispembed_encode computes. cosine is
scale-invariant, so an L2-norm mismatch is harmless; a graph scramble craters cos to ~0.

Usage:
    python tools/dump_decoder_embed_reference.py \
        --model Qwen/Qwen3-Embedding-0.6B \
        --text "The quick brown fox jumps over the lazy dog" \
        --output /tmp/qwen3-embed-ref.gguf
"""
import argparse
import sys
from pathlib import Path

import gguf
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    # MUST match the text hardcoded in tests/test_decoder_embed_diff.cpp.
    p.add_argument("--text", default="The quick brown fox jumps over the lazy dog")
    p.add_argument("--output", required=True)
    # Qwen3-Embedding = last-token; BidirLM-Omni (bidirectional) = mean.
    p.add_argument("--pooling", choices=["last", "mean"], default="last")
    args = p.parse_args()

    from transformers import AutoModel, AutoTokenizer

    print(f"Loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, dtype=torch.float32).eval()

    # Qwen3-Embedding pools the LAST token; the tokenizer appends the EOS the model
    # pools. Encode the raw text (no instruction — crispembed_encode gets a plain string).
    enc = tok(args.text, return_tensors="pt")
    ids = enc["input_ids"][0].tolist()
    print(f"tokens: {len(ids)}  last id: {ids[-1]}")

    with torch.no_grad():
        out = model(**enc)
    last_hidden = out.last_hidden_state[0]          # [T, H]
    if args.pooling == "mean":
        emb = last_hidden.mean(dim=0)
    else:
        emb = last_hidden[-1]                        # last-token pooling
    emb = emb / emb.norm()                           # L2 normalize
    print(f"pooling: {args.pooling}")
    emb = emb.float().cpu().numpy().astype(np.float32)
    print(f"final_embedding: {emb.shape}  range=[{emb.min():.4f},{emb.max():.4f}]")

    w = gguf.GGUFWriter(str(args.output), "decoder_embed_ref")
    w.add_string("ref.text", args.text)
    w.add_tensor("final_embedding", emb)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"Wrote {args.output} ({Path(args.output).stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
