#!/usr/bin/env python3
"""Dump decoder-embedding per-stage HF reference for crispembed_diff.

Captures the token IDs, embedding lookup, every transformer block, final norm,
pre-normalization pooled vector, and final L2-normalized embedding from the
actual transformers model. This is deliberately not a NumPy reimplementation.

Usage:
    python tools/dump_decoder_embed_reference.py \
        --model Qwen/Qwen3-Embedding-0.6B \
        --text "The quick brown fox jumps over the lazy dog" \
        --output /tmp/qwen3-embed-ref.gguf
"""
import argparse
import json
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
    p.add_argument("--prompt-kind", choices=["raw", "query", "document"], default="raw",
                   help="Apply the prompt from config_sentence_transformers.json")
    args = p.parse_args()

    from transformers import AutoModel, AutoTokenizer

    print(f"Loading {args.model} ...")
    # trust_remote_code: BidirLM-Omni ships custom modeling code and otherwise blocks on an
    # interactive "run custom code? [y/N]" prompt in headless (Kaggle) runs. Qwen3-Embedding
    # doesn't need it but is unaffected.
    try:
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    except TypeError:
        # Some transformers builds crash in the fast TokenizersBackend for Qwen2-based
        # tokenizers (_patch_mistral_regex kwarg clash); the slow tokenizer avoids it.
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(args.model, dtype=torch.float32, trust_remote_code=True).eval()

    text = args.text
    model_dir = Path(args.model)
    if args.prompt_kind != "raw":
        cfg_path = model_dir / "config_sentence_transformers.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"prompt config not found: {cfg_path}")
        prompts = json.load(open(cfg_path)).get("prompts", {})
        text = prompts.get(args.prompt_kind, "") + text
    enc = tok(text, return_tensors="pt")
    ids = enc["input_ids"][0].tolist()
    print(f"text: {text!r}")
    print(f"tokens ({len(ids)}): {ids}")

    stages = {}
    backbone = model
    if hasattr(backbone, "model") and hasattr(backbone.model, "layers"):
        backbone = backbone.model

    def capture(name):
        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                output = output[0]
            stages[name] = output.detach().float().cpu().squeeze(0).numpy()
        return hook

    hooks = [backbone.embed_tokens.register_forward_hook(capture("post_embed"))]
    hooks.extend(layer.register_forward_hook(capture(f"layer_{i}"))
                 for i, layer in enumerate(backbone.layers))
    hooks.append(backbone.norm.register_forward_hook(capture("final_norm")))

    with torch.no_grad():
        out = model(**enc)
    for hook in hooks:
        hook.remove()
    last_hidden = out.last_hidden_state[0]          # [T, H]
    if args.pooling == "mean":
        mask = enc["attention_mask"][0].to(last_hidden.dtype).unsqueeze(-1)
        emb = (last_hidden * mask).sum(dim=0) / mask.sum().clamp_min(1)
    else:
        emb = last_hidden[-1]                        # last-token pooling
    stages["pooled_raw"] = emb.float().cpu().numpy().astype(np.float32)
    emb = emb / emb.norm()                           # L2 normalize
    print(f"pooling: {args.pooling}")
    emb = emb.float().cpu().numpy().astype(np.float32)
    print(f"final_embedding: {emb.shape}  range=[{emb.min():.4f},{emb.max():.4f}]")

    w = gguf.GGUFWriter(str(args.output), "decoder_embed_ref")
    w.add_string("ref.text", text)
    w.add_string("ref.prompt_kind", args.prompt_kind)
    w.add_string("ref.pooling", args.pooling)
    w.add_tensor("input_ids", np.asarray(ids, dtype=np.float32))
    for name, arr in stages.items():
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        w.add_tensor(name, arr)
        print(f"{name}: {arr.shape} norm={np.linalg.norm(arr):.6f}")
    w.add_tensor("final_embedding", emb)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"Wrote {args.output} ({Path(args.output).stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
