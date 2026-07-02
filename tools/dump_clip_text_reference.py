#!/usr/bin/env python3
"""Dump CLIP/SigLIP text-encoder reference embedding to GGUF for crispembed_diff.

Independent HF-AutoModel reference for the standalone clip_text engine
(crispembed_clip_text_encode). We compare the final pooled+projected text
embedding — cosine is scale-invariant, so an L2-norm mismatch between engine and
ref is harmless; a graph-scramble regression craters cos to ~0.

Usage:
    python tools/dump_clip_text_reference.py \
        --model openai/clip-vit-base-patch16 \
        --text "a photo of a fox" \
        --output /tmp/clip-text-ref.gguf
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
    # MUST match the text hardcoded in tests/test_clip_text_diff.cpp.
    p.add_argument("--text", default="a photo of a fox")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    from transformers import AutoModel, AutoTokenizer, AutoConfig

    print(f"Loading: {args.model}")
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.float32,
                                      trust_remote_code=True).eval()
    tok = AutoTokenizer.from_pretrained(args.model)
    is_clip = "clip" in config.model_type.lower()

    enc = tok([args.text], return_tensors="pt", padding=True)
    print(f"text: {args.text!r}  tokens: {enc['input_ids'].shape[1]}")

    with torch.no_grad():
        if is_clip:
            # CLIP: projected text features (what the standalone encoder returns).
            feat = model.get_text_features(**enc)
            # Also capture the PRE-projection pooled EOS hidden (after final_ln), to
            # disambiguate whether the C++ engine applies text_projection.
            tm = model.text_model(**enc)
            last = tm.last_hidden_state[0]                     # [T, H]
            eos_id = tok.eos_token_id
            ids = enc["input_ids"][0].tolist()
            eos_pos = max(i for i, t in enumerate(ids) if t == eos_id)
            pre = last[eos_pos]
        else:
            out = model(**enc) if hasattr(model, "text_model") is False else model.text_model(**enc)
            feat = getattr(out, "text_embeds", None)
            if feat is None:
                feat = getattr(out, "pooler_output", out.last_hidden_state.mean(dim=1))
            pre = feat[0]
        feat = feat / feat.norm(dim=-1, keepdim=True)
        pre = pre / pre.norm(dim=-1, keepdim=True)

    emb = feat[0].float().cpu().numpy().astype(np.float32)
    pre_emb = pre.float().cpu().numpy().astype(np.float32)
    print(f"final_embedding: {emb.shape}  range=[{emb.min():.4f},{emb.max():.4f}]")
    print(f"pre_proj:        {pre_emb.shape}")

    w = gguf.GGUFWriter(str(args.output), "clip_text_ref")
    w.add_string("ref.text", args.text)
    w.add_tensor("final_embedding", emb)
    w.add_tensor("pre_proj", pre_emb)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"Wrote {args.output} ({Path(args.output).stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
