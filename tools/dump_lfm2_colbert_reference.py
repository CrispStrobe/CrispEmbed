#!/usr/bin/env python3
"""Dump LFM2.5-ColBERT reference for crispembed_diff parity testing.

Uses AutoModel (the real Lfm2BidirectionalModel) for the backbone — the previous
hand-written per-layer forward diverged from the actual model (hidden_states cos
~ -0.54 vs the C++ engine, which shares the verified lfm2-embedding backbone). We
capture the post-final-norm hidden (embedding_norm output == the engine's
pre-projection `cur`), then apply the 1_Dense ColBERT head + per-token L2-norm.

Usage:
    python tools/dump_lfm2_colbert_reference.py \
        --model LiquidAI/LFM2.5-ColBERT-350M \
        --output lfm2-colbert-ref.gguf
"""
import argparse, os, sys
from pathlib import Path
import numpy as np

try:
    import gguf
except ImportError:
    sys.exit("pip install gguf")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF id or local dir of the ColBERT model")
    p.add_argument("--output", "-o", required=True)
    p.add_argument("--text", default="query: The quick brown fox")
    args = p.parse_args()

    import torch
    from safetensors import safe_open
    from transformers import AutoTokenizer, AutoModel

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, dtype=torch.float32).eval()

    enc = tok(args.text, return_tensors="pt")
    input_ids = enc["input_ids"]
    T = input_ids.shape[1]
    print(f"Input: {T} tokens  ids={input_ids[0].tolist()}")

    # Capture embedding_norm output — exactly the engine's post-final-norm hidden.
    grab = {}
    h = model.embedding_norm.register_forward_hook(
        lambda m, i, o: grab.__setitem__("hidden", (o[0] if isinstance(o, tuple) else o).detach().float()))
    with torch.no_grad():
        out = model(**enc)
    h.remove()
    hidden = grab.get("hidden")
    if hidden is None:
        hidden = out.last_hidden_state.float()
    hidden = hidden.squeeze(0)  # [T, H]
    print(f"hidden (post embedding_norm): {tuple(hidden.shape)} range=[{hidden.min():.3f},{hidden.max():.3f}]")

    # ColBERT head: 1_Dense linear (no bias) + per-token L2 normalize.
    dense_dir = args.model if os.path.isdir(args.model) else None
    if dense_dir is None:
        from huggingface_hub import snapshot_download
        dense_dir = snapshot_download(args.model, allow_patterns=["1_Dense/*"])
    with safe_open(os.path.join(dense_dir, "1_Dense", "model.safetensors"), framework="pt") as f:
        proj_w = f.get_tensor("linear.weight").float()  # [128, H]
    projected = torch.nn.functional.normalize(hidden @ proj_w.T, p=2, dim=-1)  # [T, 128]
    print(f"colbert_output: {tuple(projected.shape)} range=[{projected.min():.3f},{projected.max():.3f}]")

    w = gguf.GGUFWriter(args.output, "lfm2-colbert-ref")
    w.add_uint32("ref.n_tokens", T)
    w.add_uint32("ref.colbert_dim", proj_w.shape[0])
    w.add_tensor("hidden_states", hidden.numpy().astype(np.float32),
                 raw_dtype=gguf.GGMLQuantizationType.F32)
    w.add_tensor("colbert_output", projected.numpy().astype(np.float32),
                 raw_dtype=gguf.GGMLQuantizationType.F32)
    w.add_tensor("input_ids", input_ids.squeeze(0).numpy().astype(np.float32),
                 raw_dtype=gguf.GGMLQuantizationType.F32)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"\nReference: {args.output} ({Path(args.output).stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
