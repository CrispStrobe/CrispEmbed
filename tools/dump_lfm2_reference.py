#!/usr/bin/env python3
"""Dump LFM2.5-Embedding per-layer reference activations for crispembed_diff parity testing.

Captures:
  post_embed    after embedding lookup (before any layers)
  layer_N       output of LFM2 layer N  (0 .. n_layers-1)
  final_norm    after embedding_norm RMSNorm
  cls_raw       position-0 (CLS) vector before L2 normalisation
  cls_norm      L2-normalised CLS vector (final model output)

Usage:
  PYTHONPATH=python HF_HOME=... TRANSFORMERS_OFFLINE=1 \\
  python tools/dump_lfm2_reference.py \\
      --model /tmp/lfm2-embed-model \\
      --text "hello world" \\
      --output /tmp/lfm2-ref.gguf
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "ggml" / "scripts"))
try:
    import gguf
except ImportError:
    print("pip install gguf", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="Local directory for LFM2.5-Embedding-350M")
    parser.add_argument("--text", default="hello world",
                        help="Input text (without prefix)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # Load tokenizer + model via AutoModel (needs trust_remote_code for
    # Lfm2BidirectionalModel class registered in modeling_lfm2_bidirectional.py)
    from transformers import AutoTokenizer, AutoModel
    print(f"Loading tokenizer from {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model)

    print(f"Loading model from {args.model} ...")
    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=torch.float32,
    )
    model.eval()

    # Tokenise
    enc = tok(args.text, return_tensors="pt")
    input_ids = enc["input_ids"][0]
    print(f"Text: {args.text!r}")
    print(f"Token IDs: {input_ids.tolist()}")

    # ----------------------------------------------------------------
    # Forward hooks
    # ----------------------------------------------------------------
    intermediates = {}

    def hook_embed(module, inp, out):
        intermediates["post_embed"] = out.detach().float().cpu().squeeze(0).numpy()

    # The backbone is model itself (Lfm2BidirectionalModel / Lfm2Model).
    # embed_tokens + layers + embedding_norm live directly on it.
    backbone = model

    embed_hook = backbone.embed_tokens.register_forward_hook(hook_embed)

    layer_hooks = []
    for i, layer in enumerate(backbone.layers):
        def make_hook(idx):
            def hook(module, inp, out):
                if isinstance(out, tuple):
                    out = out[0]
                intermediates[f"layer_{idx}"] = out.detach().float().cpu().squeeze(0).numpy()
            return hook
        layer_hooks.append(layer.register_forward_hook(make_hook(i)))

    if hasattr(backbone, "embedding_norm"):
        def hook_fn(module, inp, out):
            intermediates["final_norm"] = out.detach().float().cpu().squeeze(0).numpy()
        norm_hook = backbone.embedding_norm.register_forward_hook(hook_fn)
    else:
        norm_hook = None

    # ----------------------------------------------------------------
    # Run
    # ----------------------------------------------------------------
    with torch.no_grad():
        out = model(**enc)

    lhs = out.last_hidden_state[0].float()  # (T, H)

    # CLS = position 0
    cls_raw = lhs[0].numpy().astype(np.float32)
    cls_norm = cls_raw / (np.linalg.norm(cls_raw) + 1e-12)

    intermediates["cls_raw"]  = cls_raw
    intermediates["cls_norm"] = cls_norm

    # Cleanup hooks
    embed_hook.remove()
    for h in layer_hooks:
        h.remove()
    if norm_hook:
        norm_hook.remove()

    print(f"\nCaptured: {sorted(intermediates.keys())}")

    # ----------------------------------------------------------------
    # Write GGUF reference archive
    # ----------------------------------------------------------------
    writer = gguf.GGUFWriter(args.output, arch="lfm2_ref")
    writer.add_string("general.architecture", "lfm2_ref")
    writer.add_string("ref.text", args.text)
    writer.add_array("ref.input_ids", input_ids.tolist())

    for name in sorted(intermediates.keys()):
        arr = np.ascontiguousarray(intermediates[name], dtype=np.float32)
        # Shapes: post_embed / layer_N / final_norm are (T, H); cls_* are (H,)
        writer.add_tensor(name, arr, raw_dtype=gguf.GGMLQuantizationType.F32)
        print(f"  {name}: shape={arr.shape}, mean={arr.mean():.6f}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    import os
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\nWrote: {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
