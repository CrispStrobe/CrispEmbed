#!/usr/bin/env python3
"""Dump per-stage encoder intermediates from the ORIGINAL (HF/PyTorch) model.

Pairs with `CRISPEMBED_DUMP_LAYERS_GGUF=<path> crispembed -m model.gguf "<text>"`,
which writes the same stage names from the ggml graph. tests/test_encoder_diff.py
then compares the two GGUFs stage by stage.

Why per-stage and not just the final embedding: a final-embedding cosine tells you
THAT something diverged, never WHERE. Per the repo's diff methodology, the first
failing stage IS the bug — and a small drift at layer 0 can look identical to
"quantization noise" at the output while actually being a structural mismatch
(wrong rope theta, wrong LN epsilon, a shifted token sequence).

Stage mapping (no transpose on either side — ggml [H,T] has ne[0] as the fast
axis, so its flat memory is row-major (T,H), exactly HF's hidden_states[i][0]):

    emb_ln_out  ==  hidden_states[0]      (embeddings + LayerNorm, pre-block-0)
    layer_i     ==  hidden_states[i+1]    (output of encoder block i)

`emb_ln_out` is the STRUCTURAL GATE: it precedes every block, so it depends only
on tokenization + embeddings. If it does not match to ~0.99999, the two sides are
not even reading the same input and every later per-layer number is meaningless —
check tokenization/prefix/special tokens before touching the graph.

Usage:
  python tools/dump_encoder_reference.py --model sentence-transformers/all-MiniLM-L6-v2 \
      --text "hello world" --output /tmp/ref.gguf
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _find_layers(model):
    """Locate the ModuleList of transformer blocks.

    Custom architectures put it in different places (bert: encoder.layer;
    nomic: encoder.layers), so probe the known paths rather than assume.
    """
    import torch.nn as nn

    for path in ("encoder.layers", "encoder.layer", "layers", "blocks", "encoder.blocks",
                 "transformer.layers", "transformer.layer"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if isinstance(obj, (nn.ModuleList, list)) and len(obj) > 0:
            print(f"layers found at: {path} ({len(obj)} blocks)")
            return obj
    return None


def hook_hidden_states(model, enc):
    """Capture per-stage activations with forward hooks.

    Returns [emb_out, layer_0_out, ..., layer_{n-1}_out] matching the shape
    contract of HF's output_hidden_states.
    """
    import torch

    layers = _find_layers(model)
    if layers is None:
        return None

    captured: dict = {}
    handles = []

    def grab(key):
        def hook(_mod, _inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            captured[key] = t.detach()

        return hook

    def grab_input(key):
        # Capture what actually ENTERS block 0. Hooking the `embeddings` module's
        # OUTPUT is wrong on some architectures: BERT's embeddings include the
        # LayerNorm, but nomic's do not (the LN lives outside), so that output is
        # pre-LN while crispembed's emb_ln_out is post-LN — comparing them read
        # cos=0.69 while every layer read 1.000000, i.e. a pure harness artifact.
        # Block 0's input is "pre-block-0" by definition, on any architecture.
        def hook(_mod, inp, _out):
            t = inp[0] if isinstance(inp, (tuple, list)) else inp
            captured[key] = t.detach()

        return hook

    handles.append(layers[0].register_forward_pre_hook(
        lambda m, i: captured.__setitem__("emb", (i[0] if isinstance(i, (tuple, list)) else i).detach())))
    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(grab(f"l{i}")))

    try:
        with torch.no_grad():
            model(**enc)
    finally:
        for h in handles:
            h.remove()

    if "emb" not in captured:
        return None
    out = [captured["emb"]]
    for i in range(len(layers)):
        if f"l{i}" not in captured:
            print(f"WARNING: layer {i} produced no activation", file=sys.stderr)
            return None
        out.append(captured[f"l{i}"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Dump HF encoder per-stage intermediates to GGUF")
    ap.add_argument("--model", required=True, help="HF repo id or local path")
    ap.add_argument("--text", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--max-length", type=int, default=512)
    args = ap.parse_args()

    import numpy as np
    import torch
    from gguf import GGUFWriter
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModel.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code, torch_dtype=torch.float32
    ).eval()

    enc = tok(args.text, return_tensors="pt", truncation=True, max_length=args.max_length)

    hs = None
    try:
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        hs = getattr(out, "hidden_states", None)
        if hs is not None:
            print("stages via output_hidden_states")
    except TypeError:
        # Custom remote-code models often don't accept output_hidden_states
        # (nomic's NomicBertModel.forward() rejects the kwarg outright). Fall
        # back to forward hooks, which work on any nn.Module tree.
        print("model rejects output_hidden_states -> falling back to forward hooks")

    if hs is None:
        hs = hook_hidden_states(model, enc)
    if not hs:
        print("ERROR: could not capture hidden states (no hidden_states, no layer list found)",
              file=sys.stderr)
        return 1

    ids = enc["input_ids"][0].tolist()
    n_tok = len(ids)
    print(f"tokens ({n_tok}): {ids}")
    print(f"decoded: {tok.convert_ids_to_tokens(ids)}")
    print(f"stages: {len(hs)} hidden_states (emb_ln_out + {len(hs) - 1} layers)")

    w = GGUFWriter(args.output, "crispembed-encoder-dump")
    w.add_uint32("dump.n_layer", len(hs) - 1)
    w.add_uint32("dump.n_embd", int(hs[0].shape[-1]))
    w.add_uint32("dump.n_tokens", n_tok)
    # Token ids let the comparer assert both sides tokenized identically BEFORE
    # trusting any per-layer cosine (a shifted sequence mimics numeric drift).
    w.add_array("dump.input_ids", [int(i) for i in ids])

    def add(name: str, t: torch.Tensor) -> None:
        # (1, T, H) -> (T, H) float32, C-contiguous == ggml [H, T] flat memory.
        a = t[0].detach().cpu().numpy().astype(np.float32)
        w.add_tensor(name, np.ascontiguousarray(a))
        print(f"  {name:16s} shape={a.shape}")

    add("emb_ln_out", hs[0])
    for i in range(1, len(hs)):
        add(f"layer_{i - 1}", hs[i])

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
