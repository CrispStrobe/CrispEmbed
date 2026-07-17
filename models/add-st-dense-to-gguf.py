#!/usr/bin/env python
"""Bake SentenceTransformers Dense projection modules into an existing
decoder-embedding GGUF.

Community/official llama.cpp SPM exports of SentenceTransformer models (e.g.
`ggml-org/embeddinggemma-300m-*-GGUF`, arch `gemma-embedding`) ship the
transformer backbone but omit the post-pooling Dense/Matryoshka head — llama.cpp
applies it from an external `--sentence-transformers-dense-modules` file. Without
it the embedding is the raw backbone mean-pool, orthogonal to the real model's
output (cos ~0 vs HF SentenceTransformer).

This copies such a GGUF verbatim (all tensors kept at their original quant, all
metadata preserved) and appends the Dense layers as `dense.{i}.weight` (F32),
which `src/decoder_embed.cpp` already applies after mean-pooling. The Dense
weights come from the model's ST snapshot dirs (`2_Dense/`, `3_Dense/`, ... each
holding `model.safetensors` with a `linear.weight` [out, in]).

Usage:
  python models/add-st-dense-to-gguf.py IN.gguf ST_SNAPSHOT_DIR OUT.gguf

Verified for google/embeddinggemma-300m: cos(GGUF+dense, HF full) = 0.984,
matching CrispEmbed's native EmbeddingGemma ceiling.
"""
import argparse
import glob
import os

import numpy as np
import torch
from gguf import GGMLQuantizationType, GGUFReader, GGUFWriter
from safetensors.torch import load_file


def main() -> int:
    ap = argparse.ArgumentParser(description="Bake ST Dense modules into a GGUF")
    ap.add_argument("input_gguf")
    ap.add_argument("st_snapshot", help="SentenceTransformer snapshot dir with N_Dense/ subdirs")
    ap.add_argument("output_gguf")
    args = ap.parse_args()

    reader = GGUFReader(args.input_gguf)
    arch = reader.fields["general.architecture"].contents()
    writer = GGUFWriter(args.output_gguf, arch)

    # GGUFWriter's constructor already emits general.architecture; copy the rest.
    copied = 0
    for key, field in reader.fields.items():
        if key == "general.architecture":
            continue
        writer.add_key_value(key, field.contents(), field.types[0])
        copied += 1
    print(f"copied {copied} metadata KV; arch={arch}")

    # Copy every existing tensor verbatim (raw bytes, original quant preserved).
    for t in reader.tensors:
        writer.add_tensor(t.name, t.data, raw_dtype=t.tensor_type)
    print(f"copied {len(reader.tensors)} tensors")

    # Discover the Dense module dirs (2_Dense, 3_Dense, ...) in snapshot order.
    dense_dirs = sorted(
        d
        for d in glob.glob(os.path.join(args.st_snapshot, "*_Dense"))
        if os.path.isfile(os.path.join(d, "model.safetensors"))
    )
    if not dense_dirs:
        raise SystemExit(f"no *_Dense/model.safetensors under {args.st_snapshot}")
    for i, d in enumerate(dense_dirs):
        # linear.weight is [out, in]; stored as dense.{i}.weight, no transpose
        # (matches models/convert-decoder-embed-to-gguf.py's emission).
        w = load_file(os.path.join(d, "model.safetensors"))["linear.weight"].float().numpy().astype(np.float32)
        writer.add_tensor(f"dense.{i}.weight", w, raw_dtype=GGMLQuantizationType.F32)
        print(f"  dense.{i}.weight {w.shape}  <- {os.path.basename(d)}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {args.output_gguf} ({os.path.getsize(args.output_gguf)/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
