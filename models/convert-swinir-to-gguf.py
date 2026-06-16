#!/usr/bin/env python3
"""Convert SwinIR-light (JingyunLiang/SwinIR) PyTorch checkpoint to GGUF.

Usage:
    python convert-swinir-to-gguf.py \
        --model 001_classicalSR_DIV2K_s48w8_SwinIR-S_x4.pth \
        --output swinir-light-x4-f16.gguf --fp16

SwinIR-light architecture:
    conv_first(3->60) -> 4x RSTB(6 Swin blocks, embed_dim=60, 6 heads, window=8, mlp_ratio=2)
    -> norm -> conv_after_body + skip -> upsample(PixelShuffle)
"""

import argparse
import struct
import sys

import numpy as np
import torch

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_I32 = 26
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_STRING = 8


def write_string(f, s):
    b = s.encode("utf-8")
    f.write(struct.pack("<Q", len(b)))
    f.write(b)


def write_kv_string(f, key, val):
    write_string(f, key)
    f.write(struct.pack("<I", GGUF_TYPE_STRING))
    write_string(f, val)


def write_kv_u32(f, key, val):
    write_string(f, key)
    f.write(struct.pack("<I", GGUF_TYPE_UINT32))
    f.write(struct.pack("<I", val))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    state = torch.load(args.model, map_location="cpu", weights_only=True)
    if "params" in state:
        state = state["params"]

    # Convert all tensors to numpy
    sd = {}
    for k, v in state.items():
        sd[k] = v.numpy()

    tensors = {}

    def add(name, key, dtype_override=None):
        if key not in sd:
            print(f"  WARNING: missing {key}")
            return
        arr = sd[key]
        if dtype_override == "int32":
            tensors[name] = (arr.astype(np.int32), GGML_TYPE_I32)
        elif dtype_override == "float32":
            tensors[name] = (arr.astype(np.float32), GGML_TYPE_F32)
        else:
            arr = arr.astype(np.float32)
            if args.fp16 and arr.ndim >= 2:
                tensors[name] = (arr.astype(np.float16), GGML_TYPE_F16)
            else:
                tensors[name] = (arr.astype(np.float32), GGML_TYPE_F32)

    # Detect config
    embed_dim = sd["conv_first.weight"].shape[0]
    n_rstb = 0
    while f"layers.{n_rstb}.residual_group.blocks.0.attn.qkv.weight" in sd:
        n_rstb += 1
    n_blocks = 0
    while f"layers.0.residual_group.blocks.{n_blocks}.attn.qkv.weight" in sd:
        n_blocks += 1
    n_heads = sd["layers.0.residual_group.blocks.0.attn.relative_position_bias_table"].shape[1]
    # Detect scale from upsample conv
    upsample_out = sd["upsample.0.weight"].shape[0]
    scale = int(np.sqrt(upsample_out / 3))

    print(f"SwinIR-light: embed_dim={embed_dim}, n_rstb={n_rstb}, n_blocks={n_blocks}, "
          f"n_heads={n_heads}, scale={scale}")
    print(f"Total params: {sum(v.size for v in sd.values() if isinstance(v, np.ndarray)):,}")

    # conv_first
    add("conv_first.weight", "conv_first.weight")
    add("conv_first.bias", "conv_first.bias")

    # RSTB layers
    for i in range(n_rstb):
        for j in range(n_blocks):
            sp = f"layers.{i}.residual_group.blocks.{j}"
            dp = f"rstb.{i}.block.{j}"

            # Attention
            add(f"{dp}.attn.qkv.weight", f"{sp}.attn.qkv.weight")
            add(f"{dp}.attn.qkv.bias", f"{sp}.attn.qkv.bias")
            add(f"{dp}.attn.proj.weight", f"{sp}.attn.proj.weight")
            add(f"{dp}.attn.proj.bias", f"{sp}.attn.proj.bias")
            add(f"{dp}.attn.rpb_table", f"{sp}.attn.relative_position_bias_table")
            add(f"{dp}.attn.rpb_index", f"{sp}.attn.relative_position_index", dtype_override="int32")

            # Attention mask (only on odd-indexed blocks)
            mask_key = f"{sp}.attn_mask"
            if mask_key in sd:
                add(f"{dp}.attn_mask", mask_key, dtype_override="float32")

            # Norms
            add(f"{dp}.norm1.weight", f"{sp}.norm1.weight")
            add(f"{dp}.norm1.bias", f"{sp}.norm1.bias")
            add(f"{dp}.norm2.weight", f"{sp}.norm2.weight")
            add(f"{dp}.norm2.bias", f"{sp}.norm2.bias")

            # MLP
            add(f"{dp}.mlp.up.weight", f"{sp}.mlp.fc1.weight")
            add(f"{dp}.mlp.up.bias", f"{sp}.mlp.fc1.bias")
            add(f"{dp}.mlp.down.weight", f"{sp}.mlp.fc2.weight")
            add(f"{dp}.mlp.down.bias", f"{sp}.mlp.fc2.bias")

        # RSTB conv
        add(f"rstb.{i}.conv.weight", f"layers.{i}.conv.weight")
        add(f"rstb.{i}.conv.bias", f"layers.{i}.conv.bias")

    # patch_embed norm
    add("patch_norm.weight", "patch_embed.norm.weight")
    add("patch_norm.bias", "patch_embed.norm.bias")

    # Final norm
    add("norm.weight", "norm.weight")
    add("norm.bias", "norm.bias")

    # conv_after_body
    add("conv_after_body.weight", "conv_after_body.weight")
    add("conv_after_body.bias", "conv_after_body.bias")

    # Upsample
    add("upsample.weight", "upsample.0.weight")
    add("upsample.bias", "upsample.0.bias")

    print(f"GGUF tensors: {len(tensors)}")

    n_kv = 9
    with open(args.output, "wb") as f:
        f.write(struct.pack("<I", GGUF_MAGIC))
        f.write(struct.pack("<I", GGUF_VERSION))
        f.write(struct.pack("<Q", len(tensors)))
        f.write(struct.pack("<Q", n_kv))

        write_kv_string(f, "general.architecture", "swinir")
        write_kv_string(f, "general.name", f"SwinIR-light-x{scale}")
        write_kv_u32(f, "swinir.embed_dim", embed_dim)
        write_kv_u32(f, "swinir.n_rstb", n_rstb)
        write_kv_u32(f, "swinir.n_blocks", n_blocks)
        write_kv_u32(f, "swinir.n_heads", n_heads)
        write_kv_u32(f, "swinir.window_size", 8)
        write_kv_u32(f, "swinir.mlp_ratio", 2)
        write_kv_u32(f, "swinir.upscale", scale)

        offset = 0
        tensor_list = list(tensors.items())
        for name, (data, dtype_id) in tensor_list:
            write_string(f, name)
            f.write(struct.pack("<I", len(data.shape)))
            for d in data.shape:
                f.write(struct.pack("<Q", d))
            f.write(struct.pack("<I", dtype_id))
            f.write(struct.pack("<Q", offset))
            offset += data.nbytes
            offset = (offset + 31) & ~31

        pos = f.tell()
        aligned = (pos + 31) & ~31
        f.write(b"\x00" * (aligned - pos))

        for name, (data, dtype_id) in tensor_list:
            f.write(data.tobytes())
            pad = ((data.nbytes + 31) & ~31) - data.nbytes
            if pad > 0:
                f.write(b"\x00" * pad)

    size_mb = sum(d.nbytes for d, _ in tensors.values()) / 1024 / 1024
    print(f"Written: {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
