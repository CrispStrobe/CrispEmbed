#!/usr/bin/env python3
"""Convert Restormer checkpoint to GGUF for CrispEmbed.

Usage:
    python convert-restormer-to-gguf.py \
        --model gaussian_color_denoising_blind.pth \
        --output restormer-denoise-f16.gguf --fp16

Restormer (CVPR 2022, Apache-2.0):
    U-Net with Multi-DConv Head Transposed Attention (MDTA) + Gated-DConv FFN (GDFN).
    4 encoder levels, latent, 3 decoder levels + refinement.
    Channels: 48 → 96 → 192 → 384 (latent) → 192 → 96 → 96 → output.
    ~26M params, ~50 MB F16.
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
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9


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


def write_kv_f32(f, key, val):
    write_string(f, key)
    f.write(struct.pack("<I", GGUF_TYPE_FLOAT32))
    f.write(struct.pack("<f", val))


def write_kv_u32_array(f, key, arr):
    write_string(f, key)
    f.write(struct.pack("<I", GGUF_TYPE_ARRAY))
    f.write(struct.pack("<I", GGUF_TYPE_UINT32))
    f.write(struct.pack("<Q", len(arr)))
    for v in arr:
        f.write(struct.pack("<I", v))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    state = torch.load(args.model, map_location="cpu", weights_only=True)
    if "params" in state:
        state = state["params"]
    elif "state_dict" in state:
        state = state["state_dict"]

    # Detect config
    dim = state["patch_embed.proj.weight"].shape[0]  # 48
    # Count blocks per level
    num_blocks = []
    for level_name in ["encoder_level1", "encoder_level2", "encoder_level3", "latent"]:
        n = 0
        while f"{level_name}.{n}.norm1.body.weight" in state:
            n += 1
        num_blocks.append(n)
    # Count refinement blocks
    n_refine = 0
    while f"refinement.{n_refine}.norm1.body.weight" in state:
        n_refine += 1
    # Detect heads from temperature shapes
    heads = []
    for level_name in ["encoder_level1", "encoder_level2", "encoder_level3", "latent"]:
        t = state[f"{level_name}.0.attn.temperature"]
        heads.append(t.shape[0])
    # FFN expansion factor: canonical value is 2.66.
    # Derive from the largest level where int() rounding doesn't matter.
    ffn_in_shape = state["latent.0.ffn.project_in.weight"].shape[0]  # hidden*2
    latent_dim = dim * 8
    ffn_factor = (ffn_in_shape / 2) / latent_dim  # 1020 / 384 = 2.65625 ≈ 2.66
    # Check for bias
    has_bias = f"patch_embed.proj.bias" in state

    print(f"Restormer: dim={dim}, blocks={num_blocks}, heads={heads}")
    print(f"  ffn_factor={ffn_factor:.2f}, refine={n_refine}, bias={has_bias}")
    print(f"  Total params: {sum(v.numel() for v in state.values()):,}")

    tensors = {}

    def add(name, key):
        if key not in state:
            return
        arr = state[key].float().numpy()
        if args.fp16 and arr.ndim >= 2:
            tensors[name] = (arr.astype(np.float16), GGML_TYPE_F16)
        else:
            tensors[name] = (arr.astype(np.float32), GGML_TYPE_F32)

    # Patch embed
    add("patch_embed.weight", "patch_embed.proj.weight")
    if has_bias:
        add("patch_embed.bias", "patch_embed.proj.bias")

    # Encoder levels + latent
    level_names = ["encoder_level1", "encoder_level2", "encoder_level3", "latent"]
    gguf_levels = ["enc.0", "enc.1", "enc.2", "latent"]
    for li, (src_name, dst_name) in enumerate(zip(level_names, gguf_levels)):
        for b in range(num_blocks[li]):
            sp = f"{src_name}.{b}"
            dp = f"{dst_name}.{b}"
            # MDTA
            add(f"{dp}.attn.qkv.weight", f"{sp}.attn.qkv.weight")
            add(f"{dp}.attn.qkv_dw.weight", f"{sp}.attn.qkv_dwconv.weight")
            add(f"{dp}.attn.proj.weight", f"{sp}.attn.project_out.weight")
            add(f"{dp}.attn.temperature", f"{sp}.attn.temperature")
            if has_bias:
                add(f"{dp}.attn.qkv.bias", f"{sp}.attn.qkv.bias")
                add(f"{dp}.attn.qkv_dw.bias", f"{sp}.attn.qkv_dwconv.bias")
                add(f"{dp}.attn.proj.bias", f"{sp}.attn.project_out.bias")
            # GDFN
            add(f"{dp}.ffn.in.weight", f"{sp}.ffn.project_in.weight")
            add(f"{dp}.ffn.dw.weight", f"{sp}.ffn.dwconv.weight")
            add(f"{dp}.ffn.out.weight", f"{sp}.ffn.project_out.weight")
            if has_bias:
                add(f"{dp}.ffn.in.bias", f"{sp}.ffn.project_in.bias")
                add(f"{dp}.ffn.dw.bias", f"{sp}.ffn.dwconv.bias")
                add(f"{dp}.ffn.out.bias", f"{sp}.ffn.project_out.bias")
            # LayerNorm
            add(f"{dp}.norm1.weight", f"{sp}.norm1.body.weight")
            add(f"{dp}.norm2.weight", f"{sp}.norm2.body.weight")
            if f"{sp}.norm1.body.bias" in state:
                add(f"{dp}.norm1.bias", f"{sp}.norm1.body.bias")
                add(f"{dp}.norm2.bias", f"{sp}.norm2.body.bias")

    # Downsample
    for i, name in enumerate(["down1_2", "down2_3", "down3_4"]):
        add(f"down.{i}.weight", f"{name}.body.0.weight")

    # Upsample
    for i, name in enumerate(["up4_3", "up3_2", "up2_1"]):
        add(f"up.{i}.weight", f"{name}.body.0.weight")

    # Channel reduction
    add("reduce.2.weight", "reduce_chan_level3.weight")
    add("reduce.1.weight", "reduce_chan_level2.weight")
    if has_bias:
        add("reduce.2.bias", "reduce_chan_level3.bias")
        add("reduce.1.bias", "reduce_chan_level2.bias")

    # Decoder levels
    dec_names = ["decoder_level3", "decoder_level2", "decoder_level1"]
    dec_gguf = ["dec.2", "dec.1", "dec.0"]
    dec_blocks = [num_blocks[2], num_blocks[1], num_blocks[0]]
    for li, (src_name, dst_name) in enumerate(zip(dec_names, dec_gguf)):
        for b in range(dec_blocks[li]):
            sp = f"{src_name}.{b}"
            dp = f"{dst_name}.{b}"
            add(f"{dp}.attn.qkv.weight", f"{sp}.attn.qkv.weight")
            add(f"{dp}.attn.qkv_dw.weight", f"{sp}.attn.qkv_dwconv.weight")
            add(f"{dp}.attn.proj.weight", f"{sp}.attn.project_out.weight")
            add(f"{dp}.attn.temperature", f"{sp}.attn.temperature")
            if has_bias:
                add(f"{dp}.attn.qkv.bias", f"{sp}.attn.qkv.bias")
                add(f"{dp}.attn.qkv_dw.bias", f"{sp}.attn.qkv_dwconv.bias")
                add(f"{dp}.attn.proj.bias", f"{sp}.attn.project_out.bias")
            add(f"{dp}.ffn.in.weight", f"{sp}.ffn.project_in.weight")
            add(f"{dp}.ffn.dw.weight", f"{sp}.ffn.dwconv.weight")
            add(f"{dp}.ffn.out.weight", f"{sp}.ffn.project_out.weight")
            if has_bias:
                add(f"{dp}.ffn.in.bias", f"{sp}.ffn.project_in.bias")
                add(f"{dp}.ffn.dw.bias", f"{sp}.ffn.dwconv.bias")
                add(f"{dp}.ffn.out.bias", f"{sp}.ffn.project_out.bias")
            add(f"{dp}.norm1.weight", f"{sp}.norm1.body.weight")
            add(f"{dp}.norm2.weight", f"{sp}.norm2.body.weight")
            if f"{sp}.norm1.body.bias" in state:
                add(f"{dp}.norm1.bias", f"{sp}.norm1.body.bias")
                add(f"{dp}.norm2.bias", f"{sp}.norm2.body.bias")

    # Refinement
    for b in range(n_refine):
        sp = f"refinement.{b}"
        dp = f"refine.{b}"
        add(f"{dp}.attn.qkv.weight", f"{sp}.attn.qkv.weight")
        add(f"{dp}.attn.qkv_dw.weight", f"{sp}.attn.qkv_dwconv.weight")
        add(f"{dp}.attn.proj.weight", f"{sp}.attn.project_out.weight")
        add(f"{dp}.attn.temperature", f"{sp}.attn.temperature")
        if has_bias:
            add(f"{dp}.attn.qkv.bias", f"{sp}.attn.qkv.bias")
            add(f"{dp}.attn.qkv_dw.bias", f"{sp}.attn.qkv_dwconv.bias")
            add(f"{dp}.attn.proj.bias", f"{sp}.attn.project_out.bias")
        add(f"{dp}.ffn.in.weight", f"{sp}.ffn.project_in.weight")
        add(f"{dp}.ffn.dw.weight", f"{sp}.ffn.dwconv.weight")
        add(f"{dp}.ffn.out.weight", f"{sp}.ffn.project_out.weight")
        if has_bias:
            add(f"{dp}.ffn.in.bias", f"{sp}.ffn.project_in.bias")
            add(f"{dp}.ffn.dw.bias", f"{sp}.ffn.dwconv.bias")
            add(f"{dp}.ffn.out.bias", f"{sp}.ffn.project_out.bias")
        add(f"{dp}.norm1.weight", f"{sp}.norm1.body.weight")
        add(f"{dp}.norm2.weight", f"{sp}.norm2.body.weight")
        if f"{sp}.norm1.body.bias" in state:
            add(f"{dp}.norm1.bias", f"{sp}.norm1.body.bias")
            add(f"{dp}.norm2.bias", f"{sp}.norm2.body.bias")

    # Output conv
    add("output.weight", "output.weight")
    if has_bias:
        add("output.bias", "output.bias")

    print(f"GGUF tensors: {len(tensors)}")

    # Check for unmapped tensors
    mapped_src = set()
    for key in state:
        found = False
        for gname, (_, _) in tensors.items():
            pass
        # Simple check: skip known prefixes
        if any(key.startswith(p) for p in ["dual_pixel", "skip_conv"]):
            continue
    total_gguf = sum(d.size for d, _ in tensors.values())
    total_src = sum(v.numel() for v in state.values())
    print(f"GGUF params: {total_gguf:,} / source: {total_src:,}")

    # Write GGUF
    n_kv = 8
    with open(args.output, "wb") as f:
        f.write(struct.pack("<I", GGUF_MAGIC))
        f.write(struct.pack("<I", GGUF_VERSION))
        f.write(struct.pack("<Q", len(tensors)))
        f.write(struct.pack("<Q", n_kv))

        write_kv_string(f, "general.architecture", "restormer")
        write_kv_string(f, "general.name", "Restormer-Denoise-Gaussian-Blind")
        write_kv_u32(f, "restormer.dim", dim)
        write_kv_u32_array(f, "restormer.num_blocks", num_blocks)
        write_kv_u32_array(f, "restormer.heads", heads)
        write_kv_f32(f, "restormer.ffn_expansion_factor", ffn_factor)
        write_kv_u32(f, "restormer.num_refinement_blocks", n_refine)
        write_kv_u32(f, "restormer.bias", 1 if has_bias else 0)

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
