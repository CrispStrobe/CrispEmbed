#!/usr/bin/env python3
"""Convert HAT (Hybrid Attention Transformer) checkpoint to GGUF.

Usage:
    python convert-hat-to-gguf.py \
        --model HAT_SRx4_ImageNet-pretrain.pth \
        --output hat-sr-x4-f16.gguf --fp16

HAT (CVPR 2023, MIT):
    Swin-style window attention + overlapping cross-attention (OCAB) +
    channel attention blocks (CAB). 6 RHAG layers × 6 HABs + 1 OCAB each.
    ~21M params for HAT-S x4. PixelShuffle upsampling.
"""

import argparse
import struct
import sys
import math

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
    write_string(f, key); f.write(struct.pack("<I", GGUF_TYPE_STRING)); write_string(f, val)

def write_kv_u32(f, key, val):
    write_string(f, key); f.write(struct.pack("<I", GGUF_TYPE_UINT32)); f.write(struct.pack("<I", val))

def write_kv_f32(f, key, val):
    write_string(f, key); f.write(struct.pack("<I", GGUF_TYPE_FLOAT32)); f.write(struct.pack("<f", val))

def write_kv_u32_array(f, key, arr):
    write_string(f, key); f.write(struct.pack("<I", GGUF_TYPE_ARRAY))
    f.write(struct.pack("<I", GGUF_TYPE_UINT32)); f.write(struct.pack("<Q", len(arr)))
    for v in arr: f.write(struct.pack("<I", v))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    state = torch.load(args.model, map_location="cpu", weights_only=True)
    if "params_ema" in state: state = state["params_ema"]
    elif "params" in state: state = state["params"]

    # Detect config
    embed_dim = state["conv_first.weight"].shape[0]
    n_layers = 0
    while f"layers.{n_layers}.residual_group.blocks.0.norm1.weight" in state:
        n_layers += 1
    depths = []
    for i in range(n_layers):
        d = 0
        while f"layers.{i}.residual_group.blocks.{d}.norm1.weight" in state:
            d += 1
        depths.append(d)
    heads = []
    for i in range(n_layers):
        t = state[f"layers.{i}.residual_group.blocks.0.attn.relative_position_bias_table"]
        heads.append(t.shape[1])

    # Window size from relative_position_bias_table shape: (2*ws-1)^2
    rpb_size = state["layers.0.residual_group.blocks.0.attn.relative_position_bias_table"].shape[0]
    window_size = int((math.sqrt(rpb_size) + 1) / 2)

    # Upscale factor from upsample layers
    n_upsample = 0
    while f"upsample.{n_upsample * 2}.weight" in state:
        n_upsample += 1
    upscale = 2 ** n_upsample

    # Overlap ratio: rpb shape is (ws + ws_ext - 1)^2, ws_ext = int(ws*ratio) + ws
    ocab_rpb = state["layers.0.residual_group.overlap_attn.relative_position_bias_table"].shape[0]
    ws_plus_ext_minus1 = int(math.sqrt(ocab_rpb))
    overlap_win_size = ws_plus_ext_minus1 - window_size + 1
    overlap_ratio = (overlap_win_size - window_size) / window_size

    # Conv scale, compress ratio, squeeze factor from CAB
    compress_ratio = embed_dim // state["layers.0.residual_group.blocks.0.conv_block.cab.0.weight"].shape[0]
    squeeze_factor = embed_dim // state["layers.0.residual_group.blocks.0.conv_block.cab.3.attention.1.weight"].shape[0]

    num_feat = state["conv_before_upsample.0.weight"].shape[0]
    mlp_ratio = state["layers.0.residual_group.blocks.0.mlp.fc1.weight"].shape[0] / embed_dim

    print(f"HAT: embed_dim={embed_dim}, n_layers={n_layers}, depths={depths}")
    print(f"  heads={heads}, window_size={window_size}, upscale={upscale}x")
    print(f"  overlap_ratio={overlap_ratio:.1f}, compress_ratio={compress_ratio}")
    print(f"  squeeze_factor={squeeze_factor}, num_feat={num_feat}, mlp_ratio={mlp_ratio:.0f}")
    print(f"  Total params: {sum(v.numel() for v in state.values()):,}")

    tensors = {}
    def add(name, key):
        if key not in state: return
        arr = state[key].float().numpy()
        if args.fp16 and arr.ndim >= 2:
            tensors[name] = (arr.astype(np.float16), GGML_TYPE_F16)
        else:
            tensors[name] = (arr.astype(np.float32), GGML_TYPE_F32)

    # Shallow feature extraction
    add("conv_first.weight", "conv_first.weight")
    add("conv_first.bias", "conv_first.bias")

    # Patch embed norm (applied when patch_norm=True)
    add("patch_embed.norm.weight", "patch_embed.norm.weight")
    add("patch_embed.norm.bias", "patch_embed.norm.bias")

    # RHAG layers
    for li in range(n_layers):
        sp = f"layers.{li}"
        dp = f"layer.{li}"

        # HAB blocks
        for bi in range(depths[li]):
            sb = f"{sp}.residual_group.blocks.{bi}"
            db = f"{dp}.hab.{bi}"

            # WindowAttention
            add(f"{db}.attn.qkv.weight", f"{sb}.attn.qkv.weight")
            add(f"{db}.attn.qkv.bias", f"{sb}.attn.qkv.bias")
            add(f"{db}.attn.proj.weight", f"{sb}.attn.proj.weight")
            add(f"{db}.attn.proj.bias", f"{sb}.attn.proj.bias")
            add(f"{db}.attn.rpb", f"{sb}.attn.relative_position_bias_table")

            # CAB
            add(f"{db}.cab.conv1.weight", f"{sb}.conv_block.cab.0.weight")
            add(f"{db}.cab.conv1.bias", f"{sb}.conv_block.cab.0.bias")
            add(f"{db}.cab.conv2.weight", f"{sb}.conv_block.cab.2.weight")
            add(f"{db}.cab.conv2.bias", f"{sb}.conv_block.cab.2.bias")
            # ChannelAttention: pool → conv_down → relu → conv_up → sigmoid
            add(f"{db}.cab.ca_down.weight", f"{sb}.conv_block.cab.3.attention.1.weight")
            add(f"{db}.cab.ca_down.bias", f"{sb}.conv_block.cab.3.attention.1.bias")
            add(f"{db}.cab.ca_up.weight", f"{sb}.conv_block.cab.3.attention.3.weight")
            add(f"{db}.cab.ca_up.bias", f"{sb}.conv_block.cab.3.attention.3.bias")

            # MLP
            add(f"{db}.mlp.fc1.weight", f"{sb}.mlp.fc1.weight")
            add(f"{db}.mlp.fc1.bias", f"{sb}.mlp.fc1.bias")
            add(f"{db}.mlp.fc2.weight", f"{sb}.mlp.fc2.weight")
            add(f"{db}.mlp.fc2.bias", f"{sb}.mlp.fc2.bias")

            # LayerNorm
            add(f"{db}.norm1.weight", f"{sb}.norm1.weight")
            add(f"{db}.norm1.bias", f"{sb}.norm1.bias")
            add(f"{db}.norm2.weight", f"{sb}.norm2.weight")
            add(f"{db}.norm2.bias", f"{sb}.norm2.bias")

            # conv_scale parameter (scalar, stored as 1-elem tensor for compat)
            if f"{sb}.conv_scale" in state:
                add(f"{db}.conv_scale", f"{sb}.conv_scale")

        # OCAB (overlapping cross-attention)
        so = f"{sp}.residual_group.overlap_attn"
        do = f"{dp}.ocab"
        add(f"{do}.qkv.weight", f"{so}.qkv.weight")
        add(f"{do}.qkv.bias", f"{so}.qkv.bias")
        add(f"{do}.proj.weight", f"{so}.proj.weight")
        add(f"{do}.proj.bias", f"{so}.proj.bias")
        add(f"{do}.rpb", f"{so}.relative_position_bias_table")
        add(f"{do}.norm1.weight", f"{so}.norm1.weight")
        add(f"{do}.norm1.bias", f"{so}.norm1.bias")
        add(f"{do}.norm2.weight", f"{so}.norm2.weight")
        add(f"{do}.norm2.bias", f"{so}.norm2.bias")
        add(f"{do}.mlp.fc1.weight", f"{so}.mlp.fc1.weight")
        add(f"{do}.mlp.fc1.bias", f"{so}.mlp.fc1.bias")
        add(f"{do}.mlp.fc2.weight", f"{so}.mlp.fc2.weight")
        add(f"{do}.mlp.fc2.bias", f"{so}.mlp.fc2.bias")

        # RHAG conv + patch embed/unembed (identity for patch_size=1)
        add(f"{dp}.conv.weight", f"{sp}.conv.weight")
        add(f"{dp}.conv.bias", f"{sp}.conv.bias")

    # Final norm
    add("norm.weight", "norm.weight")
    add("norm.bias", "norm.bias")

    # Deep feature extraction output conv
    add("conv_after_body.weight", "conv_after_body.weight")
    add("conv_after_body.bias", "conv_after_body.bias")

    # Reconstruction
    add("conv_before_upsample.weight", "conv_before_upsample.0.weight")
    add("conv_before_upsample.bias", "conv_before_upsample.0.bias")
    for i in range(n_upsample):
        add(f"upsample.{i}.weight", f"upsample.{i * 2}.weight")
        add(f"upsample.{i}.bias", f"upsample.{i * 2}.bias")
    add("conv_last.weight", "conv_last.weight")
    add("conv_last.bias", "conv_last.bias")

    # Relative position index buffers (precomputed, store for C++ use)
    add("rpi_sa", "relative_position_index_SA")
    add("rpi_oca", "relative_position_index_OCA")

    print(f"GGUF tensors: {len(tensors)}")
    gguf_params = sum(d.size for d, _ in tensors.values())
    print(f"GGUF params: {gguf_params:,}")

    # Write GGUF
    n_kv = 11
    with open(args.output, "wb") as f:
        f.write(struct.pack("<I", GGUF_MAGIC))
        f.write(struct.pack("<I", GGUF_VERSION))
        f.write(struct.pack("<Q", len(tensors)))
        f.write(struct.pack("<Q", n_kv))

        write_kv_string(f, "general.architecture", "hat")
        write_kv_string(f, "general.name", f"HAT-SRx{upscale}")
        write_kv_u32(f, "hat.embed_dim", embed_dim)
        write_kv_u32_array(f, "hat.depths", depths)
        write_kv_u32_array(f, "hat.heads", heads)
        write_kv_u32(f, "hat.window_size", window_size)
        write_kv_u32(f, "hat.upscale", upscale)
        write_kv_u32(f, "hat.num_feat", num_feat)
        write_kv_u32(f, "hat.compress_ratio", compress_ratio)
        write_kv_u32(f, "hat.squeeze_factor", squeeze_factor)
        write_kv_f32(f, "hat.overlap_ratio", overlap_ratio)

        offset = 0
        tensor_list = list(tensors.items())
        for name, (data, dtype_id) in tensor_list:
            write_string(f, name)
            f.write(struct.pack("<I", len(data.shape)))
            for d in data.shape: f.write(struct.pack("<Q", d))
            f.write(struct.pack("<I", dtype_id))
            f.write(struct.pack("<Q", offset))
            offset += data.nbytes; offset = (offset + 31) & ~31

        pos = f.tell(); aligned = (pos + 31) & ~31; f.write(b"\x00" * (aligned - pos))
        for name, (data, dtype_id) in tensor_list:
            f.write(data.tobytes())
            pad = ((data.nbytes + 31) & ~31) - data.nbytes
            if pad > 0: f.write(b"\x00" * pad)

    size_mb = sum(d.nbytes for d, _ in tensors.values()) / 1024 / 1024
    print(f"Written: {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
