#!/usr/bin/env python3
"""Convert DAT (Dual Aggregation Transformer) super-resolution model to GGUF.

Supports DAT-light, DAT-S, DAT, DAT-2 variants for 2x/3x/4x upscaling.
Source: https://github.com/zhengchen1999/DAT (Apache-2.0, ICCV 2023).

Architecture:
  Shallow feature extraction (Conv3x3)
  → N ResidualGroups of M DATB blocks (Spatial+Channel dual attention)
  → LayerNorm → residual Conv → PixelShuffle upsampler

Usage:
    python models/convert-dat-to-gguf.py \\
        --model DAT_light_x2.pth \\
        --output /mnt/storage/gguf-models/dat-light-x2-f16.gguf --fp16
"""

import argparse
import sys
from pathlib import Path

import gguf
import numpy as np


def main():
    p = argparse.ArgumentParser(description="Convert DAT SR model to GGUF")
    p.add_argument("--model", required=True, help="Path to .pth checkpoint")
    p.add_argument("--output", required=True, help="Output GGUF path")
    p.add_argument("--fp16", action="store_true", help="Store as FP16")
    args = p.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: {model_path} not found")
        sys.exit(1)

    # Load checkpoint
    import torch
    sd = torch.load(str(model_path), map_location="cpu", weights_only=False)
    # Handle nested state dict (some checkpoints wrap in 'params' or 'params_ema')
    if "params_ema" in sd:
        sd = sd["params_ema"]
    elif "params" in sd:
        sd = sd["params"]
    elif "state_dict" in sd:
        sd = sd["state_dict"]

    print(f"Loaded {len(sd)} tensors from {model_path.name}")

    # Detect architecture from weight shapes
    # conv_first.weight: (embed_dim, 3, 3, 3)
    embed_dim = sd["conv_first.weight"].shape[0]

    # Count residual groups and DATB blocks
    rg_count = 0
    while f"layers.{rg_count}.blocks.0.norm1.weight" in sd:
        rg_count += 1
    depth = []
    for rg in range(rg_count):
        block_count = 0
        while f"layers.{rg}.blocks.{block_count}.norm1.weight" in sd:
            block_count += 1
        depth.append(block_count)

    # Detect num_heads from attention bias shape
    first_attn_key = "layers.0.blocks.0.attn.attns.0.pos.pos3.0.weight"
    if first_attn_key in sd:
        # Indirect detection
        pass
    # Try qkv projection shape
    qkv_key = "layers.0.blocks.0.attn.attns.0.qkv.weight"
    if qkv_key in sd:
        qkv_dim = sd[qkv_key].shape[0]
        num_heads = qkv_dim // (3 * (embed_dim // (qkv_dim // (3 * embed_dim) if qkv_dim > 3 * embed_dim else 1)))
    else:
        num_heads = 6  # default

    # Detect upscale factor
    if "upsample.0.weight" in sd:
        # pixelshuffledirect: Conv2d(embed_dim, 3*upscale^2, 3, 1, 1)
        out_ch = sd["upsample.0.weight"].shape[0]
        upscale_sq = out_ch // 3
        upscale = int(upscale_sq ** 0.5 + 0.5)
    elif "conv_last.weight" in sd:
        # pixelshuffle: separate upsample blocks
        upscale = 2  # default guess
        for key in sd:
            if "upsample" in key and "weight" in key:
                upscale = max(upscale, 2)  # at least 2x
    else:
        upscale = 2

    # Detect resi_connection type
    resi = "3conv" if "conv_after_body.0.weight" in sd else "1conv"

    # Detect upsampler type
    upsampler = "pixelshuffledirect" if "upsample.0.weight" in sd else "pixelshuffle"

    print(f"Architecture: embed_dim={embed_dim}, depth={depth}, "
          f"upscale={upscale}x, resi={resi}, upsampler={upsampler}")

    # Write GGUF
    writer = gguf.GGUFWriter(str(args.output), arch="dat")

    writer.add_string("general.name", f"dat-sr-{upscale}x")
    writer.add_string("general.license", "Apache-2.0")
    writer.add_string("general.source", "https://github.com/zhengchen1999/DAT")

    writer.add_uint32("dat.embed_dim", embed_dim)
    writer.add_uint32("dat.upscale", upscale)
    writer.add_uint32("dat.num_heads", num_heads)
    writer.add_array("dat.depth", depth)
    writer.add_string("dat.resi_connection", resi)
    writer.add_string("dat.upsampler", upsampler)

    dtype_np = np.float16 if args.fp16 else np.float32
    dtype_gguf = gguf.GGMLQuantizationType.F16 if args.fp16 else gguf.GGMLQuantizationType.F32

    total_params = 0
    tensor_count = 0
    # Tensors that contain indices or BN stats — must stay FP32 to avoid overflow
    keep_f32 = ("relative_position_index", "attn_mask")
    for name, tensor in sorted(sd.items()):
        # Skip num_batches_tracked (unused in eval mode, names too long for GGUF)
        if "num_batches_tracked" in name:
            continue
        use_f32 = any(k in name for k in keep_f32)
        d_np = np.float32 if use_f32 else dtype_np
        d_gguf = gguf.GGMLQuantizationType.F32 if use_f32 else dtype_gguf
        data = tensor.float().numpy().astype(d_np)
        # Flatten 4D conv weights to 2D for quantization
        if data.ndim == 4:
            data = data.reshape(data.shape[0], -1)
        total_params += data.size
        writer.add_tensor(name, data, raw_dtype=d_gguf)
        tensor_count += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    out_size = Path(args.output).stat().st_size
    print(f"\nWrote {args.output}")
    print(f"  Tensors: {tensor_count}")
    print(f"  Parameters: {total_params:,}")
    print(f"  File size: {out_size:,} bytes ({out_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
