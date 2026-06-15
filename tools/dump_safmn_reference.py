#!/usr/bin/env python3
"""Dump SAFMN per-layer reference activations for crispembed-diff parity testing.

Usage:
    python tools/dump_safmn_reference.py \
        --model /mnt/storage/models/SAFMN_DF2K_x4.pth \
        --output /tmp/safmn-ref.gguf \
        [--image test.png] [--width 64] [--height 64]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    import gguf
except ImportError:
    sys.exit("pip install gguf")


# ── Minimal SAFMN reimplementation (for reference inference) ──

class ChannelLayerNorm:
    def __init__(self, weight, bias, eps=1e-6):
        self.weight = weight  # [C]
        self.bias = bias
        self.eps = eps

    def __call__(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / (var + self.eps).sqrt()
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


def safm_forward(x, mfr_weights, aggr_w, aggr_b):
    """SAFM: multi-scale feature modulation."""
    B, C, H, W = x.shape
    n_levels = len(mfr_weights)
    chunk_dim = C // n_levels
    chunks = x.chunk(n_levels, dim=1)

    outs = []
    for i, chunk in enumerate(chunks):
        w, b = mfr_weights[i]
        if i == 0:
            out = F.conv2d(chunk, w, b, padding=1, groups=chunk_dim)
        else:
            s = 2 ** i
            ph, pw = max(1, H // s), max(1, W // s)
            pooled = F.adaptive_max_pool2d(chunk, (ph, pw))
            convd = F.conv2d(pooled, w, b, padding=1, groups=chunk_dim)
            out = F.interpolate(convd, size=(H, W), mode='nearest')
        outs.append(out)

    out = torch.cat(outs, dim=1)
    out = F.conv2d(out, aggr_w, aggr_b)  # 1x1 conv
    out = F.gelu(out)
    out = out * x
    return out


def ccm_forward(x, conv1_w, conv1_b, conv2_w, conv2_b):
    """CCM: convolutional channel mixing."""
    x = F.conv2d(x, conv1_w, conv1_b, padding=1)
    x = F.gelu(x)
    x = F.conv2d(x, conv2_w, conv2_b)
    return x


def pixel_shuffle(x, scale):
    B, C, H, W = x.shape
    c_out = C // (scale * scale)
    x = x.view(B, c_out, scale, scale, H, W)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    x = x.view(B, c_out, H * scale, W * scale)
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--image", help="Test image (PNG/JPG)")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--scale", type=int, default=4)
    args = parser.parse_args()

    # Load weights
    sd = torch.load(args.model, map_location='cpu', weights_only=True)
    if 'params' in sd: sd = sd['params']
    elif 'state_dict' in sd: sd = sd['state_dict']

    # Prepare input
    if args.image:
        from PIL import Image
        img = Image.open(args.image).convert("RGB").resize((args.width, args.height))
        inp = np.array(img).astype(np.float32) / 255.0
    else:
        np.random.seed(42)
        inp = np.random.rand(args.height, args.width, 3).astype(np.float32)

    # [H, W, 3] → [1, 3, H, W]
    x = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).float()
    H, W = x.shape[2], x.shape[3]
    print(f"Input: {W}x{H}, scale={args.scale}")

    stages = {}
    stages["input"] = x.squeeze(0).numpy().copy()  # [3, H, W]

    # to_feat: Conv3x3
    x = F.conv2d(x, sd['to_feat.weight'], sd['to_feat.bias'], padding=1)
    stages["to_feat"] = x.squeeze(0).detach().numpy().copy()
    print(f"  to_feat: {list(x.shape[1:])}, range=[{x.min():.4f}, {x.max():.4f}]")

    residual = x.clone()

    # 8 AttBlocks
    n_blocks = 8
    for i in range(n_blocks):
        p = f"feats.{i}"

        # SAFM branch
        norm1 = ChannelLayerNorm(sd[f'{p}.norm1.weight'], sd[f'{p}.norm1.bias'])
        y = norm1(x)

        mfr = [(sd[f'{p}.safm.mfr.{j}.weight'], sd[f'{p}.safm.mfr.{j}.bias']) for j in range(4)]
        y = safm_forward(y, mfr, sd[f'{p}.safm.aggr.weight'], sd[f'{p}.safm.aggr.bias'])
        x = x + y

        # CCM branch
        norm2 = ChannelLayerNorm(sd[f'{p}.norm2.weight'], sd[f'{p}.norm2.bias'])
        y = norm2(x)
        y = ccm_forward(y, sd[f'{p}.ccm.ccm.0.weight'], sd[f'{p}.ccm.ccm.0.bias'],
                         sd[f'{p}.ccm.ccm.2.weight'], sd[f'{p}.ccm.ccm.2.bias'])
        x = x + y

        stages[f"block_{i}"] = x.squeeze(0).detach().numpy().copy()
        print(f"  block_{i}: range=[{x.min():.4f}, {x.max():.4f}]")

    # Global skip
    x = x + residual
    stages["post_skip"] = x.squeeze(0).detach().numpy().copy()

    # to_img: Conv3x3 → PixelShuffle
    x = F.conv2d(x, sd['to_img.0.weight'], sd['to_img.0.bias'], padding=1)
    stages["to_img_conv"] = x.squeeze(0).detach().numpy().copy()

    x = pixel_shuffle(x, args.scale)
    stages["output"] = x.squeeze(0).detach().numpy().copy()
    print(f"  output: {list(x.shape[1:])}, range=[{x.min():.4f}, {x.max():.4f}]")

    # Write GGUF
    writer = gguf.GGUFWriter(args.output, "safmn-reference")
    writer.add_uint32("safmn.ref.width", W)
    writer.add_uint32("safmn.ref.height", H)
    writer.add_uint32("safmn.ref.scale", args.scale)
    writer.add_uint32("safmn.ref.n_blocks", n_blocks)

    for name, arr in stages.items():
        writer.add_tensor(name, arr.astype(np.float32),
                          raw_dtype=gguf.GGMLQuantizationType.F32)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"\nReference: {args.output} ({Path(args.output).stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
