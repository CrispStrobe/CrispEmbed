#!/usr/bin/env python3
"""Dump Real-ESRGAN (SRVGGNetCompact) reference activations for parity testing.

Usage:
    python tools/dump_esrgan_reference.py \
        --model /mnt/storage/models/realesr-animevideov3.pth \
        --output /tmp/esrgan-ref.gguf [--width 64] [--height 64]
"""
import argparse, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
try:
    import gguf
except ImportError:
    sys.exit("pip install gguf")


def pixel_shuffle(x, scale):
    B, C, H, W = x.shape
    c_out = C // (scale * scale)
    return x.view(B, c_out, scale, scale, H, W).permute(0, 1, 4, 2, 5, 3).contiguous().view(B, c_out, H * scale, W * scale)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--output", "-o", required=True)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--height", type=int, default=64)
    p.add_argument("--scale", type=int, default=4)
    args = p.parse_args()

    sd = torch.load(args.model, map_location='cpu', weights_only=False)
    if 'params' in sd: sd = sd['params']
    elif 'params_ema' in sd: sd = sd['params_ema']

    W, H = args.width, args.height
    np.random.seed(42)
    inp = np.random.rand(H, W, 3).astype(np.float32)
    x = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).float()

    stages = {}
    stages["input"] = x.squeeze(0).numpy().copy()

    # Forward pass: sequential Conv+PReLU stack
    out = x
    # body.0 = Conv, body.1 = PReLU, body.2 = Conv, body.3 = PReLU, ... body.34 = Conv (output)
    n_layers = max(int(k.split('.')[1]) for k in sd.keys()) + 1
    for i in range(n_layers):
        w = sd.get(f'body.{i}.weight')
        b = sd.get(f'body.{i}.bias')
        if b is not None:
            # Conv2d
            out = F.conv2d(out, w, b, padding=1)
            stages[f"conv_{i}"] = out.squeeze(0).detach().numpy().copy()
        else:
            # PReLU (weight shape [num_parameters])
            out = F.prelu(out, w)

    print(f"Input: {W}x{H}, scale={args.scale}")
    print(f"  Pre-shuffle: {list(out.shape[1:])}")

    # PixelShuffle
    sr = pixel_shuffle(out, args.scale)

    # Global residual: nearest-upsample input + sr
    base = F.interpolate(x, scale_factor=float(args.scale), mode='nearest')
    result = sr + base
    stages["output"] = result.squeeze(0).detach().numpy().copy()
    print(f"  Output: {list(result.shape[1:])}, range=[{result.min():.4f}, {result.max():.4f}]")

    # Write GGUF
    writer = gguf.GGUFWriter(args.output, "esrgan-reference")
    writer.add_uint32("esrgan.ref.width", W)
    writer.add_uint32("esrgan.ref.height", H)
    writer.add_uint32("esrgan.ref.scale", args.scale)
    for name, arr in stages.items():
        writer.add_tensor(name, arr.astype(np.float32), raw_dtype=gguf.GGMLQuantizationType.F32)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"\nReference: {args.output} ({Path(args.output).stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
