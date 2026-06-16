#!/usr/bin/env python3
"""Dump InstructIR per-stage reference activations for parity testing.

Usage:
    python tools/dump_instructir_reference.py \
        --model /mnt/storage/models/im_instructir-7d.pt \
        --embeddings /tmp/instructir_task_embeddings.npz \
        --output /tmp/instructir-ref.gguf [--task denoise] [--width 64] [--height 64]
"""
import argparse, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
try:
    import gguf
except ImportError:
    sys.exit("pip install gguf")

# ── NAFBlock forward (matches nafnet_denoise pattern) ──

def layernorm2d(x, w, b):
    B, C, H, W = x.shape
    mean = x.mean(dim=1, keepdim=True)
    var = x.var(dim=1, keepdim=True, unbiased=False)
    return (x - mean) / (var + 1e-6).sqrt() * w.view(1, C, 1, 1) + b.view(1, C, 1, 1)

def simple_gate(x):
    c = x.shape[1] // 2
    return x[:, :c] * x[:, c:]

def nafblock(x, sd, prefix):
    beta = sd[f'{prefix}.beta']
    gamma = sd[f'{prefix}.gamma']
    # Spatial mixing
    y = layernorm2d(x, sd[f'{prefix}.norm1.weight'], sd[f'{prefix}.norm1.bias'])
    y = F.conv2d(y, sd[f'{prefix}.conv1.weight'], sd[f'{prefix}.conv1.bias'])
    y = F.conv2d(y, sd[f'{prefix}.conv2.weight'], sd[f'{prefix}.conv2.bias'],
                 padding=1, groups=y.shape[1])
    y = simple_gate(y)
    # SCA
    pool = y.mean(dim=[2, 3], keepdim=True)
    pool = F.conv2d(pool, sd[f'{prefix}.sca.1.weight'], sd[f'{prefix}.sca.1.bias'])
    y = y * pool
    y = F.conv2d(y, sd[f'{prefix}.conv3.weight'], sd[f'{prefix}.conv3.bias'])
    x = x + y * beta
    # Channel mixing
    y = layernorm2d(x, sd[f'{prefix}.norm2.weight'], sd[f'{prefix}.norm2.bias'])
    y = F.conv2d(y, sd[f'{prefix}.conv4.weight'], sd[f'{prefix}.conv4.bias'])
    y = simple_gate(y)
    y = F.conv2d(y, sd[f'{prefix}.conv5.weight'], sd[f'{prefix}.conv5.bias'])
    return x + y * gamma

def icb(x, text_embd, sd, prefix):
    """Instruction Condition Block: sigmoid gating + NAFBlock."""
    beta = sd[f'{prefix}.beta']
    gamma = sd[f'{prefix}.gamma']
    # Gating from text embedding
    gate = torch.sigmoid(F.linear(text_embd, sd[f'{prefix}.fc.weight'], sd[f'{prefix}.fc.bias']))
    gate = gate.view(1, -1, 1, 1)  # [1, C, 1, 1]
    y = x * gamma + beta
    y = y * gate
    y = nafblock(y, sd, f'{prefix}.block')
    return y + x

def pixel_unshuffle(x, scale):
    B, C, H, W = x.shape
    return x.view(B, C, H // scale, scale, W // scale, scale).permute(0, 1, 3, 5, 2, 4).contiguous().view(B, C * scale * scale, H // scale, W // scale)

def pixel_shuffle(x, scale):
    B, C, H, W = x.shape
    c_out = C // (scale * scale)
    return x.view(B, c_out, scale, scale, H, W).permute(0, 1, 4, 2, 5, 3).contiguous().view(B, c_out, H * scale, W * scale)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--embeddings", required=True)
    p.add_argument("--output", "-o", required=True)
    p.add_argument("--task", default="denoise", choices=['denoise','deblur','dehaze','derain','super_resolution','low_light','enhance'])
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--height", type=int, default=64)
    args = p.parse_args()

    sd = torch.load(args.model, map_location='cpu', weights_only=False)
    if 'params' in sd: sd = sd['params']
    embs = np.load(args.embeddings)
    text_embd = torch.from_numpy(embs[args.task]).float()  # [256]

    W, H = args.width, args.height
    np.random.seed(42)
    inp = np.random.rand(H, W, 3).astype(np.float32)
    x = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).float()
    print(f"Input: {W}x{H}, task={args.task}")

    stages = {}
    stages["input"] = x.squeeze(0).numpy().copy()

    # Intro conv
    x = F.conv2d(x, sd['intro.weight'], sd['intro.bias'], padding=1)
    stages["intro"] = x.squeeze(0).detach().numpy().copy()

    # Encoder
    enc_blocks = [2, 2, 4, 8]
    skips = []
    channels = [32, 64, 128, 256]
    for lvl in range(4):
        n_blks = enc_blocks[lvl]
        for i in range(n_blks):
            x = nafblock(x, sd, f'encoders.{lvl}.{i}')
        x = icb(x, text_embd, sd, f'enc_cond.{lvl}')
        skips.append(x.clone())
        stages[f"enc_{lvl}"] = x.squeeze(0).detach().numpy().copy()
        print(f"  enc_{lvl}: {list(x.shape[1:])}")
        # Downsample: Conv2d(C→2C, k=2, s=2)
        x = F.conv2d(x, sd[f'downs.{lvl}.weight'], sd[f'downs.{lvl}.bias'], stride=2)

    # Middle
    for i in range(4):
        x = nafblock(x, sd, f'middle_blks.{i}')
    stages["middle"] = x.squeeze(0).detach().numpy().copy()
    print(f"  middle: {list(x.shape[1:])}")

    # Decoder
    dec_blocks = [2, 2, 2, 2]
    for lvl in range(4):
        # Upsample
        up_w = sd[f'ups.{lvl}.0.weight']
        up_b = sd.get(f'ups.{lvl}.0.bias')
        x = F.conv2d(x, up_w, up_b)
        x = pixel_shuffle(x, 2)
        # Skip connection
        x = x + skips[3 - lvl]
        n_blks = dec_blocks[lvl]
        for i in range(n_blks):
            x = nafblock(x, sd, f'decoders.{lvl}.{i}')
        x = icb(x, text_embd, sd, f'dec_cond.{lvl}')
        stages[f"dec_{lvl}"] = x.squeeze(0).detach().numpy().copy()
        print(f"  dec_{lvl}: {list(x.shape[1:])}")

    # Ending conv + residual
    x = F.conv2d(x, sd['ending.weight'], sd['ending.bias'], padding=1)
    inp_tensor = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).float()
    x = x + inp_tensor
    stages["output"] = x.squeeze(0).detach().numpy().copy()
    print(f"  output: {list(x.shape[1:])}, range=[{x.min():.4f}, {x.max():.4f}]")

    # Write GGUF
    writer = gguf.GGUFWriter(args.output, "instructir-reference")
    writer.add_uint32("instructir.ref.width", W)
    writer.add_uint32("instructir.ref.height", H)
    for name, arr in stages.items():
        writer.add_tensor(name, arr.astype(np.float32), raw_dtype=gguf.GGMLQuantizationType.F32)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"\nReference: {args.output} ({Path(args.output).stat().st_size / 1024:.0f} KB)")

if __name__ == "__main__":
    main()
