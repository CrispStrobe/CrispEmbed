#!/usr/bin/env python3
"""Dump NAFNet-SR per-stage reference activations for parity testing.

Loads a NAFNet-SR checkpoint, runs forward pass on a deterministic input,
captures intermediate activations at each encoder stage, middle, decoder
stage, and the final PixelShuffle output. Writes to GGUF for crispembed_diff.

Usage:
    python tools/dump_text_sr_reference.py \
        --model /mnt/storage/models/nafnet-sr-x2.pth \
        --output /tmp/text-sr-ref.gguf \
        --upscale 2
"""

import argparse
import struct

import numpy as np
import torch
import torch.nn as nn


# ── Minimal NAFNet-SR reimplementation ─────────────────────────────

class LayerNorm2d(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        var = x.var(1, keepdim=True, unbiased=False)
        return (x - mean) / (var + 1e-6).sqrt() * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        dw = c * 2
        self.conv1 = nn.Conv2d(c, dw, 1)
        self.conv2 = nn.Conv2d(dw, dw, 3, padding=1, groups=dw)
        self.sg = SimpleGate()
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c, 1))
        self.conv3 = nn.Conv2d(c, c, 1)
        self.conv4 = nn.Conv2d(c, dw, 1)
        self.conv5 = nn.Conv2d(c, c, 1)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        inp = x
        x = self.norm1(x); x = self.conv1(x); x = self.conv2(x); x = self.sg(x)
        x = x * self.sca(x); x = self.conv3(x); x = inp + x * self.beta
        inp2 = x
        x = self.norm2(x); x = self.conv4(x); x = self.sg(x); x = self.conv5(x)
        return inp2 + x * self.gamma


class NAFNetSR(nn.Module):
    """NAFNet for super-resolution: PixelShuffle ending + bicubic residual."""
    def __init__(self, w=32, upscale=2, enc=[2, 2, 4, 8], mid=12, dec=[2, 2, 2, 2]):
        super().__init__()
        self.upscale = upscale
        self.intro = nn.Conv2d(3, w, 3, padding=1)
        self.ending = nn.Conv2d(w, 3 * upscale * upscale, 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        c = w
        for n in enc:
            self.encoders.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
            self.downs.append(nn.Conv2d(c, c * 2, 2, 2))
            c *= 2
        self.middle_blks = nn.Sequential(*[NAFBlock(c) for _ in range(mid)])
        for n in dec:
            self.ups.append(nn.Sequential(nn.Conv2d(c, c * 2, 1, bias=False), nn.PixelShuffle(2)))
            c //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))

    def forward_with_intermediates(self, x):
        intermediates = {}
        inp = x

        x = self.intro(x)
        intermediates["intro"] = x[0].numpy().copy()

        skips = []
        for i, (enc, down) in enumerate(zip(self.encoders, self.downs)):
            x = enc(x)
            intermediates[f"enc_{i}"] = x[0].numpy().copy()
            skips.append(x)
            x = down(x)
            intermediates[f"down_{i}"] = x[0].numpy().copy()

        x = self.middle_blks(x)
        intermediates["middle"] = x[0].numpy().copy()

        for i, (dec, up, skip) in enumerate(zip(self.decoders, self.ups, reversed(skips))):
            x = up(x)
            x = x + skip
            x = dec(x)
            intermediates[f"dec_{i}"] = x[0].numpy().copy()

        # Ending: conv -> PixelShuffle (SR-specific)
        x = self.ending(x)
        intermediates["ending_pre_ps"] = x[0].numpy().copy()
        x = self.pixel_shuffle(x)
        intermediates["ending_post_ps"] = x[0].numpy().copy()

        # Bicubic residual
        bicubic = nn.functional.interpolate(
            inp, scale_factor=self.upscale, mode="bicubic", align_corners=False)
        x = x + bicubic
        intermediates["output"] = x[0].numpy().copy()

        return x, intermediates


# ── GGUF writer ──────────────────────────────────────────────────────

def write_gguf(path, tensors):
    MAGIC = 0x46554747
    VERSION = 3
    TYPE_STRING = 8
    TYPE_F32 = 0

    def write_string(f, s):
        b = s.encode("utf-8")
        f.write(struct.pack("<Q", len(b)))
        f.write(b)

    tensor_list = list(tensors.items())
    n_kv = 1
    n_tensors = len(tensor_list)

    with open(path, "wb") as f:
        f.write(struct.pack("<I", MAGIC))
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<Q", n_tensors))
        f.write(struct.pack("<Q", n_kv))

        write_string(f, "general.architecture")
        f.write(struct.pack("<I", TYPE_STRING))
        write_string(f, "text_sr_ref")

        offset = 0
        for name, data in tensor_list:
            write_string(f, name)
            f.write(struct.pack("<I", len(data.shape)))
            for d in data.shape:
                f.write(struct.pack("<Q", d))
            f.write(struct.pack("<I", TYPE_F32))
            f.write(struct.pack("<Q", offset))
            nbytes = data.nbytes
            offset += nbytes
            offset = (offset + 31) & ~31

        pos = f.tell()
        aligned = (pos + 31) & ~31
        f.write(b"\x00" * (aligned - pos))

        for name, data in tensor_list:
            f.write(data.astype(np.float32).tobytes())
            pad = ((data.nbytes + 31) & ~31) - data.nbytes
            if pad > 0:
                f.write(b"\x00" * pad)

    print(f"Written {path}: {n_tensors} tensors")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="NAFNet-SR .pth checkpoint")
    parser.add_argument("--output", required=True, help="Output reference GGUF")
    parser.add_argument("--upscale", type=int, default=2, choices=[2, 4],
                        help="Upscale factor (default: 2)")
    parser.add_argument("--size", type=int, default=64, help="Test image size (default 64)")
    args = parser.parse_args()

    state = torch.load(args.model, map_location="cpu", weights_only=True)
    if "params" in state:
        state = state["params"]
    elif "state_dict" in state:
        state = state["state_dict"]

    model = NAFNetSR(upscale=args.upscale)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded NAFNet-SR: {sum(p.numel() for p in model.parameters()):,} params, "
          f"{args.upscale}x upscale")

    # Deterministic input
    torch.manual_seed(42)
    inp = torch.rand(1, 3, args.size, args.size)

    with torch.no_grad():
        out, intermediates = model.forward_with_intermediates(inp)

    intermediates["input"] = inp[0].numpy().copy()

    print(f"\nIntermediate activations:")
    for name, data in intermediates.items():
        print(f"  {name:20s}  shape={list(data.shape)}  mean={data.mean():.6f}")

    write_gguf(args.output, intermediates)


if __name__ == "__main__":
    main()
