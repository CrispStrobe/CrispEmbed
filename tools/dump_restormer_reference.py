#!/usr/bin/env python3
"""Dump Restormer per-stage reference activations for crispembed-diff.

Uses PyTorch with the original Restormer architecture for exact reference.
Captures intermediate activations at each U-Net stage.

Usage:
    PYTHONNOUSERSITE=1 python tools/dump_restormer_reference.py \
        --model gaussian_color_denoising_blind.pth \
        --output /tmp/restormer-ref.gguf --size 64
"""

import argparse
import struct
import sys
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Minimal Restormer reimplementation ─────────────────────────────────

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))

    def forward(self, x):
        # x: [B, C, H, W] → reshape to [B, HW, C], norm, reshape back
        B, C, H, W = x.shape
        x3 = x.reshape(B, C, H * W).permute(0, 2, 1)  # [B, HW, C]
        sigma = x3.var(-1, keepdim=True, unbiased=False)
        x3 = x3 / torch.sqrt(sigma + 1e-5) * self.weight
        return x3.permute(0, 2, 1).reshape(B, C, H, W)


class WithBias_LayerNorm(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))

    def forward(self, x):
        B, C, H, W = x.shape
        x3 = x.reshape(B, C, H * W).permute(0, 2, 1)
        mu = x3.mean(-1, keepdim=True)
        sigma = x3.var(-1, keepdim=True, unbiased=False)
        x3 = (x3 - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias
        return x3.permute(0, 2, 1).reshape(B, C, H, W)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, 3, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.reshape(B, self.num_heads, C // self.num_heads, H * W)
        k = k.reshape(B, self.num_heads, C // self.num_heads, H * W)
        v = v.reshape(B, self.num_heads, C // self.num_heads, H * W)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v).reshape(B, C, H, W)
        return self.project_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()
        hidden = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, ln_type):
        super().__init__()
        LN = BiasFree_LayerNorm if ln_type == "BiasFree" else WithBias_LayerNorm
        self.norm1 = LN(dim)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LN(dim)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class Restormer(nn.Module):
    def __init__(self, dim=48, num_blocks=[4, 6, 6, 8], heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66, bias=False, ln_type="WithBias",
                 num_refinement_blocks=4):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, 3, padding=1, bias=bias)

        self.encoder_level1 = nn.Sequential(
            *[TransformerBlock(dim, heads[0], ffn_expansion_factor, bias, ln_type)
              for _ in range(num_blocks[0])])
        self.down1_2 = nn.Sequential(nn.Conv2d(dim, dim // 2, 3, padding=1, bias=False), nn.PixelUnshuffle(2))

        self.encoder_level2 = nn.Sequential(
            *[TransformerBlock(dim * 2, heads[1], ffn_expansion_factor, bias, ln_type)
              for _ in range(num_blocks[1])])
        self.down2_3 = nn.Sequential(nn.Conv2d(dim * 2, dim, 3, padding=1, bias=False), nn.PixelUnshuffle(2))

        self.encoder_level3 = nn.Sequential(
            *[TransformerBlock(dim * 4, heads[2], ffn_expansion_factor, bias, ln_type)
              for _ in range(num_blocks[2])])
        self.down3_4 = nn.Sequential(nn.Conv2d(dim * 4, dim * 2, 3, padding=1, bias=False), nn.PixelUnshuffle(2))

        self.latent = nn.Sequential(
            *[TransformerBlock(dim * 8, heads[3], ffn_expansion_factor, bias, ln_type)
              for _ in range(num_blocks[3])])

        self.up4_3 = nn.Sequential(nn.Conv2d(dim * 8, dim * 16, 3, padding=1, bias=False), nn.PixelShuffle(2))
        self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, 1, bias=bias)
        self.decoder_level3 = nn.Sequential(
            *[TransformerBlock(dim * 4, heads[2], ffn_expansion_factor, bias, ln_type)
              for _ in range(num_blocks[2])])

        self.up3_2 = nn.Sequential(nn.Conv2d(dim * 4, dim * 8, 3, padding=1, bias=False), nn.PixelShuffle(2))
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, 1, bias=bias)
        self.decoder_level2 = nn.Sequential(
            *[TransformerBlock(dim * 2, heads[1], ffn_expansion_factor, bias, ln_type)
              for _ in range(num_blocks[1])])

        self.up2_1 = nn.Sequential(nn.Conv2d(dim * 2, dim * 4, 3, padding=1, bias=False), nn.PixelShuffle(2))
        self.decoder_level1 = nn.Sequential(
            *[TransformerBlock(dim * 2, heads[0], ffn_expansion_factor, bias, ln_type)
              for _ in range(num_blocks[0])])

        self.refinement = nn.Sequential(
            *[TransformerBlock(dim * 2, heads[0], ffn_expansion_factor, bias, ln_type)
              for _ in range(num_refinement_blocks)])

        self.output = nn.Conv2d(dim * 2, 3, 3, padding=1, bias=bias)

    def forward_with_intermediates(self, inp_img):
        intermediates = {"input": inp_img[0].detach().numpy().copy()}

        x = self.patch_embed(inp_img)
        intermediates["patch_embed"] = x[0].detach().numpy().copy()

        enc1 = self.encoder_level1(x)
        intermediates["enc1"] = enc1[0].detach().numpy().copy()

        x2 = self.down1_2(enc1)
        enc2 = self.encoder_level2(x2)
        intermediates["enc2"] = enc2[0].detach().numpy().copy()

        x3 = self.down2_3(enc2)
        enc3 = self.encoder_level3(x3)
        intermediates["enc3"] = enc3[0].detach().numpy().copy()

        x4 = self.down3_4(enc3)
        lat = self.latent(x4)
        intermediates["latent"] = lat[0].detach().numpy().copy()

        d3 = self.up4_3(lat)
        d3 = torch.cat([d3, enc3], 1)
        d3 = self.reduce_chan_level3(d3)
        d3 = self.decoder_level3(d3)
        intermediates["dec3"] = d3[0].detach().numpy().copy()

        d2 = self.up3_2(d3)
        d2 = torch.cat([d2, enc2], 1)
        d2 = self.reduce_chan_level2(d2)
        d2 = self.decoder_level2(d2)
        intermediates["dec2"] = d2[0].detach().numpy().copy()

        d1 = self.up2_1(d2)
        d1 = torch.cat([d1, enc1], 1)
        d1 = self.decoder_level1(d1)
        intermediates["dec1"] = d1[0].detach().numpy().copy()

        d1 = self.refinement(d1)
        intermediates["refined"] = d1[0].detach().numpy().copy()

        out = self.output(d1) + inp_img
        intermediates["output"] = out[0].detach().numpy().copy()

        return out, intermediates


# ── GGUF writer ─────────────────────────────────────────────────────────

def write_gguf(path, tensors):
    MAGIC = 0x46554747; VERSION = 3; TYPE_STRING = 8; TYPE_F32 = 0
    def ws(f, s):
        b = s.encode("utf-8"); f.write(struct.pack("<Q", len(b))); f.write(b)
    tensor_list = list(tensors.items())
    with open(path, "wb") as f:
        f.write(struct.pack("<I", MAGIC)); f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<Q", len(tensor_list))); f.write(struct.pack("<Q", 1))
        ws(f, "general.architecture"); f.write(struct.pack("<I", TYPE_STRING)); ws(f, "restormer_ref")
        offset = 0
        for name, data in tensor_list:
            ws(f, name); f.write(struct.pack("<I", len(data.shape)))
            for d in data.shape: f.write(struct.pack("<Q", d))
            f.write(struct.pack("<I", TYPE_F32)); f.write(struct.pack("<Q", offset))
            offset += data.nbytes; offset = (offset + 31) & ~31
        pos = f.tell(); aligned = (pos + 31) & ~31; f.write(b"\x00" * (aligned - pos))
        for name, data in tensor_list:
            f.write(data.astype(np.float32).tobytes())
            pad = ((data.nbytes + 31) & ~31) - data.nbytes
            if pad > 0: f.write(b"\x00" * pad)
    print(f"Written {path}: {len(tensor_list)} tensors")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, default=64)
    args = parser.parse_args()

    state = torch.load(args.model, map_location="cpu", weights_only=True)
    if "params" in state: state = state["params"]

    # Detect config from weights
    dim = state["patch_embed.proj.weight"].shape[0]
    num_blocks = []
    for name in ["encoder_level1", "encoder_level2", "encoder_level3", "latent"]:
        n = 0
        while f"{name}.{n}.norm1.body.weight" in state: n += 1
        num_blocks.append(n)
    heads = [state[f"{n}.0.attn.temperature"].shape[0]
             for n in ["encoder_level1", "encoder_level2", "encoder_level3", "latent"]]
    # Use 2.66 (the canonical Restormer value) — deriving from weights loses
    # precision due to int() rounding at different dim levels.
    ffn_factor = 2.66
    n_refine = 0
    while f"refinement.{n_refine}.norm1.body.weight" in state: n_refine += 1
    has_bias = "patch_embed.proj.bias" in state
    ln_type = "WithBias" if any("norm1.body.bias" in k for k in state) else "BiasFree"

    print(f"Restormer: dim={dim}, blocks={num_blocks}, heads={heads}, "
          f"ffn={ffn_factor:.2f}, refine={n_refine}, bias={has_bias}, ln={ln_type}")

    model = Restormer(dim=dim, num_blocks=num_blocks, heads=heads,
                      ffn_expansion_factor=ffn_factor, bias=has_bias,
                      ln_type=ln_type, num_refinement_blocks=n_refine)
    # Remap checkpoint keys to match our minimal reimplementation.
    remapped = {}
    for k, v in state.items():
        nk = k
        # LayerNorm: .norm1.body.weight → .norm1.weight
        nk = nk.replace(".body.weight", ".weight").replace(".body.bias", ".bias")
        # patch_embed: .proj.weight → .weight
        nk = nk.replace("patch_embed.proj.", "patch_embed.")
        # Downsample/Upsample: .body.0.weight → .0.weight
        for prefix in ["down1_2", "down2_3", "down3_4", "up4_3", "up3_2", "up2_1"]:
            nk = nk.replace(f"{prefix}.body.0.", f"{prefix}.0.")
        remapped[nk] = v
    result = model.load_state_dict(remapped, strict=False)
    if result.missing_keys:
        print(f"WARNING: {len(result.missing_keys)} missing keys: {result.missing_keys[:5]}")
    if result.unexpected_keys:
        print(f"WARNING: {len(result.unexpected_keys)} unexpected keys: {result.unexpected_keys[:5]}")
    model.eval()
    print(f"Loaded: {sum(p.numel() for p in model.parameters()):,} params")

    torch.manual_seed(42)
    inp = torch.rand(1, 3, args.size, args.size)

    with torch.no_grad():
        out, intermediates = model.forward_with_intermediates(inp)

    print(f"\nIntermediate activations:")
    for name, data in intermediates.items():
        print(f"  {name:15s}  shape={str(list(data.shape)):30s}  mean={data.mean():.6f}")

    write_gguf(args.output, intermediates)


if __name__ == "__main__":
    main()
