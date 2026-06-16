#!/usr/bin/env python3
"""Dump HAT per-stage reference activations for crispembed-diff.

Uses PyTorch with the original HAT architecture for exact reference.

Usage:
    PYTHONNOUSERSITE=1 python tools/dump_hat_reference.py \
        --model HAT_SRx4_ImageNet-pretrain.pth \
        --output /tmp/hat-ref.gguf --size 64
"""

import argparse
import math
import struct
import sys
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add the hat_arch module path
sys.path.insert(0, os.path.dirname(__file__))

# We need basicsr utilities. Provide minimal stubs.
class _ArchUtil:
    @staticmethod
    def to_2tuple(x):
        return (x, x) if isinstance(x, int) else x
    @staticmethod
    def trunc_normal_(tensor, std=0.02):
        nn.init.trunc_normal_(tensor, std=std)

# Monkey-patch basicsr imports for standalone use
import types
basicsr = types.ModuleType("basicsr")
basicsr.utils = types.ModuleType("basicsr.utils")
basicsr.utils.registry = types.ModuleType("basicsr.utils.registry")
basicsr.archs = types.ModuleType("basicsr.archs")
basicsr.archs.arch_util = _ArchUtil
class _FakeRegistry:
    def register(self):
        return lambda cls: cls
basicsr.utils.registry.ARCH_REGISTRY = _FakeRegistry()
sys.modules["basicsr"] = basicsr
sys.modules["basicsr.utils"] = basicsr.utils
sys.modules["basicsr.utils.registry"] = basicsr.utils.registry
sys.modules["basicsr.archs"] = basicsr.archs
sys.modules["basicsr.archs.arch_util"] = basicsr.archs.arch_util

# Now we can import the real HAT arch
exec(open("tmp/hat_arch.py").read())


def write_gguf(path, tensors):
    MAGIC = 0x46554747; VERSION = 3; TYPE_STRING = 8; TYPE_F32 = 0
    def ws(f, s):
        b = s.encode("utf-8"); f.write(struct.pack("<Q", len(b))); f.write(b)
    tensor_list = list(tensors.items())
    with open(path, "wb") as f:
        f.write(struct.pack("<I", MAGIC)); f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<Q", len(tensor_list))); f.write(struct.pack("<Q", 1))
        ws(f, "general.architecture"); f.write(struct.pack("<I", TYPE_STRING)); ws(f, "hat_ref")
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
    if "params_ema" in state: state = state["params_ema"]
    elif "params" in state: state = state["params"]

    # Detect config from weights
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
    rpb_size = state["layers.0.residual_group.blocks.0.attn.relative_position_bias_table"].shape[0]
    window_size = int((math.sqrt(rpb_size) + 1) / 2)
    n_upsample = 0
    while f"upsample.{n_upsample * 2}.weight" in state:
        n_upsample += 1
    upscale = 2 ** n_upsample

    # Overlap ratio: rpb shape is (ws_ori + ws_ext - 1)^2
    # and ws_ext = int(ws * overlap_ratio) + ws
    # So (ws + ws_ext - 1)^2 = rpb_size → ws_ext = sqrt(rpb_size) - ws + 1
    ocab_rpb = state["layers.0.residual_group.overlap_attn.relative_position_bias_table"].shape[0]
    ws_plus_ext_minus1 = int(math.sqrt(ocab_rpb))
    overlap_win = ws_plus_ext_minus1 - window_size + 1
    overlap_ratio = (overlap_win - window_size) / window_size

    compress_ratio = embed_dim // state["layers.0.residual_group.blocks.0.conv_block.cab.0.weight"].shape[0]
    squeeze_factor = embed_dim // state["layers.0.residual_group.blocks.0.conv_block.cab.3.attention.1.weight"].shape[0]

    print(f"HAT: embed={embed_dim}, layers={n_layers}, depths={depths}")
    print(f"  heads={heads}, ws={window_size}, upscale={upscale}x, overlap={overlap_ratio:.1f}")

    # Build model
    model = HAT(
        img_size=args.size,
        in_chans=3,
        embed_dim=embed_dim,
        depths=tuple(depths),
        num_heads=tuple(heads),
        window_size=window_size,
        compress_ratio=compress_ratio,
        squeeze_factor=squeeze_factor,
        overlap_ratio=overlap_ratio,
        mlp_ratio=2.,
        upscale=upscale,
        upsampler='pixelshuffle',
        resi_connection='1conv',
    )
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded: {sum(p.numel() for p in model.parameters()):,} params")

    torch.manual_seed(42)
    inp = torch.rand(1, 3, args.size, args.size)

    intermediates = {"input": inp[0].detach().numpy().copy()}

    with torch.no_grad():
        # Step through forward manually to capture intermediates
        x = inp
        model.mean = model.mean.type_as(x)
        x = (x - model.mean) * model.img_range

        x = model.conv_first(x)
        intermediates["conv_first"] = x[0].detach().numpy().copy()

        residual = x
        x_size = (x.shape[2], x.shape[3])
        attn_mask = model.calculate_mask(x_size).to(x.device)
        params = {
            'attn_mask': attn_mask,
            'rpi_sa': model.relative_position_index_SA,
            'rpi_oca': model.relative_position_index_OCA
        }

        x = model.patch_embed(x)
        for i, layer in enumerate(model.layers):
            x = layer(x, x_size, params)
            # Unembed for visualization
            x_vis = model.patch_unembed(x, x_size)
            intermediates[f"rhag_{i}"] = x_vis[0].detach().numpy().copy()

        x = model.norm(x)
        x = model.patch_unembed(x, x_size)
        x = model.conv_after_body(x) + residual
        intermediates["deep_features"] = x[0].detach().numpy().copy()

        x = model.conv_before_upsample(x)
        x = model.conv_last(model.upsample(x))
        intermediates["pre_output"] = x[0].detach().numpy().copy()

        x = x / model.img_range + model.mean
        intermediates["output"] = np.clip(x[0].detach().numpy(), 0, 1).copy()

    print(f"\nIntermediate activations:")
    for name, data in intermediates.items():
        print(f"  {name:20s}  shape={str(list(data.shape)):30s}  mean={data.mean():.6f}")

    write_gguf(args.output, intermediates)


if __name__ == "__main__":
    main()
