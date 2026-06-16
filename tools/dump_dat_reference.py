#!/usr/bin/env python3
"""Dump DAT-light x2 reference outputs for parity testing.

Runs the PyTorch reference implementation on a test image and dumps
intermediate/final tensors as .npy files for comparison with the C engine.

Usage:
    python tools/dump_dat_reference.py --model DAT_light_x2.pth --image test.png --outdir /tmp/dat_ref
"""

import argparse
import sys
import os
import numpy as np
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to DAT_light_x2.pth")
    p.add_argument("--image", required=True, help="Input image path")
    p.add_argument("--outdir", default="/tmp/dat_ref", help="Output directory for .npy files")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    import torch
    from PIL import Image

    # Mock basicsr.utils.registry so we don't need the full basicsr package
    import types
    import importlib
    _basicsr = types.ModuleType("basicsr")
    _basicsr_utils = types.ModuleType("basicsr.utils")
    _basicsr_registry = types.ModuleType("basicsr.utils.registry")
    class _FakeRegistry:
        def register(self): return lambda cls: cls
    _basicsr_registry.ARCH_REGISTRY = _FakeRegistry()
    _basicsr.utils = _basicsr_utils
    _basicsr_utils.registry = _basicsr_registry
    sys.modules["basicsr"] = _basicsr
    sys.modules["basicsr.utils"] = _basicsr_utils
    sys.modules["basicsr.utils.registry"] = _basicsr_registry

    # Also need to mock basicsr.archs since it's a subpackage
    _basicsr_archs = types.ModuleType("basicsr.archs")
    sys.modules["basicsr.archs"] = _basicsr_archs
    _basicsr.archs = _basicsr_archs

    # Load the DAT arch file directly
    # dat-src may be in main repo or worktree
    dat_arch_path = Path("/mnt/volume1/CrispEmbed/tmp/dat-src/basicsr/archs/dat_arch.py")
    spec = importlib.util.spec_from_file_location("basicsr.archs.dat_arch", str(dat_arch_path))
    dat_mod = importlib.util.module_from_spec(spec)
    sys.modules["basicsr.archs.dat_arch"] = dat_mod
    spec.loader.exec_module(dat_mod)
    DAT = dat_mod.DAT

    model = DAT(
        upscale=2,
        in_chans=3,
        img_size=64,
        img_range=1.,
        depth=[18],
        embed_dim=60,
        num_heads=[6],
        expansion_factor=2,
        resi_connection='3conv',
        split_size=[8, 32],
        qkv_bias=True,
        upsampler='pixelshuffledirect',
    )

    sd = torch.load(args.model, map_location="cpu", weights_only=False)
    if "params_ema" in sd: sd = sd["params_ema"]
    elif "params" in sd: sd = sd["params"]
    model.load_state_dict(sd, strict=True)
    model.eval()

    # Load image
    img = Image.open(args.image).convert("RGB")
    img_np = np.array(img).astype(np.float32) / 255.0  # (H, W, 3)
    # Pad to multiple of 32
    h, w = img_np.shape[:2]
    max_sp = 32
    pad_h = ((h + max_sp - 1) // max_sp) * max_sp
    pad_w = ((w + max_sp - 1) // max_sp) * max_sp
    padded = np.zeros((pad_h, pad_w, 3), dtype=np.float32)
    padded[:h, :w, :] = img_np
    # Mirror pad
    if pad_h > h:
        for y in range(h, pad_h):
            sy = min(2*h - y - 2, h-1)
            if sy < 0: sy = 0
            padded[y, :w, :] = padded[sy, :w, :]
    if pad_w > w:
        for x in range(w, pad_w):
            sx = min(2*w - x - 2, w-1)
            if sx < 0: sx = 0
            padded[:, x, :] = padded[:, sx, :]

    x = torch.from_numpy(padded).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    print(f"Input shape: {x.shape}")

    # Hook to capture intermediates
    captures = {}

    def save_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                captures[name] = output.detach().cpu().numpy()
            elif isinstance(output, (tuple, list)) and len(output) > 0:
                captures[name] = output[0].detach().cpu().numpy() if isinstance(output[0], torch.Tensor) else None
        return hook

    # Register hooks
    model.conv_first.register_forward_hook(save_hook("conv_first"))
    for i, blk in enumerate(model.layers[0].blocks):
        blk.register_forward_hook(save_hook(f"block_{i}"))

    with torch.no_grad():
        out = model(x)

    print(f"Output shape: {out.shape}")

    # Save
    np.save(os.path.join(args.outdir, "input.npy"), x.numpy())
    np.save(os.path.join(args.outdir, "output.npy"), out.numpy())
    np.save(os.path.join(args.outdir, "conv_first.npy"), captures.get("conv_first"))
    for i in range(18):
        key = f"block_{i}"
        if key in captures:
            np.save(os.path.join(args.outdir, f"{key}.npy"), captures[key])

    # Also save final RGB
    out_img = out.squeeze(0).permute(1, 2, 0).numpy()
    out_img = np.clip(out_img * 255, 0, 255).astype(np.uint8)
    Image.fromarray(out_img[:h*2, :w*2]).save(os.path.join(args.outdir, "output.png"))

    print(f"Saved {len(captures) + 2} tensors to {args.outdir}")
    print(f"Output pixel range: [{out.min():.4f}, {out.max():.4f}]")


if __name__ == "__main__":
    main()
