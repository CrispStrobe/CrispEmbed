#!/usr/bin/env python3
"""Dump AdaIR output reference for parity testing.

Runs the actual PyTorch AdaIR model (from source) to get ground truth.
For simplicity, only dumps the final output — per-stage debugging can
be added later if needed.

Usage:
    python tools/dump_adair_reference.py \
        --model /mnt/storage/models/adair5d.ckpt \
        --source /tmp/adair_model.py \
        --output /tmp/adair-ref.gguf [--width 64] [--height 64]
"""
import argparse, sys, importlib.util
from pathlib import Path
import numpy as np, torch
try:
    import gguf
except ImportError:
    sys.exit("pip install gguf")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--source", required=True, help="Path to adair model.py")
    p.add_argument("--output", "-o", required=True)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--height", type=int, default=32)
    args = p.parse_args()

    # Dynamically load the AdaIR model class
    spec = importlib.util.spec_from_file_location("adair_model", args.source)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    model = mod.AdaIR()
    ckpt = torch.load(args.model, map_location='cpu', weights_only=False)
    if 'state_dict' in ckpt:
        sd = {k.replace('net.', ''): v for k, v in ckpt['state_dict'].items()}
    else:
        sd = ckpt
    model.load_state_dict(sd, strict=False)
    model.eval()

    W, H = args.width, args.height
    np.random.seed(42)
    inp = np.random.rand(H, W, 3).astype(np.float32)
    x = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).float()

    with torch.no_grad():
        out = model(x)

    print(f"Input: {W}x{H}")
    print(f"Output: {list(out.shape[1:])}, range=[{out.min():.4f}, {out.max():.4f}]")

    writer = gguf.GGUFWriter(args.output, "adair-reference")
    writer.add_tensor("input", x.squeeze(0).numpy().astype(np.float32),
                      raw_dtype=gguf.GGMLQuantizationType.F32)
    writer.add_tensor("output", out.squeeze(0).detach().numpy().astype(np.float32),
                      raw_dtype=gguf.GGMLQuantizationType.F32)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Reference: {args.output} ({Path(args.output).stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
