#!/usr/bin/env python3
"""Convert SCUNet denoising model to GGUF.

Usage:
    python models/convert-scunet-to-gguf.py \
        --model /mnt/storage/models/scunet_color_real_psnr.pth \
        --output scunet-color-f32.gguf [--fp16]
"""
import argparse, sys
from collections import OrderedDict
from pathlib import Path
import numpy as np, torch
try:
    import gguf
except ImportError:
    sys.exit("pip install gguf")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--output", "-o", required=True)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--win-size", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=32)
    args = p.parse_args()

    sd = torch.load(args.model, map_location='cpu', weights_only=False)

    writer = gguf.GGUFWriter(args.output, "scunet")
    writer.add_uint32("scunet.dim", args.dim)
    writer.add_uint32("scunet.win_size", args.win_size)
    writer.add_uint32("scunet.head_dim", args.head_dim)
    # config: [n_blocks_down1, down2, down3, body, up3, up2, up1]
    writer.add_array("scunet.config", [4, 4, 4, 4, 4, 4, 4])

    dtype = gguf.GGMLQuantizationType.F16 if args.fp16 else gguf.GGMLQuantizationType.F32
    total = 0
    for k in sorted(sd.keys()):
        arr = sd[k].float().numpy()
        total += arr.size
        # Keep biases, norms, RPB, and small tensors at F32
        if ".bias" in k or "ln" in k or "relative_position" in k or arr.size < 256:
            writer.add_tensor(k, arr, raw_dtype=gguf.GGMLQuantizationType.F32)
        else:
            writer.add_tensor(k, arr, raw_dtype=dtype)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Written {args.output}: {total:,} params, {Path(args.output).stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    main()
