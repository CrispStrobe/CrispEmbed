#!/usr/bin/env python3
"""Convert Real-ESRGAN SRVGGNetCompact to GGUF.

Usage:
    python models/convert-esrgan-to-gguf.py \
        --model /mnt/storage/models/realesr-animevideov3.pth \
        --output esrgan-x4-f32.gguf [--fp16] [--scale 4]
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
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--num-feat", type=int, default=64)
    p.add_argument("--num-conv", type=int, default=16)
    args = p.parse_args()

    sd = torch.load(args.model, map_location='cpu', weights_only=False)
    if 'params' in sd: sd = sd['params']
    elif 'params_ema' in sd: sd = sd['params_ema']

    writer = gguf.GGUFWriter(args.output, "esrgan")
    writer.add_uint32("esrgan.scale", args.scale)
    writer.add_uint32("esrgan.num_feat", args.num_feat)
    writer.add_uint32("esrgan.num_conv", args.num_conv)

    dtype = gguf.GGMLQuantizationType.F16 if args.fp16 else gguf.GGMLQuantizationType.F32
    total = 0
    for k in sorted(sd.keys(), key=lambda x: (int(x.split('.')[1]), x)):
        arr = sd[k].float().numpy()
        total += arr.size
        # Biases and PReLU slopes always F32
        if ".bias" in k or (arr.ndim == 1 and arr.shape[0] <= 64):
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
