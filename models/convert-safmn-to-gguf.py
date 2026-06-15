#!/usr/bin/env python3
"""Convert SAFMN super-resolution model to GGUF.

Usage:
    python models/convert-safmn-to-gguf.py \
        --model /mnt/storage/models/SAFMN_DF2K_x4.pth \
        --output safmn-x4-f32.gguf [--fp16] [--scale 4]
"""

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

try:
    import gguf
except ImportError:
    sys.exit("pip install gguf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--dim", type=int, default=36)
    parser.add_argument("--n-blocks", type=int, default=8)
    args = parser.parse_args()

    sd = torch.load(args.model, map_location='cpu', weights_only=True)
    if 'params' in sd: sd = sd['params']
    elif 'state_dict' in sd: sd = sd['state_dict']

    tensors = OrderedDict()
    for k, v in sorted(sd.items()):
        tensors[k] = v.float().numpy()

    # Write GGUF
    writer = gguf.GGUFWriter(args.output, "safmn")

    writer.add_uint32("safmn.scale", args.scale)
    writer.add_uint32("safmn.dim", args.dim)
    writer.add_uint32("safmn.n_blocks", args.n_blocks)
    writer.add_uint32("safmn.n_levels", 4)
    writer.add_uint32("safmn.ffn_scale", 2)

    dtype = gguf.GGMLQuantizationType.F16 if args.fp16 else gguf.GGMLQuantizationType.F32

    total_params = 0
    for name, arr in tensors.items():
        total_params += arr.size
        arr = arr.astype(np.float32)
        if ".bias" in name or "norm" in name:
            writer.add_tensor(name, arr, raw_dtype=gguf.GGMLQuantizationType.F32)
        else:
            writer.add_tensor(name, arr, raw_dtype=dtype)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size = Path(args.output).stat().st_size
    print(f"Written {args.output}: {total_params:,} params, {size/1024:.0f} KB")
    print(f"  Scale: x{args.scale}, dim: {args.dim}, blocks: {args.n_blocks}")


if __name__ == "__main__":
    main()
