#!/usr/bin/env python3
"""Convert AdaIR checkpoint to GGUF.

Usage:
    python models/convert-adair-to-gguf.py \
        --model /mnt/storage/models/adair5d.ckpt \
        --output adair-5d-f32.gguf [--fp16]
"""
import argparse, sys
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
    args = p.parse_args()

    sd = torch.load(args.model, map_location='cpu', weights_only=False)
    if 'state_dict' in sd: sd = sd['state_dict']

    writer = gguf.GGUFWriter(args.output, "adair")
    writer.add_uint32("adair.base_dim", 48)
    writer.add_array("adair.enc_blocks", [4, 6, 6])
    writer.add_uint32("adair.latent_blocks", 8)
    writer.add_array("adair.dec_blocks", [6, 6, 4])
    writer.add_uint32("adair.refine_blocks", 4)

    dtype = gguf.GGMLQuantizationType.F16 if args.fp16 else gguf.GGMLQuantizationType.F32
    total = 0
    for k in sorted(sd.keys()):
        arr = sd[k].float().numpy()
        total += arr.size
        if ".bias" in k or "norm" in k or "temperature" in k or "para" in k or arr.size < 256:
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
