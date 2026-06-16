#!/usr/bin/env python3
"""Convert InstructIR to GGUF with pre-computed task embeddings.

Bakes the 7 task prompt embeddings directly into the GGUF so no text
encoder is needed at runtime. Task selection by integer ID (0-6).

Usage:
    python models/convert-instructir-to-gguf.py \
        --model /mnt/storage/models/im_instructir-7d.pt \
        --embeddings /tmp/instructir_task_embeddings.npz \
        --output instructir-f32.gguf [--fp16]
"""
import argparse, sys
from collections import OrderedDict
from pathlib import Path
import numpy as np, torch
try:
    import gguf
except ImportError:
    sys.exit("pip install gguf")

TASK_NAMES = ['denoise', 'deblur', 'dehaze', 'derain', 'super_resolution', 'low_light', 'enhance']

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--embeddings", required=True, help="Task embeddings .npz from pre-compute step")
    p.add_argument("--output", "-o", required=True)
    p.add_argument("--fp16", action="store_true")
    args = p.parse_args()

    sd = torch.load(args.model, map_location='cpu', weights_only=False)
    if 'params' in sd: sd = sd['params']
    embs = np.load(args.embeddings)

    writer = gguf.GGUFWriter(args.output, "instructir")
    writer.add_uint32("instructir.n_tasks", len(TASK_NAMES))
    writer.add_uint32("instructir.emb_dim", 256)
    writer.add_uint32("instructir.base_dim", 32)
    writer.add_array("instructir.enc_blocks", [2, 2, 4, 8])
    writer.add_array("instructir.dec_blocks", [2, 2, 2, 2])
    writer.add_uint32("instructir.middle_blocks", 4)
    writer.add_array("instructir.task_names", TASK_NAMES)

    dtype = gguf.GGMLQuantizationType.F16 if args.fp16 else gguf.GGMLQuantizationType.F32
    total = 0

    # Write task embeddings as a single [7, 256] tensor
    emb_matrix = np.stack([embs[name] for name in TASK_NAMES], axis=0).astype(np.float32)
    writer.add_tensor("task_embeddings", emb_matrix, raw_dtype=gguf.GGMLQuantizationType.F32)
    total += emb_matrix.size

    # Write all image model weights
    for k in sorted(sd.keys()):
        arr = sd[k].float().numpy()
        total += arr.size
        if ".bias" in k or "norm" in k or "beta" in k or "gamma" in k or arr.size < 256:
            writer.add_tensor(k, arr, raw_dtype=gguf.GGMLQuantizationType.F32)
        else:
            writer.add_tensor(k, arr, raw_dtype=dtype)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Written {args.output}: {total:,} params, {Path(args.output).stat().st_size/1024:.0f} KB")
    print(f"  Tasks: {', '.join(TASK_NAMES)}")

if __name__ == "__main__":
    main()
