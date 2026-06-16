#!/usr/bin/env python3
"""Dump DeepSeek-OCR-2 reference outputs for crispembed-diff parity testing.

Runs the PyTorch model on a test image, captures per-layer intermediates,
and writes them as a binary reference file for the C++ diff harness.

Usage:
    python tools/dump_deepseek_ocr2_reference.py \
        --model-dir /path/to/DeepSeek-OCR-2 \
        --image test.png \
        --output /tmp/deepseek-ocr2-ref.bin

Reference file format (crispembed_diff compatible):
    For each stage: [name_len: u64] [name: bytes] [n_floats: u64] [data: f32...]
"""

import argparse
import struct
import sys
import os
import numpy as np
from pathlib import Path


def write_stage(f, name: str, data: np.ndarray):
    """Write one named stage to the binary reference file."""
    flat = data.astype(np.float32).flatten()
    name_bytes = name.encode("utf-8")
    f.write(struct.pack("<Q", len(name_bytes)))
    f.write(name_bytes)
    f.write(struct.pack("<Q", len(flat)))
    f.write(flat.tobytes())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True, help="HF model directory with safetensors + config")
    p.add_argument("--image", required=True, help="Input image path")
    p.add_argument("--output", required=True, help="Output binary reference file")
    p.add_argument("--max-tokens", type=int, default=64, help="Max generated tokens")
    args = p.parse_args()

    import torch
    from PIL import Image

    # Load model via transformers with trust_remote_code
    from transformers import AutoModel, AutoTokenizer

    print("Loading model...")
    model = AutoModel.from_pretrained(args.model_dir, trust_remote_code=True,
                                       torch_dtype=torch.float32)
    model.eval()

    # Load image
    img = Image.open(args.image).convert("RGB")
    print(f"Image: {img.size[0]}x{img.size[1]}")

    # Hook to capture intermediates
    captures = {}

    def hook(name):
        def fn(module, input, output):
            if isinstance(output, torch.Tensor):
                captures[name] = output.detach().cpu().numpy()
            elif isinstance(output, tuple) and len(output) > 0:
                if isinstance(output[0], torch.Tensor):
                    captures[name] = output[0].detach().cpu().numpy()
        return fn

    # Register hooks on key components
    if hasattr(model, 'sam_model'):
        model.sam_model.register_forward_hook(hook("sam_output"))
    if hasattr(model, 'qwen2_model'):
        model.qwen2_model.register_forward_hook(hook("qwen2_enc_output"))
    if hasattr(model, 'projector'):
        model.projector.register_forward_hook(hook("projector_output"))

    # Hook on LLM layers
    if hasattr(model, 'layers'):
        for i, layer in enumerate(model.layers):
            layer.register_forward_hook(hook(f"llm_layer_{i}"))

    # Run inference
    print("Running inference...")
    with torch.no_grad():
        result = model.generate_ocr_text(img) if hasattr(model, 'generate_ocr_text') else None

    if result:
        print(f"Output text: {result.get('text', '')[:200]}")

    # Write reference file
    print(f"Writing reference to {args.output}...")
    with open(args.output, "wb") as f:
        for name in sorted(captures.keys()):
            data = captures[name]
            write_stage(f, name, data)
            print(f"  {name}: shape={data.shape}, range=[{data.min():.4f}, {data.max():.4f}]")

    print(f"Wrote {len(captures)} stages to {args.output}")


if __name__ == "__main__":
    main()
