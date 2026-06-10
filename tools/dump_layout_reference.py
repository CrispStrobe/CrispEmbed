#!/usr/bin/env python3
"""Dump per-stage reference activations for RT-DETRv2 layout detection model.

Loads the HuggingFace docling-project/docling-layout-heron model, runs inference
on a test image, captures intermediate activations at every architectural boundary
via forward hooks, and writes to a GGUF tensor archive for comparison with the
C++ crispembed implementation.

Stages captured:
  ip3    (H*W, 256)  encoder_input_proj[0] output (backbone P3 projection)
  ip4    (H*W, 256)  encoder_input_proj[1] output (backbone P4 projection)
  ip5    (H*W, 256)  encoder_input_proj[2] output (backbone P5 projection)
  s3     (H*W, 256)  encoder output feature map 0 (after AIFI+FPN+PAN)
  s4     (H*W, 256)  encoder output feature map 1
  s5     (H*W, 256)  encoder output feature map 2
  enc_output  (8400, 256)  enc_output (linear + layernorm) before decoder

Usage:
    python tools/dump_layout_reference.py [--image /path/to/image.png]

Default image: /tmp/test_layout.png
Output:        /tmp/layout-ref.gguf
"""

import sys
import importlib.util

_orig_find_spec = importlib.util.find_spec
def _patched_find_spec(name, *args, **kwargs):
    if name == 'mlx' or name.startswith('mlx.'):
        return None
    return _orig_find_spec(name, *args, **kwargs)
importlib.util.find_spec = _patched_find_spec

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import gguf


def create_test_layout_image(path="/tmp/test_layout.png"):
    """Create a synthetic 800x1100 document-like test image."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (800, 1100), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Text-like gray blocks (simulating paragraphs)
    for row in range(8):
        y = 60 + row * 40
        # Full-width line
        draw.rectangle([60, y, 740, y + 10], fill=(80, 80, 80))
        # Shorter line (paragraph end)
        if row % 4 == 3:
            draw.rectangle([60, y, 500, y + 10], fill=(80, 80, 80))

    # Second text block
    for row in range(6):
        y = 420 + row * 40
        draw.rectangle([60, y, 740, y + 10], fill=(80, 80, 80))

    # Table rectangle
    draw.rectangle([60, 650, 740, 850], outline=(50, 50, 50), width=2)
    # Table inner lines (columns)
    for x in [230, 400, 570]:
        draw.line([x, 650, x, 850], fill=(50, 50, 50), width=1)
    # Table inner lines (rows)
    for y in [700, 750, 800]:
        draw.line([60, y, 740, y], fill=(50, 50, 50), width=1)
    # Table header fill
    draw.rectangle([60, 650, 740, 700], fill=(200, 200, 200))

    # Figure placeholder (dashed rectangle with X)
    draw.rectangle([150, 900, 650, 1060], outline=(120, 120, 120), width=2)
    draw.line([150, 900, 650, 1060], fill=(180, 180, 180), width=1)
    draw.line([650, 900, 150, 1060], fill=(180, 180, 180), width=1)

    # Caption text block under figure
    draw.rectangle([200, 1070, 600, 1082], fill=(100, 100, 100))

    img.save(path)
    print(f"Created test image: {path} (800x1100)")
    return path


def main():
    p = argparse.ArgumentParser(description="RT-DETRv2 layout reference dumper")
    p.add_argument("--image", default="/tmp/test_layout.png",
                   help="Test image path (default: /tmp/test_layout.png)")
    p.add_argument("--output", default="/tmp/layout-ref.gguf",
                   help="Output GGUF path (default: /tmp/layout-ref.gguf)")
    p.add_argument("--model", default="docling-project/docling-layout-heron",
                   help="HuggingFace model ID or local path")
    args = p.parse_args()

    # Create test image if it doesn't exist
    if not os.path.exists(args.image):
        create_test_layout_image(args.image)

    # Load model and processor
    print(f"Loading model: {args.model}")
    from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor

    processor = RTDetrImageProcessor.from_pretrained(args.model)
    model = RTDetrV2ForObjectDetection.from_pretrained(args.model)
    model.eval()
    print("Model loaded successfully")

    # Print model structure summary
    inner = model.model
    print("\nModel structure:")
    print(f"  model.backbone:            {type(inner.backbone).__name__}")
    print(f"  model.encoder_input_proj:  {len(inner.encoder_input_proj)} projections")
    print(f"  model.encoder:             {type(inner.encoder).__name__}")
    print(f"  model.enc_output:          {type(inner.enc_output).__name__}")
    print(f"  model.decoder:             {type(inner.decoder).__name__}")

    # Preprocess image
    from PIL import Image
    print(f"\nPreprocessing image: {args.image}")
    image = Image.open(args.image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"]
    print(f"  Input tensor: {pixel_values.shape}, range [{pixel_values.min():.3f}, {pixel_values.max():.3f}]")

    # Register forward hooks
    captures = {}

    def make_hook(name):
        def hook_fn(module, inp, output):
            # Handle BaseModelOutput (encoder returns this with last_hidden_state as a list)
            if hasattr(output, 'last_hidden_state'):
                lhs = output.last_hidden_state
                if isinstance(lhs, (list, tuple)):
                    # Encoder: last_hidden_state is list of [s3, s4, s5] tensors
                    tensors = [t for t in lhs if isinstance(t, torch.Tensor)]
                    for i, t in enumerate(tensors):
                        arr = t.detach().float().cpu().numpy()
                        captures[f"{name}_{i}"] = arr
                        print(f"    Hook [{name}_{i}]: shape={arr.shape}, "
                              f"range=[{arr.min():.4f}, {arr.max():.4f}]")
                elif isinstance(lhs, torch.Tensor):
                    arr = lhs.detach().float().cpu().numpy()
                    captures[name] = arr
                    print(f"    Hook [{name}]: shape={arr.shape}, "
                          f"range=[{arr.min():.4f}, {arr.max():.4f}]")
            elif isinstance(output, (tuple, list)):
                # Generic tuple/list of tensors
                tensors = [t for t in output if isinstance(t, torch.Tensor)]
                for i, t in enumerate(tensors):
                    arr = t.detach().float().cpu().numpy()
                    captures[f"{name}_{i}"] = arr
                    print(f"    Hook [{name}_{i}]: shape={arr.shape}, "
                          f"range=[{arr.min():.4f}, {arr.max():.4f}]")
            elif isinstance(output, torch.Tensor):
                arr = output.detach().float().cpu().numpy()
                captures[name] = arr
                print(f"    Hook [{name}]: shape={arr.shape}, "
                      f"range=[{arr.min():.4f}, {arr.max():.4f}]")
        return hook_fn

    hooks = []

    # Hook encoder_input_proj[0..2] → ip3/ip4/ip5
    for i, proj in enumerate(inner.encoder_input_proj):
        tag = f"ip{i+3}"
        hooks.append(proj.register_forward_hook(make_hook(tag)))

    # Hook the encoder (HybridEncoder) → returns tuple of 3 feature maps (s3/s4/s5)
    hooks.append(inner.encoder.register_forward_hook(make_hook("encoder")))

    # Hook enc_output (linear + layernorm)
    hooks.append(inner.enc_output.register_forward_hook(make_hook("enc_output")))

    # Run inference
    print("\nRunning inference...")
    with torch.no_grad():
        outputs = model(**inputs)

    # Remove hooks
    for h in hooks:
        h.remove()

    print("\nPost-processing captures...")

    # Rename encoder sub-outputs to s3/s4/s5
    # encoder_0 = s3 (largest, 80x80), encoder_1 = s4, encoder_2 = s5 (smallest)
    for i in range(3):
        key = f"encoder_{i}"
        if key in captures:
            captures[f"s{i+3}"] = captures.pop(key)

    # If encoder returned a single tensor (unexpected), keep as-is
    if "encoder" in captures:
        captures["encoder_raw"] = captures.pop("encoder")

    # Print detection results summary
    if hasattr(outputs, 'logits'):
        logits = outputs.logits
        boxes = outputs.pred_boxes
        print(f"\nDetection output: logits={logits.shape}, boxes={boxes.shape}")
        scores = logits.sigmoid().max(dim=-1).values[0]
        top5_idx = scores.topk(min(5, scores.shape[0])).indices
        print("Top-5 detections:")
        for idx in top5_idx:
            cls_scores = logits[0, idx].sigmoid()
            cls_id = cls_scores.argmax().item()
            conf = cls_scores[cls_id].item()
            box = boxes[0, idx].tolist()
            print(f"  det[{idx:4d}]: cls={cls_id} conf={conf:.3f} "
                  f"box=[{box[0]:.3f},{box[1]:.3f},{box[2]:.3f},{box[3]:.3f}]")

    # Write GGUF
    print(f"\nWriting GGUF: {args.output}")
    writer = gguf.GGUFWriter(str(args.output), "layout_ref")
    writer.add_string("general.name", "docling-layout-heron-reference")
    writer.add_string("layout.ref.model", args.model)
    writer.add_string("layout.ref.image", args.image)
    writer.add_uint32("layout.ref.image_w", image.width)
    writer.add_uint32("layout.ref.image_h", image.height)

    # Store input image tensor (squeeze batch dim)
    img_arr = pixel_values[0].float().cpu().numpy()  # (3, H, W)
    writer.add_tensor("input_image", img_arr,
                      raw_dtype=gguf.GGMLQuantizationType.F32)

    # Write all captured tensors
    all_names = sorted(captures.keys())
    for name in all_names:
        arr = captures[name]
        # Squeeze batch dim if present
        if arr.ndim >= 2 and arr.shape[0] == 1:
            arr = arr[0]
        writer.add_tensor(name, arr.astype(np.float32),
                          raw_dtype=gguf.GGMLQuantizationType.F32)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = Path(args.output).stat().st_size / 1024 / 1024
    print(f"\nWrote {args.output} ({size_mb:.1f} MB)")
    total = len(captures) + 1  # +1 for input_image
    print(f"Tensors ({total} total including input_image):")
    print(f"  {'input_image':30s}: {img_arr.shape}  "
          f"range=[{img_arr.min():.4f}, {img_arr.max():.4f}]")
    for name in all_names:
        arr = captures[name]
        if arr.ndim >= 2 and arr.shape[0] == 1:
            arr = arr[0]
        print(f"  {name:30s}: {arr.shape}  "
              f"range=[{arr.min():.4f}, {arr.max():.4f}]")


if __name__ == "__main__":
    main()
