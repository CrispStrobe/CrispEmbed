#!/usr/bin/env python3
"""Dump per-layer reference activations for surya-ocr-2 text detector.

Loads the EfficientViT segformer model from surya's S3 checkpoint, runs
inference on a test image, captures intermediate activations at every
architectural boundary via forward hooks, and writes to a GGUF tensor
archive for C++ parity comparison.

The C++ diff harness then loads that GGUF and compares each tensor
against what the ggml engine produces. The first layer where cos drops
below 0.999 is where the bug lives.

Stages captured:
  input_image          (3, H, W)        F32 preprocessed RGB input
  stem_output          (32, H/2, W/2)   F32 after Stem
  stage{0-3}_output    (C, H', W')      F32 after each EfficientViT stage
  decode_fused         (512, H/2, W/2)  F32 after decode_head linear_fuse+BN+ReLU
  logits               (2, H/2, W/2)    F32 raw logits (before sigmoid)
  heatmap              (2, H/2, W/2)    F32 sigmoid output

Usage:
    python tools/dump_surya_det_reference.py \\
        --output /tmp/surya-det-ref.gguf

    python tools/dump_surya_det_reference.py \\
        --image /path/to/page.png \\
        --output /tmp/surya-det-ref.gguf

    python tools/dump_surya_det_reference.py \\
        --checkpoint /path/to/local/checkpoint \\
        --output /tmp/surya-det-ref.gguf
"""

import argparse
import sys
from pathlib import Path
from typing import Dict

import gguf
import numpy as np
import torch
import torch.nn.functional as F


def make_synthetic_image(h: int = 1024, w: int = 1024) -> np.ndarray:
    """Create a synthetic test image with text-like horizontal stripes."""
    img = np.ones((h, w, 3), dtype=np.uint8) * 240  # near-white background
    # Add some dark horizontal bands to simulate text lines
    for y_start in range(100, h - 100, 80):
        y_end = min(y_start + 12, h)
        x_start = 50
        x_end = w - 50
        img[y_start:y_end, x_start:x_end, :] = 30  # dark text
    return img


def load_model_and_processor(checkpoint=None):
    """Load surya detection model and processor."""
    from surya.detection.loader import DetectionModelLoader
    loader = DetectionModelLoader(checkpoint=checkpoint)
    model = loader.model(device='cpu', dtype=torch.float32)
    processor = loader.processor()
    return model, processor


def register_hooks(model) -> Dict[str, torch.Tensor]:
    """Register forward hooks to capture intermediate activations."""
    activations = {}

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, (list, tuple)):
                # EfficientVitLarge returns list of hidden states
                for i, o in enumerate(output):
                    activations[f"{name}_{i}"] = o.detach().cpu().float()
            elif isinstance(output, torch.Tensor):
                activations[name] = output.detach().cpu().float()
            else:
                # SemanticSegmenterOutput etc.
                if hasattr(output, 'logits'):
                    activations[name] = output.logits.detach().cpu().float()
                if hasattr(output, 'hidden_states') and output.hidden_states:
                    for i, hs in enumerate(output.hidden_states):
                        activations[f"{name}_hidden_{i}"] = hs.detach().cpu().float()
        return hook

    # Stem
    model.vit.stem.register_forward_hook(make_hook("stem"))

    # Each stage
    for i, stage in enumerate(model.vit.stages):
        stage.register_forward_hook(make_hook(f"stage_{i}"))

        # Hook into individual blocks within vit stages
        if hasattr(stage, 'blocks'):
            for j, block in enumerate(stage.blocks):
                block.register_forward_hook(make_hook(f"stage_{i}_block_{j}"))

                # For EfficientVitBlock (stage 3), hook context and local modules
                if hasattr(block, 'context_module'):
                    block.context_module.register_forward_hook(
                        make_hook(f"stage_{i}_block_{j}_context"))
                    # Hook the LiteMLA attention
                    if hasattr(block.context_module, 'main') and hasattr(block.context_module.main, 'qkv'):
                        block.context_module.main.qkv.register_forward_hook(
                            make_hook(f"stage_{i}_block_{j}_qkv"))
                if hasattr(block, 'local_module'):
                    block.local_module.register_forward_hook(
                        make_hook(f"stage_{i}_block_{j}_local"))

    # Decode head sub-components
    for i, mlp in enumerate(model.decode_head.linear_c):
        mlp.register_forward_hook(make_hook(f"decode_mlp_{i}"))

    model.decode_head.linear_fuse.register_forward_hook(
        make_hook("decode_linear_fuse"))
    model.decode_head.batch_norm.register_forward_hook(
        make_hook("decode_batch_norm"))
    model.decode_head.classifier.register_forward_hook(
        make_hook("decode_classifier"))

    # Full decode head
    model.decode_head.register_forward_hook(make_hook("decode_head"))

    # Hook the linear_fuse input to capture the concatenated tensor
    original_fuse_forward = model.decode_head.linear_fuse.forward
    def fuse_hook_forward(x):
        activations["decode_cat_input"] = x.detach().cpu().float()
        return original_fuse_forward(x)
    model.decode_head.linear_fuse.forward = fuse_hook_forward

    return activations


def preprocess_image(img_np: np.ndarray, processor) -> torch.Tensor:
    """Preprocess image using surya's processor pipeline."""
    from PIL import Image
    pil_img = Image.fromarray(img_np).convert("RGB")

    # Match surya's detection pipeline preprocessing
    new_size = (processor.size["width"], processor.size["height"])
    pil_img.thumbnail(new_size, Image.Resampling.LANCZOS)
    pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)

    img_arr = np.asarray(pil_img, dtype=np.uint8)
    processed = processor(img_arr)["pixel_values"][0]
    tensor = torch.from_numpy(processed).unsqueeze(0).float()
    return tensor


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=None,
        help="Local checkpoint path or S3 path (default: surya's built-in)")
    parser.add_argument("--image", default=None,
        help="Test image path (default: synthetic)")
    parser.add_argument("--output", required=True,
        help="Output GGUF path for reference tensors")
    args = parser.parse_args()

    print("Loading model...")
    model, processor = load_model_and_processor(args.checkpoint)
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"Processor size: {processor.size}")

    # Load or create test image
    if args.image:
        from PIL import Image
        img = np.asarray(Image.open(args.image).convert("RGB"))
        print(f"Loaded image: {img.shape}")
    else:
        img = make_synthetic_image()
        print(f"Synthetic image: {img.shape}")

    # Preprocess
    pixel_values = preprocess_image(img, processor)
    print(f"Preprocessed tensor: {pixel_values.shape} (dtype={pixel_values.dtype})")

    # Register hooks and run inference
    activations = register_hooks(model)
    with torch.inference_mode():
        output = model(pixel_values)

    # Get final output
    logits = output.logits
    heatmap = torch.sigmoid(logits)

    # Store input and final outputs
    activations["input_image"] = pixel_values[0].cpu().float()
    activations["logits"] = logits[0].cpu().float()
    activations["heatmap"] = heatmap[0].cpu().float()

    # Write to GGUF
    print(f"\nWriting {len(activations)} tensors to {args.output}")
    writer = gguf.GGUFWriter(args.output, "surya-det-reference")
    writer.add_description("surya-ocr-2 detector per-layer reference activations")

    # Sort for deterministic output
    for name in sorted(activations.keys()):
        tensor = activations[name]
        if tensor.dim() == 4:
            tensor = tensor[0]  # Remove batch dim
        data = tensor.numpy().astype(np.float32)
        print(f"  {name}: {list(data.shape)} "
              f"(min={data.min():.4f} max={data.max():.4f} mean={data.mean():.4f})")
        writer.add_tensor(name, data)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    out_size = Path(args.output).stat().st_size / 1024 / 1024
    print(f"\nDone: {args.output} ({out_size:.1f} MB)")
    print(f"Activations captured: {len(activations)}")

    # Print summary of key stages
    print("\n=== Key activation shapes ===")
    for key in ["input_image", "stem", "stage_0", "stage_1", "stage_2",
                "stage_3", "decode_head", "logits", "heatmap"]:
        if key in activations:
            t = activations[key]
            if t.dim() == 4:
                t = t[0]
            print(f"  {key}: {list(t.shape)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
