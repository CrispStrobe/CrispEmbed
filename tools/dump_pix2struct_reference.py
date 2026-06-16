#!/usr/bin/env python3
"""Dump Pix2Struct per-stage reference activations for parity testing.

Usage:
    python tools/dump_pix2struct_reference.py \
        --model google/pix2struct-base \
        --output /tmp/pix2struct-ref.gguf \
        [--image test.png] [--max-patches 128]
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
    p.add_argument("--image", help="Test image (or synthetic if omitted)")
    p.add_argument("--max-patches", type=int, default=128)
    args = p.parse_args()

    from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
    from PIL import Image

    processor = Pix2StructProcessor.from_pretrained(args.model)
    model = Pix2StructForConditionalGeneration.from_pretrained(args.model)
    model.eval()

    # Create or load test image
    if args.image:
        image = Image.open(args.image).convert("RGB")
    else:
        # Synthetic: 200x100 white image with text-like dark rectangles
        np.random.seed(42)
        img = np.ones((100, 200, 3), dtype=np.uint8) * 240
        for i in range(5):
            y = 10 + i * 18
            img[y:y+12, 20:180] = 40
        image = Image.fromarray(img)

    # Preprocess
    inputs = processor(images=image, return_tensors="pt",
                       max_patches=args.max_patches)
    flattened_patches = inputs["flattened_patches"]  # [1, n_patches, patch_dim]
    attention_mask = inputs["attention_mask"]          # [1, n_patches]

    print(f"Patches: {flattened_patches.shape}, mask sum: {attention_mask.sum().item()}")

    stages = {}
    # Save preprocessed input
    stages["flattened_patches"] = flattened_patches.squeeze(0).numpy()
    stages["attention_mask"] = attention_mask.squeeze(0).float().numpy()

    # Run encoder
    with torch.no_grad():
        enc_out = model.encoder(
            flattened_patches=flattened_patches,
            attention_mask=attention_mask,
        )
        encoder_hidden = enc_out.last_hidden_state
        stages["encoder_output"] = encoder_hidden.squeeze(0).numpy()
        print(f"Encoder output: {list(encoder_hidden.shape[1:])}, "
              f"range=[{encoder_hidden.min():.4f}, {encoder_hidden.max():.4f}]")

        # Run greedy decode (first 5 tokens)
        generated = model.generate(
            flattened_patches=flattened_patches,
            attention_mask=attention_mask,
            max_new_tokens=32,
        )
        decoded = processor.batch_decode(generated, skip_special_tokens=True)
        print(f"Generated: {decoded}")
        stages["generated_ids"] = generated.squeeze(0).numpy().astype(np.float32)

    # Write GGUF
    writer = gguf.GGUFWriter(args.output, "pix2struct-reference")
    writer.add_uint32("pix2struct.ref.n_patches", flattened_patches.shape[1])
    writer.add_uint32("pix2struct.ref.max_patches", args.max_patches)
    for name, arr in stages.items():
        writer.add_tensor(name, arr.astype(np.float32),
                          raw_dtype=gguf.GGMLQuantizationType.F32)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"\nReference: {args.output} ({Path(args.output).stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
