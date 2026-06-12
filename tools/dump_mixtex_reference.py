#!/usr/bin/env python3
"""Dump per-layer reference activations for MixTex ZhEn-Latex-OCR.

Loads the HF VisionEncoderDecoderModel (Swin-Tiny + 4-layer RoBERTa),
runs inference on a test image, captures intermediate activations via
forward hooks, and writes to a GGUF tensor archive.

Architecture:
  Encoder: Swin-Tiny (patch4, window7, 224→400×500)
    - depths=[2,2,6,2], num_heads=[3,6,12,24], embed_dim=96
    - Output: [B, H/32*W/32, 768] = [B, 180, 768] for 400×500
  Decoder: RoBERTa 4-layer with cross-attention
    - hidden=768, heads=12, ffn=3072, vocab=25681
    - max_position=300, BPE tokenizer

Stages captured:
  input_image          (3, 400, 500)    F32 preprocessed
  enc_stage_{0-3}      (C, H, W)        F32 after each Swin stage
  enc_output           (N, 768)          F32 encoder sequence output
  dec_embed            (768,)            F32 first decoder token embed
  dec_layer_{0-3}      (768,)            F32 after each decoder layer (step 0)
  logits_step0         (V,)              F32 logits at first decode step
  generated_text       string            decoded LaTeX output

Usage:
    python tools/dump_mixtex_reference.py \\
        --output /tmp/mixtex-ref.gguf

    python tools/dump_mixtex_reference.py \\
        --image /path/to/formula.png \\
        --output /tmp/mixtex-ref.gguf
"""

import argparse
import sys
from pathlib import Path
from typing import Dict

import gguf
import numpy as np
import torch


def make_synthetic_formula(h: int = 400, w: int = 500) -> np.ndarray:
    """Create a synthetic formula image."""
    img = np.ones((h, w, 3), dtype=np.uint8) * 255
    # Draw a simple "x^2" shape
    # Horizontal line
    img[h//2-2:h//2+2, w//4:3*w//4, :] = 0
    # Vertical bar
    img[h//3:2*h//3, w//2-2:w//2+2, :] = 0
    # Small superscript
    img[h//3-5:h//3+5, w//2+20:w//2+30, :] = 0
    return img


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="MixTex/ZhEn-Latex-OCR",
        help="HF model ID or local path")
    parser.add_argument("--image", default=None, help="Test image (default: synthetic)")
    parser.add_argument("--output", required=True, help="Output GGUF path")
    args = parser.parse_args()

    from transformers import VisionEncoderDecoderModel, ViTImageProcessor, RobertaTokenizer
    from PIL import Image

    print(f"Loading model: {args.model}")
    model = VisionEncoderDecoderModel.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded: {total_params:,} params")

    processor = ViTImageProcessor.from_pretrained(args.model)
    tokenizer = RobertaTokenizer.from_pretrained(args.model)
    print(f"Processor size: {processor.size}")
    print(f"Vocab size: {tokenizer.vocab_size}")

    # Load or create test image
    if args.image:
        img = np.asarray(Image.open(args.image).convert("RGB"))
        print(f"Loaded image: {img.shape}")
    else:
        img = make_synthetic_formula()
        print(f"Synthetic image: {img.shape}")

    # Preprocess
    pil_img = Image.fromarray(img)
    pixel_values = processor(pil_img, return_tensors="pt").pixel_values.float()
    print(f"Preprocessed: {pixel_values.shape}")

    # Register hooks
    activations = {}

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            if isinstance(out, torch.Tensor):
                activations[name] = out.detach().cpu().float()
        return hook

    # Encoder hooks — SwinModel: encoder.embeddings, encoder.encoder.layers, encoder.layernorm
    enc = model.encoder
    if hasattr(enc, 'embeddings'):
        enc.embeddings.register_forward_hook(make_hook("enc_embed"))
    if hasattr(enc, 'encoder') and hasattr(enc.encoder, 'layers'):
        for i, layer in enumerate(enc.encoder.layers):
            layer.register_forward_hook(make_hook(f"enc_stage_{i}"))
    if hasattr(enc, 'layernorm'):
        enc.layernorm.register_forward_hook(make_hook("enc_layernorm"))
    enc.register_forward_hook(make_hook("enc_output"))

    # Decoder hooks — RobertaForCausalLM: decoder.roberta.encoder.layer
    dec = model.decoder
    if hasattr(dec, 'roberta') and hasattr(dec.roberta, 'encoder'):
        for i, layer in enumerate(dec.roberta.encoder.layer):
            layer.register_forward_hook(make_hook(f"dec_layer_{i}"))
    elif hasattr(dec, 'encoder') and hasattr(dec.encoder, 'layer'):
        for i, layer in enumerate(dec.encoder.layer):
            layer.register_forward_hook(make_hook(f"dec_layer_{i}"))

    # Run encoder
    with torch.inference_mode():
        encoder_outputs = model.encoder(pixel_values)
        enc_hidden = encoder_outputs.last_hidden_state
        print(f"Encoder output: {enc_hidden.shape}")

    # Run greedy decode
    with torch.inference_mode():
        generated = model.generate(
            pixel_values,
            max_length=296,
            num_beams=1,
        )
        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"Generated: {text}")

    # Store
    activations["input_image"] = pixel_values[0].cpu().float()
    activations["generated_ids"] = generated[0].cpu().float()

    # Write GGUF
    print(f"\nWriting {len(activations)} tensors to {args.output}")
    writer = gguf.GGUFWriter(args.output, "mixtex-reference")
    writer.add_description(f"MixTex ZhEn-Latex-OCR reference activations")
    writer.add_string("generated_text", text)

    for name in sorted(activations.keys()):
        tensor = activations[name]
        if tensor.dim() >= 4:
            tensor = tensor[0]
        elif tensor.dim() == 3:
            tensor = tensor[0]  # Remove batch
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
