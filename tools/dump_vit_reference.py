#!/usr/bin/env python3
"""Dump per-layer ViT (SigLIP/CLIP) reference activations to GGUF.

Produces a reference GGUF for crispembed_diff.h comparison.

Usage:
    python tools/dump_vit_reference.py \
        --model openai/clip-vit-base-patch16 \
        --image test.jpg \
        --output /tmp/clip-vit-ref.gguf
"""

import argparse
import sys
from pathlib import Path

import gguf
import numpy as np
import torch
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    from transformers import AutoModel, AutoProcessor, AutoConfig

    print(f"Loading: {args.model}")
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.float32,
                                       trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model)
    model.eval()

    is_clip = "clip" in config.model_type.lower()

    img = Image.open(args.image).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")

    writer = gguf.GGUFWriter(str(args.output), "vit_ref")

    # Hook into the vision model to capture intermediates
    vision = model.vision_model
    captures = {}

    def make_hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            captures[name] = out.detach().float().cpu().numpy().flatten()
        return hook_fn

    # Register hooks
    hooks = []

    # Patch embedding
    hooks.append(vision.embeddings.register_forward_hook(make_hook("embeddings")))

    # Each encoder layer
    for i, layer in enumerate(vision.encoder.layers):
        hooks.append(layer.register_forward_hook(make_hook(f"enc_layer_{i}")))

    # Post-layernorm
    if hasattr(vision, 'post_layernorm') and vision.post_layernorm is not None:
        hooks.append(vision.post_layernorm.register_forward_hook(make_hook("post_ln")))

    # Run forward pass
    with torch.no_grad():
        if is_clip:
            out = model.get_image_features(**inputs)
            out = out / out.norm(dim=-1, keepdim=True)
        else:
            out = model(**inputs)
            if hasattr(out, 'image_embeds'):
                final = out.image_embeds
            elif hasattr(out, 'pooler_output'):
                final = out.pooler_output
            else:
                final = out.last_hidden_state.mean(dim=1)
            out = final / final.norm(dim=-1, keepdim=True)

    captures["final_embedding"] = out.detach().float().cpu().numpy().flatten()

    # Write all captures
    for name, data in sorted(captures.items()):
        writer.add_tensor(name, data.astype(np.float32))
        print(f"  {name}: {len(data)} floats")

    # Remove hooks
    for h in hooks:
        h.remove()

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"\nWrote {args.output} ({Path(args.output).stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
