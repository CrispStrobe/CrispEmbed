#!/usr/bin/env python3
"""Dump per-layer reference activations for PP-FormulaNet-L.

Loads the HuggingFace model, runs inference on a test image, captures
intermediate activations at every architectural boundary via forward hooks,
and writes to a GGUF tensor archive.

Stages captured:
  input_image            (3, 768, 768)   preprocessed RGB
  patch_embed_output     (48, 48, 768)   after patch_embed + pos_embed
  enc_layer_{0..11}      (H, W, 768)     after each ViT layer (BHWC format)
  neck_output            (256, 48, 48)   after neck (NCHW)
  proj_output            (144, 512)      after multi-modal projector (decoder input)
  dec_layer_{0..7}       (512,)          decoder layer output at step 0
  logits_step0           (V,)            logits at first decode step
  generated_ids          (N,)            full greedy decode output

Usage:
    python tools/dump_ppformulanet_l_reference.py \
        --model-dir /mnt/volume1/models/PP-FormulaNet-L \
        --output /tmp/ppfnl-ref.gguf

    python tools/dump_ppformulanet_l_reference.py \
        --model-dir /mnt/volume1/models/PP-FormulaNet-L \
        --image /path/to/formula.png \
        --output /tmp/ppfnl-ref.gguf
"""

import argparse
from pathlib import Path

import gguf
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Image preprocessing (matches UniMERNet / NougatProcessor pipeline)
# ---------------------------------------------------------------------------

UNIMERNET_MEAN = 0.7931
UNIMERNET_STD = 0.1738


def preprocess_image(image_path, size=768):
    """Load and preprocess image for PP-FormulaNet-L."""
    from PIL import Image

    img = Image.open(image_path).convert("L")  # grayscale
    img.thumbnail((size, size), Image.BILINEAR)
    w, h = img.size

    padded = Image.new("L", (size, size), 0)
    padded.paste(img, ((size - w) // 2, (size - h) // 2))

    arr = np.array(padded, dtype=np.float32) / 255.0
    arr = np.stack([arr, arr, arr], axis=-1)  # HWC
    arr = (arr - UNIMERNET_MEAN) / UNIMERNET_STD
    arr = arr.transpose(2, 0, 1)  # CHW
    return arr


def create_test_image(size=768):
    """Create synthetic test image: gray 0.8 background with dark horizontal bar."""
    gray = np.ones((size, size), dtype=np.float32) * 0.8
    gray[size // 2 - 3:size // 2 + 3, size // 4:3 * size // 4] = 0.1
    rgb = np.stack([gray, gray, gray], axis=-1)
    arr = (rgb - UNIMERNET_MEAN) / UNIMERNET_STD
    arr = arr.transpose(2, 0, 1)
    return arr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="PP-FormulaNet-L reference dumper")
    p.add_argument("--model-dir", required=True, help="Path to HF model directory")
    p.add_argument("--image", default=None, help="Test image (default: synthetic)")
    p.add_argument("--output", required=True, help="Output GGUF path")
    args = p.parse_args()

    from transformers import PPFormulaNetForConditionalGeneration, AutoTokenizer

    print(f"Loading model: {args.model_dir}")
    model = PPFormulaNetForConditionalGeneration.from_pretrained(
        args.model_dir, torch_dtype=torch.float32
    )
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    print(f"Tokenizer: {len(tokenizer)} tokens")

    # Prepare input
    if args.image:
        img = preprocess_image(args.image)
        print(f"Image: {args.image} -> {img.shape}")
    else:
        img = create_test_image()
        print(f"Synthetic test image: {img.shape}")

    pixel_values = torch.from_numpy(img).unsqueeze(0).float()

    # Register hooks to capture intermediate activations
    captures = {}

    def make_hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                t = output[0]
            elif hasattr(output, 'last_hidden_state'):
                t = output.last_hidden_state
            elif hasattr(output, 'pooler_output'):
                t = output.pooler_output
            else:
                t = output
            if isinstance(t, torch.Tensor):
                captures[name] = t.detach().float().cpu().numpy()
        return hook_fn

    hooks = []
    enc = model.model.encoder  # PPFormulaNetVisionModel

    # Patch embedding
    hooks.append(enc.patch_embed.register_forward_hook(make_hook("patch_embed_raw")))

    # ViT layers (output is BHWC)
    for li, layer in enumerate(enc.layers):
        hooks.append(layer.register_forward_hook(make_hook(f"enc_layer_{li}")))

    # Neck (output is BCHW)
    hooks.append(enc.neck.register_forward_hook(make_hook("neck_output")))

    # Multi-modal projector (output is B, N, 512)
    hooks.append(enc.multi_modal_projector.register_forward_hook(make_hook("proj_output")))

    # Decoder layers
    dec = model.model.decoder
    for li, layer in enumerate(dec.layers):
        hooks.append(layer.register_forward_hook(make_hook(f"dec_layer_{li}")))

    # Decoder sublayer norms (step 0 detail)
    for li, layer in enumerate(dec.layers):
        hooks.append(layer.self_attn_layer_norm.register_forward_hook(
            make_hook(f"dec_self_attn_ln_{li}")))
        hooks.append(layer.encoder_attn_layer_norm.register_forward_hook(
            make_hook(f"dec_cross_attn_ln_{li}")))
        hooks.append(layer.final_layer_norm.register_forward_hook(
            make_hook(f"dec_ffn_ln_{li}")))
    hooks.append(dec.layer_norm.register_forward_hook(make_hook("dec_final_ln")))
    hooks.append(dec.layernorm_embedding.register_forward_hook(make_hook("dec_embed_ln")))

    # --- Run single forward step through model ---
    print("\nRunning model forward (encoder + 1 decoder step)...")
    bos = model.config.text_config.bos_token_id or 0
    decoder_input_ids = torch.tensor([[bos]], dtype=torch.long)

    with torch.no_grad():
        outputs = model(
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
        )
        logits = outputs.logits  # (1, 1, V)
        print(f"  Logits: {logits.shape}")
        top5 = torch.topk(logits[0, 0], 5)
        print(f"  Top-5 tokens: {top5.indices.tolist()} scores: "
              f"{[f'{s:.3f}' for s in top5.values.tolist()]}")

    # Remove hooks before generate
    for h in hooks:
        h.remove()

    # Store input and logits
    captures["input_image"] = img.astype(np.float32)
    captures["logits_step0"] = logits[0, 0].detach().float().cpu().numpy()

    # --- Full greedy decode ---
    print("\nRunning greedy decode...")
    with torch.no_grad():
        generated = model.generate(
            pixel_values,
            max_length=256,
            num_beams=1,
            do_sample=False,
        )
    decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"  Generated {len(generated[0])} tokens")
    print(f"  Output: {decoded[:200]}")
    captures["generated_ids"] = generated[0].numpy().astype(np.int32)

    # --- Write GGUF ---
    print(f"\nWriting GGUF: {args.output}")
    writer = gguf.GGUFWriter(str(args.output), "ppfnl_ref")
    writer.add_string("general.name", "ppformulanet-l-reference")
    writer.add_string("ppfnl.ref.model_dir", str(args.model_dir))
    writer.add_string("ppfnl.ref.decoded", decoded[:512])

    for name, arr in sorted(captures.items()):
        # Squeeze batch dim if present
        while arr.ndim >= 2 and arr.shape[0] == 1:
            arr = arr[0]

        if arr.dtype == np.int32:
            writer.add_tensor(name, arr)
        else:
            writer.add_tensor(name, arr.astype(np.float32),
                              raw_dtype=gguf.GGMLQuantizationType.F32)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = Path(args.output).stat().st_size / 1024 / 1024
    print(f"\nWrote {args.output} ({size_mb:.1f} MB, {len(captures)} tensors)")
    for name in sorted(captures.keys()):
        arr = captures[name]
        print(f"  {name}: {arr.shape} {arr.dtype}")


if __name__ == "__main__":
    main()
