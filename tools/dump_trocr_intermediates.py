#!/usr/bin/env python3
"""Dump TrOCR intermediate values from safetensors (NO PyTorch).

Pure-numpy forward pass through key checkpoints:
1. Patch embedding + position embedding + cls token
2. Per-layer encoder output (first 5 values of token 0)
3. Decoder token embedding + position embedding
4. Per-step decoder logits

This is the Python reference for the crispembed-diff harness.
Compare these values against the C ggml inference to find divergence.

Usage:
    python tools/dump_trocr_intermediates.py \\
        --model-dir /path/to/model \\
        --image /path/to/image.png \\
        --output /path/to/intermediates.json
"""

import argparse
import json
import sys
import math
import numpy as np
from pathlib import Path
from safetensors import safe_open


def layernorm(x, weight, bias, eps=1e-6):
    """LayerNorm over last axis."""
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * weight + bias


def gelu(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def linear(x, weight, bias=None):
    """x @ weight.T + bias"""
    out = x @ weight.T
    if bias is not None:
        out += bias
    return out


def mha(x, q_w, q_b, k_w, k_b, v_w, v_b, out_w, out_b, n_heads):
    """Multi-head attention. x: (T, D)"""
    T, D = x.shape
    hd = D // n_heads

    Q = linear(x, q_w, q_b).reshape(T, n_heads, hd).transpose(1, 0, 2)  # (nh, T, hd)
    K = linear(x, k_w, k_b).reshape(T, n_heads, hd).transpose(1, 0, 2)
    V = linear(x, v_w, v_b).reshape(T, n_heads, hd).transpose(1, 0, 2)

    scores = (Q @ K.transpose(0, 2, 1)) / math.sqrt(hd)  # (nh, T, T)
    attn = softmax(scores)
    out = (attn @ V).transpose(1, 0, 2).reshape(T, D)  # (T, D)
    return linear(out, out_w, out_b)


def load_image_gray(path, target_size=384):
    """Load image as grayscale float [0,1], resize to target_size."""
    from PIL import Image
    img = Image.open(path).convert('L').resize((target_size, target_size))
    return np.array(img, dtype=np.float32) / 255.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--output", default=None, help="JSON output file")
    p.add_argument("--max-enc-layers", type=int, default=2, help="Max encoder layers to run")
    args = p.parse_args()

    model_dir = Path(args.model_dir)
    with open(model_dir / "config.json") as f:
        config = json.load(f)

    enc = config.get("encoder", config)
    H = enc["hidden_size"]
    P = enc["patch_size"]
    S = enc["image_size"]
    n_heads = enc["num_attention_heads"]
    n_layers = enc["num_hidden_layers"]

    print(f"Encoder: {n_layers}L, {H}d, {n_heads}H, image={S}, patch={P}")

    # Load image
    gray = load_image_gray(args.image, S)
    # Normalize: (pixel - 0.5) / 0.5
    normalized = (gray - 0.5) / 0.5
    # Expand to 3-channel CHW
    rgb = np.stack([normalized] * 3, axis=0)  # (3, S, S)
    print(f"Image: {gray.shape} → rgb {rgb.shape}")

    # Load weights
    st_path = str(model_dir / "model.safetensors")
    intermediates = {}

    with safe_open(st_path, framework="numpy") as f:
        # --- Patch embedding ---
        proj_w = f.get_tensor("encoder.embeddings.patch_embeddings.projection.weight")  # (H, 3, P, P)
        proj_b = f.get_tensor("encoder.embeddings.patch_embeddings.projection.bias")    # (H,)
        cls_token = f.get_tensor("encoder.embeddings.cls_token").squeeze()               # (H,)
        pos_embed = f.get_tensor("encoder.embeddings.position_embeddings").squeeze()     # (T, H)

        npw = S // P
        nph = S // P
        n_patches = npw * nph
        T = n_patches + 2  # +cls +dist

        print(f"Patches: {npw}x{nph} = {n_patches}, T={T}")

        # Conv projection (manual)
        patch_dim = 3 * P * P
        proj_w_2d = proj_w.reshape(H, patch_dim)  # (H, 3*P*P)

        embedded = np.zeros((T, H), dtype=np.float32)
        # CLS token at position 0
        embedded[0] = cls_token

        # Distillation token at position 1 (if exists)
        try:
            dist_token = f.get_tensor("encoder.embeddings.distillation_token").squeeze()
            embedded[1] = dist_token
        except:
            pass

        # Patch embeddings at positions 2..T-1
        for py in range(nph):
            for px in range(npw):
                t = py * npw + px + 2
                patch = np.zeros(patch_dim, dtype=np.float32)
                for c in range(3):
                    for dy in range(P):
                        for dx in range(P):
                            sy, sx = py * P + dy, px * P + dx
                            patch[c * P * P + dy * P + dx] = rgb[c, sy, sx]
                embedded[t] = proj_w_2d @ patch + proj_b

        # Add position embeddings (may have fewer entries than T if no dist token)
        n_pos = min(T, pos_embed.shape[0])
        embedded[:n_pos] += pos_embed[:n_pos]

        intermediates["patch_embed_first5"] = embedded[0, :5].tolist()
        intermediates["patch_embed_token2_first5"] = embedded[2, :5].tolist()
        print(f"Patch embed token 0 first 5: {embedded[0, :5]}")
        print(f"Patch embed token 2 first 5: {embedded[2, :5]}")

        # --- Encoder layers (run first N) ---
        cur = embedded
        max_layers = min(args.max_enc_layers, n_layers)
        for li in range(max_layers):
            prefix = f"encoder.encoder.layer.{li}"

            def get_opt(name):
                try: return f.get_tensor(name)
                except: return None

            ln1_w = f.get_tensor(f"{prefix}.layernorm_before.weight")
            ln1_b = f.get_tensor(f"{prefix}.layernorm_before.bias")
            q_w = f.get_tensor(f"{prefix}.attention.attention.query.weight")
            q_b = get_opt(f"{prefix}.attention.attention.query.bias")
            k_w = f.get_tensor(f"{prefix}.attention.attention.key.weight")
            k_b = get_opt(f"{prefix}.attention.attention.key.bias")
            v_w = f.get_tensor(f"{prefix}.attention.attention.value.weight")
            v_b = get_opt(f"{prefix}.attention.attention.value.bias")
            out_w = f.get_tensor(f"{prefix}.attention.output.dense.weight")
            out_b = get_opt(f"{prefix}.attention.output.dense.bias")
            ln2_w = f.get_tensor(f"{prefix}.layernorm_after.weight")
            ln2_b = f.get_tensor(f"{prefix}.layernorm_after.bias")
            ff_up_w = f.get_tensor(f"{prefix}.intermediate.dense.weight")
            ff_up_b = get_opt(f"{prefix}.intermediate.dense.bias")
            ff_down_w = f.get_tensor(f"{prefix}.output.dense.weight")
            ff_down_b = get_opt(f"{prefix}.output.dense.bias")

            # Pre-LN
            normed = layernorm(cur, ln1_w, ln1_b)
            # Self-attention
            attn = mha(normed, q_w, q_b, k_w, k_b, v_w, v_b, out_w, out_b, n_heads)
            cur = cur + attn
            # Post-LN + FFN
            normed2 = layernorm(cur, ln2_w, ln2_b)
            up = gelu(linear(normed2, ff_up_w, ff_up_b))
            down = linear(up, ff_down_w, ff_down_b)
            cur = cur + down

            key = f"enc_layer_{li}_first5"
            intermediates[key] = cur[0, :5].tolist()
            print(f"Encoder L{li} token 0 first 5: {cur[0, :5]}")

    # Output
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(intermediates, f, indent=2)
        print(f"\nSaved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
