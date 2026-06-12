#!/usr/bin/env python3
"""Dump per-layer InternVL2/2.5 reference activations (safetensors, no full model load).

Pure-numpy forward pass through vision encoder + projector, and layer-by-layer
LLM decoder, loading weights one at a time via safetensors.

Stages captured (written to reference GGUF):

  Vision encoder:
    vis_patch_embed          (N+1, D_v)   patch embed + CLS + position embed
    vis_layer_{i}            (N+1, D_v)   after each ViT block
    vis_pixel_unshuffle      (M, merge_dim) after pixel unshuffle
    vis_proj_output          (M, D_llm)   after MLP projector

  LLM decoder (first N layers):
    llm_embed                (T, D)       token embedding
    llm_layer_{i}            (T, D)       after each decoder layer

Usage:
    PYTHONNOUSERSITE=1 python tools/dump_internvl2_reference.py \\
        --model OpenGVLab/InternVL2_5-2B \\
        --image test_images/invoice_de.png \\
        --output /mnt/volume1/tmp-overflow/internvl2-ref.gguf \\
        --max-vis-layers 4 \\
        --max-llm-layers 2

Requires: safetensors, gguf, numpy, Pillow, huggingface_hub
Does NOT require torch or transformers at runtime (only for download).
"""

import argparse
import json
import math
import sys
from pathlib import Path

import gguf
import numpy as np
from PIL import Image
from safetensors import safe_open


# ── Numpy ops ─────────────────────────────────────────────────────────

def layernorm(x, weight, bias, eps=1e-6):
    """Standard LayerNorm."""
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * weight + bias


def rms_norm(x, weight, eps=1e-5):
    """RMSNorm: x * weight / sqrt(mean(x²) + eps)."""
    ms = (x ** 2).mean(axis=-1, keepdims=True)
    return x / np.sqrt(ms + eps) * weight


def gelu(x):
    """GELU with tanh approximation (matches PyTorch default)."""
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


def silu(x):
    return x / (1.0 + np.exp(-x))


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def linear(x, weight, bias=None):
    """x @ weight.T + bias."""
    out = x @ weight.T
    if bias is not None:
        out += bias
    return out


# ── Multi-head attention (InternViT: no RoPE, learnable pos embed) ───

def mha_fused_qkv(x, qkv_w, qkv_b, proj_w, proj_b, n_heads, mask=None):
    """Multi-head attention with fused QKV projection.

    x: (T, D), qkv_w: (3*D, D), qkv_b: (3*D,)
    """
    T, D = x.shape
    hd = D // n_heads

    qkv = linear(x, qkv_w, qkv_b)  # (T, 3*D)
    Q, K, V = np.split(qkv, 3, axis=-1)  # each (T, D)

    Q = Q.reshape(T, n_heads, hd).transpose(1, 0, 2)  # (nh, T, hd)
    K = K.reshape(T, n_heads, hd).transpose(1, 0, 2)
    V = V.reshape(T, n_heads, hd).transpose(1, 0, 2)

    scores = (Q @ K.transpose(0, 2, 1)) / math.sqrt(hd)  # (nh, T, T)
    if mask is not None:
        scores = scores + mask[np.newaxis, :, :]

    attn = softmax(scores)
    out = (attn @ V).transpose(1, 0, 2).reshape(T, D)  # (T, D)
    return linear(out, proj_w, proj_b)


# ── GQA attention (InternLM2.5: with RoPE) ──────────────────────────

def gqa_attention(x, q_w, k_w, v_w, o_w, n_heads, n_kv_heads,
                  cos, sin, mask=None):
    """Grouped-query attention with RoPE.

    x: (T, D), q_w: (n_heads*hd, D), k_w: (n_kv_heads*hd, D), etc.
    cos/sin: (T, hd) for RoPE.
    """
    T, D = x.shape
    hd = D // n_heads
    kv_repeat = n_heads // n_kv_heads

    Q = linear(x, q_w)  # (T, n_heads*hd)
    K = linear(x, k_w)  # (T, n_kv_heads*hd)
    V = linear(x, v_w)  # (T, n_kv_heads*hd)

    Q = Q.reshape(T, n_heads, hd).transpose(1, 0, 2)     # (nh, T, hd)
    K = K.reshape(T, n_kv_heads, hd).transpose(1, 0, 2)   # (nkv, T, hd)
    V = V.reshape(T, n_kv_heads, hd).transpose(1, 0, 2)

    # Apply RoPE (rotate_half)
    cos_b = cos[np.newaxis, :, :]  # (1, T, hd)
    sin_b = sin[np.newaxis, :, :]
    Q = apply_rotary(Q, cos_b, sin_b)
    K = apply_rotary(K, cos_b, sin_b)

    # Expand K/V for GQA: (nkv, T, hd) → (nh, T, hd)
    K = np.repeat(K, kv_repeat, axis=0)
    V = np.repeat(V, kv_repeat, axis=0)

    scores = (Q @ K.transpose(0, 2, 1)) / math.sqrt(hd)  # (nh, T, T)
    if mask is not None:
        scores = scores + mask[np.newaxis, :, :]

    attn = softmax(scores)
    out = (attn @ V).transpose(1, 0, 2).reshape(T, D)
    return linear(out, o_w)


def apply_rotary(x, cos, sin):
    """Apply rotate_half RoPE. x: (nh, T, hd), cos/sin: (1, T, hd)."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([
        x1 * cos[..., :half] - x2 * sin[..., :half],
        x2 * cos[..., half:] + x1 * sin[..., half:],
    ], axis=-1)


def compute_rope_cos_sin(seq_len, head_dim, theta=1000000.0):
    """Compute standard 1D RoPE cos/sin."""
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (np.arange(0, half, dtype=np.float32) * 2.0 / head_dim))
    positions = np.arange(seq_len, dtype=np.float32)
    freqs = np.outer(positions, inv_freq)  # (T, half)
    cos_buf = np.cos(freqs)
    sin_buf = np.sin(freqs)
    # Full head_dim: [cos, cos] and [sin, sin]
    cos_full = np.concatenate([cos_buf, cos_buf], axis=-1)  # (T, hd)
    sin_full = np.concatenate([sin_buf, sin_buf], axis=-1)
    return cos_full.astype(np.float32), sin_full.astype(np.float32)


# ── Pixel unshuffle (v2) ─────────────────────────────────────────────

def pixel_unshuffle_v2(x, n_patches_per_side, downsample_ratio=0.5):
    """Pixel unshuffle from InternVL2 (ps_version='v2').

    Input:  (N, D) where N = n_patches_per_side^2, D = hidden_size
    Output: (M, D') where M = N * ratio^2, D' = D / ratio^2

    Steps (downsample_ratio=0.5):
      1. Reshape to (H, W, C) = (32, 32, 1024)
      2. View as (H, W/2, C*2) = (32, 16, 2048)
      3. Permute(1,0,2) → (16, 32, 2048) i.e. (W', H, C')
      4. View as (W', H/2, C'*2) = (16, 16, 4096)
      5. Permute(1,0,2) → (16, 16, 4096) i.e. (H', W', C'')
      6. Flatten to (256, 4096)
    """
    H = W = n_patches_per_side
    D = x.shape[-1]
    scale = int(1.0 / downsample_ratio)  # 2

    x = x.reshape(H, W, D)
    # Step 1: merge along W
    x = x.reshape(H, W // scale, D * scale)
    # Step 2: swap H and W' axes
    x = x.transpose(1, 0, 2)  # (W/s, H, D*s)
    # Step 3: merge along H (now dim 1)
    x = x.reshape(W // scale, H // scale, D * scale * scale)
    # Step 4: swap back
    x = x.transpose(1, 0, 2)  # (H/s, W/s, D*s*s)
    # Flatten
    return x.reshape(-1, D * scale * scale)


# ── Image preprocessing (InternVL2-style: simple resize to 448) ──────

def preprocess_image_internvl(img, image_size=448,
                               image_mean=(0.485, 0.456, 0.406),
                               image_std=(0.229, 0.224, 0.225)):
    """Preprocess a single tile: resize to image_size×image_size, normalize.

    Returns: (3, image_size, image_size) float32 array.
    """
    img_resized = img.resize((image_size, image_size), Image.BICUBIC)
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]  # drop alpha
    arr = arr.transpose(2, 0, 1)  # (3, H, W)
    for c in range(3):
        arr[c] = (arr[c] - image_mean[c]) / image_std[c]
    return arr


# ── GGUF reference writer ────────────────────────────────────────────

class RefWriter:
    """Write reference activations to GGUF."""

    def __init__(self, path):
        self.writer = gguf.GGUFWriter(str(path), "internvl2_ref")
        self.count = 0

    def add(self, name, data):
        """Add a reference tensor."""
        arr = np.ascontiguousarray(data, dtype=np.float32)
        self.writer.add_tensor(name, arr,
                               raw_dtype=gguf.GGMLQuantizationType.F32)
        self.count += 1
        print(f"  [{self.count:3d}] {name}: {list(arr.shape)}")

    def close(self):
        self.writer.write_header_to_file()
        self.writer.write_kv_data_to_file()
        self.writer.write_tensors_to_file()
        self.writer.close()
        print(f"  Total: {self.count} tensors")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model ID or local path")
    parser.add_argument("--image", default=None, help="Input image path (omit for synthetic gradient)")
    parser.add_argument("--output", required=True, help="Output reference GGUF path")
    parser.add_argument("--max-vis-layers", type=int, default=None,
                        help="Max vision layers to dump (default: all)")
    parser.add_argument("--max-llm-layers", type=int, default=None,
                        help="Max LLM layers to dump (default: all)")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip LLM decoder (vision only)")
    args = parser.parse_args()

    model_path = Path(args.model)
    is_local = model_path.is_dir()

    # Load config
    if is_local:
        config_path = model_path / "config.json"
    else:
        from huggingface_hub import hf_hub_download
        config_path = Path(hf_hub_download(args.model, "config.json"))
    with open(config_path) as f:
        config = json.load(f)

    def resolve_file(filename):
        if is_local:
            p = model_path / filename
            if p.exists():
                return str(p)
            raise FileNotFoundError(f"{p} not found")
        from huggingface_hub import hf_hub_download
        return hf_hub_download(args.model, filename)

    # Build tensor → shard mapping
    try:
        idx_path = resolve_file("model.safetensors.index.json")
        with open(idx_path) as f:
            idx = json.load(f)
        shard_files = sorted(set(idx["weight_map"].values()))
        tensor_to_shard = idx["weight_map"]
    except Exception:
        shard_files = ["model.safetensors"]
        tensor_to_shard = {}

    shard_paths = {}
    all_names = set()
    for shard in shard_files:
        path = resolve_file(shard)
        shard_paths[shard] = path
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                all_names.add(key)
                if key not in tensor_to_shard:
                    tensor_to_shard[key] = shard

    print(f"Model: {args.model} ({len(all_names)} tensors)")

    def get_tensor(name):
        if name not in tensor_to_shard:
            return None
        with safe_open(shard_paths[tensor_to_shard[name]],
                       framework="pt") as f:
            return f.get_tensor(name).float().numpy()

    def require(name):
        t = get_tensor(name)
        if t is None:
            raise ValueError(f"Required tensor not found: {name}")
        return t

    # Parse config
    vc = config.get("vision_config", {})
    lc = config.get("llm_config", {})
    vis_hidden = vc.get("hidden_size", 1024)
    vis_heads = vc.get("num_attention_heads", 16)
    vis_layers = vc.get("num_hidden_layers", 24)
    vis_inter = vc.get("intermediate_size", 4096)
    vis_patch = vc.get("patch_size", 14)
    vis_image_size = config.get("force_image_size", vc.get("image_size", 448))
    vis_ln_eps = vc.get("layer_norm_eps", 1e-6)
    downsample_ratio = config.get("downsample_ratio", 0.5)
    n_patches_per_side = vis_image_size // vis_patch

    llm_hidden = lc.get("hidden_size", 2048)
    llm_heads = lc.get("num_attention_heads", 16)
    llm_kv_heads = lc.get("num_key_value_heads", 8)
    llm_layers = lc.get("num_hidden_layers", 24)
    llm_inter = lc.get("intermediate_size", 8192)
    llm_rms_eps = lc.get("rms_norm_eps", 1e-5)
    llm_rope_theta = lc.get("rope_theta", 1000000.0)
    llm_head_dim = llm_hidden // llm_heads

    n_vis_dump = vis_layers if args.max_vis_layers is None else min(vis_layers, args.max_vis_layers)
    n_llm_dump = llm_layers if args.max_llm_layers is None else min(llm_layers, args.max_llm_layers)

    print(f"  Vision: {vis_layers}L, {vis_hidden}d, {vis_heads}H "
          f"(dumping {n_vis_dump})")
    print(f"  LLM: {llm_layers}L, {llm_hidden}d, {llm_heads}H/{llm_kv_heads}KV "
          f"(dumping {n_llm_dump})")

    # ── Preprocess image ─────────────────────────────────────────────
    if args.image:
        print(f"\nPreprocessing {args.image}...")
        img = Image.open(args.image).convert("RGB")
        pixels = preprocess_image_internvl(img, vis_image_size)
    else:
        print(f"\nUsing synthetic gradient image ({vis_image_size}x{vis_image_size})...")
        # Must match the C++ test_internvl2_diff.cpp synthetic image exactly
        pixels = np.zeros((3, vis_image_size, vis_image_size), dtype=np.float32)
        pp_mean = [0.485, 0.456, 0.406]
        pp_std = [0.229, 0.224, 0.225]
        for c in range(3):
            for y in range(vis_image_size):
                for x in range(vis_image_size):
                    val = float(y * vis_image_size + x) / float(vis_image_size * vis_image_size)
                    pixels[c, y, x] = (val - pp_mean[c]) / pp_std[c]
    print(f"  Tile: {pixels.shape}")

    # ── Reference writer ─────────────────────────────────────────────
    ref = RefWriter(args.output)

    # ── Vision encoder forward pass ──────────────────────────────────
    print("\nVision encoder forward pass...")

    # Patch embedding: Conv2D
    pe_w = require("vision_model.embeddings.patch_embedding.weight")
    pe_b = get_tensor("vision_model.embeddings.patch_embedding.bias")
    # pe_w: (D, 3, P, P) → flatten to (D, 3*P*P)
    D_vis = pe_w.shape[0]
    pe_w_2d = pe_w.reshape(D_vis, -1)

    # Extract patches from image: (3, H, W) → (n_patches, 3*P*P)
    P = vis_patch
    H = W = vis_image_size
    n_ph = H // P
    n_pw = W // P
    n_patches = n_ph * n_pw
    patch_dim = 3 * P * P

    patches = np.zeros((n_patches, patch_dim), dtype=np.float32)
    idx = 0
    for ph in range(n_ph):
        for pw in range(n_pw):
            patch = pixels[:, ph*P:(ph+1)*P, pw*P:(pw+1)*P]  # (3, P, P)
            patches[idx] = patch.flatten()
            idx += 1

    # Linear projection: patches @ pe_w_2d.T + pe_b
    x = patches @ pe_w_2d.T  # (n_patches, D_vis)
    if pe_b is not None:
        x += pe_b

    # Prepend CLS token
    cls_token = require("vision_model.embeddings.class_embedding")
    cls_token = cls_token.reshape(1, D_vis)
    x = np.concatenate([cls_token, x], axis=0)  # (n_patches+1, D_vis)

    # Add position embedding
    pos_embed = require("vision_model.embeddings.position_embedding")
    pos_embed = pos_embed.reshape(-1, D_vis)  # (n_positions, D_vis)
    # If position embedding is longer than needed, truncate
    n_pos = x.shape[0]
    if pos_embed.shape[0] >= n_pos:
        x = x + pos_embed[:n_pos]
    else:
        # Need to interpolate position embeddings (for different resolutions)
        print(f"  WARNING: pos_embed {pos_embed.shape[0]} < n_pos {n_pos}, skipping")

    ref.add("vis_patch_embed", x)

    # Transformer layers
    for i in range(n_vis_dump):
        p = f"vision_model.encoder.layers.{i}."
        norm1_w = require(p + "norm1.weight")
        norm1_b = require(p + "norm1.bias")
        norm2_w = require(p + "norm2.weight")
        norm2_b = require(p + "norm2.bias")
        ls1 = get_tensor(p + "ls1")
        ls2 = get_tensor(p + "ls2")
        qkv_w = require(p + "attn.qkv.weight")
        qkv_b = require(p + "attn.qkv.bias")
        proj_w = require(p + "attn.proj.weight")
        proj_b = require(p + "attn.proj.bias")
        fc1_w = require(p + "mlp.fc1.weight")
        fc1_b = require(p + "mlp.fc1.bias")
        fc2_w = require(p + "mlp.fc2.weight")
        fc2_b = require(p + "mlp.fc2.bias")

        # Pre-norm attention: x = x + ls1 * attn(norm1(x))
        h = layernorm(x, norm1_w, norm1_b, eps=vis_ln_eps)
        h = mha_fused_qkv(h, qkv_w, qkv_b, proj_w, proj_b, vis_heads)
        if ls1 is not None:
            h = h * ls1
        x = x + h

        # Pre-norm MLP: x = x + ls2 * mlp(norm2(x))
        h = layernorm(x, norm2_w, norm2_b, eps=vis_ln_eps)
        h = linear(h, fc1_w, fc1_b)
        h = gelu(h)
        h = linear(h, fc2_w, fc2_b)
        if ls2 is not None:
            h = h * ls2
        x = x + h

        ref.add(f"vis_layer_{i}", x)
        print(f"    Layer {i}: x range [{x.min():.4f}, {x.max():.4f}]")

        # Free layer weights
        del norm1_w, norm1_b, norm2_w, norm2_b, ls1, ls2
        del qkv_w, qkv_b, proj_w, proj_b, fc1_w, fc1_b, fc2_w, fc2_b

    # Remove CLS token for pixel unshuffle
    x_no_cls = x[1:]  # (n_patches, D_vis)

    # Pixel unshuffle (v2)
    x_unshuffle = pixel_unshuffle_v2(x_no_cls, n_patches_per_side, downsample_ratio)
    ref.add("vis_pixel_unshuffle", x_unshuffle)
    print(f"  Pixel unshuffle: {x_no_cls.shape} → {x_unshuffle.shape}")

    # MLP projector: LayerNorm → Linear → GELU → Linear
    proj_norm_w = require("mlp1.0.weight")
    proj_norm_b = require("mlp1.0.bias")
    proj_fc1_w = require("mlp1.1.weight")
    proj_fc1_b = require("mlp1.1.bias")
    proj_fc2_w = require("mlp1.3.weight")
    proj_fc2_b = require("mlp1.3.bias")

    x_proj = layernorm(x_unshuffle, proj_norm_w, proj_norm_b)
    x_proj = linear(x_proj, proj_fc1_w, proj_fc1_b)
    x_proj = gelu(x_proj)
    x_proj = linear(x_proj, proj_fc2_w, proj_fc2_b)
    ref.add("vis_proj_output", x_proj)
    print(f"  Projector output: {x_proj.shape}")

    del proj_norm_w, proj_norm_b, proj_fc1_w, proj_fc1_b, proj_fc2_w, proj_fc2_b

    # ── LLM decoder forward pass ─────────────────────────────────────
    if not args.skip_llm and n_llm_dump > 0:
        print(f"\nLLM decoder forward pass ({n_llm_dump} layers)...")

        # Create a simple test sequence: just a few token IDs
        # We use sequential IDs for testing (real usage would be from tokenizer)
        test_tokens = np.array([1, 100, 200, 300, 400], dtype=np.int32)
        T = len(test_tokens)

        # Token embeddings
        embed_w = require("language_model.model.tok_embeddings.weight")
        x_llm = embed_w[test_tokens]  # (T, D)
        ref.add("llm_embed", x_llm)

        # Compute RoPE
        cos, sin = compute_rope_cos_sin(T, llm_head_dim, llm_rope_theta)

        # Causal mask: upper triangular = -inf
        mask = np.full((T, T), -np.inf, dtype=np.float32)
        mask = np.triu(mask, k=1)

        del embed_w

        # Split fused wqkv helper
        def split_wqkv(wqkv, n_heads, n_kv_heads, head_dim):
            n_groups = n_kv_heads
            q_per_group = n_heads // n_kv_heads
            gs = q_per_group + 2
            hidden = wqkv.shape[1]
            w = wqkv.reshape(n_groups, gs, head_dim, hidden)
            q = w[:, :q_per_group, :, :].reshape(n_heads * head_dim, hidden)
            k = w[:, q_per_group, :, :].reshape(n_kv_heads * head_dim, hidden)
            v = w[:, q_per_group + 1, :, :].reshape(n_kv_heads * head_dim, hidden)
            return q, k, v

        for i in range(n_llm_dump):
            p = f"language_model.model.layers.{i}."
            attn_norm_w = require(p + "attention_norm.weight")
            ffn_norm_w = require(p + "ffn_norm.weight")
            wqkv = require(p + "attention.wqkv.weight")
            wo = require(p + "attention.wo.weight")
            w1 = require(p + "feed_forward.w1.weight")
            w2 = require(p + "feed_forward.w2.weight")
            w3 = require(p + "feed_forward.w3.weight")

            q_w, k_w, v_w = split_wqkv(wqkv, llm_heads, llm_kv_heads, llm_head_dim)
            del wqkv

            # Pre-norm attention
            h = rms_norm(x_llm, attn_norm_w, eps=llm_rms_eps)
            h = gqa_attention(h, q_w, k_w, v_w, wo,
                             llm_heads, llm_kv_heads, cos, sin, mask)
            x_llm = x_llm + h

            # Pre-norm SwiGLU FFN
            h = rms_norm(x_llm, ffn_norm_w, eps=llm_rms_eps)
            gate = silu(linear(h, w1))
            up = linear(h, w3)
            h = linear(gate * up, w2)
            x_llm = x_llm + h

            ref.add(f"llm_layer_{i}", x_llm)
            print(f"    Layer {i}: x range [{x_llm.min():.4f}, {x_llm.max():.4f}]")

            del attn_norm_w, ffn_norm_w, q_w, k_w, v_w, wo, w1, w2, w3

    # ── Close ────────────────────────────────────────────────────────
    print(f"\nWriting {args.output}...")
    ref.close()
    import os
    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"Done: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
