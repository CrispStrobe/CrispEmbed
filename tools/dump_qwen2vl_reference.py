#!/usr/bin/env python3
"""Dump per-layer Qwen2.5-VL reference activations (safetensors, no full model load).

Pure-numpy forward pass through the vision encoder, loading weights one
layer at a time via safetensors. This avoids the ~7.5 GB RAM spike from
loading the full model with transformers.

For the LLM decoder: we use a separate pass with layer-by-layer loading.

Stages captured (written to reference GGUF):

  Vision encoder:
    vis_patch_embed          (N, D_v)    patch embed + 2D RoPE position
    vis_layer_{i}            (N, D_v)    after each ViT block
    vis_merger_output        (M, D_llm)  after spatial merger

  LLM decoder (first N layers):
    llm_embed                (T, D)      token embedding (no vision splice yet)
    llm_layer_{i}            (T, D)      after each decoder layer
    llm_logits_last          (1, V)      logits at last position

Usage:
    python tools/dump_qwen2vl_reference.py \\
        --model Qwen/Qwen2.5-VL-3B-Instruct \\
        --image /tmp/test_invoice_de.png \\
        --output /tmp/qwen2vl-ref.gguf \\
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

def rms_norm(x, weight, eps=1e-6):
    """RMSNorm: x * weight / sqrt(mean(x²) + eps)."""
    ms = (x ** 2).mean(axis=-1, keepdims=True)
    return x / np.sqrt(ms + eps) * weight


def layernorm(x, weight, bias, eps=1e-6):
    """Standard LayerNorm."""
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * weight + bias


def silu(x):
    return x / (1.0 + np.exp(-x))


def gelu_pytorch_tanh(x):
    """GELU with tanh approximation (matches PyTorch's gelu_pytorch_tanh)."""
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def linear(x, weight, bias=None):
    """x @ weight.T + bias."""
    out = x @ weight.T
    if bias is not None:
        out += bias
    return out


# ── Multi-head attention ──────────────────────────────────────────────

def mha_fused_qkv(x, qkv_w, qkv_b, proj_w, proj_b, n_heads,
                   cos=None, sin=None, mask=None):
    """Multi-head attention with fused QKV projection.

    x: (T, D), qkv_w: (3*D, D), qkv_b: (3*D,)
    cos/sin: (T, head_dim) for RoPE, or None.
    mask: (T, T) additive attention mask, or None.
    """
    T, D = x.shape
    hd = D // n_heads

    qkv = linear(x, qkv_w, qkv_b)  # (T, 3*D)
    Q, K, V = np.split(qkv, 3, axis=-1)  # each (T, D)

    Q = Q.reshape(T, n_heads, hd).transpose(1, 0, 2)  # (nh, T, hd)
    K = K.reshape(T, n_heads, hd).transpose(1, 0, 2)
    V = V.reshape(T, n_heads, hd).transpose(1, 0, 2)

    # Apply RoPE if provided
    if cos is not None and sin is not None:
        # cos/sin: (T, hd) → broadcast over heads: (1, T, hd)
        cos_b = cos[np.newaxis, :, :]
        sin_b = sin[np.newaxis, :, :]
        # Rotate-half: split into two halves, apply rotation
        Q = apply_rotary(Q, cos_b, sin_b)
        K = apply_rotary(K, cos_b, sin_b)

    scores = (Q @ K.transpose(0, 2, 1)) / math.sqrt(hd)  # (nh, T, T)
    if mask is not None:
        scores = scores + mask[np.newaxis, :, :]  # broadcast over heads

    attn = softmax(scores)
    out = (attn @ V).transpose(1, 0, 2).reshape(T, D)  # (T, D)
    return linear(out, proj_w, proj_b)


def apply_rotary(x, cos, sin):
    """Apply rotate_half RoPE. x: (nh, T, hd), cos/sin: (1, T, hd)."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([
        x1 * cos[..., :half] - x2 * sin[..., :half],
        x2 * cos[..., half:] + x1 * sin[..., half:],
    ], axis=-1)


# ── Vision encoder: 2D RoPE computation ──────────────────────────────

def compute_vision_rope(grid_thw, head_dim, theta=10000.0):
    """Compute 2D RoPE cos/sin for vision patches.

    Returns cos, sin each of shape (n_patches, head_dim).
    Pattern: [row_freqs, col_freqs, row_freqs, col_freqs] in quarters.
    """
    t, h, w = grid_thw
    n_patches = t * h * w
    quart = head_dim // 4

    inv_freq = np.zeros(quart, dtype=np.float32)
    for j in range(quart):
        inv_freq[j] = 1.0 / (theta ** (2.0 * j / head_dim))

    cos_buf = np.zeros((n_patches, head_dim), dtype=np.float32)
    sin_buf = np.zeros((n_patches, head_dim), dtype=np.float32)

    tok = 0
    for _frame in range(t):
        for row in range(h):
            for col in range(w):
                for j in range(quart):
                    vr = float(row) * inv_freq[j]
                    vc = float(col) * inv_freq[j]
                    cos_buf[tok, j]                = math.cos(vr)
                    sin_buf[tok, j]                = math.sin(vr)
                    cos_buf[tok, j + quart]        = math.cos(vc)
                    sin_buf[tok, j + quart]        = math.sin(vc)
                    cos_buf[tok, j + 2*quart]      = math.cos(vr)
                    sin_buf[tok, j + 2*quart]      = math.sin(vr)
                    cos_buf[tok, j + 3*quart]      = math.cos(vc)
                    sin_buf[tok, j + 3*quart]      = math.sin(vc)
                tok += 1

    return cos_buf, sin_buf


# ── Image preprocessing (Qwen2VL-style) ──────────────────────────────

def preprocess_image(img, patch_size=14, merge_size=2,
                     min_pixels=3136, max_pixels=12845056,
                     image_mean=(0.4815, 0.4578, 0.4082),
                     image_std=(0.2686, 0.2613, 0.2758)):
    """Preprocess image to patches for Qwen2.5-VL vision encoder.

    Returns:
        patches: (n_patches, in_channels * temporal_patch_size * patch_size * patch_size)
        grid_thw: (t, h, w) = (1, H_patches, W_patches)
    """
    W, H = img.size
    factor = patch_size * merge_size  # 28

    # Smart resize: round to multiples of factor, clamp to pixel budget
    # Scale to fit within [min_pixels, max_pixels]
    n_pixels = W * H
    if n_pixels < min_pixels:
        scale = math.sqrt(min_pixels / n_pixels)
    elif n_pixels > max_pixels:
        scale = math.sqrt(max_pixels / n_pixels)
    else:
        scale = 1.0

    new_w = max(factor, round(W * scale / factor) * factor)
    new_h = max(factor, round(H * scale / factor) * factor)

    # Ensure within bounds
    while new_w * new_h > max_pixels:
        if new_w > new_h:
            new_w -= factor
        else:
            new_h -= factor
    while new_w * new_h < min_pixels:
        if new_w < new_h:
            new_w += factor
        else:
            new_h += factor

    img_resized = img.resize((new_w, new_h), Image.BICUBIC)

    # Convert to float array (H, W, 3) → (3, H, W), normalize
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    arr = arr.transpose(2, 0, 1)  # (3, H, W)

    for c in range(3):
        arr[c] = (arr[c] - image_mean[c]) / image_std[c]

    # Extract patches: (n_patches, C * T * P * P)
    # For temporal_patch_size=2 with a single image, we duplicate the frame
    C = 3
    T_patch = 2  # temporal_patch_size
    P = patch_size
    h_patches = new_h // P
    w_patches = new_w // P

    # Duplicate frame for temporal dim
    arr_t = np.stack([arr, arr], axis=0)  # (2, 3, H, W)

    n_patches = h_patches * w_patches
    patch_dim = C * T_patch * P * P
    patches = np.zeros((n_patches, patch_dim), dtype=np.float32)

    idx = 0
    for ph in range(h_patches):
        for pw in range(w_patches):
            patch = arr_t[:, :, ph*P:(ph+1)*P, pw*P:(pw+1)*P]  # (T, C, P, P)
            patches[idx] = patch.flatten()
            idx += 1

    grid_thw = (1, h_patches, w_patches)
    print(f"  Preprocessed: {W}x{H} → {new_w}x{new_h}, "
          f"patches={n_patches} ({h_patches}x{w_patches}), "
          f"patch_dim={patch_dim}")

    return patches, grid_thw


# ── Vision encoder forward pass ──────────────────────────────────────

def run_vision_encoder(shard_files, patches, grid_thw, config,
                       max_layers=None):
    """Run Qwen2.5-VL vision encoder through safetensors.

    Returns dict of intermediate tensors.
    """
    vc = config["vision_config"]
    D = vc["hidden_size"]  # 1280
    n_heads = vc["num_heads"]  # 16
    head_dim = D // n_heads  # 80
    n_layers = vc["depth"]  # 32
    inter_size = vc.get("intermediate_size", 3420)
    merge = vc["spatial_merge_size"]  # 2
    out_dim = vc.get("out_hidden_size", config["hidden_size"])  # 2048

    if max_layers is not None:
        n_layers = min(n_layers, max_layers)

    n_patches = patches.shape[0]
    intermediates = {}

    # Build tensor name → shard file mapping
    tensor_to_shard = {}
    for path in shard_files:
        with safe_open(str(path), framework="pt") as f:
            for key in f.keys():
                tensor_to_shard[key] = str(path)

    def get_tensor(name):
        if name not in tensor_to_shard:
            return None
        with safe_open(tensor_to_shard[name], framework="pt") as f:
            return f.get_tensor(name).float().numpy()

    def require(name):
        t = get_tensor(name)
        if t is None:
            raise ValueError(f"Required tensor not found: {name}")
        return t

    # Patch embedding
    pe_w = require("visual.patch_embed.proj.weight")
    pe_w_2d = pe_w.reshape(pe_w.shape[0], -1)  # (D, in_flat)
    x = patches @ pe_w_2d.T  # (n_patches, D)
    pe_b = get_tensor("visual.patch_embed.proj.bias")
    if pe_b is not None:
        x += pe_b

    # 2D RoPE
    cos_buf, sin_buf = compute_vision_rope(grid_thw, head_dim)

    # Attention mask: all-to-all within each frame (single image = one block)
    # No mask needed for single image — all patches attend to all patches

    intermediates["vis_patch_embed"] = x.copy()
    print(f"  Patch embed: {x.shape}, first5={x[0, :5]}")

    # ViT blocks
    for li in range(n_layers):
        prefix = f"visual.blocks.{li}."

        # Pre-attn RMSNorm
        norm1_w = require(prefix + "norm1.weight")
        normed = rms_norm(x, norm1_w)

        # Fused QKV attention
        qkv_w = require(prefix + "attn.qkv.weight")
        qkv_b = require(prefix + "attn.qkv.bias")
        proj_w = require(prefix + "attn.proj.weight")
        proj_b = require(prefix + "attn.proj.bias")

        attn_out = mha_fused_qkv(normed, qkv_w, qkv_b, proj_w, proj_b,
                                  n_heads, cos=cos_buf, sin=sin_buf)
        x = x + attn_out

        # Pre-FFN RMSNorm
        norm2_w = require(prefix + "norm2.weight")
        normed2 = rms_norm(x, norm2_w)

        # SwiGLU FFN: gate * silu(up) via down
        gate_w = require(prefix + "mlp.gate_proj.weight")
        gate_b = get_tensor(prefix + "mlp.gate_proj.bias")
        up_w = require(prefix + "mlp.up_proj.weight")
        up_b = get_tensor(prefix + "mlp.up_proj.bias")
        down_w = require(prefix + "mlp.down_proj.weight")
        down_b = get_tensor(prefix + "mlp.down_proj.bias")

        gate = linear(normed2, gate_w, gate_b)
        up = linear(normed2, up_w, up_b)
        ffn_out = linear(silu(gate) * up, down_w, down_b)
        x = x + ffn_out

        intermediates[f"vis_layer_{li}"] = x.copy()
        print(f"  ViT L{li}: first5={x[0, :5]}")

    # Merger: LayerNorm → reshape (merge_size²) → FC1 → GELU → FC2
    merger_norm_w = get_tensor("visual.merger.ln_q.weight")
    merger_norm_b = get_tensor("visual.merger.ln_q.bias")
    merger_fc1_w = require("visual.merger.mlp.0.weight")
    merger_fc1_b = get_tensor("visual.merger.mlp.0.bias")
    merger_fc2_w = require("visual.merger.mlp.2.weight")
    merger_fc2_b = get_tensor("visual.merger.mlp.2.bias")

    if merger_norm_w is not None:
        if merger_norm_b is not None:
            x_normed = layernorm(x, merger_norm_w, merger_norm_b)
        else:
            # RMSNorm if no bias
            x_normed = rms_norm(x, merger_norm_w)
    else:
        x_normed = x

    # Spatial merge: reshape patches into merge_size² groups
    _, h_p, w_p = grid_thw
    merged_h = h_p // merge
    merged_w = w_p // merge
    n_merged = merged_h * merged_w

    # Reshape: (h_p, w_p, D) → (merged_h, merge, merged_w, merge, D)
    # → (merged_h, merged_w, merge*merge*D)
    x_2d = x_normed.reshape(h_p, w_p, D)
    x_merged = np.zeros((n_merged, merge * merge * D), dtype=np.float32)

    idx = 0
    for mh in range(merged_h):
        for mw in range(merged_w):
            patches_group = []
            for ir in range(merge):
                for ic in range(merge):
                    r = mh * merge + ir
                    c = mw * merge + ic
                    patches_group.append(x_2d[r, c])
            x_merged[idx] = np.concatenate(patches_group)
            idx += 1

    # FC1 → GELU → FC2
    merged_out = linear(x_merged, merger_fc1_w, merger_fc1_b)
    # Qwen2.5-VL merger uses nn.GELU() = exact GELU (erf), not tanh approx
    try:
        from scipy.special import erf as scipy_erf
        merged_out = 0.5 * merged_out * (1.0 + scipy_erf(merged_out / np.sqrt(2.0)))
    except ImportError:
        # Fallback: use math.erf element-wise (slow but correct)
        from math import erf as _erf_s
        flat = merged_out.flatten()
        for i in range(len(flat)):
            flat[i] = 0.5 * flat[i] * (1.0 + _erf_s(float(flat[i]) / math.sqrt(2.0)))
        merged_out = flat.reshape(merged_out.shape)
    merged_out = linear(merged_out, merger_fc2_w, merger_fc2_b)

    intermediates["vis_merger_output"] = merged_out.copy()
    print(f"  Merger: {merged_out.shape}, first5={merged_out[0, :5]}")

    return intermediates


# ── mRoPE (multi-dimensional rotary position embedding) ──────────────

def apply_mrope(x, positions, sections, theta, head_dim):
    """Apply multi-dimensional RoPE to Q or K tensor.

    x: (n_heads, T, head_dim) — Q or K after reshape+transpose
    positions: (T, 3) — [temporal, height, width] positions per token
    sections: [s0, s1, s2] — how head_dim is split (e.g. [16, 24, 24])
    theta: rope base frequency (1000000.0 for Qwen2.5-VL)

    Matches ggml's GGML_ROPE_TYPE_MROPE: dimensions are processed in pairs
    (i0=0,2,4,...), using a global theta that accumulates across sections.
    Section boundaries just switch which position dimension to use.
    Rotation pattern: neghalf (x0*cos - x1*sin, x1*cos + x0*sin) where
    pairs are (i0, i0+1) — adjacent dims, not split-half.
    """
    nh, T, hd = x.shape
    out = x.copy()

    # theta_scale = base^(-2/ne0) — global frequency scaling per dim pair
    theta_scale = theta ** (-2.0 / hd)

    # Build section boundaries
    sect_dims = sum(sections)  # 64 for [16,24,24]
    sec_boundaries = [0]
    for s in sections:
        sec_boundaries.append(sec_boundaries[-1] + s)
    # sec_boundaries = [0, 16, 40, 64]

    n_dims = sect_dims * 2  # 128 for [16,24,24] (each section pair = 2 dims)
    half = n_dims // 2  # 64 — neghalf split point

    for t in range(T):
        theta_vals = [float(positions[t, i]) for i in range(len(sections))]

        for i0 in range(0, n_dims, 2):
            # i0 iterates over cache pairs 0,2,4,...,126
            # Actual dim pair in neghalf layout: (j, j+half) where j=i0/2
            j = i0 // 2
            sector = j % sect_dims

            # Determine which position dimension to use
            if sector < sec_boundaries[1]:
                pos_val = theta_vals[0]  # temporal
            elif sector < sec_boundaries[2]:
                pos_val = theta_vals[1]  # height
            elif sector < sec_boundaries[3]:
                pos_val = theta_vals[2]  # width
            else:
                pos_val = 0.0  # extra/padding

            # Frequency: theta^(-2*i0/hd) = theta^(-i0/half)
            freq = 1.0 / (theta ** (float(i0) / hd))
            angle = pos_val * freq
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)

            # Neghalf rotation: pair (j, j+half)
            d0 = j
            d1 = j + half
            if d1 < hd:
                for h in range(nh):
                    x0 = out[h, t, d0]
                    x1 = out[h, t, d1]
                    out[h, t, d0] = x0 * cos_a - x1 * sin_a
                    out[h, t, d1] = x0 * sin_a + x1 * cos_a

    return out


# ── LLM decoder forward pass (first N layers) ────────────────────────

def run_llm_decoder(shard_files, input_embeds, config, max_layers=None):
    """Run first N layers of Qwen2.5 LLM decoder.

    input_embeds: (T, D) — text token embeddings (no vision splice for now).
    Returns dict of intermediate tensors.
    """
    D = config["hidden_size"]  # 2048
    n_heads = config["num_attention_heads"]  # 16
    n_kv_heads = config["num_key_value_heads"]  # 2
    head_dim = D // n_heads  # 128
    n_layers = config["num_hidden_layers"]  # 36
    inter_size = config["intermediate_size"]  # 11008
    rms_eps = config.get("rms_norm_eps", 1e-6)

    if max_layers is not None:
        n_layers = min(n_layers, max_layers)

    intermediates = {}

    # Build tensor name → shard file mapping
    tensor_to_shard = {}
    for path in shard_files:
        with safe_open(str(path), framework="pt") as f:
            for key in f.keys():
                tensor_to_shard[key] = str(path)

    def get_tensor(name):
        if name not in tensor_to_shard:
            return None
        with safe_open(tensor_to_shard[name], framework="pt") as f:
            return f.get_tensor(name).float().numpy()

    def require(name):
        t = get_tensor(name)
        if t is None:
            raise ValueError(f"Required tensor not found: {name}")
        return t

    x = input_embeds.copy()
    intermediates["llm_embed"] = x.copy()
    T = x.shape[0]

    # Simple causal mask
    causal_mask = np.full((T, T), -np.inf, dtype=np.float32)
    for i in range(T):
        for j in range(i + 1):
            causal_mask[i, j] = 0.0

    # mRoPE positions: for text-only, all 3 dims = sequential
    positions = np.zeros((T, 3), dtype=np.float32)
    for i in range(T):
        positions[i, 0] = float(i)  # temporal
        positions[i, 1] = float(i)  # height
        positions[i, 2] = float(i)  # width

    for li in range(n_layers):
        prefix = f"model.layers.{li}."

        # Pre-attn RMSNorm
        norm_w = require(prefix + "input_layernorm.weight")
        normed = rms_norm(x, norm_w, eps=rms_eps)

        # Self-attention: separate Q/K/V projections with GQA
        q_w = require(prefix + "self_attn.q_proj.weight")
        q_b = get_tensor(prefix + "self_attn.q_proj.bias")
        k_w = require(prefix + "self_attn.k_proj.weight")
        k_b = get_tensor(prefix + "self_attn.k_proj.bias")
        v_w = require(prefix + "self_attn.v_proj.weight")
        v_b = get_tensor(prefix + "self_attn.v_proj.bias")
        o_w = require(prefix + "self_attn.o_proj.weight")

        Q = linear(normed, q_w, q_b).reshape(T, n_heads, head_dim)      # (T, nh, hd)
        K = linear(normed, k_w, k_b).reshape(T, n_kv_heads, head_dim)   # (T, nkv, hd)
        V = linear(normed, v_w, v_b).reshape(T, n_kv_heads, head_dim)

        # GQA: repeat KV heads to match Q heads
        kv_repeat = n_heads // n_kv_heads
        K = np.repeat(K, kv_repeat, axis=1)  # (T, nh, hd)
        V = np.repeat(V, kv_repeat, axis=1)

        Q = Q.transpose(1, 0, 2)  # (nh, T, hd)
        K = K.transpose(1, 0, 2)
        V = V.transpose(1, 0, 2)

        # mRoPE: multi-dimensional rotary position embedding
        # sections = [16, 24, 24, 0] — how head_dim (128) is split
        # For text tokens: all 3 dims = sequential position
        rope_sections = config.get("rope_scaling", {}).get("mrope_section", [16, 24, 24])
        rope_theta = config.get("rope_theta", 1000000.0)
        Q = apply_mrope(Q, positions, rope_sections, rope_theta, head_dim)
        K = apply_mrope(K, positions, rope_sections, rope_theta, head_dim)
        if li == 0:
            print(f"  mRoPE applied: sections={rope_sections}, theta={rope_theta}")
            print(f"    Q[0,1,:5] after mRoPE: {Q[0,1,:5]}")

        scores = (Q @ K.transpose(0, 2, 1)) / math.sqrt(head_dim)
        scores = scores + causal_mask[np.newaxis, :, :]
        attn = softmax(scores)
        attn_out = (attn @ V).transpose(1, 0, 2).reshape(T, D)
        attn_out = linear(attn_out, o_w)

        x = x + attn_out

        # Pre-FFN RMSNorm
        ffn_norm_w = require(prefix + "post_attention_layernorm.weight")
        normed2 = rms_norm(x, ffn_norm_w, eps=rms_eps)

        # SwiGLU FFN
        gate_w = require(prefix + "mlp.gate_proj.weight")
        up_w = require(prefix + "mlp.up_proj.weight")
        down_w = require(prefix + "mlp.down_proj.weight")

        gate = linear(normed2, gate_w)
        up = linear(normed2, up_w)
        ffn_out = linear(silu(gate) * up, down_w)
        x = x + ffn_out

        intermediates[f"llm_layer_{li}"] = x.copy()
        print(f"  LLM L{li}: first5={x[0, :5]}")

    # Final norm
    final_norm_w = get_tensor("model.norm.weight")
    if final_norm_w is not None:
        x = rms_norm(x, final_norm_w, eps=rms_eps)
        intermediates["llm_final_norm"] = x.copy()

    return intermediates


# ── Main ──────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Dump Qwen2.5-VL reference activations")
    p.add_argument("--model", required=True,
                   help="HF model ID (e.g. Qwen/Qwen2.5-VL-3B-Instruct)")
    p.add_argument("--image", required=True,
                   help="Path to test image")
    p.add_argument("--output", required=True,
                   help="Output GGUF path for reference tensors")
    p.add_argument("--max-vis-layers", type=int, default=None,
                   help="Only run first N vision layers")
    p.add_argument("--max-llm-layers", type=int, default=None,
                   help="Only run first N LLM layers")
    p.add_argument("--skip-llm", action="store_true",
                   help="Skip LLM decoder (vision only)")
    p.add_argument("--skip-vision", action="store_true",
                   help="Skip vision encoder (LLM only)")
    args = p.parse_args()

    # Download model files
    from huggingface_hub import hf_hub_download
    print(f"Downloading config: {args.model}")

    config_path = hf_hub_download(args.model, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    # Get safetensors shard paths
    try:
        idx_path = hf_hub_download(args.model, "model.safetensors.index.json")
        with open(idx_path) as f:
            idx = json.load(f)
        shard_names = sorted(set(idx["weight_map"].values()))
        shard_files = [Path(hf_hub_download(args.model, s)) for s in shard_names]
    except Exception:
        path = Path(hf_hub_download(args.model, "model.safetensors"))
        shard_files = [path]

    print(f"  {len(shard_files)} shards")

    # Load and preprocess image
    img = Image.open(args.image).convert("RGB")
    print(f"\nImage: {img.size} ({args.image})")

    # Get preprocessor config
    try:
        pp_path = hf_hub_download(args.model, "preprocessor_config.json")
        with open(pp_path) as f:
            pp_cfg = json.load(f)
        image_mean = tuple(pp_cfg.get("image_mean", [0.4815, 0.4578, 0.4082]))
        image_std = tuple(pp_cfg.get("image_std", [0.2686, 0.2613, 0.2758]))
        pp_size = pp_cfg.get("size", {})
        min_pixels = pp_size.get("min_pixels", pp_size.get("shortest_edge", 3136))
        max_pixels = pp_size.get("max_pixels", pp_size.get("longest_edge", 12845056))
    except Exception:
        image_mean = (0.4815, 0.4578, 0.4082)
        image_std = (0.2686, 0.2613, 0.2758)
        min_pixels = 3136
        max_pixels = 12845056

    vc = config["vision_config"]
    patch_size = vc["spatial_patch_size"]
    merge_size = vc["spatial_merge_size"]

    patches, grid_thw = preprocess_image(
        img, patch_size=patch_size, merge_size=merge_size,
        min_pixels=min_pixels, max_pixels=max_pixels,
        image_mean=image_mean, image_std=image_std,
    )

    all_intermediates = {}

    # ── Vision encoder ───────────────────────────────────────────

    if not args.skip_vision:
        print(f"\n=== Vision encoder ===")
        vis_ints = run_vision_encoder(
            shard_files, patches, grid_thw, config,
            max_layers=args.max_vis_layers,
        )
        all_intermediates.update(vis_ints)

    # ── LLM decoder ──────────────────────────────────────────────

    if not args.skip_llm and args.max_llm_layers and args.max_llm_layers > 0:
        print(f"\n=== LLM decoder (first {args.max_llm_layers} layers) ===")

        # For LLM testing, use a simple text prompt (no vision splice)
        # Load token embeddings and encode a test prompt
        # For now, just use the embed_tokens weight directly
        embed_w = None
        for path in shard_files:
            with safe_open(str(path), framework="pt") as f:
                if "model.embed_tokens.weight" in f.keys():
                    embed_w = f.get_tensor("model.embed_tokens.weight").float().numpy()
                    break

        if embed_w is not None:
            # Simple test: encode token IDs [0, 1, 2, 3, 4] (first 5 vocab entries)
            test_ids = np.array([0, 1, 2, 3, 4], dtype=np.int32)
            input_embeds = embed_w[test_ids]  # (5, D)
            all_intermediates["llm_test_token_ids"] = test_ids.astype(np.float32)

            llm_ints = run_llm_decoder(
                shard_files, input_embeds, config,
                max_layers=args.max_llm_layers,
            )
            all_intermediates.update(llm_ints)
        else:
            print("  WARNING: embed_tokens not found, skipping LLM")

    # ── Write reference GGUF ─────────────────────────────────────

    print(f"\nWriting reference GGUF: {args.output}")
    writer = gguf.GGUFWriter(str(args.output), "qwen2vl_ref")

    writer.add_string("general.name", "qwen2vl_reference")
    writer.add_string("qwen2vl.model_id", args.model)
    writer.add_string("qwen2vl.image_path", str(args.image))
    writer.add_uint32("qwen2vl.grid_t", grid_thw[0])
    writer.add_uint32("qwen2vl.grid_h", grid_thw[1])
    writer.add_uint32("qwen2vl.grid_w", grid_thw[2])

    # Also store the raw preprocessed patches for C++ comparison
    writer.add_tensor("input_patches", patches.astype(np.float32),
                      raw_dtype=gguf.GGMLQuantizationType.F32)

    n_written = 0
    for name, arr in all_intermediates.items():
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        writer.add_tensor(name, arr,
                          raw_dtype=gguf.GGMLQuantizationType.F32)
        n_written += 1
        shape_str = "x".join(str(d) for d in arr.shape)
        print(f"  {name}: {shape_str} ({arr.nbytes / 1024:.1f} KB)")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    fsize = Path(args.output).stat().st_size
    print(f"\nWrote {n_written + 1} tensors to {args.output} ({fsize / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
