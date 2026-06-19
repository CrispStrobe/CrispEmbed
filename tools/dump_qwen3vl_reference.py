#!/usr/bin/env python3
"""Dump per-layer Qwen3-VL-2B reference activations (safetensors, no full model load).

Pure-numpy forward pass through the vision encoder + LLM decoder, loading
weights one layer at a time via safetensors. Avoids the ~5 GB RAM spike from
loading the full model.

Architecture differences from Qwen2.5-VL (dump_qwen2vl_reference.py):
  Vision encoder:
    - LayerNorm (with bias) instead of RMSNorm
    - GELU fc1/fc2 FFN instead of SwiGLU gate/up/down
    - Learned position embeddings (bilinear interpolated) + RoPE
    - DeepStack: intermediate features at layers [5,11,17] → separate mergers
    - No windowed attention
    - patch_size=16, image_mean/std=0.5

  LLM decoder:
    - QK RMSNorm (per-head, applied before RoPE)
    - Interleaved mRoPE (THWTHW... pattern, not TTT...HHH...WWW...)
    - No attention bias
    - Tensor prefix: model.language_model.* (not model.*)

Stages captured (written to reference GGUF):

  Vision encoder:
    vis_patch_embed          (N, D_v)    patch embed + pos embed
    vis_layer_{i}            (N, D_v)    after each ViT block
    vis_merger_output        (M, D_llm)  main merger output
    vis_deepstack_{j}        (M, D_llm)  deepstack merger j output

  LLM decoder (first N layers):
    llm_embed                (T, D)      token embedding
    llm_layer_{i}            (T, D)      after each decoder layer
    llm_final_norm           (T, D)      after final RMSNorm

Usage:
    python tools/dump_qwen3vl_reference.py \\
        --model Qwen/Qwen3-VL-2B-Instruct \\
        --image /tmp/test.png \\
        --output /tmp/qwen3vl-ref.gguf \\
        --max-vis-layers 4 \\
        --max-llm-layers 2

Requires: safetensors, gguf, numpy, Pillow, huggingface_hub
Does NOT require torch or transformers at runtime.
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
    """Standard LayerNorm with bias."""
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * weight + bias


def silu(x):
    return x / (1.0 + np.exp(-x))


def gelu_exact(x):
    """Exact GELU using erf (matches PyTorch nn.GELU())."""
    try:
        from scipy.special import erf as scipy_erf
        return 0.5 * x * (1.0 + scipy_erf(x / np.sqrt(2.0)))
    except ImportError:
        # Fallback: tanh approximation
        return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


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


def apply_rotary(x, cos, sin):
    """Apply rotate_half RoPE. x: (nh, T, hd), cos/sin: (1, T, hd)."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([
        x1 * cos[..., :half] - x2 * sin[..., :half],
        x2 * cos[..., half:] + x1 * sin[..., half:],
    ], axis=-1)


# ── Vision encoder: 2D RoPE computation ──────────────────────────────

def compute_vision_rope(position_ids, head_dim, theta=10000.0):
    """Compute 2D RoPE cos/sin for vision patches.

    position_ids: (n_patches, 2) — (row, col) per patch
    Returns cos, sin each of shape (n_patches, head_dim).
    Pattern: [row_freqs, col_freqs, row_freqs, col_freqs] in quarters.
    """
    n_patches = position_ids.shape[0]
    # Qwen3VLVisionRotaryEmbedding uses head_dim//2 dimensions
    half_hd = head_dim // 2
    quart = half_hd // 2

    inv_freq = np.zeros(quart, dtype=np.float32)
    for j in range(quart):
        inv_freq[j] = 1.0 / (theta ** (2.0 * j / half_hd))

    cos_buf = np.zeros((n_patches, head_dim), dtype=np.float32)
    sin_buf = np.zeros((n_patches, head_dim), dtype=np.float32)

    for tok in range(n_patches):
        row = float(position_ids[tok, 0])
        col = float(position_ids[tok, 1])
        for j in range(quart):
            vr = row * inv_freq[j]
            vc = col * inv_freq[j]
            # Pattern: [row, col, row, col] in quarters
            cos_buf[tok, j]                = math.cos(vr)
            sin_buf[tok, j]                = math.sin(vr)
            cos_buf[tok, j + quart]        = math.cos(vc)
            sin_buf[tok, j + quart]        = math.sin(vc)
            cos_buf[tok, j + 2*quart]      = math.cos(vr)
            sin_buf[tok, j + 2*quart]      = math.sin(vr)
            cos_buf[tok, j + 3*quart]      = math.cos(vc)
            sin_buf[tok, j + 3*quart]      = math.sin(vc)

    return cos_buf, sin_buf


def compute_vision_position_ids(grid_thw, merge_size):
    """Compute (row, col) position IDs for vision patches.

    Matches HuggingFace get_vision_position_ids: patches are reordered
    into merge-size blocks (2x2 patches grouped together).

    Returns: (n_patches, 2) int32 array of (row, col) positions.
    """
    t, h, w = grid_thw
    # Build row/col grids
    hpos = np.arange(h, dtype=np.int32)[:, None] * np.ones(w, dtype=np.int32)[None, :]
    wpos = np.ones(h, dtype=np.int32)[:, None] * np.arange(w, dtype=np.int32)[None, :]

    # Reshape for merge reordering:
    # (h, w) → (h//m, m, w//m, m) → transpose(1,2) → (h//m, w//m, m, m) → flatten
    hm = h // merge_size
    wm = w // merge_size
    hpos_r = hpos.reshape(hm, merge_size, wm, merge_size).transpose(0, 2, 1, 3).flatten()
    wpos_r = wpos.reshape(hm, merge_size, wm, merge_size).transpose(0, 2, 1, 3).flatten()

    pos_ids = np.stack([hpos_r, wpos_r], axis=-1)  # (h*w, 2)
    # Repeat for temporal dimension
    if t > 1:
        pos_ids = np.tile(pos_ids, (t, 1))

    return pos_ids


def compute_bilinear_pos_embed(pos_embed_weight, grid_thw, merge_size, num_grid_per_side):
    """Compute bilinear-interpolated position embeddings.

    Matches HuggingFace get_vision_bilinear_indices_and_weights.

    pos_embed_weight: (num_position_embeddings, D) — learned embedding table
    Returns: (n_patches, D) position embeddings
    """
    t, h, w = grid_thw
    side = num_grid_per_side
    D = pos_embed_weight.shape[1]

    # Compute grid positions (linspace from 0 to side-1)
    h_grid = np.linspace(0, side - 1, h).astype(np.float32)
    w_grid = np.linspace(0, side - 1, w).astype(np.float32)

    h_floor = np.floor(h_grid).astype(np.int32)
    w_floor = np.floor(w_grid).astype(np.int32)
    h_ceil = np.minimum(h_floor + 1, side - 1)
    w_ceil = np.minimum(w_floor + 1, side - 1)

    h_frac = h_grid - h_floor.astype(np.float32)
    w_frac = w_grid - w_floor.astype(np.float32)

    # Compute corner indices (4 corners for bilinear interpolation)
    # Each is (h, w) → flattened to (h*w,)
    h_floor_offset = h_floor * side
    h_ceil_offset = h_ceil * side

    corner_indices = [
        (h_floor_offset[:, None] + w_floor[None, :]).flatten(),  # top-left
        (h_floor_offset[:, None] + w_ceil[None, :]).flatten(),   # top-right
        (h_ceil_offset[:, None] + w_floor[None, :]).flatten(),   # bottom-left
        (h_ceil_offset[:, None] + w_ceil[None, :]).flatten(),    # bottom-right
    ]
    corner_weights = [
        ((1 - h_frac)[:, None] * (1 - w_frac)[None, :]).flatten(),
        ((1 - h_frac)[:, None] * w_frac[None, :]).flatten(),
        (h_frac[:, None] * (1 - w_frac)[None, :]).flatten(),
        (h_frac[:, None] * w_frac[None, :]).flatten(),
    ]

    # Merge-aware reordering: (h,w) → (h//m, m, w//m, m) → transpose → flatten
    hm = h // merge_size
    wm = w // merge_size
    h_idx = np.arange(h, dtype=np.int32).reshape(hm, merge_size)
    w_idx = np.arange(w, dtype=np.int32).reshape(wm, merge_size)
    # reorder[i] maps from merge-reordered position to original (h,w) position
    reorder = (h_idx[:, :, None, None] * w + w_idx[None, None, :, :])
    reorder = reorder.transpose(0, 2, 1, 3).flatten()
    # Repeat for temporal
    if t > 1:
        reorder = np.tile(reorder, t)

    n_patches = len(reorder)

    # Apply reordering to corner indices and weights
    reordered_indices = [idx[reorder] for idx in corner_indices]
    reordered_weights = [wt[reorder] for wt in corner_weights]

    # Bilinear interpolation: sum of 4 corners weighted
    pos_embeds = np.zeros((n_patches, D), dtype=np.float32)
    for i in range(4):
        embeds = pos_embed_weight[reordered_indices[i]]  # (n_patches, D)
        pos_embeds += embeds * reordered_weights[i][:, None]

    return pos_embeds


# ── Image preprocessing ──────────────────────────────────────────────

def preprocess_image(img, patch_size=16, merge_size=2,
                     min_pixels=3136, max_pixels=12845056,
                     image_mean=(0.5, 0.5, 0.5),
                     image_std=(0.5, 0.5, 0.5)):
    """Preprocess image for Qwen3-VL vision encoder.

    Returns:
        patches: (n_patches, in_channels * temporal_patch_size * patch_size²)
        grid_thw: (t, h, w) = (1, H_patches, W_patches)
    """
    W, H = img.size
    factor = patch_size * merge_size  # 32

    n_pixels = W * H
    if n_pixels < min_pixels:
        scale = math.sqrt(min_pixels / n_pixels)
    elif n_pixels > max_pixels:
        scale = math.sqrt(max_pixels / n_pixels)
    else:
        scale = 1.0

    new_w = max(factor, round(W * scale / factor) * factor)
    new_h = max(factor, round(H * scale / factor) * factor)

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

    # Convert to float (3, H, W), normalize
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    arr = arr.transpose(2, 0, 1)  # (3, H, W)

    for c in range(3):
        arr[c] = (arr[c] - image_mean[c]) / image_std[c]

    # Extract patches: (n_patches, C * T * P * P)
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


# ── Spatial merge (used by both main merger and deepstack mergers) ────

def spatial_merge(x, grid_thw, merge_size, D):
    """Rearrange patches into merge_size² groups for merger input.

    x: (n_patches, D) — ViT output (in merge-reordered order)
    Returns: (n_merged, merge_size² * D) — grouped patches
    """
    _, h_p, w_p = grid_thw
    merged_h = h_p // merge_size
    merged_w = w_p // merge_size
    n_merged = merged_h * merged_w
    merge_unit = merge_size * merge_size

    # The patches are already in merge-reordered order from position_ids,
    # so consecutive merge_unit patches belong to the same spatial group
    x_merged = x.reshape(n_merged, merge_unit * D)
    return x_merged


# ── Vision encoder forward pass ──────────────────────────────────────

def run_vision_encoder(shard_files, patches, grid_thw, config,
                       max_layers=None):
    """Run Qwen3-VL vision encoder through safetensors.

    Returns dict of intermediate tensors.
    """
    vc = config["vision_config"]
    D = vc["hidden_size"]          # 1024
    n_heads = vc["num_heads"]      # 16
    head_dim = D // n_heads        # 64
    n_layers = vc["depth"]         # 24
    inter_size = vc.get("intermediate_size", 4096)
    merge = vc["spatial_merge_size"]  # 2
    out_dim = vc.get("out_hidden_size", config.get("text_config", config).get("hidden_size", 2048))
    hidden_act = vc.get("hidden_act", "gelu_pytorch_tanh")
    deepstack_indexes = vc.get("deepstack_visual_indexes", [])
    num_pos_embed = vc.get("num_position_embeddings", 2304)
    num_grid_per_side = int(math.sqrt(num_pos_embed))  # 48

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

    # Select activation function
    if hidden_act == "gelu_pytorch_tanh":
        act_fn = gelu_pytorch_tanh
    else:
        act_fn = gelu_exact

    # ── Patch embedding ──
    pe_w = require("model.visual.patch_embed.proj.weight")
    pe_w_2d = pe_w.reshape(pe_w.shape[0], -1)  # (D, C*T*P*P)
    x = patches @ pe_w_2d.T  # (n_patches, D)
    pe_b = get_tensor("model.visual.patch_embed.proj.bias")
    if pe_b is not None:
        x += pe_b

    # ── Learned position embeddings (bilinear interpolated) ──
    pos_embed_w = require("model.visual.pos_embed.weight")  # (2304, D)
    pos_embeds = compute_bilinear_pos_embed(
        pos_embed_w, grid_thw, merge, num_grid_per_side)
    x += pos_embeds

    intermediates["vis_patch_embed"] = x.copy()
    print(f"  Patch embed + pos: {x.shape}, first5={x[0, :5]}")

    # ── 2D RoPE ──
    position_ids = compute_vision_position_ids(grid_thw, merge)
    cos_buf, sin_buf = compute_vision_rope(position_ids, head_dim)

    # ── Multi-head attention helper ──
    def mha_fused_qkv(x_in, qkv_w, qkv_b, proj_w, proj_b):
        T_loc, D_loc = x_in.shape
        hd = D_loc // n_heads
        qkv = linear(x_in, qkv_w, qkv_b)  # (T, 3*D)
        Q, K, V = np.split(qkv, 3, axis=-1)

        Q = Q.reshape(T_loc, n_heads, hd).transpose(1, 0, 2)  # (nh, T, hd)
        K = K.reshape(T_loc, n_heads, hd).transpose(1, 0, 2)
        V = V.reshape(T_loc, n_heads, hd).transpose(1, 0, 2)

        # Apply RoPE
        cos_b = cos_buf[np.newaxis, :, :]  # (1, T, hd)
        sin_b = sin_buf[np.newaxis, :, :]
        Q = apply_rotary(Q, cos_b, sin_b)
        K = apply_rotary(K, cos_b, sin_b)

        scores = (Q @ K.transpose(0, 2, 1)) / math.sqrt(hd)
        attn = softmax(scores)
        out = (attn @ V).transpose(1, 0, 2).reshape(T_loc, D_loc)
        return linear(out, proj_w, proj_b)

    # ── ViT blocks ──
    deepstack_features = {}

    for li in range(n_layers):
        prefix = f"model.visual.blocks.{li}."

        # Pre-attn LayerNorm
        norm1_w = require(prefix + "norm1.weight")
        norm1_b = require(prefix + "norm1.bias")
        normed = layernorm(x, norm1_w, norm1_b)

        # Fused QKV attention
        qkv_w = require(prefix + "attn.qkv.weight")
        qkv_b = require(prefix + "attn.qkv.bias")
        proj_w = require(prefix + "attn.proj.weight")
        proj_b = require(prefix + "attn.proj.bias")

        attn_out = mha_fused_qkv(normed, qkv_w, qkv_b, proj_w, proj_b)
        x = x + attn_out

        # Pre-FFN LayerNorm
        norm2_w = require(prefix + "norm2.weight")
        norm2_b = require(prefix + "norm2.bias")
        normed2 = layernorm(x, norm2_w, norm2_b)

        # GELU FFN: fc1 → act → fc2
        fc1_w = require(prefix + "mlp.linear_fc1.weight")
        fc1_b = require(prefix + "mlp.linear_fc1.bias")
        fc2_w = require(prefix + "mlp.linear_fc2.weight")
        fc2_b = require(prefix + "mlp.linear_fc2.bias")

        ffn_out = linear(normed2, fc1_w, fc1_b)
        ffn_out = act_fn(ffn_out)
        ffn_out = linear(ffn_out, fc2_w, fc2_b)
        x = x + ffn_out

        intermediates[f"vis_layer_{li}"] = x.copy()
        print(f"  ViT L{li}: first5={x[0, :5]}")

        # DeepStack: capture features at indexed layers
        if li in deepstack_indexes:
            ds_idx = deepstack_indexes.index(li)
            ds_prefix = f"model.visual.deepstack_merger_list.{ds_idx}."

            # Spatial merge first (post-shuffle norm)
            x_shuffled = spatial_merge(x, grid_thw, merge, D)
            # (n_merged, merge²*D) = (n_merged, 4096)

            # LayerNorm on shuffled (post-shuffle)
            ds_norm_w = require(ds_prefix + "norm.weight")
            ds_norm_b = require(ds_prefix + "norm.bias")
            x_normed = layernorm(x_shuffled, ds_norm_w, ds_norm_b)

            # FC1 → GELU → FC2
            ds_fc1_w = require(ds_prefix + "linear_fc1.weight")
            ds_fc1_b = require(ds_prefix + "linear_fc1.bias")
            ds_fc2_w = require(ds_prefix + "linear_fc2.weight")
            ds_fc2_b = require(ds_prefix + "linear_fc2.bias")

            ds_out = linear(x_normed, ds_fc1_w, ds_fc1_b)
            ds_out = gelu_exact(ds_out)
            ds_out = linear(ds_out, ds_fc2_w, ds_fc2_b)

            deepstack_features[ds_idx] = ds_out
            intermediates[f"vis_deepstack_{ds_idx}"] = ds_out.copy()
            print(f"  DeepStack {ds_idx} (layer {li}): {ds_out.shape}, first5={ds_out[0, :5]}")

    # ── Main merger: pre-shuffle LayerNorm → spatial merge → FC1 → GELU → FC2 ──
    merger_norm_w = require("model.visual.merger.norm.weight")
    merger_norm_b = require("model.visual.merger.norm.bias")

    # LayerNorm on D (pre-shuffle)
    x_normed = layernorm(x, merger_norm_w, merger_norm_b)

    # Spatial merge
    x_merged = spatial_merge(x_normed, grid_thw, merge, D)

    merger_fc1_w = require("model.visual.merger.linear_fc1.weight")
    merger_fc1_b = require("model.visual.merger.linear_fc1.bias")
    merger_fc2_w = require("model.visual.merger.linear_fc2.weight")
    merger_fc2_b = require("model.visual.merger.linear_fc2.bias")

    merged_out = linear(x_merged, merger_fc1_w, merger_fc1_b)
    merged_out = gelu_exact(merged_out)
    merged_out = linear(merged_out, merger_fc2_w, merger_fc2_b)

    intermediates["vis_merger_output"] = merged_out.copy()
    print(f"  Merger: {merged_out.shape}, first5={merged_out[0, :5]}")

    return intermediates


# ── Interleaved mRoPE (Qwen3-VL LLM decoder) ────────────────────────

def apply_imrope(x, positions, sections, theta, head_dim):
    """Apply interleaved multi-dimensional RoPE to Q or K tensor.

    x: (n_heads, T, head_dim) — Q or K after reshape+transpose
    positions: (T, 3) — [temporal, height, width] positions per token
    sections: [s0, s1, s2] — how dims are split (e.g. [24, 20, 20])
    theta: rope base frequency

    Interleaved pattern: instead of [TTT...HHH...WWW...],
    dims cycle through [T,H,W,T,H,W,...].
    Rotation: neghalf — pairs are (j, j+half) where half = hd//2.
    """
    nh, T, hd = x.shape
    out = x.copy()
    half = hd // 2

    # Build section lookup: for each "slot" 0..sum(sections)-1,
    # which position dimension does it use?
    sect_dims = sum(sections)
    slot_to_pos = np.zeros(sect_dims, dtype=np.int32)
    offset = 0
    for si, s in enumerate(sections):
        for j in range(s):
            slot_to_pos[offset + j] = si
        offset += s

    for t_idx in range(T):
        pos_vals = [float(positions[t_idx, i]) for i in range(len(sections))]

        for i0 in range(0, hd, 2):
            j = i0 // 2  # pair index
            # Interleaved: slot = j % sect_dims, then cycle
            slot = j % sect_dims
            pos_dim = slot_to_pos[slot]
            pos_val = pos_vals[pos_dim]

            freq = 1.0 / (theta ** (float(i0) / hd))
            angle = pos_val * freq
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)

            # Neghalf rotation: pair (j, j+half)
            d0 = j
            d1 = j + half
            if d1 < hd:
                for h in range(nh):
                    x0 = out[h, t_idx, d0]
                    x1 = out[h, t_idx, d1]
                    out[h, t_idx, d0] = x0 * cos_a - x1 * sin_a
                    out[h, t_idx, d1] = x0 * sin_a + x1 * cos_a

    return out


# ── LLM decoder forward pass ────────────────────────────────────────

def run_llm_decoder(shard_files, input_embeds, config, max_layers=None):
    """Run first N layers of Qwen3-VL LLM decoder.

    input_embeds: (T, D) — text token embeddings.
    Returns dict of intermediate tensors.
    """
    tc = config.get("text_config", config)
    D = tc["hidden_size"]              # 2048
    n_heads = tc["num_attention_heads"]  # 16
    n_kv_heads = tc["num_key_value_heads"]  # 8
    head_dim = tc.get("head_dim", D // n_heads)  # 128
    n_layers = tc["num_hidden_layers"]  # 28
    rms_eps = tc.get("rms_norm_eps", 1e-6)

    rope_cfg = tc.get("rope_scaling", {})
    rope_sections = rope_cfg.get("mrope_section", [24, 20, 20])
    rope_theta = tc.get("rope_theta", 5000000.0)
    is_interleaved = rope_cfg.get("mrope_interleaved", True)

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
        positions[i, 0] = float(i)
        positions[i, 1] = float(i)
        positions[i, 2] = float(i)

    apply_rope_fn = apply_imrope if is_interleaved else None

    for li in range(n_layers):
        prefix = f"model.language_model.layers.{li}."

        # Pre-attn RMSNorm
        norm_w = require(prefix + "input_layernorm.weight")
        normed = rms_norm(x, norm_w, eps=rms_eps)

        # Q/K/V projections (no bias in Qwen3-VL)
        q_w = require(prefix + "self_attn.q_proj.weight")
        k_w = require(prefix + "self_attn.k_proj.weight")
        v_w = require(prefix + "self_attn.v_proj.weight")
        o_w = require(prefix + "self_attn.o_proj.weight")

        Q = linear(normed, q_w).reshape(T, n_heads, head_dim)
        K = linear(normed, k_w).reshape(T, n_kv_heads, head_dim)
        V = linear(normed, v_w).reshape(T, n_kv_heads, head_dim)

        # QK RMSNorm (per-head, applied before RoPE)
        q_norm_w = get_tensor(prefix + "self_attn.q_norm.weight")
        k_norm_w = get_tensor(prefix + "self_attn.k_norm.weight")
        if q_norm_w is not None:
            # RMSNorm on last dim (head_dim)
            q_ms = (Q ** 2).mean(axis=-1, keepdims=True)
            Q = Q / np.sqrt(q_ms + rms_eps) * q_norm_w
        if k_norm_w is not None:
            k_ms = (K ** 2).mean(axis=-1, keepdims=True)
            K = K / np.sqrt(k_ms + rms_eps) * k_norm_w

        # GQA: repeat KV heads
        kv_repeat = n_heads // n_kv_heads
        K = np.repeat(K, kv_repeat, axis=1)
        V = np.repeat(V, kv_repeat, axis=1)

        Q = Q.transpose(1, 0, 2)  # (nh, T, hd)
        K = K.transpose(1, 0, 2)
        V = V.transpose(1, 0, 2)

        # Apply mRoPE (interleaved)
        Q = apply_rope_fn(Q, positions, rope_sections, rope_theta, head_dim)
        K = apply_rope_fn(K, positions, rope_sections, rope_theta, head_dim)
        if li == 0:
            print(f"  mRoPE: sections={rope_sections}, theta={rope_theta}, interleaved={is_interleaved}")
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

        # SwiGLU FFN (same as Qwen2.5)
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
    final_norm_w = get_tensor("model.language_model.norm.weight")
    if final_norm_w is not None:
        x = rms_norm(x, final_norm_w, eps=rms_eps)
        intermediates["llm_final_norm"] = x.copy()

    return intermediates


# ── Main ──────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Dump Qwen3-VL-2B reference activations")
    p.add_argument("--model", required=True,
                   help="HF model ID (e.g. Qwen/Qwen3-VL-2B-Instruct)")
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
        image_mean = tuple(pp_cfg.get("image_mean", [0.5, 0.5, 0.5]))
        image_std = tuple(pp_cfg.get("image_std", [0.5, 0.5, 0.5]))
        pp_size = pp_cfg.get("size", {})
        min_pixels = pp_size.get("min_pixels", pp_size.get("shortest_edge", 65536))
        max_pixels = pp_size.get("max_pixels", pp_size.get("longest_edge", 16777216))
    except Exception:
        image_mean = (0.5, 0.5, 0.5)
        image_std = (0.5, 0.5, 0.5)
        min_pixels = 65536
        max_pixels = 16777216

    vc = config["vision_config"]
    patch_size = vc.get("patch_size", 16)
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

        embed_w = get_tensor_from_shards(shard_files, "model.language_model.embed_tokens.weight")

        if embed_w is not None:
            # Simple test: encode token IDs [0, 1, 2, 3, 4]
            test_ids = np.array([0, 1, 2, 3, 4], dtype=np.int32)
            input_embeds = embed_w[test_ids]
            all_intermediates["token_ids"] = test_ids.astype(np.float32)

            llm_ints = run_llm_decoder(
                shard_files, input_embeds, config,
                max_layers=args.max_llm_layers,
            )
            all_intermediates.update(llm_ints)
        else:
            print("  WARNING: embed_tokens not found, skipping LLM")

    # ── Write reference GGUF ─────────────────────────────────────
    print(f"\nWriting reference GGUF: {args.output}")
    writer = gguf.GGUFWriter(str(args.output), "qwen3vl_ref")

    writer.add_string("general.name", "qwen3vl_reference")
    writer.add_string("qwen3vl.model_id", args.model)
    writer.add_string("qwen3vl.image_path", str(args.image))
    writer.add_uint32("qwen3vl.grid_t", grid_thw[0])
    writer.add_uint32("qwen3vl.grid_h", grid_thw[1])
    writer.add_uint32("qwen3vl.grid_w", grid_thw[2])

    # Store raw preprocessed patches for C++ comparison
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

    print(f"\nDone: {n_written} tensors written to {args.output}")


def get_tensor_from_shards(shard_files, name):
    """Load a single tensor from the shard files."""
    for path in shard_files:
        with safe_open(str(path), framework="pt") as f:
            if name in f.keys():
                return f.get_tensor(name).float().numpy()
    return None


if __name__ == "__main__":
    main()
