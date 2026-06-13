#!/usr/bin/env python3
"""Dump per-layer GOT-OCR2 reference activations (safetensors, no full model load).

Pure-numpy forward pass through SAM-ViT vision encoder + neck + downsample +
projector, and layer-by-layer Qwen2 LLM decoder. Uses synthetic gradient image
for deterministic parity testing.

Usage:
    PYTHONNOUSERSITE=1 python tools/dump_got_ocr_reference.py \\
        --model stepfun-ai/GOT-OCR2_0 \\
        --output /mnt/storage/gguf-models/got-ocr2-ref.gguf \\
        --max-vis-layers 4 --max-llm-layers 2
"""

import argparse
import json
import math
from pathlib import Path

import gguf
import numpy as np
from safetensors import safe_open


def layernorm(x, weight, bias=None, eps=1e-6):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    out = (x - mean) / np.sqrt(var + eps) * weight
    if bias is not None:
        out += bias
    return out

def rms_norm(x, weight, eps=1e-6):
    ms = (x ** 2).mean(axis=-1, keepdims=True)
    return x / np.sqrt(ms + eps) * weight

def silu(x):
    return x / (1.0 + np.exp(-np.clip(x, -88.0, 88.0)))

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))

def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def linear(x, weight, bias=None):
    out = x @ weight.T
    if bias is not None:
        out += bias
    return out

def apply_rotary(x, cos, sin):
    """Standard RoPE: split first half / second half."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([
        x1 * cos[..., :half] - x2 * sin[..., :half],
        x2 * cos[..., half:] + x1 * sin[..., half:],
    ], axis=-1)


def get_rel_pos(q_size, k_size, rel_pos):
    """Decomposed relative position (matches SAM ViT HF)."""
    L = rel_pos.shape[0]
    hd = rel_pos.shape[1]
    max_rel_dist = 2 * max(q_size, k_size) - 1
    if L != max_rel_dist:
        # Interpolate
        from scipy.interpolate import interp1d
        x_old = np.linspace(0, 1, L)
        x_new = np.linspace(0, 1, max_rel_dist)
        resized = np.zeros((max_rel_dist, hd), dtype=np.float32)
        for c in range(hd):
            f = interp1d(x_old, rel_pos[:, c], kind='linear')
            resized[:, c] = f(x_new)
    else:
        resized = rel_pos

    q_scale = max(k_size / q_size, 1.0)
    k_scale = max(q_size / k_size, 1.0)
    q_coords = np.arange(q_size)[:, None] * q_scale
    k_coords = np.arange(k_size)[None, :] * k_scale
    relative_coords = (q_coords - k_coords) + (k_size - 1) * q_scale
    return resized[relative_coords.astype(int)]  # [q_size, k_size, hd]


def add_decomposed_rel_pos(attn, q, rel_pos_h, rel_pos_w, q_size, k_size):
    """Add decomposed relative position bias to attention scores."""
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)  # [q_h, k_h, hd]
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)  # [q_w, k_w, hd]

    B, _, dim = q.shape  # [B*nh, H*W, hd]
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = np.einsum("bhwc,hkc->bhwk", r_q, Rh)  # [B, q_h, q_w, k_h]
    rel_w = np.einsum("bhwc,wkc->bhwk", r_q, Rw)  # [B, q_h, q_w, k_w]

    attn = attn.reshape(B, q_h, q_w, k_h, k_w)
    attn = attn + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    return attn.reshape(B, q_h * q_w, k_h * k_w)


def window_partition(x, window_size):
    """x: (H, W, C) → (nW, ws, ws, C), pad_hw"""
    H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = np.pad(x, ((0, pad_h), (0, pad_w), (0, 0)))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.reshape(Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.transpose(0, 2, 1, 3, 4).reshape(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows, window_size, pad_hw, hw):
    """Reverse window partition."""
    Hp, Wp = pad_hw
    H, W = hw
    nWh = Hp // window_size
    nWw = Wp // window_size
    x = windows.reshape(nWh, nWw, window_size, window_size, -1)
    x = x.transpose(0, 2, 1, 3, 4).reshape(Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:H, :W, :]
    return x


def conv2d(inp, weight, bias=None, stride=1, padding=0):
    """Simple NCHW conv2d for numpy."""
    C_out, C_in, kH, kW = weight.shape
    C, H, W = inp.shape
    assert C == C_in
    if padding > 0:
        inp = np.pad(inp, ((0, 0), (padding, padding), (padding, padding)))
        _, H, W = inp.shape
    oH = (H - kH) // stride + 1
    oW = (W - kW) // stride + 1
    out = np.zeros((C_out, oH, oW), dtype=np.float32)
    for oc in range(C_out):
        b = bias[oc] if bias is not None else 0.0
        for oy in range(oH):
            for ox in range(oW):
                iy = oy * stride
                ix = ox * stride
                patch = inp[:, iy:iy+kH, ix:ix+kW]
                out[oc, oy, ox] = np.sum(patch * weight[oc]) + b
    return out


def layernorm2d(x, weight, bias, eps=1e-6):
    """LayerNorm over channel dim for (C, H, W) tensor."""
    C, H, W = x.shape
    out = np.zeros_like(x)
    for y in range(H):
        for w in range(W):
            col = x[:, y, w]
            mean = col.mean()
            var = ((col - mean) ** 2).mean()
            out[:, y, w] = (col - mean) / np.sqrt(var + eps) * weight + bias
    return out


class RefWriter:
    def __init__(self, path):
        self.writer = gguf.GGUFWriter(str(path), "got_ocr_ref")
        self.count = 0
    def add(self, name, data):
        arr = np.ascontiguousarray(data, dtype=np.float32)
        self.writer.add_tensor(name, arr, raw_dtype=gguf.GGMLQuantizationType.F32)
        self.count += 1
        print(f"  [{self.count:3d}] {name}: {list(arr.shape)}")
    def close(self):
        self.writer.write_header_to_file()
        self.writer.write_kv_data_to_file()
        self.writer.write_tensors_to_file()
        self.writer.close()
        print(f"  Total: {self.count} tensors")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-vis-layers", type=int, default=None)
    parser.add_argument("--max-llm-layers", type=int, default=None)
    parser.add_argument("--skip-llm", action="store_true")
    args = parser.parse_args()

    model_path = Path(args.model)
    is_local = model_path.is_dir()
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
            if p.exists(): return str(p)
            raise FileNotFoundError(f"{p}")
        from huggingface_hub import hf_hub_download
        return hf_hub_download(args.model, filename)

    # Build tensor map
    try:
        idx_path = resolve_file("model.safetensors.index.json")
        with open(idx_path) as f:
            idx = json.load(f)
        tensor_to_shard = idx["weight_map"]
    except Exception:
        tensor_to_shard = {}

    shard_paths = {}
    all_names = set()
    for shard in (sorted(set(tensor_to_shard.values())) if tensor_to_shard else ["model.safetensors"]):
        path = resolve_file(shard)
        shard_paths[shard] = path
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                all_names.add(key)
                if key not in tensor_to_shard:
                    tensor_to_shard[key] = shard

    print(f"Model: {args.model} ({len(all_names)} tensors)")

    def get_tensor(name):
        if name not in tensor_to_shard: return None
        with safe_open(shard_paths[tensor_to_shard[name]], framework="pt") as f:
            return f.get_tensor(name).float().numpy()

    def require(name):
        t = get_tensor(name)
        if t is None: raise ValueError(f"Required: {name}")
        return t

    # Vision params (hardcoded in GOT-OCR2)
    vis_hidden = 768
    vis_depth = 12
    vis_heads = 12
    vis_head_dim = vis_hidden // vis_heads  # 64
    vis_inter = 3072
    vis_patch = 16
    vis_image_size = 1024
    vis_window_size = 14
    vis_global_attn_indexes = [2, 5, 8, 11]
    vis_neck_out = 256

    # LLM params
    llm_hidden = config.get("hidden_size", 1024)
    llm_layers = config.get("num_hidden_layers", 24)
    llm_heads = config.get("num_attention_heads", 16)
    llm_kv_heads = config.get("num_key_value_heads", 16)
    llm_head_dim = llm_hidden // llm_heads
    llm_inter = config.get("intermediate_size", 2816)
    llm_rms_eps = config.get("rms_norm_eps", 1e-6)
    llm_rope_theta = config.get("rope_theta", 1000000.0)

    n_vis = min(vis_depth, args.max_vis_layers or vis_depth)
    n_llm = min(llm_layers, args.max_llm_layers or llm_layers)

    print(f"  Vision: {vis_depth}L, {vis_hidden}d (dumping {n_vis})")
    print(f"  LLM: {llm_layers}L, {llm_hidden}d (dumping {n_llm})")

    # Synthetic gradient image
    print(f"\nUsing synthetic gradient image ({vis_image_size}x{vis_image_size})...")
    pp_mean = [0.48145466, 0.4578275, 0.40821073]
    pp_std = [0.26862954, 0.26130258, 0.27577711]
    pixels = np.zeros((3, vis_image_size, vis_image_size), dtype=np.float32)
    for c in range(3):
        for y in range(vis_image_size):
            for x in range(vis_image_size):
                val = float(y * vis_image_size + x) / float(vis_image_size * vis_image_size)
                pixels[c, y, x] = (val - pp_mean[c]) / pp_std[c]

    ref = RefWriter(args.output)

    # ── Vision encoder ───────────────────────────────────────────
    print("\nVision encoder...")

    # Patch embed: Conv2D [768, 3, 16, 16]
    pe_w = require("model.vision_tower_high.patch_embed.proj.weight")
    pe_b = get_tensor("model.vision_tower_high.patch_embed.proj.bias")
    # pe_w shape: [768, 3, 16, 16]
    D_vis = pe_w.shape[0]
    P = pe_w.shape[2]
    pe_w_2d = pe_w.reshape(D_vis, -1)  # [768, 768]

    H = W = vis_image_size
    n_ph = H // P  # 64
    n_pw = W // P  # 64
    n_patches = n_ph * n_pw  # 4096

    patch_dim = 3 * P * P  # 768
    patches = np.zeros((n_patches, patch_dim), dtype=np.float32)
    idx = 0
    for ph in range(n_ph):
        for pw in range(n_pw):
            patch = pixels[:, ph*P:(ph+1)*P, pw*P:(pw+1)*P]  # [3, P, P]
            patches[idx] = patch.flatten()
            idx += 1

    x = patches @ pe_w_2d.T  # [n_patches, 768]
    if pe_b is not None:
        x += pe_b

    # Add absolute position embedding
    pos_embed = require("model.vision_tower_high.pos_embed")  # [1, 64, 64, 768]
    pos_flat = pos_embed.reshape(-1, vis_hidden)  # [4096, 768]
    x += pos_flat

    ref.add("vis_patch_embed", x)

    # Transformer layers (SAM ViT-B: LayerNorm + fused QKV + GELU MLP + decomposed RPE)
    # x: [4096, 768] — but attention operates on 2D spatial layout
    for i in range(n_vis):
        p = f"model.vision_tower_high.blocks.{i}."
        ln1_w = require(p + "norm1.weight")
        ln1_b = require(p + "norm1.bias")
        ln2_w = require(p + "norm2.weight")
        ln2_b = require(p + "norm2.bias")
        qkv_w = require(p + "attn.qkv.weight")
        qkv_b = require(p + "attn.qkv.bias")
        proj_w = require(p + "attn.proj.weight")
        proj_b = require(p + "attn.proj.bias")
        rel_pos_h = require(p + "attn.rel_pos_h")
        rel_pos_w = require(p + "attn.rel_pos_w")
        mlp_up_w = require(p + "mlp.lin1.weight")
        mlp_up_b = require(p + "mlp.lin1.bias")
        mlp_down_w = require(p + "mlp.lin2.weight")
        mlp_down_b = require(p + "mlp.lin2.bias")

        is_global = (i in vis_global_attn_indexes)

        # Reshape to spatial for attention
        x_2d = x.reshape(n_ph, n_pw, vis_hidden)

        # Pre-norm
        h = layernorm(x_2d, ln1_w, ln1_b, eps=1e-6)

        if is_global:
            # Global attention
            h_flat = h.reshape(-1, vis_hidden)
            qkv = linear(h_flat, qkv_w, qkv_b)
            T_seq = qkv.shape[0]
            Q, K, V = np.split(qkv, 3, axis=-1)

            Q = Q.reshape(T_seq, vis_heads, vis_head_dim)
            K = K.reshape(T_seq, vis_heads, vis_head_dim)
            V = V.reshape(T_seq, vis_heads, vis_head_dim)

            # Transpose to (nh, T, hd)
            Q_t = Q.transpose(1, 0, 2)
            K_t = K.transpose(1, 0, 2)
            V_t = V.transpose(1, 0, 2)

            # Attention scores
            scale = 1.0 / math.sqrt(vis_head_dim)
            # Q_t: [nh, T, hd], K_t: [nh, T, hd]
            scores = (Q_t * scale) @ K_t.transpose(0, 2, 1)  # [nh, T, T]

            # Decomposed relative position bias
            # Flatten Q back to (B*nh, H*W, hd) for add_decomposed_rel_pos
            Q_for_rpe = Q_t.reshape(vis_heads, T_seq, vis_head_dim)
            scores = add_decomposed_rel_pos(
                scores, Q_for_rpe, rel_pos_h, rel_pos_w,
                (n_ph, n_pw), (n_ph, n_pw))

            attn = softmax(scores)
            out = (attn @ V_t).transpose(1, 0, 2).reshape(T_seq, vis_hidden)
            out = linear(out, proj_w, proj_b)
            x = x + out
        else:
            # Window attention
            windows, pad_hw = window_partition(h, vis_window_size)
            # windows: [nW, ws, ws, C]
            nW = windows.shape[0]
            ws = vis_window_size
            wN = ws * ws

            for wi in range(nW):
                win = windows[wi]  # [ws, ws, C]
                win_flat = win.reshape(wN, vis_hidden)

                qkv = linear(win_flat, qkv_w, qkv_b)
                Q, K, V = np.split(qkv, 3, axis=-1)

                Q = Q.reshape(wN, vis_heads, vis_head_dim)
                K = K.reshape(wN, vis_heads, vis_head_dim)
                V = V.reshape(wN, vis_heads, vis_head_dim)

                Q_t = Q.transpose(1, 0, 2)
                K_t = K.transpose(1, 0, 2)
                V_t = V.transpose(1, 0, 2)

                scale = 1.0 / math.sqrt(vis_head_dim)
                scores = (Q_t * scale) @ K_t.transpose(0, 2, 1)

                Q_for_rpe = Q_t.reshape(vis_heads, wN, vis_head_dim)
                scores = add_decomposed_rel_pos(
                    scores, Q_for_rpe, rel_pos_h, rel_pos_w,
                    (ws, ws), (ws, ws))

                attn = softmax(scores)
                out = (attn @ V_t).transpose(1, 0, 2).reshape(wN, vis_hidden)
                out = linear(out, proj_w, proj_b)
                windows[wi] = (win_flat + out).reshape(ws, ws, vis_hidden)

            # Unpartition: add residual
            x_2d_unpart = window_unpartition(windows, vis_window_size, pad_hw, (n_ph, n_pw))
            # The residual was the original x_2d (before LN), and windows already include residual
            # Wait — SAM ViT does: shortcut = x; x = LN(x); x = attn(x); x = shortcut + x
            # So we need: x_2d (original) + attn_out
            # But in the windowed path above, I did win_flat + out inside the window.
            # That's wrong — win_flat is the LN'd version, not the original.
            # Let me fix: need to carry the residual separately.

            # Re-do: partition the original (pre-LN) x_2d too
            orig_windows, _ = window_partition(x_2d, vis_window_size)

            for wi in range(nW):
                win = windows[wi]  # currently = LN'd + attn_out
                # We need: orig + attn_out
                # attn_out = windows[wi] - LN'd
                # Actually let me just redo properly...
                pass

            # Actually the simpler fix: windows[wi] already has (LN'd_input + attn_output)
            # but we want (original_input + attn_output).
            # Let me recompute: attn_out_per_window = windows[wi] - LN'd_windows[wi]
            # Then result = orig_windows[wi] + attn_out_per_window

            # Let me redo the windowed path properly
            h_windows, pad_hw2 = window_partition(h, vis_window_size)  # LN'd
            orig_windows2, _ = window_partition(x_2d, vis_window_size)  # original

            for wi in range(nW):
                win = h_windows[wi].reshape(wN, vis_hidden)  # LN'd
                orig_win = orig_windows2[wi].reshape(wN, vis_hidden)

                qkv = linear(win, qkv_w, qkv_b)
                Q, K, V = np.split(qkv, 3, axis=-1)
                Q = Q.reshape(wN, vis_heads, vis_head_dim)
                K = K.reshape(wN, vis_heads, vis_head_dim)
                V = V.reshape(wN, vis_heads, vis_head_dim)

                Q_t = Q.transpose(1, 0, 2)
                K_t = K.transpose(1, 0, 2)
                V_t = V.transpose(1, 0, 2)

                scale = 1.0 / math.sqrt(vis_head_dim)
                scores = (Q_t * scale) @ K_t.transpose(0, 2, 1)
                Q_for_rpe = Q_t.reshape(vis_heads, wN, vis_head_dim)
                scores = add_decomposed_rel_pos(
                    scores, Q_for_rpe, rel_pos_h, rel_pos_w,
                    (ws, ws), (ws, ws))

                attn_w = softmax(scores)
                out = (attn_w @ V_t).transpose(1, 0, 2).reshape(wN, vis_hidden)
                out = linear(out, proj_w, proj_b)

                # residual with original (not LN'd)
                orig_windows2[wi] = (orig_win + out).reshape(ws, ws, vis_hidden)

            x_2d_unpart = window_unpartition(orig_windows2, vis_window_size, pad_hw2, (n_ph, n_pw))
            x = x_2d_unpart.reshape(-1, vis_hidden)

        # Pre-norm GELU MLP
        x_2d = x.reshape(n_ph, n_pw, vis_hidden)
        h = layernorm(x_2d, ln2_w, ln2_b, eps=1e-6)
        h_flat = h.reshape(-1, vis_hidden)
        up = gelu(linear(h_flat, mlp_up_w, mlp_up_b))
        down = linear(up, mlp_down_w, mlp_down_b)
        x = x + down

        ref.add(f"vis_layer_{i}", x)
        print(f"    Layer {i} ({'global' if is_global else 'window'}): "
              f"range [{x.min():.4f}, {x.max():.4f}]")

    # Neck: Conv(768→256,1x1) → LN2d → Conv(256→256,3x3,pad=1) → LN2d
    print("\n  Neck...")
    x_chw = x.reshape(n_ph, n_pw, vis_hidden).transpose(2, 0, 1)  # [768, 64, 64]

    neck_conv1_w = require("model.vision_tower_high.neck.0.weight")  # [256, 768, 1, 1]
    neck_ln1_w = require("model.vision_tower_high.neck.1.weight")
    neck_ln1_b = require("model.vision_tower_high.neck.1.bias")
    neck_conv2_w = require("model.vision_tower_high.neck.2.weight")  # [256, 256, 3, 3]
    neck_ln2_w = require("model.vision_tower_high.neck.3.weight")
    neck_ln2_b = require("model.vision_tower_high.neck.3.bias")

    x_chw = conv2d(x_chw, neck_conv1_w)           # [256, 64, 64]
    x_chw = layernorm2d(x_chw, neck_ln1_w, neck_ln1_b)
    x_chw = conv2d(x_chw, neck_conv2_w, padding=1) # [256, 64, 64]
    x_chw = layernorm2d(x_chw, neck_ln2_w, neck_ln2_b)
    ref.add("vis_neck_output", x_chw.transpose(1, 2, 0).reshape(-1, vis_neck_out))

    # Downsample: net_2(256→512, stride 2) → net_3(512→1024, stride 2)
    print("  Downsample...")
    net2_w = require("model.vision_tower_high.net_2.weight")  # [512, 256, 3, 3]
    net3_w = require("model.vision_tower_high.net_3.weight")  # [1024, 512, 3, 3]

    x_chw = conv2d(x_chw, net2_w, stride=2, padding=1)  # [512, 32, 32]
    x_chw = conv2d(x_chw, net3_w, stride=2, padding=1)  # [1024, 16, 16]

    # flatten(2).permute(0, 2, 1) → [1, 256, 1024]
    x_flat = x_chw.reshape(x_chw.shape[0], -1).T  # [256, 1024]
    ref.add("vis_downsample_output", x_flat)
    print(f"  Downsample output: {x_flat.shape}")

    # Projector: Linear(1024, 1024)
    proj_w = require("model.mm_projector_vary.weight")
    proj_b = get_tensor("model.mm_projector_vary.bias")
    x_proj = linear(x_flat, proj_w, proj_b)
    ref.add("vis_proj_output", x_proj)
    print(f"  Projector output: {x_proj.shape}")

    # ── LLM decoder ──────────────────────────────────────────────
    if not args.skip_llm and n_llm > 0:
        print(f"\nLLM decoder ({n_llm} layers)...")

        test_tokens = np.array([1, 100, 200, 300, 400], dtype=np.int32)
        T = len(test_tokens)

        embed_w = require("model.embed_tokens.weight")
        x_llm = embed_w[test_tokens]
        ref.add("llm_embed", x_llm)

        # Standard RoPE cos/sin
        half = llm_head_dim // 2
        inv_freq = 1.0 / (llm_rope_theta ** (np.arange(0, half, dtype=np.float64) * 2.0 / llm_head_dim))
        positions = np.arange(T, dtype=np.float64)
        freqs = np.outer(positions, inv_freq).astype(np.float32)
        cos_full = np.concatenate([np.cos(freqs), np.cos(freqs)], axis=-1)
        sin_full = np.concatenate([np.sin(freqs), np.sin(freqs)], axis=-1)
        cos_b = cos_full[np.newaxis, :, :]
        sin_b = sin_full[np.newaxis, :, :]

        mask = np.full((T, T), -np.inf, dtype=np.float32)
        mask = np.triu(mask, k=1)

        del embed_w

        for i in range(n_llm):
            p = f"model.layers.{i}."

            in_ln_w = require(p + "input_layernorm.weight")
            post_attn_ln_w = require(p + "post_attention_layernorm.weight")

            q_w = require(p + "self_attn.q_proj.weight")
            q_b = require(p + "self_attn.q_proj.bias")
            k_w = require(p + "self_attn.k_proj.weight")
            k_b = require(p + "self_attn.k_proj.bias")
            v_w = require(p + "self_attn.v_proj.weight")
            v_b = require(p + "self_attn.v_proj.bias")
            o_w = require(p + "self_attn.o_proj.weight")

            gate_w = require(p + "mlp.gate_proj.weight")
            up_w = require(p + "mlp.up_proj.weight")
            down_w = require(p + "mlp.down_proj.weight")

            # Standard pre-norm attention
            h = rms_norm(x_llm, in_ln_w, llm_rms_eps)

            Q = linear(h, q_w, q_b)   # [T, 1024]
            K = linear(h, k_w, k_b)   # [T, 1024]
            V = linear(h, v_w, v_b)   # [T, 1024]

            Q = Q.reshape(T, llm_heads, llm_head_dim).transpose(1, 0, 2)
            K = K.reshape(T, llm_kv_heads, llm_head_dim).transpose(1, 0, 2)
            V = V.reshape(T, llm_kv_heads, llm_head_dim).transpose(1, 0, 2)

            Q = apply_rotary(Q, cos_b, sin_b)
            K = apply_rotary(K, cos_b, sin_b)

            # MHA (not GQA) — kv_heads == heads
            scores = (Q @ K.transpose(0, 2, 1)) / math.sqrt(llm_head_dim)
            scores += mask[np.newaxis, :, :]
            attn_w = softmax(scores)
            out = (attn_w @ V).transpose(1, 0, 2).reshape(T, llm_hidden)
            out = linear(out, o_w)

            # Standard pre-norm residual
            x_llm = x_llm + out

            # FFN with pre-norm
            h = rms_norm(x_llm, post_attn_ln_w, llm_rms_eps)
            gate = silu(linear(h, gate_w))
            up = linear(h, up_w)
            ffn = linear(gate * up, down_w)
            x_llm = x_llm + ffn

            ref.add(f"llm_layer_{i}", x_llm)
            print(f"    Layer {i}: range [{x_llm.min():.4f}, {x_llm.max():.4f}]")

    print(f"\nWriting {args.output}...")
    ref.close()
    import os
    print(f"Done: {os.path.getsize(args.output)/1024/1024:.1f} MB")


if __name__ == "__main__":
    main()
