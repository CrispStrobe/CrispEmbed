#!/usr/bin/env python3
"""Dump per-layer GLM-OCR reference activations (safetensors, no full model load).

Pure-numpy forward pass through vision encoder + merger, and layer-by-layer
LLM decoder. Uses synthetic gradient image for deterministic parity testing.

Usage:
    PYTHONNOUSERSITE=1 python tools/dump_glm_ocr_reference.py \\
        --model zai-org/GLM-OCR \\
        --output /mnt/storage/gguf-models/glm-ocr-ref.gguf \\
        --max-vis-layers 4 --max-llm-layers 2
"""

import argparse
import json
import math
from pathlib import Path

import gguf
import numpy as np
from safetensors import safe_open


def rms_norm(x, weight, eps=1e-5):
    ms = (x ** 2).mean(axis=-1, keepdims=True)
    return x / np.sqrt(ms + eps) * weight

def silu(x):
    return x / (1.0 + np.exp(-x))

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
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([
        x1 * cos[..., :half] - x2 * sin[..., :half],
        x2 * cos[..., half:] + x1 * sin[..., half:],
    ], axis=-1)


class RefWriter:
    def __init__(self, path):
        self.writer = gguf.GGUFWriter(str(path), "glm_ocr_ref")
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

    vc = config.get("vision_config", {})
    tc = config.get("text_config", {})
    vis_hidden = vc.get("hidden_size", 1024)
    vis_depth = vc.get("depth", 24)
    vis_heads = vc.get("num_heads", 16)
    vis_head_dim = vis_hidden // vis_heads  # 64
    vis_inter = vc.get("intermediate_size", 4096)
    vis_patch = vc.get("patch_size", 14)
    vis_image_size = vc.get("image_size", 336)
    vis_temporal = vc.get("temporal_patch_size", 2)
    vis_merge = vc.get("spatial_merge_size", 2)
    vis_out_hidden = vc.get("out_hidden_size", 1536)
    vis_rms_eps = vc.get("rms_norm_eps", 1e-5)

    llm_hidden = tc.get("hidden_size", 1536)
    llm_layers = tc.get("num_hidden_layers", 16)
    llm_heads = tc.get("num_attention_heads", 16)
    llm_kv_heads = tc.get("num_key_value_heads", 8)
    llm_head_dim = tc.get("head_dim", 128)
    llm_inter = tc.get("intermediate_size", 4608)
    llm_rms_eps = tc.get("rms_norm_eps", 1e-5)
    llm_rope_theta = tc.get("rope_parameters", {}).get("rope_theta", 10000.0)

    n_vis = min(vis_depth, args.max_vis_layers or vis_depth)
    n_llm = min(llm_layers, args.max_llm_layers or llm_layers)

    print(f"  Vision: {vis_depth}L, {vis_hidden}d (dumping {n_vis})")
    print(f"  LLM: {llm_layers}L, {llm_hidden}d (dumping {n_llm})")

    # Synthetic image
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

    # Patch embed: Conv3D [1024, 3, T, P, P]
    pe_w = require("model.visual.patch_embed.proj.weight")
    pe_b = get_tensor("model.visual.patch_embed.proj.bias")
    # pe_w shape: [1024, 3, 2, 14, 14]
    D_vis = pe_w.shape[0]
    T_patch = pe_w.shape[2]
    P = pe_w.shape[3]
    pe_w_2d = pe_w.reshape(D_vis, -1)  # [1024, 3*2*14*14]

    # Extract patches: image → duplicate frame → patchify
    H = W = vis_image_size
    n_ph = H // P  # 24
    n_pw = W // P  # 24
    n_patches = n_ph * n_pw  # 576

    # Duplicate frame for temporal dim
    frames = np.stack([pixels, pixels], axis=0)  # [2, 3, H, W]
    patch_dim = 3 * T_patch * P * P  # 3*2*14*14 = 1176

    patches = np.zeros((n_patches, patch_dim), dtype=np.float32)
    idx = 0
    for ph in range(n_ph):
        for pw in range(n_pw):
            patch = frames[:, :, ph*P:(ph+1)*P, pw*P:(pw+1)*P]  # [T, C, P, P]
            patches[idx] = patch.flatten()
            idx += 1

    x = patches @ pe_w_2d.T  # [n_patches, 1024]
    if pe_b is not None:
        x += pe_b

    ref.add("vis_patch_embed", x)

    # 2D vision RoPE (authoritative: transformers modeling_glm_ocr.py).
    #   VisionRotaryEmbedding(dim=head_dim//2, theta=10000):
    #     inv_freq = 1/theta**(arange(0,dim,2)/dim)
    #   per-patch rotary_pos_emb = [h*inv_freq, w*inv_freq]  (len head_dim//2)
    #   emb = cat(rotary_pos_emb, rotary_pos_emb)            (len head_dim)
    #   cos/sin = emb.cos()/emb.sin(); apply neox rotate_half.
    # Patches here are raster-ordered, so patch (ph,pw) gets position (ph,pw);
    # for full (unmasked) attention this is equivalent to the merge-window
    # ordering used by the HF image processor.
    vis_head_dim_rope = vis_head_dim
    _quart = vis_head_dim_rope // 4
    _rot_dim = vis_head_dim_rope / 2.0
    _inv = 1.0 / (10000.0 ** (np.arange(_quart, dtype=np.float32) * 2.0 / _rot_dim))
    _cos = np.zeros((n_patches, vis_head_dim_rope), dtype=np.float32)
    _sin = np.zeros((n_patches, vis_head_dim_rope), dtype=np.float32)
    _tok = 0
    for _ph in range(n_ph):
        for _pw in range(n_pw):
            _vr = _ph * _inv
            _vw = _pw * _inv
            _emb = np.concatenate([_vr, _vw, _vr, _vw])
            _cos[_tok] = np.cos(_emb)
            _sin[_tok] = np.sin(_emb)
            _tok += 1

    def _rotate_half(t):
        h2 = t.shape[-1] // 2
        return np.concatenate([-t[..., h2:], t[..., :h2]], axis=-1)

    # Transformer layers (CogViT: RMSNorm + fused QKV + Q/K norm + SwiGLU)
    for i in range(n_vis):
        p = f"model.visual.blocks.{i}."
        norm1_w = require(p + "norm1.weight")
        norm2_w = require(p + "norm2.weight")
        qkv_w = require(p + "attn.qkv.weight")
        qkv_b = require(p + "attn.qkv.bias")
        proj_w = require(p + "attn.proj.weight")
        proj_b = require(p + "attn.proj.bias")
        q_norm_w = require(p + "attn.q_norm.weight")
        k_norm_w = require(p + "attn.k_norm.weight")
        gate_w = require(p + "mlp.gate_proj.weight")
        gate_b = require(p + "mlp.gate_proj.bias")
        up_w = require(p + "mlp.up_proj.weight")
        up_b = require(p + "mlp.up_proj.bias")
        down_w = require(p + "mlp.down_proj.weight")
        down_b = require(p + "mlp.down_proj.bias")

        # Pre-norm attention
        h = rms_norm(x, norm1_w, vis_rms_eps)

        # Fused QKV
        qkv = linear(h, qkv_w, qkv_b)  # [N, 3*D]
        T_seq = qkv.shape[0]
        Q, K, V = np.split(qkv, 3, axis=-1)

        Q = Q.reshape(T_seq, vis_heads, vis_head_dim)
        K = K.reshape(T_seq, vis_heads, vis_head_dim)
        V = V.reshape(T_seq, vis_heads, vis_head_dim)

        # Q/K RMSNorm (per head)
        for head in range(vis_heads):
            q_ms = (Q[:, head, :] ** 2).mean(axis=-1, keepdims=True)
            Q[:, head, :] = Q[:, head, :] / np.sqrt(q_ms + vis_rms_eps) * q_norm_w
            k_ms = (K[:, head, :] ** 2).mean(axis=-1, keepdims=True)
            K[:, head, :] = K[:, head, :] / np.sqrt(k_ms + vis_rms_eps) * k_norm_w

        # 2D vision RoPE (per-patch cos/sin broadcast over heads).
        _cb = _cos[:, None, :]
        _sb = _sin[:, None, :]
        Q = Q * _cb + _rotate_half(Q) * _sb
        K = K * _cb + _rotate_half(K) * _sb

        Q = Q.transpose(1, 0, 2)  # [nh, T, hd]
        K = K.transpose(1, 0, 2)
        V = V.transpose(1, 0, 2)

        # Attention (bidirectional, no mask)
        scores = (Q @ K.transpose(0, 2, 1)) / math.sqrt(vis_head_dim)
        attn = softmax(scores)
        out = (attn @ V).transpose(1, 0, 2).reshape(T_seq, vis_hidden)
        out = linear(out, proj_w, proj_b)
        x = x + out

        # Pre-norm SwiGLU FFN
        h = rms_norm(x, norm2_w, vis_rms_eps)
        gate = silu(linear(h, gate_w, gate_b))
        up = linear(h, up_w, up_b)
        h = linear(gate * up, down_w, down_b)
        x = x + h

        ref.add(f"vis_layer_{i}", x)
        print(f"    Layer {i}: range [{x.min():.4f}, {x.max():.4f}]")

    # Post-layernorm
    post_ln_w = require("model.visual.post_layernorm.weight")
    x = rms_norm(x, post_ln_w, vis_rms_eps)
    ref.add("vis_post_norm", x)

    # Spatial downsample: Conv2D [1536, 1024, 2, 2] stride 2
    ds_w = require("model.visual.downsample.weight")
    ds_b = get_tensor("model.visual.downsample.bias")
    # x: [576, 1024] → reshape to [24, 24, 1024]
    x_2d = x.reshape(n_ph, n_pw, vis_hidden)
    # Conv2D stride 2: output [12, 12, 1536]
    out_h = n_ph // vis_merge
    out_w = n_pw // vis_merge
    x_ds = np.zeros((out_h, out_w, vis_out_hidden), dtype=np.float32)
    # ds_w: [1536, 1024, 2, 2]
    for oh in range(out_h):
        for ow in range(out_w):
            patch = x_2d[oh*2:oh*2+2, ow*2:ow*2+2, :]  # [2, 2, 1024]
            patch_flat = patch.transpose(2, 0, 1).reshape(1024 * 4)  # [4096]
            ds_w_flat = ds_w.reshape(vis_out_hidden, -1)  # [1536, 4096]
            x_ds[oh, ow, :] = ds_w_flat @ patch_flat
            if ds_b is not None:
                x_ds[oh, ow, :] += ds_b
    x_ds = x_ds.reshape(-1, vis_out_hidden)  # [144, 1536]
    ref.add("vis_downsample", x_ds)
    print(f"  Downsample: {x.shape} → {x_ds.shape}")

    # Merger: proj → SwiGLU → LayerNorm
    merger_proj_w = require("model.visual.merger.proj.weight")
    merger_gate_w = require("model.visual.merger.gate_proj.weight")
    merger_up_w = require("model.visual.merger.up_proj.weight")
    merger_down_w = require("model.visual.merger.down_proj.weight")
    merger_norm_w = require("model.visual.merger.post_projection_norm.weight")
    merger_norm_b = get_tensor("model.visual.merger.post_projection_norm.bias")

    x_m = linear(x_ds, merger_proj_w)
    gate = silu(linear(x_m, merger_gate_w))
    up = linear(x_m, merger_up_w)
    x_m = linear(gate * up, merger_down_w)
    # LayerNorm (not RMSNorm — has bias)
    mean = x_m.mean(axis=-1, keepdims=True)
    var = ((x_m - mean) ** 2).mean(axis=-1, keepdims=True)
    x_m = (x_m - mean) / np.sqrt(var + 1e-6) * merger_norm_w
    if merger_norm_b is not None:
        x_m += merger_norm_b
    ref.add("vis_merger_output", x_m)
    print(f"  Merger output: {x_m.shape}")

    # ── LLM decoder ──────────────────────────────────────────────
    if not args.skip_llm and n_llm > 0:
        print(f"\nLLM decoder ({n_llm} layers)...")

        test_tokens = np.array([1, 100, 200, 300, 400], dtype=np.int32)
        T = len(test_tokens)

        embed_w = require("model.language_model.embed_tokens.weight")
        x_llm = embed_w[test_tokens]
        ref.add("llm_embed", x_llm)

        # mRoPE cos/sin — for text-only test, all 3 dims use same position
        # sections=[16,24,24] → head_dim/2 = 64 frequencies split across 3 dims
        # For simplicity, use standard 1D RoPE for text-only test
        half = llm_head_dim // 2
        inv_freq = 1.0 / (llm_rope_theta ** (np.arange(0, half, dtype=np.float32) * 2.0 / llm_head_dim))
        positions = np.arange(T, dtype=np.float32)
        freqs = np.outer(positions, inv_freq)
        cos_full = np.concatenate([np.cos(freqs), np.cos(freqs)], axis=-1)
        sin_full = np.concatenate([np.sin(freqs), np.sin(freqs)], axis=-1)
        cos_b = cos_full[np.newaxis, :, :]
        sin_b = sin_full[np.newaxis, :, :]

        mask = np.full((T, T), -np.inf, dtype=np.float32)
        mask = np.triu(mask, k=1)

        del embed_w

        for i in range(n_llm):
            p = f"model.language_model.layers.{i}."

            in_ln_w = require(p + "input_layernorm.weight")
            post_sa_ln_w = require(p + "post_self_attn_layernorm.weight")
            post_attn_ln_w = require(p + "post_attention_layernorm.weight")
            post_mlp_ln_w = require(p + "post_mlp_layernorm.weight")

            q_w = require(p + "self_attn.q_proj.weight")
            k_w = require(p + "self_attn.k_proj.weight")
            v_w = require(p + "self_attn.v_proj.weight")
            o_w = require(p + "self_attn.o_proj.weight")

            gate_up = require(p + "mlp.gate_up_proj.weight")
            gate_w = gate_up[:llm_inter, :]
            up_w = gate_up[llm_inter:, :]
            down_w = require(p + "mlp.down_proj.weight")

            # Pre-norm attention (post-norm with 4 norms)
            # Pattern: input_ln → attn → post_self_attn_ln → residual
            #          post_attention_ln → FFN → post_mlp_ln → residual
            h = rms_norm(x_llm, in_ln_w, llm_rms_eps)

            Q = linear(h, q_w)   # [T, 2048]
            K = linear(h, k_w)   # [T, 1024]
            V = linear(h, v_w)   # [T, 1024]

            Q = Q.reshape(T, llm_heads, llm_head_dim).transpose(1, 0, 2)
            K = K.reshape(T, llm_kv_heads, llm_head_dim).transpose(1, 0, 2)
            V = V.reshape(T, llm_kv_heads, llm_head_dim).transpose(1, 0, 2)

            Q = apply_rotary(Q, cos_b, sin_b)
            K = apply_rotary(K, cos_b, sin_b)

            kv_repeat = llm_heads // llm_kv_heads
            K_exp = np.repeat(K, kv_repeat, axis=0)
            V_exp = np.repeat(V, kv_repeat, axis=0)

            scores = (Q @ K_exp.transpose(0, 2, 1)) / math.sqrt(llm_head_dim)
            scores += mask[np.newaxis, :, :]
            attn = softmax(scores)
            out = (attn @ V_exp).transpose(1, 0, 2).reshape(T, llm_heads * llm_head_dim)
            out = linear(out, o_w)

            # Post-norm pattern: attn_out → post_self_attn_ln → + residual
            out = rms_norm(out, post_sa_ln_w, llm_rms_eps)
            x_llm = x_llm + out

            # FFN with post-norm
            h = rms_norm(x_llm, post_attn_ln_w, llm_rms_eps)
            gate = silu(linear(h, gate_w))
            up = linear(h, up_w)
            ffn = linear(gate * up, down_w)
            ffn = rms_norm(ffn, post_mlp_ln_w, llm_rms_eps)
            x_llm = x_llm + ffn

            ref.add(f"llm_layer_{i}", x_llm)
            print(f"    Layer {i}: range [{x_llm.min():.4f}, {x_llm.max():.4f}]")

    print(f"\nWriting {args.output}...")
    ref.close()
    import os
    print(f"Done: {os.path.getsize(args.output)/1024/1024:.1f} MB")


if __name__ == "__main__":
    main()
