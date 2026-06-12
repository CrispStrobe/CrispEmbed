#!/usr/bin/env python3
"""Convert InternVL2/2.5 models to CrispEmbed GGUF.

Exports:
  - Vision encoder (InternViT-300M: 24L, 1024d, LayerNorm, GELU, LayerScale)
  - MLP projector (LN + Linear + GELU + Linear, with pixel unshuffle)
  - LLM decoder (InternLM2.5-1.8B: 24L, 2048d, GQA 16/8, SwiGLU, RMSNorm)
  - Tokenizer

Supports InternVL2-1B (Qwen2-0.5B LLM) and InternVL2.5-2B (InternLM2.5-1.8B LLM).

The fused wqkv tensor uses interleaved GQA layout:
  [Q0, Q1, K, V] per KV group, each head_dim wide.
  8 KV groups × 4 slots × 128 head_dim = 4096.
We split it into separate Q/K/V for easier C++ consumption.

Usage:
    python models/convert-internvl2-to-gguf.py \\
        --model OpenGVLab/InternVL2_5-2B \\
        --output /mnt/storage/gguf-models/internvl2.5-2b-f16.gguf \\
        --dtype f16

    python models/convert-internvl2-to-gguf.py \\
        --model OpenGVLab/InternVL2-1B \\
        --output /mnt/storage/gguf-models/internvl2-1b-f16.gguf \\
        --dtype f16
"""

import argparse
import json
import sys
from pathlib import Path

import gguf
import numpy as np

ARCH = "internvl2"


def f32(t):
    return t.detach().float().cpu().numpy().astype(np.float32)


def f16(t):
    return t.detach().float().cpu().numpy().astype(np.float16)


class Q8Tensor:
    """Wrapper for Q8_0 quantized tensor data + original shape."""
    def __init__(self, data, shape):
        self.data = data
        self.shape = shape


def q8_0(t):
    """Quantize tensor to Q8_0 (block size 32)."""
    data = t.detach().float().cpu().numpy().astype(np.float32)
    if data.ndim < 2 or data.shape[-1] % 32 != 0:
        return data
    try:
        q = gguf.quantize(data, gguf.GGMLQuantizationType.Q8_0)
        return Q8Tensor(q, data.shape)
    except Exception:
        return data


def is_norm_or_bias(name):
    """Tensors that must stay F32 (norms, biases, embeddings, layerscale)."""
    return any(k in name for k in [
        "norm", "bias", "embed_tokens", "lm_head",
        "class_embedding", "position_embedding",
        ".ls1", ".ls2",  # LayerScale — used in ggml_mul which requires F32
    ])


def add_tensor(writer, name, data, wt_func):
    """Add a tensor with appropriate quantization."""
    if isinstance(data, Q8Tensor):
        writer.add_tensor(name, data.data,
                          raw_shape=np.array(data.shape, dtype=np.uint32),
                          raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    elif isinstance(data, np.ndarray):
        if data.dtype == np.float16:
            writer.add_tensor(name, data,
                              raw_dtype=gguf.GGMLQuantizationType.F16)
        else:
            writer.add_tensor(name, data,
                              raw_dtype=gguf.GGMLQuantizationType.F32)
    else:
        raise ValueError(f"Unexpected tensor type: {type(data)}")


def split_wqkv(wqkv_weight, n_heads, n_kv_heads, head_dim):
    """Split interleaved wqkv into separate Q, K, V.

    InternLM2 wqkv layout: [n_kv_heads, (n_heads/n_kv_heads + 2), head_dim, hidden]
    Each KV group has (Q_repeat + K + V) slots, each head_dim wide.
    """
    import torch
    n_groups = n_kv_heads
    q_per_group = n_heads // n_kv_heads  # typically 2
    gs = q_per_group + 2  # group size: Q slots + K + V

    # Reshape: [n_groups * gs * head_dim, hidden] -> [n_groups, gs, head_dim, hidden]
    hidden = wqkv_weight.shape[1]
    w = wqkv_weight.reshape(n_groups, gs, head_dim, hidden)

    q = w[:, :q_per_group, :, :].reshape(n_heads * head_dim, hidden)      # all Q
    k = w[:, q_per_group, :, :].reshape(n_kv_heads * head_dim, hidden)    # all K
    v = w[:, q_per_group + 1, :, :].reshape(n_kv_heads * head_dim, hidden)  # all V

    return q, k, v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="HF model ID or local path")
    parser.add_argument("--output", required=True,
                        help="Output GGUF path")
    parser.add_argument("--dtype", choices=["f16", "f32", "q8_0"], default="f32",
                        help="Weight dtype (default f32)")
    parser.add_argument("--vision-only", action="store_true",
                        help="Export only vision encoder + projector")
    parser.add_argument("--llm-only", action="store_true",
                        help="Export only LLM decoder")
    parser.add_argument("--max-llm-layers", type=int, default=None,
                        help="Export only first N LLM layers (for testing)")
    parser.add_argument("--max-vis-layers", type=int, default=None,
                        help="Export only first N vision layers (for testing)")
    args = parser.parse_args()

    import torch
    from safetensors import safe_open

    if args.dtype == "q8_0":
        wt = q8_0
    elif args.dtype == "f16":
        wt = f16
    else:
        wt = f32

    # ── Resolve model path ───────────────────────────────────────────
    model_path = Path(args.model)
    is_local = model_path.is_dir()

    if is_local:
        config_json_path = model_path / "config.json"
    else:
        from huggingface_hub import hf_hub_download
        config_json_path = Path(hf_hub_download(args.model, "config.json"))

    with open(config_json_path) as f:
        raw_config = json.load(f)

    def resolve_file(filename):
        if is_local:
            p = model_path / filename
            if p.exists():
                return str(p)
            raise FileNotFoundError(f"{p} not found")
        from huggingface_hub import hf_hub_download
        return hf_hub_download(args.model, filename)

    # ── Build tensor → shard mapping ─────────────────────────────────
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
    all_tensor_names = set()
    for shard in shard_files:
        path = resolve_file(shard)
        shard_paths[shard] = path
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                all_tensor_names.add(key)
                if key not in tensor_to_shard:
                    tensor_to_shard[key] = shard

    print(f"Model: {args.model}")
    print(f"  {len(all_tensor_names)} tensors across {len(shard_files)} shards")

    def get_tensor(name):
        """Load a single tensor lazily from the correct shard."""
        if name not in tensor_to_shard:
            return None
        shard = tensor_to_shard[name]
        with safe_open(shard_paths[shard], framework="pt", device="cpu") as f:
            return f.get_tensor(name)

    sd = all_tensor_names  # for "key exists" checks

    # ── Parse configs ────────────────────────────────────────────────
    vc = raw_config.get("vision_config", {})
    lc = raw_config.get("llm_config", {})

    # Vision config
    vis_hidden = int(vc.get("hidden_size", 1024))
    vis_inter = int(vc.get("intermediate_size", 4096))
    vis_layers = int(vc.get("num_hidden_layers", 24))
    vis_heads = int(vc.get("num_attention_heads", 16))
    vis_head_dim = vis_hidden // vis_heads
    vis_patch = int(vc.get("patch_size", 14))
    vis_image_size = int(raw_config.get("force_image_size", vc.get("image_size", 448)))
    vis_ln_eps = float(vc.get("layer_norm_eps", 1e-6))
    vis_qkv_bias = bool(vc.get("qkv_bias", True))

    # Pixel shuffle config
    downsample_ratio = float(raw_config.get("downsample_ratio", 0.5))
    ps_version = raw_config.get("ps_version", "v2")
    n_patches_per_side = vis_image_size // vis_patch  # 32
    n_patches = n_patches_per_side ** 2  # 1024
    n_merged = int(n_patches * downsample_ratio ** 2)  # 256
    merge_dim = int(vis_hidden / (downsample_ratio ** 2))  # 4096

    # LLM config
    llm_hidden = int(lc.get("hidden_size", 2048))
    llm_inter = int(lc.get("intermediate_size", 8192))
    llm_layers = int(lc.get("num_hidden_layers", 24))
    llm_heads = int(lc.get("num_attention_heads", 16))
    llm_kv_heads = int(lc.get("num_key_value_heads", 8))
    llm_head_dim = llm_hidden // llm_heads
    llm_vocab = int(lc.get("vocab_size", 92553))
    llm_max_pos = int(lc.get("max_position_embeddings", 32768))
    llm_rms_eps = float(lc.get("rms_norm_eps", 1e-5))
    llm_rope_theta = float(lc.get("rope_theta", 1000000.0))
    llm_tie = bool(lc.get("tie_word_embeddings", False))
    llm_hidden_act = lc.get("hidden_act", "silu")

    # RoPE scaling
    rope_scaling = lc.get("rope_scaling", {})
    rope_type = rope_scaling.get("type", "")
    rope_factor = float(rope_scaling.get("factor", 1.0))

    # Dynamic resolution
    max_dynamic_patch = int(raw_config.get("max_dynamic_patch", 12))
    min_dynamic_patch = int(raw_config.get("min_dynamic_patch", 1))
    use_thumbnail = bool(raw_config.get("use_thumbnail", True))

    print(f"  Vision: {vis_layers}L, {vis_hidden}d, {vis_heads}H, "
          f"patch={vis_patch}, image={vis_image_size}")
    print(f"  Pixel shuffle: {n_patches}→{n_merged} tokens, "
          f"dim {vis_hidden}→{merge_dim}, ratio={downsample_ratio}, "
          f"ps_version={ps_version}")
    print(f"  LLM: {llm_layers}L, {llm_hidden}d, {llm_heads}H/{llm_kv_heads}KV, "
          f"inter={llm_inter}, vocab={llm_vocab}")
    print(f"  RoPE: theta={llm_rope_theta}, "
          f"scaling={rope_type}(factor={rope_factor})")

    # ── GGUF writer ──────────────────────────────────────────────────
    writer = gguf.GGUFWriter(str(args.output), ARCH)

    model_name = args.model.split("/")[-1] if "/" in args.model else Path(args.model).name
    writer.add_string("general.name", model_name)

    # Apply layer limits early so metadata is written once
    n_vis_export = vis_layers
    if args.max_vis_layers is not None:
        n_vis_export = min(n_vis_export, args.max_vis_layers)
    n_llm_export = llm_layers
    if args.max_llm_layers is not None:
        n_llm_export = min(n_llm_export, args.max_llm_layers)

    # Vision metadata
    writer.add_uint32(f"{ARCH}.vision.num_hidden_layers", n_vis_export)
    writer.add_uint32(f"{ARCH}.vision.hidden_size", vis_hidden)
    writer.add_uint32(f"{ARCH}.vision.intermediate_size", vis_inter)
    writer.add_uint32(f"{ARCH}.vision.num_attention_heads", vis_heads)
    writer.add_uint32(f"{ARCH}.vision.patch_size", vis_patch)
    writer.add_uint32(f"{ARCH}.vision.image_size", vis_image_size)
    writer.add_float32(f"{ARCH}.vision.layer_norm_eps", vis_ln_eps)
    writer.add_bool(f"{ARCH}.vision.qkv_bias", vis_qkv_bias)

    # Pixel shuffle / projector metadata
    writer.add_float32(f"{ARCH}.downsample_ratio", downsample_ratio)
    writer.add_string(f"{ARCH}.ps_version", ps_version)
    writer.add_uint32(f"{ARCH}.vision.num_merged_tokens", n_merged)
    writer.add_uint32(f"{ARCH}.vision.merge_dim", merge_dim)

    # Dynamic resolution
    writer.add_uint32(f"{ARCH}.max_dynamic_patch", max_dynamic_patch)
    writer.add_uint32(f"{ARCH}.min_dynamic_patch", min_dynamic_patch)
    writer.add_bool(f"{ARCH}.use_thumbnail", use_thumbnail)

    # LLM metadata
    writer.add_uint32(f"{ARCH}.vocab_size", llm_vocab)
    writer.add_uint32(f"{ARCH}.hidden_size", llm_hidden)
    writer.add_uint32(f"{ARCH}.intermediate_size", llm_inter)
    writer.add_uint32(f"{ARCH}.num_hidden_layers", n_llm_export)
    writer.add_uint32(f"{ARCH}.num_attention_heads", llm_heads)
    writer.add_uint32(f"{ARCH}.num_key_value_heads", llm_kv_heads)
    writer.add_uint32(f"{ARCH}.max_position_embeddings", llm_max_pos)
    writer.add_float32(f"{ARCH}.rms_norm_eps", llm_rms_eps)
    writer.add_float32(f"{ARCH}.rope_theta", llm_rope_theta)
    writer.add_bool(f"{ARCH}.tie_word_embeddings", llm_tie)
    writer.add_string(f"{ARCH}.hidden_act", llm_hidden_act)

    # RoPE scaling
    if rope_type:
        writer.add_string(f"{ARCH}.rope_scaling_type", rope_type)
        writer.add_float32(f"{ARCH}.rope_scaling_factor", rope_factor)

    # Image preprocessor
    try:
        pp_path = resolve_file("preprocessor_config.json")
        with open(pp_path) as fp:
            pp_cfg = json.load(fp)
        pp_mean = list(pp_cfg.get("image_mean", [0.485, 0.456, 0.406]))[:3]
        pp_std = list(pp_cfg.get("image_std", [0.229, 0.224, 0.225]))[:3]
        writer.add_array(f"{ARCH}.vision.image_mean", [float(x) for x in pp_mean])
        writer.add_array(f"{ARCH}.vision.image_std", [float(x) for x in pp_std])
        print(f"  Image: mean={pp_mean}, std={pp_std}")
    except Exception as e:
        # ImageNet defaults
        writer.add_array(f"{ARCH}.vision.image_mean", [0.485, 0.456, 0.406])
        writer.add_array(f"{ARCH}.vision.image_std", [0.229, 0.224, 0.225])
        print(f"  preprocessor_config.json unavailable ({e}); using ImageNet defaults")

    # ── Tokenizer ────────────────────────────────────────────────────
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        vocab = tok.get_vocab()
        n_vocab = len(vocab)
        writer.add_uint32(f"{ARCH}.tokenizer.vocab_size", n_vocab)

        if hasattr(tok, "bos_token_id") and tok.bos_token_id is not None:
            writer.add_uint32(f"{ARCH}.tokenizer.bos_id", int(tok.bos_token_id))
        if hasattr(tok, "eos_token_id") and tok.eos_token_id is not None:
            writer.add_uint32(f"{ARCH}.tokenizer.eos_id", int(tok.eos_token_id))
        if hasattr(tok, "pad_token_id") and tok.pad_token_id is not None:
            writer.add_uint32(f"{ARCH}.tokenizer.pad_id", int(tok.pad_token_id))

        # Find image placeholder token
        for name_candidate in ["<IMG_CONTEXT>", "<image>", "<img>"]:
            if name_candidate in vocab:
                writer.add_uint32(f"{ARCH}.image_token_id", vocab[name_candidate])
                print(f"  image_token: '{name_candidate}' = {vocab[name_candidate]}")
                break

        print(f"  Tokenizer: {n_vocab} tokens")
    except Exception as e:
        print(f"  Tokenizer export failed: {e}")

    # ── Vision encoder tensors ───────────────────────────────────────
    n_exported = 0
    VPFX = "v."  # compact GGUF prefix

    if not args.llm_only:
        print("\nExporting vision encoder...")

        def vw(gguf_name, hf_key, force_f32=False):
            nonlocal n_exported
            if hf_key not in sd:
                return False
            t = get_tensor(hf_key)
            data = f32(t) if (force_f32 or is_norm_or_bias(gguf_name)) else wt(t)
            del t
            add_tensor(writer, gguf_name, data, wt)
            n_exported += 1
            return True

        # Patch embedding: Conv2D [1024, 3, 14, 14]
        pe_key = "vision_model.embeddings.patch_embedding.weight"
        if pe_key in sd:
            pe_w = get_tensor(pe_key)
            # Flatten Conv2D to 2D: [out_ch, in_ch*H*W]
            pe_w_flat = pe_w.reshape(pe_w.shape[0], -1).contiguous()
            data = wt(pe_w_flat)
            add_tensor(writer, VPFX + "patch_embed.weight", data, wt)
            n_exported += 1
        vw(VPFX + "patch_embed.bias", "vision_model.embeddings.patch_embedding.bias")

        # CLS token and position embedding
        vw(VPFX + "class_embedding",
           "vision_model.embeddings.class_embedding", force_f32=True)
        vw(VPFX + "position_embedding",
           "vision_model.embeddings.position_embedding", force_f32=True)

        # Vision transformer layers
        if n_vis_export < vis_layers:
            print(f"  Vision: exporting first {n_vis_export} of {vis_layers} layers")

        for i in range(n_vis_export):
            p = f"vision_model.encoder.layers.{i}."
            q = f"{VPFX}blk.{i}."

            for hf_suf, gg_suf in [
                # LayerNorm (with bias)
                ("norm1.weight",     "norm1.weight"),
                ("norm1.bias",       "norm1.bias"),
                ("norm2.weight",     "norm2.weight"),
                ("norm2.bias",       "norm2.bias"),
                # LayerScale
                ("ls1",              "ls1"),
                ("ls2",              "ls2"),
                # Fused QKV (with bias)
                ("attn.qkv.weight",  "attn_qkv.weight"),
                ("attn.qkv.bias",    "attn_qkv.bias"),
                # Output projection (with bias)
                ("attn.proj.weight", "attn_proj.weight"),
                ("attn.proj.bias",   "attn_proj.bias"),
                # GELU MLP (standard 2-layer, with bias)
                ("mlp.fc1.weight",   "ffn_fc1.weight"),
                ("mlp.fc1.bias",     "ffn_fc1.bias"),
                ("mlp.fc2.weight",   "ffn_fc2.weight"),
                ("mlp.fc2.bias",     "ffn_fc2.bias"),
            ]:
                vw(q + gg_suf, p + hf_suf)

        print(f"  Vision: {n_vis_export} layers exported ({n_exported} tensors)")

        # MLP projector (mlp1)
        print("Exporting MLP projector...")
        for hf_suf, gg_suf in [
            ("0.weight",   "norm.weight"),      # LayerNorm
            ("0.bias",     "norm.bias"),
            ("1.weight",   "fc1.weight"),        # Linear 4096→2048
            ("1.bias",     "fc1.bias"),
            ("3.weight",   "fc2.weight"),        # Linear 2048→2048
            ("3.bias",     "fc2.bias"),
        ]:
            vw(VPFX + "proj." + gg_suf, "mlp1." + hf_suf)

        print(f"  Projector exported")

    # ── LLM decoder tensors ──────────────────────────────────────────
    if not args.vision_only:
        print("\nExporting LLM decoder...")
        LPFX = "l."

        def lw(gguf_name, hf_key, force_f32=False):
            nonlocal n_exported
            if hf_key not in sd:
                return False
            t = get_tensor(hf_key)
            data = f32(t) if (force_f32 or is_norm_or_bias(gguf_name)) else wt(t)
            del t
            add_tensor(writer, gguf_name, data, wt)
            n_exported += 1
            return True

        # Token embeddings
        lw(LPFX + "embed_tokens.weight",
           "language_model.model.tok_embeddings.weight", force_f32=True)

        # Decoder layers
        if n_llm_export < llm_layers:
            print(f"  LLM: exporting first {n_llm_export} of {llm_layers} layers")

        for i in range(n_llm_export):
            p = f"language_model.model.layers.{i}."
            q = f"{LPFX}blk.{i}."

            # RMSNorm (no bias in InternLM2.5)
            lw(q + "attn_norm.weight", p + "attention_norm.weight")
            lw(q + "ffn_norm.weight", p + "ffn_norm.weight")

            # Split fused wqkv into Q/K/V
            wqkv_key = p + "attention.wqkv.weight"
            if wqkv_key in sd:
                wqkv = get_tensor(wqkv_key)
                q_w, k_w, v_w = split_wqkv(wqkv, llm_heads, llm_kv_heads, llm_head_dim)
                del wqkv
                data_q = wt(q_w) if not is_norm_or_bias("q") else f32(q_w)
                data_k = wt(k_w) if not is_norm_or_bias("k") else f32(k_w)
                data_v = wt(v_w) if not is_norm_or_bias("v") else f32(v_w)
                del q_w, k_w, v_w
                add_tensor(writer, q + "attn_q.weight", data_q, wt)
                add_tensor(writer, q + "attn_k.weight", data_k, wt)
                add_tensor(writer, q + "attn_v.weight", data_v, wt)
                n_exported += 3

            # Output projection (no bias)
            lw(q + "attn_o.weight", p + "attention.wo.weight")

            # SwiGLU FFN (no bias): w1=gate, w3=up, w2=down
            lw(q + "ffn_gate.weight", p + "feed_forward.w1.weight")
            lw(q + "ffn_up.weight", p + "feed_forward.w3.weight")
            lw(q + "ffn_down.weight", p + "feed_forward.w2.weight")

        # Final norm + LM head
        lw(LPFX + "output_norm.weight",
           "language_model.model.norm.weight")
        lw(LPFX + "lm_head.weight",
           "language_model.output.weight", force_f32=True)

        print(f"  LLM: {n_llm_export} layers exported")

    # ── Write ────────────────────────────────────────────────────────
    print(f"\nWriting {args.output} ({n_exported} tensors)...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    import os
    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"Done: {size_mb:.0f} MB")


if __name__ == "__main__":
    main()
