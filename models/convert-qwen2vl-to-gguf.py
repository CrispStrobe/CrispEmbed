#!/usr/bin/env python3
"""Convert Qwen2.5-VL models (3B/7B) to CrispEmbed GGUF.

Exports:
  - Vision encoder (ViT with windowed attention)
  - LLM decoder (Qwen2.5 with mRoPE)
  - Tokenizer (GPT-2 BPE from GGUF metadata)
  - Image preprocessor config

The output GGUF can be loaded by src/qwen2vl_ocr.cpp for OCR inference.

Usage:
    python models/convert-qwen2vl-to-gguf.py \\
        --model Qwen/Qwen2.5-VL-3B-Instruct \\
        --output /mnt/storage/gguf-models/qwen2.5-vl-3b.gguf

    python models/convert-qwen2vl-to-gguf.py \\
        --model Qwen/Qwen2.5-VL-3B-Instruct \\
        --dtype f16 \\
        --output /mnt/storage/gguf-models/qwen2.5-vl-3b-f16.gguf
"""

import argparse
import json
import sys
from pathlib import Path

import gguf
import numpy as np
import torch
from transformers import AutoConfig


ARCH = "qwen2vl"


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
    """Check if a tensor should always be stored as F32."""
    return any(k in name for k in [
        "norm", "bias", "embed_tokens", "lm_head",
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="HF model ID (e.g. Qwen/Qwen2.5-VL-3B-Instruct)")
    parser.add_argument("--output", required=True,
                        help="Output GGUF path")
    parser.add_argument("--dtype", choices=["f16", "f32", "q8_0"], default="f32",
                        help="Weight dtype (f32 default, f16 for smaller, q8_0 for quantized)")
    parser.add_argument("--load-dtype", choices=["bfloat16", "float32"],
                        default="bfloat16",
                        help="Precision for loading HF weights (bfloat16 saves RAM)")
    parser.add_argument("--vision-only", action="store_true",
                        help="Export only vision encoder (for mmproj-style GGUF)")
    parser.add_argument("--llm-only", action="store_true",
                        help="Export only LLM decoder (vision encoder excluded)")
    parser.add_argument("--max-llm-layers", type=int, default=None,
                        help="Export only first N LLM layers (for testing)")
    args = parser.parse_args()

    if args.dtype == "q8_0":
        wt = q8_0
    elif args.dtype == "f16":
        wt = f16
    else:
        wt = f32

    print(f"Loading config: {args.model}")
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)

    from safetensors import safe_open
    from huggingface_hub import hf_hub_download

    # Build tensor → shard mapping (memory-efficient: don't load all at once)
    try:
        idx_path = hf_hub_download(args.model, "model.safetensors.index.json")
        with open(idx_path) as f:
            idx = json.load(f)
        shard_files = sorted(set(idx["weight_map"].values()))
        tensor_to_shard = idx["weight_map"]
    except Exception:
        shard_files = ["model.safetensors"]
        tensor_to_shard = None  # will scan

    # Download all shards and build path map
    shard_paths = {}
    all_tensor_names = set()
    for shard in shard_files:
        path = hf_hub_download(args.model, shard)
        shard_paths[shard] = path
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                all_tensor_names.add(key)
                if tensor_to_shard is None:
                    if not hasattr(convert_qwen2vl_to_gguf, '_tsmap'):
                        convert_qwen2vl_to_gguf._tsmap = {}
                    convert_qwen2vl_to_gguf._tsmap[key] = shard

    if tensor_to_shard is None:
        tensor_to_shard = getattr(convert_qwen2vl_to_gguf, '_tsmap', {})

    print(f"  {len(all_tensor_names)} tensors across {len(shard_files)} shards")

    # Lazy tensor loader: load one tensor at a time, immediately free
    def get_tensor(name):
        if name not in tensor_to_shard:
            return None
        shard = tensor_to_shard[name]
        with safe_open(shard_paths[shard], framework="pt", device="cpu") as f:
            return f.get_tensor(name)

    sd = all_tensor_names  # use as "key exists" check

    # ── GGUF writer ──────────────────────────────────────────────────

    writer = gguf.GGUFWriter(str(args.output), ARCH)

    # ── Global metadata ──────────────────────────────────────────────

    writer.add_string("general.name", args.model.split("/")[-1])
    writer.add_string("general.architecture", ARCH)

    # LLM config
    tc = config  # text config is the top-level config for Qwen2.5-VL
    writer.add_uint32("qwen2vl.vocab_size", int(tc.vocab_size))
    writer.add_uint32("qwen2vl.hidden_size", int(tc.hidden_size))
    writer.add_uint32("qwen2vl.intermediate_size", int(tc.intermediate_size))
    writer.add_uint32("qwen2vl.num_hidden_layers", int(tc.num_hidden_layers))
    writer.add_uint32("qwen2vl.num_attention_heads", int(tc.num_attention_heads))
    writer.add_uint32("qwen2vl.num_key_value_heads", int(tc.num_key_value_heads))
    writer.add_uint32("qwen2vl.max_position_embeddings", int(tc.max_position_embeddings))
    writer.add_float32("qwen2vl.rms_norm_eps", float(tc.rms_norm_eps))
    writer.add_float32("qwen2vl.rope_theta", float(tc.rope_theta))
    writer.add_bool("qwen2vl.tie_word_embeddings", bool(tc.tie_word_embeddings))

    # mRoPE sections
    rope_scaling = getattr(tc, "rope_scaling", None)
    if rope_scaling and "mrope_section" in rope_scaling:
        sections = rope_scaling["mrope_section"]
        writer.add_array("qwen2vl.rope_sections", [int(x) for x in sections])
        print(f"  mRoPE sections: {sections}")

    # Vision config
    vc = config.vision_config
    writer.add_uint32("qwen2vl.vision.depth", int(vc.depth))
    writer.add_uint32("qwen2vl.vision.hidden_size", int(vc.hidden_size))
    writer.add_uint32("qwen2vl.vision.intermediate_size", int(vc.intermediate_size))
    writer.add_uint32("qwen2vl.vision.num_heads", int(vc.num_heads))
    writer.add_uint32("qwen2vl.vision.in_channels", int(vc.in_channels))
    writer.add_uint32("qwen2vl.vision.spatial_patch_size", int(vc.spatial_patch_size))
    writer.add_uint32("qwen2vl.vision.spatial_merge_size", int(vc.spatial_merge_size))
    writer.add_uint32("qwen2vl.vision.temporal_patch_size", int(vc.temporal_patch_size))
    writer.add_uint32("qwen2vl.vision.out_hidden_size", int(vc.out_hidden_size))

    # Windowed attention config
    if hasattr(vc, "window_size"):
        writer.add_uint32("qwen2vl.vision.window_size", int(vc.window_size))
    if hasattr(vc, "fullatt_block_indexes"):
        writer.add_array("qwen2vl.vision.fullatt_block_indexes",
                         [int(x) for x in vc.fullatt_block_indexes])
        print(f"  Vision fullatt blocks: {list(vc.fullatt_block_indexes)}")

    print(f"  LLM: {tc.num_hidden_layers}L, {tc.hidden_size}d, "
          f"{tc.num_attention_heads}H/{tc.num_key_value_heads}KV, "
          f"inter={tc.intermediate_size}")
    print(f"  Vision: {vc.depth}L, {vc.hidden_size}d, {vc.num_heads}H, "
          f"patch={vc.spatial_patch_size}, merge={vc.spatial_merge_size}, "
          f"out={vc.out_hidden_size}")

    # Image preprocessor config
    try:
        pp_path = hf_hub_download(args.model, "preprocessor_config.json")
        with open(pp_path) as fp:
            pp_cfg = json.load(fp)
        pp_mean = list(pp_cfg.get("image_mean", [0.5, 0.5, 0.5]))[:3]
        pp_std = list(pp_cfg.get("image_std", [0.5, 0.5, 0.5]))[:3]
        pp_size = pp_cfg.get("size", {})
        pp_min = int(pp_size.get("min_pixels", pp_size.get("shortest_edge", 3136)))
        pp_max = int(pp_size.get("max_pixels", pp_size.get("longest_edge", 12845056)))
        writer.add_array("qwen2vl.vision.image_mean", [float(x) for x in pp_mean])
        writer.add_array("qwen2vl.vision.image_std", [float(x) for x in pp_std])
        writer.add_uint32("qwen2vl.vision.min_pixels", pp_min)
        writer.add_uint32("qwen2vl.vision.max_pixels", pp_max)
        print(f"  Image: mean={pp_mean}, std={pp_std}, "
              f"min_px={pp_min}, max_px={pp_max}")
    except Exception as e:
        print(f"  preprocessor_config.json unavailable ({e}); using defaults")

    # ── Tokenizer ────────────────────────────────────────────────────

    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        # Store vocab as GGUF tokens
        vocab = tok.get_vocab()
        n_vocab = len(vocab)
        # Sort by token ID
        tokens = [""] * n_vocab
        for token_str, token_id in vocab.items():
            if token_id < n_vocab:
                tokens[token_id] = token_str
        writer.add_uint32("qwen2vl.tokenizer.vocab_size", n_vocab)

        # Store special token IDs
        if hasattr(tok, "bos_token_id") and tok.bos_token_id is not None:
            writer.add_uint32("qwen2vl.tokenizer.bos_id", int(tok.bos_token_id))
        if hasattr(tok, "eos_token_id") and tok.eos_token_id is not None:
            writer.add_uint32("qwen2vl.tokenizer.eos_id", int(tok.eos_token_id))
        if hasattr(tok, "pad_token_id") and tok.pad_token_id is not None:
            writer.add_uint32("qwen2vl.tokenizer.pad_id", int(tok.pad_token_id))

        # Vision special tokens
        for special_name in ["image_token_id", "video_token_id",
                             "vision_start_token_id", "vision_end_token_id"]:
            val = getattr(config, special_name, None)
            if val is not None:
                writer.add_uint32(f"qwen2vl.{special_name}", int(val))
                print(f"  {special_name}: {val}")

        print(f"  Tokenizer: {n_vocab} tokens")
    except Exception as e:
        print(f"  Tokenizer export failed: {e}")

    # ── Vision encoder tensors ───────────────────────────────────────

    if not args.llm_only:
        print("\nExporting vision encoder...")
        VPFX = "v."  # compact GGUF prefix

        def vw(gguf_name, hf_key):
            """Write a vision tensor."""
            if hf_key not in sd:
                return False
            t = get_tensor(hf_key)
            data = f32(t) if is_norm_or_bias(gguf_name) else wt(t)
            del t
            add_tensor(writer, gguf_name, data, wt)
            return True

        # Patch embedding: Conv3D weight (out, in, T, H, W) → flatten to 2D
        pe_key = "visual.patch_embed.proj.weight"
        if pe_key in sd:
            pe_w = get_tensor(pe_key)
            pe_w_flat = pe_w.reshape(pe_w.shape[0], -1).contiguous()
            data = wt(pe_w_flat)
            add_tensor(writer, VPFX + "patch_embed.weight", data, wt)
        vw(VPFX + "patch_embed.bias", "visual.patch_embed.proj.bias")

        # ViT blocks (32 layers for 3B)
        n_vis_layers = int(vc.depth)
        for i in range(n_vis_layers):
            p = f"visual.blocks.{i}."
            q = f"{VPFX}blk.{i}."

            for hf_suf, gg_suf in [
                # RMSNorm (no bias in Qwen2.5-VL vision)
                ("norm1.weight",         "norm1.weight"),
                ("norm2.weight",         "norm2.weight"),
                # Fused QKV attention
                ("attn.qkv.weight",      "attn_qkv.weight"),
                ("attn.qkv.bias",        "attn_qkv.bias"),
                ("attn.proj.weight",     "attn_proj.weight"),
                ("attn.proj.bias",       "attn_proj.bias"),
                # SwiGLU MLP (gate + up + down)
                ("mlp.gate_proj.weight", "ffn_gate.weight"),
                ("mlp.gate_proj.bias",   "ffn_gate.bias"),
                ("mlp.up_proj.weight",   "ffn_up.weight"),
                ("mlp.up_proj.bias",     "ffn_up.bias"),
                ("mlp.down_proj.weight", "ffn_down.weight"),
                ("mlp.down_proj.bias",   "ffn_down.bias"),
            ]:
                vw(q + gg_suf, p + hf_suf)

        # Merger: ln_q + mlp.0 + mlp.2
        for hf_suf, gg_suf in [
            ("ln_q.weight",    "norm.weight"),
            ("ln_q.bias",      "norm.bias"),
            ("mlp.0.weight",   "fc1.weight"),
            ("mlp.0.bias",     "fc1.bias"),
            ("mlp.2.weight",   "fc2.weight"),
            ("mlp.2.bias",     "fc2.bias"),
        ]:
            vw(VPFX + "merger." + gg_suf, "visual.merger." + hf_suf)

        print(f"  Vision: {n_vis_layers} blocks + merger exported")

    # ── LLM decoder tensors ──────────────────────────────────────────

    if not args.vision_only:
        print("\nExporting LLM decoder...")
        LPFX = "l."  # compact GGUF prefix

        def lw(gguf_name, hf_key):
            """Write an LLM tensor."""
            if hf_key not in sd:
                return False
            t = get_tensor(hf_key)
            data = f32(t) if is_norm_or_bias(gguf_name) else wt(t)
            del t
            add_tensor(writer, gguf_name, data, wt)
            return True

        # Token embeddings
        lw(LPFX + "embed_tokens.weight", "model.embed_tokens.weight")

        # Decoder layers
        n_llm_layers = int(tc.num_hidden_layers)
        if args.max_llm_layers is not None:
            n_llm_layers = min(n_llm_layers, args.max_llm_layers)
            writer.add_uint32("qwen2vl.num_hidden_layers", n_llm_layers)
            print(f"  LLM: exporting first {n_llm_layers} of {tc.num_hidden_layers} layers")
        for i in range(n_llm_layers):
            p = f"model.layers.{i}."
            q = f"{LPFX}blk.{i}."

            for hf_suf, gg_suf in [
                # Norms (RMSNorm, no bias)
                ("input_layernorm.weight",          "attn_norm.weight"),
                ("post_attention_layernorm.weight",  "ffn_norm.weight"),
                # Self-attention (Q/K/V have bias, O has no bias)
                ("self_attn.q_proj.weight",          "attn_q.weight"),
                ("self_attn.q_proj.bias",            "attn_q.bias"),
                ("self_attn.k_proj.weight",          "attn_k.weight"),
                ("self_attn.k_proj.bias",            "attn_k.bias"),
                ("self_attn.v_proj.weight",          "attn_v.weight"),
                ("self_attn.v_proj.bias",            "attn_v.bias"),
                ("self_attn.o_proj.weight",          "attn_o.weight"),
                # MLP (no biases)
                ("mlp.gate_proj.weight",             "ffn_gate.weight"),
                ("mlp.up_proj.weight",               "ffn_up.weight"),
                ("mlp.down_proj.weight",             "ffn_down.weight"),
            ]:
                lw(q + gg_suf, p + hf_suf)

        # Final norm
        lw(LPFX + "output_norm.weight", "model.norm.weight")

        # LM head (may be tied to embed_tokens)
        if "lm_head.weight" in all_tensor_names:
            lw(LPFX + "lm_head.weight", "lm_head.weight")
        elif tc.tie_word_embeddings:
            writer.add_bool("qwen2vl.tie_word_embeddings", True)
            print("  lm_head: tied to embed_tokens")
        else:
            print("  WARNING: lm_head.weight not found and not tied!")

        print(f"  LLM: {n_llm_layers} layers exported")

    # ── Finalize ─────────────────────────────────────────────────────

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    import os
    fsize = os.path.getsize(args.output)
    print(f"\nWrote {args.output} ({fsize / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
