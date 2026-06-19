#!/usr/bin/env python3
"""Convert Qwen3-VL models (2B/7B) to CrispEmbed GGUF.

Exports:
  - Vision encoder (ViT with learned pos embed + RoPE, deepstack mergers)
  - LLM decoder (Qwen3 with interleaved mRoPE, QK RMSNorm)
  - Tokenizer (GPT-2 BPE)
  - Image preprocessor config

Architecture differences from Qwen2.5-VL (convert-qwen2vl-to-gguf.py):
  - HF tensor prefix: model.visual.* (not visual.*)
  - HF LLM prefix: model.language_model.* (not model.*)
  - Vision FFN: GELU fc1/fc2 (tensor: mlp.linear_fc1/linear_fc2)
  - Vision norms: LayerNorm with bias
  - Learned position embeddings (model.visual.pos_embed.weight)
  - DeepStack mergers at intermediate layers
  - LLM QK RMSNorm (self_attn.q_norm/k_norm)
  - Interleaved mRoPE
  - No attention bias in LLM

Usage:
    python models/convert-qwen3vl-to-gguf.py \\
        --model Qwen/Qwen3-VL-2B-Instruct \\
        --output /mnt/storage/gguf-models/qwen3-vl-2b.gguf

    python models/convert-qwen3vl-to-gguf.py \\
        --model Qwen/Qwen3-VL-2B-Instruct \\
        --dtype f16 \\
        --output /mnt/storage/gguf-models/qwen3-vl-2b-f16.gguf
"""

import argparse
import json
import sys
from pathlib import Path

import gguf
import numpy as np
import torch
from transformers import AutoConfig


ARCH = "qwen3vl"


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
        "norm", "bias", "embed_tokens", "lm_head", "pos_embed",
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
                        help="HF model ID (e.g. Qwen/Qwen3-VL-2B-Instruct)")
    parser.add_argument("--output", required=True,
                        help="Output GGUF path")
    parser.add_argument("--dtype", choices=["f16", "f32", "q8_0"], default="f32",
                        help="Weight dtype (f32 default, f16 for smaller, q8_0 for quantized)")
    parser.add_argument("--vision-only", action="store_true",
                        help="Export only vision encoder")
    parser.add_argument("--llm-only", action="store_true",
                        help="Export only LLM decoder")
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

    from safetensors import safe_open

    model_path = Path(args.model)
    is_local = model_path.is_dir()

    # Load config.json directly
    if is_local:
        config_json_path = model_path / "config.json"
    else:
        from huggingface_hub import hf_hub_download
        config_json_path = Path(hf_hub_download(args.model, "config.json"))
    with open(config_json_path) as f:
        raw_config = json.load(f)

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)

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
        tensor_to_shard = None

    shard_paths = {}
    all_tensor_names = set()
    _tsmap = {}
    for shard in shard_files:
        path = resolve_file(shard)
        shard_paths[shard] = path
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                all_tensor_names.add(key)
                if tensor_to_shard is None:
                    _tsmap[key] = shard

    if tensor_to_shard is None:
        tensor_to_shard = _tsmap

    print(f"  {len(all_tensor_names)} tensors across {len(shard_files)} shards")

    def get_tensor(name):
        if name not in tensor_to_shard:
            return None
        shard = tensor_to_shard[name]
        with safe_open(shard_paths[shard], framework="pt", device="cpu") as f:
            return f.get_tensor(name)

    sd = all_tensor_names

    # ── GGUF writer ──────────────────────────────────────────────────

    writer = gguf.GGUFWriter(str(args.output), ARCH)

    # ── Global metadata ──────────────────────────────────────────────

    model_name = args.model.split("/")[-1] if "/" in args.model else Path(args.model).name
    writer.add_string("general.name", model_name)

    rc = raw_config
    tc = rc.get("text_config", rc)

    writer.add_uint32(f"{ARCH}.vocab_size", int(tc.get("vocab_size", 151936)))
    writer.add_uint32(f"{ARCH}.hidden_size", int(tc.get("hidden_size", 2048)))
    writer.add_uint32(f"{ARCH}.intermediate_size", int(tc.get("intermediate_size", 6144)))
    writer.add_uint32(f"{ARCH}.num_hidden_layers", int(tc.get("num_hidden_layers", 28)))
    writer.add_uint32(f"{ARCH}.num_attention_heads", int(tc.get("num_attention_heads", 16)))
    writer.add_uint32(f"{ARCH}.num_key_value_heads", int(tc.get("num_key_value_heads", 8)))
    writer.add_uint32(f"{ARCH}.max_position_embeddings", int(tc.get("max_position_embeddings", 262144)))
    writer.add_float32(f"{ARCH}.rms_norm_eps", float(tc.get("rms_norm_eps", 1e-6)))
    writer.add_float32(f"{ARCH}.rope_theta", float(tc.get("rope_theta", 5000000.0)))
    tie_embeddings = bool(tc.get("tie_word_embeddings", True))
    writer.add_bool(f"{ARCH}.tie_word_embeddings", tie_embeddings)

    # mRoPE sections + interleaved flag
    rope_scaling = tc.get("rope_scaling", {})
    if "mrope_section" in rope_scaling:
        sections = rope_scaling["mrope_section"]
        writer.add_array(f"{ARCH}.rope_sections", [int(x) for x in sections])
        print(f"  mRoPE sections: {sections}")
    mrope_interleaved = rope_scaling.get("mrope_interleaved", True)
    writer.add_bool(f"{ARCH}.mrope_interleaved", mrope_interleaved)
    print(f"  mRoPE interleaved: {mrope_interleaved}")

    # QK RMSNorm flag
    writer.add_bool(f"{ARCH}.has_qk_norm", True)

    # Vision config
    vc_json = rc.get("vision_config", {})
    vc_depth = int(vc_json.get("depth", 24))
    vc_hidden = int(vc_json.get("hidden_size", 1024))
    vc_inter = int(vc_json.get("intermediate_size", 4096))
    vc_heads = int(vc_json.get("num_heads", 16))
    vc_patch = int(vc_json.get("patch_size", 16))
    vc_merge = int(vc_json.get("spatial_merge_size", 2))
    vc_temporal = int(vc_json.get("temporal_patch_size", 2))
    vc_out = int(vc_json.get("out_hidden_size", 2048))
    vc_in_ch = int(vc_json.get("in_channels", 3))
    vc_num_pos = int(vc_json.get("num_position_embeddings", 2304))
    vc_deepstack = vc_json.get("deepstack_visual_indexes", [5, 11, 17])

    writer.add_uint32(f"{ARCH}.vision.depth", vc_depth)
    writer.add_uint32(f"{ARCH}.vision.hidden_size", vc_hidden)
    writer.add_uint32(f"{ARCH}.vision.intermediate_size", vc_inter)
    writer.add_uint32(f"{ARCH}.vision.num_heads", vc_heads)
    writer.add_uint32(f"{ARCH}.vision.in_channels", vc_in_ch)
    writer.add_uint32(f"{ARCH}.vision.patch_size", vc_patch)
    writer.add_uint32(f"{ARCH}.vision.spatial_patch_size", vc_patch)
    writer.add_uint32(f"{ARCH}.vision.spatial_merge_size", vc_merge)
    writer.add_uint32(f"{ARCH}.vision.temporal_patch_size", vc_temporal)
    writer.add_uint32(f"{ARCH}.vision.out_hidden_size", vc_out)
    writer.add_uint32(f"{ARCH}.vision.num_position_embeddings", vc_num_pos)
    writer.add_array(f"{ARCH}.vision.deepstack_indexes",
                     [int(x) for x in vc_deepstack])

    print(f"  LLM: {tc.get('num_hidden_layers')}L, {tc.get('hidden_size')}d, "
          f"{tc.get('num_attention_heads')}H/{tc.get('num_key_value_heads')}KV, "
          f"inter={tc.get('intermediate_size')}")
    print(f"  Vision: {vc_depth}L, {vc_hidden}d, {vc_heads}H, "
          f"patch={vc_patch}, merge={vc_merge}, out={vc_out}")
    print(f"  DeepStack indexes: {vc_deepstack}")

    # Image preprocessor config
    try:
        pp_path = resolve_file("preprocessor_config.json")
        with open(pp_path) as fp:
            pp_cfg = json.load(fp)
        pp_mean = list(pp_cfg.get("image_mean", [0.5, 0.5, 0.5]))[:3]
        pp_std = list(pp_cfg.get("image_std", [0.5, 0.5, 0.5]))[:3]
        pp_size = pp_cfg.get("size", {})
        pp_min = int(pp_size.get("min_pixels", pp_size.get("shortest_edge", 65536)))
        pp_max = int(pp_size.get("max_pixels", pp_size.get("longest_edge", 16777216)))
        writer.add_array(f"{ARCH}.vision.image_mean", [float(x) for x in pp_mean])
        writer.add_array(f"{ARCH}.vision.image_std", [float(x) for x in pp_std])
        writer.add_uint32(f"{ARCH}.vision.min_pixels", pp_min)
        writer.add_uint32(f"{ARCH}.vision.max_pixels", pp_max)
        print(f"  Image: mean={pp_mean}, std={pp_std}, "
              f"min_px={pp_min}, max_px={pp_max}")
    except Exception as e:
        print(f"  preprocessor_config.json unavailable ({e}); using defaults")

    # ── Tokenizer ────────────────────────────────────────────────────

    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        vocab = tok.get_vocab()
        n_vocab = len(vocab)
        tokens = [""] * n_vocab
        for token_str, token_id in vocab.items():
            if token_id < n_vocab:
                tokens[token_id] = token_str

        writer.add_array("tokenizer.ggml.tokens", tokens)
        writer.add_string("tokenizer.ggml.model", "gpt2")
        writer.add_uint32("tokenizer.ggml.type", 1)

        try:
            merges_file = resolve_file("merges.txt")
            with open(merges_file) as mf:
                raw_merges = [l.strip() for l in mf if l.strip() and not l.startswith("#")]
            writer.add_array("tokenizer.ggml.merges", raw_merges)
            print(f"  Merges: {len(raw_merges)}")
        except Exception:
            try:
                tok_json_file = resolve_file("tokenizer.json")
                with open(tok_json_file, "r", encoding="utf-8") as jf:
                    tok_json = json.load(jf)
                raw_merges = []
                for merge in tok_json.get("model", {}).get("merges", []):
                    if isinstance(merge, str):
                        raw_merges.append(merge)
                    elif isinstance(merge, list) and len(merge) == 2:
                        raw_merges.append(f"{merge[0]} {merge[1]}")
                if raw_merges:
                    writer.add_array("tokenizer.ggml.merges", raw_merges)
                    print(f"  Merges: {len(raw_merges)} (from tokenizer.json)")
            except Exception as e2:
                print(f"  Merges not found: {e2}")

        eos_id = getattr(tok, "eos_token_id", None)
        if eos_id is not None:
            writer.add_uint32("tokenizer.ggml.eos_token_id", int(eos_id))
        pad_id = getattr(tok, "pad_token_id", None)
        if pad_id is not None:
            writer.add_uint32("tokenizer.ggml.padding_token_id", int(pad_id))

        # Vision special tokens
        for special_name in ["image_token_id", "video_token_id",
                             "vision_start_token_id", "vision_end_token_id"]:
            val = rc.get(special_name, None)
            if val is not None:
                writer.add_uint32(f"{ARCH}.{special_name}", int(val))
                print(f"  {special_name}: {val}")

        print(f"  Tokenizer: {n_vocab} tokens")
    except Exception as e:
        print(f"  Tokenizer export failed: {e}")

    # ── Vision encoder tensors ───────────────────────────────────────

    if not args.llm_only:
        print("\nExporting vision encoder...")
        VPFX = "v."

        def vw(gguf_name, hf_key):
            if hf_key not in sd:
                return False
            t = get_tensor(hf_key)
            data = f32(t) if is_norm_or_bias(gguf_name) else wt(t)
            del t
            add_tensor(writer, gguf_name, data, wt)
            return True

        # Patch embedding: Conv3D → flatten to 2D
        pe_key = "model.visual.patch_embed.proj.weight"
        if pe_key in sd:
            pe_w = get_tensor(pe_key)
            pe_w_flat = pe_w.reshape(pe_w.shape[0], -1).contiguous()
            data = wt(pe_w_flat)
            add_tensor(writer, VPFX + "patch_embed.weight", data, wt)
        vw(VPFX + "patch_embed.bias", "model.visual.patch_embed.proj.bias")

        # Learned position embeddings
        vw(VPFX + "pos_embed.weight", "model.visual.pos_embed.weight")

        # ViT blocks
        for i in range(vc_depth):
            p = f"model.visual.blocks.{i}."
            q = f"{VPFX}blk.{i}."

            for hf_suf, gg_suf in [
                # LayerNorm (with bias)
                ("norm1.weight",             "norm1.weight"),
                ("norm1.bias",               "norm1.bias"),
                ("norm2.weight",             "norm2.weight"),
                ("norm2.bias",               "norm2.bias"),
                # Fused QKV attention
                ("attn.qkv.weight",          "attn_qkv.weight"),
                ("attn.qkv.bias",            "attn_qkv.bias"),
                ("attn.proj.weight",         "attn_proj.weight"),
                ("attn.proj.bias",           "attn_proj.bias"),
                # GELU fc1/fc2 MLP (Qwen3-VL uses linear_fc1/linear_fc2)
                ("mlp.linear_fc1.weight",    "ffn_fc1.weight"),
                ("mlp.linear_fc1.bias",      "ffn_fc1.bias"),
                ("mlp.linear_fc2.weight",    "ffn_fc2.weight"),
                ("mlp.linear_fc2.bias",      "ffn_fc2.bias"),
            ]:
                vw(q + gg_suf, p + hf_suf)

        # Main merger
        for hf_suf, gg_suf in [
            ("norm.weight",          "norm.weight"),
            ("norm.bias",            "norm.bias"),
            ("linear_fc1.weight",    "fc1.weight"),
            ("linear_fc1.bias",      "fc1.bias"),
            ("linear_fc2.weight",    "fc2.weight"),
            ("linear_fc2.bias",      "fc2.bias"),
        ]:
            vw(VPFX + "merger." + gg_suf, "model.visual.merger." + hf_suf)

        # DeepStack mergers
        for ds_idx in range(len(vc_deepstack)):
            for hf_suf, gg_suf in [
                ("norm.weight",          "norm.weight"),
                ("norm.bias",            "norm.bias"),
                ("linear_fc1.weight",    "fc1.weight"),
                ("linear_fc1.bias",      "fc1.bias"),
                ("linear_fc2.weight",    "fc2.weight"),
                ("linear_fc2.bias",      "fc2.bias"),
            ]:
                hf_key = f"model.visual.deepstack_merger_list.{ds_idx}.{hf_suf}"
                gg_key = f"{VPFX}deepstack.{ds_idx}.{gg_suf}"
                vw(gg_key, hf_key)

        print(f"  Vision: {vc_depth} blocks + merger + {len(vc_deepstack)} deepstack mergers exported")

    # ── LLM decoder tensors ──────────────────────────────────────────

    if not args.vision_only:
        print("\nExporting LLM decoder...")
        LPFX = "l."

        def lw(gguf_name, hf_key):
            if hf_key not in sd:
                return False
            t = get_tensor(hf_key)
            data = f32(t) if is_norm_or_bias(gguf_name) else wt(t)
            del t
            add_tensor(writer, gguf_name, data, wt)
            return True

        # Token embeddings (note: model.language_model.embed_tokens.weight)
        lw(LPFX + "embed_tokens.weight", "model.language_model.embed_tokens.weight")

        n_llm_layers = int(tc.get("num_hidden_layers", 28))
        if args.max_llm_layers is not None:
            n_llm_layers = min(n_llm_layers, args.max_llm_layers)
            writer.add_uint32(f"{ARCH}.num_hidden_layers", n_llm_layers)
            print(f"  LLM: exporting first {n_llm_layers} of {tc.get('num_hidden_layers')} layers")

        for i in range(n_llm_layers):
            p = f"model.language_model.layers.{i}."
            q = f"{LPFX}blk.{i}."

            for hf_suf, gg_suf in [
                # Norms (RMSNorm, no bias)
                ("input_layernorm.weight",           "attn_norm.weight"),
                ("post_attention_layernorm.weight",   "ffn_norm.weight"),
                # Self-attention (no bias in Qwen3-VL)
                ("self_attn.q_proj.weight",           "attn_q.weight"),
                ("self_attn.k_proj.weight",           "attn_k.weight"),
                ("self_attn.v_proj.weight",           "attn_v.weight"),
                ("self_attn.o_proj.weight",           "attn_o.weight"),
                # QK RMSNorm
                ("self_attn.q_norm.weight",           "attn_q_norm.weight"),
                ("self_attn.k_norm.weight",           "attn_k_norm.weight"),
                # SwiGLU MLP (no biases)
                ("mlp.gate_proj.weight",              "ffn_gate.weight"),
                ("mlp.up_proj.weight",                "ffn_up.weight"),
                ("mlp.down_proj.weight",              "ffn_down.weight"),
            ]:
                lw(q + gg_suf, p + hf_suf)

        # Final norm
        lw(LPFX + "output_norm.weight", "model.language_model.norm.weight")

        # LM head (tied to embed_tokens for Qwen3-VL)
        if "model.language_model.lm_head.weight" in all_tensor_names:
            lw(LPFX + "lm_head.weight", "model.language_model.lm_head.weight")
        elif tie_embeddings:
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
