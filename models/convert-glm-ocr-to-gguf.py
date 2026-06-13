#!/usr/bin/env python3
"""Convert GLM-OCR (zai-org/GLM-OCR) to CrispEmbed GGUF.

Exports:
  - Vision encoder: CogViT (24L, 1024d, RMSNorm+SwiGLU, Q/K norm, Conv3D patches)
  - Spatial downsample: Conv2D [1536, 1024, 2, 2]
  - Merger: proj + SwiGLU + LayerNorm
  - LLM decoder: GLM-0.5B (16L, 1536d, GQA 16/8, post-norm, fused gate_up, mRoPE)
  - Tokenizer: GPT-2 BPE (vocab 59392)

Architecture notes:
  - Vision uses RMSNorm (not LayerNorm), SwiGLU (not GELU), with Q/K RMSNorm
  - LLM has 4 norms per layer (post-norm pattern with pre+post norms)
  - Q projects from 1536 to 2048 (head_dim=128, 16 heads > hidden/head_dim)
  - gate_up_proj is fused [2*inter, hidden] — split in C++ engine
  - mRoPE with sections [16,24,24] (same as Qwen2VL)

Usage:
    python models/convert-glm-ocr-to-gguf.py \\
        --model zai-org/GLM-OCR \\
        --output /mnt/storage/gguf-models/glm-ocr-f16.gguf \\
        --dtype f16
"""

import argparse
import json
import sys
from pathlib import Path

import gguf
import numpy as np

ARCH = "glm_ocr"


def f32(t):
    return t.detach().float().cpu().numpy().astype(np.float32)

def f16(t):
    return t.detach().float().cpu().numpy().astype(np.float16)

def is_norm_or_bias(name):
    """Tensors that must stay F32."""
    return any(k in name for k in [
        "norm", "bias", "embed_tokens", "lm_head",
        "class_embedding", "position_embedding",
        ".ls1", ".ls2",
    ])

def add_tensor(writer, name, data, wt_func):
    if isinstance(data, np.ndarray):
        if data.dtype == np.float16:
            writer.add_tensor(name, data, raw_dtype=gguf.GGMLQuantizationType.F16)
        else:
            writer.add_tensor(name, data, raw_dtype=gguf.GGMLQuantizationType.F32)
    else:
        raise ValueError(f"Unexpected tensor type: {type(data)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model ID or local path")
    parser.add_argument("--output", required=True, help="Output GGUF path")
    parser.add_argument("--dtype", choices=["f16", "f32"], default="f32")
    parser.add_argument("--vision-only", action="store_true")
    parser.add_argument("--llm-only", action="store_true")
    parser.add_argument("--max-vis-layers", type=int, default=None)
    parser.add_argument("--max-llm-layers", type=int, default=None)
    args = parser.parse_args()

    import torch
    from safetensors import safe_open

    wt = f16 if args.dtype == "f16" else f32

    # Resolve model
    model_path = Path(args.model)
    is_local = model_path.is_dir()

    if is_local:
        config_path = model_path / "config.json"
    else:
        from huggingface_hub import hf_hub_download
        config_path = Path(hf_hub_download(args.model, "config.json"))
    with open(config_path) as f:
        raw_config = json.load(f)

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
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                all_names.add(key)
                if key not in tensor_to_shard:
                    tensor_to_shard[key] = shard

    print(f"Model: {args.model} ({len(all_names)} tensors)")

    def get_tensor(name):
        if name not in tensor_to_shard: return None
        shard = tensor_to_shard[name]
        with safe_open(shard_paths[shard], framework="pt", device="cpu") as f:
            return f.get_tensor(name)

    sd = all_names

    # Parse config
    vc = raw_config.get("vision_config", {})
    tc = raw_config.get("text_config", {})

    vis_hidden = vc.get("hidden_size", 1024)
    vis_depth = vc.get("depth", 24)
    vis_heads = vc.get("num_heads", 16)
    vis_inter = vc.get("intermediate_size", 4096)
    vis_patch = vc.get("patch_size", 14)
    vis_image_size = vc.get("image_size", 336)
    vis_temporal_patch = vc.get("temporal_patch_size", 2)
    vis_spatial_merge = vc.get("spatial_merge_size", 2)
    vis_out_hidden = vc.get("out_hidden_size", 1536)
    vis_rms_eps = vc.get("rms_norm_eps", 1e-5)

    llm_hidden = tc.get("hidden_size", 1536)
    llm_layers = tc.get("num_hidden_layers", 16)
    llm_heads = tc.get("num_attention_heads", 16)
    llm_kv_heads = tc.get("num_key_value_heads", 8)
    llm_inter = tc.get("intermediate_size", 4608)
    llm_vocab = tc.get("vocab_size", 59392)
    llm_head_dim = tc.get("head_dim", 128)
    llm_rms_eps = tc.get("rms_norm_eps", 1e-5)
    llm_rope_theta = tc.get("rope_parameters", {}).get("rope_theta", 10000.0)
    llm_max_pos = tc.get("max_position_embeddings", 131072)
    llm_nextn = tc.get("num_nextn_predict_layers", 1)

    # mRoPE sections
    mrope_sections = tc.get("rope_parameters", {}).get("mrope_section", [16, 24, 24])

    n_vis_export = min(vis_depth, args.max_vis_layers or vis_depth)
    n_llm_export = min(llm_layers, args.max_llm_layers or llm_layers)

    print(f"  Vision: {vis_depth}L, {vis_hidden}d, {vis_heads}H, patch={vis_patch}, image={vis_image_size}")
    print(f"  LLM: {llm_layers}L, {llm_hidden}d, {llm_heads}H/{llm_kv_heads}KV, "
          f"inter={llm_inter}, head_dim={llm_head_dim}, vocab={llm_vocab}")
    print(f"  mRoPE sections: {mrope_sections}, theta={llm_rope_theta}")

    # GGUF writer
    writer = gguf.GGUFWriter(str(args.output), ARCH)
    model_name = args.model.split("/")[-1] if "/" in args.model else Path(args.model).name
    writer.add_string("general.name", model_name)

    # Vision metadata
    writer.add_uint32(f"{ARCH}.vision.depth", n_vis_export)
    writer.add_uint32(f"{ARCH}.vision.hidden_size", vis_hidden)
    writer.add_uint32(f"{ARCH}.vision.intermediate_size", vis_inter)
    writer.add_uint32(f"{ARCH}.vision.num_heads", vis_heads)
    writer.add_uint32(f"{ARCH}.vision.patch_size", vis_patch)
    writer.add_uint32(f"{ARCH}.vision.image_size", vis_image_size)
    writer.add_uint32(f"{ARCH}.vision.temporal_patch_size", vis_temporal_patch)
    writer.add_uint32(f"{ARCH}.vision.spatial_merge_size", vis_spatial_merge)
    writer.add_uint32(f"{ARCH}.vision.out_hidden_size", vis_out_hidden)
    writer.add_float32(f"{ARCH}.vision.rms_norm_eps", vis_rms_eps)

    # LLM metadata
    writer.add_uint32(f"{ARCH}.hidden_size", llm_hidden)
    writer.add_uint32(f"{ARCH}.num_hidden_layers", n_llm_export)
    writer.add_uint32(f"{ARCH}.num_attention_heads", llm_heads)
    writer.add_uint32(f"{ARCH}.num_key_value_heads", llm_kv_heads)
    writer.add_uint32(f"{ARCH}.intermediate_size", llm_inter)
    writer.add_uint32(f"{ARCH}.vocab_size", llm_vocab)
    writer.add_uint32(f"{ARCH}.head_dim", llm_head_dim)
    writer.add_uint32(f"{ARCH}.max_position_embeddings", llm_max_pos)
    writer.add_float32(f"{ARCH}.rms_norm_eps", llm_rms_eps)
    writer.add_float32(f"{ARCH}.rope_theta", llm_rope_theta)
    writer.add_array(f"{ARCH}.rope_sections", [int(x) for x in mrope_sections])

    # Special tokens
    writer.add_uint32(f"{ARCH}.image_token_id", raw_config.get("image_token_id", 59280))
    writer.add_uint32(f"{ARCH}.image_start_token_id", raw_config.get("image_start_token_id", 59256))
    writer.add_uint32(f"{ARCH}.image_end_token_id", raw_config.get("image_end_token_id", 59257))

    # Image preprocessor
    try:
        pp_path = resolve_file("preprocessor_config.json")
        with open(pp_path) as fp:
            pp = json.load(fp)
        pp_mean = list(pp.get("image_mean", [0.5, 0.5, 0.5]))[:3]
        pp_std = list(pp.get("image_std", [0.5, 0.5, 0.5]))[:3]
        writer.add_array(f"{ARCH}.vision.image_mean", [float(x) for x in pp_mean])
        writer.add_array(f"{ARCH}.vision.image_std", [float(x) for x in pp_std])
        print(f"  Image: mean={pp_mean}, std={pp_std}")
    except Exception as e:
        writer.add_array(f"{ARCH}.vision.image_mean", [0.5, 0.5, 0.5])
        writer.add_array(f"{ARCH}.vision.image_std", [0.5, 0.5, 0.5])
        print(f"  preprocessor: using defaults ({e})")

    # Tokenizer
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        vocab = tok.get_vocab()
        n_vocab = max(vocab.values()) + 1
        tokens_list = [''] * n_vocab
        for s, i in vocab.items():
            if 0 <= i < n_vocab:
                tokens_list[i] = s
        writer.add_token_list(tokens_list)
        writer.add_token_types([0] * n_vocab)
        writer.add_uint32(f"{ARCH}.tokenizer.vocab_size", n_vocab)
        if hasattr(tok, "eos_token_id") and tok.eos_token_id is not None:
            writer.add_uint32(f"{ARCH}.tokenizer.eos_id", int(tok.eos_token_id))
        print(f"  Tokenizer: {n_vocab} tokens")
    except Exception as e:
        print(f"  Tokenizer failed: {e}")

    # ── Vision tensors ───────────────────────────────────────────
    n_exported = 0
    VPFX = "v."

    if not args.llm_only:
        print("\nExporting vision encoder...")

        def vw(gguf_name, hf_key, force_f32=False):
            nonlocal n_exported
            if hf_key not in sd: return False
            t = get_tensor(hf_key)
            data = f32(t) if (force_f32 or is_norm_or_bias(gguf_name)) else wt(t)
            del t
            add_tensor(writer, gguf_name, data, wt)
            n_exported += 1
            return True

        # Patch embed: Conv3D [1024, 3, 2, 14, 14] → flatten to 2D
        pe_key = "model.visual.patch_embed.proj.weight"
        if pe_key in sd:
            pe_w = get_tensor(pe_key)
            pe_flat = pe_w.reshape(pe_w.shape[0], -1).contiguous()
            data = wt(pe_flat)
            add_tensor(writer, VPFX + "patch_embed.weight", data, wt)
            n_exported += 1
        vw(VPFX + "patch_embed.bias", "model.visual.patch_embed.proj.bias")

        # Vision blocks
        if n_vis_export < vis_depth:
            print(f"  Exporting first {n_vis_export} of {vis_depth} layers")

        for i in range(n_vis_export):
            p = f"model.visual.blocks.{i}."
            q = f"{VPFX}blk.{i}."
            for hf_suf, gg_suf in [
                ("norm1.weight",         "norm1.weight"),
                ("norm2.weight",         "norm2.weight"),
                ("attn.qkv.weight",      "attn_qkv.weight"),
                ("attn.qkv.bias",        "attn_qkv.bias"),
                ("attn.proj.weight",     "attn_proj.weight"),
                ("attn.proj.bias",       "attn_proj.bias"),
                ("attn.q_norm.weight",   "attn_q_norm.weight"),
                ("attn.k_norm.weight",   "attn_k_norm.weight"),
                ("mlp.gate_proj.weight", "ffn_gate.weight"),
                ("mlp.gate_proj.bias",   "ffn_gate.bias"),
                ("mlp.up_proj.weight",   "ffn_up.weight"),
                ("mlp.up_proj.bias",     "ffn_up.bias"),
                ("mlp.down_proj.weight", "ffn_down.weight"),
                ("mlp.down_proj.bias",   "ffn_down.bias"),
            ]:
                vw(q + gg_suf, p + hf_suf)

        # Post-layernorm
        vw(VPFX + "post_layernorm.weight", "model.visual.post_layernorm.weight")

        # Spatial downsample: Conv2D → flatten to 2D
        ds_key = "model.visual.downsample.weight"
        if ds_key in sd:
            ds_w = get_tensor(ds_key)
            ds_flat = ds_w.reshape(ds_w.shape[0], -1).contiguous()
            data = wt(ds_flat)
            add_tensor(writer, VPFX + "downsample.weight", data, wt)
            n_exported += 1
        vw(VPFX + "downsample.bias", "model.visual.downsample.bias")

        # Merger
        for hf_suf, gg_suf in [
            ("proj.weight",                  "proj.weight"),
            ("gate_proj.weight",             "gate.weight"),
            ("up_proj.weight",               "up.weight"),
            ("down_proj.weight",             "down.weight"),
            ("post_projection_norm.weight",  "norm.weight"),
            ("post_projection_norm.bias",    "norm.bias"),
        ]:
            vw(VPFX + "merger." + gg_suf, "model.visual.merger." + hf_suf)

        print(f"  Vision: {n_vis_export} layers + downsample + merger ({n_exported} tensors)")

    # ── LLM tensors ──────────────────────────────────────────────
    if not args.vision_only:
        print("\nExporting LLM decoder...")
        LPFX = "l."

        def lw(gguf_name, hf_key, force_f32=False):
            nonlocal n_exported
            if hf_key not in sd: return False
            t = get_tensor(hf_key)
            data = f32(t) if (force_f32 or is_norm_or_bias(gguf_name)) else wt(t)
            del t
            add_tensor(writer, gguf_name, data, wt)
            n_exported += 1
            return True

        # Embeddings
        lw(LPFX + "embed_tokens.weight",
           "model.language_model.embed_tokens.weight", force_f32=True)

        if n_llm_export < llm_layers:
            print(f"  Exporting first {n_llm_export} of {llm_layers} layers")

        for i in range(n_llm_export):
            p = f"model.language_model.layers.{i}."
            q = f"{LPFX}blk.{i}."

            # 4 norms (post-norm architecture)
            lw(q + "input_layernorm.weight", p + "input_layernorm.weight")
            lw(q + "post_self_attn_layernorm.weight", p + "post_self_attn_layernorm.weight")
            lw(q + "post_attention_layernorm.weight", p + "post_attention_layernorm.weight")
            lw(q + "post_mlp_layernorm.weight", p + "post_mlp_layernorm.weight")

            # Attention: separate Q/K/V, no bias
            lw(q + "attn_q.weight", p + "self_attn.q_proj.weight")
            lw(q + "attn_k.weight", p + "self_attn.k_proj.weight")
            lw(q + "attn_v.weight", p + "self_attn.v_proj.weight")
            lw(q + "attn_o.weight", p + "self_attn.o_proj.weight")

            # SwiGLU FFN: fused gate_up_proj → split into gate + up
            fused_key = p + "mlp.gate_up_proj.weight"
            if fused_key in sd:
                fused = get_tensor(fused_key)  # [2*inter, hidden]
                gate = fused[:llm_inter, :]    # [inter, hidden]
                up = fused[llm_inter:, :]      # [inter, hidden]
                data_gate = wt(gate)
                data_up = wt(up)
                del fused, gate, up
                add_tensor(writer, q + "ffn_gate.weight", data_gate, wt)
                add_tensor(writer, q + "ffn_up.weight", data_up, wt)
                n_exported += 2
            lw(q + "ffn_down.weight", p + "mlp.down_proj.weight")

        # Final norm + LM head
        lw(LPFX + "output_norm.weight", "model.language_model.norm.weight")
        lw(LPFX + "lm_head.weight", "lm_head.weight", force_f32=True)

        print(f"  LLM: {n_llm_export} layers exported")

    # Write
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
