#!/usr/bin/env python3
"""Convert GOT-OCR2 (stepfun-ai/GOT-OCR2_0) to CrispEmbed GGUF.

Exports:
  - Vision encoder: SAM ViT-B (12L, 768d, LayerNorm+GELU, window+global attn, decomposed RPE)
  - Neck: Conv(768→256,1x1) → LN2d → Conv(256→256,3x3) → LN2d
  - Downsample: Conv(256→512,3x3,s2) → Conv(512→1024,3x3,s2)
  - Projector: Linear(1024, 1024)
  - LLM decoder: Qwen2-0.5B (24L, 1024d, MHA 16/16, RoPE, SiLU SwiGLU)
  - Tokenizer: tiktoken (vocab 151860)

Architecture notes:
  - Vision uses LayerNorm (not RMSNorm), GELU (not SiLU), no Q/K norm
  - LLM is standard Qwen2 pre-norm: 2 norms/layer, separate Q/K/V with bias
  - MHA (16/16 heads, not GQA), standard RoPE (not mRoPE)
  - tie_word_embeddings=true (no separate lm_head)

Usage:
    python models/convert-got-ocr-to-gguf.py \\
        --model stepfun-ai/GOT-OCR2_0 \\
        --output /mnt/storage/gguf-models/got-ocr2-f16.gguf \\
        --dtype f16
"""

import argparse
import json
import sys
from pathlib import Path

import gguf
import numpy as np

ARCH = "got_ocr"


def f32(t):
    return t.detach().float().cpu().numpy().astype(np.float32)

def f16(t):
    return t.detach().float().cpu().numpy().astype(np.float16)

def is_norm_or_bias(name):
    """Tensors that must stay F32."""
    return any(k in name for k in [
        "norm", "bias", "embed_tokens", "lm_head",
        "pos_embed", "rel_pos",
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

    # Build tensor map (single safetensors file for GOT-OCR2)
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

    # Parse config — GOT-OCR2 has flat config (no vision_config/text_config nesting)
    # Vision params from got_vision_b.py (hardcoded in build_GOT_vit_b)
    vis_hidden = 768
    vis_depth = 12
    vis_heads = 12
    vis_head_dim = vis_hidden // vis_heads  # 64
    vis_mlp_ratio = 4
    vis_inter = vis_hidden * vis_mlp_ratio  # 3072
    vis_patch = 16
    vis_image_size = 1024
    vis_window_size = 14
    vis_global_attn_indexes = [2, 5, 8, 11]
    vis_neck_out = 256  # prompt_embed_dim

    # LLM params from config.json (Qwen2-0.5B)
    llm_hidden = raw_config.get("hidden_size", 1024)
    llm_layers = raw_config.get("num_hidden_layers", 24)
    llm_heads = raw_config.get("num_attention_heads", 16)
    llm_kv_heads = raw_config.get("num_key_value_heads", 16)
    llm_inter = raw_config.get("intermediate_size", 2816)
    llm_vocab = raw_config.get("vocab_size", 151860)
    llm_head_dim = llm_hidden // llm_heads  # 64
    llm_rms_eps = raw_config.get("rms_norm_eps", 1e-6)
    llm_rope_theta = raw_config.get("rope_theta", 1000000.0)
    llm_max_pos = raw_config.get("max_position_embeddings", 32768)
    tie_word_embeddings = raw_config.get("tie_word_embeddings", True)

    n_vis_export = min(vis_depth, args.max_vis_layers or vis_depth)
    n_llm_export = min(llm_layers, args.max_llm_layers or llm_layers)

    print(f"  Vision: {vis_depth}L, {vis_hidden}d, {vis_heads}H, patch={vis_patch}, "
          f"image={vis_image_size}, window={vis_window_size}")
    print(f"  LLM: {llm_layers}L, {llm_hidden}d, {llm_heads}H/{llm_kv_heads}KV, "
          f"inter={llm_inter}, head_dim={llm_head_dim}, vocab={llm_vocab}")
    print(f"  RoPE theta={llm_rope_theta}, tie_embeddings={tie_word_embeddings}")

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
    writer.add_uint32(f"{ARCH}.vision.window_size", vis_window_size)
    writer.add_uint32(f"{ARCH}.vision.neck_out_channels", vis_neck_out)
    writer.add_array(f"{ARCH}.vision.global_attn_indexes",
                     [int(x) for x in vis_global_attn_indexes])
    writer.add_array(f"{ARCH}.vision.image_mean",
                     [0.48145466, 0.4578275, 0.40821073])
    writer.add_array(f"{ARCH}.vision.image_std",
                     [0.26862954, 0.26130258, 0.27577711])

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

    # Special tokens
    writer.add_uint32(f"{ARCH}.image_token_id",
                      raw_config.get("im_patch_token", 151859))
    writer.add_uint32(f"{ARCH}.image_start_token_id",
                      raw_config.get("im_start_token", 151857))
    writer.add_uint32(f"{ARCH}.image_end_token_id",
                      raw_config.get("im_end_token", 151858))
    writer.add_uint32(f"{ARCH}.image_token_len",
                      raw_config.get("image_token_len", 256))

    # Tokenizer
    try:
        tok_path = resolve_file("qwen.tiktoken")
        import base64
        # Read tiktoken BPE file
        mergeable_ranks = {}
        with open(tok_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) == 2:
                    token_bytes = base64.b64decode(parts[0])
                    rank = int(parts[1])
                    mergeable_ranks[token_bytes] = rank

        # Build vocab list
        n_base = len(mergeable_ranks)
        # Special tokens (from tokenization_qwen.py)
        ENDOFTEXT = "<|endoftext|>"
        IMSTART = "<|im_start|>"
        IMEND = "<|im_end|>"
        extras = [f"<|extra_{i}|>" for i in range(205)]
        special_tokens = [ENDOFTEXT, IMSTART, IMEND] + extras

        # GOT-OCR2 adds image tokens: <img>, </img>, <imgpad>
        # These are at 151857, 151858, 151859 (after base + 208 special)
        n_vocab = llm_vocab
        tokens_list = [''] * n_vocab

        # Fill base tokens
        for token_bytes, rank in mergeable_ranks.items():
            if 0 <= rank < n_vocab:
                try:
                    tokens_list[rank] = token_bytes.decode('utf-8', errors='replace')
                except Exception:
                    tokens_list[rank] = repr(token_bytes)

        # Fill special tokens
        special_start = n_base
        for idx, st in enumerate(special_tokens):
            pos = special_start + idx
            if 0 <= pos < n_vocab:
                tokens_list[pos] = st

        # Image tokens
        tokens_list[151857] = "<img>"
        tokens_list[151858] = "</img>"
        tokens_list[151859] = "<imgpad>"

        writer.add_token_list(tokens_list)
        writer.add_token_types([0] * n_vocab)
        writer.add_uint32(f"{ARCH}.tokenizer.vocab_size", n_vocab)
        writer.add_uint32(f"{ARCH}.tokenizer.eos_id",
                          raw_config.get("eos_token_id", 151643))
        print(f"  Tokenizer: {n_vocab} tokens ({n_base} BPE + {len(special_tokens)} special + 3 image)")
    except Exception as e:
        print(f"  Tokenizer failed: {e}")
        import traceback; traceback.print_exc()

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

        # Patch embed: Conv2D [768, 3, 16, 16] → flatten to 2D [768, 768]
        pe_key = "model.vision_tower_high.patch_embed.proj.weight"
        if pe_key in sd:
            pe_w = get_tensor(pe_key)
            pe_flat = pe_w.reshape(pe_w.shape[0], -1).contiguous()
            data = wt(pe_flat)
            add_tensor(writer, VPFX + "patch_embed.weight", data, wt)
            n_exported += 1
        vw(VPFX + "patch_embed.bias", "model.vision_tower_high.patch_embed.proj.bias")

        # Absolute position embedding: [1, 64, 64, 768] → [4096, 768]
        pos_key = "model.vision_tower_high.pos_embed"
        if pos_key in sd:
            pos_t = get_tensor(pos_key)
            pos_flat = pos_t.reshape(-1, vis_hidden).contiguous()
            data = f32(pos_flat)  # always F32
            add_tensor(writer, VPFX + "pos_embed", data, wt)
            n_exported += 1

        # Vision blocks
        if n_vis_export < vis_depth:
            print(f"  Exporting first {n_vis_export} of {vis_depth} layers")

        for i in range(n_vis_export):
            p = f"model.vision_tower_high.blocks.{i}."
            q = f"{VPFX}blk.{i}."
            for hf_suf, gg_suf in [
                ("norm1.weight",          "ln1.weight"),
                ("norm1.bias",            "ln1.bias"),
                ("norm2.weight",          "ln2.weight"),
                ("norm2.bias",            "ln2.bias"),
                ("attn.qkv.weight",       "attn_qkv.weight"),
                ("attn.qkv.bias",         "attn_qkv.bias"),
                ("attn.proj.weight",      "attn_proj.weight"),
                ("attn.proj.bias",        "attn_proj.bias"),
                ("attn.rel_pos_h",        "attn_rel_pos_h"),
                ("attn.rel_pos_w",        "attn_rel_pos_w"),
                ("mlp.lin1.weight",       "ffn_up.weight"),
                ("mlp.lin1.bias",         "ffn_up.bias"),
                ("mlp.lin2.weight",       "ffn_down.weight"),
                ("mlp.lin2.bias",         "ffn_down.bias"),
            ]:
                vw(q + gg_suf, p + hf_suf)

        # Neck: 4 layers — conv1(1x1), ln2d_1, conv2(3x3), ln2d_2
        # Conv2D weights: flatten spatial dims
        neck_prefix = "model.vision_tower_high.neck."
        for idx, (hf_suf, gg_suf) in enumerate([
            ("0.weight", "neck_conv1.weight"),   # [256, 768, 1, 1]
            ("1.weight", "neck_ln1.weight"),      # [256]
            ("1.bias",   "neck_ln1.bias"),
            ("2.weight", "neck_conv2.weight"),   # [256, 256, 3, 3]
            ("3.weight", "neck_ln2.weight"),      # [256]
            ("3.bias",   "neck_ln2.bias"),
        ]):
            hf_key = neck_prefix + hf_suf
            gg_name = VPFX + gg_suf
            if hf_key in sd:
                t = get_tensor(hf_key)
                if t.dim() == 4:
                    t = t.reshape(t.shape[0], -1).contiguous()
                data = f32(t) if is_norm_or_bias(gg_name) else wt(t)
                add_tensor(writer, gg_name, data, wt)
                n_exported += 1

        # Downsample convolutions: net_2 and net_3
        for hf_key, gg_name in [
            ("model.vision_tower_high.net_2.weight", VPFX + "net_2.weight"),
            ("model.vision_tower_high.net_3.weight", VPFX + "net_3.weight"),
        ]:
            if hf_key in sd:
                t = get_tensor(hf_key)
                t_flat = t.reshape(t.shape[0], -1).contiguous()
                data = wt(t_flat)
                add_tensor(writer, gg_name, data, wt)
                n_exported += 1

        # Projector: mm_projector_vary
        vw(VPFX + "projector.weight", "model.mm_projector_vary.weight")
        vw(VPFX + "projector.bias", "model.mm_projector_vary.bias")

        print(f"  Vision: {n_vis_export} layers + neck + downsample + projector ({n_exported} tensors)")

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
           "model.embed_tokens.weight", force_f32=True)

        if n_llm_export < llm_layers:
            print(f"  Exporting first {n_llm_export} of {llm_layers} layers")

        for i in range(n_llm_export):
            p = f"model.layers.{i}."
            q = f"{LPFX}blk.{i}."

            # 2 norms (standard pre-norm)
            lw(q + "input_layernorm.weight", p + "input_layernorm.weight")
            lw(q + "post_attention_layernorm.weight", p + "post_attention_layernorm.weight")

            # Separate Q/K/V with bias
            lw(q + "attn_q.weight", p + "self_attn.q_proj.weight")
            lw(q + "attn_q.bias",  p + "self_attn.q_proj.bias")
            lw(q + "attn_k.weight", p + "self_attn.k_proj.weight")
            lw(q + "attn_k.bias",  p + "self_attn.k_proj.bias")
            lw(q + "attn_v.weight", p + "self_attn.v_proj.weight")
            lw(q + "attn_v.bias",  p + "self_attn.v_proj.bias")
            lw(q + "attn_o.weight", p + "self_attn.o_proj.weight")

            # SwiGLU FFN: separate gate/up/down
            lw(q + "ffn_gate.weight", p + "mlp.gate_proj.weight")
            lw(q + "ffn_up.weight",   p + "mlp.up_proj.weight")
            lw(q + "ffn_down.weight", p + "mlp.down_proj.weight")

        # Final norm (no separate lm_head — tied to embed_tokens)
        lw(LPFX + "output_norm.weight", "model.norm.weight")

        # lm_head — only if not tied
        if not tie_word_embeddings:
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
