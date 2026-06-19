#!/usr/bin/env python3
"""Convert PaddlePaddle/PaddleOCR-VL to CrispEmbed GGUF.

Architecture:
  NaViT-style ViT (27L, 1152d, SigLIP 2D RoPE + learned position embed)
  + Projector (pre_norm → 2×2 spatial merge → Linear → GELU → Linear)
  + ERNIE-4.5-0.3B LLM (18L, 1024d, 16/2 GQA, MRoPE, SwiGLU)

Outputs in the same qwen2vl GGUF format (arch="qwen2vl") so the existing
qwen2vl_ocr engine can load it with minimal changes.

Usage:
    python models/convert-paddleocr-vl-to-gguf.py \\
        --model PaddlePaddle/PaddleOCR-VL \\
        --output /mnt/storage/gguf-models/paddleocr-vl-0.9b-f16.gguf \\
        --dtype f16
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "ggml" / "scripts"))
try:
    import gguf
except ImportError:
    print("ERROR: gguf package not found. pip install gguf", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Convert PaddleOCR-VL to CrispEmbed GGUF")
    parser.add_argument("--model", required=True,
                        help="HF model ID or local directory")
    parser.add_argument("--output", required=True,
                        help="Output .gguf path")
    parser.add_argument("--dtype", default="f16", choices=["f32", "f16"],
                        help="Storage dtype (default: f16)")
    args = parser.parse_args()

    # Resolve model directory
    model_dir = args.model
    if not os.path.isdir(model_dir):
        from huggingface_hub import snapshot_download
        cache_dir = os.environ.get("HF_HUB_CACHE",
                                   os.path.expanduser("~/.cache/huggingface/hub"))
        model_dir = snapshot_download(args.model, cache_dir=cache_dir)
        print(f"Downloaded to: {model_dir}")

    # Load config
    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)

    vc = config.get("vision_config", {})

    # LLM params
    hidden_size = config["hidden_size"]                      # 1024
    n_layers = config["num_hidden_layers"]                   # 18
    n_heads = config["num_attention_heads"]                  # 16
    n_kv_heads = config["num_key_value_heads"]               # 2
    intermediate_size = config["intermediate_size"]          # 3072
    vocab_size = config["vocab_size"]                        # 103424
    rms_norm_eps = config.get("rms_norm_eps", 1e-5)
    rope_theta = config.get("rope_theta", 500000.0)
    tie_word_embeddings = config.get("tie_word_embeddings", False)
    max_pos = config.get("max_position_embeddings", 131072)

    # MRoPE sections
    rope_scaling = config.get("rope_scaling", {})
    mrope_section = rope_scaling.get("mrope_section", [16, 24, 24])

    # Vision params
    vis_hidden = vc.get("hidden_size", 1152)
    vis_inter = vc.get("intermediate_size", 4304)
    vis_depth = vc.get("num_hidden_layers", 27)
    vis_heads = vc.get("num_attention_heads", 16)
    vis_patch = vc.get("patch_size", 14)
    vis_merge = vc.get("spatial_merge_size", 2)
    vis_channels = vc.get("num_channels", 3)
    # PaddleOCR-VL config says temporal_patch_size=2 but preprocessor uses 1 for images.
    # The actual Conv2d weight shape is [1152, 3, 14, 14] (no temporal dim).
    # Override to 1 to match the real patch embedding.
    vis_temporal_patch = 1  # vc.get("temporal_patch_size", 2) — forced to 1 for images
    vis_image_size = vc.get("image_size", 384)
    vis_ln_eps = vc.get("layer_norm_eps", 1e-6)

    # Special tokens
    image_token_id = config.get("image_token_id", 100295)
    video_token_id = config.get("video_token_id", 101307)
    vision_start_token_id = config.get("vision_start_token_id", 101305)
    vision_end_token_id = config.get("vision_end_token_id", 101306)

    print(f"Model: {args.model}")
    print(f"  LLM: {n_layers}L, {hidden_size}d, {n_heads}/{n_kv_heads} heads, "
          f"inter={intermediate_size}, vocab={vocab_size}")
    print(f"  Vision: {vis_depth}L, {vis_hidden}d, {vis_heads} heads, "
          f"patch={vis_patch}, merge={vis_merge}")
    print(f"  MRoPE sections: {mrope_section}")

    # Image preprocessor config
    try:
        with open(os.path.join(model_dir, "preprocessor_config.json")) as f:
            pp = json.load(f)
        pp_mean = pp.get("image_mean", [0.5, 0.5, 0.5])[:3]
        pp_std = pp.get("image_std", [0.5, 0.5, 0.5])[:3]
        pp_min = pp.get("min_pixels", 147384)
        pp_max = pp.get("max_pixels", 2822400)
    except Exception:
        pp_mean, pp_std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        pp_min, pp_max = 147384, 2822400

    # Dtype
    if args.dtype == "f16":
        np_dtype = np.float16
        gguf_dtype = gguf.GGMLQuantizationType.F16
    else:
        np_dtype = np.float32
        gguf_dtype = gguf.GGMLQuantizationType.F32

    # ── GGUF writer ──
    ARCH = "qwen2vl"
    writer = gguf.GGUFWriter(args.output, ARCH)

    writer.add_string("general.architecture", ARCH)
    writer.add_string("general.name", "PaddleOCR-VL-0.9B")
    writer.add_string("general.source", args.model)

    # LLM hyperparameters (qwen2vl.* namespace)
    writer.add_uint32("qwen2vl.vocab_size", vocab_size)
    writer.add_uint32("qwen2vl.hidden_size", hidden_size)
    writer.add_uint32("qwen2vl.intermediate_size", intermediate_size)
    writer.add_uint32("qwen2vl.num_hidden_layers", n_layers)
    writer.add_uint32("qwen2vl.num_attention_heads", n_heads)
    writer.add_uint32("qwen2vl.num_key_value_heads", n_kv_heads)
    writer.add_uint32("qwen2vl.max_position_embeddings", max_pos)
    writer.add_float32("qwen2vl.rms_norm_eps", rms_norm_eps)
    writer.add_float32("qwen2vl.rope_theta", rope_theta)
    writer.add_bool("qwen2vl.tie_word_embeddings", tie_word_embeddings)
    writer.add_array("qwen2vl.rope_sections", [int(x) for x in mrope_section])

    # Vision hyperparameters
    writer.add_uint32("qwen2vl.vision.depth", vis_depth)
    writer.add_uint32("qwen2vl.vision.hidden_size", vis_hidden)
    writer.add_uint32("qwen2vl.vision.intermediate_size", vis_inter)
    writer.add_uint32("qwen2vl.vision.num_heads", vis_heads)
    writer.add_uint32("qwen2vl.vision.in_channels", vis_channels)
    writer.add_uint32("qwen2vl.vision.spatial_patch_size", vis_patch)
    writer.add_uint32("qwen2vl.vision.spatial_merge_size", vis_merge)
    writer.add_uint32("qwen2vl.vision.temporal_patch_size", vis_temporal_patch)
    writer.add_uint32("qwen2vl.vision.out_hidden_size", hidden_size)  # projector output = LLM dim

    # PaddleOCR-VL-specific flags
    writer.add_bool("qwen2vl.vision.has_position_embed", True)
    writer.add_string("qwen2vl.vision.hidden_act", "gelu_pytorch_tanh")
    writer.add_float32("qwen2vl.vision.layer_norm_eps", vis_ln_eps)

    # Image preprocessor
    writer.add_array("qwen2vl.vision.image_mean", [float(x) for x in pp_mean])
    writer.add_array("qwen2vl.vision.image_std", [float(x) for x in pp_std])
    writer.add_uint32("qwen2vl.vision.min_pixels", pp_min)
    writer.add_uint32("qwen2vl.vision.max_pixels", pp_max)

    # Special token IDs
    writer.add_uint32("qwen2vl.image_token_id", image_token_id)
    writer.add_uint32("qwen2vl.video_token_id", video_token_id)
    writer.add_uint32("qwen2vl.vision_start_token_id", vision_start_token_id)
    writer.add_uint32("qwen2vl.vision_end_token_id", vision_end_token_id)

    # ── Tokenizer ──
    tok_json_path = os.path.join(model_dir, "tokenizer.json")
    with open(tok_json_path, encoding="utf-8") as f:
        tok_json = json.load(f)

    model_data = tok_json.get("model", {})
    vocab_dict = model_data.get("vocab", {})
    merges = model_data.get("merges", [])

    # Build token list
    max_id = max(vocab_dict.values()) if vocab_dict else 0
    added_toks = tok_json.get("added_tokens", [])
    for at in added_toks:
        max_id = max(max_id, at["id"])
    tokens_list = [""] * (max_id + 1)
    for tok, tid in vocab_dict.items():
        tokens_list[tid] = tok
    for at in added_toks:
        tokens_list[at["id"]] = at["content"]

    writer.add_array("tokenizer.ggml.tokens", tokens_list)
    writer.add_string("tokenizer.ggml.model", "gpt2")  # BPE style
    writer.add_uint32("tokenizer.ggml.type", 1)         # 1 = BPE

    # Merges
    merge_strs = []
    for m in merges:
        if isinstance(m, list):
            merge_strs.append(" ".join(m))
        else:
            merge_strs.append(str(m))
    if merge_strs:
        writer.add_array("tokenizer.ggml.merges", merge_strs)

    writer.add_uint32("tokenizer.ggml.eos_token_id", 2)
    writer.add_uint32("tokenizer.ggml.padding_token_id", 0)
    writer.add_uint32("tokenizer.ggml.bos_token_id", 1)
    writer.add_uint32("qwen2vl.tokenizer.vocab_size", len(tokens_list))

    print(f"  Tokenizer: {len(tokens_list)} tokens, {len(merge_strs)} merges")

    # ── Load weights ──
    from safetensors import safe_open

    st_path = os.path.join(model_dir, "model.safetensors")
    state_dict = {}
    with safe_open(st_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    print(f"  Loaded {len(state_dict)} tensors from {st_path}")

    def to_np(t):
        return t.detach().float().cpu().numpy().astype(np_dtype)

    def to_f32(t):
        return t.detach().float().cpu().numpy().astype(np.float32)

    def add(name, t, force_f32=False):
        data = to_f32(t) if force_f32 else to_np(t)
        dt = gguf.GGMLQuantizationType.F32 if force_f32 else gguf_dtype
        writer.add_tensor(name, data, raw_dtype=dt)

    def is_norm_or_bias(name):
        return any(k in name for k in ["norm", "bias", "embed_tokens", "lm_head",
                                        "position_embed", "packing_pos"])

    VPFX = "v."
    LPFX = "l."
    n_written = 0

    # ── Vision embeddings ──
    # Patch embedding: Conv2d [1152, 3, 14, 14] → flatten to [1152, 588]
    pe_key = "visual.vision_model.embeddings.patch_embedding.weight"
    pe_w = state_dict[pe_key]
    pe_w_flat = pe_w.reshape(pe_w.shape[0], -1).contiguous()
    add(VPFX + "patch_embed.weight", pe_w_flat)
    n_written += 1

    pe_b_key = "visual.vision_model.embeddings.patch_embedding.bias"
    if pe_b_key in state_dict:
        add(VPFX + "patch_embed.bias", state_dict[pe_b_key], force_f32=True)
        n_written += 1

    # Learned position embeddings (NEW for PaddleOCR-VL)
    pos_key = "visual.vision_model.embeddings.position_embedding.weight"
    if pos_key in state_dict:
        add(VPFX + "position_embed.weight", state_dict[pos_key], force_f32=True)
        n_written += 1
        print(f"  Position embedding: {list(state_dict[pos_key].shape)}")

    pack_pos_key = "visual.vision_model.embeddings.packing_position_embedding.weight"
    if pack_pos_key in state_dict:
        add(VPFX + "packing_pos_embed.weight", state_dict[pack_pos_key], force_f32=True)
        n_written += 1
        print(f"  Packing position embedding: {list(state_dict[pack_pos_key].shape)}")

    # ── Vision encoder layers ──
    for i in range(vis_depth):
        p = f"visual.vision_model.encoder.layers.{i}."
        q = f"{VPFX}blk.{i}."

        mappings = [
            ("layer_norm1.weight",          "norm1.weight"),
            ("layer_norm1.bias",            "norm1.bias"),
            ("layer_norm2.weight",          "norm2.weight"),
            ("layer_norm2.bias",            "norm2.bias"),
            ("self_attn.q_proj.weight",     "attn_q.weight"),
            ("self_attn.q_proj.bias",       "attn_q.bias"),
            ("self_attn.k_proj.weight",     "attn_k.weight"),
            ("self_attn.k_proj.bias",       "attn_k.bias"),
            ("self_attn.v_proj.weight",     "attn_v.weight"),
            ("self_attn.v_proj.bias",       "attn_v.bias"),
            ("self_attn.out_proj.weight",   "attn_proj.weight"),
            ("self_attn.out_proj.bias",     "attn_proj.bias"),
            ("mlp.fc1.weight",              "ffn_fc1.weight"),
            ("mlp.fc1.bias",                "ffn_fc1.bias"),
            ("mlp.fc2.weight",              "ffn_fc2.weight"),
            ("mlp.fc2.bias",                "ffn_fc2.bias"),
        ]
        for hf_suf, gg_suf in mappings:
            hf_key = p + hf_suf
            if hf_key in state_dict:
                gg_name = q + gg_suf
                f32 = is_norm_or_bias(gg_name)
                add(gg_name, state_dict[hf_key], force_f32=f32)
                n_written += 1

    # Post-layernorm (applied after last encoder layer, before projector)
    for suf in ["weight", "bias"]:
        hf_key = f"visual.vision_model.post_layernorm.{suf}"
        if hf_key in state_dict:
            add(f"{VPFX}post_layernorm.{suf}", state_dict[hf_key], force_f32=True)
            n_written += 1

    # Skip pooler head (visual.vision_model.head.*) — not used in VLM inference

    print(f"  Vision: {vis_depth} layers + embeddings + post_layernorm = {n_written} tensors")

    # ── Projector (mlp_AR) ──
    proj_mappings = [
        ("mlp_AR.pre_norm.weight",   VPFX + "merger.norm.weight"),
        ("mlp_AR.pre_norm.bias",     VPFX + "merger.norm.bias"),
        ("mlp_AR.linear_1.weight",   VPFX + "merger.fc1.weight"),
        ("mlp_AR.linear_1.bias",     VPFX + "merger.fc1.bias"),
        ("mlp_AR.linear_2.weight",   VPFX + "merger.fc2.weight"),
        ("mlp_AR.linear_2.bias",     VPFX + "merger.fc2.bias"),
    ]
    n_proj = 0
    for hf_key, gg_name in proj_mappings:
        if hf_key in state_dict:
            f32 = is_norm_or_bias(gg_name)
            add(gg_name, state_dict[hf_key], force_f32=f32)
            n_written += 1
            n_proj += 1
    print(f"  Projector: {n_proj} tensors")

    # ── LLM decoder ──
    # Token embeddings
    add(LPFX + "embed_tokens.weight", state_dict["model.embed_tokens.weight"],
        force_f32=True)
    n_written += 1

    for i in range(n_layers):
        p = f"model.layers.{i}."
        q = f"{LPFX}blk.{i}."

        llm_mappings = [
            ("input_layernorm.weight",          "attn_norm.weight"),
            ("post_attention_layernorm.weight",  "ffn_norm.weight"),
            ("self_attn.q_proj.weight",          "attn_q.weight"),
            ("self_attn.k_proj.weight",          "attn_k.weight"),
            ("self_attn.v_proj.weight",          "attn_v.weight"),
            ("self_attn.o_proj.weight",          "attn_o.weight"),
            ("mlp.gate_proj.weight",             "ffn_gate.weight"),
            ("mlp.up_proj.weight",               "ffn_up.weight"),
            ("mlp.down_proj.weight",             "ffn_down.weight"),
        ]
        for hf_suf, gg_suf in llm_mappings:
            hf_key = p + hf_suf
            if hf_key in state_dict:
                gg_name = q + gg_suf
                f32 = is_norm_or_bias(gg_name)
                add(gg_name, state_dict[hf_key], force_f32=f32)
                n_written += 1

    # Final norm
    add(LPFX + "output_norm.weight", state_dict["model.norm.weight"], force_f32=True)
    n_written += 1

    # LM head
    if "lm_head.weight" in state_dict:
        add(LPFX + "lm_head.weight", state_dict["lm_head.weight"], force_f32=True)
        n_written += 1

    print(f"  LLM: {n_layers} layers + embed + norm + lm_head")
    print(f"\nTotal: {n_written} tensors written")

    # ── Finalize ──
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    out_size = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Output: {args.output} ({out_size:.1f} MB)")


if __name__ == "__main__":
    main()
