#!/usr/bin/env python3
"""Convert Unlimited-OCR (Baidu, 3B MoE) to GGUF.

Architecture:
  SAM-ViT-B (12 blocks, 768d) + CLIP-L/14 (24 layers, 1024d)
  → Fusion (concat 2048d) → Linear projector (2048→1280)
  → DeepSeek-V2 MoE decoder (12L, 1280d, 64 experts top-6 + 2 shared,
    layer 0 dense)

The SAM output is passed to CLIP as patch_embeds (replacing CLIP's conv
patch embeddings). Fusion concatenates CLIP output (skip CLS) with SAM
output flattened along feature dim → (256, 2048).

Usage:
    python models/convert-unlimited-ocr-to-gguf.py \
        --model-dir /mnt/storage/models/Unlimited-OCR \
        --output /mnt/storage/gguf-models/unlimited-ocr-f16.gguf --fp16
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import gguf
import numpy as np
import re


def map_tensor_name(n):
    """Map HuggingFace Unlimited-OCR tensor name to the GGUF short scheme.
    Returns None for tensors the engine does not consume."""

    # --- SAM ViT-B ---
    if n.startswith("model.sam_model."):
        s = n[len("model.sam_model."):]
        if s == "patch_embed.proj.weight": return "v.patch_embed.weight"
        if s == "patch_embed.proj.bias":   return "v.patch_embed.bias"
        if s == "pos_embed":               return "v.pos_embed"
        m = re.match(r"blocks\.(\d+)\.(.+)", s)
        if m:
            sub = {
                "norm1.weight":"ln1.weight","norm1.bias":"ln1.bias",
                "norm2.weight":"ln2.weight","norm2.bias":"ln2.bias",
                "attn.qkv.weight":"attn_qkv.weight","attn.qkv.bias":"attn_qkv.bias",
                "attn.proj.weight":"attn_proj.weight","attn.proj.bias":"attn_proj.bias",
                "attn.rel_pos_h":"attn_rel_pos_h","attn.rel_pos_w":"attn_rel_pos_w",
                "mlp.lin1.weight":"ffn_up.weight","mlp.lin1.bias":"ffn_up.bias",
                "mlp.lin2.weight":"ffn_down.weight","mlp.lin2.bias":"ffn_down.bias",
            }.get(m.group(2))
            if sub: return f"v.blk.{m.group(1)}.{sub}"
        return {
            "neck.0.weight":"v.neck_conv1.weight",
            "neck.1.weight":"v.neck_ln1.weight","neck.1.bias":"v.neck_ln1.bias",
            "neck.2.weight":"v.neck_conv2.weight",
            "neck.3.weight":"v.neck_ln2.weight","neck.3.bias":"v.neck_ln2.bias",
            "net_2.weight":"v.net_2.weight",
            "net_3.weight":"v.net_3.weight",
        }.get(s)

    # --- CLIP-L/14 ---
    if n.startswith("model.vision_model."):
        s = n[len("model.vision_model."):]
        direct = {
            "embeddings.class_embedding": "c.cls_token",
            "embeddings.patch_embedding.weight": "c.patch_embed.weight",
            "embeddings.position_embedding.weight": "c.pos_embed",
            "pre_layrnorm.weight": "c.pre_ln.weight",
            "pre_layrnorm.bias": "c.pre_ln.bias",
        }
        if s in direct:
            return direct[s]
        m = re.match(r"transformer\.layers\.(\d+)\.(.+)", s)
        if m:
            sub = {
                "layer_norm1.weight":"ln1.weight","layer_norm1.bias":"ln1.bias",
                "layer_norm2.weight":"ln2.weight","layer_norm2.bias":"ln2.bias",
                "self_attn.qkv_proj.weight":"attn_qkv.weight",
                "self_attn.qkv_proj.bias":"attn_qkv.bias",
                "self_attn.out_proj.weight":"attn_proj.weight",
                "self_attn.out_proj.bias":"attn_proj.bias",
                "mlp.fc1.weight":"ffn_up.weight","mlp.fc1.bias":"ffn_up.bias",
                "mlp.fc2.weight":"ffn_down.weight","mlp.fc2.bias":"ffn_down.bias",
            }.get(m.group(2))
            if sub: return f"c.blk.{m.group(1)}.{sub}"
        return None

    # --- Projector ---
    if n == "model.projector.layers.weight": return "proj.weight"
    if n == "model.projector.layers.bias":   return "proj.bias"

    # --- Learned tokens ---
    if n == "model.view_seperator":  return "v.view_separator"
    if n == "model.image_newline":   return "v.image_newline"

    # --- LLM Decoder ---
    if n == "model.embed_tokens.weight": return "l.embed_tokens.weight"
    if n == "lm_head.weight":            return "l.lm_head.weight"
    if n == "model.norm.weight":         return "l.output_norm.weight"

    m = re.match(r"model\.layers\.(\d+)\.(.+)", n)
    if m:
        i, r = m.group(1), m.group(2)
        direct = {
            "input_layernorm.weight":"input_layernorm.weight",
            "post_attention_layernorm.weight":"post_attention_layernorm.weight",
            "self_attn.q_proj.weight":"attn_q.weight",
            "self_attn.k_proj.weight":"attn_k.weight",
            "self_attn.v_proj.weight":"attn_v.weight",
            "self_attn.o_proj.weight":"attn_o.weight",
            # Dense layer 0
            "mlp.gate_proj.weight":"ffn_gate.weight",
            "mlp.up_proj.weight":"ffn_up.weight",
            "mlp.down_proj.weight":"ffn_down.weight",
            # MoE router
            "mlp.gate.weight":"mlp_gate.weight",
            # Shared experts
            "mlp.shared_experts.gate_proj.weight":"shared_exp.ffn_gate.weight",
            "mlp.shared_experts.up_proj.weight":"shared_exp.ffn_up.weight",
            "mlp.shared_experts.down_proj.weight":"shared_exp.ffn_down.weight",
        }.get(r)
        if direct: return f"l.blk.{i}.{direct}"
        # Routed experts
        e = re.match(r"mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight", r)
        if e: return f"l.blk.{i}.exp.{e.group(1)}.ffn_{e.group(2)}.weight"
    return None


def main():
    p = argparse.ArgumentParser(description="Convert Unlimited-OCR to GGUF")
    p.add_argument("--model-dir", required=True,
                    help="Path to HF model directory (with safetensors + config)")
    p.add_argument("--output", required=True, help="Output GGUF path")
    p.add_argument("--fp16", action="store_true", help="Store as FP16")
    args = p.parse_args()

    model_dir = Path(args.model_dir)

    # Load config
    cfg = json.load(open(model_dir / "config.json"))
    lang_cfg = cfg.get("language_config", cfg)
    vis_cfg = cfg.get("vision_config", {})
    proj_cfg = cfg.get("projector_config", {})

    # Architecture params
    hidden_size = lang_cfg["hidden_size"]           # 1280
    n_layers = lang_cfg["num_hidden_layers"]        # 12
    n_heads = lang_cfg["num_attention_heads"]       # 10
    n_kv_heads = lang_cfg["num_key_value_heads"]    # 10
    intermediate = lang_cfg["intermediate_size"]    # 6848
    vocab_size = lang_cfg["vocab_size"]             # 129280
    n_experts = lang_cfg.get("n_routed_experts", 64)
    n_experts_per_tok = lang_cfg.get("num_experts_per_tok", 6)
    n_shared_experts = lang_cfg.get("n_shared_experts", 2)
    moe_intermediate = lang_cfg.get("moe_intermediate_size", 896)
    first_k_dense = lang_cfg.get("first_k_dense_replace", 1)
    sliding_window = lang_cfg.get("sliding_window_size",
                                  lang_cfg.get("sliding_window", 128))

    # SAM ViT-B
    sam_cfg = vis_cfg.get("width", {}).get("sam_vit_b", {})
    sam_width = sam_cfg.get("width", 768)
    sam_layers = sam_cfg.get("layers", 12)
    sam_heads = sam_cfg.get("heads", 12)
    sam_global_attn = sam_cfg.get("global_attn_indexes", [2, 5, 8, 11])
    sam_image_size = vis_cfg.get("image_size", 1024)
    sam_ds_channels = sam_cfg.get("downsample_channels", [512, 1024])

    # CLIP-L/14
    clip_cfg = vis_cfg.get("width", {}).get("clip-l-14-224", {})
    clip_width = clip_cfg.get("width", 1024)
    clip_layers = clip_cfg.get("layers", 24)
    clip_heads = clip_cfg.get("heads", 16)
    clip_image_size = clip_cfg.get("image_size", 224)
    clip_patch_size = clip_cfg.get("patch_size", 14)

    # Projector
    proj_input_dim = proj_cfg.get("input_dim", 2048)
    proj_output_dim = proj_cfg.get("n_embed", 1280)

    print(f"Unlimited-OCR architecture:")
    print(f"  LLM: {n_layers}L, {hidden_size}d, {n_heads}H, vocab={vocab_size}")
    print(f"  MoE: {n_experts} experts, top-{n_experts_per_tok}, "
          f"{n_shared_experts} shared, dense layers: 0..{first_k_dense-1}")
    print(f"  SAM: {sam_layers}L, {sam_width}d, {sam_heads}H, "
          f"image={sam_image_size}")
    print(f"  CLIP: {clip_layers}L, {clip_width}d, {clip_heads}H, "
          f"image={clip_image_size}, patch={clip_patch_size}")
    print(f"  Projector: {proj_input_dim}→{proj_output_dim}")
    print(f"  Sliding window: {sliding_window}")

    # Resolve safetensors files
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.load(open(index_path))
        weight_map = index["weight_map"]
        st_files = sorted(set(weight_map.values()))
    else:
        # Single file
        st_files = ["model.safetensors"]
        weight_map = None

    # Parse all safetensors headers
    all_tensors = {}  # name -> (file, info, data_offset)
    for st_file in st_files:
        st_path = model_dir / st_file
        with open(st_path, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header_json = json.loads(f.read(header_size))
        header_json.pop("__metadata__", None)
        data_offset = 8 + header_size
        for name, info in header_json.items():
            all_tensors[name] = (str(st_path), info, data_offset)

    tensor_names = sorted(all_tensors.keys())
    print(f"\nParsed {len(tensor_names)} tensors from {len(st_files)} file(s)")

    # Load tokenizer
    tok_path = model_dir / "tokenizer.json"
    tok_data = json.load(open(tok_path))
    vocab = tok_data.get("model", {}).get("vocab", {})
    merges = tok_data.get("model", {}).get("merges", [])
    print(f"Tokenizer: {len(vocab)} tokens, {len(merges)} merges")

    # Write GGUF
    writer = gguf.GGUFWriter(str(args.output), arch="unlimited_ocr",
                              use_temp_file=True)

    writer.add_string("general.name", "unlimited-ocr")
    writer.add_string("general.license", "MIT")
    writer.add_string("general.source",
                       "https://huggingface.co/baidu/Unlimited-OCR")

    # LLM hyperparams
    writer.add_uint32("unlimited_ocr.hidden_size", hidden_size)
    writer.add_uint32("unlimited_ocr.num_hidden_layers", n_layers)
    writer.add_uint32("unlimited_ocr.num_attention_heads", n_heads)
    writer.add_uint32("unlimited_ocr.num_key_value_heads", n_kv_heads)
    writer.add_uint32("unlimited_ocr.intermediate_size", intermediate)
    writer.add_uint32("unlimited_ocr.vocab_size", vocab_size)
    writer.add_uint32("unlimited_ocr.n_routed_experts", n_experts)
    writer.add_uint32("unlimited_ocr.num_experts_per_tok", n_experts_per_tok)
    writer.add_uint32("unlimited_ocr.n_shared_experts", n_shared_experts)
    writer.add_uint32("unlimited_ocr.moe_intermediate_size", moe_intermediate)
    writer.add_uint32("unlimited_ocr.first_k_dense_replace", first_k_dense)
    writer.add_uint32("unlimited_ocr.sliding_window", sliding_window)
    writer.add_uint32("unlimited_ocr.max_position_embeddings",
                      lang_cfg.get("max_position_embeddings", 32768))

    # SAM hyperparams
    writer.add_uint32("unlimited_ocr.sam.width", sam_width)
    writer.add_uint32("unlimited_ocr.sam.layers", sam_layers)
    writer.add_uint32("unlimited_ocr.sam.heads", sam_heads)
    writer.add_uint32("unlimited_ocr.sam.image_size", sam_image_size)
    writer.add_uint32("unlimited_ocr.sam.patch_size", 16)
    writer.add_uint32("unlimited_ocr.sam.window_size", 14)
    writer.add_array("unlimited_ocr.sam.global_attn_indexes", sam_global_attn)
    writer.add_array("unlimited_ocr.sam.downsample_channels", sam_ds_channels)

    # CLIP hyperparams
    writer.add_uint32("unlimited_ocr.clip.width", clip_width)
    writer.add_uint32("unlimited_ocr.clip.layers", clip_layers)
    writer.add_uint32("unlimited_ocr.clip.heads", clip_heads)
    writer.add_uint32("unlimited_ocr.clip.image_size", clip_image_size)
    writer.add_uint32("unlimited_ocr.clip.patch_size", clip_patch_size)

    # Projector
    writer.add_uint32("unlimited_ocr.projector.input_dim", proj_input_dim)
    writer.add_uint32("unlimited_ocr.projector.output_dim", proj_output_dim)

    # Tokenizer
    tokens = [""] * len(vocab)
    for tok, idx in vocab.items():
        if idx < len(tokens):
            tokens[idx] = tok
    writer.add_array("tokenizer.ggml.tokens", tokens)
    if merges:
        merges_str = []
        for mg in merges:
            if isinstance(mg, (list, tuple)) and len(mg) == 2:
                merges_str.append(f"{mg[0]} {mg[1]}")
            elif isinstance(mg, str):
                merges_str.append(mg)
        writer.add_array("tokenizer.ggml.merges", merges_str)

    eos_id = lang_cfg.get("eos_token_id", 1)
    bos_id = lang_cfg.get("bos_token_id", 0)
    writer.add_uint32("unlimited_ocr.tokenizer.eos_id", eos_id)
    writer.add_uint32("unlimited_ocr.tokenizer.bos_id", bos_id)
    writer.add_uint32("unlimited_ocr.tokenizer.image_token_id", 128815)

    # Write tensors
    dtype_np = np.float16 if args.fp16 else np.float32
    dtype_gguf = (gguf.GGMLQuantizationType.F16 if args.fp16
                  else gguf.GGMLQuantizationType.F32)

    total_params = 0
    tensor_count = 0
    skipped = 0

    def read_tensor(fpath, info, base_offset):
        """Read a single tensor from safetensors, convert bf16→f32."""
        offsets = info["data_offsets"]
        shape = info["shape"]
        dtype_str = info["dtype"]
        byte_start = base_offset + offsets[0]
        n_bytes = base_offset + offsets[1] - byte_start

        with open(fpath, "rb") as f:
            f.seek(byte_start)
            raw = f.read(n_bytes)

        if dtype_str == "BF16":
            u16 = np.frombuffer(raw, dtype=np.uint16)
            u32 = u16.astype(np.uint32) << 16
            arr = u32.view(np.float32).reshape(shape)
        elif dtype_str == "F16":
            arr = np.frombuffer(raw, dtype=np.float16).reshape(shape).astype(
                np.float32)
        elif dtype_str == "F32":
            arr = np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
        elif dtype_str == "I32":
            arr = np.frombuffer(raw, dtype=np.int32).reshape(shape).astype(
                np.float32)
        elif dtype_str == "I64":
            arr = np.frombuffer(raw, dtype=np.int64).reshape(shape).astype(
                np.float32)
        else:
            raise ValueError(f"Unsupported dtype: {dtype_str}")
        return arr

    for name in tensor_names:
        gguf_name = map_tensor_name(name)
        if gguf_name is None:
            skipped += 1
            continue

        fpath, info, data_offset = all_tensors[name]
        data = read_tensor(fpath, info, data_offset)

        # Flatten 4D conv/patch_embed weights to 2D
        if data.ndim == 4:
            data = data.reshape(data.shape[0], -1)

        data = data.astype(dtype_np)
        total_params += data.size
        writer.add_tensor(gguf_name, data, raw_dtype=dtype_gguf)
        tensor_count += 1

        if tensor_count % 100 == 0:
            print(f"  {tensor_count} tensors processed...", end="\r")

    print(f"\nWriting {tensor_count} tensors (skipped {skipped})...")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    out_size = Path(args.output).stat().st_size
    print(f"\nWrote {args.output}")
    print(f"  Tensors: {tensor_count}")
    print(f"  Parameters: {total_params:,}")
    print(f"  File size: {out_size:,} bytes ({out_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
