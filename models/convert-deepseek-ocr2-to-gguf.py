#!/usr/bin/env python3
"""Convert DeepSeek-OCR-2 (3B MoE) to GGUF.

Architecture:
  SAM-ViT-B (12 blocks, 768d) → Qwen2 encoder (24L, 896d, bidirectional)
  → Linear projector (896→1280) → DeepSeek-V2 MoE decoder (12L, 1280d,
    64 experts top-6 + 2 shared, layer 0 dense)

Usage:
    python models/convert-deepseek-ocr2-to-gguf.py \
        --model /mnt/storage/models/deepseek-ocr2.safetensors \
        --config /path/to/config.json \
        --tokenizer /path/to/tokenizer.json \
        --output /mnt/storage/gguf-models/deepseek-ocr2-f16.gguf --fp16
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import gguf
import numpy as np


def main():
    p = argparse.ArgumentParser(description="Convert DeepSeek-OCR-2 to GGUF")
    p.add_argument("--model", required=True, help="Path to .safetensors file")
    p.add_argument("--config", required=True, help="Path to config.json")
    p.add_argument("--tokenizer", required=True, help="Path to tokenizer.json")
    p.add_argument("--output", required=True, help="Output GGUF path")
    p.add_argument("--fp16", action="store_true", help="Store as FP16")
    args = p.parse_args()

    # Load config
    cfg = json.load(open(args.config))
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

    # SAM ViT-B
    sam_cfg = vis_cfg.get("width", {}).get("sam_vit_b", {})
    sam_width = sam_cfg.get("width", 768)
    sam_layers = sam_cfg.get("layers", 12)
    sam_heads = sam_cfg.get("heads", 12)
    sam_global_attn = sam_cfg.get("global_attn_indexes", [2, 5, 8, 11])
    sam_image_size = vis_cfg.get("image_size", 1024)

    # Qwen2 encoder
    qwen2_cfg = vis_cfg.get("width", {}).get("qwen2-0-5b", {})
    qwen2_dim = qwen2_cfg.get("dim", 896)

    # Projector
    proj_input_dim = proj_cfg.get("input_dim", 896)
    proj_output_dim = proj_cfg.get("n_embed", 1280)

    print(f"DeepSeek-OCR-2 architecture:")
    print(f"  LLM: {n_layers}L, {hidden_size}d, {n_heads}H, vocab={vocab_size}")
    print(f"  MoE: {n_experts} experts, top-{n_experts_per_tok}, {n_shared_experts} shared, dense layers: 0..{first_k_dense-1}")
    print(f"  SAM: {sam_layers}L, {sam_width}d, {sam_heads}H, image={sam_image_size}")
    print(f"  Qwen2 encoder: {qwen2_dim}d")
    print(f"  Projector: {proj_input_dim}→{proj_output_dim}")

    # Parse safetensors header to get tensor metadata without loading data
    import struct as _struct
    with open(args.model, "rb") as f:
        header_size = _struct.unpack("<Q", f.read(8))[0]
        header_json = json.loads(f.read(header_size))
    # Remove __metadata__ key
    header_json.pop("__metadata__", None)
    tensor_names = sorted(header_json.keys())
    data_offset = 8 + header_size
    print(f"\nParsed header: {len(tensor_names)} tensors, data starts at offset {data_offset}")

    # Load tokenizer
    tok_data = json.load(open(args.tokenizer))
    vocab = tok_data.get("model", {}).get("vocab", {})
    merges = tok_data.get("model", {}).get("merges", [])
    print(f"Tokenizer: {len(vocab)} tokens, {len(merges)} merges")

    # Write GGUF
    writer = gguf.GGUFWriter(str(args.output), arch="deepseek_ocr2", use_temp_file=True)

    writer.add_string("general.name", "deepseek-ocr2")
    writer.add_string("general.license", "Apache-2.0")
    writer.add_string("general.source", "https://huggingface.co/deepseek-ai/DeepSeek-OCR-2")

    # LLM hyperparams
    writer.add_uint32("deepseek_ocr2.hidden_size", hidden_size)
    writer.add_uint32("deepseek_ocr2.num_hidden_layers", n_layers)
    writer.add_uint32("deepseek_ocr2.num_attention_heads", n_heads)
    writer.add_uint32("deepseek_ocr2.num_key_value_heads", n_kv_heads)
    writer.add_uint32("deepseek_ocr2.intermediate_size", intermediate)
    writer.add_uint32("deepseek_ocr2.vocab_size", vocab_size)
    writer.add_uint32("deepseek_ocr2.n_routed_experts", n_experts)
    writer.add_uint32("deepseek_ocr2.num_experts_per_tok", n_experts_per_tok)
    writer.add_uint32("deepseek_ocr2.n_shared_experts", n_shared_experts)
    writer.add_uint32("deepseek_ocr2.moe_intermediate_size", moe_intermediate)
    writer.add_uint32("deepseek_ocr2.first_k_dense_replace", first_k_dense)
    writer.add_uint32("deepseek_ocr2.max_position_embeddings",
                      lang_cfg.get("max_position_embeddings", 8192))

    # SAM hyperparams
    writer.add_uint32("deepseek_ocr2.sam.width", sam_width)
    writer.add_uint32("deepseek_ocr2.sam.layers", sam_layers)
    writer.add_uint32("deepseek_ocr2.sam.heads", sam_heads)
    writer.add_uint32("deepseek_ocr2.sam.image_size", sam_image_size)
    writer.add_array("deepseek_ocr2.sam.global_attn_indexes", sam_global_attn)
    ds_channels = sam_cfg.get("downsample_channels", [512, 1024])
    writer.add_array("deepseek_ocr2.sam.downsample_channels", ds_channels)

    # Qwen2 encoder hyperparams
    writer.add_uint32("deepseek_ocr2.qwen2_enc.dim", qwen2_dim)
    # Count qwen2 encoder layers from weight names
    qwen2_layers = 0
    for name in tensor_names:
        if name.startswith("model.qwen2_model.model.model.layers."):
            parts = name.split(".")
            layer_idx = int(parts[5])
            qwen2_layers = max(qwen2_layers, layer_idx + 1)
    writer.add_uint32("deepseek_ocr2.qwen2_enc.layers", qwen2_layers)
    print(f"  Qwen2 encoder: {qwen2_layers} layers")

    # Projector
    writer.add_uint32("deepseek_ocr2.projector.input_dim", proj_input_dim)
    writer.add_uint32("deepseek_ocr2.projector.output_dim", proj_output_dim)

    # Tokenizer
    tokens = [""] * len(vocab)
    for tok, idx in vocab.items():
        if idx < len(tokens):
            tokens[idx] = tok
    writer.add_array("tokenizer.ggml.tokens", tokens)
    if merges:
        # tokenizer.json may store merges as ["a", "b"] PAIRS (newer tokenizers)
        # or as "a b" strings. GGUF arrays cannot be nested (array-of-arrays is
        # not a valid type and won't load), so flatten each pair to "a b".
        merges_str = []
        for mg in merges:
            if isinstance(mg, (list, tuple)) and len(mg) == 2:
                merges_str.append(f"{mg[0]} {mg[1]}")
            elif isinstance(mg, str):
                merges_str.append(mg)
        writer.add_array("tokenizer.ggml.merges", merges_str)
    # Find EOS token
    eos_id = lang_cfg.get("eos_token_id", 1)
    writer.add_uint32("deepseek_ocr2.tokenizer.eos_id", eos_id)

    # Write tensors
    dtype_np = np.float16 if args.fp16 else np.float32
    dtype_gguf = gguf.GGMLQuantizationType.F16 if args.fp16 else gguf.GGMLQuantizationType.F32

    total_params = 0
    tensor_count = 0

    dtype_map = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I32": 4, "I64": 8}

    def read_tensor(fpath, info, base_offset):
        """Read a single tensor from safetensors via mmap, convert bf16→f32."""
        offsets = info["data_offsets"]
        shape = info["shape"]
        dtype_str = info["dtype"]
        byte_start = base_offset + offsets[0]
        byte_end = base_offset + offsets[1]
        n_bytes = byte_end - byte_start
        n_elements = 1
        for s in shape:
            n_elements *= s

        with open(fpath, "rb") as f:
            f.seek(byte_start)
            raw = f.read(n_bytes)

        if dtype_str == "BF16":
            # Convert bf16 → f32 via bit shift
            u16 = np.frombuffer(raw, dtype=np.uint16)
            u32 = u16.astype(np.uint32) << 16
            arr = u32.view(np.float32).reshape(shape)
        elif dtype_str == "F16":
            arr = np.frombuffer(raw, dtype=np.float16).reshape(shape).astype(np.float32)
        elif dtype_str == "F32":
            arr = np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
        elif dtype_str == "I32":
            arr = np.frombuffer(raw, dtype=np.int32).reshape(shape).astype(np.float32)
        elif dtype_str == "I64":
            arr = np.frombuffer(raw, dtype=np.int64).reshape(shape).astype(np.float32)
        else:
            raise ValueError(f"Unsupported dtype: {dtype_str}")
        return arr

    for name in tensor_names:
        info = header_json[name]
        data = read_tensor(args.model, info, data_offset)

        # Flatten 4D conv/patch_embed weights
        if data.ndim == 4:
            data = data.reshape(data.shape[0], -1)

        data = data.astype(dtype_np)
        total_params += data.size
        writer.add_tensor(name, data, raw_dtype=dtype_gguf)
        tensor_count += 1

        if tensor_count % 100 == 0:
            print(f"  {tensor_count} tensors processed...", end="\r")

    print(f"\nWriting {tensor_count} tensors...")

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
