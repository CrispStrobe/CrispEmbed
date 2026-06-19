#!/usr/bin/env python3
"""Convert SmolDocling-256M (Idefics3 architecture, IBM Research, Apache-2.0) to GGUF.

Architecture:
  SigLIP vision encoder (12L, 768d, 12H, patch=16, image=512)
  → pixel_shuffle(scale=4) + Linear(12288→576) connector
  → SmolLM2-135M decoder (30L, 576d, 9H/3KV, SwiGLU, RoPE θ=100k)

Usage:
    # NOTE: Use PYTHONNOUSERSITE=1 on CIFS mounts to avoid numpy site-packages bug
    PYTHONNOUSERSITE=1 python models/convert-smoldocling-to-gguf.py \
        --model /mnt/storage/models/SmolDocling-256M-preview \
        --output /mnt/storage/gguf-models/smoldocling-f16.gguf --dtype f16
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import gguf
import numpy as np


def map_tensor_name(name):
    """Map HuggingFace Idefics3/SmolDocling tensor name to short GGUF name.

    Returns None for tensors the engine does not consume.
    All GGUF names must fit within 64 characters.
    """
    # --- Vision encoder (SigLIP) ---
    if name.startswith("model.vision_model."):
        s = name[len("model.vision_model."):]

        # Embeddings
        if s == "embeddings.patch_embedding.weight":
            return "vis.patch_embed.weight"
        if s == "embeddings.patch_embedding.bias":
            return "vis.patch_embed.bias"
        if s == "embeddings.position_embedding.weight":
            return "vis.pos_embed.weight"

        # Encoder layers
        if s.startswith("encoder.layers."):
            parts = s.split(".")
            li = int(parts[2])
            rest = ".".join(parts[3:])

            # Layer norms
            rest = rest.replace("layer_norm1", "ln1")
            rest = rest.replace("layer_norm2", "ln2")

            # Self-attention
            rest = rest.replace("self_attn.q_proj", "attn.q")
            rest = rest.replace("self_attn.k_proj", "attn.k")
            rest = rest.replace("self_attn.v_proj", "attn.v")
            rest = rest.replace("self_attn.out_proj", "attn.out")

            # MLP
            rest = rest.replace("mlp.fc1", "mlp.fc1")
            rest = rest.replace("mlp.fc2", "mlp.fc2")

            return f"vis.layers.{li}.{rest}"

        # Post-layernorm
        if s == "post_layernorm.weight":
            return "vis.post_ln.weight"
        if s == "post_layernorm.bias":
            return "vis.post_ln.bias"

        return None

    # --- Connector (pixel_shuffle + linear projection) ---
    if name == "model.connector.modality_projection.proj.weight":
        return "connector.proj.weight"

    # --- LLM decoder (SmolLM2) ---
    if name.startswith("model.text_model."):
        s = name[len("model.text_model."):]

        if s == "embed_tokens.weight":
            return "llm.embed.weight"
        if s == "norm.weight":
            return "llm.norm.weight"

        if s.startswith("layers."):
            parts = s.split(".")
            li = int(parts[1])
            rest = ".".join(parts[2:])

            # Attention norms
            rest = rest.replace("input_layernorm", "attn_norm")
            rest = rest.replace("post_attention_layernorm", "ffn_norm")

            # Self-attention projections
            rest = rest.replace("self_attn.q_proj", "attn.q")
            rest = rest.replace("self_attn.k_proj", "attn.k")
            rest = rest.replace("self_attn.v_proj", "attn.v")
            rest = rest.replace("self_attn.o_proj", "attn.o")

            # MLP (SwiGLU)
            rest = rest.replace("mlp.gate_proj", "ffn.gate")
            rest = rest.replace("mlp.up_proj", "ffn.up")
            rest = rest.replace("mlp.down_proj", "ffn.down")

            return f"llm.layers.{li}.{rest}"

        return None

    # --- LM head ---
    if name == "lm_head.weight":
        return "llm.lm_head.weight"

    return None


def read_safetensors_header(filepath):
    """Parse safetensors header without loading tensor data."""
    with open(filepath, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header_json = json.loads(f.read(header_size))
    header_json.pop("__metadata__", None)
    data_offset = 8 + header_size
    return header_json, data_offset


def read_tensor_from_safetensors(filepath, info, data_offset):
    """Read a single tensor from safetensors file, converting bf16->f32."""
    offsets = info["data_offsets"]
    shape = info["shape"]
    dtype_str = info["dtype"]
    byte_start = data_offset + offsets[0]
    n_bytes = offsets[1] - offsets[0]

    with open(filepath, "rb") as f:
        f.seek(byte_start)
        raw = f.read(n_bytes)

    if dtype_str == "BF16":
        u16 = np.frombuffer(raw, dtype=np.uint16)
        u32 = u16.astype(np.uint32) << 16
        arr = u32.view(np.float32).reshape(shape)
    elif dtype_str == "F16":
        arr = np.frombuffer(raw, dtype=np.float16).reshape(shape).astype(np.float32)
    elif dtype_str == "F32":
        arr = np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    return arr


def main():
    p = argparse.ArgumentParser(
        description="Convert SmolDocling-256M (Idefics3) to GGUF")
    p.add_argument("--model", required=True,
                    help="Path to model directory (HF download or local)")
    p.add_argument("--output", required=True, help="Output GGUF path")
    p.add_argument("--dtype", choices=["f32", "f16"], default="f16",
                    help="Storage dtype (default: f16)")
    args = p.parse_args()

    model_dir = Path(args.model)
    if not model_dir.is_dir():
        print(f"Error: {model_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Load config
    config_path = model_dir / "config.json"
    if not config_path.exists():
        print(f"Error: {config_path} not found", file=sys.stderr)
        sys.exit(1)
    cfg = json.load(open(config_path))

    tie = cfg.get("tie_word_embeddings", False)
    image_token_id = cfg.get("image_token_id", 49190)

    # Vision config
    vis_cfg = cfg.get("vision_config", {})
    vis_hidden = vis_cfg.get("hidden_size", 768)
    vis_heads = vis_cfg.get("num_attention_heads", 12)
    vis_layers = vis_cfg.get("num_hidden_layers", 12)
    vis_patch = vis_cfg.get("patch_size", 16)
    vis_image = vis_cfg.get("image_size", 512)
    vis_intermediate = vis_cfg.get("intermediate_size", 3072)

    # Connector config
    scale_factor = cfg.get("scale_factor", 4)

    # Text/LLM config
    txt_cfg = cfg.get("text_config", {})
    llm_hidden = txt_cfg.get("hidden_size", 576)
    llm_heads = txt_cfg.get("num_attention_heads", 9)
    llm_kv_heads = txt_cfg.get("num_key_value_heads", 3)
    llm_layers = txt_cfg.get("num_hidden_layers", 30)
    llm_intermediate = txt_cfg.get("intermediate_size", 1536)
    llm_head_dim = llm_hidden // llm_heads  # 64
    rms_eps = txt_cfg.get("rms_norm_eps", 1e-5)
    rope_theta = txt_cfg.get("rope_theta", 100000.0)
    vocab_size = txt_cfg.get("vocab_size", 49280)

    print(f"SmolDocling-256M architecture:")
    print(f"  Vision: {vis_layers}L, {vis_hidden}d, {vis_heads}H, "
          f"patch={vis_patch}, image={vis_image}")
    print(f"  Connector: pixel_shuffle(scale={scale_factor}) + "
          f"Linear({vis_hidden * scale_factor**2}→{llm_hidden})")
    print(f"  LLM: {llm_layers}L, {llm_hidden}d, {llm_heads}H/{llm_kv_heads}KV, "
          f"ffn={llm_intermediate}, vocab={vocab_size}")
    print(f"  RoPE theta={rope_theta}, RMSNorm eps={rms_eps}")
    print(f"  tie_word_embeddings={tie}, image_token_id={image_token_id}")

    # Find safetensors files
    st_files = sorted(str(f) for f in model_dir.glob("*.safetensors"))
    if not st_files:
        print(f"Error: no .safetensors files in {model_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"\nSafetensors files: {len(st_files)}")

    # Parse all safetensors headers to build tensor map
    # tensor_map: gguf_name -> (src_name, filepath, info, data_offset)
    tensor_map = {}
    total_src = 0
    skipped = []
    for sf_path in st_files:
        header, data_off = read_safetensors_header(sf_path)
        total_src += len(header)
        for src_name, info in header.items():
            gguf_name = map_tensor_name(src_name)
            if gguf_name is None:
                skipped.append(src_name)
                continue
            # Validate 64-char limit
            if len(gguf_name) > 64:
                print(f"WARNING: GGUF name exceeds 64 chars: {gguf_name} "
                      f"({len(gguf_name)})", file=sys.stderr)
            tensor_map[gguf_name] = (src_name, sf_path, info, data_off)

    print(f"Source tensors: {total_src}")
    print(f"Mapped tensors: {len(tensor_map)}")
    if skipped:
        print(f"Skipped tensors: {len(skipped)}")
        for s in skipped:
            print(f"  - {s}")

    # Load tokenizer
    tokenizer_path = model_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        print(f"Error: {tokenizer_path} not found", file=sys.stderr)
        sys.exit(1)
    tok_data = json.load(open(tokenizer_path))
    tok_model = tok_data.get("model", {})
    vocab = tok_model.get("vocab", {})
    merges = tok_model.get("merges", [])
    print(f"Tokenizer: {len(vocab)} tokens, {len(merges)} merges")

    # Build token list and scores arrays
    n_tokens = max(vocab.values()) + 1 if vocab else 0
    tokens = [""] * n_tokens
    scores = [0.0] * n_tokens
    for tok, idx in vocab.items():
        if idx < n_tokens:
            tokens[idx] = tok
            # BPE: use negative index as score (earlier = higher priority)
            scores[idx] = -float(idx)

    # Create GGUF writer
    use_fp16 = (args.dtype == "f16")
    dtype_np = np.float16 if use_fp16 else np.float32
    dtype_gguf = (gguf.GGMLQuantizationType.F16 if use_fp16
                  else gguf.GGMLQuantizationType.F32)

    writer = gguf.GGUFWriter(str(args.output), arch="smoldocling",
                             use_temp_file=True)

    # --- Write metadata ---
    writer.add_string("general.name", "SmolDocling-256M-preview")
    writer.add_string("general.license", "Apache-2.0")
    writer.add_string("general.source",
                       "https://huggingface.co/ds4sd/SmolDocling-256M-preview")

    # Vision hyperparams
    writer.add_uint32("smoldocling.vision.hidden_size", vis_hidden)
    writer.add_uint32("smoldocling.vision.num_heads", vis_heads)
    writer.add_uint32("smoldocling.vision.num_layers", vis_layers)
    writer.add_uint32("smoldocling.vision.patch_size", vis_patch)
    writer.add_uint32("smoldocling.vision.image_size", vis_image)
    writer.add_uint32("smoldocling.vision.intermediate_size", vis_intermediate)

    # Connector
    writer.add_uint32("smoldocling.connector.scale_factor", scale_factor)

    # LLM hyperparams
    writer.add_uint32("smoldocling.hidden_size", llm_hidden)
    writer.add_uint32("smoldocling.num_attention_heads", llm_heads)
    writer.add_uint32("smoldocling.num_key_value_heads", llm_kv_heads)
    writer.add_uint32("smoldocling.num_hidden_layers", llm_layers)
    writer.add_uint32("smoldocling.intermediate_size", llm_intermediate)
    writer.add_uint32("smoldocling.head_dim", llm_head_dim)
    writer.add_float32("smoldocling.rms_norm_eps", rms_eps)
    writer.add_float32("smoldocling.rope_theta", rope_theta)
    writer.add_uint32("smoldocling.vocab_size", vocab_size)
    writer.add_uint32("smoldocling.image_token_id", image_token_id)

    # Tokenizer data
    writer.add_array("tokenizer.tokens", tokens)
    if merges:
        merges_str = []
        for mg in merges:
            if isinstance(mg, (list, tuple)) and len(mg) == 2:
                merges_str.append(f"{mg[0]} {mg[1]}")
            elif isinstance(mg, str):
                merges_str.append(mg)
        writer.add_array("tokenizer.merges", merges_str)
    writer.add_array("tokenizer.scores", scores)

    # --- Write tensors ---
    total_params = 0
    tensor_count = 0

    # Sort by GGUF name for deterministic output
    for gguf_name in sorted(tensor_map.keys()):
        src_name, sf_path, info, data_off = tensor_map[gguf_name]

        # Read tensor data (bf16 auto-converted to f32)
        data = read_tensor_from_safetensors(sf_path, info, data_off)

        # Flatten 4D conv weights (patch embedding)
        if data.ndim == 4:
            data = data.reshape(data.shape[0], -1)

        # Convert to target dtype
        data = data.astype(dtype_np)
        total_params += data.size

        writer.add_tensor(gguf_name, data, raw_dtype=dtype_gguf)
        tensor_count += 1

        if tensor_count % 50 == 0:
            print(f"  {tensor_count}/{len(tensor_map)} tensors processed...",
                  end="\r")

    print(f"\nWriting {tensor_count} tensors to {args.output}...")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    out_size = Path(args.output).stat().st_size
    print(f"\nDone: {args.output}")
    print(f"  Tensors: {tensor_count}")
    print(f"  Parameters: {total_params:,}")
    print(f"  File size: {out_size:,} bytes ({out_size / 1024 / 1024:.1f} MB)")
    print(f"  Dtype: {args.dtype}")


if __name__ == "__main__":
    main()
