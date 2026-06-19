#!/usr/bin/env python3
"""Convert LiquidAI/LFM2.5-Embedding-350M → GGUF for CrispEmbed.

Architecture:
  LFM2.5-350M bidirectional backbone (16 layers: 10 ShortConv + 6 GQA)
  + CLS-token pooling. Same weights as the GLiNER-LFM backbone but without
  the layer fuser / BiLSTM / GLiNER head tensors.

GGUF arch tag: "lfm2"
Tensor prefix: "lfm.*" (identical naming to the gliner GGUF backbone section)

Usage:
  python models/convert-lfm2-embed-to-gguf.py \\
      --model LiquidAI/LFM2.5-Embedding-350M \\
      --output lfm2-embed-f16.gguf --dtype f16
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


# ---------------------------------------------------------------------------
# Tensor name mapping
# ---------------------------------------------------------------------------

def remap_tensor_name(hf_name: str) -> str | None:
    """Map HuggingFace state_dict key → GGUF tensor name. Return None to skip."""
    n = hf_name

    # Skip the LM head (tied to embed_tokens — not needed for embedding)
    if n == "lm_head.weight":
        return None
    # Skip rotary embeddings (computed at runtime)
    if "rotary_emb" in n:
        return None

    # ColBERT projection head (from sentence-transformers 1_Dense module)
    if n == "2_Dense.linear.weight" or n == "1_Dense.linear.weight" or n == "dense.weight":
        return "colbert.projection.weight"

    # Strip optional "model." prefix (some checkpoints include it, some don't)
    prefix = "model."
    rest = n[len(prefix):] if n.startswith(prefix) else n

    if rest == "embed_tokens.weight":
        return "lfm.embed_tokens.weight"
    if rest == "embedding_norm.weight":
        return "lfm.embedding_norm.weight"
    # Some HF checkpoints use "norm" as the final norm
    if rest == "norm.weight":
        return "lfm.embedding_norm.weight"

    if rest.startswith("layers."):
        parts = rest.split(".", 2)   # layers, idx, remainder
        idx = parts[1]
        remainder = parts[2]

        # Conv layers
        if remainder.startswith("conv."):
            return f"lfm.layers.{idx}.conv.{remainder[len('conv.'):]}"

        # Attention layers
        if remainder.startswith("self_attn."):
            attn_part = remainder[len("self_attn."):]
            return f"lfm.layers.{idx}.attn.{attn_part}"

        # FFN (feed_forward.w1/w2/w3)
        if remainder.startswith("feed_forward."):
            ff_part = remainder[len("feed_forward."):]
            return f"lfm.layers.{idx}.ff.{ff_part}"

        # Norms
        if remainder == "operator_norm.weight":
            return f"lfm.layers.{idx}.operator_norm.weight"
        if remainder == "ffn_norm.weight":
            return f"lfm.layers.{idx}.ffn_norm.weight"

    print(f"  WARN: unmapped tensor: {n}", file=sys.stderr)
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert LFM2.5-Embedding to GGUF")
    parser.add_argument("--model", required=True,
                        help="HuggingFace model ID or local directory")
    parser.add_argument("--output", required=True, help="Output .gguf path")
    parser.add_argument("--dtype", default="f16", choices=["f32", "f16"],
                        help="Storage dtype (default: f16)")
    args = parser.parse_args()

    model_dir = args.model
    if not os.path.isdir(model_dir):
        from huggingface_hub import snapshot_download
        cache_dir = os.environ.get("HF_HUB_CACHE",
                                   os.path.expanduser("~/.cache/huggingface/hub"))
        model_dir = snapshot_download(args.model, cache_dir=cache_dir)
        print(f"Downloaded to: {model_dir}")

    # Load config.json
    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)

    hidden_size    = config["hidden_size"]             # 1024
    n_layers       = config["num_hidden_layers"]       # 16
    n_heads        = config["num_attention_heads"]     # 16
    n_kv_heads     = config["num_key_value_heads"]     # 8
    head_dim       = hidden_size // n_heads            # 64
    vocab_size     = config["vocab_size"]              # 65536
    conv_kernel    = config.get("conv_L_cache", 3)
    norm_eps       = config.get("norm_eps", config.get("block_norm_eps", 1e-5))
    rope_theta     = config.get("rope_parameters", {}).get("rope_theta",
                        config.get("rope_theta", 1000000.0))

    # SwiGLU effective hidden dim (2/3 of intermediate_size, rounded to multiple_of)
    intermediate_size  = config.get("intermediate_size", config.get("block_ff_dim", 6656))
    ff_multiplier      = config.get("block_ffn_dim_multiplier", 1.0)
    block_multiple     = config.get("block_multiple_of", 256)
    ff_dim = int(2 * intermediate_size / 3 * ff_multiplier)
    ff_dim = ((ff_dim + block_multiple - 1) // block_multiple) * block_multiple

    # Layer types string (c=conv, a=attention)
    layer_types_list = config.get("layer_types", [])
    layer_types_str = ""
    for lt in layer_types_list:
        if "conv" in lt:
            layer_types_str += "c"
        elif "attention" in lt:
            layer_types_str += "a"
        else:
            layer_types_str += "?"

    bos_id = config.get("bos_token_id", 1)
    eos_id = config.get("eos_token_id", 7)
    pad_id = config.get("pad_token_id", 0)

    print(f"Model: {args.model}")
    print(f"  hidden={hidden_size}, layers={n_layers}, heads={n_heads}/{n_kv_heads}")
    print(f"  ff_dim={ff_dim} (intermediate={intermediate_size}), conv_kernel={conv_kernel}")
    print(f"  norm_eps={norm_eps}, rope_theta={rope_theta}")
    print(f"  layer_types={layer_types_str}")
    print(f"  vocab={vocab_size}, bos={bos_id}, eos={eos_id}")

    # Dtype
    if args.dtype == "f16":
        np_dtype    = np.float16
        gguf_dtype  = gguf.GGMLQuantizationType.F16
    else:
        np_dtype    = np.float32
        gguf_dtype  = gguf.GGMLQuantizationType.F32

    # --- GGUF writer ---
    writer = gguf.GGUFWriter(args.output, arch="lfm2")

    writer.add_string("general.architecture", "lfm2")
    writer.add_string("general.name", "LFM2.5-Embedding-350M")
    writer.add_string("general.license", "lfm1.0")
    writer.add_string("general.source", "LiquidAI/LFM2.5-Embedding-350M")

    # Hyperparameters
    writer.add_uint32("lfm2.hidden_size",  hidden_size)
    writer.add_uint32("lfm2.n_layers",     n_layers)
    writer.add_uint32("lfm2.n_heads",      n_heads)
    writer.add_uint32("lfm2.n_kv_heads",   n_kv_heads)
    writer.add_uint32("lfm2.head_dim",     head_dim)
    writer.add_uint32("lfm2.ff_dim",       ff_dim)
    writer.add_uint32("lfm2.conv_kernel",  conv_kernel)
    writer.add_float32("lfm2.rope_theta",  rope_theta)
    writer.add_float32("lfm2.norm_eps",    norm_eps)
    writer.add_string("lfm2.layer_types",  layer_types_str)
    writer.add_uint32("lfm2.vocab_size",   vocab_size)

    # Tokenizer
    tok_path = os.path.join(model_dir, "tokenizer.json")
    with open(tok_path) as f:
        tokenizer = json.load(f)

    model_data  = tokenizer.get("model", {})
    vocab_dict  = model_data.get("vocab", {})
    merges      = model_data.get("merges", [])
    added_toks  = tokenizer.get("added_tokens", [])

    max_id = max(vocab_dict.values()) if vocab_dict else 0
    for at in added_toks:
        max_id = max(max_id, at["id"])
    tokens_list = [""] * (max_id + 1)
    for tok, tid in vocab_dict.items():
        tokens_list[tid] = tok
    for at in added_toks:
        tokens_list[at["id"]] = at["content"]

    writer.add_array("tokenizer.ggml.tokens", tokens_list)
    writer.add_uint32("tokenizer.ggml.model", 0)  # 0 = GPT-2 BPE style

    # Store merges as array (standard GGUF format)
    merge_strs = []
    for m in merges:
        if isinstance(m, list):
            merge_strs.append(" ".join(m))
        else:
            merge_strs.append(str(m))
    if merge_strs:
        writer.add_array("tokenizer.ggml.merges", merge_strs)

    writer.add_uint32("tokenizer.ggml.bos_token_id",     bos_id)
    writer.add_uint32("tokenizer.ggml.eos_token_id",     eos_id)
    writer.add_uint32("tokenizer.ggml.padding_token_id", pad_id)

    print(f"  tokenizer: {len(tokens_list)} tokens, {len(merge_strs)} merges")

    # --- Load weights ---
    # Try safetensors first (the embedding model ships as model.safetensors)
    st_path = os.path.join(model_dir, "model.safetensors")
    pt_path = os.path.join(model_dir, "pytorch_model.bin")

    if os.path.exists(st_path):
        try:
            from safetensors import safe_open
            state_dict = {}
            with safe_open(st_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
            print(f"Loading weights from: {st_path}")
        except ImportError:
            print("safetensors not available, trying torch.load", file=sys.stderr)
            import torch
            state_dict = torch.load(pt_path, map_location="cpu", weights_only=True)
            print(f"Loading weights from: {pt_path}")
    else:
        import torch
        state_dict = torch.load(pt_path, map_location="cpu", weights_only=True)
        print(f"Loading weights from: {pt_path}")

    n_written = 0
    n_skipped = 0
    for hf_name in sorted(state_dict.keys()):
        gguf_name = remap_tensor_name(hf_name)
        if gguf_name is None:
            n_skipped += 1
            continue

        tensor = state_dict[hf_name]
        data = tensor.float().numpy().astype(np_dtype)

        writer.add_tensor(gguf_name, data, raw_dtype=gguf_dtype)
        n_written += 1
        if n_written % 20 == 0:
            print(f"  [{n_written}] {gguf_name} {list(data.shape)}")

    print(f"\nWritten: {n_written} tensors, skipped: {n_skipped}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    out_size = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Output: {args.output} ({out_size:.1f} MB)")


if __name__ == "__main__":
    main()
