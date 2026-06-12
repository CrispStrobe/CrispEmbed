#!/usr/bin/env python3
"""Convert SauerkrautLM-LFM2.5-GLiNER → GGUF for CrispEmbed NER inference.

Architecture:
  LFM2.5-350M bidirectional backbone (16 layers: 10 ShortConv + 6 GQA)
  + layer fusion (attention-weighted sum of all layers)
  + BiLSTM (1-layer, bidirectional, hidden=512)
  + GLiNER head: span_rep (markerV1) + prompt_rep + dot-product scorer

Usage:
  python models/convert-gliner-lfm-to-gguf.py \
      --model VAGOsolutions/SauerkrautLM-LFM2.5-GLiNER \
      --output gliner-lfm-f32.gguf
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

# gguf package from the ggml ecosystem
sys.path.insert(0, str(Path(__file__).parent.parent / "ggml" / "scripts"))
try:
    import gguf
except ImportError:
    print("ERROR: gguf package not found. pip install gguf", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# GGUF tensor naming
# ---------------------------------------------------------------------------

# HuggingFace prefix → GGUF prefix mapping for the LFM2 backbone
# HF: token_rep_layer.bert_layer.model.layers.{i}.{conv|self_attn|feed_forward|operator_norm|ffn_norm}
# GGUF: lfm.layers.{i}.{conv|attn|ff|operator_norm|ffn_norm}

def remap_tensor_name(hf_name: str) -> str | None:
    """Map HuggingFace state_dict key → GGUF tensor name. Return None to skip."""
    n = hf_name

    # --- LFM2 backbone (inside token_rep_layer.bert_layer.model) ---
    prefix = "token_rep_layer.bert_layer.model."
    if n.startswith(prefix):
        rest = n[len(prefix):]

        if rest == "embed_tokens.weight":
            return "lfm.embed_tokens.weight"
        if rest == "embedding_norm.weight":
            return "lfm.embedding_norm.weight"

        if rest.startswith("layers."):
            parts = rest.split(".", 2)  # layers, idx, remainder
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

        # Skip rotary embeddings (computed at runtime)
        if "rotary_emb" in rest:
            return None

    # --- Layer fuser ---
    fuser_prefix = "token_rep_layer.bert_layer.layers_fuser."
    if n.startswith(fuser_prefix):
        rest = n[len(fuser_prefix):]
        return f"fuser.{rest}"

    # --- BiLSTM (rnn.lstm) ---
    if n.startswith("rnn.lstm."):
        rest = n[len("rnn.lstm."):]
        return f"lstm.{rest}"

    # --- GLiNER span representation ---
    span_prefix = "span_rep_layer.span_rep_layer."
    if n.startswith(span_prefix):
        rest = n[len(span_prefix):]
        return f"span.{rest}"

    # --- Prompt/entity representation ---
    if n.startswith("prompt_rep_layer."):
        rest = n[len("prompt_rep_layer."):]
        return f"prompt_rep.{rest}"

    # --- Scorer temperature ---
    if n == "log_score_temperature":
        return "scorer.log_temperature"

    print(f"  WARN: unmapped tensor: {n}", file=sys.stderr)
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert GLiNER-LFM to GGUF")
    parser.add_argument("--model", required=True, help="HuggingFace model ID or local path")
    parser.add_argument("--output", required=True, help="Output GGUF path")
    parser.add_argument("--dtype", default="f32", choices=["f32", "f16"],
                        help="Storage dtype (default: f32)")
    args = parser.parse_args()

    model_dir = args.model
    # If it's a HF model ID, download it
    if not os.path.isdir(model_dir):
        from huggingface_hub import snapshot_download
        cache_dir = os.environ.get("HF_HUB_CACHE",
                                   "/mnt/akademie_storage/huggingface/hub")
        model_dir = snapshot_download(args.model, cache_dir=cache_dir)
        print(f"Downloaded to: {model_dir}")

    # Load config
    config_path = os.path.join(model_dir, "gliner_config.json")
    with open(config_path) as f:
        config = json.load(f)
    enc_cfg = config["encoder_config"]

    # Load tokenizer
    tok_path = os.path.join(model_dir, "tokenizer.json")
    with open(tok_path) as f:
        tokenizer = json.load(f)

    # Extract hyperparameters
    hidden_size = enc_cfg["hidden_size"]           # 1024
    n_layers = enc_cfg["num_hidden_layers"]        # 16
    n_heads = enc_cfg["num_attention_heads"]       # 16
    n_kv_heads = enc_cfg["num_key_value_heads"]    # 8
    head_dim = hidden_size // n_heads              # 64
    vocab_size = enc_cfg["vocab_size"]             # 64404
    conv_kernel = enc_cfg.get("conv_L_cache", 3)
    rope_theta = enc_cfg.get("rope_parameters", {}).get("rope_theta", 1000000.0)

    # Compute FF dim (SwiGLU adjusted, rounded to block_multiple_of)
    intermediate_size = enc_cfg.get("intermediate_size", 6656)
    ff_multiplier = enc_cfg.get("block_ffn_dim_multiplier", 1.0)
    block_multiple = enc_cfg.get("block_multiple_of", 256)
    ff_dim = int(2 * intermediate_size / 3 * ff_multiplier)
    ff_dim = ((ff_dim + block_multiple - 1) // block_multiple) * block_multiple

    # Layer types
    layer_types_list = enc_cfg.get("layer_types", [])
    layer_types_str = ""
    for lt in layer_types_list:
        if "conv" in lt:
            layer_types_str += "c"
        elif "attention" in lt:
            layer_types_str += "a"
        else:
            layer_types_str += "?"

    # GLiNER-specific params
    max_width = config.get("max_width", 12)
    ent_token_id = config.get("class_token_index", 64402)
    sep_token = config.get("sep_token", "<<SEP>>")
    ent_token = config.get("ent_token", "<<ENT>>")
    span_mode = config.get("span_mode", "markerV1")

    print(f"Model: {args.model}")
    print(f"  hidden={hidden_size}, layers={n_layers}, heads={n_heads}/{n_kv_heads}")
    print(f"  ff_dim={ff_dim}, conv_kernel={conv_kernel}")
    print(f"  layer_types={layer_types_str}")
    print(f"  vocab={vocab_size}, max_width={max_width}")
    print(f"  ent_token={ent_token} (id={ent_token_id}), sep_token={sep_token}")
    print(f"  span_mode={span_mode}")

    # Determine dtype
    if args.dtype == "f16":
        np_dtype = np.float16
        gguf_dtype = gguf.GGMLQuantizationType.F16
    else:
        np_dtype = np.float32
        gguf_dtype = gguf.GGMLQuantizationType.F32

    # --- Initialize GGUF writer ---
    writer = gguf.GGUFWriter(args.output, arch="gliner")

    # General metadata
    writer.add_string("general.architecture", "gliner")
    writer.add_string("general.name", "SauerkrautLM-LFM2.5-GLiNER")
    writer.add_string("general.license", "lfm1.0")
    writer.add_string("general.source", "VAGOsolutions/SauerkrautLM-LFM2.5-GLiNER")

    # LFM2 backbone hyperparameters
    writer.add_uint32("gliner.hidden_size", hidden_size)
    writer.add_uint32("gliner.n_layers", n_layers)
    writer.add_uint32("gliner.n_heads", n_heads)
    writer.add_uint32("gliner.n_kv_heads", n_kv_heads)
    writer.add_uint32("gliner.head_dim", head_dim)
    writer.add_uint32("gliner.ff_dim", ff_dim)
    writer.add_uint32("gliner.conv_kernel", conv_kernel)
    writer.add_float32("gliner.rope_theta", rope_theta)
    writer.add_string("gliner.layer_types", layer_types_str)
    writer.add_uint32("gliner.vocab_size", vocab_size)

    # GLiNER head params
    writer.add_uint32("gliner.max_width", max_width)
    writer.add_uint32("gliner.ent_token_id", ent_token_id)
    writer.add_string("gliner.span_mode", span_mode)

    # --- Tokenizer ---
    # Extract vocab tokens and merges from tokenizer.json
    model_data = tokenizer.get("model", {})
    vocab_dict = model_data.get("vocab", {})
    merges = model_data.get("merges", [])
    added_tokens = tokenizer.get("added_tokens", [])

    # Build token list ordered by ID
    max_id = max(vocab_dict.values()) if vocab_dict else 0
    for at in added_tokens:
        max_id = max(max_id, at["id"])
    tokens = [""] * (max_id + 1)
    for tok, tid in vocab_dict.items():
        tokens[tid] = tok
    for at in added_tokens:
        tokens[at["id"]] = at["content"]

    writer.add_array("tokenizer.tokens", tokens)
    writer.add_uint32("tokenizer.vocab_size", len(tokens))
    # Store merges as a single newline-separated string (avoids GGUF large
    # string array limitation in the C reader).
    if merges:
        # Each merge is either a string "a b" or a list ["a", "b"]
        merge_strs = []
        for m in merges:
            if isinstance(m, list):
                merge_strs.append(" ".join(m))
            else:
                merge_strs.append(str(m))
        writer.add_string("tokenizer.merges_blob", "\n".join(merge_strs))
    writer.add_uint32("tokenizer.bos_id", 1)
    writer.add_uint32("tokenizer.eos_id", 7)
    writer.add_uint32("tokenizer.pad_id", 0)

    # Find <<ENT>> and <<SEP>> token IDs
    ent_id = None
    sep_id = None
    for at in added_tokens:
        if at["content"] == ent_token:
            ent_id = at["id"]
        if at["content"] == sep_token:
            sep_id = at["id"]
    if ent_id is not None:
        writer.add_uint32("gliner.ent_token_id", ent_id)
    if sep_id is not None:
        writer.add_uint32("gliner.sep_token_id", sep_id)

    print(f"  tokenizer: {len(tokens)} tokens, {len(merges)} merges")
    print(f"  <<ENT>>={ent_id}, <<SEP>>={sep_id}")

    # --- Load and write tensors ---
    import torch

    pt_path = os.path.join(model_dir, "pytorch_model.bin")
    print(f"Loading weights from: {pt_path}")
    state_dict = torch.load(pt_path, map_location="cpu", weights_only=False)

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

    # Finalize
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    out_size = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Output: {args.output} ({out_size:.1f} MB)")


if __name__ == "__main__":
    main()
