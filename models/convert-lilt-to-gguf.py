#!/usr/bin/env python3
"""Convert a HuggingFace LiLT model to GGUF format.

Supports base model (SCUT-DLVCLab/lilt-roberta-en-base) and fine-tuned
variants (e.g. philschmid/lilt-en-funsd for token classification).

    pip install torch transformers gguf safetensors
    python convert-lilt-to-gguf.py \
        --model SCUT-DLVCLab/lilt-roberta-en-base \
        --output lilt-base-f32.gguf

    python convert-lilt-to-gguf.py \
        --model philschmid/lilt-en-funsd \
        --output lilt-funsd-f32.gguf
"""

import argparse
import json
import sys
from pathlib import Path

import gguf
import numpy as np

ARCH = "lilt"


def f32(t) -> np.ndarray:
    """Convert tensor to float32 numpy."""
    if hasattr(t, 'detach'):
        return t.detach().float().cpu().numpy().astype(np.float32)
    return np.array(t, dtype=np.float32)


def f16(t) -> np.ndarray:
    if hasattr(t, 'detach'):
        return t.detach().float().cpu().numpy().astype(np.float16)
    return np.array(t, dtype=np.float16)


def main():
    parser = argparse.ArgumentParser(description="Convert LiLT model to GGUF")
    parser.add_argument("--model", required=True, help="HF model ID or local path")
    parser.add_argument("--output", required=True, help="Output .gguf path")
    parser.add_argument("--dtype", choices=["f16", "f32"], default="f32",
                        help="Weight dtype for linear layers (default: f32)")
    args = parser.parse_args()

    wt = f16 if args.dtype == "f16" else f32

    print(f"Loading model: {args.model}")

    # Load config
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)

    # Load weights via safetensors (memory-efficient)
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    import os
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

    model_path = Path(args.model)
    if model_path.is_dir():
        sf_path = model_path / "model.safetensors"
        if not sf_path.exists():
            # Try pytorch bin
            import torch
            sd = torch.load(model_path / "pytorch_model.bin", map_location="cpu", weights_only=True)
        else:
            sd = {}
            with safe_open(str(sf_path), framework="pt") as f:
                for k in f.keys():
                    sd[k] = f.get_tensor(k)
    else:
        try:
            sf_path = hf_hub_download(args.model, "model.safetensors",
                cache_dir='/mnt/akademie_storage/huggingface/hub/')
        except Exception:
            sf_path = hf_hub_download(args.model, "pytorch_model.bin",
                cache_dir='/mnt/akademie_storage/huggingface/hub/')
        sd = {}
        if sf_path.endswith('.safetensors'):
            with safe_open(sf_path, framework="pt") as f:
                for k in f.keys():
                    sd[k] = f.get_tensor(k)
        else:
            import torch
            sd = torch.load(sf_path, map_location="cpu", weights_only=True)

    # Strip "lilt." prefix if present (fine-tuned models wrap in LiltForTokenClassification)
    stripped = {}
    for k, v in sd.items():
        nk = k.replace("lilt.", "", 1) if k.startswith("lilt.") else k
        stripped[nk] = v
    sd = stripped

    n_layers = config.num_hidden_layers
    hidden = config.hidden_size
    n_heads = config.num_attention_heads
    inter = config.intermediate_size
    shrink = getattr(config, 'channel_shrink_ratio', 4)
    layout_dim = hidden // shrink  # 192 for base
    max_2d = getattr(config, 'max_2d_position_embeddings', 1024)
    vocab = config.vocab_size

    print(f"  hidden={hidden}, layers={n_layers}, heads={n_heads}, inter={inter}")
    print(f"  layout_dim={layout_dim} (shrink={shrink}), max_2d={max_2d}, vocab={vocab}")

    # Detect classifier head (token classification fine-tuned model)
    has_classifier = "classifier.weight" in sd
    num_labels = 0
    id2label = {}
    if has_classifier:
        num_labels = sd["classifier.weight"].shape[0]
        id2label = getattr(config, 'id2label', {})
        print(f"  classifier: {num_labels} labels: {id2label}")

    # Build GGUF
    writer = gguf.GGUFWriter(str(args.output), arch=ARCH)

    # Metadata
    writer.add_uint32("lilt.vocab_size", vocab)
    writer.add_uint32("lilt.hidden_size", hidden)
    writer.add_uint32("lilt.num_hidden_layers", n_layers)
    writer.add_uint32("lilt.num_attention_heads", n_heads)
    writer.add_uint32("lilt.intermediate_size", inter)
    writer.add_uint32("lilt.max_position_embeddings", config.max_position_embeddings)
    writer.add_uint32("lilt.max_2d_position_embeddings", max_2d)
    writer.add_uint32("lilt.channel_shrink_ratio", shrink)
    writer.add_uint32("lilt.layout_dim", layout_dim)
    writer.add_float32("lilt.layer_norm_eps", getattr(config, "layer_norm_eps", 1e-5))

    if has_classifier:
        writer.add_uint32("lilt.num_labels", num_labels)
        if id2label:
            for i, label in id2label.items():
                writer.add_string(f"lilt.label.{i}", str(label))

    # --- Text embeddings ---
    writer.add_tensor("token_embd.weight", f32(sd["embeddings.word_embeddings.weight"]))
    writer.add_tensor("position_embd.weight", f32(sd["embeddings.position_embeddings.weight"]))
    if "embeddings.token_type_embeddings.weight" in sd:
        writer.add_tensor("token_type_embd.weight", f32(sd["embeddings.token_type_embeddings.weight"]))
    writer.add_tensor("embd_ln.weight", f32(sd["embeddings.LayerNorm.weight"]))
    writer.add_tensor("embd_ln.bias", f32(sd["embeddings.LayerNorm.bias"]))

    # --- Layout embeddings ---
    writer.add_tensor("layout.x_embd.weight", f32(sd["layout_embeddings.x_position_embeddings.weight"]))
    writer.add_tensor("layout.y_embd.weight", f32(sd["layout_embeddings.y_position_embeddings.weight"]))
    writer.add_tensor("layout.h_embd.weight", f32(sd["layout_embeddings.h_position_embeddings.weight"]))
    writer.add_tensor("layout.w_embd.weight", f32(sd["layout_embeddings.w_position_embeddings.weight"]))
    writer.add_tensor("layout.box_proj.weight", wt(sd["layout_embeddings.box_linear_embeddings.weight"]))
    writer.add_tensor("layout.box_proj.bias", f32(sd["layout_embeddings.box_linear_embeddings.bias"]))
    writer.add_tensor("layout.pos_embd.weight", f32(sd["layout_embeddings.box_position_embeddings.weight"]))
    writer.add_tensor("layout.ln.weight", f32(sd["layout_embeddings.LayerNorm.weight"]))
    writer.add_tensor("layout.ln.bias", f32(sd["layout_embeddings.LayerNorm.bias"]))

    # --- Encoder layers ---
    for i in range(n_layers):
        pfx = f"encoder.layer.{i}"

        # Text attention
        writer.add_tensor(f"blk.{i}.attn_q.weight", wt(sd[f"{pfx}.attention.self.query.weight"]))
        writer.add_tensor(f"blk.{i}.attn_q.bias", f32(sd[f"{pfx}.attention.self.query.bias"]))
        writer.add_tensor(f"blk.{i}.attn_k.weight", wt(sd[f"{pfx}.attention.self.key.weight"]))
        writer.add_tensor(f"blk.{i}.attn_k.bias", f32(sd[f"{pfx}.attention.self.key.bias"]))
        writer.add_tensor(f"blk.{i}.attn_v.weight", wt(sd[f"{pfx}.attention.self.value.weight"]))
        writer.add_tensor(f"blk.{i}.attn_v.bias", f32(sd[f"{pfx}.attention.self.value.bias"]))
        writer.add_tensor(f"blk.{i}.attn_o.weight", wt(sd[f"{pfx}.attention.output.dense.weight"]))
        writer.add_tensor(f"blk.{i}.attn_o.bias", f32(sd[f"{pfx}.attention.output.dense.bias"]))
        writer.add_tensor(f"blk.{i}.attn_ln.weight", f32(sd[f"{pfx}.attention.output.LayerNorm.weight"]))
        writer.add_tensor(f"blk.{i}.attn_ln.bias", f32(sd[f"{pfx}.attention.output.LayerNorm.bias"]))

        # Layout attention
        writer.add_tensor(f"blk.{i}.layout_q.weight", wt(sd[f"{pfx}.attention.self.layout_query.weight"]))
        writer.add_tensor(f"blk.{i}.layout_q.bias", f32(sd[f"{pfx}.attention.self.layout_query.bias"]))
        writer.add_tensor(f"blk.{i}.layout_k.weight", wt(sd[f"{pfx}.attention.self.layout_key.weight"]))
        writer.add_tensor(f"blk.{i}.layout_k.bias", f32(sd[f"{pfx}.attention.self.layout_key.bias"]))
        writer.add_tensor(f"blk.{i}.layout_v.weight", wt(sd[f"{pfx}.attention.self.layout_value.weight"]))
        writer.add_tensor(f"blk.{i}.layout_v.bias", f32(sd[f"{pfx}.attention.self.layout_value.bias"]))
        writer.add_tensor(f"blk.{i}.layout_attn_o.weight", wt(sd[f"{pfx}.attention.layout_output.dense.weight"]))
        writer.add_tensor(f"blk.{i}.layout_attn_o.bias", f32(sd[f"{pfx}.attention.layout_output.dense.bias"]))
        writer.add_tensor(f"blk.{i}.layout_attn_ln.weight", f32(sd[f"{pfx}.attention.layout_output.LayerNorm.weight"]))
        writer.add_tensor(f"blk.{i}.layout_attn_ln.bias", f32(sd[f"{pfx}.attention.layout_output.LayerNorm.bias"]))

        # Text FFN
        writer.add_tensor(f"blk.{i}.ffn_up.weight", wt(sd[f"{pfx}.intermediate.dense.weight"]))
        writer.add_tensor(f"blk.{i}.ffn_up.bias", f32(sd[f"{pfx}.intermediate.dense.bias"]))
        writer.add_tensor(f"blk.{i}.ffn_down.weight", wt(sd[f"{pfx}.output.dense.weight"]))
        writer.add_tensor(f"blk.{i}.ffn_down.bias", f32(sd[f"{pfx}.output.dense.bias"]))
        writer.add_tensor(f"blk.{i}.ffn_ln.weight", f32(sd[f"{pfx}.output.LayerNorm.weight"]))
        writer.add_tensor(f"blk.{i}.ffn_ln.bias", f32(sd[f"{pfx}.output.LayerNorm.bias"]))

        # Layout FFN
        writer.add_tensor(f"blk.{i}.layout_ffn_up.weight", wt(sd[f"{pfx}.layout_intermediate.dense.weight"]))
        writer.add_tensor(f"blk.{i}.layout_ffn_up.bias", f32(sd[f"{pfx}.layout_intermediate.dense.bias"]))
        writer.add_tensor(f"blk.{i}.layout_ffn_down.weight", wt(sd[f"{pfx}.layout_output.dense.weight"]))
        writer.add_tensor(f"blk.{i}.layout_ffn_down.bias", f32(sd[f"{pfx}.layout_output.dense.bias"]))
        writer.add_tensor(f"blk.{i}.layout_ffn_ln.weight", f32(sd[f"{pfx}.layout_output.LayerNorm.weight"]))
        writer.add_tensor(f"blk.{i}.layout_ffn_ln.bias", f32(sd[f"{pfx}.layout_output.LayerNorm.bias"]))

    # --- Classifier head (token classification) ---
    if has_classifier:
        writer.add_tensor("classifier.weight", f32(sd["classifier.weight"]))
        writer.add_tensor("classifier.bias", f32(sd["classifier.bias"]))

    # --- Tokenizer ---
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    vocab_list = [tokenizer.convert_ids_to_tokens(i) for i in range(vocab)]
    writer.add_token_list(vocab_list)
    writer.add_uint32("tokenizer.ggml.bos_token_id", tokenizer.bos_token_id or 0)
    writer.add_uint32("tokenizer.ggml.eos_token_id", tokenizer.eos_token_id or 2)
    writer.add_uint32("tokenizer.ggml.padding_token_id", tokenizer.pad_token_id or 1)

    # Write
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    import os
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\nWrote {args.output} ({size_mb:.1f} MB)")
    print(f"  arch={ARCH}, layers={n_layers}, hidden={hidden}, layout_dim={layout_dim}")
    if has_classifier:
        print(f"  classifier: {num_labels} labels")


if __name__ == "__main__":
    main()
