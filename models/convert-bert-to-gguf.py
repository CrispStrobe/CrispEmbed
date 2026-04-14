#!/usr/bin/env python3
"""Convert a HuggingFace BERT/MiniLM/E5 model to CrispEmbed GGUF format.

    pip install torch transformers gguf
    python convert-bert-to-gguf.py \
        --model sentence-transformers/all-MiniLM-L6-v2 \
        --output all-MiniLM-L6-v2.gguf
"""

import argparse
import json
import sys
from pathlib import Path

import gguf
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig


ARCH = "bert"


def f32(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().numpy().astype(np.float32)


def f16(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().numpy().astype(np.float16)


def main():
    parser = argparse.ArgumentParser(description="Convert BERT-family model to GGUF")
    parser.add_argument("--model", required=True, help="HF model ID or local path")
    parser.add_argument("--output", required=True, help="Output .gguf path")
    parser.add_argument("--dtype", choices=["f16", "f32"], default="f32",
                        help="Weight dtype for linear layers (default: f32)")
    args = parser.parse_args()

    wt = f32 if args.dtype == "f32" else f16

    print(f"Loading model: {args.model}")
    config = AutoConfig.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.eval()
    sd = model.state_dict()

    print(f"Config: hidden={config.hidden_size} layers={config.num_hidden_layers} "
          f"heads={config.num_attention_heads} intermediate={config.intermediate_size} "
          f"vocab={config.vocab_size}")

    writer = gguf.GGUFWriter(str(args.output), arch=ARCH)

    # Metadata
    writer.add_uint32("bert.vocab_size", config.vocab_size)
    writer.add_uint32("bert.max_position_embeddings", config.max_position_embeddings)
    writer.add_uint32("bert.hidden_size", config.hidden_size)
    writer.add_uint32("bert.num_attention_heads", config.num_attention_heads)
    writer.add_uint32("bert.num_hidden_layers", config.num_hidden_layers)
    writer.add_uint32("bert.intermediate_size", config.intermediate_size)
    writer.add_float32("bert.layer_norm_eps", getattr(config, "layer_norm_eps", 1e-12))
    writer.add_uint32("bert.output_dim", config.hidden_size)

    # Tokenizer vocab
    vocab = tokenizer.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}
    tokens = [id_to_token.get(i, f"[UNK_{i}]") for i in range(config.vocab_size)]
    writer.add_array("tokenizer.ggml.tokens", tokens)
    writer.add_uint32("tokenizer.ggml.cls_token_id", tokenizer.cls_token_id or 101)
    writer.add_uint32("tokenizer.ggml.sep_token_id", tokenizer.sep_token_id or 102)
    writer.add_uint32("tokenizer.ggml.unknown_token_id", tokenizer.unk_token_id or 100)
    writer.add_uint32("tokenizer.ggml.padding_token_id", tokenizer.pad_token_id or 0)

    # Embeddings
    writer.add_tensor("token_embd.weight", f32(sd["embeddings.word_embeddings.weight"]))
    writer.add_tensor("position_embd.weight", f32(sd["embeddings.position_embeddings.weight"]))
    if "embeddings.token_type_embeddings.weight" in sd:
        writer.add_tensor("token_type_embd.weight", f32(sd["embeddings.token_type_embeddings.weight"]))
    writer.add_tensor("embd_ln.weight", f32(sd["embeddings.LayerNorm.weight"]))
    writer.add_tensor("embd_ln.bias", f32(sd["embeddings.LayerNorm.bias"]))
    print("  embeddings: ok")

    # Encoder layers
    for i in range(config.num_hidden_layers):
        pfx = f"encoder.layer.{i}"

        # Pre-attention LN (BERT uses post-LN but we normalize it)
        writer.add_tensor(f"enc.{i}.ln1.weight", f32(sd[f"{pfx}.attention.output.LayerNorm.weight"]))
        writer.add_tensor(f"enc.{i}.ln1.bias", f32(sd[f"{pfx}.attention.output.LayerNorm.bias"]))

        # Attention
        for proj, hf_name in [("q", "query"), ("k", "key"), ("v", "value")]:
            writer.add_tensor(f"enc.{i}.attn.{proj}.weight", wt(sd[f"{pfx}.attention.self.{hf_name}.weight"]))
            writer.add_tensor(f"enc.{i}.attn.{proj}.bias", f32(sd[f"{pfx}.attention.self.{hf_name}.bias"]))
        writer.add_tensor(f"enc.{i}.attn.o.weight", wt(sd[f"{pfx}.attention.output.dense.weight"]))
        writer.add_tensor(f"enc.{i}.attn.o.bias", f32(sd[f"{pfx}.attention.output.dense.bias"]))

        # Post-FFN LN
        writer.add_tensor(f"enc.{i}.ln2.weight", f32(sd[f"{pfx}.output.LayerNorm.weight"]))
        writer.add_tensor(f"enc.{i}.ln2.bias", f32(sd[f"{pfx}.output.LayerNorm.bias"]))

        # FFN
        writer.add_tensor(f"enc.{i}.ffn.fc1.weight", wt(sd[f"{pfx}.intermediate.dense.weight"]))
        writer.add_tensor(f"enc.{i}.ffn.fc1.bias", f32(sd[f"{pfx}.intermediate.dense.bias"]))
        writer.add_tensor(f"enc.{i}.ffn.fc2.weight", wt(sd[f"{pfx}.output.dense.weight"]))
        writer.add_tensor(f"enc.{i}.ffn.fc2.bias", f32(sd[f"{pfx}.output.dense.bias"]))

        print(f"  enc.{i}: ok")

    # Pooler (optional)
    if "pooler.dense.weight" in sd:
        writer.add_tensor("pooler.weight", f32(sd["pooler.dense.weight"]))
        writer.add_tensor("pooler.bias", f32(sd["pooler.dense.bias"]))
        print("  pooler: ok")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    import os
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\nWrote {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
