#!/usr/bin/env python3
"""Convert decoder-style embedding models (Qwen3/LLaMA) to GGUF.

Supports: Qwen3-Embedding, Octen-Embedding, F2LLM-v2, Jina v5, Harrier.
These models use causal transformer decoders with last-token pooling.

    python convert-decoder-embed-to-gguf.py \
        --model Alibaba-NLP/gte-Qwen2-1.5B-instruct \
        --output gte-qwen2-1.5b.gguf
"""

import argparse
import json
import sys
from pathlib import Path

import gguf
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig


ARCH = "decoder_embed"


def f32(t):
    return t.detach().float().cpu().numpy().astype(np.float32)


def f16(t):
    return t.detach().float().cpu().numpy().astype(np.float16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", choices=["f16", "f32"], default="f32")
    args = parser.parse_args()

    wt = f32 if args.dtype == "f32" else f16

    print(f"Loading: {args.model}")
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model.eval()
    sd = model.state_dict()

    print(f"Architecture: {config.architectures}")
    print(f"Hidden: {config.hidden_size}, Layers: {config.num_hidden_layers}, "
          f"Heads: {config.num_attention_heads}, Vocab: {config.vocab_size}")

    writer = gguf.GGUFWriter(str(args.output), arch=ARCH)

    # Metadata
    writer.add_uint32("decoder.vocab_size", config.vocab_size)
    writer.add_uint32("decoder.hidden_size", config.hidden_size)
    writer.add_uint32("decoder.num_hidden_layers", config.num_hidden_layers)
    writer.add_uint32("decoder.num_attention_heads", config.num_attention_heads)
    n_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    writer.add_uint32("decoder.num_key_value_heads", n_kv_heads)
    writer.add_uint32("decoder.intermediate_size", config.intermediate_size)
    writer.add_uint32("decoder.max_position_embeddings",
                       getattr(config, "max_position_embeddings", 8192))
    writer.add_float32("decoder.rms_norm_eps",
                        getattr(config, "rms_norm_eps", 1e-6))
    rope_theta = getattr(config, "rope_theta", 10000.0)
    writer.add_float32("decoder.rope_theta", rope_theta)
    writer.add_uint32("decoder.pooling_method", 2)  # last-token

    # Tokenizer
    vocab = tokenizer.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}
    tokens = [id_to_token.get(i, f"<unk_{i}>") for i in range(config.vocab_size)]
    writer.add_array("tokenizer.ggml.tokens", tokens)
    writer.add_uint32("tokenizer.ggml.type", 1)  # BPE
    if tokenizer.bos_token_id is not None:
        writer.add_uint32("tokenizer.ggml.bos_token_id", tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        eos = tokenizer.eos_token_id
        if isinstance(eos, list):
            eos = eos[0]
        writer.add_uint32("tokenizer.ggml.eos_token_id", eos)
    if tokenizer.pad_token_id is not None:
        writer.add_uint32("tokenizer.ggml.padding_token_id", tokenizer.pad_token_id)

    # Token embeddings — search multiple naming conventions
    embd_keys = ["model.embed_tokens.weight", "embed_tokens.weight",
                  "embeddings.word_embeddings.weight"]
    for key in embd_keys:
        if key in sd:
            writer.add_tensor("token_embd.weight", f32(sd[key]))
            print(f"  token_embd: {sd[key].shape}")
            break
    else:
        print("  WARNING: token_embd not found!")

    # Detect layer prefix — models vary: "model.layers.{i}" vs "layers.{i}"
    layer_prefix = None
    for candidate in ["model.layers", "layers", "encoder.layer"]:
        if f"{candidate}.0.self_attn.q_proj.weight" in sd:
            layer_prefix = candidate
            break
        if f"{candidate}.0.attention.self.query.weight" in sd:
            layer_prefix = candidate
            break
    if not layer_prefix:
        print("  WARNING: cannot detect layer naming convention")
        # Try to find any layer key
        for key in sd:
            if ".self_attn.q_proj.weight" in key:
                parts = key.split(".self_attn")[0]
                # e.g. "layers.0" → prefix is "layers"
                idx = parts.rfind(".")
                if idx >= 0:
                    layer_prefix = parts[:idx]
                    print(f"  Detected layer prefix: '{layer_prefix}'")
                break

    # Decoder layers
    for i in range(config.num_hidden_layers):
        pfx = f"{layer_prefix}.{i}" if layer_prefix else f"layers.{i}"

        # Check this layer exists
        has_layer = any(k.startswith(pfx + ".") for k in sd)
        if not has_layer:
            print(f"  WARNING: layer {i} not found (prefix: {pfx})")
            continue

        # RMSNorm / LayerNorm
        for norm_key in [f"{pfx}.input_layernorm.weight",
                          f"{pfx}.attention.output.LayerNorm.weight"]:
            if norm_key in sd:
                writer.add_tensor(f"dec.{i}.attn_norm.weight", f32(sd[norm_key]))
                break

        # Attention Q/K/V/O
        for proj, names in [
            ("q", ["self_attn.q_proj", "attention.self.query"]),
            ("k", ["self_attn.k_proj", "attention.self.key"]),
            ("v", ["self_attn.v_proj", "attention.self.value"]),
            ("o", ["self_attn.o_proj", "attention.output.dense"]),
        ]:
            for n in names:
                wkey = f"{pfx}.{n}.weight"
                if wkey in sd:
                    writer.add_tensor(f"dec.{i}.attn.{proj}.weight", wt(sd[wkey]))
                    bkey = f"{pfx}.{n}.bias"
                    if bkey in sd:
                        writer.add_tensor(f"dec.{i}.attn.{proj}.bias", f32(sd[bkey]))
                    break

        # QK norm (Qwen3 feature)
        for norm_name, out_name in [("self_attn.q_norm", "q_norm"),
                                     ("self_attn.k_norm", "k_norm")]:
            nkey = f"{pfx}.{norm_name}.weight"
            if nkey in sd:
                writer.add_tensor(f"dec.{i}.attn.{out_name}.weight", f32(sd[nkey]))

        # Post-attention norm
        for norm_key in [f"{pfx}.post_attention_layernorm.weight",
                          f"{pfx}.output.LayerNorm.weight"]:
            if norm_key in sd:
                writer.add_tensor(f"dec.{i}.ffn_norm.weight", f32(sd[norm_key]))
                break

        # FFN (SwiGLU: gate + up + down, or standard: fc1 + fc2)
        gate_key = f"{pfx}.mlp.gate_proj.weight"
        if gate_key in sd:
            writer.add_tensor(f"dec.{i}.ffn.gate.weight", wt(sd[gate_key]))
            writer.add_tensor(f"dec.{i}.ffn.up.weight", wt(sd[f"{pfx}.mlp.up_proj.weight"]))
            writer.add_tensor(f"dec.{i}.ffn.down.weight", wt(sd[f"{pfx}.mlp.down_proj.weight"]))
        else:
            fc1_key = f"{pfx}.intermediate.dense.weight"
            if fc1_key in sd:
                writer.add_tensor(f"dec.{i}.ffn.fc1.weight", wt(sd[fc1_key]))
                writer.add_tensor(f"dec.{i}.ffn.fc1.bias", f32(sd[f"{pfx}.intermediate.dense.bias"]))
                writer.add_tensor(f"dec.{i}.ffn.fc2.weight", wt(sd[f"{pfx}.output.dense.weight"]))
                writer.add_tensor(f"dec.{i}.ffn.fc2.bias", f32(sd[f"{pfx}.output.dense.bias"]))

        print(f"  dec.{i}: ok")

    # Final norm
    for key in ["model.norm.weight", "norm.weight", "encoder.layer_norm.weight"]:
        if key in sd:
            writer.add_tensor("output_norm.weight", f32(sd[key]))
            print(f"  output_norm: ok")
            break

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    import os
    print(f"\nWrote {args.output} ({os.path.getsize(args.output)/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
