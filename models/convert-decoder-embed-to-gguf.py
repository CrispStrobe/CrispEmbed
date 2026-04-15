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


class Q8Tensor:
    """Wrapper for Q8_0 quantized tensor data + original shape."""
    def __init__(self, data, shape):
        self.data = data      # quantized bytes (numpy uint8 array)
        self.shape = shape    # original f32 shape


def q8_0(t):
    """Quantize tensor to Q8_0 (block size 32)."""
    data = t.detach().float().cpu().numpy().astype(np.float32)
    if data.ndim < 2 or data.shape[-1] % 32 != 0:
        return data  # keep f32 for norms/biases or non-aligned
    try:
        q = gguf.quantize(data, gguf.GGMLQuantizationType.Q8_0)
        return Q8Tensor(q, data.shape)
    except Exception:
        return data  # fallback to f32


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", choices=["f16", "f32", "q8_0"], default="f32")
    args = parser.parse_args()

    if args.dtype == "q8_0":
        wt = q8_0
    elif args.dtype == "f16":
        wt = f16
    else:
        wt = f32

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

    def add_tensor(name, data):
        """Add tensor, handling Q8_0 quantized data."""
        if isinstance(data, Q8Tensor):
            # raw_shape must be the byte-level shape, not the logical shape
            # For Q8_0: each block of 32 f32 = 34 bytes (32 int8 + 2 byte scale)
            # So row_bytes = (row_width / 32) * 34
            shape = data.shape
            row_width = shape[-1]
            row_bytes = (row_width // 32) * 34
            byte_shape = list(shape[:-1]) + [row_bytes]
            writer.add_tensor(name, data.data,
                              raw_shape=byte_shape,
                              raw_dtype=gguf.GGMLQuantizationType.Q8_0)
        else:
            writer.add_tensor(name, data)

    # Metadata
    writer.add_uint32("decoder.vocab_size", config.vocab_size)
    writer.add_uint32("decoder.hidden_size", config.hidden_size)
    writer.add_uint32("decoder.num_hidden_layers", config.num_hidden_layers)
    writer.add_uint32("decoder.num_attention_heads", config.num_attention_heads)
    n_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    writer.add_uint32("decoder.num_key_value_heads", n_kv_heads)
    writer.add_uint32("decoder.intermediate_size", config.intermediate_size)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    writer.add_uint32("decoder.head_dim", head_dim)
    writer.add_uint32("decoder.max_position_embeddings",
                       getattr(config, "max_position_embeddings", 8192))
    writer.add_float32("decoder.rms_norm_eps",
                        getattr(config, "rms_norm_eps", 1e-6))
    # Rope theta — check multiple locations
    rope_theta = getattr(config, "rope_theta", None)
    if rope_theta is None:
        rp = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None)
        if isinstance(rp, dict):
            rope_theta = rp.get("rope_theta", None)
            # Gemma3 has nested rope configs — try full_attention.rope_theta
            if rope_theta is None and "full_attention" in rp:
                rope_theta = rp["full_attention"].get("rope_theta", 10000.0)
            if rope_theta is None:
                rope_theta = 10000.0
        else:
            rope_theta = 10000.0
    writer.add_float32("decoder.rope_theta", float(rope_theta))
    print(f"  rope_theta: {rope_theta}")
    writer.add_uint32("decoder.pooling_method", 2)  # last-token

    # Hidden activation: silu (SwiGLU) vs gelu (GeGLU) vs gelu_pytorch_tanh
    act = getattr(config, "hidden_act", getattr(config, "hidden_activation", "silu"))
    act_str = str(act).lower()
    if "gelu_pytorch_tanh" in act_str:
        act_id = 2  # gelu_pytorch_tanh (Gemma3)
    elif "gelu" in act_str:
        act_id = 1  # gelu (GeGLU)
    else:
        act_id = 0  # silu (SwiGLU)
    writer.add_uint32("decoder.activation", act_id)
    act_names = {0: "silu", 1: "gelu", 2: "gelu_pytorch_tanh"}
    print(f"  activation: {act_names[act_id]} (config: {act})")

    # Gemma3-specific: query_pre_attn_scalar (attention scale)
    qpas = getattr(config, "query_pre_attn_scalar", 0)
    if qpas:
        writer.add_float32("decoder.attn_scale", float(qpas))
        print(f"  attn_scale: {qpas}")

    # Embedding scale: Gemma3 multiplies token embeddings by sqrt(hidden_size)
    # Detect from model class: if it passes embed_scale to the embedding layer
    embed_scale = 1.0
    try:
        m = model
        if hasattr(m, 'model'):
            m = m.model
        if hasattr(m, 'embed_tokens') and hasattr(m.embed_tokens, 'embed_scale'):
            embed_scale = float(m.embed_tokens.embed_scale)
        elif "gemma" in config.model_type.lower():
            embed_scale = float(config.hidden_size ** 0.5)
    except:
        pass
    if embed_scale != 1.0:
        writer.add_float32("decoder.embed_scale", embed_scale)
        print(f"  embed_scale: {embed_scale:.4f}")

    # Gemma-style RMSNorm: uses (1 + weight) instead of weight
    gemma_norm = "gemma" in config.model_type.lower()
    writer.add_uint32("decoder.gemma_norm", int(gemma_norm))
    if gemma_norm:
        print(f"  gemma_norm: true (RMSNorm uses 1+weight)")

    # Detect if bidirectional (encoder) vs causal (decoder)
    # EuroBERT/ModernBERT are encoder models with decoder-style weights
    is_bidirectional = "bert" in config.model_type.lower() or "encoder" in str(config.architectures).lower()
    writer.add_uint32("decoder.is_bidirectional", int(is_bidirectional))
    if is_bidirectional:
        print(f"  attention: bidirectional (no causal mask)")

    # Tokenizer
    vocab = tokenizer.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}
    tokens = [id_to_token.get(i, f"<unk_{i}>") for i in range(config.vocab_size)]
    writer.add_array("tokenizer.ggml.tokens", tokens)
    writer.add_uint32("tokenizer.ggml.type", 1)  # BPE

    # Store BPE merges if available
    try:
        from huggingface_hub import hf_hub_download
        tok_json_path = hf_hub_download(repo_id=args.model, filename="tokenizer.json")
        with open(tok_json_path) as f:
            tok_json = json.load(f)
        raw_merges = tok_json.get("model", {}).get("merges", [])
        if raw_merges:
            # Merges can be list[str] ("a b") or list[list[str]] (["a", "b"])
            merges = []
            for m in raw_merges:
                if isinstance(m, list):
                    merges.append(" ".join(m))
                else:
                    merges.append(str(m))
            writer.add_array("tokenizer.ggml.merges", merges)
            print(f"  merges: {len(merges)}")
    except Exception as e:
        print(f"  merges: not found ({e})")
    if tokenizer.bos_token_id is not None:
        writer.add_uint32("tokenizer.ggml.bos_token_id", tokenizer.bos_token_id)

    # Detect SentencePiece-style BPE (Gemma) vs GPT-2-style BPE (Qwen3)
    # SentencePiece BPE uses ▁ as space marker; GPT-2 uses byte-level encoding
    is_spm_bpe = False
    vocab_dict = tokenizer.get_vocab()
    for token_str in ["▁the", "▁a", "▁world"]:
        if token_str in vocab_dict:
            is_spm_bpe = True
            break
    # Also check: Gemma tokenizers have ▁ tokens in the vocab
    spm_count = sum(1 for t in vocab_dict if t.startswith("▁"))
    if spm_count > 1000:
        is_spm_bpe = True
    writer.add_uint32("tokenizer.ggml.is_spm_bpe", int(is_spm_bpe))
    if is_spm_bpe:
        print(f"  tokenizer_style: SentencePiece BPE ({spm_count} ▁-prefixed tokens)")
    if tokenizer.eos_token_id is not None:
        eos = tokenizer.eos_token_id
        if isinstance(eos, list):
            eos = eos[0]
        writer.add_uint32("tokenizer.ggml.eos_token_id", eos)
    if tokenizer.pad_token_id is not None:
        writer.add_uint32("tokenizer.ggml.padding_token_id", tokenizer.pad_token_id)

    # Detect suffix token: what gets appended after the text
    # Compare encode("a") with just tokenizing "a" to see if tokenizer adds a suffix
    test_ids = tokenizer.encode("a")
    raw_ids = tokenizer.convert_tokens_to_ids(["a"])
    if len(test_ids) > len(raw_ids) and test_ids[-1] != raw_ids[-1]:
        suffix_id = test_ids[-1]
        print(f"  suffix_token_id: {suffix_id} (appended by tokenizer)")
    else:
        suffix_id = -1  # no suffix
        print(f"  suffix_token_id: none (tokenizer does not append special tokens)")
    writer.add_int32("tokenizer.ggml.suffix_token_id", suffix_id)

    # Token embeddings — search multiple naming conventions
    embd_keys = ["model.embed_tokens.weight", "embed_tokens.weight",
                  "embeddings.word_embeddings.weight"]
    for key in embd_keys:
        if key in sd:
            add_tensor("token_embd.weight", f32(sd[key]))
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
                add_tensor(f"dec.{i}.attn_norm.weight", f32(sd[norm_key]))
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
                    add_tensor(f"dec.{i}.attn.{proj}.weight", wt(sd[wkey]))
                    bkey = f"{pfx}.{n}.bias"
                    if bkey in sd:
                        add_tensor(f"dec.{i}.attn.{proj}.bias", f32(sd[bkey]))
                    break

        # QK norm (Qwen3 feature)
        for norm_name, out_name in [("self_attn.q_norm", "q_norm"),
                                     ("self_attn.k_norm", "k_norm")]:
            nkey = f"{pfx}.{norm_name}.weight"
            if nkey in sd:
                add_tensor(f"dec.{i}.attn.{out_name}.weight", f32(sd[nkey]))

        # Post-attention norm
        for norm_key in [f"{pfx}.post_attention_layernorm.weight",
                          f"{pfx}.output.LayerNorm.weight"]:
            if norm_key in sd:
                add_tensor(f"dec.{i}.ffn_norm.weight", f32(sd[norm_key]))
                break

        # Gemma3 extra norms: pre/post feedforward layernorms
        pre_ffn_key = f"{pfx}.pre_feedforward_layernorm.weight"
        if pre_ffn_key in sd:
            add_tensor(f"dec.{i}.pre_ffn_norm.weight", f32(sd[pre_ffn_key]))
        post_ffn_key = f"{pfx}.post_feedforward_layernorm.weight"
        if post_ffn_key in sd:
            add_tensor(f"dec.{i}.post_ffn_norm.weight", f32(sd[post_ffn_key]))

        # FFN (SwiGLU: gate + up + down, or standard: fc1 + fc2)
        gate_key = f"{pfx}.mlp.gate_proj.weight"
        if gate_key in sd:
            add_tensor(f"dec.{i}.ffn.gate.weight", wt(sd[gate_key]))
            add_tensor(f"dec.{i}.ffn.up.weight", wt(sd[f"{pfx}.mlp.up_proj.weight"]))
            add_tensor(f"dec.{i}.ffn.down.weight", wt(sd[f"{pfx}.mlp.down_proj.weight"]))
        else:
            fc1_key = f"{pfx}.intermediate.dense.weight"
            if fc1_key in sd:
                add_tensor(f"dec.{i}.ffn.fc1.weight", wt(sd[fc1_key]))
                add_tensor(f"dec.{i}.ffn.fc1.bias", f32(sd[f"{pfx}.intermediate.dense.bias"]))
                add_tensor(f"dec.{i}.ffn.fc2.weight", wt(sd[f"{pfx}.output.dense.weight"]))
                add_tensor(f"dec.{i}.ffn.fc2.bias", f32(sd[f"{pfx}.output.dense.bias"]))

        print(f"  dec.{i}: ok")

    # Final norm
    for key in ["model.norm.weight", "norm.weight", "encoder.layer_norm.weight"]:
        if key in sd:
            add_tensor("output_norm.weight", f32(sd[key]))
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
