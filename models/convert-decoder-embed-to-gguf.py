#!/usr/bin/env python3
"""Convert decoder-style embedding models (Qwen3/Gemma3) to GGUF.

Supports: Qwen3-Embedding, Octen-Embedding, F2LLM-v2, Jina v5, Harrier.
These models use causal transformer decoders with last-token pooling.

    python convert-decoder-embed-to-gguf.py \
        --model Qwen/Qwen3-Embedding-0.6B \
        --output qwen3-embed-0.6b.gguf

Use --ollama (default) for Ollama-compatible output, --crisp for CrispEmbed-native.
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
    fmt_group = parser.add_mutually_exclusive_group()
    fmt_group.add_argument("--ollama", action="store_true", default=True,
                           help="Ollama-compatible naming (default)")
    fmt_group.add_argument("--crisp", action="store_true",
                           help="CrispEmbed-native naming")
    args = parser.parse_args()

    ollama_mode = not args.crisp

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

    # Detect and merge LoRA adapters (e.g. Jina v5 task-specific adapters)
    has_lora = any("lora_A" in k or "base_layer" in k for k in model.state_dict())
    if has_lora:
        try:
            # Select retrieval adapter if available (most common use case)
            if hasattr(model, 'active_adapters'):
                adapters = list(getattr(model, 'peft_config', {}).keys())
                target = "retrieval" if "retrieval" in adapters else (adapters[0] if adapters else None)
                if target and hasattr(model, 'set_adapter'):
                    model.set_adapter(target)
                    print(f"  LoRA: selected adapter '{target}' from {adapters}")
            model = model.merge_and_unload()
            print(f"  LoRA: merged ({len(model.state_dict())} weights)")
        except Exception as e:
            print(f"  WARNING: LoRA merge failed ({e}), using raw weights")

    sd = model.state_dict()

    print(f"Architecture: {config.architectures}")
    print(f"Hidden: {config.hidden_size}, Layers: {config.num_hidden_layers}, "
          f"Heads: {config.num_attention_heads}, Vocab: {config.vocab_size}")

    # Detect architecture: qwen3 vs gemma3
    is_gemma = "gemma" in config.model_type.lower()
    n_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

    # Rope theta — check multiple locations
    rope_theta = getattr(config, "rope_theta", None)
    if rope_theta is None:
        rp = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None)
        if isinstance(rp, dict):
            rope_theta = rp.get("rope_theta", None)
            if rope_theta is None and "full_attention" in rp:
                rope_theta = rp["full_attention"].get("rope_theta", 10000.0)
            if rope_theta is None:
                rope_theta = 10000.0
        else:
            rope_theta = 10000.0
    print(f"  rope_theta: {rope_theta}")

    # Hidden activation
    act = getattr(config, "hidden_act", getattr(config, "hidden_activation", "silu"))
    act_str = str(act).lower()
    if "gelu_pytorch_tanh" in act_str:
        act_id = 2
    elif "gelu" in act_str:
        act_id = 1
    else:
        act_id = 0
    act_names = {0: "silu", 1: "gelu", 2: "gelu_pytorch_tanh"}
    print(f"  activation: {act_names[act_id]} (config: {act})")

    # Gemma3-specific features
    gemma_norm = is_gemma
    qpas = getattr(config, "query_pre_attn_scalar", 0)
    embed_scale = 1.0
    if is_gemma:
        embed_scale = float(config.hidden_size ** 0.5)
        try:
            m = model
            if hasattr(m, 'model'):
                m = m.model
            if hasattr(m, 'embed_tokens') and hasattr(m.embed_tokens, 'embed_scale'):
                embed_scale = float(m.embed_tokens.embed_scale)
        except Exception:
            pass

    # Detect if bidirectional
    is_bidirectional = "bert" in config.model_type.lower() or "encoder" in str(config.architectures).lower()

    if ollama_mode:
        # Ollama arch: "qwen3" or "gemma3"
        arch = "gemma3" if is_gemma else "qwen3"
        # Ollama pooling: 3 = Last token
        pool_ollama = 3
    else:
        arch = ARCH

    writer = gguf.GGUFWriter(str(args.output), arch=arch)

    def add_tensor(name, data):
        """Add tensor, handling Q8_0 quantized data."""
        if isinstance(data, Q8Tensor):
            shape = data.shape
            row_width = shape[-1]
            row_bytes = (row_width // 32) * 34
            byte_shape = list(shape[:-1]) + [row_bytes]
            writer.add_tensor(name, data.data,
                              raw_shape=byte_shape,
                              raw_dtype=gguf.GGMLQuantizationType.Q8_0)
        else:
            writer.add_tensor(name, data)

    if ollama_mode:
        # Ollama-compatible metadata: {arch}.key_name
        writer.add_uint32(f"{arch}.embedding_length", config.hidden_size)
        writer.add_uint32(f"{arch}.block_count", config.num_hidden_layers)
        writer.add_uint32(f"{arch}.attention.head_count", config.num_attention_heads)
        writer.add_uint32(f"{arch}.attention.head_count_kv", n_kv_heads)
        writer.add_uint32(f"{arch}.attention.key_length", head_dim)
        writer.add_uint32(f"{arch}.attention.value_length", head_dim)
        writer.add_uint32(f"{arch}.feed_forward_length", config.intermediate_size)
        writer.add_float32(f"{arch}.attention.layer_norm_rms_epsilon",
                           getattr(config, "rms_norm_eps", 1e-6))
        writer.add_float32(f"{arch}.rope.freq_base", float(rope_theta))
        writer.add_uint32(f"{arch}.context_length",
                          getattr(config, "max_position_embeddings", 8192))
        writer.add_uint32(f"{arch}.pooling_type", pool_ollama)
        writer.add_bool(f"{arch}.normalize_embeddings", True)
        if qpas:
            writer.add_float32(f"{arch}.attention.key_length_scale", float(qpas))
        print(f"  format: Ollama (arch={arch})")
    else:
        # CrispEmbed-native metadata
        writer.add_uint32("decoder.vocab_size", config.vocab_size)
        writer.add_uint32("decoder.hidden_size", config.hidden_size)
        writer.add_uint32("decoder.num_hidden_layers", config.num_hidden_layers)
        writer.add_uint32("decoder.num_attention_heads", config.num_attention_heads)
        writer.add_uint32("decoder.num_key_value_heads", n_kv_heads)
        writer.add_uint32("decoder.intermediate_size", config.intermediate_size)
        writer.add_uint32("decoder.head_dim", head_dim)
        writer.add_uint32("decoder.max_position_embeddings",
                          getattr(config, "max_position_embeddings", 8192))
        writer.add_float32("decoder.rms_norm_eps",
                           getattr(config, "rms_norm_eps", 1e-6))
        writer.add_float32("decoder.rope_theta", float(rope_theta))
        writer.add_uint32("decoder.pooling_method", 2)  # last-token
        writer.add_uint32("decoder.activation", act_id)
        if qpas:
            writer.add_float32("decoder.attn_scale", float(qpas))
        if embed_scale != 1.0:
            writer.add_float32("decoder.embed_scale", embed_scale)
        writer.add_uint32("decoder.gemma_norm", int(gemma_norm))
        writer.add_uint32("decoder.is_bidirectional", int(is_bidirectional))
        print(f"  format: CrispEmbed")

    # Tokenizer
    vocab = tokenizer.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}
    tokens = [id_to_token.get(i, f"<unk_{i}>") for i in range(config.vocab_size)]
    writer.add_array("tokenizer.ggml.tokens", tokens)
    if ollama_mode:
        # Gemma3 uses SentencePiece BPE ("llama"); Qwen3 uses GPT-2 BPE ("gpt2")
        tok_model = "llama" if is_gemma else "gpt2"
        writer.add_string("tokenizer.ggml.model", tok_model)
    else:
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

    if ollama_mode:
        writer.add_bool("tokenizer.ggml.add_bos_token", True)
        writer.add_bool("tokenizer.ggml.add_eos_token", False)
        # Token types for Ollama
        token_types = []
        for i in range(config.vocab_size):
            tok = id_to_token.get(i, "")
            if tok.startswith("<") and tok.endswith(">"):
                token_types.append(3)  # control
            elif i == (tokenizer.unk_token_id or 0):
                token_types.append(2)  # unknown
            else:
                token_types.append(1)  # normal
        writer.add_array("tokenizer.ggml.token_type", token_types)

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
        # Gemma3 SentencePiece needs scores for Ollama
        if ollama_mode:
            try:
                from huggingface_hub import hf_hub_download
                tok_json_path = hf_hub_download(repo_id=args.model, filename="tokenizer.json")
                with open(tok_json_path) as f:
                    tok_json = json.load(f)
                tj_vocab = tok_json.get("model", {}).get("vocab", {})
                if isinstance(tj_vocab, dict):
                    # BPE vocab is dict: {token: score}
                    scores = [0.0] * config.vocab_size
                    for tok_str, score in tj_vocab.items():
                        tid = vocab.get(tok_str, -1)
                        if 0 <= tid < config.vocab_size:
                            scores[tid] = float(score)
                    writer.add_array("tokenizer.ggml.scores", scores)
                    print(f"  scores: loaded from tokenizer.json (dict)")
                elif isinstance(tj_vocab, list) and tj_vocab and isinstance(tj_vocab[0], list):
                    # Unigram vocab is list: [[token, score], ...]
                    scores = [0.0] * config.vocab_size
                    for i2, (tok_str, score) in enumerate(tj_vocab):
                        if i2 < config.vocab_size:
                            scores[i2] = float(score)
                    writer.add_array("tokenizer.ggml.scores", scores)
                    print(f"  scores: loaded from tokenizer.json (list)")
            except Exception as e:
                print(f"  scores: not loaded ({e})")
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

    # Gemma3 RMSNorm uses (1 + weight). In Ollama mode, pre-bake the +1
    # since Ollama's RMSNorm doesn't handle the offset.
    # In CrispEmbed mode, store raw weights (runtime adds +1 via ones tensor).
    def norm_weight(t):
        """Convert norm weight tensor: add +1 for Gemma3 in Ollama mode."""
        data = f32(t)
        if ollama_mode and is_gemma:
            data = data + 1.0
        return data

    # Layer prefix for output tensors: "blk" for Ollama, "dec" for CrispEmbed
    LP = "blk" if ollama_mode else "dec"

    # Tensor name maps for attention projections
    if ollama_mode:
        # Ollama: blk.N.attn_q, blk.N.attn_output, blk.N.attn_q_norm
        ATTN_MAP = {"q": "attn_q", "k": "attn_k", "v": "attn_v", "o": "attn_output"}
        NORM_MAP = {"q_norm": "attn_q_norm", "k_norm": "attn_k_norm"}
        FFN_MAP = {"gate": "ffn_gate", "up": "ffn_up", "down": "ffn_down"}
    else:
        # CrispEmbed: dec.N.attn.q, dec.N.attn.o, dec.N.attn.q_norm
        ATTN_MAP = {"q": "attn.q", "k": "attn.k", "v": "attn.v", "o": "attn.o"}
        NORM_MAP = {"q_norm": "attn.q_norm", "k_norm": "attn.k_norm"}
        FFN_MAP = {"gate": "ffn.gate", "up": "ffn.up", "down": "ffn.down"}

    # Decoder layers
    for i in range(config.num_hidden_layers):
        pfx = f"{layer_prefix}.{i}" if layer_prefix else f"layers.{i}"

        has_layer = any(k.startswith(pfx + ".") for k in sd)
        if not has_layer:
            print(f"  WARNING: layer {i} not found (prefix: {pfx})")
            continue

        # RMSNorm / LayerNorm (pre-attention)
        for norm_key in [f"{pfx}.input_layernorm.weight",
                          f"{pfx}.attention.output.LayerNorm.weight"]:
            if norm_key in sd:
                add_tensor(f"{LP}.{i}.attn_norm.weight", norm_weight(sd[norm_key]))
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
                    add_tensor(f"{LP}.{i}.{ATTN_MAP[proj]}.weight", wt(sd[wkey]))
                    bkey = f"{pfx}.{n}.bias"
                    if bkey in sd:
                        add_tensor(f"{LP}.{i}.{ATTN_MAP[proj]}.bias", f32(sd[bkey]))
                    break

        # QK norm (Qwen3 feature)
        for norm_name, out_name in [("self_attn.q_norm", "q_norm"),
                                     ("self_attn.k_norm", "k_norm")]:
            nkey = f"{pfx}.{norm_name}.weight"
            if nkey in sd:
                add_tensor(f"{LP}.{i}.{NORM_MAP[out_name]}.weight", norm_weight(sd[nkey]))

        # Post-attention / pre-FFN norms
        # Gemma3 has 4 norms: attn_norm, post_attention_norm, ffn_norm, post_ffw_norm
        # Qwen3 has 2 norms: attn_norm, ffn_norm (post_attention_layernorm IS the pre-FFN norm)
        has_pre_ffn = f"{pfx}.pre_feedforward_layernorm.weight" in sd

        if has_pre_ffn:
            # Gemma3-style: 4 norms per layer
            post_attn_key = f"{pfx}.post_attention_layernorm.weight"
            if post_attn_key in sd:
                out_name = "post_attention_norm" if ollama_mode else "ffn_norm"
                add_tensor(f"{LP}.{i}.{out_name}.weight", norm_weight(sd[post_attn_key]))

            pre_ffn_key = f"{pfx}.pre_feedforward_layernorm.weight"
            out_name = "ffn_norm" if ollama_mode else "pre_ffn_norm"
            add_tensor(f"{LP}.{i}.{out_name}.weight", norm_weight(sd[pre_ffn_key]))

            post_ffn_key = f"{pfx}.post_feedforward_layernorm.weight"
            if post_ffn_key in sd:
                out_name = "post_ffw_norm" if ollama_mode else "post_ffn_norm"
                add_tensor(f"{LP}.{i}.{out_name}.weight", norm_weight(sd[post_ffn_key]))
        else:
            # Qwen3-style: 2 norms per layer
            for norm_key in [f"{pfx}.post_attention_layernorm.weight",
                              f"{pfx}.output.LayerNorm.weight"]:
                if norm_key in sd:
                    add_tensor(f"{LP}.{i}.ffn_norm.weight", norm_weight(sd[norm_key]))
                    break

        # FFN (SwiGLU: gate + up + down, or standard: fc1 + fc2)
        gate_key = f"{pfx}.mlp.gate_proj.weight"
        if gate_key in sd:
            add_tensor(f"{LP}.{i}.{FFN_MAP['gate']}.weight", wt(sd[gate_key]))
            add_tensor(f"{LP}.{i}.{FFN_MAP['up']}.weight", wt(sd[f"{pfx}.mlp.up_proj.weight"]))
            add_tensor(f"{LP}.{i}.{FFN_MAP['down']}.weight", wt(sd[f"{pfx}.mlp.down_proj.weight"]))
        else:
            fc1_key = f"{pfx}.intermediate.dense.weight"
            if fc1_key in sd:
                add_tensor(f"{LP}.{i}.{FFN_MAP.get('fc1', 'ffn.fc1')}.weight", wt(sd[fc1_key]))
                add_tensor(f"{LP}.{i}.{FFN_MAP.get('fc1', 'ffn.fc1')}.bias", f32(sd[f"{pfx}.intermediate.dense.bias"]))
                add_tensor(f"{LP}.{i}.{FFN_MAP.get('fc2', 'ffn.fc2')}.weight", wt(sd[f"{pfx}.output.dense.weight"]))
                add_tensor(f"{LP}.{i}.{FFN_MAP.get('fc2', 'ffn.fc2')}.bias", f32(sd[f"{pfx}.output.dense.bias"]))

        print(f"  {LP}.{i}: ok")

    # Final norm
    for key in ["model.norm.weight", "norm.weight", "encoder.layer_norm.weight"]:
        if key in sd:
            add_tensor("output_norm.weight", norm_weight(sd[key]))
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
