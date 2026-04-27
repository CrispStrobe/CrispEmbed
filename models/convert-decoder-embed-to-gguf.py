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
    try:
        model = AutoModel.from_pretrained(args.model, trust_remote_code=True,
                                          use_safetensors=True)
    except Exception:
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

    # BidirLM-Omni: text settings live under config.text_config; non-text towers
    # (audio_tower.*, visual.*) are skipped by Phase 1 — text-only export.
    is_bidirlm_omni = (getattr(config, "model_type", "") == "bidirlm_omni"
                       or hasattr(config, "text_config") and getattr(config, "model_type", "").endswith("omni"))
    if is_bidirlm_omni and hasattr(config, "text_config"):
        text_config = config.text_config
        print(f"BidirLM-Omni detected — using config.text_config for text hyperparams")
        # Promote text fields to top-level access for the rest of the script
        for attr in ("vocab_size", "hidden_size", "num_hidden_layers",
                     "num_attention_heads", "num_key_value_heads", "head_dim",
                     "intermediate_size", "max_position_embeddings",
                     "rms_norm_eps", "rope_theta", "rope_scaling",
                     "hidden_act", "tie_word_embeddings", "attention_bias"):
            if hasattr(text_config, attr):
                setattr(config, attr, getattr(text_config, attr))

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

    # Detect if bidirectional. BidirLM-Omni is bidirectional by design (Qwen3
    # body with the causal mask removed — see modeling_bidirlm_omni.py L793,
    # `self.is_causal = False`).
    is_bidirectional = (
        "bert" in config.model_type.lower()
        or "encoder" in str(config.architectures).lower()
        or is_bidirlm_omni
    )

    # Pooling: 1 = mean (BidirLM uses sentence_transformers Pooling, mean by
    # default), 2 = last-token (Qwen3/Gemma3 default).
    pool_crisp = 1 if is_bidirlm_omni else 2
    # Ollama pooling enum: 0=None, 1=Mean, 2=CLS, 3=Last
    pool_ollama_default = 1 if is_bidirlm_omni else 3

    if ollama_mode:
        # Ollama arch: "qwen3" or "gemma3"
        arch = "gemma3" if is_gemma else "qwen3"
        pool_ollama = pool_ollama_default
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
        writer.add_uint32("decoder.pooling_method", pool_crisp)  # 1=mean, 2=last
        writer.add_uint32("decoder.activation", act_id)
        if qpas:
            writer.add_float32("decoder.attn_scale", float(qpas))
        if embed_scale != 1.0:
            writer.add_float32("decoder.embed_scale", embed_scale)
        writer.add_uint32("decoder.gemma_norm", int(gemma_norm))
        writer.add_uint32("decoder.is_bidirectional", int(is_bidirectional))
        # MRoPE / multimodal metadata (BidirLM-Omni). Absent or [0,0,0]
        # means the runtime falls back to standard RoPE (identical behavior
        # for text-only inputs since all 3 channels share the same position).
        rs = getattr(config, "rope_scaling", None) or {}
        if isinstance(rs, dict) and "mrope_section" in rs:
            secs = list(rs["mrope_section"])[:3]
            while len(secs) < 3: secs.append(0)
            writer.add_array("decoder.mrope_section", [int(x) for x in secs])
            print(f"  mrope_section: {secs}")
        # Multimodal token IDs (defaults from BidirLMOmniConfig).
        for tk in ("vision_start_token_id", "vision_end_token_id",
                   "image_token_id", "video_token_id",
                   "audio_token_id", "audio_start_token_id",
                   "audio_end_token_id"):
            v = getattr(config, tk, None)
            if v is not None:
                writer.add_uint32(f"decoder.{tk}", int(v))
        # Spatial merge size (mirror of vision_config.spatial_merge_size,
        # exported here for the decoder so encode_image_text can compute
        # n_image_tokens from grid_thw without loading vision_config).
        vc = getattr(config, "vision_config", None)
        if vc is not None and getattr(vc, "spatial_merge_size", None):
            writer.add_uint32("decoder.spatial_merge_size",
                              int(vc.spatial_merge_size))
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
                  "language_model.embed_tokens.weight",  # BidirLM-Omni
                  "embeddings.word_embeddings.weight"]
    for key in embd_keys:
        if key in sd:
            add_tensor("token_embd.weight", f32(sd[key]))
            print(f"  token_embd: {sd[key].shape}")
            break
    else:
        print("  WARNING: token_embd not found!")

    # Detect layer prefix — models vary: "model.layers.{i}" vs "layers.{i}"
    # vs "language_model.layers.{i}" (BidirLM-Omni)
    layer_prefix = None
    for candidate in ["model.layers", "language_model.layers", "layers", "encoder.layer"]:
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
    for key in ["model.norm.weight", "language_model.norm.weight",
                 "norm.weight", "encoder.layer_norm.weight"]:
        if key in sd:
            add_tensor("output_norm.weight", norm_weight(sd[key]))
            print(f"  output_norm: ok")
            break

    # ---------------------------------------------------------------------
    # BidirLM-Omni audio tower export (Phase 2 — crisp_audio integration).
    #
    # Writes audio_tower.* tensors with the names crisp_audio expects, plus
    # bidirlm.audio.* hparam metadata, plus the WhisperFeatureExtractor mel
    # filterbank + Hann window so crisp_audio can compute log-mel from raw
    # PCM at runtime without re-implementing the Slaney filterbank in C++.
    # ---------------------------------------------------------------------
    # Audio + vision exports run regardless of ollama_mode. The
    # bidirlm.{audio,vision}.* metadata + audio_tower.* / visual.* tensor
    # names don't collide with Ollama's schema, so the same GGUF works in
    # both runtimes (Ollama just ignores the extras). This is what enables
    # a single bidirlm-omni-2.5b-{f16,q8_0,...}.gguf to carry text + audio
    # + vision in one file.
    #
    # is_f32_only is defined here once so both the audio block (just below)
    # and the vision block (further down) can share it. Earlier the audio
    # block was gated on `not ollama_mode`, leaving is_f32_only undefined
    # in ollama mode — which broke the vision export when it tried to use
    # it. Hoisting it out fixes that and makes the rule apply uniformly.
    def is_f32_only(name):
        if name.endswith(".bias"):
            return True
        if name.endswith(".ln_post.weight") or name.endswith("ln_post.weight"):
            return True
        # *attn_norm.weight, *ffn_norm.weight, *_norm.weight,
        # plus visual.blk.*.norm{1,2}.weight (vision tower uses numbered
        # LayerNorms — must stay f32 or ggml_mul corrupts).
        if name.endswith("_norm.weight") or name.endswith("norm.weight"):
            return True
        if ".norm" in name and name.endswith(".weight"):
            return True
        return False

    if is_bidirlm_omni:
        # config.audio_config still present — we only promoted text_config
        # fields earlier, leaving audio_config untouched.
        ac = getattr(config, "audio_config", None)
        if ac is None:
            print("  audio: no audio_config — skipping")
        else:
            print("  audio: exporting BidirLM audio tower")
            ATPFX = "audio_tower."
            # Hparams (Phase 2 picks these up via bidirlm.audio.* meta_prefix).
            writer.add_uint32("bidirlm.audio.n_layers",       int(ac.encoder_layers))
            writer.add_uint32("bidirlm.audio.d_model",        int(ac.d_model))
            writer.add_uint32("bidirlm.audio.n_heads",        int(ac.encoder_attention_heads))
            writer.add_uint32("bidirlm.audio.head_dim",       int(ac.d_model // ac.encoder_attention_heads))
            writer.add_uint32("bidirlm.audio.ff_dim",         int(ac.encoder_ffn_dim))
            writer.add_uint32("bidirlm.audio.conv_channels",  int(ac.downsample_hidden_size))
            writer.add_uint32("bidirlm.audio.output_dim",     int(ac.output_dim))
            writer.add_uint32("bidirlm.audio.max_source_pos", int(ac.max_source_positions))
            writer.add_uint32("bidirlm.audio.n_window",       int(ac.n_window))
            writer.add_uint32("bidirlm.audio.n_window_infer", int(ac.n_window_infer))
            writer.add_uint32("bidirlm.audio.n_mels",         int(ac.num_mel_bins))
            # Whisper-v3 mel spec (matches preprocessor_config.json:
            # n_fft=400, hop_length=160, sampling_rate=16000)
            writer.add_uint32("bidirlm.audio.n_fft",       400)
            writer.add_uint32("bidirlm.audio.hop_length",  160)
            writer.add_uint32("bidirlm.audio.win_length",  400)
            writer.add_uint32("bidirlm.audio.sample_rate", 16000)
            # BidirLM uses block-diagonal windowed attention over groups of
            # (n_window_infer / (n_window*2)) post-cnn chunks; padding-frame
            # keys are masked off. Without this the encoder would treat
            # padding tokens and across-window tokens as fully attendable,
            # which diverges from HF's _prepare_attention_mask. See
            # crisp_audio/src/audio_tower.cpp for the mask construction.
            writer.add_uint32("bidirlm.audio.attn_window_mode", 1)

            # Tensor name remap (HF safetensors → crisp_audio GGUF suffix).
            # Pick weight-dtype helper: matmul weights go through `wt`
            # (--dtype-driven), but biases AND every kind of LayerNorm
            # scale/shift stay f32. Metal's binary-op kernels assert src
            # F32, and quantize.cpp skips norm-named tensors from later
            # K-quant requantize, so leaving any of these f16 would
            # propagate and either crash on Metal or silently lose
            # precision after the encoder body.
            #
            # The pattern below catches:
            #   *.bias            — every linear-layer bias
            #   *_norm.weight     — per-block attn_norm / ffn_norm
            #   *.ln_post.weight  — final LayerNorm scale (encoder output)
            # Conv kernels are 4D and go through `wt`; they happen to be
            # f16-or-f32 per --dtype but are only ever multiplied with F32
            # inputs (mel_batched), which Metal's conv2d kernel accepts.
            # is_f32_only is hoisted to the function scope above.

            def aw(name, hf_key):
                if hf_key not in sd:
                    return
                add_tensor(name, f32(sd[hf_key]) if is_f32_only(name) else wt(sd[hf_key]))

            audio_remap_static = {
                "audio_tower.conv2d1.weight": ATPFX + "conv.1.weight",
                "audio_tower.conv2d1.bias":   ATPFX + "conv.1.bias",
                "audio_tower.conv2d2.weight": ATPFX + "conv.2.weight",
                "audio_tower.conv2d2.bias":   ATPFX + "conv.2.bias",
                "audio_tower.conv2d3.weight": ATPFX + "conv.3.weight",
                "audio_tower.conv2d3.bias":   ATPFX + "conv.3.bias",
                "audio_tower.conv_out.weight": ATPFX + "conv_out.weight",
                # conv_out has bias=False in BidirLM — no entry.
                "audio_tower.ln_post.weight": ATPFX + "ln_post.weight",
                "audio_tower.ln_post.bias":   ATPFX + "ln_post.bias",
                "audio_tower.proj1.weight":   ATPFX + "proj1.weight",
                "audio_tower.proj1.bias":     ATPFX + "proj1.bias",
                "audio_tower.proj2.weight":   ATPFX + "proj2.weight",
                "audio_tower.proj2.bias":     ATPFX + "proj2.bias",
            }
            for hf, gg in audio_remap_static.items():
                aw(gg, hf)

            n_audio_layers = int(ac.encoder_layers)
            for i in range(n_audio_layers):
                p = f"audio_tower.layers.{i}."
                q = f"{ATPFX}blk.{i}."
                for hf_suf, gg_suf in [
                    ("self_attn_layer_norm.weight", "attn_norm.weight"),
                    ("self_attn_layer_norm.bias",   "attn_norm.bias"),
                    ("self_attn.q_proj.weight",     "attn_q.weight"),
                    ("self_attn.q_proj.bias",       "attn_q.bias"),
                    ("self_attn.k_proj.weight",     "attn_k.weight"),
                    ("self_attn.k_proj.bias",       "attn_k.bias"),
                    ("self_attn.v_proj.weight",     "attn_v.weight"),
                    ("self_attn.v_proj.bias",       "attn_v.bias"),
                    ("self_attn.out_proj.weight",   "attn_out.weight"),
                    ("self_attn.out_proj.bias",     "attn_out.bias"),
                    ("final_layer_norm.weight",     "ffn_norm.weight"),
                    ("final_layer_norm.bias",       "ffn_norm.bias"),
                    ("fc1.weight",                  "ffn_up.weight"),
                    ("fc1.bias",                    "ffn_up.bias"),
                    ("fc2.weight",                  "ffn_down.weight"),
                    ("fc2.bias",                    "ffn_down.bias"),
                ]:
                    aw(q + gg_suf, p + hf_suf)
            print(f"  audio: {n_audio_layers} encoder layers exported")

            # Mel filterbank + Hann window (baked from WhisperFeatureExtractor).
            try:
                from transformers import WhisperFeatureExtractor
                fe = WhisperFeatureExtractor.from_pretrained(args.model,
                                                             trust_remote_code=True)
                mel_filters = np.asarray(fe.mel_filters, dtype=np.float32)
                writer.add_tensor(ATPFX + "mel_filters", mel_filters)
                print(f"  audio.mel_filters: {mel_filters.shape}")
            except Exception as e:
                print(f"  audio.mel_filters: WhisperFeatureExtractor unavailable ({e})")
            n_fft_w = 400
            win = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n_fft_w) / n_fft_w)).astype(np.float32)
            writer.add_tensor(ATPFX + "mel_window", win)
            print(f"  audio.mel_window: {win.shape}")

    # ---------------------------------------------------------------------
    # BidirLM-Omni vision tower export (Phase 3 — bidirlm_vision integration).
    #
    # Writes visual.* tensors with crisp_vision's expected names, plus
    # bidirlm.vision.* hparam metadata. The tensor naming convention
    # mirrors the audio export: `visual.patch_embed.*`, `visual.pos_embed.*`,
    # `visual.blk.{i}.*`, `visual.merger.*`, `visual.deepstack.{0,1,2}.*`.
    # ---------------------------------------------------------------------
    # Vision tower export — runs regardless of ollama_mode. The bidirlm.vision.*
    # metadata + visual.* tensor names don't collide with Ollama's schema, so
    # the same GGUF works in both runtimes (Ollama just ignores the extras).
    if is_bidirlm_omni:
        vc = getattr(config, "vision_config", None)
        if vc is None:
            print("  vision: no vision_config — skipping")
        else:
            print("  vision: exporting BidirLM vision tower")
            VPFX = "visual."

            writer.add_uint32("bidirlm.vision.depth",            int(vc.depth))
            writer.add_uint32("bidirlm.vision.hidden_size",      int(vc.hidden_size))
            writer.add_uint32("bidirlm.vision.intermediate_size", int(vc.intermediate_size))
            writer.add_uint32("bidirlm.vision.num_heads",        int(vc.num_heads))
            writer.add_uint32("bidirlm.vision.in_channels",      int(vc.in_channels))
            writer.add_uint32("bidirlm.vision.patch_size",       int(vc.patch_size))
            writer.add_uint32("bidirlm.vision.spatial_merge_size", int(vc.spatial_merge_size))
            writer.add_uint32("bidirlm.vision.temporal_patch_size", int(vc.temporal_patch_size))
            writer.add_uint32("bidirlm.vision.out_hidden_size",  int(vc.out_hidden_size))
            writer.add_uint32("bidirlm.vision.num_position_embeddings", int(vc.num_position_embeddings))
            ds_idx = list(vc.deepstack_visual_indexes)
            writer.add_array("bidirlm.vision.deepstack_visual_indexes",
                              [int(x) for x in ds_idx])
            print(f"  vision: {vc.depth} layers, {vc.hidden_size}d, "
                  f"out={vc.out_hidden_size}d, deepstack={ds_idx}")

            # Image preprocessor scalars (Qwen2VLImageProcessorFast). These
            # come from preprocessor_config.json, not vision_config — but we
            # store them under bidirlm.vision.* alongside the tower metadata
            # so the C++ preprocessor (`image_preprocess.cpp`) can read them
            # without depending on `transformers`.
            try:
                from huggingface_hub import hf_hub_download
                pp_path = hf_hub_download(repo_id=args.model,
                                            filename="preprocessor_config.json")
                with open(pp_path) as fp:
                    pp_cfg = json.load(fp)
                pp_mean = list(pp_cfg.get("image_mean", [0.5, 0.5, 0.5]))[:3]
                pp_std  = list(pp_cfg.get("image_std",  [0.5, 0.5, 0.5]))[:3]
                pp_size = pp_cfg.get("size", {})
                pp_min  = int(pp_size.get("shortest_edge", 256 * 256))
                pp_max  = int(pp_size.get("longest_edge",  1024 * 1024))
                writer.add_array("bidirlm.vision.image_mean",
                                  [float(x) for x in pp_mean])
                writer.add_array("bidirlm.vision.image_std",
                                  [float(x) for x in pp_std])
                writer.add_uint32("bidirlm.vision.min_pixels", pp_min)
                writer.add_uint32("bidirlm.vision.max_pixels", pp_max)
                print(f"  vision: image_mean={pp_mean}, image_std={pp_std}, "
                      f"min_pixels={pp_min}, max_pixels={pp_max}")
            except Exception as e:
                print(f"  vision: preprocessor_config.json unavailable ({e}); "
                      f"runtime will use BidirLM-Omni defaults (mean=std=0.5).")

            def vw(name, hf_key):
                """Vision-tensor writer: same f32-only-for-norms-and-biases rule."""
                if hf_key not in sd:
                    return False
                add_tensor(name, f32(sd[hf_key]) if is_f32_only(name) else wt(sd[hf_key]))
                return True

            # Top-level tensors. patch_embed.proj.weight is a 5D Conv3d kernel
            # (out, in, T, H, W); GGUF/ggml support up to 4 dims, so we flatten
            # the kernel side into a single matmul axis: (out, in*T*H*W). The
            # runtime treats it as (in_flat, out) post-reshape anyway.
            pe_key = "visual.patch_embed.proj.weight"
            if pe_key in sd:
                pe_w = sd[pe_key]
                pe_w_flat = pe_w.reshape(pe_w.shape[0], -1).contiguous()
                add_tensor(VPFX + "patch_embed.weight", wt(pe_w_flat))
            vw(VPFX + "patch_embed.bias",   "visual.patch_embed.proj.bias")
            vw(VPFX + "pos_embed.weight",   "visual.pos_embed.weight")

            # ViT blocks
            for i in range(int(vc.depth)):
                p = f"visual.blocks.{i}."
                q = f"{VPFX}blk.{i}."
                for hf_suf, gg_suf in [
                    ("norm1.weight", "norm1.weight"),
                    ("norm1.bias",   "norm1.bias"),
                    ("norm2.weight", "norm2.weight"),
                    ("norm2.bias",   "norm2.bias"),
                    ("attn.qkv.weight",  "attn_qkv.weight"),
                    ("attn.qkv.bias",    "attn_qkv.bias"),
                    ("attn.proj.weight", "attn_proj.weight"),
                    ("attn.proj.bias",   "attn_proj.bias"),
                    ("mlp.linear_fc1.weight", "ffn_fc1.weight"),
                    ("mlp.linear_fc1.bias",   "ffn_fc1.bias"),
                    ("mlp.linear_fc2.weight", "ffn_fc2.weight"),
                    ("mlp.linear_fc2.bias",   "ffn_fc2.bias"),
                ]:
                    vw(q + gg_suf, p + hf_suf)

            # Final patch merger
            for hf_suf, gg_suf in [
                ("norm.weight",       "norm.weight"),
                ("norm.bias",         "norm.bias"),
                ("linear_fc1.weight", "fc1.weight"),
                ("linear_fc1.bias",   "fc1.bias"),
                ("linear_fc2.weight", "fc2.weight"),
                ("linear_fc2.bias",   "fc2.bias"),
            ]:
                vw(VPFX + "merger." + gg_suf, "visual.merger." + hf_suf)

            # DeepStack mergers (use_postshuffle_norm=True; norm is over hidden*4096)
            for i in range(len(ds_idx)):
                for hf_suf, gg_suf in [
                    ("norm.weight",       "norm.weight"),
                    ("norm.bias",         "norm.bias"),
                    ("linear_fc1.weight", "fc1.weight"),
                    ("linear_fc1.bias",   "fc1.bias"),
                    ("linear_fc2.weight", "fc2.weight"),
                    ("linear_fc2.bias",   "fc2.bias"),
                ]:
                    vw(f"{VPFX}deepstack.{i}.{gg_suf}",
                       f"visual.deepstack_merger_list.{i}.{hf_suf}")
            print(f"  vision: {vc.depth} blocks + {len(ds_idx)} deepstack mergers exported")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    import os
    print(f"\nWrote {args.output} ({os.path.getsize(args.output)/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
