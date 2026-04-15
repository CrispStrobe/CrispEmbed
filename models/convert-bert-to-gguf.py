#!/usr/bin/env python3
"""Convert a HuggingFace BERT/MiniLM/E5/XLM-R model to GGUF format.

Supports two output modes:
  --ollama   Ollama-compatible tensor names and metadata (default)
  --crisp    CrispEmbed-native tensor names and metadata

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


def q8_0(t: torch.Tensor) -> np.ndarray:
    """Quantize to Q8_0 if dimensions allow."""
    data = t.detach().float().cpu().numpy().astype(np.float32)
    if data.ndim < 2 or data.shape[-1] % 32 != 0:
        return data
    try:
        return gguf.quantize(data, gguf.GGMLQuantizationType.Q8_0)
    except Exception:
        return data


def main():
    parser = argparse.ArgumentParser(description="Convert BERT-family model to GGUF")
    parser.add_argument("--model", required=True, help="HF model ID or local path")
    parser.add_argument("--output", required=True, help="Output .gguf path")
    parser.add_argument("--dtype", choices=["f16", "f32", "q8_0"], default="f32",
                        help="Weight dtype for linear layers (default: f32)")
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

    print(f"Loading model: {args.model}")
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model.eval()
    sd = model.state_dict()

    print(f"Config: hidden={config.hidden_size} layers={config.num_hidden_layers} "
          f"heads={config.num_attention_heads} intermediate={config.intermediate_size} "
          f"vocab={config.vocab_size}")

    # Detect architecture: XLM-R vs BERT
    # True XLM-R: model_type is "roberta" or "xlm-roberta" → needs position offset
    # SentencePiece BERT: model_type is "bert" but uses SP tokenizer → no position offset
    is_true_xlmr = config.model_type in ("roberta", "xlm-roberta")
    is_sentencepiece_model = hasattr(tokenizer, 'sp_model') or config.vocab_size > 100000
    # For Ollama: use xlmr arch for true XLM-R, bert arch for SP-BERT
    # (Ollama's bert model only supports WordPiece, SP-BERT needs xlmr arch but no offset)
    needs_xlmr_arch = is_true_xlmr or (is_sentencepiece_model and ollama_mode)
    arch = "xlmr" if needs_xlmr_arch else ARCH

    # Position embedding offset: only true RoBERTa/XLM-R uses padding_idx + 1
    pos_offset = 0
    if is_true_xlmr and hasattr(config, "pad_token_id") and config.pad_token_id is not None:
        pos_offset = config.pad_token_id + 1
        print(f"  position_offset: {pos_offset} (RoBERTa-style)")

    # Detect pooling method from sentence-transformers config
    pool_method_crisp = 0  # CrispEmbed: 0=mean, 1=CLS, 2=last
    try:
        from huggingface_hub import hf_hub_download
        pool_path = hf_hub_download(repo_id=args.model, filename="1_Pooling/config.json")
        with open(pool_path) as f:
            pool_cfg = json.load(f)
        if pool_cfg.get("pooling_mode_cls_token", False):
            pool_method_crisp = 1
            print(f"  pooling: CLS (from 1_Pooling/config.json)")
        elif pool_cfg.get("pooling_mode_lasttoken", False):
            pool_method_crisp = 2
            print(f"  pooling: last-token (from 1_Pooling/config.json)")
        else:
            print(f"  pooling: mean (from 1_Pooling/config.json)")
    except Exception:
        print(f"  pooling: mean (default, no 1_Pooling/config.json)")

    writer = gguf.GGUFWriter(str(args.output), arch=arch)

    if ollama_mode:
        # Ollama-compatible metadata: {arch}.key_name
        # Ollama pooling: 0=None, 1=Mean, 2=CLS, 3=Last
        pool_ollama = {0: 1, 1: 2, 2: 3}[pool_method_crisp]
        writer.add_uint32(f"{arch}.embedding_length", config.hidden_size)
        writer.add_uint32(f"{arch}.block_count", config.num_hidden_layers)
        writer.add_uint32(f"{arch}.attention.head_count", config.num_attention_heads)
        writer.add_uint32(f"{arch}.feed_forward_length", config.intermediate_size)
        writer.add_float32(f"{arch}.attention.layer_norm_epsilon",
                           getattr(config, "layer_norm_eps", 1e-12))
        writer.add_uint32(f"{arch}.context_length", config.max_position_embeddings)
        writer.add_uint32(f"{arch}.pooling_type", pool_ollama)
        writer.add_bool(f"{arch}.normalize_embeddings", True)
        if pos_offset > 0:
            writer.add_uint32(f"{arch}.position_offset", pos_offset)
        print(f"  format: Ollama (arch={arch})")
    else:
        # CrispEmbed-native metadata
        writer.add_uint32("bert.vocab_size", config.vocab_size)
        writer.add_uint32("bert.max_position_embeddings", config.max_position_embeddings)
        writer.add_uint32("bert.hidden_size", config.hidden_size)
        writer.add_uint32("bert.num_attention_heads", config.num_attention_heads)
        writer.add_uint32("bert.num_hidden_layers", config.num_hidden_layers)
        writer.add_uint32("bert.intermediate_size", config.intermediate_size)
        writer.add_float32("bert.layer_norm_eps", getattr(config, "layer_norm_eps", 1e-12))
        writer.add_uint32("bert.output_dim", config.hidden_size)
        writer.add_uint32("bert.position_offset", pos_offset)
        writer.add_uint32("bert.pooling_method", pool_method_crisp)
        print(f"  format: CrispEmbed")

    # Tokenizer vocab
    vocab = tokenizer.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}
    tokens = [id_to_token.get(i, f"[UNK_{i}]") for i in range(config.vocab_size)]

    # Detect tokenizer type
    is_sentencepiece = hasattr(tokenizer, 'sp_model') or config.vocab_size > 100000

    # Ollama's WordPiece tokenizer expects phantom-space tokens:
    # "hello" -> "▁hello", "##ing" -> "ing", "[CLS]" -> "[CLS]"
    if ollama_mode and not is_sentencepiece:
        for i, tok in enumerate(tokens):
            if tok.startswith("[") and tok.endswith("]"):
                pass
            elif tok.startswith("##"):
                tokens[i] = tok[2:]
            else:
                tokens[i] = "\u2581" + tok

    writer.add_array("tokenizer.ggml.tokens", tokens)

    if is_sentencepiece:
        if ollama_mode:
            # Ollama uses string model name: "llama" for SentencePiece
            writer.add_string("tokenizer.ggml.model", "llama")
        else:
            writer.add_uint32("tokenizer.ggml.type", 2)

        writer.add_uint32("tokenizer.ggml.bos_token_id", tokenizer.bos_token_id or 0)
        writer.add_uint32("tokenizer.ggml.eos_token_id", tokenizer.eos_token_id or 2)
        writer.add_uint32("tokenizer.ggml.unknown_token_id", tokenizer.unk_token_id or 3)
        writer.add_uint32("tokenizer.ggml.padding_token_id", tokenizer.pad_token_id or 1)
        if ollama_mode:
            writer.add_bool("tokenizer.ggml.add_bos_token", True)
            writer.add_bool("tokenizer.ggml.add_eos_token", True)

        # Store vocab scores (for SentencePiece unigram model)
        scores_loaded = False
        # Method 1: from sp_model (classic SentencePiece)
        if hasattr(tokenizer, 'sp_model') and tokenizer.sp_model:
            try:
                sp = tokenizer.sp_model
                scores = [sp.GetScore(i) for i in range(config.vocab_size)]
                writer.add_array("tokenizer.ggml.scores", scores)
                scores_loaded = True
                print(f"  tokenizer: SentencePiece ({config.vocab_size} tokens, scores from sp_model)")
            except Exception:
                pass
        # Method 2: from tokenizer.json Unigram vocab (HF fast tokenizer)
        if not scores_loaded:
            try:
                from huggingface_hub import hf_hub_download
                tok_json_path = hf_hub_download(repo_id=args.model, filename="tokenizer.json")
                with open(tok_json_path) as f:
                    tok_json = json.load(f)
                tj_vocab = tok_json.get("model", {}).get("vocab", [])
                if tj_vocab and isinstance(tj_vocab[0], list) and len(tj_vocab[0]) == 2:
                    scores = [0.0] * config.vocab_size
                    for i, (tok_str, score) in enumerate(tj_vocab):
                        if i < config.vocab_size:
                            scores[i] = float(score)
                    writer.add_array("tokenizer.ggml.scores", scores)
                    scores_loaded = True
                    print(f"  tokenizer: SentencePiece ({config.vocab_size} tokens, scores from tokenizer.json)")
            except Exception as e:
                print(f"  warning: could not load scores from tokenizer.json: {e}")

        # Token types for SentencePiece (Ollama needs this)
        if ollama_mode:
            # Token types: 1=normal, 2=unknown, 3=control, 6=byte
            token_types = []
            for i in range(config.vocab_size):
                tok = id_to_token.get(i, "")
                if tok.startswith("<") and tok.endswith(">"):
                    token_types.append(3)  # control
                elif tok.startswith("<0x") and tok.endswith(">"):
                    token_types.append(6)  # byte
                elif i == (tokenizer.unk_token_id or 3):
                    token_types.append(2)  # unknown
                else:
                    token_types.append(1)  # normal
            writer.add_array("tokenizer.ggml.token_type", token_types)

        if not scores_loaded:
            print(f"  tokenizer: SentencePiece ({config.vocab_size} tokens, NO SCORES)")
    else:
        if ollama_mode:
            writer.add_string("tokenizer.ggml.model", "bert")
            writer.add_uint32("tokenizer.ggml.cls_token_id", tokenizer.cls_token_id or 101)
            writer.add_uint32("tokenizer.ggml.separator_token_id", tokenizer.sep_token_id or 102)
            writer.add_bool("tokenizer.ggml.add_bos_token", True)
            writer.add_bool("tokenizer.ggml.add_eos_token", True)
            # Token types for WordPiece
            token_types = []
            for i in range(config.vocab_size):
                tok = id_to_token.get(i, "")
                if tok in ("[CLS]", "[SEP]", "[PAD]", "[MASK]"):
                    token_types.append(3)  # control
                elif tok == "[UNK]":
                    token_types.append(2)  # unknown
                else:
                    token_types.append(1)  # normal
            writer.add_array("tokenizer.ggml.token_type", token_types)
        else:
            writer.add_uint32("tokenizer.ggml.type", 0)
            writer.add_uint32("tokenizer.ggml.cls_token_id", tokenizer.cls_token_id or 101)
            writer.add_uint32("tokenizer.ggml.sep_token_id", tokenizer.sep_token_id or 102)
            writer.add_uint32("tokenizer.ggml.unknown_token_id", tokenizer.unk_token_id or 100)
            writer.add_uint32("tokenizer.ggml.padding_token_id", tokenizer.pad_token_id or 0)
        print(f"  tokenizer: WordPiece ({config.vocab_size} tokens)")

    # Tensor naming: Ollama uses blk.N.attn_q, CrispEmbed uses enc.N.attn.q
    if ollama_mode:
        LP = "blk"  # layer prefix
        TN = {  # tensor name mapping
            "type_embd": "token_types",
            "embd_ln": "token_embd_norm",
            "attn_q": "attn_q", "attn_k": "attn_k", "attn_v": "attn_v",
            "attn_o": "attn_output",
            "ln1": "attn_output_norm", "ln2": "layer_output_norm",
            "ffn_up": "ffn_up", "ffn_down": "ffn_down",
        }
    else:
        LP = "enc"
        TN = {
            "type_embd": "token_type_embd",
            "embd_ln": "embd_ln",
            "attn_q": "attn.q", "attn_k": "attn.k", "attn_v": "attn.v",
            "attn_o": "attn.o",
            "ln1": "ln1", "ln2": "ln2",
            "ffn_up": "ffn.fc1", "ffn_down": "ffn.fc2",
        }

    # Embeddings
    writer.add_tensor("token_embd.weight", f32(sd["embeddings.word_embeddings.weight"]))
    writer.add_tensor("position_embd.weight", f32(sd["embeddings.position_embeddings.weight"]))
    if "embeddings.token_type_embeddings.weight" in sd:
        writer.add_tensor(f"{TN['type_embd']}.weight", f32(sd["embeddings.token_type_embeddings.weight"]))
    writer.add_tensor(f"{TN['embd_ln']}.weight", f32(sd["embeddings.LayerNorm.weight"]))
    writer.add_tensor(f"{TN['embd_ln']}.bias", f32(sd["embeddings.LayerNorm.bias"]))
    print("  embeddings: ok")

    # Encoder layers
    for i in range(config.num_hidden_layers):
        pfx = f"encoder.layer.{i}"

        # Post-attention LayerNorm
        writer.add_tensor(f"{LP}.{i}.{TN['ln1']}.weight", f32(sd[f"{pfx}.attention.output.LayerNorm.weight"]))
        writer.add_tensor(f"{LP}.{i}.{TN['ln1']}.bias", f32(sd[f"{pfx}.attention.output.LayerNorm.bias"]))

        # Attention Q/K/V
        for proj, hf_name in [("attn_q", "query"), ("attn_k", "key"), ("attn_v", "value")]:
            writer.add_tensor(f"{LP}.{i}.{TN[proj]}.weight", wt(sd[f"{pfx}.attention.self.{hf_name}.weight"]))
            writer.add_tensor(f"{LP}.{i}.{TN[proj]}.bias", f32(sd[f"{pfx}.attention.self.{hf_name}.bias"]))
        # Attention output
        writer.add_tensor(f"{LP}.{i}.{TN['attn_o']}.weight", wt(sd[f"{pfx}.attention.output.dense.weight"]))
        writer.add_tensor(f"{LP}.{i}.{TN['attn_o']}.bias", f32(sd[f"{pfx}.attention.output.dense.bias"]))

        # Post-FFN LayerNorm
        writer.add_tensor(f"{LP}.{i}.{TN['ln2']}.weight", f32(sd[f"{pfx}.output.LayerNorm.weight"]))
        writer.add_tensor(f"{LP}.{i}.{TN['ln2']}.bias", f32(sd[f"{pfx}.output.LayerNorm.bias"]))

        # FFN
        writer.add_tensor(f"{LP}.{i}.{TN['ffn_up']}.weight", wt(sd[f"{pfx}.intermediate.dense.weight"]))
        writer.add_tensor(f"{LP}.{i}.{TN['ffn_up']}.bias", f32(sd[f"{pfx}.intermediate.dense.bias"]))
        writer.add_tensor(f"{LP}.{i}.{TN['ffn_down']}.weight", wt(sd[f"{pfx}.output.dense.weight"]))
        writer.add_tensor(f"{LP}.{i}.{TN['ffn_down']}.bias", f32(sd[f"{pfx}.output.dense.bias"]))

        print(f"  {LP}.{i}: ok")

    # Pooler (optional, skip in Ollama mode — not used)
    if not ollama_mode and "pooler.dense.weight" in sd:
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
