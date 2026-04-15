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
    args = parser.parse_args()

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

    # Position embedding offset: RoBERTa/XLM-R uses padding_idx + 1
    pos_offset = 0
    if hasattr(config, "pad_token_id") and config.pad_token_id is not None:
        if config.model_type in ("roberta", "xlm-roberta"):
            pos_offset = config.pad_token_id + 1
            print(f"  position_offset: {pos_offset} (RoBERTa-style)")
    writer.add_uint32("bert.position_offset", pos_offset)

    # Detect pooling method from sentence-transformers config
    pool_method = 0  # default: mean
    try:
        from huggingface_hub import hf_hub_download
        pool_path = hf_hub_download(repo_id=args.model, filename="1_Pooling/config.json")
        with open(pool_path) as f:
            pool_cfg = json.load(f)
        if pool_cfg.get("pooling_mode_cls_token", False):
            pool_method = 1  # CLS
            print(f"  pooling: CLS (from 1_Pooling/config.json)")
        elif pool_cfg.get("pooling_mode_lasttoken", False):
            pool_method = 2  # last token
            print(f"  pooling: last-token (from 1_Pooling/config.json)")
        else:
            print(f"  pooling: mean (from 1_Pooling/config.json)")
    except Exception:
        print(f"  pooling: mean (default, no 1_Pooling/config.json)")
    writer.add_uint32("bert.pooling_method", pool_method)

    # Tokenizer vocab
    vocab = tokenizer.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}
    tokens = [id_to_token.get(i, f"[UNK_{i}]") for i in range(config.vocab_size)]
    writer.add_array("tokenizer.ggml.tokens", tokens)

    # Detect tokenizer type
    is_sentencepiece = hasattr(tokenizer, 'sp_model') or config.vocab_size > 100000
    if is_sentencepiece:
        writer.add_uint32("tokenizer.ggml.type", 2)  # SentencePiece
        writer.add_uint32("tokenizer.ggml.bos_token_id", tokenizer.bos_token_id or 0)
        writer.add_uint32("tokenizer.ggml.eos_token_id", tokenizer.eos_token_id or 2)
        writer.add_uint32("tokenizer.ggml.unknown_token_id", tokenizer.unk_token_id or 3)
        writer.add_uint32("tokenizer.ggml.padding_token_id", tokenizer.pad_token_id or 1)
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
                    # Unigram model: vocab is [[token, score], ...]
                    scores = [0.0] * config.vocab_size
                    for i, (tok_str, score) in enumerate(tj_vocab):
                        if i < config.vocab_size:
                            scores[i] = float(score)
                    writer.add_array("tokenizer.ggml.scores", scores)
                    scores_loaded = True
                    print(f"  tokenizer: SentencePiece ({config.vocab_size} tokens, scores from tokenizer.json)")
            except Exception as e:
                print(f"  warning: could not load scores from tokenizer.json: {e}")
        if not scores_loaded:
            print(f"  tokenizer: SentencePiece ({config.vocab_size} tokens, NO SCORES — tokenization may be wrong)")
    else:
        writer.add_uint32("tokenizer.ggml.type", 0)  # WordPiece
        writer.add_uint32("tokenizer.ggml.cls_token_id", tokenizer.cls_token_id or 101)
        writer.add_uint32("tokenizer.ggml.sep_token_id", tokenizer.sep_token_id or 102)
        writer.add_uint32("tokenizer.ggml.unknown_token_id", tokenizer.unk_token_id or 100)
        writer.add_uint32("tokenizer.ggml.padding_token_id", tokenizer.pad_token_id or 0)
        print(f"  tokenizer: WordPiece ({config.vocab_size} tokens)")

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
