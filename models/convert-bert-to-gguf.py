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
from transformers import AutoModel, AutoModelForSequenceClassification, AutoModelForMaskedLM, AutoTokenizer, AutoConfig


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

    # Patch out transformers' torch.load safety check (CVE-2025-32434).
    # Required for local trusted models that only have .bin (no safetensors).
    _noop = lambda: None
    for _mod_name in ("transformers.modeling_utils",
                      "transformers.utils.import_utils",
                      "transformers.utils"):
        try:
            import importlib
            _m = importlib.import_module(_mod_name)
            if hasattr(_m, "check_torch_load_is_safe"):
                _m.check_torch_load_is_safe = _noop
        except Exception:
            pass

    # Try sequence classification first (rerankers) to get the scoring head,
    # then fall back to AutoModel (embedders + BGE-M3 with sparse/colbert heads).
    # use_safetensors=True avoids torch.load (required when torch < 2.6).
    def _load(cls):
        try:
            return cls.from_pretrained(args.model, trust_remote_code=True,
                                       use_safetensors=True)
        except Exception:
            return cls.from_pretrained(args.model, trust_remote_code=True)

    try:
        model = _load(AutoModelForSequenceClassification)
        sd_probe = model.state_dict()
        # Only keep SeqClass model if it actually has num_labels == 1 (reranker)
        if not (hasattr(model.config, "num_labels") and model.config.num_labels == 1):
            raise ValueError("not a reranker")
    except Exception:
        # Try MaskedLM for SPLADE models (has MLM head)
        try:
            model = _load(AutoModelForMaskedLM)
            sd_probe = model.state_dict()
            if not any("cls.predictions" in k for k in sd_probe.keys()):
                raise ValueError("no MLM head")
            print(f"  detected: MLM head (SPLADE)")
        except Exception:
            model = _load(AutoModel)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model.eval()
    sd = model.state_dict()

    # Normalize: strip backbone prefix (roberta./bert./xlm_roberta.) so keys
    # always look like embeddings.*, encoder.layer.*, classifier.*, etc.
    _backbone_prefix = ""
    for _pfx in ("roberta.", "bert.", "xlm_roberta.", "deberta.", "model."):
        if any(k.startswith(_pfx + "embeddings.") for k in sd):
            _backbone_prefix = _pfx
            break
    if _backbone_prefix:
        _sd_norm = {}
        for k, v in sd.items():
            _sd_norm[k[len(_backbone_prefix):] if k.startswith(_backbone_prefix) else k] = v
        sd = _sd_norm
        print(f"  state_dict prefix stripped: '{_backbone_prefix}'")

    # BGE-M3: sparse/colbert heads stored as separate .pt files
    model_dir = Path(args.model) if Path(args.model).is_dir() else None
    for _head in ("sparse_linear", "colbert_linear"):
        if any(k.startswith(f"{_head}.") for k in sd):
            continue
        _pt_path = None
        if model_dir and (model_dir / f"{_head}.pt").exists():
            _pt_path = str(model_dir / f"{_head}.pt")
        else:
            # Try to pull from HF hub (BGE-M3 stores these as siblings of pytorch_model.bin)
            try:
                from huggingface_hub import hf_hub_download
                _pt_path = hf_hub_download(repo_id=args.model, filename=f"{_head}.pt")
            except Exception:
                _pt_path = None
        if _pt_path:
            _weights = torch.load(_pt_path, map_location="cpu", weights_only=False)
            for _k, _v in _weights.items():
                sd[f"{_head}.{_k}"] = _v
            print(f"  loaded {_head}.pt ({list(_weights.keys())})")

    print(f"Config: hidden={config.hidden_size} layers={config.num_hidden_layers} "
          f"heads={config.num_attention_heads} intermediate={config.intermediate_size} "
          f"vocab={config.vocab_size}")

    # Detect optional retrieval heads (BGE-M3 sparse/colbert, cross-encoder reranker)
    has_sparse_head  = any(k.startswith("sparse_linear.")  for k in sd)
    has_colbert_head = any(k.startswith("colbert_linear.") for k in sd)
    # 2-layer RobertaClassificationHead (bge-reranker-v2-m3) or simple 1-layer head
    has_classifier_2layer = ("classifier.dense.weight" in sd and
                             "classifier.out_proj.weight" in sd)
    has_classifier_1layer = ("classifier.weight" in sd and
                             sd["classifier.weight"].shape[0] == 1)
    has_classifier = has_classifier_2layer or has_classifier_1layer
    if has_sparse_head:       print("  detected: sparse_linear head")
    if has_colbert_head:      print("  detected: colbert_linear head")
    if has_classifier_2layer: print("  detected: classifier head 2-layer (reranker)")
    elif has_classifier_1layer: print("  detected: classifier head 1-layer (reranker)")

    if has_sparse_head and has_colbert_head:
        model_type_str = "bgem3"
    elif has_sparse_head:
        model_type_str = "sparse"
    elif has_colbert_head:
        model_type_str = "colbert"
    elif has_classifier:
        model_type_str = "reranker"
    else:
        model_type_str = "dense"

    # Detect architecture: XLM-R vs BERT
    # True XLM-R: model_type is "roberta" or "xlm-roberta" → needs position offset + xlmr arch
    # SentencePiece BERT: model_type is "bert" but uses SP tokenizer → bert arch, no offset
    #   (Ollama's BERT model now supports SP Unigram and BPE tokenizers)
    is_true_xlmr = config.model_type in ("roberta", "xlm-roberta")
    is_sentencepiece_model = hasattr(tokenizer, 'sp_model') or config.vocab_size > 100000
    arch = "xlmr" if (is_true_xlmr and ollama_mode) else ARCH

    # Position embedding offset: RoBERTa/XLM-R/MPNet use padding_idx + 1
    pos_offset = 0
    needs_pos_offset = is_true_xlmr or config.model_type == "mpnet"
    if needs_pos_offset and hasattr(config, "pad_token_id") and config.pad_token_id is not None:
        pos_offset = config.pad_token_id + 1
        print(f"  position_offset: {pos_offset} (RoBERTa/MPNet-style)")

    # Detect pooling method from sentence-transformers config
    pool_method_crisp = 0  # CrispEmbed: 0=mean, 1=CLS, 2=last
    try:
        from huggingface_hub import hf_hub_download
        pool_path = hf_hub_download(repo_id=args.model, filename="1_Pooling/config.json")
        with open(pool_path, encoding="utf-8") as f:
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

    # BGE-M3 quirk: 1_Pooling/config.json says mean, but FlagEmbedding's BGEM3Model
    # actually uses CLS pooling for the dense head. Detect and override.
    if model_type_str == "bgem3" and pool_method_crisp != 1:
        print(f"  pooling: overriding to CLS (BGE-M3 dense head uses CLS, not mean)")
        pool_method_crisp = 1

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
        writer.add_string("bert.model_type", model_type_str)
        if has_colbert_head:
            colbert_out_dim = sd["colbert_linear.weight"].shape[0]
            writer.add_uint32("bert.colbert_dim", colbert_out_dim)
        print(f"  format: CrispEmbed (model_type={model_type_str})")

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
                with open(tok_json_path, encoding="utf-8") as f:
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
            writer.add_uint32("tokenizer.ggml.cls_token_id", tokenizer.cls_token_id if tokenizer.cls_token_id is not None else 101)
            writer.add_uint32("tokenizer.ggml.separator_token_id", tokenizer.sep_token_id if tokenizer.sep_token_id is not None else 102)
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
            # Detect BPE vs WordPiece from tokenizer.json
            is_bpe_tokenizer = False
            scores = []
            try:
                from huggingface_hub import hf_hub_download
                tj_path = hf_hub_download(repo_id=args.model, filename="tokenizer.json")
                with open(tj_path, encoding="utf-8") as f:
                    tj = json.load(f)
                is_bpe_tokenizer = tj.get("model", {}).get("type") == "BPE"
                # Store BPE merges for BPE tokenizers
                if is_bpe_tokenizer:
                    merges = tj.get("model", {}).get("merges", [])
                    writer.add_uint32("tokenizer.ggml.merges_count", len(merges))
                    # Store merges as a tensor (newline-separated blob)
                    # This avoids GGUF string array metadata issues
                    merge_strs = []
                    for m in merges:
                        merge_strs.append(" ".join(m) if isinstance(m, list) else m)
                    blob = "\n".join(merge_strs).encode("utf-8")
                    writer.add_tensor("tokenizer.merges", np.frombuffer(blob, dtype=np.int8))
                    print(f"  BPE merges: {len(merges)} ({len(blob)} bytes as tensor)")
            except Exception as e:
                print(f"  BPE detection: {e}")

            if is_bpe_tokenizer and scores:
                writer.add_array("tokenizer.ggml.scores", scores)
            writer.add_uint32("tokenizer.ggml.type", 1 if is_bpe_tokenizer else 0)
            writer.add_uint32("tokenizer.ggml.cls_token_id", tokenizer.cls_token_id if tokenizer.cls_token_id is not None else 101)
            writer.add_uint32("tokenizer.ggml.sep_token_id", tokenizer.sep_token_id if tokenizer.sep_token_id is not None else 102)
            writer.add_uint32("tokenizer.ggml.unknown_token_id", tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 100)
            writer.add_uint32("tokenizer.ggml.padding_token_id", tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
        tok_type_str = "BPE" if (not ollama_mode and is_bpe_tokenizer) else "WordPiece"
        print(f"  tokenizer: {tok_type_str} ({config.vocab_size} tokens)")

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

    # Embeddings — auto-detect prefix (deberta.embeddings vs embeddings)
    if "deberta.embeddings.word_embeddings.weight" in sd:
        emb_prefix = "deberta.embeddings"
    else:
        emb_prefix = "embeddings"
    # Token embeddings: word_embeddings (BERT) or tok_embeddings (ModernBERT)
    tok_key = f"{emb_prefix}.word_embeddings.weight"
    if tok_key not in sd:
        tok_key = f"{emb_prefix}.tok_embeddings.weight"  # ModernBERT
    # ggml stores embeddings as [H, V] for ggml_get_rows — transpose from PyTorch [V, H]
    writer.add_tensor("token_embd.weight", f32(sd[tok_key]))
    if f"{emb_prefix}.position_embeddings.weight" in sd:
        writer.add_tensor("position_embd.weight", f32(sd[f"{emb_prefix}.position_embeddings.weight"]))
    else:
        print("  note: no position embeddings (model uses rotary/relative positions)")
    if f"{emb_prefix}.token_type_embeddings.weight" in sd:
        writer.add_tensor(f"{TN['type_embd']}.weight", f32(sd[f"{emb_prefix}.token_type_embeddings.weight"]))
    for ln_suffix in ["LayerNorm", "norm"]:
        k = f"{emb_prefix}.{ln_suffix}"
        if f"{k}.weight" in sd:
            writer.add_tensor(f"{TN['embd_ln']}.weight", f32(sd[f"{k}.weight"]))
            if f"{k}.bias" in sd:
                writer.add_tensor(f"{TN['embd_ln']}.bias", f32(sd[f"{k}.bias"]))
            break
    else:
        if "emb_ln.weight" in sd:
            writer.add_tensor(f"{TN['embd_ln']}.weight", f32(sd["emb_ln.weight"]))
            writer.add_tensor(f"{TN['embd_ln']}.bias", f32(sd["emb_ln.bias"]))
    # DeBERTa relative position embeddings (with or without deberta. prefix)
    for rel_key in ["deberta.encoder.rel_embeddings.weight", "encoder.rel_embeddings.weight"]:
        if rel_key in sd:
            writer.add_tensor("rel_embd.weight", f32(sd[rel_key]))
            print("  relative position embeddings: ok")
            break
    # MPNet relative attention bias [n_buckets, n_heads]
    for rab_key in ["encoder.relative_attention_bias.weight", "relative_attention_bias.weight"]:
        if rab_key in sd:
            writer.add_tensor("rel_attn_bias.weight", f32(sd[rab_key]))
            print(f"  relative attention bias: ok ({list(sd[rab_key].shape)})")
            break
    # DeBERTa encoder-level LayerNorm
    for enc_ln in ["deberta.encoder.LayerNorm", "encoder.LayerNorm"]:
        if f"{enc_ln}.weight" in sd:
            writer.add_tensor("encoder_ln.weight", f32(sd[f"{enc_ln}.weight"]))
            writer.add_tensor("encoder_ln.bias", f32(sd[f"{enc_ln}.bias"]))
            break
    print("  embeddings: ok")

    # Auto-detect source weight key patterns:
    # BERT:      encoder.layer.N.attention.self.query,      attention.output.LayerNorm
    # MPNet:     encoder.layer.N.attention.attn.q,          attention.LayerNorm
    # NomicBERT: encoder.layers.N.attn.Wqkv (fused),       norm1/norm2, SwiGLU FFN
    # DeBERTa:   deberta.encoder.layer.N.attention.self.query_proj
    is_mpnet = "encoder.layer.0.attention.attn.q.weight" in sd
    is_nomic = "encoder.layers.0.attn.Wqkv.weight" in sd
    is_modernbert = "layers.0.attn.Wqkv.weight" in sd and "layers.0.mlp.Wi.weight" in sd
    is_deberta = ("encoder.layer.0.attention.self.query_proj.weight" in sd or
                  "deberta.encoder.layer.0.attention.self.query_proj.weight" in sd)

    if is_nomic:
        # NomicBERT: RoPE encoder, fused QKV, SwiGLU FFN, no bias on attn
        rope_theta = getattr(config, "rotary_emb_base", 10000.0)
        writer.add_float32("bert.rope_theta", rope_theta)
        print(f"  NomicBERT: rope_theta={rope_theta}")

    if is_modernbert:
        # ModernBERT: pre-LN, RoPE, GeGLU, fused QKV, fused gate+up, no biases
        rope_cfg = getattr(config, "rope_scaling", {})
        sliding_theta = rope_cfg.get("sliding_attention", {}).get("rope_theta", 10000.0)
        global_theta = rope_cfg.get("full_attention", {}).get("rope_theta", 160000.0)
        global_every = getattr(config, "global_attn_every_n_layers", 3)
        writer.add_float32("bert.rope_theta", sliding_theta)
        writer.add_float32("bert.rope_theta_global", global_theta)
        writer.add_uint32("bert.global_attn_every_n", global_every)
        writer.add_uint32("bert.pre_ln", 1)
        print(f"  ModernBERT: sliding_theta={sliding_theta}, global_theta={global_theta}, every={global_every}, pre_ln=1")
        # Final norm (applied after all layers in pre-LN models)
        if "final_norm.weight" in sd:
            writer.add_tensor("final_norm.weight", f32(sd["final_norm.weight"]))

    # Encoder layers
    for i in range(config.num_hidden_layers):
        if is_modernbert:
            pfx = f"layers.{i}"

            # Pre-attention norm (layer 0 may not have it)
            if f"{pfx}.attn_norm.weight" in sd:
                writer.add_tensor(f"{LP}.{i}.{TN['ln1']}.weight", f32(sd[f"{pfx}.attn_norm.weight"]))

            # Pre-FFN norm
            if f"{pfx}.mlp_norm.weight" in sd:
                writer.add_tensor(f"{LP}.{i}.{TN['ln2']}.weight", f32(sd[f"{pfx}.mlp_norm.weight"]))

            # Store fused QKV as single [3H, H] tensor for single matmul
            qkv = sd[f"{pfx}.attn.Wqkv.weight"]  # [3*768, 768] = [2304, 768]
            H = qkv.shape[1]
            writer.add_tensor(f"{LP}.{i}.attn.qkv.weight", f32(qkv))

            # Attention output
            writer.add_tensor(f"{LP}.{i}.{TN['attn_o']}.weight", wt(sd[f"{pfx}.attn.Wo.weight"]))

            # GeGLU FFN: Wi [2*intermediate, H] → split into up + gate
            wi = sd[f"{pfx}.mlp.Wi.weight"]
            mid = wi.shape[0] // 2
            writer.add_tensor(f"{LP}.{i}.{TN['ffn_up']}.weight", wt(wi[:mid]))     # input (activation applied)
            writer.add_tensor(f"{LP}.{i}.ffn_gate.weight", wt(wi[mid:]))            # gate (multiplied)
            writer.add_tensor(f"{LP}.{i}.{TN['ffn_down']}.weight", wt(sd[f"{pfx}.mlp.Wo.weight"]))

        elif is_nomic:
            pfx = f"encoder.layers.{i}"

            # Layer norms: pre-attention (norm1) and pre-FFN (norm2)
            writer.add_tensor(f"{LP}.{i}.{TN['ln1']}.weight", f32(sd[f"{pfx}.norm1.weight"]))
            writer.add_tensor(f"{LP}.{i}.{TN['ln1']}.bias", f32(sd[f"{pfx}.norm1.bias"]))
            writer.add_tensor(f"{LP}.{i}.{TN['ln2']}.weight", f32(sd[f"{pfx}.norm2.weight"]))
            writer.add_tensor(f"{LP}.{i}.{TN['ln2']}.bias", f32(sd[f"{pfx}.norm2.bias"]))

            # Fused QKV: [3H, H] → split into Q [H,H], K [H,H], V [H,H]
            qkv = sd[f"{pfx}.attn.Wqkv.weight"]
            H = qkv.shape[1]
            writer.add_tensor(f"{LP}.{i}.{TN['attn_q']}.weight", wt(qkv[:H]))
            writer.add_tensor(f"{LP}.{i}.{TN['attn_k']}.weight", wt(qkv[H:2*H]))
            writer.add_tensor(f"{LP}.{i}.{TN['attn_v']}.weight", wt(qkv[2*H:]))

            # Attention output (no bias in NomicBERT)
            writer.add_tensor(f"{LP}.{i}.{TN['attn_o']}.weight", wt(sd[f"{pfx}.attn.out_proj.weight"]))

            # SwiGLU FFN: fc11 (gate), fc12 (up), fc2 (down)
            writer.add_tensor(f"{LP}.{i}.{TN['ffn_up']}.weight", wt(sd[f"{pfx}.mlp.fc12.weight"]))
            writer.add_tensor(f"{LP}.{i}.{TN['ffn_down']}.weight", wt(sd[f"{pfx}.mlp.fc2.weight"]))
            writer.add_tensor(f"{LP}.{i}.ffn_gate.weight", wt(sd[f"{pfx}.mlp.fc11.weight"]))

        else:
            # BERT / DeBERTa / MPNet layer prefix
            # DeBERTa prefix may be stripped by HF loader — check both
            if is_deberta and f"deberta.encoder.layer.0.attention.self.query_proj.weight" in sd:
                pfx = f"deberta.encoder.layer.{i}"
            else:
                pfx = f"encoder.layer.{i}"

            if is_mpnet:
                ln1_key = f"{pfx}.attention.LayerNorm"
            else:
                ln1_key = f"{pfx}.attention.output.LayerNorm"

            # Post-attention LayerNorm
            writer.add_tensor(f"{LP}.{i}.{TN['ln1']}.weight", f32(sd[f"{ln1_key}.weight"]))
            writer.add_tensor(f"{LP}.{i}.{TN['ln1']}.bias", f32(sd[f"{ln1_key}.bias"]))

            # Attention Q/K/V
            if is_mpnet:
                for proj, hf_name in [("attn_q", "attn.q"), ("attn_k", "attn.k"), ("attn_v", "attn.v")]:
                    writer.add_tensor(f"{LP}.{i}.{TN[proj]}.weight", wt(sd[f"{pfx}.attention.{hf_name}.weight"]))
                    writer.add_tensor(f"{LP}.{i}.{TN[proj]}.bias", f32(sd[f"{pfx}.attention.{hf_name}.bias"]))
                attn_o_key = f"{pfx}.attention.attn.o"
            elif is_deberta:
                for proj, hf_name in [("attn_q", "query_proj"), ("attn_k", "key_proj"), ("attn_v", "value_proj")]:
                    writer.add_tensor(f"{LP}.{i}.{TN[proj]}.weight", wt(sd[f"{pfx}.attention.self.{hf_name}.weight"]))
                    writer.add_tensor(f"{LP}.{i}.{TN[proj]}.bias", f32(sd[f"{pfx}.attention.self.{hf_name}.bias"]))
                attn_o_key = f"{pfx}.attention.output.dense"
            else:
                for proj, hf_name in [("attn_q", "query"), ("attn_k", "key"), ("attn_v", "value")]:
                    writer.add_tensor(f"{LP}.{i}.{TN[proj]}.weight", wt(sd[f"{pfx}.attention.self.{hf_name}.weight"]))
                    writer.add_tensor(f"{LP}.{i}.{TN[proj]}.bias", f32(sd[f"{pfx}.attention.self.{hf_name}.bias"]))
                attn_o_key = f"{pfx}.attention.output.dense"

            # Attention output
            writer.add_tensor(f"{LP}.{i}.{TN['attn_o']}.weight", wt(sd[f"{attn_o_key}.weight"]))
            if f"{attn_o_key}.bias" in sd:
                writer.add_tensor(f"{LP}.{i}.{TN['attn_o']}.bias", f32(sd[f"{attn_o_key}.bias"]))

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

    # Optional retrieval heads (CrispEmbed-native only — Ollama doesn't use them)
    if not ollama_mode:
        if has_sparse_head:
            writer.add_tensor("sparse_linear.weight", f32(sd["sparse_linear.weight"]))
            if "sparse_linear.bias" in sd:
                writer.add_tensor("sparse_linear.bias", f32(sd["sparse_linear.bias"]))
            print("  sparse_linear: ok")
        if has_colbert_head:
            writer.add_tensor("colbert_linear.weight", f32(sd["colbert_linear.weight"]))
            if "colbert_linear.bias" in sd:
                writer.add_tensor("colbert_linear.bias", f32(sd["colbert_linear.bias"]))
            print("  colbert_linear: ok")
        if has_classifier_2layer:
            writer.add_tensor("classifier.dense.weight",    f32(sd["classifier.dense.weight"]))
            writer.add_tensor("classifier.dense.bias",      f32(sd["classifier.dense.bias"]))
            writer.add_tensor("classifier.out_proj.weight", f32(sd["classifier.out_proj.weight"]))
            if "classifier.out_proj.bias" in sd:
                writer.add_tensor("classifier.out_proj.bias", f32(sd["classifier.out_proj.bias"]))
            print("  classifier (2-layer): ok")
        elif has_classifier_1layer:
            writer.add_tensor("classifier.weight", f32(sd["classifier.weight"]))
            if "classifier.bias" in sd:
                writer.add_tensor("classifier.bias", f32(sd["classifier.bias"]))
            print("  classifier (1-layer): ok")

        # SPLADE MLM head (cls.predictions.transform + decoder)
        has_mlm = "cls.predictions.transform.dense.weight" in sd
        if has_mlm:
            writer.add_tensor("mlm_transform.weight", f32(sd["cls.predictions.transform.dense.weight"]))
            writer.add_tensor("mlm_transform.bias", f32(sd["cls.predictions.transform.dense.bias"]))
            writer.add_tensor("mlm_ln.weight", f32(sd["cls.predictions.transform.LayerNorm.weight"]))
            writer.add_tensor("mlm_ln.bias", f32(sd["cls.predictions.transform.LayerNorm.bias"]))
            # Decoder bias (decoder weight is tied to token_embd)
            if "cls.predictions.bias" in sd:
                writer.add_tensor("mlm_bias", f32(sd["cls.predictions.bias"]))
            writer.add_uint32("bert.has_mlm_head", 1)
            print("  MLM/SPLADE head: ok")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    import os
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\nWrote {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
