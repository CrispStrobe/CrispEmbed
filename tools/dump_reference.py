#!/usr/bin/env python3
"""CrispEmbed — per-layer reference activation dumper.

Loads a HuggingFace encoder model in PyTorch, runs it on test texts,
captures intermediate activations at every architectural boundary via
forward hooks, and writes the collection to a single GGUF tensor archive.
The C++ diff harness (`crispembed-diff`) then loads that GGUF and
compares each tensor against what the ggml encoder graph produces.

Stages captured (per text, suffix _0.._N-1 for each text):

  token_ids_K          (T,)             I32 token IDs
  embed_output_K       (T, D)           F32 after token+pos+type embeddings + LN
  enc_layer_K_N        (T, D)           F32 after encoder layer N (post-LN output)
  enc_attn_K_N         (T, D)           F32 attention sub-layer output (pre-residual)
  enc_ffn_K_N          (T, D)           F32 FFN sub-layer output (pre-residual)
  final_output_K       (T, D)           F32 final encoder output (all layers)
  pooled_K             (D,)             F32 pooled embedding (mean/CLS)
  normalized_K         (D,)             F32 L2-normalized embedding

Usage:

  python tools/dump_reference.py \\
      --model sentence-transformers/all-MiniLM-L6-v2 \\
      --output /tmp/minilm-ref.gguf

  python tools/dump_reference.py \\
      --model Alibaba-NLP/gte-base-en-v1.5 \\
      --pooling cls \\
      --output /tmp/gte-base-ref.gguf

  python tools/dump_reference.py \\
      --model BAAI/bge-reranker-v2-m3 \\
      --reranker \\
      --output /tmp/bge-reranker-ref.gguf

The GGUF archive stores each activation as a named F32 tensor. Load it
from C++ with `crispembed_diff::Ref` and call `ref.compare(name, ...)`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch


# ── Default test texts ─────────────────────────────────────────────────
DEFAULT_TEXTS = [
    "Hello world",
    "The quick brown fox jumps over the lazy dog",
    "Machine learning is transforming natural language processing",
    "This is a test of the embedding system",
]

DEFAULT_RERANKER_PAIRS = [
    ("What is the capital of France?", "Paris is the capital of France."),
    ("What is the capital of France?", "Cats are popular pets."),
    ("How do I cook pasta?", "Boil water, add pasta, cook 8-10 minutes."),
    ("How do I cook pasta?", "The stock market hit a record high today."),
]


# ── Forward hook helpers (adapted from CrispASR _hooks.py) ────────────

def _hook_factory(captured: Dict, name: str):
    """Return a forward_hook that stores output under captured[name]."""
    def hook(_module, _inp, output):
        if hasattr(output, "last_hidden_state"):
            t = output.last_hidden_state
        elif isinstance(output, (tuple, list)):
            t = output[0]
        else:
            t = output
        if isinstance(t, torch.Tensor):
            captured[name] = t.detach().cpu().float()
    return hook


def capture_modules(captured: Dict, stages):
    """Register forward hooks. Returns handles for drop_hooks()."""
    handles = []
    for name, module in stages:
        if module is None:
            continue
        handles.append(module.register_forward_hook(_hook_factory(captured, name)))
    return handles


def drop_hooks(handles):
    for h in handles:
        h.remove()


# ── Architecture-aware layer enumeration ──────────────────────────────

def get_encoder_layers(model):
    """Return list of (layer_index, layer_module) for any supported arch."""
    # Standard BERT/XLM-R/MPNet
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
        return list(enumerate(model.encoder.layer))
    # NomicBERT / Jina v2 (encoder.layers)
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):
        return list(enumerate(model.encoder.layers))
    # ModernBERT (model.layers)
    if hasattr(model, 'layers'):
        return list(enumerate(model.layers))
    return []


def get_embeddings_module(model):
    """Return the embeddings module."""
    if hasattr(model, 'embeddings'):
        return model.embeddings
    return None


def get_attention_sublayer(layer, arch_hint: str):
    """Return the attention sub-module of a layer (varies by arch)."""
    # Standard BERT
    if hasattr(layer, 'attention'):
        return layer.attention
    # NomicBERT/Jina v2
    if hasattr(layer, 'mixer'):
        return layer.mixer
    if hasattr(layer, 'attn'):
        return layer.attn
    return None


def get_ffn_sublayer(layer, arch_hint: str):
    """Return the FFN sub-module of a layer."""
    # Standard BERT
    if hasattr(layer, 'output') and hasattr(layer, 'intermediate'):
        return layer.output  # BERT output = FFN down + residual + LN
    # NomicBERT / Jina v2 / ModernBERT
    if hasattr(layer, 'mlp'):
        return layer.mlp
    return None


# ── Main dumper ───────────────────────────────────────────────────────

def dump_embeddings(model_id: str, texts: List[str], pooling: str = "mean",
                    trust_remote_code: bool = True) -> Dict[str, np.ndarray]:
    """Run HF model with forward hooks, capture all intermediates."""
    from transformers import AutoModel, AutoTokenizer

    print(f"Loading model: {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model.eval()

    out_dict = {}
    layers = get_encoder_layers(model)
    n_layers = len(layers)
    print(f"  {n_layers} encoder layers detected")

    for text_idx, text in enumerate(texts):
        print(f"  text {text_idx}: '{text[:60]}...' " if len(text) > 60 else f"  text {text_idx}: '{text}'")

        inputs = tok(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        token_ids = inputs['input_ids'][0].numpy().astype(np.int32)
        out_dict[f"token_ids_{text_idx}"] = token_ids

        # Register hooks on embedding output + each encoder layer + sublayers
        captured = {}
        stages = []

        emb_mod = get_embeddings_module(model)
        if emb_mod:
            stages.append((f"embed_output_{text_idx}", emb_mod))

        for layer_idx, layer in layers:
            stages.append((f"enc_layer_{text_idx}_{layer_idx}", layer))
            attn = get_attention_sublayer(layer, "")
            if attn:
                stages.append((f"enc_attn_{text_idx}_{layer_idx}", attn))
            ffn = get_ffn_sublayer(layer, "")
            if ffn:
                stages.append((f"enc_ffn_{text_idx}_{layer_idx}", ffn))

        handles = capture_modules(captured, stages)

        with torch.no_grad():
            outputs = model(**inputs)

        drop_hooks(handles)

        # Finalize captured tensors: (B, T, D) -> (T, D)
        for name, t in captured.items():
            if t.ndim == 3:
                arr = t[0].numpy()
            elif t.ndim == 2:
                arr = t.numpy()
            elif t.ndim == 1:
                arr = t.numpy()
            else:
                continue
            out_dict[name] = arr.astype(np.float32)

        # Final encoder output
        hidden = outputs.last_hidden_state[0].detach().cpu().numpy()
        out_dict[f"final_output_{text_idx}"] = hidden.astype(np.float32)

        # Pooling
        mask = inputs['attention_mask'][0].numpy()
        if pooling == "cls":
            pooled = hidden[0]
        else:  # mean
            mask_f = mask.astype(np.float32)
            pooled = (hidden * mask_f[:, None]).sum(0) / mask_f.sum()
        out_dict[f"pooled_{text_idx}"] = pooled.astype(np.float32)

        # L2 normalize
        norm = np.linalg.norm(pooled)
        if norm > 1e-9:
            normalized = pooled / norm
        else:
            normalized = pooled
        out_dict[f"normalized_{text_idx}"] = normalized.astype(np.float32)

    return out_dict


def dump_reranker(model_id: str, pairs: List[tuple],
                  trust_remote_code: bool = True) -> Dict[str, np.ndarray]:
    """Run HF reranker with forward hooks, capture intermediates."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print(f"Loading reranker: {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, trust_remote_code=trust_remote_code)
    model.eval()

    # Get the base model (before classifier)
    base = None
    for attr in ('roberta', 'bert', 'base_model', 'model'):
        if hasattr(model, attr):
            base = getattr(model, attr)
            break
    if base is None:
        base = model

    out_dict = {}
    layers = get_encoder_layers(base)
    n_layers = len(layers)
    print(f"  {n_layers} encoder layers detected")

    for pair_idx, (query, doc) in enumerate(pairs):
        print(f"  pair {pair_idx}: '{query[:30]}' / '{doc[:30]}'")

        inputs = tok(query, doc, return_tensors="pt", padding=True,
                     truncation=True, max_length=512)
        token_ids = inputs['input_ids'][0].numpy().astype(np.int32)
        out_dict[f"token_ids_{pair_idx}"] = token_ids

        # Hooks on base model layers
        captured = {}
        stages = []
        emb_mod = get_embeddings_module(base)
        if emb_mod:
            stages.append((f"embed_output_{pair_idx}", emb_mod))
        for layer_idx, layer in layers:
            stages.append((f"enc_layer_{pair_idx}_{layer_idx}", layer))

        handles = capture_modules(captured, stages)

        with torch.no_grad():
            logits = model(**inputs).logits

        drop_hooks(handles)

        for name, t in captured.items():
            if t.ndim == 3:
                arr = t[0].numpy()
            elif t.ndim == 2:
                arr = t.numpy()
            else:
                continue
            out_dict[name] = arr.astype(np.float32)

        # Reranker score
        score = logits[0].detach().cpu().numpy().astype(np.float32)
        out_dict[f"reranker_score_{pair_idx}"] = score

    return out_dict


# ── GGUF writer ───────────────────────────────────────────────────────

def write_gguf(path: str, tensors: Dict[str, np.ndarray],
               metadata: Optional[Dict[str, str]] = None):
    """Write captured tensors to a GGUF archive."""
    import gguf

    writer = gguf.GGUFWriter(path, "crispembed-ref")

    if metadata:
        for k, v in metadata.items():
            writer.add_string(k, v)

    for name, arr in tensors.items():
        if arr.dtype == np.int32:
            writer.add_tensor(name, arr)
        else:
            writer.add_tensor(name, arr.astype(np.float32))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = Path(path).stat().st_size / 1024 / 1024
    print(f"\nWrote {path} ({size_mb:.1f} MB, {len(tensors)} tensors)")
    for name in sorted(tensors.keys()):
        arr = tensors[name]
        print(f"  {name}: {arr.shape} {arr.dtype}")


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="CrispEmbed reference activation dumper")
    p.add_argument("--model", required=True, help="HuggingFace model ID or local path")
    p.add_argument("--output", required=True, help="Output GGUF path")
    p.add_argument("--pooling", default="mean", choices=["mean", "cls"],
                   help="Pooling strategy (default: mean)")
    p.add_argument("--reranker", action="store_true",
                   help="Treat model as a cross-encoder reranker")
    p.add_argument("--texts", nargs="+", default=None,
                   help="Custom test texts (default: built-in set)")
    p.add_argument("--no-trust-remote-code", action="store_true",
                   help="Disable trust_remote_code")
    args = p.parse_args()

    trust = not args.no_trust_remote_code

    if args.reranker:
        tensors = dump_reranker(args.model, DEFAULT_RERANKER_PAIRS,
                                trust_remote_code=trust)
    else:
        texts = args.texts or DEFAULT_TEXTS
        tensors = dump_embeddings(args.model, texts, pooling=args.pooling,
                                  trust_remote_code=trust)

    metadata = {
        "crispembed.ref.model": args.model,
        "crispembed.ref.pooling": args.pooling,
        "crispembed.ref.is_reranker": "1" if args.reranker else "0",
    }

    write_gguf(args.output, tensors, metadata)


if __name__ == "__main__":
    main()
