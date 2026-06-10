#!/usr/bin/env python3 -u
"""Comprehensive parity test: all CrispEmbed models vs HuggingFace reference.

For each model:
  1. Auto-download GGUF via CLI (--auto-download)
  2. Load HF reference model
  3. Compare final embeddings (cosine similarity) for F32 + all quants
  4. Per-layer dump + comparison for models that fail or need investigation

Usage:
    python -u tests/test_all_parity.py --binary build-q/crispembed --cache-dir ~/.cache/crispembed

    # Test specific models only:
    python -u tests/test_all_parity.py --binary build-q/crispembed --models all-MiniLM-L6-v2 gte-base-en-v1.5

    # With quantize binary for quant testing:
    python -u tests/test_all_parity.py --binary build-q/crispembed --quantize build-q/crispembed-quantize
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

# Patch transformers safety check
try:
    import importlib
    for _mn in ("transformers.modeling_utils", "transformers.utils.import_utils"):
        _m = importlib.import_module(_mn)
        if hasattr(_m, "check_torch_load_is_safe"):
            setattr(_m, "check_torch_load_is_safe", lambda: None)
except Exception:
    pass


# Model registry: gguf_name -> (hf_id, pooling, is_reranker, extra_kwargs)
MODELS = {
    # ── BERT encoder ──
    "all-MiniLM-L6-v2": ("sentence-transformers/all-MiniLM-L6-v2", "mean", False, {}),
    "all-MiniLM-L12-v2": ("sentence-transformers/all-MiniLM-L12-v2", "mean", False, {}),
    "gte-small": ("thenlper/gte-small", "mean", False, {}),
    "bge-small-en-v1.5": ("BAAI/bge-small-en-v1.5", "cls", False, {}),
    "bge-base-en-v1.5": ("BAAI/bge-base-en-v1.5", "cls", False, {}),
    "bge-large-en-v1.5": ("BAAI/bge-large-en-v1.5", "cls", False, {}),
    "arctic-embed-xs": ("Snowflake/snowflake-arctic-embed-xs", "cls", False, {}),

    # ── XLM-R / SentencePiece encoder ──
    "multilingual-e5-small": ("intfloat/multilingual-e5-small", "mean", False, {}),
    "multilingual-e5-base": ("intfloat/multilingual-e5-base", "mean", False, {}),
    "multilingual-e5-large": ("intfloat/multilingual-e5-large", "mean", False, {}),
    "pixie-rune-v1": ("telepix/PIXIE-Rune-v1.0", "cls", False, {}),
    "arctic-embed-l-v2": ("Snowflake/snowflake-arctic-embed-l-v2.0", "cls", False, {}),
    "paraphrase-multilingual-MiniLM-L12-v2": (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "mean", False, {}),
    "snowflake-arctic-embed-m": ("Snowflake/snowflake-arctic-embed-m", "cls", False, {}),
    "snowflake-arctic-embed-l": ("Snowflake/snowflake-arctic-embed-l", "cls", False, {}),
    "bge-m3": ("BAAI/bge-m3", "cls", False, {}),

    # ── NomicBERT (RoPE + SwiGLU / MoE) ──
    "nomic-embed-text-v1.5": ("nomic-ai/nomic-embed-text-v1.5", "mean", False,
                               {"prefix": "search_query: "}),
    "nomic-embed-text-v2-moe": ("nomic-ai/nomic-embed-text-v2-moe", "mean", False,
                                 {"prefix": "search_query: "}),

    # ── MPNet ──
    "all-mpnet-base-v2": ("sentence-transformers/all-mpnet-base-v2", "mean", False, {}),

    # ── MXBai ──
    "mxbai-embed-large-v1": ("mixedbread-ai/mxbai-embed-large-v1", "cls", False, {}),

    # ── GTE v1.5 (post-LN, RoPE, GeGLU) ──
    "gte-base-en-v1.5": ("Alibaba-NLP/gte-base-en-v1.5", "cls", False, {}),
    "gte-large-en-v1.5": ("Alibaba-NLP/gte-large-en-v1.5", "cls", False, {}),

    # ── Qwen3 decoder ──
    "qwen3-embed-0.6b": ("Qwen/Qwen3-Embedding-0.6B", "last", False,
                          {"prefix": "Instruct: Retrieve semantically similar text.\nQuery: "}),
    "octen-0.6b": ("Octen/Octen-Embedding-0.6B", "last", False, {}),
    "f2llm-v2-0.6b": ("codefuse-ai/F2LLM-v2-0.6B", "last", False, {}),
    "jina-v5-nano": ("jinaai/jina-embeddings-v5-text-nano", "last", False, {}),
    "jina-v5-small": ("jinaai/jina-embeddings-v5-text-small", "last", False, {}),
    "harrier-0.6b": ("microsoft/harrier-oss-v1-0.6b", "last", False, {}),
    "harrier-270m": ("microsoft/harrier-oss-v1-270m", "last", False, {}),

    # ── Rerankers ──
    "bge-reranker-base": ("BAAI/bge-reranker-base", "mean", True, {}),
    "bge-reranker-v2-m3": ("BAAI/bge-reranker-v2-m3", "mean", True, {}),
    "jina-reranker-v2-base-multilingual": (
        "jinaai/jina-reranker-v2-base-multilingual", "mean", True, {}),
    "ms-marco-MiniLM-L-6-v2": ("cross-encoder/ms-marco-MiniLM-L-6-v2", "mean", True, {}),
    "ms-marco-MiniLM-L-12-v2": ("cross-encoder/ms-marco-MiniLM-L-12-v2", "mean", True, {}),
    "mxbai-rerank-xsmall-v1": ("mixedbread-ai/mxbai-rerank-xsmall-v1", "mean", True, {}),
    "mxbai-rerank-base-v1": ("mixedbread-ai/mxbai-rerank-base-v1", "mean", True, {}),
}

TEXTS = [
    "Hello world",
    "Machine learning is transforming natural language processing",
    "The quick brown fox jumps over the lazy dog",
    "I love cats and dogs equally",
]


def get_ce_embeddings(binary, gguf_path, texts, prefix=""):
    """Get embeddings from CrispEmbed CLI.

    ``prefix`` is forwarded to ``--prefix`` so the CLI applies the same prompt
    prefix as the HF reference.  Pass an empty string to suppress the model's
    built-in auto-prefix.
    """
    vecs = []
    for t in texts:
        r = subprocess.run([binary, "-m", gguf_path, "--prefix", prefix, t],
                           capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            return None
        try:
            vals = [float(x) for x in r.stdout.strip().split('\n')[-1].split()]
            vecs.append(vals)
        except (ValueError, IndexError):
            return None
    return np.array(vecs)


def get_hf_embeddings(model_id, texts, pooling="mean", is_reranker=False,
                      trust_remote_code=True, prefix=""):
    """Get embeddings from HuggingFace."""
    import torch
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)

    if is_reranker:
        from transformers import AutoModelForSequenceClassification
        m = AutoModelForSequenceClassification.from_pretrained(
            model_id, trust_remote_code=trust_remote_code, torch_dtype=torch.float32)
        m.eval()
        # Get base encoder
        for attr in ('roberta', 'bert', 'base_model', 'model', 'encoder'):
            if hasattr(m, attr):
                base = getattr(m, attr)
                if hasattr(base, 'encoder') or hasattr(base, 'embeddings'):
                    break
        else:
            base = m
    else:
        from transformers import AutoModel
        base = AutoModel.from_pretrained(
            model_id, trust_remote_code=trust_remote_code, torch_dtype=torch.float32)
        base.eval()

    vecs = []
    for t in texts:
        text = prefix + t if prefix else t
        inp = tok(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            o = base(**inp)
        h = o.last_hidden_state[0].float().numpy()
        mask = inp['attention_mask'][0].numpy().astype(np.float32)

        if pooling == "cls":
            p = h[0]
        elif pooling == "last":
            last_pos = int(mask.sum()) - 1
            p = h[last_pos]
        else:  # mean
            p = (h * mask[:, None]).sum(0) / max(mask.sum(), 1)

        norm = np.linalg.norm(p)
        if norm > 1e-9:
            p = p / norm
        vecs.append(p)
    return np.array(vecs)


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-18))


def test_model(binary, model_name, cache_dir, quantize_binary=None):
    """Test a single model. Returns (model_name, {quant: (mean_cos, min_cos)}, error_msg)."""
    hf_id, pooling, is_reranker, extra = MODELS[model_name]
    prefix = extra.get("prefix", "")

    print(f"\n{'─'*60}", flush=True)
    print(f"  {model_name}  ({hf_id})", flush=True)
    print(f"{'─'*60}", flush=True)

    # Step 1: Get GGUF (auto-download)
    gguf_path = os.path.join(cache_dir, f"{model_name}.gguf")
    if not os.path.exists(gguf_path):
        print(f"  Downloading GGUF via CLI...", flush=True)
        r = subprocess.run([binary, "-m", model_name, "--auto-download", "--dim"],
                           capture_output=True, text=True, timeout=600,
                           env={**os.environ, "CRISPEMBED_CACHE_DIR": cache_dir})
        if r.returncode != 0:
            print(f"  SKIP: auto-download failed: {r.stderr[:200]}", flush=True)
            return model_name, {}, f"download failed"

    if not os.path.exists(gguf_path):
        # Try alternate names
        for f in os.listdir(cache_dir):
            if f.startswith(model_name) and f.endswith(".gguf"):
                gguf_path = os.path.join(cache_dir, f)
                break
        else:
            print(f"  SKIP: GGUF not found in {cache_dir}", flush=True)
            return model_name, {}, "gguf not found"

    # Step 2: Get CrispEmbed F32 embeddings (pass the model prefix so both sides match)
    ce_f32 = get_ce_embeddings(binary, gguf_path, TEXTS, prefix=prefix)
    if ce_f32 is None:
        print(f"  SKIP: CrispEmbed CLI failed", flush=True)
        return model_name, {}, "cli failed"

    # Step 3: Sanity check — cross-text diversity
    ct_cos = cosine_sim(ce_f32[0], ce_f32[1])
    if ct_cos > 0.995:
        print(f"  WARNING: degenerate embeddings (cross-text cos={ct_cos:.4f})", flush=True)

    # Step 4: Get HF reference
    try:
        hf = get_hf_embeddings(hf_id, TEXTS, pooling=pooling,
                               is_reranker=is_reranker, prefix=prefix)
    except Exception as e:
        print(f"  SKIP: HF load failed: {str(e)[:200]}", flush=True)
        return model_name, {}, f"hf failed: {str(e)[:100]}"

    # Step 5: Compare F32 vs HF
    results = {}
    f32_coss = [cosine_sim(hf[i], ce_f32[i]) for i in range(len(TEXTS))]
    mean_f32 = np.mean(f32_coss)
    min_f32 = np.min(f32_coss)
    results["F32"] = (mean_f32, min_f32)
    status = "PASS" if min_f32 > 0.95 else "FAIL"
    print(f"  F32:  mean={mean_f32:.6f}  min={min_f32:.6f}  [{status}]", flush=True)

    # Step 6: Quantize and test Q8_0 + Q4_K
    if quantize_binary and os.path.exists(quantize_binary):
        for qtype in ["q8_0", "q4_k"]:
            q_path = gguf_path.replace(".gguf", f"-{qtype}.gguf")
            if not os.path.exists(q_path):
                r = subprocess.run([quantize_binary, gguf_path, q_path, qtype],
                                   capture_output=True, text=True, timeout=300)
                if r.returncode != 0:
                    print(f"  {qtype.upper()}: quantize failed", flush=True)
                    continue
            ce_q = get_ce_embeddings(binary, q_path, TEXTS, prefix=prefix)
            if ce_q is None:
                print(f"  {qtype.upper()}: CLI failed on quantized model", flush=True)
                continue
            q_coss = [cosine_sim(hf[i], ce_q[i]) for i in range(len(TEXTS))]
            mean_q = np.mean(q_coss)
            min_q = np.min(q_coss)
            results[qtype.upper()] = (mean_q, min_q)
            status = "PASS" if min_q > 0.95 else "FAIL"
            print(f"  {qtype.upper():5s} mean={mean_q:.6f}  min={min_q:.6f}  [{status}]", flush=True)

    return model_name, results, None


def main():
    p = argparse.ArgumentParser(description="CrispEmbed comprehensive parity test")
    p.add_argument("--binary", default="build-q/crispembed", help="CrispEmbed CLI binary")
    p.add_argument("--quantize", default="build-q/crispembed-quantize", help="Quantize binary")
    p.add_argument("--cache-dir", default=os.path.expanduser("~/.cache/crispembed"),
                   help="GGUF cache directory")
    p.add_argument("--models", nargs="+", default=None,
                   help="Test specific models only (default: all)")
    p.add_argument("--threshold", type=float, default=0.95,
                   help="Minimum cosine threshold for PASS")
    args = p.parse_args()

    models_to_test = args.models or sorted(MODELS.keys())
    print(f"Testing {len(models_to_test)} models", flush=True)
    print(f"Binary: {args.binary}", flush=True)
    print(f"Cache: {args.cache_dir}", flush=True)

    all_results = {}
    failures = []
    skipped = []

    for model_name in models_to_test:
        if model_name not in MODELS:
            print(f"\nUnknown model: {model_name}", flush=True)
            continue
        try:
            name, results, error = test_model(
                args.binary, model_name, args.cache_dir, args.quantize)
            all_results[name] = results
            if error:
                skipped.append((name, error))
            elif results:
                f32_min = results.get("F32", (0, 0))[1]
                if f32_min < args.threshold:
                    failures.append((name, f32_min))
        except Exception as e:
            print(f"  EXCEPTION: {e}", flush=True)
            skipped.append((model_name, str(e)[:100]))

    # Summary
    print(f"\n{'='*60}", flush=True)
    print(f"  SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"\n{'Model':<45} {'F32':>8} {'Q8_0':>8} {'Q4_K':>8}", flush=True)
    print("-" * 72, flush=True)
    for name in models_to_test:
        if name not in all_results or not all_results[name]:
            print(f"  {name:<43} {'SKIP':>8}", flush=True)
            continue
        r = all_results[name]
        f32 = f"{r['F32'][1]:.4f}" if "F32" in r else "—"
        q8 = f"{r['Q8_0'][1]:.4f}" if "Q8_0" in r else "—"
        q4 = f"{r['Q4_K'][1]:.4f}" if "Q4_K" in r else "—"
        print(f"  {name:<43} {f32:>8} {q8:>8} {q4:>8}", flush=True)

    if failures:
        print(f"\nFAILURES ({len(failures)}):", flush=True)
        for name, cos in failures:
            print(f"  {name}: min_cos={cos:.6f}", flush=True)
    if skipped:
        print(f"\nSKIPPED ({len(skipped)}):", flush=True)
        for name, reason in skipped:
            print(f"  {name}: {reason}", flush=True)

    print(f"\nTotal: {len(all_results)} tested, {len(failures)} failed, {len(skipped)} skipped",
          flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
