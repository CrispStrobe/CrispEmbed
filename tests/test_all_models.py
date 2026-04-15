#!/usr/bin/env python3
"""Test all converted CrispEmbed models against HuggingFace references.

Usage:
    python tests/test_all_models.py --binary ./build/crispembed --gguf-dir /path/to/ggufs

Automatically discovers .gguf files and maps them to HF model IDs.
"""

import argparse
import os
import sys
import subprocess
import numpy as np
from pathlib import Path

# Model registry: GGUF filename pattern → HF model ID
MODEL_MAP = {
    # BERT encoder models
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "gte-small": "thenlper/gte-small",
    "arctic-embed-xs": "Snowflake/snowflake-arctic-embed-xs",
    # XLM-R encoder models
    "multilingual-e5-small": "intfloat/multilingual-e5-small",
    "pixie-rune-v1": "CrispStrobe/PIXIE-Rune-v1.0",
    "arctic-embed-l-v2": "Snowflake/snowflake-arctic-embed-l-v2.0",
    # Qwen3 decoder models
    "octen-0.6b": "Octen/Octen-Embedding-0.6B",
    "f2llm-v2-0.6b": "codefuse-ai/F2LLM-v2-0.6B",
    "jina-v5-nano": "jinaai/jina-embeddings-v5-nano",
    "jina-v5-small": "jinaai/jina-embeddings-v5-small",
    "harrier-0.6b": "Harrier/Harrier-OSS-v1-0.6B",
    "qwen3-embed-0.6b": "Qwen/Qwen3-Embedding-0.6B",
    # Gemma3 decoder models
    "harrier-270m": "Harrier/Harrier-OSS-v1-270M",
}

TEST_TEXTS = [
    "Hello world",
    "The quick brown fox jumps over the lazy dog",
    "Machine learning is transforming NLP",
    "This is a test of the emergency broadcast system",
]


def get_hf_embeddings(model_id, texts):
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_id, trust_remote_code=True)
        vecs = model.encode(texts, normalize_embeddings=True)
        return vecs
    except Exception as e:
        print(f"  HF error: {e}")
        return None


def get_crispembed(binary, gguf, texts):
    results = []
    for text in texts:
        try:
            r = subprocess.run([binary, "-m", gguf, text],
                               capture_output=True, text=True, timeout=120)
            if r.returncode != 0:
                results.append(None)
                continue
            line = r.stdout.strip()
            if not line:
                results.append(None)
                continue
            vec = np.array([float(x) for x in line.split()])
            results.append(vec)
        except Exception:
            results.append(None)
    if not any(v is not None for v in results):
        return None
    dim = next(v.shape[0] for v in results if v is not None)
    out = np.zeros((len(texts), dim))
    for i, v in enumerate(results):
        if v is not None:
            out[i] = v
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default="./build/crispembed")
    parser.add_argument("--gguf-dir", default=".")
    parser.add_argument("--model", help="Test only this model pattern")
    args = parser.parse_args()

    gguf_dir = Path(args.gguf_dir)
    ggufs = sorted(gguf_dir.glob("*.gguf"))

    if not ggufs:
        print(f"No .gguf files found in {gguf_dir}")
        return 1

    print(f"Found {len(ggufs)} GGUF files in {gguf_dir}")
    print(f"Binary: {args.binary}")
    print()

    results = []
    for gguf_path in ggufs:
        name = gguf_path.stem
        if args.model and args.model not in name:
            continue

        # Find matching HF model
        hf_id = None
        for pattern, mid in MODEL_MAP.items():
            if pattern in name:
                hf_id = mid
                break

        if not hf_id:
            print(f"SKIP {name} — no HF model mapping")
            continue

        print(f"TEST {name} → {hf_id}")

        hf_vecs = get_hf_embeddings(hf_id, TEST_TEXTS)
        if hf_vecs is None:
            print(f"  FAIL: HF load failed")
            results.append((name, "FAIL", 0))
            continue

        ce_vecs = get_crispembed(args.binary, str(gguf_path), TEST_TEXTS)
        if ce_vecs is None:
            print(f"  FAIL: CrispEmbed failed")
            results.append((name, "FAIL", 0))
            continue

        if hf_vecs.shape[1] != ce_vecs.shape[1]:
            print(f"  FAIL: dim mismatch ({hf_vecs.shape[1]} vs {ce_vecs.shape[1]})")
            results.append((name, "FAIL", 0))
            continue

        cos_sims = []
        for i in range(len(TEST_TEXTS)):
            cs = np.dot(hf_vecs[i], ce_vecs[i]) / (
                np.linalg.norm(hf_vecs[i]) * np.linalg.norm(ce_vecs[i]) + 1e-12)
            cos_sims.append(cs)

        avg_cos = np.mean(cos_sims)
        min_cos = np.min(cos_sims)
        status = "PASS" if min_cos > 0.99 else ("OK" if min_cos > 0.95 else "FAIL")
        print(f"  avg_cos={avg_cos:.6f} min_cos={min_cos:.6f} → {status}")
        results.append((name, status, avg_cos))

    print("\n" + "=" * 70)
    print(f"{'Model':<45s} {'Status':>8s} {'AvgCos':>10s}")
    print("-" * 70)
    for name, status, cos in results:
        print(f"{name:<45s} {status:>8s} {cos:>10.6f}")

    n_pass = sum(1 for _, s, _ in results if s == "PASS")
    n_ok = sum(1 for _, s, _ in results if s == "OK")
    n_fail = sum(1 for _, s, _ in results if s == "FAIL")
    print(f"\n{n_pass} PASS, {n_ok} OK (>0.95), {n_fail} FAIL")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
