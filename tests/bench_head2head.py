#!/usr/bin/env python3
"""Head-to-head benchmark: CrispEmbed vs fastembed-rs vs HuggingFace.

Tests the same models on the same texts across all engines.
Measures single-text latency, batch throughput, and embedding quality.

Usage:
    python tests/bench_head2head.py --lib build/libcrispembed.so
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

TEXTS_SINGLE = "The quick brown fox jumps over the lazy dog near the river bank"

TEXTS_BATCH = [
    "Machine learning is a subset of artificial intelligence.",
    "The weather in Paris is mild in spring.",
    "Neural networks learn patterns from training data.",
    "Python is a popular programming language.",
    "Transformers use self-attention mechanisms.",
    "Cloud computing provides scalable infrastructure.",
    "Natural language processing enables text understanding.",
    "Data privacy is critical in modern AI systems.",
    "Gradient descent optimizes neural network weights.",
    "Retrieval-augmented generation combines search with LLMs.",
]

# Models available in both CrispEmbed and fastembed-rs
SHARED_MODELS = [
    {
        "name": "all-MiniLM-L6-v2",
        "gguf": "all-MiniLM-L6-v2.gguf",
        "fe_enum": "AllMiniLML6V2",
        "hf": "sentence-transformers/all-MiniLM-L6-v2",
        "dim": 384,
    },
    {
        "name": "bge-small-en-v1.5",
        "gguf": "bge-small-en-v1.5.gguf",
        "fe_enum": "BGESmallENV15",
        "hf": "BAAI/bge-small-en-v1.5",
        "dim": 384,
    },
    {
        "name": "nomic-embed-text-v1.5",
        "gguf": "nomic-embed-text-v1.5.gguf",
        "fe_enum": "NomicEmbedTextV1_5",
        "hf": "nomic-ai/nomic-embed-text-v1.5",
        "dim": 768,
    },
    {
        "name": "snowflake-arctic-embed-m",
        "gguf": "snowflake-arctic-embed-m.gguf",
        "fe_enum": "SnowflakeArcticEmbedM",
        "hf": "Snowflake/snowflake-arctic-embed-m",
        "dim": 768,
    },
]


def bench_crispembed(gguf_path, lib_path, n_runs=20):
    """Benchmark CrispEmbed Python wrapper."""
    from crispembed import CrispEmbed

    model = CrispEmbed(gguf_path, lib_path=lib_path)

    # Warmup
    model.encode(TEXTS_SINGLE)
    model.encode(TEXTS_BATCH)

    # Single text
    t0 = time.perf_counter()
    for _ in range(n_runs):
        model.encode(TEXTS_SINGLE)
    single_ms = (time.perf_counter() - t0) / n_runs * 1000

    # Batch (10 texts)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        model.encode(TEXTS_BATCH)
    batch_ms = (time.perf_counter() - t0) / n_runs * 1000

    # Get embedding for quality check
    vec = model.encode(TEXTS_SINGLE)
    del model

    return {
        "single_ms": single_ms,
        "batch_ms": batch_ms,
        "batch_tps": len(TEXTS_BATCH) * 1000 / batch_ms,
        "dim": len(vec),
        "vec": vec,
    }


def bench_fastembed_rs(model_enum, n_runs=20):
    """Benchmark fastembed-rs via Rust API."""
    try:
        # Try to use fastembed Python package (wraps ONNX)
        from fastembed import TextEmbedding

        model = TextEmbedding(model_name=model_enum)

        # Warmup
        list(model.embed([TEXTS_SINGLE]))
        list(model.embed(TEXTS_BATCH))

        # Single text
        t0 = time.perf_counter()
        for _ in range(n_runs):
            list(model.embed([TEXTS_SINGLE]))
        single_ms = (time.perf_counter() - t0) / n_runs * 1000

        # Batch
        t0 = time.perf_counter()
        for _ in range(n_runs):
            list(model.embed(TEXTS_BATCH))
        batch_ms = (time.perf_counter() - t0) / n_runs * 1000

        vec = list(model.embed([TEXTS_SINGLE]))[0]
        del model

        return {
            "single_ms": single_ms,
            "batch_ms": batch_ms,
            "batch_tps": len(TEXTS_BATCH) * 1000 / batch_ms,
            "dim": len(vec),
            "vec": np.array(vec),
        }
    except Exception as e:
        return {"error": str(e)}


def bench_huggingface(model_name, n_runs=20):
    """Benchmark HuggingFace sentence-transformers."""
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)

        # Warmup
        model.encode(TEXTS_SINGLE, normalize_embeddings=True)
        model.encode(TEXTS_BATCH, normalize_embeddings=True)

        # Single
        t0 = time.perf_counter()
        for _ in range(n_runs):
            model.encode(TEXTS_SINGLE, normalize_embeddings=True)
        single_ms = (time.perf_counter() - t0) / n_runs * 1000

        # Batch
        t0 = time.perf_counter()
        for _ in range(n_runs):
            model.encode(TEXTS_BATCH, normalize_embeddings=True)
        batch_ms = (time.perf_counter() - t0) / n_runs * 1000

        vec = model.encode(TEXTS_SINGLE, normalize_embeddings=True)
        del model

        return {
            "single_ms": single_ms,
            "batch_ms": batch_ms,
            "batch_tps": len(TEXTS_BATCH) * 1000 / batch_ms,
            "dim": len(vec),
            "vec": vec,
        }
    except Exception as e:
        return {"error": str(e)}


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    parser = argparse.ArgumentParser(description="Head-to-head benchmark")
    parser.add_argument("--lib", required=True, help="Path to libcrispembed.so")
    parser.add_argument("--gguf-dir", default="/mnt/akademie_storage/test_cohere")
    parser.add_argument("--n-runs", type=int, default=20)
    parser.add_argument("--skip-hf", action="store_true")
    parser.add_argument("--skip-fastembed", action="store_true")
    args = parser.parse_args()

    print("=" * 90)
    print("  CrispEmbed vs fastembed (ONNX) vs HuggingFace — Head-to-Head Benchmark")
    print("=" * 90)
    print(f"  Runs per measurement: {args.n_runs}")
    print(f"  Single text: {len(TEXTS_SINGLE)} chars")
    print(f"  Batch: {len(TEXTS_BATCH)} texts")
    print()

    for model_info in SHARED_MODELS:
        name = model_info["name"]
        gguf_path = os.path.join(args.gguf_dir, model_info["gguf"])

        if not os.path.exists(gguf_path):
            print(f"\n--- {name} --- SKIP (no GGUF)")
            continue

        print(f"\n--- {name} ({model_info['dim']}d) ---")

        results = {}

        # CrispEmbed
        print(f"  CrispEmbed...", end="", flush=True)
        r = bench_crispembed(gguf_path, args.lib, args.n_runs)
        results["CrispEmbed"] = r
        print(f" single={r['single_ms']:.1f}ms  batch={r['batch_ms']:.1f}ms  {r['batch_tps']:.0f} t/s")

        # FastEmbed (Python ONNX)
        if not args.skip_fastembed:
            print(f"  FastEmbed...", end="", flush=True)
            r = bench_fastembed_rs(model_info["hf"], args.n_runs)
            if "error" in r:
                print(f" SKIP ({r['error'][:50]})")
            else:
                results["FastEmbed"] = r
                print(f" single={r['single_ms']:.1f}ms  batch={r['batch_ms']:.1f}ms  {r['batch_tps']:.0f} t/s")

        # HuggingFace
        if not args.skip_hf:
            print(f"  HuggingFace...", end="", flush=True)
            r = bench_huggingface(model_info["hf"], args.n_runs)
            if "error" in r:
                print(f" SKIP ({r['error'][:50]})")
            else:
                results["HuggingFace"] = r
                print(f" single={r['single_ms']:.1f}ms  batch={r['batch_ms']:.1f}ms  {r['batch_tps']:.0f} t/s")

        # Cross-engine cosine similarity
        if len(results) >= 2:
            engines = list(results.keys())
            ref = results[engines[0]]
            for eng in engines[1:]:
                if "vec" in results[eng] and "vec" in ref:
                    cos = cosine(ref["vec"], results[eng]["vec"])
                    print(f"  cos({engines[0]} vs {eng}): {cos:.6f}")

    # Summary table
    print("\n" + "=" * 90)
    print(f"{'Model':<25} {'Engine':<15} {'Single(ms)':>10} {'Batch(ms)':>10} {'Texts/s':>8}")
    print("-" * 90)
    for model_info in SHARED_MODELS:
        name = model_info["name"]
        # Re-run would be needed for full table — just print what we have
    print("=" * 90)


if __name__ == "__main__":
    main()
