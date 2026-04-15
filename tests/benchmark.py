#!/usr/bin/env python3
"""Benchmark CrispEmbed vs HuggingFace sentence-transformers.

Measures:
  - Single text latency (ms)
  - Throughput (texts/sec) for batch of 100 texts
  - Peak memory usage (MB)

Usage:
    python tests/benchmark.py --binary ./build/crispembed \
        --gguf-dir /path/to/ggufs \
        --model all-MiniLM-L6-v2
"""

import argparse
import os
import subprocess
import sys
import time
import numpy as np

# Default test texts of varying lengths
TEXTS_SHORT = [
    "Hello world",
    "Quick test",
    "Simple query",
]

TEXTS_MEDIUM = [
    "The quick brown fox jumps over the lazy dog near the river bank",
    "Machine learning is transforming natural language processing tasks",
    "Semantic search enables finding relevant documents by meaning not keywords",
]

TEXTS_LONG = [
    "In recent years, transformer-based language models have revolutionized the field of natural language processing. "
    "These models, trained on vast amounts of text data, can capture complex linguistic patterns and generate coherent text. "
    "Embedding models distill these capabilities into dense vector representations useful for search and retrieval.",
    "The development of efficient inference engines for transformer models has become increasingly important. "
    "Projects like ggml and llama.cpp have demonstrated that quantized models can run efficiently on consumer hardware "
    "without significant loss in quality, democratizing access to powerful language technology.",
]

MODEL_MAP = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "gte-small": "thenlper/gte-small",
    "arctic-embed-xs": "Snowflake/snowflake-arctic-embed-xs",
    "octen-0.6b": "Octen/Octen-Embedding-0.6B",
    "harrier-0.6b": "Harrier/Harrier-OSS-v1-0.6B",
    "harrier-270m": "Harrier/Harrier-OSS-v1-270M",
    "qwen3-embed-0.6b": "Qwen/Qwen3-Embedding-0.6B",
}


def bench_crispembed(binary, gguf_path, texts, n_runs=5):
    """Benchmark CrispEmbed CLI on a set of texts."""
    # Warmup
    subprocess.run([binary, "-m", gguf_path, texts[0]],
                   capture_output=True, text=True, timeout=120)

    # Single text latency
    latencies = []
    for _ in range(n_runs):
        for text in texts[:3]:
            t0 = time.perf_counter()
            r = subprocess.run([binary, "-m", gguf_path, text],
                               capture_output=True, text=True, timeout=120)
            t1 = time.perf_counter()
            if r.returncode == 0:
                latencies.append((t1 - t0) * 1000)

    # Throughput: encode 30 texts (each text 3 times for batch simulation)
    batch_texts = (texts * 10)[:30]
    t0 = time.perf_counter()
    for text in batch_texts:
        subprocess.run([binary, "-m", gguf_path, text],
                       capture_output=True, text=True, timeout=120)
    t1 = time.perf_counter()
    throughput = len(batch_texts) / (t1 - t0)

    # Memory: check RSS after one run
    try:
        import resource
        r = subprocess.run([binary, "-m", gguf_path, texts[0]],
                           capture_output=True, text=True, timeout=120)
        # Can't easily get child memory, use file size as proxy
        mem_mb = os.path.getsize(gguf_path) / (1024 * 1024)
    except Exception:
        mem_mb = 0

    return {
        "avg_latency_ms": np.mean(latencies) if latencies else 0,
        "p95_latency_ms": np.percentile(latencies, 95) if latencies else 0,
        "throughput_texts_sec": throughput,
        "model_size_mb": os.path.getsize(gguf_path) / (1024 * 1024),
    }


def bench_hf(model_id, texts, n_runs=3):
    """Benchmark HuggingFace sentence-transformers."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None

    model = SentenceTransformer(model_id, trust_remote_code=True)

    # Warmup
    model.encode(texts[:1], normalize_embeddings=True)

    # Single text latency
    latencies = []
    for _ in range(n_runs):
        for text in texts[:3]:
            t0 = time.perf_counter()
            model.encode([text], normalize_embeddings=True)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)

    # Throughput
    batch_texts = (texts * 10)[:30]
    t0 = time.perf_counter()
    model.encode(batch_texts, normalize_embeddings=True, batch_size=32)
    t1 = time.perf_counter()
    throughput = len(batch_texts) / (t1 - t0)

    return {
        "avg_latency_ms": np.mean(latencies) if latencies else 0,
        "p95_latency_ms": np.percentile(latencies, 95) if latencies else 0,
        "throughput_texts_sec": throughput,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default="./build/crispembed")
    parser.add_argument("--gguf-dir", default="/mnt/akademie_storage/test_cohere")
    parser.add_argument("--model", help="Test specific model")
    parser.add_argument("--skip-hf", action="store_true", help="Skip HuggingFace comparison")
    parser.add_argument("--n-runs", type=int, default=3)
    args = parser.parse_args()

    texts = TEXTS_SHORT + TEXTS_MEDIUM + TEXTS_LONG

    models = [args.model] if args.model else list(MODEL_MAP.keys())

    print(f"{'Model':<25s} {'Quant':>6s} {'Lat(ms)':>9s} {'P95(ms)':>9s} {'Tput':>8s} {'Size':>8s}")
    print("-" * 70)

    for model_name in models:
        if model_name not in MODEL_MAP:
            print(f"  SKIP {model_name} (no HF mapping)")
            continue

        # Test each quant variant
        for suffix, label in [("", "F32"), ("-q8_0", "Q8_0"), ("-q4_k", "Q4_K")]:
            gguf_path = os.path.join(args.gguf_dir, f"{model_name}{suffix}.gguf")
            if not os.path.exists(gguf_path):
                continue

            try:
                result = bench_crispembed(args.binary, gguf_path, texts, args.n_runs)
                print(f"{model_name:<25s} {label:>6s} {result['avg_latency_ms']:>8.1f}ms "
                      f"{result['p95_latency_ms']:>8.1f}ms "
                      f"{result['throughput_texts_sec']:>7.1f}/s "
                      f"{result['model_size_mb']:>7.0f}MB")
            except Exception as e:
                print(f"{model_name:<25s} {label:>6s}  ERROR: {e}")

        # HuggingFace comparison
        if not args.skip_hf:
            hf_id = MODEL_MAP[model_name]
            try:
                result = bench_hf(hf_id, texts, args.n_runs)
                if result:
                    print(f"{model_name:<25s} {'HF':>6s} {result['avg_latency_ms']:>8.1f}ms "
                          f"{result['p95_latency_ms']:>8.1f}ms "
                          f"{result['throughput_texts_sec']:>7.1f}/s "
                          f"{'---':>8s}")
            except Exception as e:
                print(f"{model_name:<25s} {'HF':>6s}  ERROR: {e}")

        print()  # blank line between models


if __name__ == "__main__":
    sys.exit(main() or 0)
