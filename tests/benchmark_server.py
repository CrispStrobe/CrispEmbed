#!/usr/bin/env python3
"""Server-mode benchmark: CrispEmbed vs HuggingFace sentence-transformers.

Uses CrispEmbed's HTTP server (model loaded once) for fair comparison.
Falls back to CLI with model pre-loading trick if server not available.

Usage:
    # Start server first:
    ./build/crispembed-server -m model.gguf --port 8090 &

    # Run benchmark:
    python tests/benchmark_server.py --port 8090 \
        --model sentence-transformers/all-MiniLM-L6-v2

    # Or standalone (uses CLI with subprocess, measures per-text):
    python tests/benchmark_server.py --binary ./build/crispembed \
        --gguf /path/to/model.gguf --model MODEL_ID
"""

import argparse
import json
import os
import subprocess
import sys
import time
import numpy as np

# Test corpus of varying lengths
TEXTS = {
    "short": [
        "Hello world",
        "Quick test",
        "Search query",
        "Semantic matching",
        "Vector database",
    ],
    "medium": [
        "The quick brown fox jumps over the lazy dog near the river bank",
        "Machine learning is transforming natural language processing tasks",
        "Semantic search enables finding relevant documents by meaning not keywords",
        "Embeddings compress text into dense vectors for efficient similarity computation",
        "Transformers use self-attention to capture long-range dependencies in text",
    ],
    "long": [
        "In recent years, transformer-based language models have revolutionized the field of natural language processing. "
        "These models, trained on vast amounts of text data, can capture complex linguistic patterns and generate coherent text. "
        "Embedding models distill these capabilities into dense vector representations useful for search and retrieval.",
    ] * 5,
}


def bench_server(port, texts, n_warmup=3, n_runs=10):
    """Benchmark CrispEmbed HTTP server."""
    import urllib.request
    url = f"http://localhost:{port}/embed"

    # Warmup
    for _ in range(n_warmup):
        data = json.dumps({"texts": texts[:1]}).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=30)

    # Single-text latency
    latencies = []
    for _ in range(n_runs):
        for text in texts[:5]:
            data = json.dumps({"texts": [text]}).encode()
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            t0 = time.perf_counter()
            urllib.request.urlopen(req, timeout=30)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)

    # Batch throughput
    t0 = time.perf_counter()
    for _ in range(n_runs):
        data = json.dumps({"texts": texts}).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=60)
    t1 = time.perf_counter()
    batch_throughput = (n_runs * len(texts)) / (t1 - t0)

    return {
        "avg_latency_ms": np.mean(latencies),
        "p50_latency_ms": np.percentile(latencies, 50),
        "p95_latency_ms": np.percentile(latencies, 95),
        "batch_throughput": batch_throughput,
    }


def bench_cli(binary, gguf, texts, n_warmup=2, n_runs=5):
    """Benchmark CrispEmbed CLI (includes model load per call)."""
    # Warmup
    for _ in range(n_warmup):
        subprocess.run([binary, "-m", gguf, texts[0]], capture_output=True, timeout=120)

    latencies = []
    for _ in range(n_runs):
        for text in texts[:5]:
            t0 = time.perf_counter()
            r = subprocess.run([binary, "-m", gguf, text], capture_output=True, text=True, timeout=120)
            t1 = time.perf_counter()
            if r.returncode == 0:
                latencies.append((t1 - t0) * 1000)

    # Batch (sequential CLI calls)
    t0 = time.perf_counter()
    for text in texts:
        subprocess.run([binary, "-m", gguf, text], capture_output=True, timeout=120)
    t1 = time.perf_counter()
    batch_throughput = len(texts) / (t1 - t0)

    return {
        "avg_latency_ms": np.mean(latencies),
        "p50_latency_ms": np.percentile(latencies, 50),
        "p95_latency_ms": np.percentile(latencies, 95),
        "batch_throughput": batch_throughput,
    }


def bench_hf(model_id, texts, n_warmup=2, n_runs=5):
    """Benchmark HuggingFace sentence-transformers."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_id, trust_remote_code=True)

    # Warmup
    for _ in range(n_warmup):
        model.encode(texts[:1], normalize_embeddings=True)

    # Single-text latency
    latencies = []
    for _ in range(n_runs):
        for text in texts[:5]:
            t0 = time.perf_counter()
            model.encode([text], normalize_embeddings=True)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)

    # Batch throughput
    t0 = time.perf_counter()
    for _ in range(n_runs):
        model.encode(texts, normalize_embeddings=True, batch_size=32)
    t1 = time.perf_counter()
    batch_throughput = (n_runs * len(texts)) / (t1 - t0)

    return {
        "avg_latency_ms": np.mean(latencies),
        "p50_latency_ms": np.percentile(latencies, 50),
        "p95_latency_ms": np.percentile(latencies, 95),
        "batch_throughput": batch_throughput,
    }


def fmt_result(name, result):
    return (f"  {name:<20s}  avg={result['avg_latency_ms']:7.1f}ms  "
            f"p50={result['p50_latency_ms']:7.1f}ms  "
            f"p95={result['p95_latency_ms']:7.1f}ms  "
            f"throughput={result['batch_throughput']:6.1f} texts/s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, help="CrispEmbed server port")
    parser.add_argument("--binary", default="./build/crispembed")
    parser.add_argument("--gguf", help="GGUF model path (for CLI benchmark)")
    parser.add_argument("--model", help="HF model ID for comparison")
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--skip-hf", action="store_true")
    parser.add_argument("--corpus", choices=["short", "medium", "long", "all"], default="medium")
    args = parser.parse_args()

    if args.corpus == "all":
        texts = TEXTS["short"] + TEXTS["medium"] + TEXTS["long"]
    else:
        texts = TEXTS[args.corpus]

    print(f"Corpus: {args.corpus} ({len(texts)} texts)")
    print()

    results = {}

    # CrispEmbed server benchmark
    if args.port:
        print("CrispEmbed Server:")
        try:
            r = bench_server(args.port, texts, n_runs=args.n_runs)
            print(fmt_result("server", r))
            results["crispembed_server"] = r
        except Exception as e:
            print(f"  ERROR: {e}")

    # CrispEmbed CLI benchmark
    if args.gguf:
        print("CrispEmbed CLI (includes model load per call):")
        for suffix, label in [("", "F32"), ("-q8_0", "Q8_0"), ("-q4_k", "Q4_K")]:
            gguf_path = args.gguf.replace(".gguf", f"{suffix}.gguf") if suffix else args.gguf
            if not os.path.exists(gguf_path):
                continue
            try:
                r = bench_cli(args.binary, gguf_path, texts, n_runs=args.n_runs)
                print(fmt_result(f"CLI {label}", r))
                results[f"crispembed_cli_{label}"] = r
            except Exception as e:
                print(f"  ERROR {label}: {e}")

    # HuggingFace benchmark
    if args.model and not args.skip_hf:
        print("\nHuggingFace sentence-transformers:")
        try:
            r = bench_hf(args.model, texts, n_runs=args.n_runs)
            print(fmt_result("HF PyTorch", r))
            results["hf_pytorch"] = r
        except Exception as e:
            print(f"  ERROR: {e}")

    # Summary
    if len(results) > 1:
        print("\n=== Summary ===")
        baseline = results.get("hf_pytorch", results.get("crispembed_server", {}))
        for name, r in sorted(results.items()):
            speedup = ""
            if baseline and "avg_latency_ms" in baseline and r["avg_latency_ms"] > 0:
                s = baseline["avg_latency_ms"] / r["avg_latency_ms"]
                speedup = f" ({s:.1f}x vs HF)" if "hf" not in name else ""
            print(f"  {name:<25s} {r['avg_latency_ms']:7.1f}ms  {r['batch_throughput']:6.1f}/s{speedup}")


if __name__ == "__main__":
    sys.exit(main() or 0)
