#!/usr/bin/env python3
"""Reranking Benchmark: CrispEmbed cross-encoder vs bi-encoder reranking.

Compares:
  1. CrispEmbed cross-encoder reranking (crispembed_rerank)
  2. CrispEmbed bi-encoder reranking (encode + cosine similarity)
  3. fastembed-rs cross-encoder reranking (if available)

Usage:
    python tests/bench_rerank.py \
        --lib build/libcrispembed.so \
        --embed-gguf all-MiniLM-L6-v2.gguf \
        --reranker-gguf bge-reranker-v2-m3.gguf
"""

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Test data: queries with documents and relevance grades (0-2)
# ---------------------------------------------------------------------------

TEST_CASES = [
    {
        "query": "What is machine learning?",
        "documents": [
            ("Machine learning is a branch of AI that enables computers to learn from data without being explicitly programmed.", 2),
            ("The weather forecast for tomorrow shows partly cloudy skies with a high of 72 degrees.", 0),
            ("Deep learning is a subset of machine learning that uses neural networks with many layers.", 1),
            ("Python is a popular programming language used in data science and web development.", 0),
            ("Supervised learning requires labeled training data to make predictions on new examples.", 1),
        ],
    },
    {
        "query": "How does photosynthesis work?",
        "documents": [
            ("Photosynthesis converts light energy into chemical energy, producing glucose and oxygen from CO2 and water.", 2),
            ("The stock market experienced significant volatility during the trading session.", 0),
            ("Chloroplasts contain chlorophyll, the green pigment that absorbs light for photosynthesis.", 2),
            ("Plants use sunlight to grow, which involves various metabolic processes.", 1),
            ("The recipe calls for two cups of flour and one cup of sugar.", 0),
        ],
    },
    {
        "query": "What are the benefits of exercise?",
        "documents": [
            ("Regular exercise improves cardiovascular health, reduces stress, and helps maintain a healthy weight.", 2),
            ("The new smartphone features a 6.7-inch OLED display and a 108MP camera.", 0),
            ("Physical activity releases endorphins which improve mood and reduce symptoms of depression.", 2),
            ("Walking for 30 minutes daily can lower the risk of chronic diseases.", 1),
            ("The library has an extensive collection of books on various subjects.", 0),
        ],
    },
    {
        "query": "Explain retrieval-augmented generation",
        "documents": [
            ("RAG combines a retrieval system with a language model, fetching relevant documents to ground LLM responses in factual information.", 2),
            ("GPU prices have been declining steadily throughout the year.", 0),
            ("Dense retrieval encodes queries and documents into vector embeddings for semantic similarity search.", 1),
            ("Language models can hallucinate facts, and retrieval helps mitigate this by providing source documents.", 1),
            ("The restaurant serves an excellent selection of Italian cuisine.", 0),
        ],
    },
    {
        "query": "What is cosine similarity used for?",
        "documents": [
            ("Cosine similarity measures the cosine of the angle between two vectors, commonly used to compare document embeddings in information retrieval.", 2),
            ("The museum exhibit features paintings from the Renaissance period.", 0),
            ("In NLP, cosine similarity helps determine how semantically similar two pieces of text are.", 2),
            ("Vector databases use distance metrics like cosine similarity and Euclidean distance for nearest neighbor search.", 1),
            ("The hiking trail offers beautiful views of the mountain range.", 0),
        ],
    },
]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def ndcg_at_k(scores: List[float], relevances: List[int], k: int) -> float:
    """NDCG@k with graded relevance."""
    # Sort by predicted score
    paired = sorted(zip(scores, relevances), key=lambda x: -x[0])
    dcg = sum(
        (2 ** rel - 1) / math.log2(i + 2)
        for i, (_, rel) in enumerate(paired[:k])
    )
    # Ideal: sort by relevance
    ideal_rels = sorted(relevances, reverse=True)
    idcg = sum(
        (2 ** rel - 1) / math.log2(i + 2)
        for i, rel in enumerate(ideal_rels[:k])
    )
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(scores: List[float], relevances: List[int], k: int,
                   threshold: int = 1) -> float:
    """Precision@k: fraction of top-k results with relevance >= threshold."""
    paired = sorted(zip(scores, relevances), key=lambda x: -x[0])
    relevant = sum(1 for _, rel in paired[:k] if rel >= threshold)
    return relevant / k


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------

def rerank_crispembed_cross(query: str, documents: List[str],
                             reranker_path: str, lib_path: str) -> List[float]:
    """Cross-encoder reranking with CrispEmbed."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
    from crispembed import CrispEmbed
    model = CrispEmbed(reranker_path, lib_path=lib_path)
    if not model.is_reranker:
        print("  WARNING: Model is not a reranker")
        return [0.0] * len(documents)
    return [model.rerank(query, doc) for doc in documents]


def rerank_crispembed_bienc(query: str, documents: List[str],
                             embed_path: str, lib_path: str) -> List[float]:
    """Bi-encoder reranking with CrispEmbed (cosine similarity)."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
    from crispembed import CrispEmbed
    model = CrispEmbed(embed_path, lib_path=lib_path)
    results = model.rerank_biencoder(query, documents)
    # Convert back to scores in original order
    scores = [0.0] * len(documents)
    for r in results:
        scores[r["index"]] = r["score"]
    return scores


def rerank_huggingface(query: str, documents: List[str],
                       model_name: str) -> List[float]:
    """Cross-encoder reranking with HuggingFace."""
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(model_name)
    pairs = [(query, doc) for doc in documents]
    scores = model.predict(pairs)
    return scores.tolist()


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(args):
    print("Reranking Benchmark")
    print(f"  Test cases: {len(TEST_CASES)}")
    print()

    engines = {}

    # CrispEmbed cross-encoder
    if args.reranker_gguf and args.lib:
        engines["CrispEmbed Cross-Encoder"] = lambda q, docs: rerank_crispembed_cross(
            q, docs, args.reranker_gguf, args.lib
        )

    # CrispEmbed bi-encoder
    if args.embed_gguf and args.lib:
        engines["CrispEmbed Bi-Encoder"] = lambda q, docs: rerank_crispembed_bienc(
            q, docs, args.embed_gguf, args.lib
        )

    # HuggingFace cross-encoder
    if not args.skip_hf:
        try:
            from sentence_transformers import CrossEncoder
            engines["HuggingFace Cross-Encoder"] = lambda q, docs: rerank_huggingface(
                q, docs, args.hf_reranker
            )
        except ImportError:
            pass

    if not engines:
        print("No engines configured. Use --lib + --reranker-gguf or --embed-gguf")
        return

    results = {}
    for engine_name, rerank_fn in engines.items():
        print(f"Running {engine_name}...")
        ndcg_scores = []
        prec_scores = []
        t0 = time.time()

        for tc in TEST_CASES:
            query = tc["query"]
            docs = [d for d, _ in tc["documents"]]
            rels = [r for _, r in tc["documents"]]

            scores = rerank_fn(query, docs)
            ndcg_scores.append(ndcg_at_k(scores, rels, 3))
            prec_scores.append(precision_at_k(scores, rels, 3))

        elapsed = time.time() - t0
        results[engine_name] = {
            "NDCG@3": np.mean(ndcg_scores),
            "P@3": np.mean(prec_scores),
            "time_s": elapsed,
            "per_query_ms": elapsed / len(TEST_CASES) * 1000,
        }

    # Print results
    print()
    print("=" * 75)
    print(f"{'Engine':<30} {'NDCG@3':>8} {'P@3':>8} {'Time':>8} {'Per-Q':>10}")
    print("-" * 75)
    for engine, m in results.items():
        print(f"{engine:<30} {m['NDCG@3']:.4f}   {m['P@3']:.4f}   "
              f"{m['time_s']:.2f}s   {m['per_query_ms']:.1f}ms")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="Reranking Benchmark")
    parser.add_argument("--lib", help="Path to libcrispembed.so/.dylib")
    parser.add_argument("--reranker-gguf", help="Path to reranker GGUF")
    parser.add_argument("--embed-gguf", help="Path to embedding GGUF (for bi-encoder)")
    parser.add_argument("--hf-reranker", default="cross-encoder/ms-marco-MiniLM-L-6-v2",
                        help="HuggingFace cross-encoder model")
    parser.add_argument("--skip-hf", action="store_true")
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
