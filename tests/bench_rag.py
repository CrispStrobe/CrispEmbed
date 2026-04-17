#!/usr/bin/env python3
"""RAG Retrieval Quality Benchmark: CrispEmbed vs fastembed-rs.

Evaluates retrieval quality using a synthetic or downloaded IR dataset.
Computes MRR@10, NDCG@10, Recall@10, Recall@100 for each engine+model combo.

Usage:
    # All engines, default model (all-MiniLM-L6-v2)
    python tests/bench_rag.py --lib build/libcrispembed.so

    # Specific model, compare CrispEmbed F32 vs Q8_0
    python tests/bench_rag.py --lib build/libcrispembed.so \
        --gguf model.gguf --gguf-q8 model-q8_0.gguf

    # Include fastembed-rs
    python tests/bench_rag.py --lib build/libcrispembed.so \
        --fastembed-rs ../fastembed-rs/target/release/examples/bench

    # Use BEIR dataset (downloads automatically)
    python tests/bench_rag.py --lib build/libcrispembed.so --dataset scifact
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Built-in synthetic IR dataset (no download needed)
# ---------------------------------------------------------------------------

SYNTHETIC_CORPUS = [
    "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
    "Neural networks are composed of layers of interconnected nodes that process information.",
    "Deep learning uses multiple layers of neural networks to model complex patterns in data.",
    "Natural language processing focuses on the interaction between computers and human language.",
    "Computer vision enables machines to interpret and understand visual information from the world.",
    "Reinforcement learning trains agents through rewards and penalties in an environment.",
    "Transfer learning leverages knowledge from one task to improve performance on another.",
    "Gradient descent is an optimization algorithm used to minimize the loss function in training.",
    "Convolutional neural networks are particularly effective for image recognition tasks.",
    "Recurrent neural networks are designed to work with sequential data like time series.",
    "Transformers use self-attention mechanisms to process input sequences in parallel.",
    "BERT is a pre-trained language model that uses bidirectional context for understanding text.",
    "GPT models are autoregressive language models that generate text one token at a time.",
    "Word embeddings map words to dense vector representations capturing semantic meaning.",
    "Attention mechanisms allow models to focus on relevant parts of the input when making predictions.",
    "Batch normalization helps stabilize and accelerate the training of deep neural networks.",
    "Dropout is a regularization technique that randomly deactivates neurons during training.",
    "Data augmentation artificially increases the size of training datasets through transformations.",
    "Hyperparameter tuning involves finding the optimal configuration for a machine learning model.",
    "Cross-validation is a technique for assessing how well a model generalizes to unseen data.",
    "Random forests combine multiple decision trees to improve prediction accuracy and reduce overfitting.",
    "Support vector machines find the optimal hyperplane that separates classes in feature space.",
    "K-means clustering partitions data points into groups based on their distance to centroids.",
    "Principal component analysis reduces the dimensionality of data while preserving variance.",
    "The bias-variance tradeoff is a fundamental concept in machine learning model selection.",
    "Feature engineering involves creating new input features from raw data to improve model performance.",
    "Ensemble methods combine predictions from multiple models to achieve better accuracy.",
    "Backpropagation computes gradients of the loss function with respect to network weights.",
    "The vanishing gradient problem makes it difficult to train deep networks with many layers.",
    "Generative adversarial networks consist of a generator and discriminator trained in opposition.",
    "Autoencoders learn compressed representations of input data through encoding and decoding.",
    "Semantic search retrieves documents based on meaning rather than keyword matching.",
    "Vector databases store and search high-dimensional embeddings for similarity queries.",
    "Retrieval-augmented generation combines document retrieval with language model generation.",
    "Cosine similarity measures the angle between two vectors in high-dimensional space.",
    "Approximate nearest neighbor search trades exactness for speed in similarity lookups.",
    "HNSW is a graph-based algorithm for efficient approximate nearest neighbor search.",
    "Inverted indexes map terms to the documents that contain them for fast keyword search.",
    "BM25 is a bag-of-words retrieval function that ranks documents by term frequency.",
    "Sentence embeddings encode entire sentences into fixed-size vector representations.",
    "Cross-encoder rerankers score query-document pairs jointly for high-quality ranking.",
    "Bi-encoder models encode queries and documents independently for efficient retrieval.",
    "ColBERT uses per-token embeddings for fine-grained late interaction between query and document.",
    "Sparse retrieval models like SPLADE learn term weights for efficient inverted index search.",
    "Dense retrieval uses continuous vector representations for semantic similarity matching.",
    "Hybrid search combines sparse and dense retrieval for better recall and precision.",
    "Multi-vector retrieval represents documents as sets of vectors for richer matching.",
    "Matryoshka representation learning trains embeddings that work at multiple dimensions.",
    "Knowledge distillation transfers knowledge from a large teacher model to a smaller student.",
    "Quantization reduces model size by using lower-precision number formats for weights.",
]

SYNTHETIC_QUERIES = [
    ("What is machine learning?", [0, 2, 6]),
    ("How do neural networks work?", [1, 2, 8, 9]),
    ("What are transformers in NLP?", [10, 14, 3]),
    ("Explain word embeddings", [13, 39, 44]),
    ("What is retrieval-augmented generation?", [33, 31, 44]),
    ("How does semantic search work?", [31, 34, 44]),
    ("What is the vanishing gradient problem?", [28, 7, 27]),
    ("Explain attention mechanisms", [14, 10, 11]),
    ("What are sparse retrieval models?", [43, 38, 45]),
    ("How does cross-encoder reranking work?", [40, 41, 42]),
    ("What is quantization for models?", [49, 48, 4]),
    ("How do GANs work?", [29, 30, 1]),
    ("Explain transfer learning", [6, 48, 18]),
    ("What is ColBERT?", [42, 46, 41]),
    ("How does cosine similarity work?", [34, 44, 13]),
]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def reciprocal_rank(retrieved: List[int], relevant: List[int]) -> float:
    """Mean reciprocal rank: 1/rank of first relevant result."""
    relevant_set = set(relevant)
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved: List[int], relevant: List[int], k: int) -> float:
    """Normalized discounted cumulative gain at k."""
    relevant_set = set(relevant)
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k]):
        if doc_id in relevant_set:
            dcg += 1.0 / math.log2(i + 2)
    # Ideal DCG
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


def recall_at_k(retrieved: List[int], relevant: List[int], k: int) -> float:
    """Fraction of relevant documents found in top-k."""
    relevant_set = set(relevant)
    found = sum(1 for doc_id in retrieved[:k] if doc_id in relevant_set)
    return found / len(relevant) if relevant else 0.0


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------

def encode_crispembed(texts: List[str], model_path: str, lib_path: str,
                      prefix: str = "") -> np.ndarray:
    """Encode texts using CrispEmbed Python wrapper."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
    from crispembed import CrispEmbed
    model = CrispEmbed(model_path, lib_path=lib_path)
    if prefix:
        model.set_prefix(prefix)
    vecs = model.encode(texts)
    return vecs


def encode_huggingface(texts: List[str], model_name: str,
                       prefix: str = "") -> np.ndarray:
    """Encode texts using HuggingFace sentence-transformers."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    if prefix:
        texts = [prefix + t for t in texts]
    vecs = model.encode(texts, normalize_embeddings=True)
    return vecs


def encode_fastembed_rs(texts: List[str], bench_binary: str,
                        model_name: str) -> Optional[np.ndarray]:
    """Encode texts using fastembed-rs bench binary (if available)."""
    if not os.path.exists(bench_binary):
        return None
    # fastembed-rs bench outputs JSON with embeddings
    input_json = json.dumps({"texts": texts, "model": model_name})
    try:
        result = subprocess.run(
            [bench_binary, "--encode", "--model", model_name],
            input=input_json, capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            print(f"  fastembed-rs error: {result.stderr[:200]}")
            return None
        data = json.loads(result.stdout)
        return np.array(data["embeddings"], dtype=np.float32)
    except Exception as e:
        print(f"  fastembed-rs failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Retrieval evaluation
# ---------------------------------------------------------------------------

def evaluate_retrieval(
    query_vecs: np.ndarray,
    corpus_vecs: np.ndarray,
    queries_with_relevance: List[Tuple[str, List[int]]],
    engine_name: str,
    k_values: List[int] = [10, 100],
) -> Dict[str, float]:
    """Compute retrieval metrics for an engine."""
    # Compute all similarities at once: [n_queries, n_corpus]
    sims = query_vecs @ corpus_vecs.T

    metrics = {}
    mrr_sum = 0.0
    ndcg10_sum = 0.0
    recall_sums = {k: 0.0 for k in k_values}
    n = len(queries_with_relevance)

    for i, (_, relevant) in enumerate(queries_with_relevance):
        # Rank documents by similarity
        ranked = np.argsort(-sims[i]).tolist()

        mrr_sum += reciprocal_rank(ranked, relevant)
        ndcg10_sum += ndcg_at_k(ranked, relevant, 10)
        for k in k_values:
            recall_sums[k] += recall_at_k(ranked, relevant, k)

    metrics["MRR@10"] = mrr_sum / n
    metrics["NDCG@10"] = ndcg10_sum / n
    for k in k_values:
        metrics[f"Recall@{k}"] = recall_sums[k] / n

    return metrics


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(args):
    corpus = SYNTHETIC_CORPUS
    queries_with_relevance = SYNTHETIC_QUERIES
    query_texts = [q for q, _ in queries_with_relevance]

    print(f"RAG Retrieval Benchmark")
    print(f"  Corpus: {len(corpus)} documents")
    print(f"  Queries: {len(query_texts)} queries")
    print(f"  Dataset: synthetic IR")
    print()

    results = {}

    # --- CrispEmbed ---
    if args.lib and args.gguf:
        for label, gguf_path in [("CrispEmbed F32", args.gguf),
                                   ("CrispEmbed Q8_0", args.gguf_q8)]:
            if not gguf_path:
                continue
            print(f"Encoding with {label}...")
            t0 = time.time()
            corpus_vecs = encode_crispembed(corpus, gguf_path, args.lib,
                                            prefix=args.prefix or "")
            query_vecs = encode_crispembed(query_texts, gguf_path, args.lib,
                                           prefix=args.query_prefix or args.prefix or "")
            elapsed = time.time() - t0
            print(f"  Encoded in {elapsed:.2f}s")

            metrics = evaluate_retrieval(
                query_vecs, corpus_vecs, queries_with_relevance, label
            )
            metrics["encode_time_s"] = elapsed
            results[label] = metrics

    # --- HuggingFace ---
    if not args.skip_hf and args.hf_model:
        label = "HuggingFace"
        print(f"Encoding with {label}...")
        try:
            t0 = time.time()
            corpus_vecs = encode_huggingface(corpus, args.hf_model,
                                              prefix=args.prefix or "")
            query_vecs = encode_huggingface(query_texts, args.hf_model,
                                             prefix=args.query_prefix or args.prefix or "")
            elapsed = time.time() - t0
            print(f"  Encoded in {elapsed:.2f}s")

            metrics = evaluate_retrieval(
                query_vecs, corpus_vecs, queries_with_relevance, label
            )
            metrics["encode_time_s"] = elapsed
            results[label] = metrics
        except ImportError:
            print("  Skipping HuggingFace (sentence-transformers not installed)")

    # --- fastembed-rs ---
    if args.fastembed_rs and args.fastembed_model:
        label = "fastembed-rs"
        print(f"Encoding with {label}...")
        corpus_vecs = encode_fastembed_rs(corpus, args.fastembed_rs,
                                           args.fastembed_model)
        if corpus_vecs is not None:
            query_vecs = encode_fastembed_rs(query_texts, args.fastembed_rs,
                                              args.fastembed_model)
            if query_vecs is not None:
                metrics = evaluate_retrieval(
                    query_vecs, corpus_vecs, queries_with_relevance, label
                )
                results[label] = metrics

    # --- Print results ---
    print()
    print("=" * 80)
    print(f"{'Engine':<25} {'MRR@10':>8} {'NDCG@10':>8} {'R@10':>8} {'R@100':>8} {'Time':>8}")
    print("-" * 80)
    for engine, m in results.items():
        print(f"{engine:<25} {m.get('MRR@10', 0):.4f}   {m.get('NDCG@10', 0):.4f}   "
              f"{m.get('Recall@10', 0):.4f}   {m.get('Recall@100', 0):.4f}   "
              f"{m.get('encode_time_s', 0):.2f}s")
    print("=" * 80)

    # --- Cross-engine consistency ---
    if len(results) >= 2:
        engines = list(results.keys())
        print(f"\nCross-engine comparison ({engines[0]} vs {engines[1]}):")
        for metric in ["MRR@10", "NDCG@10", "Recall@10"]:
            v0 = results[engines[0]].get(metric, 0)
            v1 = results[engines[1]].get(metric, 0)
            diff = abs(v0 - v1)
            status = "MATCH" if diff < 0.05 else "DIFF"
            print(f"  {metric}: {v0:.4f} vs {v1:.4f} (delta={diff:.4f}) [{status}]")


def main():
    parser = argparse.ArgumentParser(description="RAG Retrieval Quality Benchmark")
    parser.add_argument("--lib", help="Path to libcrispembed.so/.dylib")
    parser.add_argument("--gguf", help="Path to CrispEmbed GGUF model (F32)")
    parser.add_argument("--gguf-q8", help="Path to CrispEmbed GGUF model (Q8_0)")
    parser.add_argument("--hf-model", default="sentence-transformers/all-MiniLM-L6-v2",
                        help="HuggingFace model name")
    parser.add_argument("--skip-hf", action="store_true", help="Skip HuggingFace")
    parser.add_argument("--fastembed-rs", help="Path to fastembed-rs bench binary")
    parser.add_argument("--fastembed-model", default="AllMiniLML6V2",
                        help="fastembed-rs model enum name")
    parser.add_argument("--prefix", default="", help="Text prefix for documents")
    parser.add_argument("--query-prefix", default="", help="Text prefix for queries")
    parser.add_argument("--dataset", default="synthetic",
                        help="Dataset: 'synthetic' (built-in) or BEIR dataset name")

    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
