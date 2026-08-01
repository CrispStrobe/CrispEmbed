#!/usr/bin/env python3
"""RAG retrieval quality benchmark: CrispEmbed F32/Q8_0/Q4_K vs HuggingFace.

Encodes a corpus + queries, retrieves top-k by cosine similarity,
computes MRR@10 and Recall@10. Self-contained test dataset (no download).

Usage:
    python tests/bench_rag.py                    # all models
    python tests/bench_rag.py all-MiniLM-L6-v2   # single model
"""

import subprocess, sys, time
import numpy as np
from pathlib import Path

import os
CLI = os.environ.get("CRISPEMBED_CLI") or str(Path(__file__).parent.parent / "build" / "crispembed")
if not Path(CLI).exists():
    CLI = "/tmp/crispembed-build/crispembed"
CACHE = os.environ.get("CRISPEMBED_RAG_CACHE", "/mnt/storage/crispembed_cache")

# Quant flavors to compare, in report order: (label, filename-suffix). The imatrix
# flavors are the shipped defaults — this benchmark checks they preserve actual
# retrieval RANKING (MRR/Recall), not just cosine agreement with f32.
FLAVORS = [
    ("Q8_0",        "-q8_0.gguf"),
    ("Q4_K",        "-q4_k.gguf"),
    ("Q4_K+imat",   "-q4_k-imatrix.gguf"),
    ("IQ4_XS+imat", "-iq4_xs.gguf"),
]

# ── Built-in IR test dataset ──────────────────────────────────────────
CORPUS = [
    "Artificial intelligence is a branch of computer science that aims to create intelligent machines.",
    "Machine learning is a subset of AI that enables systems to learn from data.",
    "Deep learning uses neural networks with many layers to model complex patterns.",
    "Natural language processing deals with interactions between computers and human language.",
    "Computer vision is a field of AI that trains computers to interpret visual information.",
    "Reinforcement learning is a type of machine learning where agents learn by interacting with environments.",
    "Transfer learning leverages pre-trained models to solve new but related problems.",
    "The Transformer architecture revolutionized NLP with its self-attention mechanism.",
    "BERT is a pre-trained language model that uses bidirectional context for understanding text.",
    "GPT models generate text by predicting the next token in a sequence.",
    "Convolutional neural networks are primarily used for image recognition tasks.",
    "Recurrent neural networks process sequential data using internal memory.",
    "Generative adversarial networks consist of a generator and discriminator competing against each other.",
    "Embeddings represent words or sentences as dense vectors in a continuous space.",
    "Vector databases store and search high-dimensional embeddings for similarity retrieval.",
    "RAG combines retrieval with generation to produce factually grounded responses.",
    "Fine-tuning adapts a pre-trained model to a specific downstream task.",
    "Tokenization splits text into subword units for processing by language models.",
    "Attention mechanisms allow models to focus on relevant parts of the input.",
    "Knowledge distillation transfers knowledge from a large model to a smaller one.",
    # ── Adjacent-topic distractors: semantically close, NOT answers — they make
    #    top-1 ranking harder so quant flavors can differentiate (indices 20+).
    "Supervised learning trains models on labelled examples to predict outcomes.",
    "Unsupervised learning finds structure in data without labelled targets.",
    "Gradient descent optimizes model parameters by following the loss gradient.",
    "Overfitting happens when a model memorizes training data and fails to generalize.",
    "Batch normalization stabilizes training by normalizing layer activations.",
    "Dropout regularizes neural networks by randomly deactivating units during training.",
    "A learning rate schedule adjusts the step size as training progresses.",
    "Cross-entropy loss measures the difference between predicted and true distributions.",
    "Data augmentation expands a training set by transforming existing examples.",
    "Hyperparameter tuning searches for the configuration that maximizes validation performance.",
    "Semantic search retrieves documents by meaning rather than exact keyword overlap.",
    "Cosine similarity measures the angle between two vectors regardless of magnitude.",
    "A bi-encoder embeds queries and documents independently for fast retrieval.",
    "A cross-encoder jointly scores a query-document pair for higher accuracy.",
    "Approximate nearest neighbour search speeds up similarity lookup in large indexes.",
]

QUERIES = [
    ("What is artificial intelligence?", [0, 1]),
    ("How does deep learning work?", [2, 11]),
    ("What is NLP?", [3, 7, 8]),
    ("How do computers understand images?", [4, 10]),
    ("What is reinforcement learning?", [5]),
    ("How does BERT work?", [8, 7]),
    ("What are word embeddings?", [13, 14]),
    ("What is retrieval augmented generation?", [15, 14]),
    ("How is a model fine-tuned?", [16, 6]),
    ("What is the attention mechanism?", [18, 7]),
    # Harder queries — the answer sits among close distractors (indices 20–34)
    ("How does semantic search find relevant documents?", [30]),
    ("What causes a neural network to overfit?", [23]),
    ("How are embeddings compared for similarity?", [31]),
    ("What is a bi-encoder used for in retrieval?", [32]),
    ("How is a query-document pair scored jointly?", [33]),
    ("What regularizes a network by dropping units?", [25]),
]

MODELS = [
    ("all-MiniLM-L6-v2",      "sentence-transformers/all-MiniLM-L6-v2"),
    ("bge-small-en-v1.5",     "BAAI/bge-small-en-v1.5"),
    ("bge-base-en-v1.5",      "BAAI/bge-base-en-v1.5"),
    ("all-mpnet-base-v2",     "sentence-transformers/all-mpnet-base-v2"),
    ("nomic-embed-text-v1.5", "nomic-ai/nomic-embed-text-v1.5"),
    ("mxbai-embed-large-v1",  "mixedbread-ai/mxbai-embed-large-v1"),
]


def encode_ce(model_path, texts):
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for t in texts:
            f.write(t + '\n')
        f.flush()
        r = subprocess.run([CLI, "-m", model_path, "-f", f.name, "--prefix", ""],
                           capture_output=True, text=True, timeout=120)
    Path(f.name).unlink()
    if r.returncode != 0:
        return None
    return np.array([[float(x) for x in line.split()] for line in r.stdout.strip().split('\n')])


def encode_hf(model_name, texts):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_name, trust_remote_code=True)
    return m.encode(texts, normalize_embeddings=True)


def metrics(q_embs, c_embs, queries, k=10):
    mrr, recall = 0, 0
    for i, (_, rels) in enumerate(queries):
        scores = c_embs @ q_embs[i]
        top = np.argsort(scores)[::-1][:k]
        for rank, d in enumerate(top, 1):
            if d in rels:
                mrr += 1.0 / rank
                break
        recall += sum(1 for d in top if d in rels) / len(rels)
    n = len(queries)
    return mrr / n, recall / n


def run(name, gguf, hf_name):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    qtexts = [q for q, _ in QUERIES]

    # F32
    f32 = gguf
    if Path(f32).exists():
        t0 = time.time()
        c = encode_ce(f32, CORPUS); q = encode_ce(f32, qtexts)
        dt = time.time() - t0
        if c is not None and q is not None:
            m, r = metrics(q, c, QUERIES)
            print(f"  F32:  MRR@10={m:.4f}  Recall@10={r:.4f}  {dt:.1f}s  dim={c.shape[1]}")
            f32_c = c
        else:
            print(f"  F32: FAILED"); f32_c = None
    else:
        print(f"  F32: not found"); f32_c = None

    # Reference for cos_vs_ref: f32 if available, else the first flavor that loads.
    ref_c = f32_c
    for label, suffix in FLAVORS:
        fp = gguf.replace('.gguf', suffix)
        if not Path(fp).exists():
            continue
        c = encode_ce(fp, CORPUS); q = encode_ce(fp, qtexts)
        if c is None or q is None:
            print(f"  {label:11s} FAILED to encode"); continue
        if ref_c is None:
            ref_c = c  # anchor on the highest-precision flavor present (Q8_0)
        m, r = metrics(q, c, QUERIES)
        cos = np.mean([np.dot(c[i], ref_c[i]) for i in range(len(CORPUS))])
        print(f"  {label:11s} MRR@10={m:.4f}  Recall@10={r:.4f}  cos_vs_ref={cos:.6f}")

    # HuggingFace (skip with CRISPEMBED_RAG_NO_HF=1 — avoids loading torch/ST)
    if hf_name and not os.environ.get("CRISPEMBED_RAG_NO_HF"):
        try:
            t0 = time.time()
            hc = encode_hf(hf_name, CORPUS); hq = encode_hf(hf_name, qtexts)
            dt = time.time() - t0
            m, r = metrics(hq, hc, QUERIES)
            cos = np.mean([np.dot(hc[i], f32_c[i]) for i in range(len(CORPUS))]) if f32_c is not None else 0
            print(f"  HF:   MRR@10={m:.4f}  Recall@10={r:.4f}  {dt:.1f}s  cos_vs_CE={cos:.6f}")
        except Exception as e:
            print(f"  HF: FAILED ({e})")


def main():
    print("RAG Retrieval Quality Benchmark")
    print(f"Corpus: {len(CORPUS)} docs, Queries: {len(QUERIES)}, CLI: {CLI}")

    target = sys.argv[1] if len(sys.argv) > 1 else None
    for name, hf in MODELS:
        if target and target != name:
            continue
        gguf = f"{CACHE}/{name}.gguf"
        run(name, gguf, hf)

    print(f"\n{'='*60}")
    print("Done.")


if __name__ == "__main__":
    main()
