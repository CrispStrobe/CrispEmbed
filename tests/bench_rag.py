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

CLI = str(Path(__file__).parent.parent / "build" / "crispembed")
if not Path(CLI).exists():
    CLI = "/tmp/crispembed-build/crispembed"
CACHE = "/mnt/storage/crispembed_cache"

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

    # Q8_0
    q8 = gguf.replace('.gguf', '-q8_0.gguf')
    if Path(q8).exists() and f32_c is not None:
        c = encode_ce(q8, CORPUS); q = encode_ce(q8, qtexts)
        if c is not None:
            m, r = metrics(q, c, QUERIES)
            cos = np.mean([np.dot(c[i], f32_c[i]) for i in range(len(CORPUS))])
            print(f"  Q8_0: MRR@10={m:.4f}  Recall@10={r:.4f}  cos_vs_f32={cos:.6f}")

    # Q4_K
    q4 = gguf.replace('.gguf', '-q4_k.gguf')
    if Path(q4).exists() and f32_c is not None:
        c = encode_ce(q4, CORPUS); q = encode_ce(q4, qtexts)
        if c is not None:
            m, r = metrics(q, c, QUERIES)
            cos = np.mean([np.dot(c[i], f32_c[i]) for i in range(len(CORPUS))])
            print(f"  Q4_K: MRR@10={m:.4f}  Recall@10={r:.4f}  cos_vs_f32={cos:.6f}")

    # HuggingFace
    if hf_name:
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
