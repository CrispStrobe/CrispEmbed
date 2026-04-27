#!/usr/bin/env python3
"""CrispEmbed Python demo — text embedding, reranking, and RAG search."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))
from crispembed import CrispEmbed
import numpy as np

MODEL = os.environ.get("CRISPEMBED_MODEL")
if not MODEL:
    raise SystemExit("Set CRISPEMBED_MODEL to a GGUF path before running this demo.")
LIB = os.environ.get("CRISPEMBED_LIB", None)

print("=== CrispEmbed Python Demo ===\n")
model = CrispEmbed(MODEL, lib_path=LIB)
print(f"Model loaded: dim={model.dim}")

# 1. Dense encoding
vec = model.encode("Hello world")
print(f"\n1. Dense encode: shape={vec.shape}, norm={np.linalg.norm(vec):.4f}")

# 2. Batch encoding
texts = ["Machine learning is amazing", "The weather is nice", "Deep learning uses neural networks"]
vecs = model.encode(texts)
print(f"2. Batch encode: {len(texts)} texts -> shape={vecs.shape}")

# 3. Similarity search
query = "What is deep learning?"
q_vec = model.encode(query)
sims = vecs @ q_vec
print(f"\n3. Similarity search for: '{query}'")
for i, (text, sim) in enumerate(sorted(zip(texts, sims), key=lambda x: -x[1])):
    print(f"   {sim:.4f}  {text}")

# 4. Bi-encoder reranking
results = model.rerank_biencoder(query, texts, top_n=2)
print(f"\n4. Bi-encoder reranking (top 2):")
for r in results:
    print(f"   [{r['index']}] {r['score']:.4f}  {r['document']}")

# 5. Matryoshka truncation
model.set_dim(128)
vec128 = model.encode("Hello world")
print(f"\n5. Matryoshka: dim=128 -> shape={vec128.shape}, norm={np.linalg.norm(vec128):.4f}")
model.set_dim(0)

# 6. Prefix
model.set_prefix("query: ")
vec_pf = model.encode("Hello world")
print(f"6. With prefix '{model.prefix}': shape={vec_pf.shape}")
model.set_prefix("")

print("\nAll Python demo tests passed!")
