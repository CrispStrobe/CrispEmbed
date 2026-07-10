#!/usr/bin/env python3
"""tests/test_embeddinggemma_parity.py — EmbeddingGemma parity + matryoshka test.

Validates:
  1. Model loads and produces non-zero embeddings
  2. Cosine similarity between related/unrelated pairs is sane
  3. Matryoshka dimension truncation works (128, 256 dims)
  4. Truncated embeddings preserve ranking (related > unrelated)
  5. Batch encoding matches single encoding

Usage:
    PYTHONNOUSERSITE=1 python tests/test_embeddinggemma_parity.py \
        --model /mnt/storage/gguf-models/embeddinggemma-300m-q8_0.gguf

Requires: the crispembed shared library (build with CRISPEMBED_BUILD_SHARED=ON).
"""

import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
from crispembed import CrispEmbed


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to EmbeddingGemma GGUF')
    parser.add_argument('--lib', default=None, help='Path to libcrispembed.so')
    args = parser.parse_args()

    passed, failed = 0, 0
    failures = []

    def check(cond, msg):
        nonlocal passed, failed
        if cond:
            passed += 1
        else:
            failed += 1
            failures.append(msg)
            print(f'  FAIL: {msg}')

    print(f'Model: {args.model}')
    embed = CrispEmbed(args.model, lib_path=args.lib)
    print(f'  dim={embed.dim}')

    # --- Test 1: Basic encoding ---
    print('\n=== Basic encoding ===')
    v1 = embed.encode('The quick brown fox jumps over the lazy dog')
    check(v1 is not None, 'encode returns non-None')
    check(len(v1.shape) == 1, f'single text returns 1D array, got {v1.shape}')
    check(v1.shape[0] > 0, f'embedding dim > 0: {v1.shape[0]}')
    check(np.linalg.norm(v1) > 0.99, f'L2 normalized: norm={np.linalg.norm(v1):.4f}')
    print(f'  dim={v1.shape[0]}, norm={np.linalg.norm(v1):.4f}, v[:5]={v1[:5]}')

    # --- Test 2: Semantic similarity ---
    print('\n=== Semantic similarity ===')
    v_dog = embed.encode('A dog playing in the park')
    v_cat = embed.encode('A cat sleeping on the couch')
    v_code = embed.encode('The Python programming language uses indentation')

    sim_dog_cat = cosine(v_dog, v_cat)
    sim_dog_code = cosine(v_dog, v_code)
    print(f'  dog↔cat: {sim_dog_cat:.4f}')
    print(f'  dog↔code: {sim_dog_code:.4f}')
    check(sim_dog_cat > sim_dog_code, f'dog↔cat ({sim_dog_cat:.3f}) > dog↔code ({sim_dog_code:.3f})')
    check(sim_dog_cat > 0.3, f'related pair similarity > 0.3: {sim_dog_cat:.3f}')

    # --- Test 3: Matryoshka dimension truncation ---
    print('\n=== Matryoshka dimensions ===')
    full_dim = v1.shape[0]

    for mdim in [128, 256]:
        if mdim >= full_dim:
            print(f'  skip dim={mdim} (>= full_dim={full_dim})')
            continue

        v_trunc = embed.encode('The quick brown fox jumps over the lazy dog', matryoshka_dim=mdim)
        check(v_trunc.shape[0] == mdim, f'matryoshka_dim={mdim}: got shape {v_trunc.shape}')
        check(np.linalg.norm(v_trunc) > 0.99, f'matryoshka L2 normalized: {np.linalg.norm(v_trunc):.4f}')

        # Verify truncated embeddings preserve ranking
        vd_t = embed.encode('A dog playing in the park', matryoshka_dim=mdim)
        vc_t = embed.encode('A cat sleeping on the couch', matryoshka_dim=mdim)
        vx_t = embed.encode('The Python programming language uses indentation', matryoshka_dim=mdim)

        sim_dc = cosine(vd_t, vc_t)
        sim_dx = cosine(vd_t, vx_t)
        print(f'  dim={mdim}: dog↔cat={sim_dc:.4f}, dog↔code={sim_dx:.4f}')
        check(sim_dc > sim_dx, f'dim={mdim}: ranking preserved (dog↔cat > dog↔code)')

    # Verify full dim restored after matryoshka
    v_after = embed.encode('test')
    check(v_after.shape[0] == full_dim, f'full dim restored: {v_after.shape[0]} == {full_dim}')

    # --- Test 4: Batch encoding ---
    print('\n=== Batch encoding ===')
    texts = ['Hello world', 'Good morning', 'Machine learning is fun']
    batch = embed.encode(texts)
    check(batch.shape == (3, full_dim), f'batch shape: {batch.shape}')

    # Batch result should match individual encodes
    for i, t in enumerate(texts):
        v_single = embed.encode(t)
        sim = cosine(batch[i], v_single)
        check(sim > 0.999, f'batch[{i}] ≈ single: cos={sim:.6f}')

    # --- Results ---
    print(f'\n{"=" * 50}')
    print(f'Results: {passed} passed, {failed} failed')
    if failures:
        print('\nFailures:')
        for f in failures:
            print(f'  - {f}')
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
