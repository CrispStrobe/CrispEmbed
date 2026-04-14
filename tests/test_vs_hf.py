#!/usr/bin/env python3
"""Compare CrispEmbed GGUF output with HuggingFace reference.

Usage:
    python tests/test_vs_hf.py --model MODEL_ID --gguf PATH.gguf --binary ./build/crispembed
"""

import argparse
import subprocess
import sys
import numpy as np


TEST_TEXTS = [
    "Hello world",
    "The quick brown fox jumps over the lazy dog",
    "Machine learning is transforming the world of natural language processing",
    "This is a test of the emergency broadcast system",
    "",  # empty string edge case
]


def get_hf_embeddings(model_id: str, texts: list[str], pooling: str = "mean") -> np.ndarray:
    """Get embeddings from HuggingFace sentence-transformers."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_id, trust_remote_code=True)
        # Filter empty strings (HF handles them but may produce zeros)
        valid = [(i, t) for i, t in enumerate(texts) if t.strip()]
        if not valid:
            return np.zeros((len(texts), model.get_sentence_embedding_dimension()))
        indices, valid_texts = zip(*valid)
        vecs = model.encode(list(valid_texts), normalize_embeddings=True)
        result = np.zeros((len(texts), vecs.shape[1]))
        for idx, vec in zip(indices, vecs):
            result[idx] = vec
        return result
    except Exception as e:
        print(f"HF error: {e}", file=sys.stderr)
        return None


def get_crispembed_embeddings(binary: str, gguf: str, texts: list[str]) -> np.ndarray:
    """Get embeddings from CrispEmbed CLI."""
    results = []
    for text in texts:
        if not text.strip():
            results.append(None)
            continue
        try:
            r = subprocess.run(
                [binary, "-m", gguf, text],
                capture_output=True, text=True, timeout=60
            )
            if r.returncode != 0:
                print(f"CrispEmbed error: {r.stderr[:200]}", file=sys.stderr)
                results.append(None)
                continue
            # Parse space-separated floats from stdout (skip stderr)
            line = r.stdout.strip()
            if not line:
                results.append(None)
                continue
            vec = np.array([float(x) for x in line.split()])
            results.append(vec)
        except Exception as e:
            print(f"CrispEmbed error: {e}", file=sys.stderr)
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
    parser.add_argument("--model", required=True, help="HF model ID")
    parser.add_argument("--gguf", required=True, help="Path to GGUF file")
    parser.add_argument("--binary", default="./build/crispembed", help="CrispEmbed binary")
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"GGUF:  {args.gguf}")
    print(f"Texts: {len(TEST_TEXTS)}")
    print()

    # Get HF reference
    print("Getting HF reference embeddings...")
    hf_vecs = get_hf_embeddings(args.model, TEST_TEXTS)
    if hf_vecs is None:
        print("FAIL: could not get HF embeddings")
        return 1

    # Get CrispEmbed output
    print("Getting CrispEmbed embeddings...")
    ce_vecs = get_crispembed_embeddings(args.binary, args.gguf, TEST_TEXTS)
    if ce_vecs is None:
        print("FAIL: could not get CrispEmbed embeddings")
        return 1

    print(f"\nDim: HF={hf_vecs.shape[1]}, CE={ce_vecs.shape[1]}")
    if hf_vecs.shape[1] != ce_vecs.shape[1]:
        print(f"FAIL: dimension mismatch ({hf_vecs.shape[1]} vs {ce_vecs.shape[1]})")
        return 1

    # Compare
    print("\n{:<60s} {:>10s} {:>10s} {:>10s}".format("Text", "MaxDiff", "CosSim", "Status"))
    print("-" * 95)
    all_pass = True
    for i, text in enumerate(TEST_TEXTS):
        if not text.strip():
            print(f"{'(empty)':<60s} {'skip':>10s} {'skip':>10s} {'SKIP':>10s}")
            continue

        hf = hf_vecs[i]
        ce = ce_vecs[i]
        max_diff = np.max(np.abs(hf - ce))
        cos_sim = np.dot(hf, ce) / (np.linalg.norm(hf) * np.linalg.norm(ce) + 1e-12)
        ok = cos_sim > 0.99 and max_diff < 0.05
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        label = text[:57] + "..." if len(text) > 60 else text
        print(f"{label:<60s} {max_diff:>10.6f} {cos_sim:>10.6f} {status:>10s}")

    print()
    if all_pass:
        print("ALL PASS — CrispEmbed matches HuggingFace reference")
        return 0
    else:
        print("SOME TESTS FAILED — check output above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
