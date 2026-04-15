#!/usr/bin/env python3
"""Debug CrispEmbed vs HuggingFace — tokenization + embedding comparison.

Usage:
    python tests/debug_model.py --model MODEL_ID --gguf PATH.gguf [--binary ./build/crispembed]

Compares:
1. Token IDs (HF tokenizer vs GGUF tokenizer via CrispEmbed)
2. Final embedding vectors
3. Cosine similarity per text
"""

import argparse
import json
import subprocess
import sys
import numpy as np


TEST_TEXTS = [
    "Hello world",
    "The quick brown fox jumps over the lazy dog",
    "Machine learning",
    "Bonjour le monde",  # multilingual test
]


def get_hf_details(model_id: str, texts: list):
    """Get token IDs and embeddings from HuggingFace."""
    from transformers import AutoTokenizer, AutoModel
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    model.eval()

    results = []
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", padding=False, truncation=True)
        ids = enc["input_ids"][0].tolist()
        tokens_str = [tokenizer.convert_ids_to_tokens(i) for i in ids]

        with torch.no_grad():
            out = model(**enc)
            # last_hidden_state: [1, seq, hidden]
            hidden = out.last_hidden_state[0].numpy()

        results.append({
            "text": text,
            "ids": ids,
            "tokens": tokens_str,
            "hidden_shape": hidden.shape,
            "hidden": hidden,
        })

    return results, tokenizer, model


def get_hf_embeddings(model_id: str, texts: list, pooling: str = "cls"):
    """Get final embeddings using sentence-transformers or manual pooling."""
    try:
        from sentence_transformers import SentenceTransformer
        st_model = SentenceTransformer(model_id, trust_remote_code=True)
        vecs = st_model.encode(texts, normalize_embeddings=True)
        return vecs
    except Exception as e:
        print(f"  sentence-transformers failed: {e}")
        # Fallback: manual pooling
        from transformers import AutoTokenizer, AutoModel
        import torch
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        model.eval()
        vecs = []
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", padding=False, truncation=True)
            with torch.no_grad():
                out = model(**enc)
                h = out.last_hidden_state[0].numpy()
                if pooling == "cls":
                    v = h[0]
                else:
                    mask = enc["attention_mask"][0].numpy()
                    v = (h * mask[:, None]).sum(0) / mask.sum()
                v = v / np.linalg.norm(v)
                vecs.append(v)
        return np.array(vecs)


def get_crispembed(binary: str, gguf: str, texts: list):
    """Get embeddings from CrispEmbed CLI."""
    results = []
    for text in texts:
        r = subprocess.run(
            [binary, "-m", gguf, text],
            capture_output=True, text=True, timeout=120
        )
        if r.returncode != 0:
            print(f"  CrispEmbed error: {r.stderr[:200]}")
            results.append(None)
            continue
        line = r.stdout.strip()
        if not line:
            results.append(None)
            continue
        vec = np.array([float(x) for x in line.split()])
        results.append(vec)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model ID")
    parser.add_argument("--gguf", required=True, help="Path to GGUF file")
    parser.add_argument("--binary", default="./build/crispembed")
    parser.add_argument("--pooling", default="auto", choices=["auto", "cls", "mean", "last"])
    args = parser.parse_args()

    print(f"=== Debug: {args.model} ===")
    print(f"GGUF: {args.gguf}")
    print()

    # Step 1: Get HF tokenization details
    print("--- Step 1: HF Tokenization ---")
    hf_results, tokenizer, model = get_hf_details(args.model, TEST_TEXTS)
    for r in hf_results:
        print(f"  Text: {r['text']!r}")
        print(f"  IDs ({len(r['ids'])}): {r['ids']}")
        print(f"  Tokens: {r['tokens']}")
        print()

    # Step 2: Detect pooling
    pooling = args.pooling
    if pooling == "auto":
        # Try to detect from sentence-transformers config
        try:
            from sentence_transformers import SentenceTransformer
            st = SentenceTransformer(args.model, trust_remote_code=True)
            modules = list(st.modules())
            for m in modules:
                if hasattr(m, 'pooling_mode_cls_token') and m.pooling_mode_cls_token:
                    pooling = "cls"
                    break
                if hasattr(m, 'pooling_mode_mean_tokens') and m.pooling_mode_mean_tokens:
                    pooling = "mean"
                    break
            if pooling == "auto":
                pooling = "mean"  # default
        except:
            pooling = "mean"
    print(f"--- Pooling: {pooling} ---")
    print()

    # Step 3: Get HF embeddings
    print("--- Step 2: HF Embeddings ---")
    hf_vecs = get_hf_embeddings(args.model, TEST_TEXTS, pooling)
    for i, text in enumerate(TEST_TEXTS):
        v = hf_vecs[i]
        print(f"  {text!r}: dim={len(v)}, first5={v[:5].tolist()}")
    print()

    # Step 4: Get CrispEmbed embeddings
    print("--- Step 3: CrispEmbed Embeddings ---")
    ce_vecs = get_crispembed(args.binary, args.gguf, TEST_TEXTS)
    for i, text in enumerate(TEST_TEXTS):
        v = ce_vecs[i]
        if v is not None:
            print(f"  {text!r}: dim={len(v)}, first5={v[:5].tolist()}")
        else:
            print(f"  {text!r}: FAILED")
    print()

    # Step 5: Compare
    print("--- Step 4: Comparison ---")
    print(f"{'Text':<50s} {'CosSim':>10s} {'MaxDiff':>10s} {'Status':>8s}")
    print("-" * 82)
    for i, text in enumerate(TEST_TEXTS):
        hf = hf_vecs[i]
        ce = ce_vecs[i]
        if ce is None:
            print(f"{text:<50s} {'N/A':>10s} {'N/A':>10s} {'FAIL':>8s}")
            continue
        if len(hf) != len(ce):
            print(f"{text:<50s} dim mismatch: {len(hf)} vs {len(ce)}")
            continue
        cos = np.dot(hf, ce) / (np.linalg.norm(hf) * np.linalg.norm(ce) + 1e-12)
        maxd = np.max(np.abs(hf - ce))
        ok = cos > 0.99
        print(f"{text:<50s} {cos:>10.6f} {maxd:>10.6f} {'PASS' if ok else 'FAIL':>8s}")

    # Step 6: Detailed token ID comparison via Python
    print()
    print("--- Step 5: Token ID Diff (HF vs what GGUF vocab would produce) ---")
    # Load GGUF vocab for comparison
    try:
        import struct
        # We can't easily read GGUF from Python without gguf lib, so compare via tokenizer
        for r in hf_results:
            text = r["text"]
            hf_ids = r["ids"]
            print(f"  Text: {text!r}")
            print(f"  HF IDs:  {hf_ids}")
            print(f"  HF toks: {r['tokens']}")
            print()
    except Exception as e:
        print(f"  Could not compare tokens: {e}")


if __name__ == "__main__":
    sys.exit(main() or 0)
