#!/usr/bin/env python3
"""Debug CrispEmbed vs HuggingFace — tokenization + embedding comparison.

Auto-detects architecture (BERT/XLM-R/Qwen3/Gemma3) from HF config.
Can also compare quantized GGUF variants against the F32 GGUF.

Usage:
    # Compare GGUF against HuggingFace ground truth
    python tests/debug_model.py --model sentence-transformers/all-MiniLM-L6-v2 \
        --gguf "$CRISPEMBED_GGUF"

    # Auto-detect HF model from GGUF filename
    python tests/debug_model.py --gguf "$CRISPEMBED_GGUF"

    # Compare quantized vs F32 GGUF (no HF needed)
    python tests/debug_model.py --gguf "$CRISPEMBED_Q8_GGUF" \
        --ref-gguf "$CRISPEMBED_F16_GGUF"

    # Just show tokenization details
    python tests/debug_model.py --model intfloat/multilingual-e5-small --tokens-only
"""

import argparse
import os
import subprocess
import sys
import numpy as np


TEST_TEXTS = [
    "Hello world",
    "The quick brown fox jumps over the lazy dog",
    "Machine learning",
    "Bonjour le monde",
]

# Auto-detect HF model ID from GGUF filename
GGUF_TO_HF = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "gte-small": "thenlper/gte-small",
    "arctic-embed-xs": "Snowflake/snowflake-arctic-embed-xs",
    "multilingual-e5-small": "intfloat/multilingual-e5-small",
    "pixie-rune-v1": "CrispStrobe/PIXIE-Rune-v1.0",
    "arctic-embed-l-v2": "Snowflake/snowflake-arctic-embed-l-v2.0",
    "octen-0.6b": "Octen/Octen-Embedding-0.6B",
    "f2llm-v2-0.6b": "codefuse-ai/F2LLM-v2-0.6B",
    "jina-v5-nano": "jinaai/jina-embeddings-v5-nano",
    "jina-v5-small": "jinaai/jina-embeddings-v5-small",
    "harrier-0.6b": "Harrier/Harrier-OSS-v1-0.6B",
    "harrier-270m": "Harrier/Harrier-OSS-v1-270M",
    "qwen3-embed-0.6b": "Qwen/Qwen3-Embedding-0.6B",
}


def guess_hf_id(gguf_path):
    """Guess HF model ID from GGUF filename."""
    base = os.path.basename(gguf_path).replace(".gguf", "")
    # Strip quant suffixes
    for suffix in ["-q8_0", "-q4_k", "-q5_k", "-q6_k", "-q4_0", "-q5_0", "-f16"]:
        base = base.replace(suffix, "")
    return GGUF_TO_HF.get(base)


def detect_arch(model_id):
    """Detect model architecture from HF config."""
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    model_type = getattr(config, "model_type", "unknown")
    arch_map = {
        "bert": "BERT",
        "roberta": "XLM-R",
        "xlm-roberta": "XLM-R",
        "qwen2": "Qwen3",
        "qwen3": "Qwen3",
        "gemma": "Gemma3",
        "gemma2": "Gemma3",
        "gemma3_text": "Gemma3",
    }
    arch = arch_map.get(model_type, model_type)
    return arch, config


def get_hf_tokens(model_id, texts):
    """Get tokenization details from HuggingFace."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    results = []
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", padding=False, truncation=True)
        ids = enc["input_ids"][0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(ids)
        results.append({"text": text, "ids": ids, "tokens": tokens})
    return results, tokenizer


def get_hf_embeddings(model_id, texts):
    """Get normalized embeddings from sentence-transformers."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_id, trust_remote_code=True)
        return model.encode(texts, normalize_embeddings=True)
    except Exception as e:
        print(f"  WARNING: sentence-transformers failed: {e}")
        return None


def get_crispembed(binary, gguf, texts):
    """Get embeddings from CrispEmbed CLI."""
    results = []
    for text in texts:
        try:
            r = subprocess.run([binary, "-m", gguf, text],
                               capture_output=True, text=True, timeout=120)
            if r.returncode != 0:
                results.append(None)
                continue
            vals = r.stdout.strip().split()
            if not vals:
                results.append(None)
                continue
            results.append(np.array([float(x) for x in vals]))
        except Exception:
            results.append(None)
    return results


def cos_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    parser = argparse.ArgumentParser(
        description="Debug CrispEmbed vs HuggingFace — auto-detects architecture")
    parser.add_argument("--model", help="HF model ID (auto-detected from GGUF if omitted)")
    parser.add_argument("--gguf", help="Path to GGUF file")
    parser.add_argument("--ref-gguf", help="Reference GGUF (e.g. F32) for quant-vs-quant comparison")
    parser.add_argument("--binary", default="./build/crispembed")
    parser.add_argument("--tokens-only", action="store_true", help="Only show tokenization")
    parser.add_argument("--texts", nargs="+", help="Custom test texts")
    args = parser.parse_args()

    if not args.model and not args.gguf:
        parser.print_help()
        return 1

    texts = args.texts or TEST_TEXTS

    # Auto-detect model
    model_id = args.model
    if not model_id and args.gguf:
        model_id = guess_hf_id(args.gguf)
        if model_id:
            print(f"Auto-detected HF model: {model_id}")
        else:
            print(f"Could not auto-detect HF model from {args.gguf}")
            if not args.ref_gguf:
                return 1

    # Detect architecture
    if model_id:
        arch, config = detect_arch(model_id)
        print(f"Architecture: {arch}")
        print(f"  model_type: {config.model_type}")
        print(f"  hidden_size: {getattr(config, 'hidden_size', '?')}")
        print(f"  num_layers: {getattr(config, 'num_hidden_layers', '?')}")
        print(f"  num_heads: {getattr(config, 'num_attention_heads', '?')}")
        if hasattr(config, 'num_key_value_heads'):
            print(f"  num_kv_heads: {config.num_key_value_heads}")
        print()

    # Step 1: Tokenization
    if model_id:
        print("=== Tokenization ===")
        tok_results, tokenizer = get_hf_tokens(model_id, texts)
        for r in tok_results:
            print(f"  '{r['text']}'")
            print(f"    IDs ({len(r['ids'])}): {r['ids'][:20]}{'...' if len(r['ids'])>20 else ''}")
            print(f"    Tokens: {r['tokens'][:15]}{'...' if len(r['tokens'])>15 else ''}")
        print()

    if args.tokens_only:
        return 0

    # Step 2: Get embeddings
    hf_vecs = None
    if model_id and not args.ref_gguf:
        print("=== HuggingFace Embeddings ===")
        hf_vecs = get_hf_embeddings(model_id, texts)
        if hf_vecs is not None:
            for i, text in enumerate(texts):
                v = hf_vecs[i]
                print(f"  '{text}': dim={len(v)}, norm={np.linalg.norm(v):.4f}, "
                      f"first3=[{v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f}]")
        print()

    ref_vecs = None
    if args.ref_gguf:
        print(f"=== Reference GGUF: {os.path.basename(args.ref_gguf)} ===")
        ref_vecs = get_crispembed(args.binary, args.ref_gguf, texts)
        for i, text in enumerate(texts):
            v = ref_vecs[i]
            if v is not None:
                print(f"  '{text}': dim={len(v)}, first3=[{v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f}]")
            else:
                print(f"  '{text}': FAILED")
        print()

    if args.gguf:
        print(f"=== CrispEmbed GGUF: {os.path.basename(args.gguf)} ===")
        ce_vecs = get_crispembed(args.binary, args.gguf, texts)
        for i, text in enumerate(texts):
            v = ce_vecs[i]
            if v is not None:
                print(f"  '{text}': dim={len(v)}, first3=[{v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f}]")
            else:
                print(f"  '{text}': FAILED")
        print()

        # Step 3: Comparison
        compare_to = hf_vecs if hf_vecs is not None else ref_vecs
        compare_label = "HF" if hf_vecs is not None else "Ref"

        if compare_to is not None:
            print(f"=== Comparison (CrispEmbed vs {compare_label}) ===")
            print(f"{'Text':<45s} {'CosSim':>10s} {'MaxDiff':>10s} {'Status':>8s}")
            print("-" * 77)
            all_pass = True
            for i, text in enumerate(texts):
                ref = compare_to[i] if isinstance(compare_to, np.ndarray) else compare_to[i]
                test = ce_vecs[i]
                if ref is None or test is None:
                    print(f"{text[:44]:<45s} {'N/A':>10s} {'N/A':>10s} {'FAIL':>8s}")
                    all_pass = False
                    continue
                if len(ref) != len(test):
                    print(f"{text[:44]:<45s} dim mismatch: {len(ref)} vs {len(test)}")
                    all_pass = False
                    continue
                cs = cos_sim(ref, test)
                md = float(np.max(np.abs(ref - test)))
                ok = cs > 0.99
                if not ok:
                    all_pass = False
                print(f"{text[:44]:<45s} {cs:>10.6f} {md:>10.6f} {'PASS' if ok else 'FAIL':>8s}")

            print()
            if all_pass:
                print("RESULT: ALL PASS")
            else:
                print("RESULT: SOME FAILURES — check tokenization and architecture details above")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
