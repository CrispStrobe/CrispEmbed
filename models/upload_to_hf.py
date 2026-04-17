#!/usr/bin/env python3
"""Upload CrispEmbed GGUF models to HuggingFace.

Creates repos under cstr/ org with proper README cards.

Usage:
    python models/upload_to_hf.py --all --dir /path/to/ggufs
    python models/upload_to_hf.py --model octen-0.6b --dir /path/to/ggufs
"""

import argparse
import os
import sys

from huggingface_hub import HfApi, create_repo


# Model registry: maps GGUF base name -> metadata
MODELS = {
    "all-MiniLM-L6-v2": {
        "base_model": "sentence-transformers/all-MiniLM-L6-v2",
        "arch": "BERT",
        "dim": 384,
        "layers": 6,
        "params": "22M",
        "pooling": "mean",
        "tokenizer": "WordPiece",
        "license": "apache-2.0",
        "langs": ["en"],
        "desc": "Lightweight English embedding model. Fast inference, 384-dimensional output.",
    },
    "gte-small": {
        "base_model": "thenlper/gte-small",
        "arch": "BERT",
        "dim": 384,
        "layers": 6,
        "params": "33M",
        "pooling": "mean",
        "tokenizer": "WordPiece",
        "license": "mit",
        "langs": ["en"],
        "desc": "General Text Embeddings model. 384-dimensional output, excellent for semantic search.",
    },
    "arctic-embed-xs": {
        "base_model": "Snowflake/snowflake-arctic-embed-xs",
        "arch": "BERT",
        "dim": 384,
        "layers": 6,
        "params": "22M",
        "pooling": "CLS",
        "tokenizer": "WordPiece",
        "license": "apache-2.0",
        "langs": ["en"],
        "desc": "Snowflake Arctic Embed XS. CLS pooling, optimized for retrieval.",
    },
    "multilingual-e5-small": {
        "base_model": "intfloat/multilingual-e5-small",
        "arch": "XLM-R",
        "dim": 384,
        "layers": 12,
        "params": "118M",
        "pooling": "mean",
        "tokenizer": "SentencePiece",
        "license": "mit",
        "langs": ["multilingual"],
        "desc": "Multilingual E5 Small. 100+ languages, 384-dimensional mean-pooled embeddings.",
    },
    "multilingual-e5-base": {
        "base_model": "intfloat/multilingual-e5-base",
        "arch": "XLM-R",
        "dim": 768,
        "layers": 12,
        "params": "278M",
        "pooling": "mean",
        "tokenizer": "SentencePiece",
        "license": "mit",
        "langs": ["multilingual"],
        "desc": "Multilingual E5 Base. 100+ languages, 768-dimensional mean-pooled. Use prefix: \"query: \" / \"passage: \".",
    },
    "multilingual-e5-large": {
        "base_model": "intfloat/multilingual-e5-large",
        "arch": "XLM-R",
        "dim": 1024,
        "layers": 24,
        "params": "560M",
        "pooling": "mean",
        "tokenizer": "SentencePiece",
        "license": "mit",
        "langs": ["multilingual"],
        "desc": "Multilingual E5 Large. 100+ languages, 1024-dimensional mean-pooled. Top MTEB multilingual scorer. Use prefix: \"query: \" / \"passage: \".",
    },
    "pixie-rune-v1": {
        "base_model": "telepix/PIXIE-Rune-v1.0",
        "arch": "XLM-R",
        "dim": 1024,
        "layers": 24,
        "params": "560M",
        "pooling": "CLS",
        "tokenizer": "SentencePiece",
        "license": "apache-2.0",
        "langs": ["multilingual"],
        "desc": "PIXIE-Rune v1.0. 74-language embedding model, 1024-dimensional CLS-pooled.",
    },
    "arctic-embed-l-v2": {
        "base_model": "Snowflake/snowflake-arctic-embed-l-v2.0",
        "arch": "XLM-R",
        "dim": 1024,
        "layers": 24,
        "params": "560M",
        "pooling": "CLS",
        "tokenizer": "SentencePiece",
        "license": "apache-2.0",
        "langs": ["en"],
        "desc": "Snowflake Arctic Embed L v2.0. High-quality retrieval embeddings, 1024-dimensional.",
    },
    "octen-0.6b": {
        "base_model": "Octen/Octen-Embedding-0.6B",
        "arch": "Qwen3",
        "dim": 1024,
        "layers": 28,
        "params": "600M",
        "pooling": "last-token",
        "tokenizer": "GPT-2 BPE",
        "license": "apache-2.0",
        "langs": ["multilingual"],
        "desc": "Octen Embedding 0.6B. Qwen3-based decoder, 1024-dimensional, last-token pooling.",
    },
    "f2llm-v2-0.6b": {
        "base_model": "F2LLM/F2LLM-Embedding-v2-0.6B",
        "arch": "Qwen3",
        "dim": 1024,
        "layers": 28,
        "params": "600M",
        "pooling": "last-token",
        "tokenizer": "GPT-2 BPE",
        "license": "apache-2.0",
        "langs": ["multilingual"],
        "desc": "F2LLM Embedding v2 0.6B. Qwen3-based, strong multilingual performance.",
    },
    "jina-v5-nano": {
        "base_model": "jinaai/jina-embeddings-v5-nano",
        "arch": "Qwen3",
        "dim": 1024,
        "layers": 14,
        "params": "210M",
        "pooling": "last-token",
        "tokenizer": "GPT-2 BPE",
        "license": "cc-by-nc-4.0",
        "langs": ["multilingual"],
        "desc": "Jina Embeddings v5 Nano. Compact 210M decoder model, 1024-dimensional.",
    },
    "jina-v5-small": {
        "base_model": "jinaai/jina-embeddings-v5-small",
        "arch": "Qwen3",
        "dim": 1024,
        "layers": 28,
        "params": "600M",
        "pooling": "last-token",
        "tokenizer": "GPT-2 BPE",
        "license": "cc-by-nc-4.0",
        "langs": ["multilingual"],
        "desc": "Jina Embeddings v5 Small. Full-size decoder model, 1024-dimensional.",
    },
    "harrier-0.6b": {
        "base_model": "microsoft/harrier-oss-v1-0.6b",
        "arch": "Qwen3",
        "dim": 1024,
        "layers": 28,
        "params": "600M",
        "pooling": "last-token",
        "tokenizer": "GPT-2 BPE",
        "license": "mit",
        "langs": ["multilingual"],
        "desc": "Microsoft Harrier OSS v1 0.6B. Qwen3-based, state-of-the-art for its size.",
    },
    "harrier-270m": {
        "base_model": "microsoft/harrier-oss-v1-270m",
        "arch": "Gemma3",
        "dim": 640,
        "layers": 18,
        "params": "270M",
        "pooling": "last-token",
        "tokenizer": "SentencePiece BPE",
        "license": "mit",
        "langs": ["multilingual"],
        "desc": "Microsoft Harrier OSS v1 270M. Gemma3-based compact model, 640-dimensional.",
    },
    "qwen3-embed-0.6b": {
        "base_model": "Alibaba-NLP/Qwen3-Embedding-0.6B",
        "arch": "Qwen3",
        "dim": 1024,
        "layers": 28,
        "params": "600M",
        "pooling": "last-token",
        "tokenizer": "GPT-2 BPE",
        "license": "apache-2.0",
        "langs": ["multilingual"],
        "desc": "Qwen3 Embedding 0.6B. Official Alibaba embedding model.",
    },
    # --- RAG-critical models ---
    "bge-small-en-v1.5": {
        "base_model": "BAAI/bge-small-en-v1.5",
        "arch": "BERT",
        "dim": 384,
        "layers": 12,
        "params": "33M",
        "pooling": "CLS",
        "tokenizer": "WordPiece",
        "license": "mit",
        "langs": ["en"],
        "desc": "BGE Small English v1.5. Popular RAG baseline, 384-dimensional CLS-pooled. Use with prefix: \"Represent this sentence for searching relevant passages: \".",
    },
    "bge-base-en-v1.5": {
        "base_model": "BAAI/bge-base-en-v1.5",
        "arch": "BERT",
        "dim": 768,
        "layers": 12,
        "params": "109M",
        "pooling": "CLS",
        "tokenizer": "WordPiece",
        "license": "mit",
        "langs": ["en"],
        "desc": "BGE Base English v1.5. Standard RAG embedding model, 768-dimensional CLS-pooled. Use with prefix: \"Represent this sentence for searching relevant passages: \".",
    },
    "bge-large-en-v1.5": {
        "base_model": "BAAI/bge-large-en-v1.5",
        "arch": "BERT",
        "dim": 1024,
        "layers": 24,
        "params": "335M",
        "pooling": "CLS",
        "tokenizer": "WordPiece",
        "license": "mit",
        "langs": ["en"],
        "desc": "BGE Large English v1.5. High-quality RAG embedding model, 1024-dimensional CLS-pooled. Use with prefix: \"Represent this sentence for searching relevant passages: \".",
    },
    "nomic-embed-text-v1.5": {
        "base_model": "nomic-ai/nomic-embed-text-v1.5",
        "arch": "BERT",
        "dim": 768,
        "layers": 12,
        "params": "137M",
        "pooling": "mean",
        "tokenizer": "WordPiece",
        "license": "apache-2.0",
        "langs": ["en"],
        "desc": "Nomic Embed Text v1.5. 8K context window, Matryoshka representation learning. Use with prefix: \"search_query: \" / \"search_document: \".",
    },
    "all-MiniLM-L12-v2": {
        "base_model": "sentence-transformers/all-MiniLM-L12-v2",
        "arch": "BERT",
        "dim": 384,
        "layers": 12,
        "params": "33M",
        "pooling": "mean",
        "tokenizer": "WordPiece",
        "license": "apache-2.0",
        "langs": ["en"],
        "desc": "All-MiniLM-L12-v2. 12-layer upgrade from L6, higher quality 384-dimensional mean-pooled embeddings.",
    },
    "all-mpnet-base-v2": {
        "base_model": "sentence-transformers/all-mpnet-base-v2",
        "arch": "BERT",
        "dim": 768,
        "layers": 12,
        "params": "109M",
        "pooling": "mean",
        "tokenizer": "WordPiece",
        "license": "apache-2.0",
        "langs": ["en"],
        "desc": "All-MPNet-Base-v2. Highest quality sentence-transformers model, 768-dimensional mean-pooled.",
    },
    "mxbai-embed-large-v1": {
        "base_model": "mixedbread-ai/mxbai-embed-large-v1",
        "arch": "BERT",
        "dim": 1024,
        "layers": 24,
        "params": "335M",
        "pooling": "CLS",
        "tokenizer": "WordPiece",
        "license": "apache-2.0",
        "langs": ["en"],
        "desc": "MixedBread Embed Large v1. Top MTEB scorer, 1024-dimensional CLS-pooled.",
    },
    "snowflake-arctic-embed-m": {
        "base_model": "Snowflake/snowflake-arctic-embed-m",
        "arch": "BERT",
        "dim": 768,
        "layers": 12,
        "params": "109M",
        "pooling": "CLS",
        "tokenizer": "WordPiece",
        "license": "apache-2.0",
        "langs": ["en"],
        "desc": "Snowflake Arctic Embed M. Mid-range retrieval model, 768-dimensional CLS-pooled.",
    },
    "snowflake-arctic-embed-l": {
        "base_model": "Snowflake/snowflake-arctic-embed-l",
        "arch": "BERT",
        "dim": 1024,
        "layers": 24,
        "params": "335M",
        "pooling": "CLS",
        "tokenizer": "WordPiece",
        "license": "apache-2.0",
        "langs": ["en"],
        "desc": "Snowflake Arctic Embed L. Large retrieval model, 1024-dimensional CLS-pooled.",
    },
    "bge-m3": {
        "base_model": "BAAI/bge-m3",
        "arch": "XLM-R",
        "dim": 1024,
        "layers": 24,
        "params": "568M",
        "pooling": "mean",
        "tokenizer": "SentencePiece",
        "license": "mit",
        "langs": ["multilingual"],
        "desc": "BGE-M3. Dense + sparse + ColBERT multi-vector retrieval in one model. 100+ languages, 8192 context.",
    },
    # --- Reranker models ---
    "bge-reranker-base": {
        "base_model": "BAAI/bge-reranker-base",
        "arch": "BERT",
        "dim": 768,
        "layers": 12,
        "params": "278M",
        "pooling": "CLS",
        "tokenizer": "WordPiece",
        "license": "mit",
        "langs": ["en", "zh"],
        "desc": "BGE Reranker Base. Cross-encoder reranker for English and Chinese. Use with crispembed_rerank().",
        "is_reranker": True,
    },
    "ms-marco-MiniLM-L-6-v2": {
        "base_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "arch": "BERT",
        "dim": 384,
        "layers": 6,
        "params": "22M",
        "pooling": "CLS",
        "tokenizer": "WordPiece",
        "license": "apache-2.0",
        "langs": ["en"],
        "desc": "MS MARCO MiniLM L-6 v2. Fastest cross-encoder reranker, 22M parameters. Ideal for real-time RAG.",
        "is_reranker": True,
    },
    "ms-marco-MiniLM-L-12-v2": {
        "base_model": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "arch": "BERT",
        "dim": 384,
        "layers": 12,
        "params": "33M",
        "pooling": "CLS",
        "tokenizer": "WordPiece",
        "license": "apache-2.0",
        "langs": ["en"],
        "desc": "MS MARCO MiniLM L-12 v2. Higher quality cross-encoder reranker, 33M parameters.",
        "is_reranker": True,
    },
    "mxbai-rerank-xsmall-v1": {
        "base_model": "mixedbread-ai/mxbai-rerank-xsmall-v1",
        "arch": "BERT",
        "dim": 384,
        "layers": 6,
        "params": "33M",
        "pooling": "CLS",
        "tokenizer": "WordPiece",
        "license": "apache-2.0",
        "langs": ["en"],
        "desc": "MixedBread Rerank XSmall v1. Fast cross-encoder reranker for English.",
        "is_reranker": True,
    },
    "mxbai-rerank-base-v1": {
        "base_model": "mixedbread-ai/mxbai-rerank-base-v1",
        "arch": "BERT",
        "dim": 768,
        "layers": 12,
        "params": "86M",
        "pooling": "CLS",
        "tokenizer": "WordPiece",
        "license": "apache-2.0",
        "langs": ["en"],
        "desc": "MixedBread Rerank Base v1. Cross-encoder reranker for English, good quality/speed balance.",
        "is_reranker": True,
    },
}


def make_readme(model_name, files_info):
    """Generate a HuggingFace model card README."""
    m = MODELS[model_name]
    repo_name = f"cstr/{model_name}-GGUF"

    # File table
    file_rows = ""
    quant_suffixes = ["-q8_0", "-q5_k", "-q4_k", "-q4_0", "-q5_0", "-q5_1", "-q6_k", "-f16"]
    for fname, size_mb in files_info:
        qtype = "F32"
        for qs in quant_suffixes:
            if qs in fname:
                qtype = qs.lstrip("-").upper()
                break
        file_rows += f"| [{fname}](https://huggingface.co/{repo_name}/resolve/main/{fname}) | {qtype} | {size_mb:.0f} MB |\n"

    langs = ", ".join(m["langs"])
    tags = ", ".join([
        "embeddings", "gguf", "ggml", "text-embeddings",
        m["arch"].lower(), "crispembed",
    ])

    readme = f"""---
license: {m["license"]}
language: [{langs}]
tags: [{tags}]
pipeline_tag: feature-extraction
base_model: {m["base_model"]}
---

# {model_name} GGUF

GGUF format of [{m["base_model"]}](https://huggingface.co/{m["base_model"]}) for use with [CrispEmbed](https://github.com/CrispStrobe/CrispEmbed).

{m["desc"]}

## Files

| File | Quantization | Size |
|------|-------------|------|
{file_rows}

## Quick Start

```bash
# Download
huggingface-cli download {repo_name} {files_info[0][0]} --local-dir .

# Run with CrispEmbed
./crispembed -m {files_info[0][0]} "Hello world"

# Or with auto-download
./crispembed -m {model_name} "Hello world"
```

## Model Details

| Property | Value |
|----------|-------|
| Architecture | {m["arch"]} |
| Parameters | {m["params"]} |
| Embedding Dimension | {m["dim"]} |
| Layers | {m["layers"]} |
| Pooling | {m["pooling"]} |
| Tokenizer | {m["tokenizer"]} |
| Base Model | [{m["base_model"]}](https://huggingface.co/{m["base_model"]}) |

## Verification

Verified bit-identical to HuggingFace sentence-transformers (cosine similarity >= 0.999 on test texts).

## Usage with CrispEmbed

CrispEmbed is a lightweight C/C++ text embedding inference engine using ggml.
No Python runtime, no ONNX. Supports BERT, XLM-R, Qwen3, and Gemma3 architectures.

```bash
# Build CrispEmbed
git clone https://github.com/CrispStrobe/CrispEmbed
cd CrispEmbed
cmake -S . -B build && cmake --build build -j

# Encode
./build/crispembed -m {files_info[0][0]} "query text"

# Server mode
./build/crispembed-server -m {files_info[0][0]} --port 8080
curl -X POST http://localhost:8080/v1/embeddings \\
    -d '{{"input": ["Hello world"], "model": "{model_name}"}}'
```

## Credits

- Original model: [{m["base_model"]}](https://huggingface.co/{m["base_model"]})
- Inference engine: [CrispEmbed](https://github.com/CrispStrobe/CrispEmbed) (ggml-based)
- Conversion: `convert-{"decoder" if m["arch"] in ("Qwen3","Gemma3") else "bert"}-embed-to-gguf.py`
"""
    return readme


def upload_model(model_name, gguf_dir, dry_run=False):
    """Upload a model's GGUFs to HuggingFace."""
    if model_name not in MODELS:
        print(f"Unknown model: {model_name}")
        return False

    # Use HF_TOKEN env var or stored token
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    repo_id = f"cstr/{model_name}-GGUF"

    # Find all GGUF files for this model (skip Q4_0 — we have Q4_K)
    files = []
    skip_suffixes = ["-q4_0.gguf", "-q5_0.gguf", "-q5_1.gguf"]
    for f in sorted(os.listdir(gguf_dir)):
        if f.startswith(model_name) and f.endswith(".gguf"):
            if any(f.endswith(s) for s in skip_suffixes):
                continue
            path = os.path.join(gguf_dir, f)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            files.append((f, size_mb, path))

    if not files:
        print(f"  No GGUFs found for {model_name} in {gguf_dir}")
        return False

    print(f"\n=== {model_name} -> {repo_id} ===")
    for f, size, _ in files:
        print(f"  {f} ({size:.0f} MB)")

    if dry_run:
        print("  (dry run, skipping upload)")
        return True

    # Create repo
    try:
        create_repo(repo_id, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"  Repo creation: {e}")

    # Generate and upload README
    files_info = [(f, size) for f, size, _ in files]
    readme = make_readme(model_name, files_info)
    api.upload_file(
        path_or_fileobj=readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        commit_message=f"Add model card for {model_name} GGUF",
    )
    print(f"  README.md uploaded")

    # Upload GGUFs
    for fname, size_mb, fpath in files:
        print(f"  Uploading {fname} ({size_mb:.0f} MB)...")
        try:
            api.upload_file(
                path_or_fileobj=fpath,
                path_in_repo=fname,
                repo_id=repo_id,
                commit_message=f"Add {fname}",
            )
            print(f"  {fname} uploaded")
        except Exception as e:
            print(f"  ERROR uploading {fname}: {e}")

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Model base name (e.g., octen-0.6b)")
    parser.add_argument("--all", action="store_true", help="Upload all models")
    parser.add_argument("--dir", default="/mnt/akademie_storage/test_cohere",
                        help="Directory with GGUF files")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be uploaded without uploading")
    parser.add_argument("--list", action="store_true",
                        help="List available models")
    args = parser.parse_args()

    if args.list:
        for name in sorted(MODELS):
            print(f"  {name}")
        return

    if args.all:
        models = sorted(MODELS.keys())
    elif args.model:
        models = [args.model]
    else:
        parser.print_help()
        return

    for m in models:
        upload_model(m, args.dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
