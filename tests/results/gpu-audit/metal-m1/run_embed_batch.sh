#!/bin/bash
# Embedding/rerank/sparse/ColBERT Metal-vs-CPU batch (2026-11-01). Run under m1-gpu-lock.sh.
B=/Volumes/backups/code/crispembed-metal-audit-20261101/build-metal/crispembed
M=/Volumes/backups/code/model-cache-20261101/metal-audit
H=$(dirname "$0"); P=~/miniconda3/bin/python; T=$H/embed_texts.txt
mkdir -p $H/embed
ab() { $P $H/ab_embed.py $B $M/$1 $2 $T $H/embed/${1%.gguf}.$2.json "${@:3}" || echo "FAIL $1 $2"; }
ab all-MiniLM-L6-v2-q8_0.gguf embed
ab bge-small-en-v1.5-q8_0.gguf embed
ab multilingual-e5-small-q8_0.gguf embed
ab arctic-embed-m-v2-q4_k-imatrix.gguf embed
ab bge-m3-iq4_xs.gguf embed
ab bge-m3-iq4_xs.gguf sparse
ab bge-m3-iq4_xs.gguf colbert
ab ms-marco-MiniLM-L-6-v2-q8_0-g7c.gguf rerank
ab mxbai-rerank-xsmall-v1-q8_0-g7c.gguf rerank
ab ettin-reranker-17m-v1-q8_0.gguf rerank
ab Qwen3-Embedding-0.6B-Q8_0.gguf embed
ab f2llm-v2-0.6b-q4_k-imatrix.gguf embed
ab jina-v5-nano-q4_k.gguf embed
ab harrier-270m-q4_k.gguf embed
ab embeddinggemma-300m-qat-Q8_0.gguf embed
ab LFM2.5-Embedding-350M-Q8_0.gguf embed
