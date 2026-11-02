#!/bin/bash
# Batch 2: reference-anchored arms (higher-precision GGUF on CPU) + float-weight Metal-vs-CPU arms.
B=/Volumes/backups/code/crispembed-metal-audit-20261101/build-metal/crispembed
M=/Volumes/backups/code/model-cache-20261101/metal-audit
H=$(cd "$(dirname "$0")" && pwd); P=~/miniconda3/bin/python; T=$H/embed_texts.txt
ab() { $P $H/ab_embed.py $B $M/$1 $2 $T $H/embed/${1%.gguf}.$2.json "${@:3}" || echo "FAIL $1 $2"; }
ab all-MiniLM-L6-v2-q8_0.gguf embed --ref $M/all-MiniLM-L6-v2-f32.gguf
ab all-MiniLM-L6-v2-f32.gguf embed
ab bge-small-en-v1.5-q8_0.gguf embed --ref $M/bge-small-en-v1.5.gguf
ab f2llm-v2-0.6b-q4_k-imatrix.gguf embed --ref $M/f2llm-v2-0.6b.gguf
ab f2llm-v2-0.6b.gguf embed
ab jina-v5-nano-q4_k.gguf embed --ref $M/jina-v5-nano.gguf
ab jina-v5-nano.gguf embed
ab ms-marco-MiniLM-L-6-v2-q8_0-g7c.gguf rerank --ref $M/ms-marco-MiniLM-L-6-v2-g7c.gguf
ab ms-marco-MiniLM-L-6-v2-g7c.gguf rerank
ab mxbai-rerank-xsmall-v1-q8_0-g7c.gguf rerank
ab ettin-reranker-17m-v1-q8_0.gguf rerank --repeat 3
