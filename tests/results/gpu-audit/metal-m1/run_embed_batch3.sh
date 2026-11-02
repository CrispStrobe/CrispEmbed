#!/bin/bash
B=/Volumes/backups/code/crispembed-metal-audit-20261101/build-metal/crispembed
M=/Volumes/backups/code/model-cache-20261101/metal-audit
H=$(cd "$(dirname "$0")" && pwd); P=~/miniconda3/bin/python; T=$H/embed_texts.txt
ab() { $P $H/ab_embed.py $B $M/$1 $2 $T $H/embed/${1%.gguf}.$2.json "${@:3}" || echo "FAIL $1 $2"; }
ab ettin-reranker-17m-v1-q8_0.gguf rerank --ref $M/ettin-reranker-17m-v1-f32.gguf
ab ettin-reranker-17m-v1-f32.gguf rerank
