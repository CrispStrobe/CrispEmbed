#!/bin/bash
# VLM / document-OCR Metal-vs-CPU batch (2026-11-01). Greedy decode, capped tokens. Run under m1-gpu-lock.sh.
B=/Volumes/backups/code/crispembed-metal-audit-20261101/build-metal/crispembed
M=/Volumes/backups/code/model-cache-20261101/metal-audit
I=/Volumes/backups/code/crispembed-metal-audit-20261101/tests/regression/images
H=$(cd "$(dirname "$0")" && pwd); P=~/miniconda3/bin/python; O=$H/vlm; mkdir -p $O
LOCK=/Volumes/backups/code/model-cache-20261101/m1-gpu-lock.sh
ab() { local out=$1; shift; $LOCK $P $H/ab_cmd.py $O/$out --repeat 1 "$@" || echo "FAIL $out"; }
ab got-ocr2-q4k.fox.json -- $B -t 4 -m $M/got-ocr2-q4_k.gguf --ocr $I/fox.png --ocr-max-tokens 128
ab smoldocling-q8.fox.json -- $B -t 4 -m $M/smoldocling-q8_0.gguf --ocr $I/fox.png --ocr-max-tokens 128
ab h2ovl-800m-f16.fox.json -- $B -t 4 -m $M/h2ovl-800m-f16.gguf --ocr $I/fox.png --ocr-max-tokens 128
ab lightonocr-1b-q4k-gpu.fox.json --metal-env CRISPEMBED_LIGHTONOCR_GPU=1 -- $B -t 4 -m $M/lightonocr-1b-q4_k.gguf --ocr $I/fox.png --ocr-max-tokens 128
ab granite-vision-q4k.fox.json --cpu-env CRISPEMBED_GRANITE_CPU=1 -- $B -t 4 -m $M/granite-vision-3.3-2b-q4_k-visq8.tok.gguf --ocr $I/fox.png --ocr-max-tokens 128
# GOT-OCR2 needs the decoder licence acknowledgement for any run (evaluation only, nothing redistributed)
ab got-ocr2-q4k.fox.json --metal-env CRISPEMBED_ACCEPT_TONGYI_QIANWEN_LICENSE=1 --cpu-env CRISPEMBED_ACCEPT_TONGYI_QIANWEN_LICENSE=1 -- $B -t 4 -m $M/got-ocr2-q4_k.gguf --ocr $I/fox.png --ocr-max-tokens 128
