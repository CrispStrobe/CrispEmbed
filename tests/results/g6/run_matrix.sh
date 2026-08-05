#!/bin/bash
# G6 (F6) F16-KV measurement matrix — serialized, one heavy process at a time.
# The ONLY variable vs the f1 F32 baselines (tests/results/f1/*-guard-persist-*)
# is DS2_KV_F16=1; the no-repeat-ngram guard is ON (default) in both.
# Timing medians are reported with the shared-box caveat; the interleaved A/B
# (interleave_kv.py) is the controlled timing instrument.
set -u
cd "$(dirname "$0")/../../.."
BIN=build/crispembed
MODEL=$HOME/.cache/crispembed-local/deepseek-ocr2-q4_k-stacked.gguf
SYNTH=$HOME/crispembed-ocr-synth
CC0=tests/regression/images/cc0
OUT=tests/results/g6
RUN="/Users/christianstrobele/miniconda3/bin/python tests/run_deepseek_ocr2_bench.py --binary $BIN --model $MODEL"

set -x
# Metal arms
$RUN --images $CC0 --labelled-only --out $OUT/m-kvf16-cc0 --gpu-backend metal --env DS2_FAST_DECODE=1 --env DS2_KV_F16=1
$RUN --images $SYNTH --out $OUT/m-kvf16-synth --gpu-backend metal --env DS2_FAST_DECODE=1 --env DS2_KV_F16=1
# CPU arms
$RUN --images $CC0 --labelled-only --out $OUT/c-kvf16-cc0 --gpu-backend cpu --env DS2_FAST_DECODE=1 --env DS2_KV_F16=1
$RUN --images $SYNTH --limit 5 --out $OUT/c-kvf16-synth --gpu-backend cpu --env DS2_FAST_DECODE=1 --env DS2_KV_F16=1
echo "G6 MATRIX DONE"
