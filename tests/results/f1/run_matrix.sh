#!/bin/bash
# F1 acceptance matrix — serialized (one heavy process at a time).
# Correctness/termination gates only; no timing claims are made from these runs
# (the box carries concurrent agent load).
set -u
cd "$(dirname "$0")/../../.."
BIN=build/crispembed
MODEL=$HOME/.cache/crispembed-local/deepseek-ocr2-q4_k-stacked.gguf
SYNTH=$HOME/crispembed-ocr-synth
CC0=tests/regression/images/cc0
OUT=tests/results/f1
RUN="python3 tests/run_deepseek_ocr2_bench.py --binary $BIN --model $MODEL"

set -x
# Metal arms
$RUN --images $SYNTH --out $OUT/m-guard-persist-synth --gpu-backend metal --env DS2_FAST_DECODE=1
$RUN --images $CC0 --labelled-only --out $OUT/m-guard-persist-cc0 --gpu-backend metal --env DS2_FAST_DECODE=1
$RUN --images $SYNTH --out $OUT/m-guard-legacy-synth --gpu-backend metal --env DS2_LEGACY_DECODE=1
$RUN --images $CC0 --labelled-only --out $OUT/m-guard-legacy-cc0 --gpu-backend metal --env DS2_LEGACY_DECODE=1
$RUN --images $SYNTH --out $OUT/m-base-persist-synth --gpu-backend metal --env DS2_FAST_DECODE=1 --env DS2_NO_REPEAT_NGRAM=0
$RUN --images $CC0 --labelled-only --out $OUT/m-base-persist-cc0 --gpu-backend metal --env DS2_FAST_DECODE=1 --env DS2_NO_REPEAT_NGRAM=0
# CPU arms
$RUN --images $CC0 --labelled-only --out $OUT/c-guard-persist-cc0 --gpu-backend cpu --env DS2_FAST_DECODE=1
$RUN --images $CC0 --labelled-only --out $OUT/c-guard-legacy-cc0 --gpu-backend cpu --env DS2_LEGACY_DECODE=1
$RUN --images $SYNTH --limit 5 --out $OUT/c-guard-persist-synth --gpu-backend cpu --env DS2_FAST_DECODE=1
$RUN --images $SYNTH --limit 5 --out $OUT/c-guard-legacy-synth --gpu-backend cpu --env DS2_LEGACY_DECODE=1
$RUN --images $SYNTH --limit 5 --out $OUT/c-base-persist-synth --gpu-backend cpu --env DS2_FAST_DECODE=1 --env DS2_NO_REPEAT_NGRAM=0
echo "F1 MATRIX DONE"
