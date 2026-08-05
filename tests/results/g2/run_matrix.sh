#!/bin/bash
# G2 (F5) acceptance matrix — serialized, one heavy process at a time.
# Correctness/termination/CER gates only; no timing claims (box carries load).
# Baselines for crop-OFF byte-identity: tests/results/f1/{m,c}-guard-persist-cc0.
set -u
cd "$(dirname "$0")/../../.."
BIN=build/crispembed
MODEL=$HOME/.cache/crispembed-local/deepseek-ocr2-q4_k-stacked.gguf
SYNTH=$HOME/crispembed-ocr-synth
CC0=tests/regression/images/cc0
OUT=tests/results/g2
RUN="/Users/christianstrobele/miniconda3/bin/python tests/run_deepseek_ocr2_bench.py --binary $BIN --model $MODEL"

set -x
# Metal: crop-on headline, then crop-off identity arm (vs f1 baseline files)
$RUN --images $CC0 --labelled-only --out $OUT/m-crop-cc0 --gpu-backend metal --env DS2_FAST_DECODE=1 --env DS2_CROP_MODE=1
$RUN --images $CC0 --labelled-only --out $OUT/m-base-cc0 --gpu-backend metal --env DS2_FAST_DECODE=1
# CPU arms
$RUN --images $CC0 --labelled-only --out $OUT/c-crop-cc0 --gpu-backend cpu --env DS2_FAST_DECODE=1 --env DS2_CROP_MODE=1
$RUN --images $CC0 --labelled-only --out $OUT/c-base-cc0 --gpu-backend cpu --env DS2_FAST_DECODE=1
# Synth no-regression (all <=768 -> crop-on must be a byte-level no-op)
$RUN --images $SYNTH --limit 10 --out $OUT/m-crop-synth --gpu-backend metal --env DS2_FAST_DECODE=1 --env DS2_CROP_MODE=1
$RUN --images $SYNTH --limit 10 --out $OUT/m-base-synth --gpu-backend metal --env DS2_FAST_DECODE=1
echo "G2 MATRIX DONE"
