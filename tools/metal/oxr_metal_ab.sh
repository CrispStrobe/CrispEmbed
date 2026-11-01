#!/usr/bin/env bash
# OXR-1.0 recognition: CPU vs Metal decoded-output A/B on the five reference cases
# (local Apple-silicon run, original-HF generation profile of the hosted lane).
#
#   tools/metal/oxr_metal_ab.sh BUILD_DIR MODEL.gguf PAGES_DIR OUT_DIR arm[,arm...]
#
# Reference token IDs/text: tests/results/oxr/hosted-41277219829/<case>/python.json
# (original Transformers F32, 200704..921600 pixels, completion-only history).
# Arms: cpu metal metal-flash metal-flash-cpu metal-reuse cpu-reuse
# Writes OUT_DIR/<case>/<arm>.json (native) and <arm>.cmp.json (comparison).
set -uo pipefail
B=$1 MODEL=$2 PAGES=$3 OUT=$4; ARMS=${5:-"cpu metal"}
CASES=${CASES:-"formula table receipt german japanese"}
REFDIR=${REFDIR:-tests/results/oxr/hosted-41277219829}
export CRISPEMBED_OXR=1 CRISPEMBED_ACCEPT_OXR_LICENSE=1
prompt() { case $1 in
    formula) echo "Recognize the formula of the image into KaTex format." ;;
    table) echo "Recognize the table of the image into html format." ;;
    *) echo "Recognize the content of the image." ;; esac; }
for id in $CASES; do
    mkdir -p "$OUT/$id"
    for arm in ${ARMS//,/ }; do
        case $arm in
            cpu) e=(CRISPEMBED_OXR_FORCE_CPU=1) ;;
            cpu-reuse) e=(CRISPEMBED_OXR_FORCE_CPU=1 CRISPEMBED_OXR_REUSE_DECODE_GRAPH=1) ;;
            metal) e=(CRISPEMBED_OXR_GPU=1) ;;
            metal-reuse) e=(CRISPEMBED_OXR_GPU=1 CRISPEMBED_OXR_REUSE_DECODE_GRAPH=1) ;;
            metal-flash) e=(CRISPEMBED_OXR_GPU=1 CRISPEMBED_OXR_GPU_FLASH=1) ;;
            metal-flash-cpu) e=(CRISPEMBED_OXR_GPU=1 CRISPEMBED_OXR_GPU_FLASH_CPU=1) ;;
            *) echo "unknown arm $arm"; exit 2 ;;
        esac
        env "${e[@]}" ${EXTRA_ENV:-} "$B/test-oxr-run" --model "$MODEL" --image "$PAGES/$id.png" --prompt "$(prompt $id)" \
            --threads ${THREADS:-4} --repeat ${REPEAT:-3} --max-tokens 1024 --hf-generation 1 \
            > "$OUT/$id/$arm.json" 2> "$OUT/$id/$arm.log"
        rc=$?
        python3 tools/oxr_decode_compare.py --reference "$REFDIR/$id/python.json" --native "$OUT/$id/$arm.json" \
            --label "$id/$arm" > "$OUT/$id/$arm.cmp.json" 2>> "$OUT/$id/$arm.log"
        echo "$id $arm run=$rc $(cat "$OUT/$id/$arm.cmp.json")"
    done
done
