#!/usr/bin/env bash
# OXR D-FINE layout: CPU vs Metal A/B on the five reference pages (local Apple-silicon run).
#
#   tools/metal/oxr_dfine_metal_ab.sh BUILD_DIR MODEL.gguf REF_DIR OUT_DIR [arm...]
#
# REF_DIR holds <page>.png and <page>-generic.gguf from tools/oxr_dfine_reference.py
# (--resize generic: this host's OpenCV has no IPP, and native preprocessing equals
# the generic resize bit for bit). Each arm writes OUT_DIR/<page>/diff-<arm>.json via
# test-oxr-dfine-diff, whose budgets are fixed in tests/oxr_dfine_diff.cpp.
# Arms: cpu metal metal-halfmm metal-conv-f16 metal-dw-direct metal-conv-direct metal-flash metal-split metal-split-dw
set -uo pipefail
B=$1 MODEL=$2 REF=$3 OUT=$4; shift 4
ARMS=${*:-"cpu metal"}
PAGES=${PAGES:-"receipt german scan table japanese"}
export CRISPEMBED_OXR=1 CRISPEMBED_ACCEPT_OXR_LICENSE=1
for page in $PAGES; do
    mkdir -p "$OUT/$page"
    for arm in $ARMS; do
        env_args=()
        case $arm in
            cpu) env_args=(LAYOUT_DETECT_FORCE_CPU=1) ;;
            metal) env_args=(CRISPEMBED_OXR_LAYOUT_GPU=1) ;;
            metal-halfmm) env_args=(CRISPEMBED_OXR_LAYOUT_GPU=1 CRISPEMBED_OXR_LAYOUT_GPU_PREC_F32=0) ;;
            metal-conv-f16) env_args=(CRISPEMBED_OXR_LAYOUT_GPU=1 LAYOUT_CONV_F16=1) ;;
            metal-dw-direct) env_args=(CRISPEMBED_OXR_LAYOUT_GPU=1 CRISPEMBED_OXR_LAYOUT_DW_DIRECT=1) ;;
            metal-conv-direct) env_args=(CRISPEMBED_OXR_LAYOUT_GPU=1 LAYOUT_CONV_DIRECT=1) ;;
            metal-flash) env_args=(CRISPEMBED_OXR_LAYOUT_GPU=1 LAYOUT_DETECT_FLASH=1) ;;
            metal-split) env_args=(CRISPEMBED_OXR_LAYOUT_GPU=1 CRISPEMBED_OXR_LAYOUT_GPU_SPLIT_F16=1) ;;
            metal-split-dw) env_args=(CRISPEMBED_OXR_LAYOUT_GPU=1 CRISPEMBED_OXR_LAYOUT_GPU_SPLIT_F16=1 CRISPEMBED_OXR_LAYOUT_DW_DIRECT=1) ;;
            *) echo "unknown arm $arm"; exit 2 ;;
        esac
        env "${env_args[@]}" "$B/test-oxr-dfine-diff" --model "$MODEL" --reference "$REF/$page-generic.gguf" \
            --image "$REF/pages/$page.png" --threads ${THREADS:-4} --repeat ${REPEAT:-5} \
            > "$OUT/$page/diff-$arm.json" 2> "$OUT/$page/diff-$arm.log"
        echo "$page $arm exit=$?"
    done
done
