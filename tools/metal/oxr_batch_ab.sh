#!/usr/bin/env bash
# OXR-1.0 five-case A/B with one model load per arm (tests/oxr_batch.cpp).
#
#   tools/metal/oxr_batch_ab.sh BUILD_DIR MODEL.gguf PAGES_DIR OUT_DIR arm[,arm...]
#
# Per case and arm: decoded IDs/text vs the original Python reference
# (tests/results/oxr/hosted-41277219829/<case>/python.json, original HF F32,
# 200704..921600 px, completion-only history, 1024-token cap), warm timings,
# and a prefill stage dump (STAGES) for cosine/abs-error A/B between arms.
# Arm = device[+gate...]: cpu, metal, metal+flash, metal+vflash, metal+f16half,
# metal+flashcpu, plus any extra KEY=VALUE env via EXTRA_ENV.
set -uo pipefail
B=$1 MODEL=$2 PAGES=$3 OUT=$4; ARMS=${5:-"cpu,metal"}
CASES=${CASES:-"formula table receipt german japanese"}
REFDIR=${REFDIR:-tests/results/oxr/hosted-41277219829}
STAGES=${STAGES-"vision.merge,visual.embeddings,llm.input,llm.layer.0,llm.layer.13,llm.layer.27,llm.norm,llm.logits"}
export CRISPEMBED_OXR=1 CRISPEMBED_ACCEPT_OXR_LICENSE=1
prompt() { case $1 in
    formula) echo "Recognize the formula of the image into KaTex format." ;;
    table) echo "Recognize the table of the image into html format." ;;
    *) echo "Recognize the content of the image." ;; esac; }
for arm in ${ARMS//,/ }; do
    e=()
    for part in ${arm//+/ }; do
        case $part in
            cpu) e+=(CRISPEMBED_OXR_FORCE_CPU=1) ;;
            metal) e+=(CRISPEMBED_OXR_GPU=1) ;;
            flash) e+=(CRISPEMBED_OXR_GPU_FLASH=1) ;;
            vflash) e+=(CRISPEMBED_OXR_GPU_VISION_FLASH=1) ;;
            flashcpu) e+=(CRISPEMBED_OXR_GPU_FLASH_CPU=1) ;;
            f16half) e+=(CRISPEMBED_OXR_GPU_F16_HALF_ACT=1) ;;
            vteq8) e+=(CRISPEMBED_OXR_VTE_Q8=1) ;;
            split) e+=(CRISPEMBED_OXR_GPU_SPLIT_F16=1) ;;
            hpmm) e+=(GGML_METAL_PREC_F32_MM=1) ;;
            nohpmm) e+=(CRISPEMBED_OXR_GPU_HP_MM=0) ;;
            auto) e+=(OXR_ARM_AUTO=1) ;;
            *) echo "unknown arm part $part"; exit 2 ;;
        esac
    done
    tsv=$OUT/$arm.tsv; mkdir -p "$OUT"; : > "$tsv"
    for id in $CASES; do mkdir -p "$OUT/$id"; printf '%s\t%s\t%s\n' "$OUT/$id/$arm" "$PAGES/$id.png" "$(prompt $id)" >> "$tsv"; done
    first=$(echo $CASES | cut -d' ' -f1)
    sargs=(); [ -n "$STAGES" ] && sargs=(--stages "$STAGES")
    env "${e[@]}" ${EXTRA_ENV:-} CRISPEMBED_OXR_TIMING=1 "$B/test-oxr-batch" --batch "$tsv" --model "$MODEL" \
        --image "$PAGES/$first.png" --threads ${THREADS:-4} --repeat ${REPEAT:-2} --max-tokens 1024 --hf-generation 1 \
        ${sargs[@]+"${sargs[@]}"} 2> "$OUT/$arm.log"
    echo "$arm batch=$?"
    for id in $CASES; do
        python3 tools/oxr_decode_compare.py --reference "$REFDIR/$id/python.json" --native "$OUT/$id/$arm.json" \
            --label "$id/$arm" > "$OUT/$id/$arm.cmp.json" 2>/dev/null
        echo "  $(cat "$OUT/$id/$arm.cmp.json" | cut -c1-260)"
    done
done
