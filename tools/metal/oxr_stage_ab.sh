#!/usr/bin/env bash
# OXR-1.0 prefill stage dumps for precision/backend cosine A/B (local run).
#
#   tools/metal/oxr_stage_ab.sh BUILD_DIR MODEL.gguf PAGES_DIR OUT_DIR arm[,arm...]
#
# Dumps the final prefill hidden state (llm.norm), last-position logits
# (llm.logits) and the vision/visual-vocabulary boundaries per case into
# OUT_DIR/<case>/<arm>/ (tests/oxr_stage_dump.cpp). Compare two arms with
# tools/oxr_stage_compare.py. Arms: cpu metal metal-flash
set -uo pipefail
B=$1 MODEL=$2 PAGES=$3 OUT=$4; ARMS=${5:-"cpu metal"}
CASES=${CASES:-"formula table receipt german japanese"}
STAGES=${STAGES:-"vision.merge,visual.embeddings,llm.input,llm.layer.0,llm.layer.13,llm.layer.27,llm.norm,llm.logits"}
export CRISPEMBED_OXR=1 CRISPEMBED_ACCEPT_OXR_LICENSE=1
prompt() { case $1 in
    formula) echo "Recognize the formula of the image into KaTex format." ;;
    table) echo "Recognize the table of the image into html format." ;;
    *) echo "Recognize the content of the image." ;; esac; }
for id in $CASES; do
    for arm in ${ARMS//,/ }; do
        case $arm in
            cpu) e=(CRISPEMBED_OXR_FORCE_CPU=1) ;;
            metal) e=(CRISPEMBED_OXR_GPU=1) ;;
            metal-flash) e=(CRISPEMBED_OXR_GPU=1 CRISPEMBED_OXR_GPU_FLASH=1) ;;
            *) echo "unknown arm $arm"; exit 2 ;;
        esac
        D=$OUT/$id/$arm; mkdir -p "$D"
        env "${e[@]}" ${EXTRA_ENV:-} "$B/test-oxr-stage-dump" --dump-dir "$D" --only "$STAGES" --model "$MODEL" \
            --image "$PAGES/$id.png" --prompt "$(prompt $id)" --threads ${THREADS:-4} --repeat 1 --hf-generation 1 \
            > /dev/null 2> "$D.log"
        echo "$id $arm dump=$?"
    done
done
