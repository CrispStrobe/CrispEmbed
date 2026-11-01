#!/usr/bin/env bash
# OXR-1.0 F32 CPU qualification against the pinned original Python model.
#
#   gh workflow run heavy-cpu.yml --ref <branch> -f script=tools/ci-heavy/oxr_qualify.sh \
#       -f isolated_lane=oxr [-f args="formula table"]
#
# The .github/workflows/oxr.yml model-smoke job for several cases: build once,
# convert the authenticated checkpoint once, then per case freeze the original
# Python result and all 66 stages (tools/oxr_validate.py reference), run the
# native stage diff (default and dense-attention control) and the same-host
# decoded output/speed comparison (default and reused decode graph).
# Each case writes $HEAVY_OUT/<id>/; results.txt lists every gate verbatim.
set -uo pipefail
TOP=${HEAVY_OUT:?}
S=${HEAVY_SCRATCH:?}
# Optional first argument dtype=f16|q8_0|q4_k qualifies that native GGUF (quantized
# variants via crispembed-quantize from the F32 conversion); the original
# Python reference stays F32 either way.
DTYPE=f32
if [[ "${1:-}" == dtype=* ]]; then DTYPE=${1#dtype=}; shift; fi
# Optional quantizer policy (tools/quantize.cpp, opt-in): large=q8_0|mix quantizes
# embed/lm_head/vte/visual head too; novisf16=1 keeps Q8_0 vision instead of F16.
QENV=()
while [[ "${1:-}" == large=* || "${1:-}" == novisf16=* ]]; do
    case $1 in large=*) QENV+=(CRISPEMBED_QUANTIZE_OXR_LARGE=${1#large=}) ;; novisf16=*) QENV+=(CRISPEMBED_QUANTIZE_NO_VISION_F16=${1#novisf16=}) ;; esac
    shift
done
case "$DTYPE" in f32 | f16 | q8_0 | q4_k) ;; *) echo "unsupported dtype $DTYPE"; exit 2 ;; esac
CASES=${*:-"formula table receipt german japanese"}
export CRISPEMBED_OXR=1 CRISPEMBED_ACCEPT_OXR_LICENSE=1 CRISPEMBED_OXR_FORCE_CPU=1 \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
declare -A IMAGE=([formula]=formula_quadratic.png [table]=cc0/simple_table.jpg [receipt]=cc0/receipt_example.png
    [german]=cc0/german_official_print.jpg [japanese]=japanese_print.png)
declare -A TASK=([formula]=formula [table]=table [receipt]=text [german]=text [japanese]=text)
declare -A PROMPT=([formula]="Recognize the formula of the image into KaTex format."
    [table]="Recognize the table of the image into html format." [text]="Recognize the content of the image.")

{ echo "source=$(git rev-parse HEAD)"; git submodule status; } > "$TOP/provenance.txt"
echo "::group::build"
cmake -S . -B "$S/build" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DCRISPEMBED_NATIVE=OFF >/dev/null &&
    cmake --build "$S/build" --target test-oxr-run test-oxr-diff test-oxr-profile test-oxr-graph test-oxr-stage-dump crispembed-quantize -j4 > "$S/build.log" 2>&1 ||
    { tail -60 "$S/build.log"; exit 4; }
"$S/build/test-oxr-profile" > "$TOP/profile.txt" 2>&1; echo "profile=$([ $? = 0 ] && echo pass || echo fail)" >> "$TOP/results.txt"
"$S/build/test-oxr-graph" > "$TOP/graph.txt" 2>&1; echo "graph=$([ $? = 0 ] && echo pass || echo fail)" >> "$TOP/results.txt"
echo "::endgroup::"

echo "::group::reference runtime and conversion"
python -m pip install -q torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu &&
    python -m pip install -q -r tools/oxr-reference-requirements.txt -r tools/oxr-conversion-requirements.txt || exit 3
python -m pip freeze > "$TOP/packages.txt"
python - "$S/checkpoint" <<'PY' || exit 5
import json, sys
from huggingface_hub import snapshot_download
m = json.load(open("models/oxr-source.json"))
snapshot_download(m["model_id"], revision=m["revision"], allow_patterns=list(m["files"]), local_dir=sys.argv[1])
PY
SRC_DTYPE=$DTYPE; [[ "$DTYPE" == q* ]] && SRC_DTYPE=f32
python models/convert-oxr-to-gguf.py --model "$S/checkpoint" --dtype "$SRC_DTYPE" --output "$S/oxr-$SRC_DTYPE.gguf" > "$S/convert.log" 2>&1 ||
    { tail -30 "$S/convert.log"; exit 5; }
cp "$S/oxr-$SRC_DTYPE.gguf.json" "$TOP/conversion.json" 2>/dev/null
if [[ "$DTYPE" != "$SRC_DTYPE" ]]; then
    QUANT=$(find "$S/build" -type f -name crispembed-quantize -perm -u+x | head -1)
    env ${QENV[@]+"${QENV[@]}"} "$QUANT" "$S/oxr-f32.gguf" "$S/oxr-$DTYPE.gguf" "$DTYPE" > "$TOP/quantize.log" 2>&1 || { tail -30 "$TOP/quantize.log"; exit 5; }
    ls -l "$S/oxr-$DTYPE.gguf" >> "$TOP/quantize.log"
fi
sha256sum "$S/oxr-$DTYPE.gguf" > "$TOP/gguf-sha256.txt"
echo "::endgroup::"

for id in $CASES; do
    O=$TOP/$id; mkdir -p "$O"
    note() { echo "$1=$2" >> "$O/results.txt"; }
    # Decode JPEG once with PIL so both sides see identical pixels (stb and PIL differ).
    python -c "import sys; from PIL import Image; Image.open(sys.argv[1]).convert('RGB').save(sys.argv[2])" \
        "tests/regression/images/${IMAGE[$id]}" "$S/$id.png"
    P=${PROMPT[${TASK[$id]}]}
    echo "::group::$id reference"
    if python tools/oxr_validate.py reference --model "$S/checkpoint" --image "$S/$id.png" --task "${TASK[$id]}" \
        --dtype f32 --device cpu --attention sdpa --threads 4 --repeat 3 --max-tokens 1024 \
        --output "$O/python.json" --dump-reference "$S/$id-python.gguf" > "$O/reference.log" 2>&1; then
        note reference pass
    else note reference fail; tail -30 "$O/reference.log"; echo "::endgroup::"; continue; fi
    echo "::endgroup::"
    for arm in default dense; do
        [ $arm = dense ] && export CRISPEMBED_OXR_FLASH_ATTN=0 || unset CRISPEMBED_OXR_FLASH_ATTN
        "$S/build/test-oxr-diff" --model "$S/oxr-$DTYPE.gguf" --reference "$S/$id-python.gguf" --image "$S/$id.png" \
            --prompt "$P" --threads 4 --max-abs 0.002 --min-cos 0.99999 \
            > "$O/stages-$arm.json" 2> "$O/stages-$arm.stderr.log"
        note "stages-$arm" "$([ $? = 0 ] && echo pass || echo "fail($?)")"
    done
    unset CRISPEMBED_OXR_FLASH_ATTN
    for arm in default reuse; do
        [ $arm = reuse ] && export CRISPEMBED_OXR_REUSE_DECODE_GRAPH=1 || unset CRISPEMBED_OXR_REUSE_DECODE_GRAPH
        python tools/oxr_validate.py compare --reference "$O/python.json" --native "$S/build/test-oxr-run" \
            --gguf "$S/oxr-$DTYPE.gguf" --image "$S/$id.png" --max-slowdown 1.0 --output "$O/native-$arm.json" \
            > "$O/compare-$arm.log" 2>&1
        note "decode-$arm" "$([ $? = 0 ] && echo pass || echo "fail($?)")"
    done
    unset CRISPEMBED_OXR_REUSE_DECODE_GRAPH
    if [[ "$DTYPE" != "$SRC_DTYPE" ]]; then
        # Owner acceptance for quantized recipes: prefill final hidden state and
        # logits cosine vs the native F32 conversion (diagnostic stage A/B).
        ST="vision.merge,visual.embeddings,llm.input,llm.layer.27,llm.norm,llm.logits"
        for m in f32 "$DTYPE"; do
            "$S/build/test-oxr-stage-dump" --dump-dir "$S/$id-dump-$m" --only "$ST" --model "$S/oxr-$m.gguf" \
                --image "$S/$id.png" --prompt "$P" --threads 4 --repeat 1 > /dev/null 2>> "$O/stage-ab.log"
        done
        python tools/oxr_stage_compare.py "$S/$id-dump-f32" "$S/$id-dump-$DTYPE" --output "$O/stage-ab.json" >> "$O/stage-ab.log" 2>&1
        note stage-ab "$([ $? = 0 ] && echo recorded || echo fail)"
        rm -rf "$S/$id-dump-f32" "$S/$id-dump-$DTYPE"
    fi
    rm -f "$S/$id-python.gguf"
    cat "$O/results.txt"
done

{ echo "### OXR-1.0 native $DTYPE CPU qualification ($(git rev-parse --short HEAD))"; echo '```'
  cat "$TOP/results.txt"; for id in $CASES; do echo "-- $id"; cat "$TOP/$id/results.txt" 2>/dev/null; done; echo '```'; } \
    > "$TOP/summary.md"
! grep -rqs '=fail' "$TOP"/results.txt "$TOP"/*/results.txt
