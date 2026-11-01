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
# Optional first argument dtype=f16 qualifies the native F16 GGUF; the original
# Python reference stays F32 either way.
DTYPE=f32
if [[ "${1:-}" == dtype=* ]]; then DTYPE=${1#dtype=}; shift; fi
[[ "$DTYPE" == f32 || "$DTYPE" == f16 ]] || { echo "unsupported dtype $DTYPE"; exit 2; }
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
    cmake --build "$S/build" --target test-oxr-run test-oxr-diff test-oxr-profile test-oxr-graph -j4 > "$S/build.log" 2>&1 ||
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
python models/convert-oxr-to-gguf.py --model "$S/checkpoint" --dtype "$DTYPE" --output "$S/oxr-$DTYPE.gguf" > "$S/convert.log" 2>&1 ||
    { tail -30 "$S/convert.log"; exit 5; }
cp "$S/oxr-$DTYPE.gguf.json" "$TOP/conversion.json" 2>/dev/null
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
    rm -f "$S/$id-python.gguf"
    cat "$O/results.txt"
done

{ echo "### OXR-1.0 native $DTYPE CPU qualification ($(git rev-parse --short HEAD))"; echo '```'
  cat "$TOP/results.txt"; for id in $CASES; do echo "-- $id"; cat "$TOP/$id/results.txt" 2>/dev/null; done; echo '```'; } \
    > "$TOP/summary.md"
! grep -rqs '=fail' "$TOP"/results.txt "$TOP"/*/results.txt
