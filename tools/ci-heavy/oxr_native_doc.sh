#!/usr/bin/env bash
# Native OXR document route (CRISPEMBED_OXR_NATIVE_DOC) parity against the
# pinned original Python document helper.
#
#   gh workflow run heavy-cpu.yml -R CrispStrobe/CrispEmbed --ref feat/oxr-native-doc \
#       -f script=tools/ci-heavy/oxr_native_doc.sh -f isolated_lane=oxr-native-doc \
#       [-f args="helpers replay dfine native"]
#
# Stages (each optional through args, default all):
#   helpers  randomized helper differential tests (model-free)
#   replay   full process_pages with a scripted recognizer (model-free)
#   dfine    original D-FINE (SoMarkAI/OXR services/layout_analyze, CPU F32) on
#            the five qualification images, frozen as layout fixtures
#   native   real F32 OXR model: original helper + native recognition, recorded
#            once; C++ route replay of those exact results; live C++ route
# Results land in $HEAVY_OUT; results.txt lists every gate verbatim.
set -uo pipefail
TOP=${HEAVY_OUT:?}
S=${HEAVY_SCRATCH:?}
STAGES=${*:-"helpers replay bindings dfine native"}
BLUEPRINT=c1a1dcb312ee9f691404f127785510453c0d9350
export CRISPEMBED_OXR=1 CRISPEMBED_ACCEPT_OXR_LICENSE=1 CRISPEMBED_OXR_FORCE_CPU=1 \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH=$PWD/python
declare -A IMAGE=([formula]=formula_quadratic.png [table]=cc0/simple_table.jpg [receipt]=cc0/receipt_example.png
    [german]=cc0/german_official_print.jpg [japanese]=japanese_print.png)
has() { [[ " $STAGES " == *" $1 "* ]]; }
note() { echo "$1=$2" | tee -a "$TOP/results.txt"; }

{ echo "source=$(git rev-parse HEAD)"; git submodule status; python --version; } > "$TOP/provenance.txt"

echo "::group::build"
cmake -S . -B "$S/build" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DCRISPEMBED_NATIVE=OFF >/dev/null &&
    cmake --build "$S/build" --target test-oxr-document crispembed-shared crispembed-cli crispembed-server -j4 > "$S/build.log" 2>&1 ||
    { tail -80 "$S/build.log"; exit 4; }
"$S/build/test-oxr-document" > "$TOP/self-checks.txt" 2>&1
note self_checks "$([ $? = 0 ] && echo pass || echo fail)"
LIB=$(ls "$S"/build/libcrispembed*.so 2>/dev/null | head -1)
echo "::endgroup::"

echo "::group::pinned helper environment"
python -m pip install -q -r tools/oxr-blueprint-requirements.txt || exit 3
git clone -q https://github.com/SoMarkAI/OXR.git "$S/oxr-src" && git -C "$S/oxr-src" checkout -q $BLUEPRINT || exit 3
python -m pip freeze > "$TOP/packages-helper.txt"
echo "::endgroup::"

if has helpers; then
    echo "::group::helper differential"
    python tools/oxr_doc_parity.py helpers --binary "$S/build/test-oxr-document" --cases 400 --seed 20261031 \
        --output "$TOP/helpers.json" > "$TOP/helpers.log" 2>&1
    note helpers "$([ $? = 0 ] && echo pass || echo fail)"
    echo "::endgroup::"
fi
if has replay; then
    echo "::group::scripted full-document replay"
    python tools/oxr_doc_parity.py replay --binary "$S/build/test-oxr-document" --cases 80 --seed 20261031 \
        --output "$TOP/replay.json" > "$TOP/replay.log" 2>&1
    note replay "$([ $? = 0 ] && echo pass || echo fail)"
    echo "::endgroup::"
fi

if has bindings && command -v cargo >/dev/null; then
    echo "::group::Rust wrapper compile check"
    (cd crispembed && CARGO_TARGET_DIR="$S/cargo-target" timeout 3000 cargo check -q) > "$TOP/cargo-check.log" 2>&1
    note rust_check "$([ $? = 0 ] && echo pass || echo fail)"
    tail -20 "$TOP/cargo-check.log"; rm -rf "$S/cargo-target"
    echo "::endgroup::"
fi

CASES=()
for id in formula table receipt german japanese; do CASES+=("$id=tests/regression/images/${IMAGE[$id]}"); done
if has dfine || has native; then
    echo "::group::original D-FINE layout fixtures"
    python -m pip install -q torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu || exit 3
    python -m pip freeze > "$TOP/packages-dfine.txt"
    mkdir -p "$TOP/layouts"
    python tools/oxr_dfine_fixtures.py --oxr-source "$S/oxr-src" --out "$TOP/layouts" "${CASES[@]}" \
        > "$TOP/dfine.log" 2>&1
    note dfine "$([ $? = 0 ] && echo pass || echo fail)"
    cat "$TOP/dfine.log"
    echo "::endgroup::"
fi

if has native; then
    echo "::group::checkpoint conversion"
    python -m pip install -q -r tools/oxr-conversion-requirements.txt || exit 3
    python - "$S/checkpoint" <<'PY' || exit 5
import json, sys
from huggingface_hub import snapshot_download
m = json.load(open("models/oxr-source.json"))
snapshot_download(m["model_id"], revision=m["revision"], allow_patterns=list(m["files"]), local_dir=sys.argv[1])
PY
    python models/convert-oxr-to-gguf.py --model "$S/checkpoint" --dtype f32 --output "$S/oxr-f32.gguf" \
        > "$S/convert.log" 2>&1 || { tail -30 "$S/convert.log"; exit 5; }
    cp "$S/oxr-f32.gguf.json" "$TOP/conversion.json" 2>/dev/null
    rm -rf "$S/checkpoint"
    echo "::endgroup::"
    echo "::group::native model parity"
    ARGS=()
    for id in formula table receipt german japanese; do
        [ -f "$TOP/layouts/$id.json" ] && ARGS+=(--case "$id=tests/regression/images/${IMAGE[$id]}:$TOP/layouts/$id.json")
    done
    python tools/oxr_doc_parity.py native --binary "$S/build/test-oxr-document" --lib "$LIB" \
        --gguf "$S/oxr-f32.gguf" --workdir "$TOP/native" --threads 4 "${ARGS[@]}" \
        --output "$TOP/native.json" > "$TOP/native.log" 2>&1
    note native "$([ $? = 0 ] && echo pass || echo fail)"
    tail -20 "$TOP/native.log"
    rm -rf "$TOP"/native/*.rgb
    echo "::endgroup::"
fi

{ echo "### OXR native document route ($(git rev-parse --short HEAD))"; echo '```'; cat "$TOP/results.txt"; echo '```'; } \
    > "$TOP/summary.md"
! grep -qs '=fail' "$TOP/results.txt"
