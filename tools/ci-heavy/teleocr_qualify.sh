#!/usr/bin/env bash
# TeleOCR independent hosted re-run of the delivered validation protocol.
#
#   gh workflow run heavy-cpu.yml --ref <branch> -f script=tools/ci-heavy/teleocr_qualify.sh \
#       -f args="[VARIANTS]" -f isolated_lane=teleocr-<variants>
#
# VARIANTS defaults to f32,f16,q8_0,q4_k. Follows docs/teleocr.md "Tests and
# interpretation" exactly: pinned CPU reference (two threads, 1,024-token cap,
# six frozen cases), the delivered conversion/Q4_K recipes, and
# tools/run_teleocr_validation.py per precision with unchanged budgets. Also
# runs the library-backed loader/quantizer controls, the public-interface
# consistency harness and the native parser smoke on F16 when it is selected.
# Every step runs after a failure; exit is nonzero when any step failed.
set -uo pipefail

VARIANTS=${1:-f32,f16,q8_0,q4_k}
MODEL=XingChen-AGI/TeleOCR
REV=b3f6a0c084db72f971e19a779d2fd8a6bc9e3b8d
OUT=${HEAVY_OUT:?}
S=${HEAVY_SCRATCH:?}
export CRISPEMBED_TELEOCR=1 QWEN2VL_OCR_FORCE_CPU=1

failed=0
step() { # name, command...
    local name=$1; shift
    echo "::group::$name"
    if "$@"; then echo "$name=pass" >> "$OUT/results.txt"; else
        echo "$name=fail($?)" >> "$OUT/results.txt"; failed=1; fi
    echo "::endgroup::"
}

{
    echo "source=$(git rev-parse HEAD)"; git submodule status
    echo "variants=$VARIANTS"; lscpu 2>/dev/null | grep -E 'Model name|^CPU\(s\)'
} > "$OUT/provenance.txt"

echo "::group::reference environment"
python -m pip install -q torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu &&
    python -m pip install -q -r models/requirements-teleocr-reference.txt || exit 3
python -m pip freeze > "$OUT/packages.txt"
echo "::endgroup::"

echo "::group::build"
cmake -S . -B "$S/build" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DCRISPEMBED_NATIVE=OFF -DGGML_CUDA=OFF \
    -DCRISPEMBED_BUILD_SHARED=ON >/dev/null &&
    cmake --build "$S/build" -j4 --target test-teleocr-diff crispembed-quantize crispembed-shared crispembed-cli crispembed-server \
        > "$S/build.log" 2>&1 || { tail -60 "$S/build.log"; exit 4; }
B=$S/build
LIB=$(find "$B" -name 'libcrispembed.so' | head -1)
QUANT=$(find "$B" -type f -name crispembed-quantize -perm -u+x | head -1)
DIFF=$(find "$B" -type f -name test-teleocr-diff -perm -u+x | head -1)
echo "::endgroup::"

step loader python tests/test_teleocr_loader.py --lib "$LIB" --report "$OUT/teleocr-loader.json"
step quantizer-provenance python tests/test_teleocr_quantizer.py --quantizer "$QUANT" \
    --report "$OUT/quantizer-provenance-tests.json"

echo "::group::checkpoint"
python - "$MODEL" "$REV" "$S/TeleOCR" <<'PY' || exit 5
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3])
PY
echo "::endgroup::"

IFS=, read -ra VARS <<< "$VARIANTS"
mkdir -p "$S/gguf" "$OUT/manifests"
need_f16=0
for v in "${VARS[@]}"; do [ "$v" = q4_k ] && need_f16=1; [ "$v" = f16 ] && need_f16=1; done
convert() { python models/convert-teleocr-to-gguf.py --model "$S/TeleOCR" --output "$S/gguf/teleocr-$1.gguf" \
    --dtype "$1" --manifest "$OUT/manifests/teleocr-$1.gguf.json"; }
for v in "${VARS[@]}"; do
    case "$v" in
        f32 | q8_0) step "convert-$v" convert "$v" ;;
    esac
done
[ $need_f16 = 1 ] && step convert-f16 convert f16
for v in "${VARS[@]}"; do
    [ "$v" = q4_k ] && step quantize-q4_k "$QUANT" "$S/gguf/teleocr-f16.gguf" "$S/gguf/teleocr-q4_k.gguf" q4_k \
        --tensor-type 'v.*=f16' --tensor-type 'l.embed_tokens.weight=f16'
done
for f in "$S"/gguf/*.gguf; do printf '%s %s %s\n' "$(sha256sum "$f" | cut -d' ' -f1)" "$(stat -c %s "$f")" "$(basename "$f")"; done |
    tee "$OUT/gguf-sha256.txt"

step reference python tools/dump_teleocr_reference.py --model-dir "$S/TeleOCR" --out "$S/reference" \
    --threads 2 --max-new-tokens 1024
mkdir -p "$OUT/reference" && find "$S/reference" -maxdepth 1 -name '*.json' -exec cp {} "$OUT/reference/" \;

for v in "${VARS[@]}"; do
    [ -f "$S/gguf/teleocr-$v.gguf" ] || continue
    step "differential-$v" python tools/run_teleocr_validation.py --binary "$DIFF" --model "$S/gguf/teleocr-$v.gguf" \
        --reference-dir "$S/reference" --out "$OUT/native-$v" --threads 2
done

if [[ ",$VARIANTS," == *,f16,* ]] && [ -f "$S/gguf/teleocr-f16.gguf" ]; then
    step public-f16 python tests/test_teleocr_public.py --lib "$LIB" --cli "$(find "$B" -type f -name crispembed -perm -u+x | head -1)" \
        --server "$(find "$B" -type f -name crispembed-server -perm -u+x | head -1)" --model "$S/gguf/teleocr-f16.gguf" \
        --image tests/fixtures/teleocr/synthetic-text.png --max-tokens 8 --threads 2 --report "$OUT/teleocr-public.json"
    step parser-f16 python tests/test_teleocr_parser_native.py --lib "$LIB" --model "$S/gguf/teleocr-f16.gguf" \
        --image tests/fixtures/teleocr/synthetic-text.png --mode detection --out "$OUT/parser-f16" \
        --max-tokens 1024 --threads 2
fi

{
    echo "### TeleOCR hosted re-run ($(git rev-parse --short HEAD), $VARIANTS)"
    echo; echo '```'; cat "$OUT/results.txt"; echo '```'
} > "$OUT/summary.md"
exit $failed
