#!/usr/bin/env bash
# MinerU2.5-Pro full native document route: parity against the pinned Python
# helper (mineru-vl-utils c9797cde) with the SAME native GGML recognition.
#
#   gh workflow run heavy-cpu.yml -R CrispStrobe/CrispEmbed --ref feat/mineru-native-doc \
#       -f script=tools/ci-heavy/mineru25_doc_parity.sh -f isolated_lane=mineru-native-doc \
#       [-f args="CASE_IDS MAX_TOKENS SYNTHETIC"]
#
# 1. Weight-free: test-mineru-doc self, test-otsl-table, regex and function
#    differential fuzz, synthetic page parity (every helper branch).
# 2. Model-backed, published F16 GGUF, each protocol case (docs/mineru25-pro-
#    publication-v1.json case_inputs): Python CrispMinerU.two_step_extract
#    (pinned helper + native inference) vs CrispMinerU.native_two_step_extract
#    (pure C++), both with table_token seed S, compared byte-for-byte as JSON
#    and Markdown; per-region native requests/outputs are compared too. The
#    recorded Python recognition is then replayed through test-mineru-doc.
# Results: $HEAVY_OUT/{weight-free,cases}/..., summary.json, summary.md.
set -uo pipefail
CASE_IDS=${1:-all}
MAX_TOKENS=${2:-2048}
SYNTHETIC=${3:-300}
THREADS=4
GGUF_URL=https://huggingface.co/cstr/MinerU2.5-Pro-2604-1.2B-GGUF/resolve/e63c95cfd4e9f753783f588a187b9a3b89a7c201/mineru25-pro-2604-f16.gguf
GGUF_SHA=7d464d375f7c20dd3e2bae7e20fed1f80485ae9ca69aaad201897652c74d746b
OUT=${HEAVY_OUT:?}
S=${HEAVY_SCRATCH:?}
mkdir -p "$OUT/weight-free" "$OUT/cases"
status=0
note() { echo "$1: $2" | tee -a "$OUT/results.txt"; }
gate() { local n=$1; shift; if "$@"; then note "$n" pass; else note "$n" "fail($?)"; status=1; fi; }

{ echo "source=$(git rev-parse HEAD)"; python --version; lscpu 2>/dev/null | grep 'Model name'; } > "$OUT/provenance.txt"

echo "::group::python environment"
python -m pip install -q -r tools/mineru25-blueprint-requirements.txt numpy || exit 3
python -m pip freeze > "$OUT/packages.txt"
echo "::endgroup::"

echo "::group::unicode tables for this CPython"
python -I tools/gen/gen_pyunicode_tables.py "$S/pyunicode_tables.h" &&
    if cmp -s <(tail -n +2 "$S/pyunicode_tables.h") <(tail -n +2 src/pyunicode_tables.h); then
        note unicode_tables "identical to committed (CPython $(python -c 'import sys;print(sys.version.split()[0])'))"
    else
        cp "$S/pyunicode_tables.h" src/pyunicode_tables.h
        note unicode_tables "regenerated for reference CPython $(python -c 'import sys,unicodedata;print(sys.version.split()[0], unicodedata.unidata_version)')"
    fi
echo "::endgroup::"

echo "::group::build"
cmake -S . -B "$S/build" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DCRISPEMBED_NATIVE=OFF -DGGML_CUDA=OFF >/dev/null &&
    cmake --build "$S/build" -j4 --target crispembed-shared crispembed-cli test-mineru-doc test-otsl-table \
        test-mineru-layout test-marker-document > "$S/build.log" 2>&1 || { tail -80 "$S/build.log"; exit 4; }
B=$S/build
LIB=$(find "$B" -name 'libcrispembed.so' | head -1)
DOC=$(find "$B" -type f -name test-mineru-doc -perm -u+x | head -1)
OTSL=$(find "$B" -type f -name test-otsl-table -perm -u+x | head -1)
echo "::endgroup::"

echo "::group::weight-free"
gate self "$DOC" self
gate otsl_golden "$OTSL" tests/data/otsl_profiles_golden.tsv
gate mineru_layout_unit "$(find "$B" -type f -name test-mineru-layout -perm -u+x | head -1)"
gate marker_document_unit "$(find "$B" -type f -name test-marker-document -perm -u+x | head -1)"
gate fuzz_regex python -I tools/mineru_doc/fuzz_regex.py "$DOC" --cases 400 > "$OUT/weight-free/fuzz_regex.txt" 2>&1
gate fuzz_postprocess python -I tools/mineru_doc/fuzz_postprocess.py "$DOC" --cases 1500 --seed 2 \
    --report "$OUT/weight-free/fuzz_postprocess.json" > "$OUT/weight-free/fuzz_postprocess.txt" 2>&1
gate synthetic_pages python -I tools/mineru_doc/page_parity.py "$DOC" "$S/synthetic" --synthetic "$SYNTHETIC" --seed 21 \
    --report "$OUT/weight-free/synthetic_pages.json" > "$OUT/weight-free/synthetic_pages.txt" 2>&1
for f in "$OUT"/weight-free/*.txt; do tail -n 3 "$f"; done
echo "::endgroup::"

echo "::group::model"
curl -fsSL -o "$S/mineru25-pro-f16.gguf" "$GGUF_URL" || exit 5
echo "$GGUF_SHA  $S/mineru25-pro-f16.gguf" | sha256sum -c - || exit 5
echo "::endgroup::"

echo "::group::model-backed protocol cases"
export CRISPEMBED_MINERU25_PRO=1 CRISPEMBED_MINERU25_PRO_FULL_DOC=1 QWEN2VL_OCR_FORCE_CPU=1
export OMP_NUM_THREADS=$THREADS PYTHONPATH=python
gate protocol_cases python tools/mineru_doc/model_parity.py --lib "$LIB" --gguf "$S/mineru25-pro-f16.gguf" \
    --cases "$CASE_IDS" --max-tokens "$MAX_TOKENS" --threads "$THREADS" --out "$OUT/cases" \
    --record "$OUT/cases/recorded.json"
gate recorded_replay python -I tools/mineru_doc/page_parity.py "$DOC" "$S/replay" --recorded "$OUT/cases/recorded.json" \
    --report "$OUT/cases/replay.json" > "$OUT/cases/replay.txt" 2>&1
tail -15 "$OUT/cases/replay.txt"
echo "::endgroup::"

{
    echo "## MinerU native document parity ($(git rev-parse --short HEAD))"
    echo '```'; cat "$OUT/results.txt"; echo '```'
    [ -f "$OUT/cases/summary.txt" ] && { echo '```'; cat "$OUT/cases/summary.txt"; echo '```'; }
} > "$OUT/summary.md"
exit $status
