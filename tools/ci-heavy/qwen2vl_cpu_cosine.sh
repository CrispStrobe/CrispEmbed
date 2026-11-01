#!/usr/bin/env bash
# CPU arm of the qwen2vl_ocr device A/B (docs/qwen2vl-metal.md): decoded IDs
# against the committed original-Python references plus final-state cosine
# (prefill final-norm rows, prefill logits, first 32 decode steps) against a
# native CPU F32 reference, for MinerU2.5-Pro-2604 precisions.
#
#   gh workflow run heavy-cpu.yml --ref feat/metal-qwen2vl \
#       -f script=tools/ci-heavy/qwen2vl_cpu_cosine.sh -f args="f16,q8_0,q4_k-vf16,q4_k" \
#       -f pip="numpy==2.2.6 safetensors==0.6.2 huggingface_hub==0.36.0 gguf==0.17.1" \
#       -f isolated_lane=qwen2vl-cpu-cosine-mineru
#
# The F32 reference is converted from the pinned checkpoint; the other variants
# are the published F16/Q8_0 GGUFs or quantized from that F32 here. Writes
# receipts/cosine JSON to $HEAVY_OUT and a summary.md table.
set -uo pipefail
VARIANTS=${1:-f16,q8_0,q4_k-vf16,q4_k}
CASES=${2:-}
ROOT=$(pwd)
OUT=${HEAVY_OUT:?}
S=${HEAVY_SCRATCH:?}
PUB=https://huggingface.co/cstr/MinerU2.5-Pro-2604-1.2B-GGUF/resolve/e63c95cfd4e9f753783f588a187b9a3b89a7c201
cmake -S . -B "$S/build" -DCMAKE_BUILD_TYPE=Release > "$OUT/cmake.log" 2>&1 &&
    cmake --build "$S/build" --target test-mineru25-run crispembed-quantize -j"$(nproc)" > "$OUT/build.log" 2>&1 ||
    { tail -40 "$OUT/build.log"; exit 2; }
B=$S/build
[ -x "$B/test-mineru25-run" ] || B=$(dirname "$(find "$S/build" -name test-mineru25-run -type f | head -1)")
python models/convert-mineru25-pro-to-gguf.py --output "$S/f32.gguf" --dtype f32 > "$OUT/convert-f32.log" 2>&1 ||
    { tail -20 "$OUT/convert-f32.log"; exit 2; }
gguf_for() {
    case $1 in
    f32) echo "$S/f32.gguf" ;;
    f16 | q8_0)
        [ -f "$S/$1.gguf" ] || curl -sSL --retry 5 -o "$S/$1.gguf" "$PUB/mineru25-pro-2604-$1.gguf"
        echo "$S/$1.gguf" ;;
    q4_k-vf16)
        [ -f "$S/$1.gguf" ] || "$B/crispembed-quantize" "$S/f32.gguf" "$S/$1.gguf" q4_k --tensor-type 'v.*=f16' \
            --tensor-type 'l.embed_tokens.weight=q8_0' > "$OUT/quantize-$1.log" 2>&1
        echo "$S/$1.gguf" ;;
    q4_k-dq8)
        [ -f "$S/$1.gguf" ] || "$B/crispembed-quantize" "$S/f32.gguf" "$S/$1.gguf" q4_k --tensor-type 'l.*=q8_0' \
            > "$OUT/quantize-$1.log" 2>&1
        echo "$S/$1.gguf" ;;
    q4_k)
        [ -f "$S/$1.gguf" ] || "$B/crispembed-quantize" "$S/f32.gguf" "$S/$1.gguf" q4_k > "$OUT/quantize-$1.log" 2>&1
        echo "$S/$1.gguf" ;;
    esac
}
extra=()
[ -n "$CASES" ] && extra=(--cases "$CASES")
run() {
    python tools/qwen2vl_device_ab.py --model mineru --gguf "$(gguf_for "$1")" --precision "$1" --device cpu \
        --build "$B" --out "$OUT" --repeat 2 --hash --dump "$S/dump/$1-cpu" "${extra[@]}"
}
run f32
IFS=, read -ra vs <<< "$VARIANTS"
for v in "${vs[@]}"; do
    run "$v"
    python tools/qwen2vl_stage_cosine.py --ref "$S/dump/f32-cpu" --cand "$S/dump/$v-cpu" --ref-receipt "$OUT/f32-cpu.json" \
        --cand-receipt "$OUT/$v-cpu.json" --width 896 --out "$OUT/$v-cpu.cosine.json"
done
python tools/qwen2vl_device_summary.py "$OUT" --cosine-dir "$OUT" --out "$OUT/summary.json" > "$OUT/summary.md"
cat "$OUT/summary.md"
