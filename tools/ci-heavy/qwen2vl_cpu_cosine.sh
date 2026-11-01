#!/usr/bin/env bash
# CPU arm of the qwen2vl_ocr precision/device study (docs/qwen2vl-metal.md,
# docs/precision-acceptance.md): for every variant, free-running decoded IDs
# against the committed original-Python IDs, and the final-output cosine — the
# logits at every generated position, teacher-forced on those reference IDs —
# against a native CPU reference, plus the worst intermediate-stage cosine.
#
#   gh workflow run heavy-cpu.yml --ref feat/metal-qwen2vl \
#       -f script=tools/ci-heavy/qwen2vl_cpu_cosine.sh -f args="mineru f16,q8_0,q4_k,q4_k-dq8,q4_k-df16,q4_k-vf16" \
#       -f pip="numpy==2.2.6 safetensors==0.6.2 huggingface_hub==0.36.0 gguf==0.17.1" \
#       -f isolated_lane=qwen2vl-cpu-cosine-mineru
#   ... -f args="teleocr q8_0,q4_k-m,q4_k,q4_k-attn8" -f isolated_lane=qwen2vl-cpu-cosine-teleocr
#
# References: MinerU = F32 converted from the pinned checkpoint; TeleOCR = the
# published F16 GGUF on CPU (its CPU path dequantizes to F32 and computes in
# F32; the F16 weights are the bf16 originals). The reference's teacher-forced
# logit dumps are uploaded (reference-dump.tar) for the local Metal scoring.
set -uo pipefail
MODEL=${1:?mineru|teleocr}
VARIANTS=${2:?comma list}
CASES=${3:-}
OUT=${HEAVY_OUT:?}
S=${HEAVY_SCRATCH:?}
cmake -S . -B "$S/build" -DCMAKE_BUILD_TYPE=Release > "$OUT/cmake.log" 2>&1 &&
    cmake --build "$S/build" --target test-mineru25-run test-teleocr-run crispembed-quantize -j"$(nproc)" \
        > "$OUT/build.log" 2>&1 || { tail -40 "$OUT/build.log"; exit 2; }
B=$(dirname "$(find "$S/build" -name test-mineru25-run -type f | head -1)")
Q=$B/crispembed-quantize
[ -x "$Q" ] || Q=$(find "$S/build" -name crispembed-quantize -type f | head -1)
if [ "$MODEL" = mineru ]; then
    PUB=https://huggingface.co/cstr/MinerU2.5-Pro-2604-1.2B-GGUF/resolve/e63c95cfd4e9f753783f588a187b9a3b89a7c201
    PRE=mineru25-pro-2604-
    REF=f32
    WIDTH=896
    python models/convert-mineru25-pro-to-gguf.py --output "$S/f32.gguf" --dtype f32 > "$OUT/convert-f32.log" 2>&1 ||
        { tail -20 "$OUT/convert-f32.log"; exit 2; }
    SRC=$S/f32.gguf
    PROTECT=()
else
    PUB=https://huggingface.co/cstr/TeleOCR-GGUF/resolve/1de0119f27f470cfc7ef0be14327f5953730c7e8
    PRE=teleocr-
    REF=f16
    WIDTH=1024
    curl -sSL --retry 5 -o "$S/f16.gguf" "${PUB}/teleocr-f16.gguf" || exit 2
    SRC=$S/f16.gguf
    PROTECT=(--tensor-type 'v.*=f16' --tensor-type 'l.embed_tokens.weight=f16')
fi
gguf_for() {
    local f=$S/$1.gguf
    [ -f "$f" ] && { echo "$f"; return; }
    case $1 in
    f16 | q8_0 | q4_k-m) curl -sSL --retry 5 -o "$f" "${PUB}/${PRE}$1.gguf" ;;
    q4_k) "$Q" "$SRC" "$f" q4_k "${PROTECT[@]}" > "$OUT/quantize-$1.log" 2>&1 ;;
    q4_k-dq8) "$Q" "$SRC" "$f" q4_k --tensor-type 'l.*=q8_0' > "$OUT/quantize-$1.log" 2>&1 ;;
    q4_k-df16) "$Q" "$SRC" "$f" q4_k --decoder-f16 > "$OUT/quantize-$1.log" 2>&1 ;;
    q4_k-vf16) "$Q" "$SRC" "$f" q4_k --tensor-type 'v.*=f16' --tensor-type 'l.embed_tokens.weight=q8_0' \
        > "$OUT/quantize-$1.log" 2>&1 ;;
    q4_k-attn8) "$Q" "$SRC" "$f" q4_k "${PROTECT[@]}" --tensor-type 'l.blk.*.attn_*=q8_0' > "$OUT/quantize-$1.log" 2>&1 ;;
    esac
    echo "$f"
}
extra=()
[ -n "$CASES" ] && extra=(--cases "$CASES")
ab() { # variant, extra args...
    local v=$1
    shift
    python tools/qwen2vl_device_ab.py --model "$MODEL" --gguf "$(gguf_for "$v")" --precision "$v" --device cpu \
        --build "$B" --out "$OUT" --repeat 1 --hash "${extra[@]}" "$@"
}
ab "$REF" --force-reference --label forced --dump "$S/dump/$REF-forced"
tar -C "$S/dump" -cf "$OUT/reference-dump.tar" "$REF-forced"
IFS=, read -ra vs <<< "$VARIANTS"
for v in "${vs[@]}"; do
    ab "$v"
    ab "$v" --force-reference --label forced --dump "$S/dump/$v-forced"
    python tools/qwen2vl_stage_cosine.py --ref "$S/dump/$REF-forced" --cand "$S/dump/$v-forced" \
        --ref-receipt "$OUT/$REF-cpu-forced.json" --cand-receipt "$OUT/$v-cpu-forced.json" --width $WIDTH \
        --out "$OUT/$v-cpu.cosine.json" | tail -2
    rm -rf "$S/dump/$v-forced" "$S/$v.gguf"
done
ls -la "$S"/*.gguf > "$OUT/gguf-files.txt" 2>&1
python tools/qwen2vl_device_summary.py "$OUT" --cosine-dir "$OUT" --out "$OUT/summary.json" > "$OUT/summary.md"
cat "$OUT/summary.md"
