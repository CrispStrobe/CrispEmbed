#!/usr/bin/env bash
# Unlimited-OCR-2 Q4_K recipe sweep (docs/unlimited-ocr2-q4k-recipes.md).
# For each named recipe in tools/unlimited_ocr2_quant_recipes.json: quantize the
# published F16 GGUF, run the four frozen cases teacher-forced on the original-
# Python IDs (tests/results/unlimited-ocr2/trial.json) and score the final-output
# cosine (lm_head logits at every forced position plus the EOS position) against
# the native F16 run on the same forced IDs. Native F16 tiled+fast reproduces the
# original IDs/text exactly on all four cases (trial.json, mixed-q4-v3-review.json).
# Free-running decoded IDs are compared with the original IDs for the recipes in
# the second argument. Portable build (GGML_NATIVE=OFF): CPU scores do not depend
# on the runner's ISA extensions.
#
#   gh workflow run heavy-cpu.yml --ref <branch> -f script=tools/ci-heavy/unlimited_ocr2_q4k_sweep.sh \
#       -f args="q8,q4k,exp4 f16,q8,exp4" -f pip="numpy==2.2.6 gguf==0.17.1 pillow==11.3.0 matplotlib==3.10.7" \
#       -f isolated_lane=q4k-unlimited-ocr2-a
set -uo pipefail
RECIPES=${1:?comma list of recipe names}
FREE=${2:-}  # comma list (may include f16) of free-running decoded checks
KEEP=${3:-}  # comma list of recipes whose GGUF is copied to $HEAVY_OUT
OUT=${HEAVY_OUT:?}
S=${HEAVY_SCRATCH:?}
T=$(nproc)
# DEVICE=gpu (Kaggle CUDA confirmation, tools/kaggle/unlimited-ocr2-q4k-cuda): the F16
# reference stays on the CPU; every candidate, plus an F16 control "f16cuda", runs on CUDA.
DEVICE=${DEVICE:-cpu}
CMAKE_EXTRA=${CMAKE_EXTRA:-}
H=tools/unlimited_ocr2_q4k.py
CASES="paragraphs table receipt german"
{
    echo "build_mode: portable (-DGGML_NATIVE=OFF -DCRISPEMBED_NATIVE=OFF)"
    echo "commit: $(git rev-parse HEAD)"
    lscpu 2>/dev/null
} > "$OUT/cpu.txt"
echo "script_build_flags=-DGGML_NATIVE=OFF -DCRISPEMBED_NATIVE=OFF (portable) device=$DEVICE extra=$CMAKE_EXTRA" >> "$OUT/build-mode.txt"

python $H render "$S/pages" > "$OUT/render.log" 2>&1 ||
    { tail -20 "$OUT/render.log"; exit 2; }
python $H cases "$S/pages" "$S/ids" > "$OUT/cases.json" || { cat "$OUT/cases.json"; echo "frozen PNG reproduction failed"; exit 2; }
cp "$S/pages"/*.png "$OUT/" 2>/dev/null
[ "$RECIPES" = render ] && { cat "$OUT/cases.json"; exit 0; }

cmake -S . -B "$S/build" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DCRISPEMBED_NATIVE=OFF \
    -DGGML_METAL=OFF -DGGML_VULKAN=OFF ${CMAKE_EXTRA:--DGGML_CUDA=OFF} > "$OUT/cmake.log" 2>&1 &&
    cmake --build "$S/build" --target crispembed-cli crispembed-quantize -j"$T" > "$OUT/build.log" 2>&1 ||
    { tail -40 "$OUT/build.log"; exit 2; }
CLI=$(find "$S/build" -name crispembed -type f -perm -u+x | head -1)
Q=$(find "$S/build" -name crispembed-quantize -type f -perm -u+x | head -1)
grep -E "GGML_NATIVE|CRISPEMBED_NATIVE" "$S/build/CMakeCache.txt" >> "$OUT/build-mode.txt"

F16=$S/f16.gguf
curl -sSL --retry 5 -o "$F16" \
    "https://huggingface.co/cstr/Unlimited-OCR-2-GGUF/resolve/e3af0c0c4bfa241c161db64ca36f3c63d3ca8378/unlimited-ocr-2-f16.gguf" ||
    exit 2
echo "487b887eddaf73e070cd98c3e10d112d21749036ca06b5e7fc103c3fb4def517  $F16" | sha256sum -c - > "$OUT/source-sha256.txt" ||
    { echo "published F16 SHA differs"; exit 2; }
mkdir -p "$S/logits"

# run <label> <gguf> <case> forced|free
run() {
    local env=(CRISPEMBED_UNLIMITED_OCR2=1 CRISPEMBED_UNLIMITED_OCR2_FAST=1 CRISPEMBED_UNLIMITED_OCR2_TILES=1
        UOCR_DBG=1 OMP_NUM_THREADS="$T")
    [ "$4" = forced ] && env+=(CRISPEMBED_UNLIMITED_OCR2_FORCE_IDS="$S/ids/$3.ids"
        CRISPEMBED_UNLIMITED_OCR2_LOGITS_OUT="$S/logits/$1.$3.f32")
    [ "$DEVICE" = gpu ] && [ "$1" = f16 ] && env+=(CUDA_VISIBLE_DEVICES=)
    env "${env[@]}" "$CLI" -m "$2" --ocr "$S/pages/$3.png" --ocr-max-tokens 1024 -t "$T" \
        > "$OUT/$1.$3.$4.txt" 2> "$S/$1.$3.$4.err"
    local rc=$?
    grep -E "gen_ids|decode done|error|unlimited_ocr2:" "$S/$1.$3.$4.err" > "$OUT/$1.$3.$4.log"
    return $rc
}

# panel <label> <gguf>
panel() {
    local t0=$SECONDS c n
    for c in $CASES; do
        run "$1" "$2" "$c" forced || echo "$1/$c forced run failed"
        if [ "$1" != f16 ]; then
            n=$(wc -w < "$S/ids/$c.ids")
            python $H cosine "$S/logits/f16.$c.f32" "$S/logits/$1.$c.f32" "$n" "$OUT/$1.$c.cosine.json" | tail -1
            rm -f "$S/logits/$1.$c.f32"
        fi
        case ",$FREE," in *",$1,"*)
            run "$1" "$2" "$c" free || echo "$1/$c free run failed"
            python $H free "$S/$1.$c.free.err" "$c" "$OUT/$1.$c.free.json" ;;
        esac
    done
    echo $((SECONDS - t0)) > "$OUT/$1.wall"
}

panel f16 "$F16"
[ "$DEVICE" = gpu ] && panel f16cuda "$F16"
fail=0
IFS=, read -ra names <<< "$RECIPES"
for n in "${names[@]}"; do
    mapfile -t qargs < <(python $H args "$n") || { fail=1; continue; }
    f=$S/$n.gguf
    "$Q" "$F16" "$f" "${qargs[@]}" > "$OUT/quantize-$n.log" 2>&1 ||
        { echo "$n: quantize failed"; tail -5 "$OUT/quantize-$n.log"; fail=1; continue; }
    python $H census "$f" "$OUT/$n.census.json" "$F16" || { echo "$n: protected tensor check failed"; fail=1; }
    panel "$n" "$f"
    case ",$KEEP," in *",$n,"*) mv "$f" "$OUT/unlimited-ocr-2-$n.gguf" ;; esac
    rm -f "$f"
    python $H summary "$OUT" > "$OUT/summary.md"
done
python $H summary "$OUT" > "$OUT/summary.md"
cat "$OUT/summary.md"
exit $fail
