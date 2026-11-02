#!/usr/bin/env bash
# Q4_K recipe sweep for TeleOCR and MinerU2.5-Pro (docs/qwen2vl-q4k-recipes.md).
# For each named recipe in tools/qwen2vl_quant_recipes.json: quantize, run every
# frozen case teacher-forced on the committed original-Python IDs (CPU), and
# score the final-output cosine (logits at every generated position) against
# the native CPU reference of docs/qwen2vl-metal.md, plus the file size and the
# stored tensor-type census. Teacher-forced only: free-running decoded agreement
# of a winner is measured separately (tools/ci-heavy/qwen2vl_cpu_cosine.sh).
#
#   gh workflow run heavy-cpu.yml --ref feat/qwen2vl-q4k-recipes \
#       -f script=tools/ci-heavy/qwen2vl_q4k_sweep.sh -f args="teleocr q8,aq,ak" \
#       -f pip="numpy==2.2.6 safetensors==0.6.2 huggingface_hub==0.36.0 gguf==0.17.1" \
#       -f isolated_lane=qwen2vl-q4k-sweep-teleocr-a
set -uo pipefail
MODEL=${1:?mineru|teleocr}
RECIPES=${2:?comma list of recipe names}
KEEP=${3:-} # comma list of recipes whose GGUF is copied to $HEAVY_OUT (device confirmation elsewhere)
OUT=${HEAVY_OUT:?}
S=${HEAVY_SCRATCH:?}
(lscpu 2>/dev/null || sysctl -a 2>/dev/null | grep -i brand) > "$OUT/cpu.txt" # -march=native: kernels follow the runner ISA
cmake -S . -B "$S/build" -DCMAKE_BUILD_TYPE=Release > "$OUT/cmake.log" 2>&1 &&
    cmake --build "$S/build" --target test-mineru25-run test-teleocr-run crispembed-quantize -j"$(nproc)" \
        > "$OUT/build.log" 2>&1 || { tail -40 "$OUT/build.log"; exit 2; }
B=$(dirname "$(find "$S/build" -name test-mineru25-run -type f | head -1)")
Q=$B/crispembed-quantize
[ -x "$Q" ] || Q=$(find "$S/build" -name crispembed-quantize -type f | head -1)
if [ "$MODEL" = mineru ]; then
    REF=f32
    WIDTH=896
    python models/convert-mineru25-pro-to-gguf.py --output "$S/f32.gguf" --dtype f32 > "$OUT/convert-f32.log" 2>&1 ||
        { tail -20 "$OUT/convert-f32.log"; exit 2; }
else
    REF=f16
    WIDTH=1024
    curl -sSL --retry 5 -o "$S/f16.gguf" \
        "https://huggingface.co/cstr/TeleOCR-GGUF/resolve/1de0119f27f470cfc7ef0be14327f5953730c7e8/teleocr-f16.gguf" ||
        exit 2
fi
sha256sum "$S/$REF.gguf" > "$OUT/source-sha256.txt"
python tools/qwen2vl_device_ab.py --model "$MODEL" --gguf "$S/$REF.gguf" --precision "$REF" --device cpu \
    --build "$B" --out "$OUT" --repeat 1 --force-reference --label forced --dump "$S/dump/ref"
# Importance matrix (recipes with "imatrix": true): activation statistics of the
# unquantized source on calibration images that are NOT frozen evaluation cases.
calibrate() {
    local I=tests/regression/images C=$S/imatrix.gguf
    [ -f "$C" ] && return 0
    if [ "$MODEL" = mineru ]; then
        for c in "$I/cc0/commons_test_ocr_document.jpg text" "$I/cc0/simple_form.png text" \
            "$I/cc0/receipt_historical.png text" "$I/cc0/public_domain_formula_photo.jpg formula" \
            "$I/page_transcoda.png layout" "$I/mixtex_pow.png formula"; do
            set -- $c
            CRISPEMBED_MINERU25_PRO=1 QWEN2VL_OCR_FORCE_CPU=1 CRISPEMBED_IMATRIX_OUT=$C "$B/test-mineru25-run" \
                --model "$S/$REF.gguf" --image "$1" --task "$2" --max-tokens 256 --threads 4 --repeat 1 \
                >> "$OUT/imatrix-calibration.log" 2>&1
        done
    else
        local t="Please output the text content from the image."
        for c in "$I/cc0/commons_test_ocr_document.jpg|$t" "$I/cc0/simple_form.png|$t" "$I/cc0/receipt_historical.png|$t" \
            "$I/cc0/public_domain_formula_photo.jpg|Please write out the expression of the formula in the image using LaTeX format." \
            "$I/page_transcoda.png|Analyze the image layout." "$I/mixtex_pow.png|Please write out the expression of the formula in the image using LaTeX format."; do
            CRISPEMBED_TELEOCR=1 CRISPEMBED_ACCEPT_QWEN_RESEARCH_LICENSE=1 QWEN2VL_OCR_FORCE_CPU=1 CRISPEMBED_IMATRIX_OUT=$C \
                "$B/test-teleocr-run" "$S/$REF.gguf" "${c%%|*}" "${c#*|}" 256 4 1 >> "$OUT/imatrix-calibration.log" 2>&1
        done
    fi
    [ -f "$C" ] && cp "$C" "$OUT/imatrix.gguf"
}
fail=0
IFS=, read -ra names <<< "$RECIPES"
for n in "${names[@]}"; do
    mapfile -t qargs < <(python tools/qwen2vl_quant_census.py args "$MODEL" "$n") || { fail=1; continue; }
    mapfile -t qenv < <(python tools/qwen2vl_quant_census.py env "$MODEL" "$n")
    src=$(python -c "import json,sys;print(json.load(open('tools/qwen2vl_quant_recipes.json'))[sys.argv[1]][sys.argv[2]]['source'])" "$MODEL" "$n")
    f=$S/$n.gguf
    if python -c "import json,sys;sys.exit(0 if json.load(open('tools/qwen2vl_quant_recipes.json'))[sys.argv[1]][sys.argv[2]].get('imatrix') else 1)" "$MODEL" "$n"; then
        calibrate && qargs+=(--imatrix "$S/imatrix.gguf") || { echo "$n: calibration failed"; fail=1; continue; }
    fi
    env "${qenv[@]}" "$Q" "$S/$src.gguf" "$f" "${qargs[@]}" > "$OUT/quantize-$n.log" 2>&1 ||
        { echo "$n: quantize failed"; tail -5 "$OUT/quantize-$n.log"; fail=1; continue; }
    python tools/qwen2vl_quant_census.py census "$f" "$OUT/$n.census.json"
    python tools/qwen2vl_device_ab.py --model "$MODEL" --gguf "$f" --precision "$n" --device cpu --build "$B" \
        --out "$OUT" --repeat 1 --hash --force-reference --label forced --dump "$S/dump/$n"
    python tools/qwen2vl_stage_cosine.py --ref "$S/dump/ref" --cand "$S/dump/$n" \
        --ref-receipt "$OUT/$REF-cpu-forced.json" --cand-receipt "$OUT/$n-cpu-forced.json" --width $WIDTH \
        --out "$OUT/$n-cpu.cosine.json" | tail -2
    case ",$KEEP," in *",$n,"*) mv "$f" "$OUT/$MODEL-$n.gguf" ;; esac
    rm -rf "$S/dump/$n" "$f"
    python tools/qwen2vl_quant_census.py summary "$MODEL" "$OUT" > "$OUT/summary.md"
done
cat "$OUT/summary.md"
exit $fail
