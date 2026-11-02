#!/usr/bin/env bash
# OXR D-FINE layout: F16 and quantized recipes vs the original PyTorch service (CPU),
# judged by docs/precision-acceptance.md (final selected boxes/scores cosine, region
# match) with the strict stage/region budgets of tests/oxr_dfine_diff.cpp recorded.
#   gh workflow run heavy-cpu.yml --ref <branch> -f script=tools/ci-heavy/oxr_dfine_quant.sh -f isolated_lane=oxr-dfine-quant
# Recipes: f32 f16 f16-as-f32, q8_0/q4_k with conv kernels quantized (convq), kept F32
# (convkeep, CRISPEMBED_QUANTIZE_DFINE_CONV=keep) or F16 (convf16).
set -uo pipefail
TOP=${HEAVY_OUT:?}; S=${HEAVY_SCRATCH:?}
REV=c708407f9d2b4df658f2543d9132b62bb7221df4; BP=c1a1dcb312ee9f691404f127785510453c0d9350
export CRISPEMBED_OXR=1 CRISPEMBED_ACCEPT_OXR_LICENSE=1 LAYOUT_DETECT_FORCE_CPU=1 OMP_NUM_THREADS=4
note() { echo "$1=$2" >> "$TOP/results.txt"; }
echo "source=$(git rev-parse HEAD)" > "$TOP/provenance.txt"
cmake -S . -B "$S/build" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DCRISPEMBED_NATIVE=OFF >/dev/null &&
    cmake --build "$S/build" --target test-oxr-dfine-diff crispembed-quantize -j4 > "$S/build.log" 2>&1 || { tail -40 "$S/build.log"; exit 4; }
python -m pip install -q torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu &&
    python -m pip install -q numpy==1.26.4 opencv-python-headless==4.11.0.86 gguf pillow huggingface_hub pydantic httpx || exit 3
git clone -q https://github.com/SoMarkAI/OXR.git "$S/oxr" && git -C "$S/oxr" checkout -q $BP || exit 3
CKPT=$(python -c "from huggingface_hub import hf_hub_download as d; d('SoMarkAI/OXR-1.0','layout_analysis/metadata.json',revision='$REV'); print(d('SoMarkAI/OXR-1.0','layout_analysis/last.pth',revision='$REV'))") || exit 5
for dt in f32 f16 f16-as-f32; do
    python models/convert-oxr-dfine-to-gguf.py --model SoMarkAI/OXR-1.0 --dtype $dt --output "$S/d-$dt.gguf" > "$TOP/convert-$dt.log" 2>&1
    note "convert_$dt" "$([ $? = 0 ] && echo pass || echo fail)"
done
Q=$(find "$S/build" -type f -name crispembed-quantize -perm -u+x | head -1)
RECIPES="f32 f16 f16-as-f32"
for q in q8_0 q4_k; do
    for c in convq:"" convkeep:keep convf16:f16; do
        name=$q-${c%%:*}; v=${c#*:}
        env ${v:+CRISPEMBED_QUANTIZE_DFINE_CONV=$v} "$Q" "$S/d-f32.gguf" "$S/d-$name.gguf" $q > "$TOP/quantize-$name.log" 2>&1
        note "quantize_$name" "$([ $? = 0 ] && echo pass || echo fail)"; RECIPES="$RECIPES $name"
    done
done
ls -l "$S"/d-*.gguf > "$TOP/sizes.txt"
for e in receipt:cc0/receipt_example.png german:cc0/german_official_print.jpg scan:scan_page_pd.png table:cc0/simple_table.jpg japanese:japanese_print.png; do
    id=${e%%:*}; src=${e#*:}; O=$TOP/$id; mkdir -p "$O"
    python -c "import sys; from PIL import Image; Image.open(sys.argv[1]).convert('RGB').save(sys.argv[2])" "tests/regression/images/$src" "$S/$id.png"
    python tools/oxr_dfine_reference.py --blueprint "$S/oxr" --checkpoint "$CKPT" --image "$S/$id.png" --resize generic \
        --threads 4 --repeat 1 --dump "$S/$id-ref.gguf" --output "$O/reference-generic.json" > "$O/reference.log" 2>&1
    for r in $RECIPES; do
        "$S/build/test-oxr-dfine-diff" --model "$S/d-$r.gguf" --reference "$S/$id-ref.gguf" --image "$S/$id.png" \
            --threads 4 --repeat 3 > "$O/diff-$r.json" 2> "$O/diff-$r.log"
        echo "$id $r exit=$?" >> "$TOP/results.txt"
    done
    rm -f "$S/$id-ref.gguf"
done
python tools/metal/summarize_dfine_ab.py "$TOP" > "$TOP/summary.json"
cat "$TOP/results.txt"
