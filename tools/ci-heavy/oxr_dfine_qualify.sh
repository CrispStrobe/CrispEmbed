#!/usr/bin/env bash
# OXR-1.0 D-FINE layout: native layout_detect (layout.arch=dfine) vs the pinned
# original PyTorch service, plus heron (rt_detr_v2) byte-identity vs the base.
#
#   gh workflow run heavy-cpu.yml -R CrispStrobe/CrispEmbed --ref feat/oxr-dfine \
#       -f script=tools/ci-heavy/oxr_dfine_qualify.sh -f isolated_lane=oxr-dfine [-f args="base=<ref>"]
#
# Writes $HEAVY_OUT/{provenance,results,summary}.* and per-page directories.
# Every gate is recorded verbatim in results.txt; budgets are fixed in
# tests/oxr_dfine_diff.cpp (stage rel <= 5e-3, selected boxes <= 0.5 px,
# selected scores <= 1e-3) and are not changed per run.
set -uo pipefail
TOP=${HEAVY_OUT:?}
S=${HEAVY_SCRATCH:?}
BASE=origin/feat/teleocr
for a in "$@"; do case "$a" in base=*) BASE=${a#base=} ;; esac; done
export CRISPEMBED_OXR=1 CRISPEMBED_ACCEPT_OXR_LICENSE=1 LAYOUT_DETECT_FORCE_CPU=1 \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
REV=c708407f9d2b4df658f2543d9132b62bb7221df4
BP=c1a1dcb312ee9f691404f127785510453c0d9350
PAGES="receipt:cc0/receipt_example.png german:cc0/german_official_print.jpg scan:scan_page_pd.png table:cc0/simple_table.jpg japanese:japanese_print.png"
note() { echo "$1=$2" >> "$TOP/results.txt"; }

{ echo "source=$(git rev-parse HEAD)"; git submodule status; nproc; lscpu | grep -E 'Model name|^CPU\(s\)'; } > "$TOP/provenance.txt"

echo "::group::build (branch)"
cmake -S . -B "$S/build" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DCRISPEMBED_NATIVE=OFF >/dev/null &&
    cmake --build "$S/build" --target crispembed-shared test-oxr-dfine-unit test-oxr-dfine-diff test-layout-detect \
        -j4 > "$S/build.log" 2>&1 || { tail -60 "$S/build.log"; exit 4; }
"$S/build/test-oxr-dfine-unit" > "$TOP/unit.txt" 2>&1; note unit "$([ $? = 0 ] && echo pass || echo fail)"
LIB=$(find "$S/build" -name 'libcrispembed.so*' -type f | head -1)
echo "::endgroup::"

echo "::group::reference runtime"
python -m pip install -q torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu &&
    python -m pip install -q numpy==1.26.4 opencv-python-headless==4.11.0.86 gguf pillow huggingface_hub pytest \
        pydantic httpx || exit 3
python -m pip freeze > "$TOP/packages.txt"
python -m pytest -q tests/test_oxr_dfine_conversion.py > "$TOP/converter-test.txt" 2>&1
note converter_tests "$([ $? = 0 ] && echo pass || echo fail)"
git clone -q https://github.com/SoMarkAI/OXR.git "$S/oxr" && git -C "$S/oxr" checkout -q $BP || exit 3
echo "::endgroup::"

echo "::group::authenticated download and conversion"
CKPT=$(python -c "from huggingface_hub import hf_hub_download as d; d('SoMarkAI/OXR-1.0','layout_analysis/metadata.json',revision='$REV'); print(d('SoMarkAI/OXR-1.0','layout_analysis/last.pth',revision='$REV'))") || exit 5
for dt in f32 f16; do
    python models/convert-oxr-dfine-to-gguf.py --model SoMarkAI/OXR-1.0 --dtype $dt --output "$S/oxr-dfine-$dt.gguf" \
        > "$TOP/convert-$dt.log" 2>&1
    note "convert_$dt" "$([ $? = 0 ] && echo pass || echo fail)"
    cp "$S/oxr-dfine-$dt.gguf.json" "$TOP/conversion-$dt.json" 2>/dev/null
done
sha256sum "$S"/oxr-dfine-*.gguf > "$TOP/gguf-sha256.txt"
python tools/oxr_dfine_checks.py lineage --checkpoint "$CKPT" --output "$TOP/lineage.json" --workdir "$S/lineage" \
    > "$TOP/lineage.log" 2>&1
note lineage "$([ $? = 0 ] && echo recorded || echo fail)"
rm -rf "$S/lineage"
echo "::endgroup::"

for entry in $PAGES; do
    id=${entry%%:*}; src=${entry#*:}; O=$TOP/$id; mkdir -p "$O"
    pn() { echo "$1=$2" >> "$O/results.txt"; }
    # Decode JPEGs once with PIL; both sides read the same lossless RGB PNG.
    python -c "import sys; from PIL import Image; Image.open(sys.argv[1]).convert('RGB').save(sys.argv[2])" \
        "tests/regression/images/$src" "$S/$id.png"
    echo "::group::$id reference"
    for mode in service generic; do
        python tools/oxr_dfine_reference.py --blueprint "$S/oxr" --checkpoint "$CKPT" --image "$S/$id.png" \
            --resize $mode --threads 4 --repeat 5 --dump "$S/$id-$mode.gguf" --output "$O/reference-$mode.json" \
            > "$O/reference-$mode.log" 2>&1
        pn "reference_$mode" "$([ $? = 0 ] && echo pass || echo fail)"
    done
    # Exact-arithmetic yardstick: the same original model in float64 (diagnostic only).
    python tools/oxr_dfine_reference.py --blueprint "$S/oxr" --checkpoint "$CKPT" --image "$S/$id.png" \
        --resize generic --threads 4 --repeat 1 --float64 --dump "$S/$id-f64.gguf" --output "$O/reference-f64.json" \
        > "$O/reference-f64.log" 2>&1
    pn reference_f64 "$([ $? = 0 ] && echo pass || echo fail)"
    # PyTorch thread noise floor: same reference with one thread.
    python tools/oxr_dfine_reference.py --blueprint "$S/oxr" --checkpoint "$CKPT" --image "$S/$id.png" \
        --resize generic --threads 1 --repeat 1 --dump "$S/$id-generic-t1.gguf" --output "$O/reference-generic-t1.json" \
        > /dev/null 2>&1 && python tools/oxr_dfine_checks.py refdiff "$S/$id-generic-t1.gguf" "$S/$id-generic.gguf" \
        --output "$O/reference-noise-floor.json"
    echo "::endgroup::"
    echo "::group::$id native"
    # Arms: generic (native resize = OpenCV generic Lanczos4; reference with IPP off),
    # service (shipped cv2 default, IPP on x86), inject (reference input tensor), f16.
    run_arm() {
        local arm=$1 model=$2 ref=$3; shift 3
        "$S/build/test-oxr-dfine-diff" --model "$model" --reference "$ref" --image "$S/$id.png" --threads 4 \
            --repeat 5 "$@" > "$O/diff-$arm.json" 2> "$O/diff-$arm.log"
        local rc=$?
        # Stage budget and final selection are separate gates, both recorded.
        python - "$O/diff-$arm.json" "$arm" "$rc" >> "$O/results.txt" <<'PY'
import json, sys
path, arm, rc = sys.argv[1:]
try:
    d = json.load(open(path))
except Exception:
    print(f"diff_{arm}=fail(rc={rc},no-json)"); sys.exit()
for key, name in (("stage_budget_pass", "stages"), ("selection_pass", "selection")):
    print(f"diff_{arm}_{name}={'pass' if d.get(key) else 'fail'}")
print(f"diff_{arm}_earliest_divergence={d.get('earliest_divergence')}")
PY
    }
    mkdir -p "$S/$id-native"
    run_arm generic "$S/oxr-dfine-f32.gguf" "$S/$id-generic.gguf" --dump-dir "$S/$id-native"
    python tools/oxr_dfine_checks.py yardstick "$S/$id-generic.gguf" "$S/$id-f64.gguf" "$S/$id-native" \
        --output "$O/yardstick-f64.json" > "$O/yardstick.log" 2>&1
    pn yardstick_f64 "$([ $? = 0 ] && echo recorded || echo fail)"
    rm -rf "$S/$id-native"
    run_arm service "$S/oxr-dfine-f32.gguf" "$S/$id-service.gguf"
    run_arm inject-service "$S/oxr-dfine-f32.gguf" "$S/$id-service.gguf" --inject-input
    run_arm f16-generic "$S/oxr-dfine-f16.gguf" "$S/$id-generic.gguf"
    # Speed A/B of opt-in convolution paths (same budgets; defaults unchanged).
    CRISPEMBED_OXR_LAYOUT_DW_DIRECT=1 run_arm dw-direct "$S/oxr-dfine-f32.gguf" "$S/$id-generic.gguf"
    LAYOUT_CONV_DIRECT=1 run_arm conv-direct "$S/oxr-dfine-f32.gguf" "$S/$id-generic.gguf"
    python tools/oxr_dfine_checks.py api --lib "$LIB" --model "$S/oxr-dfine-f32.gguf" --image "$S/$id.png" \
        --reference "$O/reference-generic.json" --blueprint "$S/oxr" --output "$O/api.json" > "$O/api.log" 2>&1
    pn api "$([ $? = 0 ] && echo pass || echo fail)"
    rm -f "$S/$id"-*.gguf
    echo "::endgroup::"
    cat "$O/results.txt"
done

echo "::group::heron (rt_detr_v2) identity vs $BASE"
HERON=$(python -c "from huggingface_hub import hf_hub_download as d; print(d('cstr/layout-heron-gguf','layout-heron-f32.gguf'))")
git fetch -q origin "+refs/heads/feat/teleocr:refs/remotes/origin/feat/teleocr" 2>/dev/null
git worktree add -q "$S/base" "$BASE" && git -C "$S/base" submodule update -q --init ggml &&
    cmake -S "$S/base" -B "$S/build-base" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DCRISPEMBED_NATIVE=OFF >/dev/null &&
    cmake --build "$S/build-base" --target crispembed-shared -j4 > "$S/build-base.log" 2>&1
BASELIB=$(find "$S/build-base" -name 'libcrispembed.so*' -type f | head -1)
IMGS=$(for entry in $PAGES; do echo "$S/${entry%%:*}.png"; done)
python tools/oxr_dfine_checks.py heron --lib "$BASELIB" --model "$HERON" --output "$TOP/heron-base.json" --images $IMGS \
    > "$TOP/heron-base.log" 2>&1
python tools/oxr_dfine_checks.py heron --lib "$LIB" --model "$HERON" --output "$TOP/heron-branch.json" --images $IMGS \
    > "$TOP/heron-branch.log" 2>&1
note heron_byte_identical "$(cmp -s "$TOP/heron-base.json" "$TOP/heron-branch.json" && echo pass || echo fail)"
{ echo "base=$(git -C "$S/base" rev-parse HEAD)"; cat "$TOP/heron-base.log" "$TOP/heron-branch.log"; } > "$TOP/heron.txt"
echo "::endgroup::"

python - "$TOP" > "$TOP/summary.json" <<'PY'
import json, sys, pathlib
top = pathlib.Path(sys.argv[1]); out = {}
for d in sorted(p for p in top.iterdir() if p.is_dir()):
    page = {}
    for f in sorted(d.glob("diff-*.json")):
        try:
            j = json.loads(f.read_text())
        except Exception as e:
            page[f.stem] = {"error": str(e)}; continue
        worst = max(j.get("stages", []), key=lambda s: s["rel"], default={})
        page[f.stem] = {k: j.get(k) for k in ("pass", "earliest_divergence", "topk_query_selection_equal",
                                               "query_alignment_applied", "timing_ms", "candidates",
                                               "matched_candidates_score_ge_0.05")}
        page[f.stem]["worst_stage"] = {k: worst.get(k) for k in ("name", "max_abs", "rel", "cosine")}
        sel = j.get("oxr_selection", {})
        page[f.stem]["oxr_selection"] = {k: v for k, v in sel.items() if not k.endswith("regions")}
    for mode in ("generic", "service"):
        r = d / f"reference-{mode}.json"
        if r.exists():
            j = json.loads(r.read_text())
            page[f"reference_{mode}"] = {k: j.get(k) for k in ("resize", "ipp_available",
                                         "ipp_vs_generic_input_max_abs", "forward_ms_median", "preprocess_ms_median")}
    out[d.name] = page
print(json.dumps(out, indent=1))
PY
{ echo "### OXR D-FINE layout qualification ($(git rev-parse --short HEAD))"; echo '```'
  cat "$TOP/results.txt"; for entry in $PAGES; do id=${entry%%:*}; echo "-- $id"; cat "$TOP/$id/results.txt" 2>/dev/null; done
  echo '```'; } > "$TOP/summary.md"
! grep -rqs '=fail' "$TOP"/results.txt "$TOP"/*/results.txt
