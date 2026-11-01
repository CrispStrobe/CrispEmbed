#!/usr/bin/env bash
# INTERNAL RESEARCH ONLY (feat/issue71-five-engines): Big-LaMa F16 / Q8_0 / Q4_K
# (crispembed-quantize policy lama-v2) on the delivery-3 core.
#
#   gh workflow run heavy-cpu.yml --ref feat/issue71-five-engines \
#       -f script=tools/ci-heavy/lama_quant_research.sh -f args=fixtures -f isolated_lane=lama-quant-fixtures
#   ... -f args=live -f isolated_lane=lama-quant-live
#
# Both phases first run the unchanged strict protocol
#   tools/ci-heavy/lama_inpaint.py --phase original-native
# (original capture BEFORE conversion, F32 replay under unchanged thresholds; a
# strict failure is recorded, not hidden), then quantize its F32 GGUF.
#   fixtures: every variant x runtime mode replayed through test-lama-diff on the
#             original capture (tools/lama_variant_report.py), all eight fixtures.
#   live:     burned-in subtitles on real photos (tools/inpaint_live_eval.py), each
#             variant x mode, plus the original PyTorch generator on the same inputs.
# Runtime modes: storage (default: kernels dequantized to F32 at load), matmul
# (CRISPEMBED_LAMA_QUANT_MATMUL=1: patch-row kernels run as ggml_mul_mat on their
# stored type), graph / graph-matmul (the same on the persistent ggml graph,
# CRISPEMBED_INPAINT_GRAPH=1, CPU here); f32-stored is the F32 file through the stored-weight ops
# (CRISPEMBED_LAMA_STORED_OPS=1), an A/B control against the shared Engine.
# Checkpoint rights are unresolved: no checkpoint or GGUF leaves scratch.
set -uo pipefail
PHASE=${1:-fixtures}
OUT=${HEAVY_OUT:?}
S=${HEAVY_SCRATCH:?}
THREADS=4
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONHASHSEED=0
failed=0
step() { # name, command...
    local name=$1; shift
    echo "::group::$name"
    local started=$SECONDS
    if "$@" > "$OUT/$name.log" 2>&1; then echo "$name=pass ($((SECONDS - started))s)" >> "$OUT/results.txt"; else
        echo "$name=fail($?) ($((SECONDS - started))s)" >> "$OUT/results.txt"; failed=1; fi
    tail -20 "$OUT/$name.log"
    echo "::endgroup::"
}
info() { # recorded, never fails the lane (the strict F32 protocol may be known-red)
    local name=$1; shift
    local started=$SECONDS
    if "$@" > "$OUT/$name.log" 2>&1; then echo "$name=pass ($((SECONDS - started))s)" >> "$OUT/results.txt"; else
        echo "$name=fail($?) [informational] ($((SECONDS - started))s)" >> "$OUT/results.txt"; fi
    tail -20 "$OUT/$name.log"
}
echo "source=$(git rev-parse HEAD) phase=$PHASE" > "$OUT/provenance.txt"

step deps bash -c "python -m pip install -q torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu &&
    python -m pip install -q numpy==2.1.3 Pillow==11.3.0 omegaconf==2.3.0 kornia==0.8.1 pytorch-lightning==2.5.3 \
        torchmetrics==1.8.2 opencv-python-headless==4.12.0.88 PyYAML==6.0.2"
A=$OUT/acceptance
info strict-f32 python tools/ci-heavy/lama_inpaint.py --phase original-native --cases "$([ "$PHASE" = fixtures ] && echo full-panel || echo subtitle_bar)" \
    --output "$A" --scratch "$S/acc" --deadline-minutes 170
B=$S/acc/build
F32=$S/acc/big-lama-f32.gguf
[ -f "$F32" ] || { echo "no F32 conversion; stopping" >> "$OUT/results.txt"; exit 1; }
# The strict driver keeps its native stage dumps; they are the F32 comparison base.
step build-quantize cmake --build "$B" --target crispembed-quantize -j4
Q=$(find "$B" -name crispembed-quantize -type f | head -1)
DIFF=$(find "$B" -name test-lama-diff -type f | head -1)
CLI=$(find "$B" -name crispembed -type f | head -1)
M=$S/models; mkdir -p "$M"; cp "$F32" "$M/big-lama-f32.gguf"
# Recipes: name|requested type|--tensor-type overrides. q4_k-s8e8 holds every global
# spectral-branch kernel and the encoder downsampling convs (model.2-4) at Q8_0; plain
# q4_k fails the full-mask fixture (lanes 41292697462, 41295796193).
RECIPES=("f16|f16|" "q8_0|q8_0|" "q4_k|q4_k|"
         "q4_k-s8e8|q4_k|generator.*.convg2g.*=q8_0 generator.model.2.*=q8_0 generator.model.3.*=q8_0 generator.model.4.*=q8_0")
for recipe in "${RECIPES[@]}"; do
    IFS='|' read -r q type overrides <<< "$recipe"
    set -f; qa=""; for o in $overrides; do qa+=" --tensor-type '$o'"; done; set +f
    step "quantize-$q" bash -c "'$Q' '$M/big-lama-f32.gguf' '$M/big-lama-$q.gguf' $type $qa > '$OUT/quantize-$q.stdout' &&
        python tools/lama_quant_census.py --source '$M/big-lama-f32.gguf' --variant '$M/big-lama-$q.gguf' \
            --requested $type $qa --output '$OUT/census-$q.json'"
done
sha256sum "$M"/*.gguf | sed "s#$M/##" > "$OUT/gguf-sha256.txt"
ls -l "$M"/*.gguf | awk '{print $5, $NF}' | sed "s#$M/##" > "$OUT/gguf-bytes.txt"
sha() { awk -v f="big-lama-$1.gguf" '$2==f{print $1}' "$OUT/gguf-sha256.txt"; }
# "opt" is delivery 5's optimized CPU graph (tests/results/inpainting/lama-cpu-2026-11-01.json):
# persistent graph + CPU FFT + reflect gather + GEMM layout, F32 weights by default.
OPT="CRISPEMBED_INPAINT_GRAPH=1 CRISPEMBED_INPAINT_CPU_FFT=1 CRISPEMBED_INPAINT_REFLECT_GATHER=1 CRISPEMBED_INPAINT_GEMM_LAYOUT=1"
mode_env() {
    case $1 in
        matmul) echo CRISPEMBED_LAMA_QUANT_MATMUL=1 ;;
        stored) echo CRISPEMBED_LAMA_STORED_OPS=1 ;;
        graph) echo CRISPEMBED_INPAINT_GRAPH=1 ;;
        graph-matmul) echo CRISPEMBED_INPAINT_GRAPH=1 CRISPEMBED_LAMA_QUANT_MATMUL=1 ;;
        opt) echo $OPT ;;
        opt-matmul) echo $OPT CRISPEMBED_LAMA_QUANT_MATMUL=1 ;;
    esac
}
# Delivery 5 established F32 eager/portable/optimized; earlier lanes established the
# StoredOps and portable-graph matrices. Here: the optimized graph only.
CONFIGS=(f32/opt)
for q in f16 q8_0 q4_k q4_k-s8e8; do for mode in opt opt-matmul; do CONFIGS+=("$q/$mode"); done; done

if [ "$PHASE" = fixtures ]; then
    for config in "${CONFIGS[@]}"; do
        q=${config%/*}; mode=${config#*/}; label=$q-$mode
        envs=(); for kv in $(mode_env "$mode"); do envs+=(--env "$kv"); done
        step "fixtures-$label" python tools/lama_variant_report.py --binary "$DIFF" --model "$M/big-lama-$q.gguf" \
            --model-sha256 "$(sha "$q")" --reference-dir "$A/original" --f32-native-dir "$A" \
            --output-dir "$S/fixtures-$label" --label "$label" "${envs[@]}"
        mkdir -p "$OUT/fixtures/$label"
        cp "$S/fixtures-$label"/report.json "$S/fixtures-$label"/*.log "$OUT/fixtures/$label/" 2>/dev/null
        tar -czf "$OUT/fixtures/$label/stages.tgz" -C "$S/fixtures-$label" $(cd "$S/fixtures-$label" && ls *.stages.json 2>/dev/null)
        for d in "$S/fixtures-$label"/native-*; do
            [ -f "$d/native.rgb8" ] && cp "$d/native.rgb8" "$OUT/fixtures/$label/$(basename "$d").rgb8"
        done
    done
    # Drop the bulky per-stage F32 dumps from the strict F32 evidence; keep reports, logs and RGB.
    find "$A" -name '*.f32' -delete
fi

if [ "$PHASE" = live ]; then
    mkdir -p "$S/photos-src" "$S/photos"
    cp tests/regression/images/fox.png tests/regression/images/face.png "$S/photos-src/"
    python - "$S/photos-src" "$S/photos" <<'PY' > "$OUT/photos.json"
import hashlib, json, sys, urllib.request
from pathlib import Path
from PIL import Image
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
urls = {"landscape.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a8/Tour_Eiffel_Wikimedia_Commons.jpg/960px-Tour_Eiffel_Wikimedia_Commons.jpg",
        "street.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/New_york_times_square-terabass.jpg/1280px-New_york_times_square-terabass.jpg"}
rec = {}
for name, url in urls.items():
    try:
        data = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "CrispEmbed-research"}), timeout=60).read()
        (src / name).write_bytes(data); rec[name] = {"url": url, "sha256": hashlib.sha256(data).hexdigest()}
    except Exception as e:
        rec[name] = {"url": url, "error": str(e)}
# The delivery-3 LaMa profile caps each side at 1024; larger photos are downscaled
# (Lanczos) before subtitles are rendered, and saved losslessly as PNG.
for p in sorted(src.iterdir()):
    image = Image.open(p).convert("RGB")
    scale = min(1.0, 1024 / max(image.size))
    if scale < 1:
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    image.save(dst / (p.stem + ".png"))
    rec.setdefault(p.name, {})
    rec[p.name].update({"sha256": rec[p.name].get("sha256") or hashlib.sha256(p.read_bytes()).hexdigest(),
                        "evaluated_size": list(image.size)})
print(json.dumps(rec, indent=2))
PY
    LIVE=()
    for config in "${CONFIGS[@]}"; do
        q=${config%/*}; mode=${config#*/}; label=$q-$mode
        step "live-$label" env $(mode_env "$mode") python tools/inpaint_live_eval.py --binary "$CLI" --backend lama \
            --model "$M/big-lama-$q.gguf" --model-sha256 "$(sha "$q")" --photos "$S"/photos/*.png \
            --output-dir "$OUT/live/$label" --threads $THREADS
        [ "$label" != f32-opt ] && LIVE+=(--variant "$label=$OUT/live/$label")
        # Peak RSS / wall for one photo (street) through the production CLI.
        /usr/bin/time -v env CRISPEMBED_LAMA=1 $(mode_env "$mode") "$CLI" --inpaint-engine lama \
            --inpaint-model "$M/big-lama-$q.gguf" --inpaint-model-sha256 "$(sha "$q")" \
            --inpaint "$OUT/live/f32-opt/street.input.png" --mask "$OUT/live/f32-opt/street.mask.png" \
            --inpaint-output "$S/rss-$label.png" --threads $THREADS > /dev/null 2> "$OUT/rss-$label.txt" ||
            echo "rss-$label=fail" >> "$OUT/results.txt"
    done
    SRC=$(find "$S/acc" -maxdepth 1 -name original-source -type d)
    ZIP=$S/acc/big-lama.zip
    step live-original python tools/lama_original_photos.py --source-dir "$SRC" --checkpoint-archive "$ZIP" \
        --eval-dir "$OUT/live/f32-opt" --output-dir "$OUT/live/original" --threads $THREADS
    step live-compare python tools/lama_live_compare.py --original-dir "$OUT/live/original" \
        --f32-dir "$OUT/live/f32-opt" "${LIVE[@]}" --output "$OUT/live-comparison.json"
    find "$OUT/live" -mindepth 2 -path '*/f32-opt/*' -prune -o \
        \( -name '*.clean.png' -o -name '*.input.png' -o -name '*.mask.png' \) -print | xargs rm -f
    find "$A" -name '*.f32' -delete
fi

{ echo "### Big-LaMa lama-v2 quantization research ($PHASE, $(git rev-parse --short HEAD)) — internal only"
  echo '```'; cat "$OUT/results.txt"; echo; cat "$OUT/gguf-bytes.txt"; echo '```'; } > "$OUT/summary.md"
exit $failed
