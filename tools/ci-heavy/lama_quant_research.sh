#!/usr/bin/env bash
# INTERNAL RESEARCH ONLY (branch feat/issue71-lama-research): Big-LaMa F16 / Q8_0 /
# Q4_K variants from the F32 conversion, runtime modes, original/F32 parity and live
# ground-truth evaluation.
#
#   gh workflow run heavy-cpu.yml --ref feat/issue71-lama-research \
#       -f script=tools/ci-heavy/lama_quant_research.sh -f isolated_lane=lama-quant
#
# Checkpoint rights are unresolved: the checkpoint and every GGUF stay in scratch and
# are never uploaded as artifacts. Only reports, manifests, logs and small images are.
#
# Runtime modes per model (all default-off gates; storage is the default path):
#   storage  every non-F32 kernel dequantized to F32 at load (unchanged F32 graph)
#   matmul   CRISPEMBED_LAMA_QUANT_MATMUL=1: 1x1 kernels run as ggml_mul_mat on the stored type
#   lowmem   matmul + CRISPEMBED_LAMA_F16_CONV=1: k>1 conv kernels held as F16
set -uo pipefail
OUT=${HEAVY_OUT:?}
S=${HEAVY_SCRATCH:?}
ZIP_URL=https://huggingface.co/smartywu/big-lama/resolve/05cb2be7f8dbe6ca7c6e78f4fc827a4b2baaa4a9/big-lama.zip
ZIP_SHA=f1b358ca24093b93a106183b98a3dea6e8ed09f3b43ea7251eb2c81e7b4575f6
LAMA_SRC=786f5936b27fb3dacd2b1ad799e4de968ea697e7
THREADS=4
export CRISPEMBED_LAMA=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
failed=0
step() { # name, command...
    local name=$1; shift
    echo "::group::$name"
    local started=$SECONDS
    if "$@" > "$OUT/$name.log" 2>&1; then echo "$name=pass ($((SECONDS - started))s)" >> "$OUT/results.txt"; else
        echo "$name=fail($?) ($((SECONDS - started))s)" >> "$OUT/results.txt"; failed=1; fi
    tail -25 "$OUT/$name.log"
    echo "::endgroup::"
}
info() { # name, command... : recorded, never fails the lane (strict F32 budget is known-red)
    local name=$1; shift
    if "$@" > "$OUT/$name.log" 2>&1; then echo "$name=pass" >> "$OUT/results.txt"; else
        echo "$name=fail($?) [informational]" >> "$OUT/results.txt"; fi
    tail -8 "$OUT/$name.log"
}
echo "source=$(git rev-parse HEAD)" > "$OUT/provenance.txt"
B=$S/build
M=$S/models
mkdir -p "$M"

step build bash -c "cmake -S . -B '$B' -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DCRISPEMBED_NATIVE=OFF \
    -DBUILD_SHARED_LIBS=OFF -DGGML_BLAS=OFF -DGGML_METAL=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_OPENMP=OFF \
    -DGGML_LLAMAFILE=OFF -DCRISPEMBED_BUILD_SHARED=OFF >/dev/null && cmake --build '$B' --parallel 4 --target \
    crispembed-cli crispembed-quantize test-lama-fft test-lama test-lama-native"
step native-units bash -c "'$B/test-lama-fft' && '$B/test-lama'"
step py-units bash -c "python tests/test_lama_converter.py -v && python tests/test_lama_reference.py -v"

step lama-deps bash -c "python -m pip install -q numpy==2.2.6 pillow==11.3.0 PyYAML==6.0.2 &&
    python -m pip install -q torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu &&
    git clone -q https://github.com/advimman/lama.git '$S/upstream-lama' && git -C '$S/upstream-lama' checkout -q $LAMA_SRC"
step lama-fetch bash -c "curl -sfL -o '$S/big-lama.zip' '$ZIP_URL' && echo '$ZIP_SHA  $S/big-lama.zip' | sha256sum -c - &&
    cd '$M' && unzip -q -j ../big-lama.zip '*best.ckpt' '*config.yaml' && rm ../big-lama.zip &&
    sha256sum best.ckpt config.yaml | tee '$OUT/lama-checkpoint-sha256.txt'"
CK=$(awk '/best.ckpt/{print $1}' "$OUT/lama-checkpoint-sha256.txt" 2>/dev/null)
F32=$M/big-lama-f32.gguf
step lama-convert python models/convert-lama-to-gguf.py --checkpoint "$M/best.ckpt" --checkpoint-sha256 "$CK" \
    --config "$M/config.yaml" --output "$F32"
cp "$M/big-lama-f32.json" "$OUT/" 2>/dev/null

# ---- quantize + policy census --------------------------------------------------
for q in f16 q8_0 q4_k; do
    step "quantize-$q" bash -c "'$B/crispembed-quantize' '$F32' '$M/big-lama-$q.gguf' $q > '$OUT/quantize-$q.stdout' &&
        python tools/lama_quant_census.py --source '$F32' --variant '$M/big-lama-$q.gguf' --requested $q \
            --output '$OUT/census-$q.json'"
done
sha256sum "$M"/*.gguf | sed "s#$M/##" > "$OUT/gguf-sha256.txt"
ls -l "$M"/*.gguf | awk '{print $5, $NF}' | sed "s#$M/##" > "$OUT/gguf-bytes.txt"

# ---- original-module fixtures, native F32 (strict protocol, known-red), variants ---
step lama-reference python tools/dump_lama_reference.py --source-dir "$S/upstream-lama" --checkpoint "$M/best.ckpt" \
    --checkpoint-sha256 "$CK" --config "$M/config.yaml" --converted-model "$F32" \
    --output "$OUT/lama-original" --threads $THREADS
info f32-strict python tools/run_lama_native.py --binary "$B/test-lama-native" --model "$F32" \
    --original-dir "$OUT/lama-original" --output-dir "$OUT/native-f32" --threads $THREADS
mode_env() {
    case $1 in
        storage) ;;
        matmul) echo "CRISPEMBED_LAMA_QUANT_MATMUL=1" ;;
        lowmem) echo "CRISPEMBED_LAMA_QUANT_MATMUL=1 CRISPEMBED_LAMA_F16_CONV=1" ;;
    esac
}
CONFIGS=()
for q in f32 f16 q8_0 q4_k; do
    for mode in storage matmul lowmem; do
        [ "$q/$mode" = f32/storage ] && continue
        CONFIGS+=("$q/$mode")
    done
done
for config in "${CONFIGS[@]}"; do
    q=${config%/*}; mode=${config#*/}; label=$q-$mode
    envs=(); for kv in $(mode_env "$mode"); do envs+=(--env "$kv"); done
    step "fixtures-$label" python tools/compare_lama_variant.py --binary "$B/test-lama-native" \
        --model "$M/big-lama-$q.gguf" --original-dir "$OUT/lama-original" --f32-native-dir "$OUT/native-f32" \
        --output-dir "$S/fixtures-$label" --label "$label" --threads $THREADS "${envs[@]}"
    mkdir -p "$OUT/fixtures/$label"
    cp "$S/fixtures-$label/report.json" "$OUT/fixtures/$label/" 2>/dev/null
    cp "$S"/fixtures-"$label"/*.log "$OUT/fixtures/$label/" 2>/dev/null
    gzip -c "$S/fixtures-$label/stages.json" > "$OUT/fixtures/$label/stages.json.gz" 2>/dev/null
done

# ---- live: burned-in subtitles on real photos (exact mask), all configs ---------
mkdir -p "$S/photos"
cp tests/regression/images/fox.png tests/regression/images/face.png "$S/photos/"
python - "$S/photos" <<'PY' > "$OUT/photos.json"
import hashlib, json, sys, urllib.request
from pathlib import Path
d = Path(sys.argv[1])
urls = {"landscape.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a8/Tour_Eiffel_Wikimedia_Commons.jpg/960px-Tour_Eiffel_Wikimedia_Commons.jpg",
        "street.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/New_york_times_square-terabass.jpg/1280px-New_york_times_square-terabass.jpg"}
rec = {}
for name, url in urls.items():
    try:
        data = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "CrispEmbed-research"}), timeout=60).read()
        (d / name).write_bytes(data); rec[name] = {"url": url, "sha256": hashlib.sha256(data).hexdigest()}
    except Exception as e:
        rec[name] = {"url": url, "error": str(e)}
for p in sorted(d.iterdir()):
    rec.setdefault(p.name, {"sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
print(json.dumps(rec, indent=2))
PY
LIVE=()
for config in f32/storage "${CONFIGS[@]}"; do
    q=${config%/*}; mode=${config#*/}; label=$q-$mode
    step "live-$label" env $(mode_env "$mode") python tools/inpaint_live_eval.py --binary "$B/crispembed" \
        --backend lama --model "$M/big-lama-$q.gguf" --photos "$S"/photos/* --output-dir "$OUT/live/$label" \
        --threads $THREADS
    [ "$label" != f32-storage ] && LIVE+=(--variant "$label=$OUT/live/$label")
    # Peak RSS / wall for one full-size photo (street, 1280x853) through the production CLI.
    /usr/bin/time -v env $(mode_env "$mode") "$B/crispembed" --inpaint-model "$M/big-lama-$q.gguf" \
        --inpaint-backend lama --inpaint "$OUT/live/f32-storage/street.input.png" \
        --mask "$OUT/live/f32-storage/street.mask.png" --inpaint-output "$S/rss-$label.png" --threads $THREADS \
        > /dev/null 2> "$OUT/rss-$label.txt" || echo "rss-$label=fail" >> "$OUT/results.txt"
done
step live-original python tools/lama_original_photos.py --source-dir "$S/upstream-lama" --checkpoint "$M/best.ckpt" \
    --checkpoint-sha256 "$CK" --config "$M/config.yaml" --converted-model "$F32" \
    --eval-dir "$OUT/live/f32-storage" --output-dir "$OUT/live/original" --threads $THREADS
step live-compare python tools/lama_live_compare.py --original-dir "$OUT/live/original" \
    --f32-dir "$OUT/live/f32-storage" "${LIVE[@]}" --output "$OUT/live-comparison.json"
# Keep reports and the small composed outputs; drop the duplicated clean/input/mask copies.
find "$OUT/live" -mindepth 2 -path '*/f32-storage/*' -prune -o \( -name '*.clean.png' -o -name '*.input.png' -o -name '*.mask.png' \) -print | xargs rm -f

{ echo "### Big-LaMa quantization research lane ($(git rev-parse --short HEAD)) — internal only"; echo '```'
  cat "$OUT/results.txt"; echo; cat "$OUT/gguf-bytes.txt"; echo '```'; } > "$OUT/summary.md"
exit $failed
