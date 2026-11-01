#!/usr/bin/env bash
# Moebius / PixelHacker native CPU replay against the Kaggle CUDA original capture.
#
#   gh workflow run heavy-cpu.yml --ref <branch> -f script=tools/ci-heavy/latent_native.sh \
#     -f pip="kaggle numpy pillow" -f isolated_lane=latent-<model>-<arms> \
#     -f args="<model> f32 q8_0 q8_0-mm ..."
#
# The original capture (tools/kaggle/latent-inpaint-capture) is fetched from
# that kernel's output with the repository Kaggle secret and verified part by
# part. Pinned checkpoints are converted here (F32, lossless), precision
# variants are quantized from them (denoiser and VAE at the same type), and
# every arm runs the unchanged frozen-gate comparator test-latent-inpaint.
# An "-mm" arm is the same file with CRISPEMBED_INPAINT_QUANT_MATMUL=1; a
# trailing "-g" runs the persistent ggml graph path (CRISPEMBED_INPAINT_GRAPH=1)
# instead of the eager CPU path, e.g. f32 f32-g q8_0-mm q8_0-mm-g.
set -uo pipefail
OUT=${HEAVY_OUT:?}
S=${HEAVY_SCRATCH:?}
MODEL=${1:?model}
shift
# A literal "live" argument adds the real-photo ground-truth stage; "nonative"
# skips the synthetic replay (e.g. a live-only rerun).
ARMS=(); LIVE=0; NATIVE=1
for a in "$@"; do case $a in live) LIVE=1 ;; nonative) NATIVE=0 ;; *) ARMS+=("$a") ;; esac; done
[ ${#ARMS[@]} -gt 0 ] || ARMS=(f32)
GATE=CRISPEMBED_$(echo "$MODEL" | tr a-z A-Z)
log() { echo "$*" | tee -a "$OUT/summary.md"; }
step() {
    local name=$1; shift
    echo "::group::$name"
    local start=$(date +%s)
    if "$@" > "$OUT/$name.log" 2>&1; then rc=0; else rc=$?; fi
    echo "$name rc=$rc seconds=$(( $(date +%s) - start ))" | tee -a "$OUT/results.txt"
    tail -30 "$OUT/$name.log"
    echo "::endgroup::"
    return $rc
}
echo "source=$(git rev-parse HEAD) model=$MODEL arms=${ARMS[*]}" > "$OUT/provenance.txt"
nproc >> "$OUT/provenance.txt"; free -g >> "$OUT/provenance.txt"; df -h "$S" >> "$OUT/provenance.txt"

: "${KAGGLE_API_TOKEN:?repository Kaggle secret unavailable}" "${KAGGLE_ACCOUNT:?}"
step fetch-capture python -m kaggle kernels output "$KAGGLE_ACCOUNT/crispembed-latent-capture-$MODEL" -p "$S/kout" || exit 1
cp "$S/kout/capture-manifest.json" "$S/kout/original-report.json" "$OUT/" 2>/dev/null
step verify-capture python tools/ci-heavy/latent_fetch.py capture "$MODEL" "$S/kout" "$S/original" || exit 1
rm -rf "$S/kout"
step sources bash -c "python tools/ci-heavy/latent_fetch.py sources '$MODEL' '$S/src/$MODEL' && \
    { [ '$MODEL' = pixelhacker ] || python tools/ci-heavy/latent_fetch.py sources pixelhacker '$S/src/pixelhacker'; }" || exit 1
step checkpoints bash -c "python tools/ci-heavy/latent_fetch.py checkpoint '$MODEL' '$S/ckpt/$MODEL.bin' && \
    python tools/ci-heavy/latent_fetch.py checkpoint sdxl_vae '$S/ckpt/sdxl_vae.bin'" || exit 1
mkdir -p "$S/gguf" "$OUT/conversions"
D=$S/gguf/$MODEL-f32.gguf; V=$S/gguf/sdxl-vae-f32.gguf
step convert-denoiser python models/convert-latent-inpaint-to-gguf.py --model "$MODEL" --source-dir "$S/src/$MODEL" \
    --checkpoint "$S/ckpt/$MODEL.bin" --output "$D" || exit 1
step convert-vae python models/convert-latent-inpaint-to-gguf.py --model sdxl_vae --source-dir "$S/src/pixelhacker" \
    --checkpoint "$S/ckpt/sdxl_vae.bin" --output "$V" || exit 1
rm -f "$S/ckpt/"*.bin
cp "$D.json" "$V.json" "$OUT/conversions/"; cp -r "$D.licenses" "$V.licenses" "$OUT/conversions/"

B=$S/build
step configure cmake -S . -B "$B" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DCRISPEMBED_NATIVE=OFF \
    -DBUILD_SHARED_LIBS=OFF -DGGML_BLAS=OFF -DGGML_METAL=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_OPENMP=OFF \
    -DGGML_LLAMAFILE=OFF -DCRISPEMBED_BUILD_SHARED=OFF || exit 1
step build cmake --build "$B" --parallel 4 --target test-latent-inpaint crispembed-cli crispembed-quantize || exit 1
BIN=$B; [ -x "$B/bin/test-latent-inpaint" ] && BIN=$B/bin

python - "$S/original/subtitle_bar" "$S/cli" <<'EOF'
import sys
from pathlib import Path
from PIL import Image
src, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(exist_ok=True)
Image.frombytes("RGB", (512, 512), (src / "input.rgb").read_bytes()).save(out / "input.png")
Image.frombytes("L", (512, 512), (src / "fill-mask.u8").read_bytes()).save(out / "mask.png")
EOF
failed=0
for arm in "${ARMS[@]}"; do
    base=${arm%-g}; q=${base%-mm}
    env_extra=("$GATE=1")
    [ "$base" = "$arm" ] || env_extra+=(CRISPEMBED_INPAINT_GRAPH=1)
    if [ "$q" = f32 ]; then
        DQ=$D; VQ=$V
    else
        DQ=$S/gguf/$MODEL-$q.gguf; VQ=$S/gguf/sdxl-vae-$q.gguf
        [ -f "$DQ" ] || step "quantize-denoiser-$q" "$BIN/crispembed-quantize" "$D" "$DQ" "$q" || { failed=1; continue; }
        [ -f "$VQ" ] || step "quantize-vae-$q" "$BIN/crispembed-quantize" "$V" "$VQ" "$q" || { failed=1; continue; }
        env_extra+=(CRISPEMBED_INPAINT_QUANT=1)
        [ "$base" = "$q" ] || env_extra+=(CRISPEMBED_INPAINT_QUANT_MATMUL=1)
    fi
    dsha=$(sha256sum "$DQ" | cut -d' ' -f1); vsha=$(sha256sum "$VQ" | cut -d' ' -f1)
    echo "{\"arm\":\"$arm\",\"denoiser_bytes\":$(stat -c %s "$DQ"),\"denoiser_sha256\":\"$dsha\",\"vae_bytes\":$(stat -c %s "$VQ"),\"vae_sha256\":\"$vsha\"}" >> "$OUT/variants.jsonl"
    # Frozen comparator, unchanged budgets, every captured case.
    [ $NATIVE = 1 ] && step "native-$arm" env "${env_extra[@]}" /usr/bin/time -v "$BIN/test-latent-inpaint" --model "$MODEL" \
        --denoiser "$DQ" --vae "$VQ" --model-sha256 "$dsha" --vae-sha256 "$vsha" \
        --reference-dir "$S/original" --out "$S/native-$arm" --threads 4
    mkdir -p "$OUT/native-$arm"
    [ -d "$S/native-$arm" ] && (cd "$S/native-$arm" && find . -name "*.json" -o -name "*.rgb" | cpio -pdm "$OUT/native-$arm" 2>/dev/null)
    # Real CLI, native seed stream, subtitle fixture: latency and peak RSS.
    step "cli-$arm" env "${env_extra[@]}" /usr/bin/time -v "$BIN/crispembed" --inpaint-engine "$MODEL" \
        --inpaint-model "$DQ" --inpaint-model-sha256 "$dsha" --inpaint-vae "$VQ" --inpaint-vae-sha256 "$vsha" \
        --inpaint "$S/cli/input.png" --mask "$S/cli/mask.png" --inpaint-output "$OUT/cli-$arm-subtitle_bar.png" \
        --inpaint-seed 0 --threads 4
    grep -E "Maximum resident|Elapsed" "$OUT/native-$arm.log" "$OUT/cli-$arm.log" >> "$OUT/resources.txt"
done
if [ $LIVE = 1 ]; then
    # Live ground truth: subtitles burned into real photos, 512 centre crops
    # (the strict latent profile), exact and DBNet masks, every arm.
    bash tools/ci-heavy/inpaint_live_inputs.sh "$S/photos" "$S/dbnet.gguf" "$OUT/photos.json"
    TM=(); [ -f "$S/dbnet.gguf" ] && TM=(--text-model "$S/dbnet.gguf")
    for arm in "${ARMS[@]}"; do
        base=${arm%-g}; q=${base%-mm}
        if [ "$q" = f32 ]; then DQ=$D; VQ=$V; else DQ=$S/gguf/$MODEL-$q.gguf; VQ=$S/gguf/sdxl-vae-$q.gguf; fi
        [ -f "$DQ" ] && [ -f "$VQ" ] || continue
        LE=(); [ "$q" = f32 ] || LE=(--env CRISPEMBED_INPAINT_QUANT=1)
        [ "$base" = "$q" ] || LE+=(--env CRISPEMBED_INPAINT_QUANT_MATMUL=1)
        [ "$base" = "$arm" ] || LE+=(--env CRISPEMBED_INPAINT_GRAPH=1)
        REF=(); [ "$arm" = f32 ] || [ ! -d "$OUT/live/f32" ] || REF=(--reference-run "$OUT/live/f32")
        step "live-$arm" python tools/inpaint_live_eval.py --binary "$BIN/crispembed" --engine "$MODEL" \
            --model "$DQ" --model-sha256 "$(sha256sum "$DQ" | cut -d' ' -f1)" --vae "$VQ" \
            --vae-sha256 "$(sha256sum "$VQ" | cut -d' ' -f1)" --seed 0 --crop 512 --label "$arm" "${LE[@]}" \
            "${REF[@]}" --photos "$S"/photos/* --output-dir "$OUT/live/$arm" "${TM[@]}" --threads 4
    done
fi
python - "$OUT" "${ARMS[@]}" <<'EOF' | tee -a "$OUT/summary.md"
import json, sys
from pathlib import Path
out, arms = Path(sys.argv[1]), sys.argv[2:]
for arm in arms:
    path = out / f"native-{arm}" / "report.json"
    if not path.exists():
        print(arm, "no report"); continue
    report = json.loads(path.read_text())
    print(arm, "complete", report.get("complete"), "synthetic_parity_pass", report.get("synthetic_parity_pass"))
    for case in report.get("cases", []):
        print(" ", json.dumps(case)[:600])
EOF
# Keep small receipts; GGUFs are reproducible from the pinned checkpoints.
ls -l "$S/gguf" > "$OUT/gguf-sizes.txt"
for f in "$S"/gguf/*.gguf; do sha256sum "$f"; done > "$OUT/gguf-sha256.txt"
exit $failed
