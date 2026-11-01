#!/usr/bin/env bash
# MI-GAN512 hosted qualification (heavy-cpu.yml; pip: torch numpy pillow huggingface_hub).
#
#   gh workflow run heavy-cpu.yml --ref <branch> -f script=tools/ci-heavy/migan512_qualify.sh \
#     -f pip="torch numpy pillow huggingface_hub" -f isolated_lane=migan512 [-f args="f16 q8_0 q4_k"]
#
# Original first: pinned upstream source files, the README-linked checkpoint
# (first-observed SHA-256 recorded before any torch load), the unchanged
# original Generator(resolution=512) capture of all eight fixtures, then
# lossless F32 conversion and the frozen-gate native comparator. Optional
# precision variants (args) are quantized from the F32 GGUF and scored
# against native F32 and the original; the frozen F32 gates are unchanged.
set -uo pipefail
OUT=${HEAVY_OUT:?}
S=${HEAVY_SCRATCH:?}
QUANTS=("$@")
ROOT=$(pwd)
failed=0
log() { echo "$*" | tee -a "$OUT/summary.md"; }
step() { # name, command...
    local name=$1; shift
    echo "::group::$name"
    local start=$(date +%s)
    if "$@" > "$OUT/$name.log" 2>&1; then rc=0; else rc=$?; failed=1; fi
    echo "$name rc=$rc seconds=$(( $(date +%s) - start ))" >> "$OUT/results.txt"
    tail -40 "$OUT/$name.log"
    echo "::endgroup::"
    return $rc
}
echo "source=$(git rev-parse HEAD)" > "$OUT/provenance.txt"
python - <<'EOF' > "$OUT/sources.sh"
import json
c = json.load(open("docs/migan512-reference-contract.json"))
src = c["original_source"]
for name, rec in src["files"].items():
    print(f'fetch "{name}" {rec["bytes"]} {rec["sha256"]}')
print(f'COMMIT={src["commit"]}')
print(f'CKPT_URL="{c["checkpoint"]["download_url"]}"')
print(f'CKPT_BYTES={c["checkpoint"]["declared_bytes"]}')
EOF
SRC=$S/migan-src
fetch() {
    mkdir -p "$SRC/$(dirname "$1")"
    curl -fsSL "https://raw.githubusercontent.com/Picsart-AI-Research/MI-GAN/$COMMIT/$1" -o "$SRC/$1" || return 1
    [ "$(stat -c %s "$SRC/$1")" = "$2" ] && [ "$(sha256sum "$SRC/$1" | cut -d' ' -f1)" = "$3" ] || { echo "source identity differs: $1"; return 1; }
}
# shellcheck disable=SC1090
eval "$(grep '^COMMIT=\|^CKPT_' "$OUT/sources.sh")"
while read -r _ name bytes sha; do fetch "${name//\"/}" "$bytes" "$sha" || failed=1; done < <(grep '^fetch' "$OUT/sources.sh")
[ $failed = 0 ] || { log "source download failed"; exit 1; }
mkdir -p "$S/ckpt"
CKPT=$S/ckpt/migan_512_places2.pt
curl -fsSL "$CKPT_URL" -o "$CKPT" -D "$OUT/checkpoint-headers.txt" || { log "checkpoint download failed"; exit 1; }
BYTES=$(stat -c %s "$CKPT"); SHA=$(sha256sum "$CKPT" | cut -d' ' -f1)
head -c 4 "$CKPT" | od -An -c > "$OUT/checkpoint-magic.txt"
cat > "$OUT/checkpoint-identity.json" <<EOF
{"schema":"crispembed-migan512-checkpoint-first-observation-v1","filename":"migan_512_places2.pt",
 "bytes":$BYTES,"declared_bytes":$CKPT_BYTES,"sha256":"$SHA","upstream_sha256_declared":false,
 "source_commit":"$COMMIT","github_sha":"${GITHUB_SHA:-}","github_run_id":"${GITHUB_RUN_ID:-}"}
EOF
log "checkpoint bytes=$BYTES sha256=$SHA"
[ "$BYTES" = "$CKPT_BYTES" ] || { log "checkpoint byte count differs (HTML/quota page?)"; exit 1; }

export PYTHONPATH=$ROOT/tools
step original python tools/dump_migan512_reference.py --source-dir "$SRC" --checkpoint "$CKPT" \
    --checkpoint-sha256 "$SHA" --output "$S/original512" --threads 4 || { log "original capture failed"; exit 1; }
cp "$S/original512/report.json" "$OUT/original-report.json"
cp "$S/original512/tensor-inventory.json" "$OUT/original-tensor-inventory.json"
GGUF=$S/migan-512-places2-f32-experimental.gguf
step convert python models/convert-migan-to-gguf.py --image-size 512 --source-dir "$SRC" --checkpoint "$CKPT" \
    --checkpoint-sha256 "$SHA" --original-report "$S/original512/report.json" --output "$GGUF" || exit 1
cp "${GGUF%.gguf}.json" "$OUT/conversion-receipt.json"
mkdir -p "$OUT/licenses"; cp "$SRC/LICENSE" "$SRC/LICENSE-WEIGHTS" "$OUT/licenses/"

B=$S/build
step configure cmake -S . -B "$B" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DCRISPEMBED_NATIVE=OFF \
    -DBUILD_SHARED_LIBS=OFF -DGGML_BLAS=OFF -DGGML_METAL=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_OPENMP=OFF \
    -DGGML_LLAMAFILE=OFF -DCRISPEMBED_BUILD_SHARED=OFF || exit 1
step build cmake --build "$B" --parallel 4 --target test-migan512-diff test-inpaint crispembed-cli crispembed-quantize || exit 1
BIN=$B; [ -x "$B/bin/test-migan512-diff" ] && BIN=$B/bin
step native-controls bash -c "'$BIN/test-migan512-diff' --contracts && '$BIN/test-migan512-diff' --gate-controls && '$BIN/test-migan512-diff' --observer-retention"

export CRISPEMBED_MIGAN=1 CRISPEMBED_MIGAN512=1
declare -A FILES=([f32]=$GGUF)
ARMS=(f32)
for q in "${QUANTS[@]}"; do
    FILES[$q]=$S/migan-512-places2-$q.gguf
    if step "quantize-$q" "$BIN/crispembed-quantize" "$GGUF" "${FILES[$q]}" "$q"; then
        # Two arms per file: dequantize-at-load (default) and the gated
        # stored-type 1x1 matmul.
        FILES[$q-mm]=${FILES[$q]}
        ARMS+=("$q" "$q-mm")
    fi
done
CASES="subtitle_bar edge_touch empty full center one_pixel checkerboard thin_strip"
python - "$S/original512/subtitle_bar" "$S/cli" <<'EOF'
import sys
from pathlib import Path
from PIL import Image
src, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(exist_ok=True)
Image.frombytes("RGB", (512, 512), (src / "input.rgb8").read_bytes()).save(out / "input.png")
Image.frombytes("L", (512, 512), (src / "fill-mask.u8").read_bytes()).save(out / "mask.png")
EOF
for v in "${ARMS[@]}"; do
    f=${FILES[$v]}; fsha=$(sha256sum "$f" | cut -d' ' -f1)
    echo "{\"variant\":\"$v\",\"bytes\":$(stat -c %s "$f"),\"sha256\":\"$fsha\"}" >> "$OUT/variants.jsonl"
    env_extra=(CRISPEMBED_ARM="$v")
    case $v in f32) ;; *-mm) env_extra+=(CRISPEMBED_INPAINT_QUANT=1 CRISPEMBED_INPAINT_QUANT_MATMUL=1) ;; *) env_extra+=(CRISPEMBED_INPAINT_QUANT=1) ;; esac
    for c in $CASES; do
        R=$S/original512/$c
        # The frozen comparator: F32 gates are unchanged for every variant.
        env "${env_extra[@]}" /usr/bin/time -v "$BIN/test-migan512-diff" "$f" "$R/reference.gguf" "$SHA" \
            "$R/input.rgb8" "$R/fill-mask.u8" "$R/composited.rgb8" "$S/native-$v/$c" \
            > "$OUT/native-$v-$c.log" 2>&1
        rc=$?
        grep -E "MIGAN512_(SUMMARY|IMAGE)" "$OUT/native-$v-$c.log" | sed "s/^/$v $c rc=$rc /" >> "$OUT/native-summary.txt"
        grep -E "Maximum resident|Elapsed" "$OUT/native-$v-$c.log" | sed "s/^/$v $c /" >> "$OUT/native-resources.txt"
    done
    # Real CLI on the subtitle fixture, 512 engine, every arm.
    step "cli-$v" env "${env_extra[@]}" /usr/bin/time -v "$BIN/crispembed" --inpaint-engine migan512 --inpaint-model "$f" \
        --inpaint-model-sha256 "$fsha" --inpaint-checkpoint-sha256 "$SHA" --inpaint "$S/cli/input.png" \
        --mask "$S/cli/mask.png" --inpaint-output "$OUT/cli-$v-subtitle_bar.png" --threads 4
done
# Live ground truth: burned-in subtitles on real photos, exact and DBNet masks,
# both as 512 centre crops (fixed512 profile) and full photos (512 tiles).
bash tools/ci-heavy/inpaint_live_inputs.sh "$S/photos" "$S/dbnet.gguf" "$OUT/photos.json"
TM=(); [ -f "$S/dbnet.gguf" ] && TM=(--text-model "$S/dbnet.gguf")
for geometry in crop512 full; do
    GEO=(); [ $geometry = crop512 ] && GEO=(--crop 512)
    for v in "${ARMS[@]}"; do
        f=${FILES[$v]}; fsha=$(sha256sum "$f" | cut -d' ' -f1)
        LE=(); case $v in f32) ;; *-mm) LE=(--env CRISPEMBED_INPAINT_QUANT=1 --env CRISPEMBED_INPAINT_QUANT_MATMUL=1) ;; *) LE=(--env CRISPEMBED_INPAINT_QUANT=1) ;; esac
        REF=(); [ "$v" = f32 ] || REF=(--reference-run "$OUT/live-$geometry/f32")
        step "live-$geometry-$v" python tools/inpaint_live_eval.py --binary "$BIN/crispembed" --engine migan512 \
            --model "$f" --model-sha256 "$fsha" --checkpoint-sha256 "$SHA" --label "$v" "${GEO[@]}" "${LE[@]}" \
            "${REF[@]}" --photos "$S"/photos/* --output-dir "$OUT/live-$geometry/$v" "${TM[@]}" --threads 4
    done
done
python - "$S" "$OUT" "${ARMS[@]}" <<'EOF'
import json, sys
from pathlib import Path
S, OUT, variants = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
cases = {}
for c in "subtitle_bar edge_touch empty full center one_pixel checkerboard thin_strip".split():
    R = S / "original512" / c
    v = {name: {"float": str(S / f"native-{name}" / c / "controls" / "ordinary-output.f32"),
                "rgb": str(S / f"native-{name}" / c / "composited.rgb8")} for name in variants
         if (S / f"native-{name}" / c / "controls" / "ordinary-output.f32").exists()}
    cases[c] = {"original_float": str(R / "generator.output.npy"), "original_rgb": str(R / "composited.rgb8"),
                "mask": str(R / "fill-mask.u8"), "input": str(R / "input.rgb8"), "variants": v}
(OUT / "score-spec.json").write_text(json.dumps({"cases": cases}, indent=2))
EOF
step score python tools/inpaint_precision_score.py table "$OUT/score-spec.json" "$OUT/precision-score.json"
mkdir -p "$OUT/images"
for v in "${ARMS[@]}"; do cp "$S/native-$v/subtitle_bar/composited.rgb8" "$OUT/images/$v-subtitle_bar.rgb8" 2>/dev/null; done
cp "$S/original512/subtitle_bar/composited.png" "$OUT/images/original-subtitle_bar.png" 2>/dev/null
# GGUF files are retained as a run artifact for operator-side private
# preservation (the repository HF_TOKEN secret cannot write that repo).
mkdir -p "$OUT/gguf"
for q in f32 "${QUANTS[@]}"; do [ -f "${FILES[$q]:-}" ] && cp "${FILES[$q]}" "$OUT/gguf/"; done
cat "$OUT/native-summary.txt" >> "$OUT/summary.md" 2>/dev/null
exit $failed
