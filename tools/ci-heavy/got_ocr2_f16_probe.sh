#!/usr/bin/env bash
# GOT-OCR2 F16 artifact investigation (fix/got-ocr2-f16).
#   1. Reproduce: native CPU decode of the published F16 and Q4_K files on fox.
#   2. Ground truth: original model (HF transformers port, F32) decode plus the
#      per-module activation magnitudes (tools/ci-heavy/got_ocr2_hf_probe.py).
#   3. Bisect the F16 file by precision: variants rebuilt from the same F16
#      bytes with selected tensor groups widened to F32 (or quantized to Q8_0),
#      to separate a data defect from an F16-activation runtime defect.
#   4. Stage diff (test-got-ocr-diff) of each arm against got-ocr2-ref-full.gguf.
#
#   gh workflow run heavy-cpu.yml --ref fix/got-ocr2-f16 \
#       -f script=tools/ci-heavy/got_ocr2_f16_probe.sh -f isolated_lane=got-f16 \
#       -f pip="torch transformers>=4.49 accelerate pillow gguf numpy"
set -uo pipefail
OUT=${HEAVY_OUT:?}
S=${HEAVY_SCRATCH:?}
export CRISPEMBED_ACCEPT_TONGYI_QIANWEN_LICENSE=1
MODE=${1:-all}
echo "source=$(git rev-parse HEAD)" > "$OUT/provenance.txt"
I=tests/regression/images
B=$S/build-cpu
cmake -S . -B "$B" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DCRISPEMBED_NATIVE=OFF -DGGML_METAL=OFF \
    -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DCRISPEMBED_BUILD_SHARED=OFF > "$OUT/configure.log" 2>&1 || exit 1
cmake --build "$B" --parallel 4 --target crispembed-cli test-got-ocr-diff crispembed-quantize \
    > "$OUT/build.log" 2>&1 || { tail -40 "$OUT/build.log"; exit 1; }
BIN="$B/crispembed"
DIFF=$(find "$B" -name test-got-ocr-diff -type f | head -1)
QZ=$(find "$B" -name crispembed-quantize -type f | head -1)
M=$S/models; mkdir -p "$M"
REV=${GOT_REV:-5a0b0a1adc}
get() { [ -s "$M/$1" ] || curl -sfL --retry 3 -o "$M/$1" "https://huggingface.co/cstr/got-ocr2-crispembed-GGUF/resolve/$REV/$1"; }
get got-ocr2-f16.gguf; get got-ocr2-q4_k.gguf; get got-ocr2-ref-full.gguf
sha256sum "$M"/*.gguf > "$OUT/model-sha256.txt"

# Inf/NaN/range scan of the published F16 file.
python - "$M/got-ocr2-f16.gguf" > "$OUT/f16-tensor-scan.txt" 2>&1 <<'EOF'
import sys, numpy as np, gguf
r = gguf.GGUFReader(sys.argv[1])
bad = 0
for t in r.tensors:
    a = np.asarray(t.data)
    if a.dtype.kind != 'f':
        continue
    a = a.astype(np.float32)
    nf = int((~np.isfinite(a)).sum()); mx = float(np.nanmax(np.abs(a[np.isfinite(a)]))) if a.size else 0
    sub = int(((a != 0) & (np.abs(a) < 6.1e-5)).sum()) if t.tensor_type == gguf.GGMLQuantizationType.F16 else 0
    if nf or mx > 1000:
        bad += 1
    print(f"{t.name:45s} {t.tensor_type.name:5s} max={mx:.4g} nonfinite={nf} f16_subnormal={sub}")
print("tensors_flagged", bad)
EOF
tail -1 "$OUT/f16-tensor-scan.txt"

run_ocr() { # tag model
    local t0=$EPOCHREALTIME
    "$BIN" -t 4 -m "$2" --ocr "$I/fox.png" --ocr-max-tokens 128 > "$OUT/ocr.$1.txt" 2> "$OUT/ocr.$1.err"
    local rc=$?
    echo "ocr.$1 rc=$rc secs=$(python -c "print(round($EPOCHREALTIME-$t0,2))") text=$(head -c 160 "$OUT/ocr.$1.txt" | tr '\n' ' ')" \
        | tee -a "$OUT/results.txt"
}
run_diff() { # tag model
    "$DIFF" "$2" "$M/got-ocr2-ref-full.gguf" > "$OUT/diff.$1.out" 2> "$OUT/diff.$1.err"
    echo "diff.$1 rc=$?" >> "$OUT/results.txt"
    grep -E "cos_min" "$OUT/diff.$1.err" | sed "s/^/  diff.$1 /" >> "$OUT/results.txt"
}

run_ocr f16 "$M/got-ocr2-f16.gguf"
run_ocr q4_k "$M/got-ocr2-q4_k.gguf"
run_diff f16 "$M/got-ocr2-f16.gguf"
run_diff q4_k "$M/got-ocr2-q4_k.gguf"

# Precision bisection from the same F16 bytes.
variant() { # tag target [--tensor-type ...]
    local tag=$1; shift
    "$QZ" "$M/got-ocr2-f16.gguf" "$S/got-$tag.gguf" "$@" > "$OUT/quantize.$tag.log" 2>&1 || {
        echo "variant.$tag quantize FAILED" >> "$OUT/results.txt"; tail -5 "$OUT/quantize.$tag.log"; return; }
    run_ocr "$tag" "$S/got-$tag.gguf"
    run_diff "$tag" "$S/got-$tag.gguf"
    rm -f "$S/got-$tag.gguf"
}
variant q8_0-from-f16 q8_0
variant llm-f32 f16 --tensor-type 'l.*=f32'
variant vis-f32 f16 --tensor-type 'v.*=f32'
variant ffn-down-f32 f16 --tensor-type 'l.blk.*.ffn_down.weight=f32'

# Original model ground truth + activation magnitudes.
if [ "$MODE" = all ]; then
    python -I tools/ci-heavy/got_ocr2_hf_probe.py "$I/fox.png" "$OUT/hf-probe.fox.json" 128 \
        > "$OUT/hf-probe.fox.log" 2>&1
    echo "hf-probe rc=$?" >> "$OUT/results.txt"
    head -40 "$OUT/hf-probe.fox.log" | sed 's/^/  hf /' >> "$OUT/results.txt"
fi
{ echo "### GOT-OCR2 F16 probe ($(git rev-parse --short HEAD))"; echo '```'; cat "$OUT/results.txt"; echo '```'; } > "$OUT/summary.md"
