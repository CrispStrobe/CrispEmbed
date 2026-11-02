#!/usr/bin/env bash
# GPU audit (feat/metal-audit) CPU-only controls that need no Metal device:
#   1. DBNet deconv form: CRISPEMBED_DBNET_DECONV_MATMUL=1 vs the default
#      conv_transpose_2d on the CPU backend (DBNet + Tesseract, both CPU).
#   2. ESRGAN / InstructIR default CPU output is unchanged by the gated code
#      (gate unset vs gate=0 must be pixel-identical).
#
#   gh workflow run heavy-cpu.yml --ref feat/metal-audit \
#       -f script=tools/ci-heavy/gpu_audit_cpu_controls.sh -f isolated_lane=gpu-audit
set -uo pipefail
OUT=${HEAVY_OUT:?}
S=${HEAVY_SCRATCH:?}
failed=0
echo "source=$(git rev-parse HEAD)" > "$OUT/provenance.txt"
B=$S/build-cpu
cmake -S . -B "$B" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DCRISPEMBED_NATIVE=OFF -DGGML_METAL=OFF \
    -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DCRISPEMBED_BUILD_SHARED=OFF > "$OUT/configure.log" 2>&1 || exit 1
cmake --build "$B" --parallel 4 --target crispembed-cli > "$OUT/build.log" 2>&1 || { tail -40 "$OUT/build.log"; exit 1; }
M=$S/models; mkdir -p "$M"
get() { [ -s "$M/$2" ] || curl -sfL --retry 3 -o "$M/$2" "https://huggingface.co/$1/resolve/main/$2"; }
get cstr/dbnet-ic15-GGUF dbnet-ic15-q8_0.gguf
get cstr/tesseract-lstm-GGUF tesseract-eng-q8_0.gguf || true
get cstr/esrgan-sr-GGUF esrgan-x4-f32.gguf
get cstr/instructir-GGUF instructir-f16.gguf
I=tests/regression/images
BIN="$B/crispembed"
check() { # name file-a file-b
    if cmp -s "$2" "$3"; then echo "$1=identical" >> "$OUT/results.txt"; else
        echo "$1=DIFFERS" >> "$OUT/results.txt"; failed=1; diff <(head -c 4000 "$2") <(head -c 4000 "$3") | head -20; fi
}
if [ -s "$M/tesseract-eng-q8_0.gguf" ]; then
  for img in fox scan_page_pd scan_strip; do
    for arm in 0 1; do
      CRISPEMBED_DBNET_DECONV_MATMUL=$arm CRISPEMBED_OCR_DETECT_BENCH=1 "$BIN" -t 4 --json -m "$M/tesseract-eng-q8_0.gguf" \
        --ocr-det "$M/dbnet-ic15-q8_0.gguf" --ocr-rec "$M/tesseract-eng-q8_0.gguf" --ocr "$I/$img.png" \
        > "$OUT/dbnet-deconv$arm.$img.json" 2> "$OUT/dbnet-deconv$arm.$img.err"
    done
    check "dbnet-deconv-matmul.$img" "$OUT/dbnet-deconv0.$img.json" "$OUT/dbnet-deconv1.$img.json"
  done
else
  echo "dbnet-deconv=SKIPPED(no tesseract model)" >> "$OUT/results.txt"
fi
for arm in unset 0; do
  if [ $arm = unset ]; then E=(); else E=(CRISPEMBED_ESRGAN_GPU=0 CRISPEMBED_INSTRUCTIR_GPU=0); fi
  env "${E[@]}" "$BIN" -t 4 --esrgan-model "$M/esrgan-x4-f32.gguf" --esrgan-sr "$I/scan_strip.png" > "$OUT/esrgan.$arm.png" 2>/dev/null
  env "${E[@]}" "$BIN" -t 4 --instructir-model "$M/instructir-f16.gguf" --instructir "$I/scan_strip.png" > "$OUT/instructir.$arm.png" 2>/dev/null
done
check esrgan-default-unchanged "$OUT/esrgan.unset.png" "$OUT/esrgan.0.png"
check instructir-default-unchanged "$OUT/instructir.unset.png" "$OUT/instructir.0.png"
{ echo "### GPU-audit CPU controls ($(git rev-parse --short HEAD))"; echo '```'; cat "$OUT/results.txt"; echo '```'; } > "$OUT/summary.md"
exit $failed
