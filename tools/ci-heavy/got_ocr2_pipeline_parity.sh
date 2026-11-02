#!/usr/bin/env bash
# GOT-OCR2 pipeline parity against the ORIGINAL stepfun-ai/GOT-OCR2_0 chat() pipeline.
#
#   1. Original remote code (transformers==4.37.2 venv, CPU F32): exact pixel
#      tensor, prompt IDs, generation config and greedy IDs (no_repeat_ngram 20,
#      plus an ngram-off control) for the frozen cases.
#   2. HF port preprocessing capture (prompt IDs, pixels, generation config).
#   3. Native CPU arms, one factor at a time, every factor forced explicitly:
#        R = CRISPEMBED_GOT_OCR_PIL_BICUBIC  (resize)
#        S = CRISPEMBED_GOT_OCR_ORIG_SYSTEM  (prompt)
#        N = CRISPEMBED_GOT_OCR_ORIG_NGRAM   (no-repeat-ngram 20 over prompt+generated)
#      on got-ocr2-f16-r2 (Hub c0f1aa7b) and got-ocr2-q4_k.
#   4. got_ocr2_parity_compare.py -> parity.json + table.
#
#   gh workflow run heavy-cpu.yml --ref fix/got-ocr2-pipeline-parity \
#       -f script=tools/ci-heavy/got_ocr2_pipeline_parity.sh -f isolated_lane=got-parity \
#       -f pip="torch transformers>=4.49 accelerate pillow numpy safetensors huggingface_hub"
#   args: [max_new_tokens] [f16 arm list] [q4 arm list]   (arms are RSN bit strings)
set -uo pipefail
OUT=${HEAVY_OUT:?}
S=${HEAVY_SCRATCH:?}
export CRISPEMBED_ACCEPT_TONGYI_QIANWEN_LICENSE=1
MAXNEW=${1:-1024}
F16_ARMS=${2:-000,100,010,001,110,101,011,111}
Q4_ARMS=${3:-000,111}
REPO=cstr/got-ocr2-crispembed-GGUF
REV=c0f1aa7b5b52c15002e911d5d630e6d0dcb3ba8f
I=tests/regression/images
CASES=(fox=$I/fox.png receipt_example=$I/cc0/receipt_example.png simple_form=$I/cc0/simple_form.png)
fail=0
note() { echo "$*" | tee -a "$OUT/results.txt"; }
echo "source=$(git rev-parse HEAD) max_new=$MAXNEW f16_arms=$F16_ARMS q4_arms=$Q4_ARMS" > "$OUT/provenance.txt"

# 1. original pipeline in its own venv (remote code is pinned to transformers 4.37.2)
python -m venv "$S/venv-orig"
VP="$S/venv-orig/bin/python"
"$VP" -m pip install -q --upgrade pip
"$VP" -m pip install -q torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
"$VP" -m pip install -q "transformers==4.37.2" "tokenizers<0.16" tiktoken accelerate pillow "numpy<2" safetensors \
    > "$OUT/venv-orig.log" 2>&1 || { note "venv-orig install FAILED"; tail -20 "$OUT/venv-orig.log"; exit 1; }
"$VP" -m pip freeze | grep -i -E "^(torch|torchvision|transformers|tiktoken|pillow|numpy)=" >> "$OUT/provenance.txt"
"$VP" -I tools/ci-heavy/got_ocr2_orig_probe.py "$OUT" "$MAXNEW" 20,0 "${CASES[@]}" > "$OUT/orig.log" 2>&1 \
    || { note "original probe FAILED"; tail -40 "$OUT/orig.log"; exit 1; }
grep -E "ngram=" "$OUT/orig.log" | tee -a "$OUT/results.txt"

# 2. HF port preprocessing (main env, recent transformers)
python -c "import transformers; print('hf-port env transformers', transformers.__version__)" >> "$OUT/provenance.txt"
python -I tools/ci-heavy/got_ocr2_hf_preproc.py "$OUT" "${CASES[@]}" > "$OUT/hf_preproc.log" 2>&1 \
    || { note "hf preproc FAILED (diagnostic)"; tail -20 "$OUT/hf_preproc.log"; }

# 3. native
B=$S/build-cpu
cmake -S . -B "$B" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DCRISPEMBED_NATIVE=OFF -DGGML_METAL=OFF \
    -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DCRISPEMBED_BUILD_SHARED=OFF > "$OUT/configure.log" 2>&1 || exit 1
cmake --build "$B" --parallel 4 --target crispembed-cli > "$OUT/build.log" 2>&1 || { tail -40 "$OUT/build.log"; exit 1; }
BIN="$B/crispembed"
M=$S/models; mkdir -p "$M"
for f in got-ocr2-f16-r2.gguf got-ocr2-q4_k.gguf; do
    [ -s "$M/$f" ] || curl -sfL --retry 3 -o "$M/$f" "https://huggingface.co/$REPO/resolve/$REV/$f" || { note "download $f FAILED"; exit 1; }
done
sha256sum "$M"/*.gguf | sed "s|$M/||" > "$OUT/model-sha256.txt"
mkdir -p "$OUT/native"
run_arm() {  # model_tag file arm
    local tag=$1 f=$2 arm=$3
    for c in "${CASES[@]}"; do
        local id=${c%%=*} img=${c#*=}
        local d="$OUT/native/$tag.$arm.$id"; mkdir -p "$d"
        local t0=$EPOCHREALTIME
        CRISPEMBED_GOT_OCR_PIL_BICUBIC=${arm:0:1} CRISPEMBED_GOT_OCR_ORIG_SYSTEM=${arm:1:1} \
        CRISPEMBED_GOT_OCR_ORIG_NGRAM=${arm:2:1} CRISPEMBED_GOT_OCR_DUMP_DIR="$d" \
            "$BIN" -t 4 -m "$f" --ocr "$img" --ocr-max-tokens "$MAXNEW" > "$d/text.txt" 2> "$d/stderr.txt"
        local rc=$?
        python -c "print(round($EPOCHREALTIME-$t0,2))" > "$d/secs.txt"
        [ $rc = 0 ] || { note "native $tag.$arm.$id rc=$rc"; fail=1; }
    done
}
IFS=, read -ra fa <<< "$F16_ARMS"; for a in "${fa[@]}"; do run_arm f16 "$M/got-ocr2-f16-r2.gguf" "$a"; done
IFS=, read -ra qa <<< "$Q4_ARMS";  for a in "${qa[@]}"; do run_arm q4k "$M/got-ocr2-q4_k.gguf" "$a"; done
# pixels are 12 MiB per arm/case: keep the comparison, not the raw dumps
python -I tools/ci-heavy/got_ocr2_parity_compare.py "$OUT" \
    tests/results/got-ocr2-f16/qualify-41391040980-rejected/compare.json | tee -a "$OUT/results.txt" || fail=1
find "$OUT/native" -name pixels.f32 -delete
python - "$OUT" <<'EOF'
import os, sys, numpy as np, hashlib, json
out = sys.argv[1]; rec = {}
for f in sorted(os.listdir(out)):
    if f.endswith(".pixels.npy"):
        a = np.load(os.path.join(out, f)); rec[f] = hashlib.sha256(a.astype(np.float32).tobytes()).hexdigest()
        os.remove(os.path.join(out, f))
json.dump(rec, open(os.path.join(out, "pixels-sha256.json"), "w"), indent=1)
EOF
{ echo "### GOT-OCR2 pipeline parity ($(git rev-parse --short HEAD))"; echo '```'; cat "$OUT/results.txt"; echo '```'; } > "$OUT/summary.md"
exit $fail
