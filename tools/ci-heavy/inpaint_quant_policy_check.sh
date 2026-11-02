#!/usr/bin/env bash
# Byte check of the built-in inpaint quantizer policies against the qualified,
# privately preserved GGUFs (issue 71). Read-only on Hugging Face: HF_TOKEN_2
# only downloads from the private repositories; nothing is uploaded.
#
#   1. CRISPEMBED_QUANTIZE_INPAINT_MIX=1 + q4_k on moebius-f32.gguf must
#      reproduce moebius-q4_k-mix.gguf (qualified in run 41328479130 with 630
#      explicit --tensor-type overrides); on sdxl-vae-f32.gguf it must
#      reproduce sdxl-vae-q8_0.gguf (the recipe's VAE half).
#   2. Without the gate, q4_k must still reproduce the plain moebius-q4_k.gguf
#      and q8_0 the plain sdxl-vae-q8_0.gguf (default unchanged).
#   3. MI-GAN512 f16/q8_0/q4_k from its F32 GGUF with the default policy
#      dispatch (fixed512 -> inpaint-v1) must reproduce the preserved variants.
# Every comparison records SHA-256 and the per-tensor type census.
set -uo pipefail
OUT=${HEAVY_OUT:?}
S=${HEAVY_SCRATCH:?}
: "${HF_TOKEN_2:?HF_TOKEN_2 read access to the private inpaint repositories is required}"
log() { echo "$*" | tee -a "$OUT/summary.md"; }
echo "source=$(git rev-parse HEAD)" > "$OUT/provenance.txt"
python -m pip install -q "huggingface_hub>=0.36,<2" gguf numpy || exit 1

B=$S/build
cmake -S . -B "$B" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DCRISPEMBED_NATIVE=OFF -DBUILD_SHARED_LIBS=OFF \
    -DGGML_BLAS=OFF -DGGML_METAL=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_OPENMP=OFF -DGGML_LLAMAFILE=OFF \
    -DCRISPEMBED_BUILD_SHARED=OFF > "$OUT/configure.log" 2>&1 || { tail -40 "$OUT/configure.log"; exit 1; }
cmake --build "$B" --parallel 4 --target crispembed-quantize > "$OUT/build.log" 2>&1 || { tail -40 "$OUT/build.log"; exit 1; }
Q=$B/crispembed-quantize; [ -x "$B/bin/crispembed-quantize" ] && Q=$B/bin/crispembed-quantize

G=$S/gguf; mkdir -p "$G"
fetch() { # REPO FILE -> $G/basename
    HF_TOKEN=$HF_TOKEN_2 python - "$1" "$2" "$G" <<'EOF'
import sys
from huggingface_hub import hf_hub_download
repo, name, dest = sys.argv[1:]
print(hf_hub_download(repo, name, local_dir=dest))
EOF
}
census() { # GGUF -> name<TAB>type lines
    python - "$1" <<'EOF'
import sys
from gguf import GGUFReader
for t in GGUFReader(sys.argv[1]).tensors:
    print(f"{t.name}\t{t.tensor_type.name}")
EOF
}
failed=0
compared=0
compare() { # LABEL PRODUCED EXPECTED
    local label=$1 got=$2 want=$3 a b
    compared=$((compared + 1))
    a=$(sha256sum "$got" | cut -d' ' -f1); b=$(sha256sum "$want" | cut -d' ' -f1)
    census "$got" > "$OUT/$label.produced.types"; census "$want" > "$OUT/$label.expected.types"
    local types=identical
    cmp -s "$OUT/$label.produced.types" "$OUT/$label.expected.types" || types=DIFFER
    local counts; counts=$(cut -f2 "$OUT/$label.produced.types" | sort | uniq -c | tr -s ' ' | tr '\n' ';')
    if [ "$a" = "$b" ]; then
        log "- $label: bytes IDENTICAL sha256=$a, tensor types $types ($counts)"
    else
        log "- $label: bytes DIFFER produced=$a expected=$b, tensor types $types ($counts)"
        failed=1
    fi
    [ "$types" = identical ] || failed=1
}
quant() { # LABEL ENV... -- IN OUT TYPE
    local label=$1; shift
    local envs=()
    while [ "$1" != -- ]; do envs+=("$1"); shift; done; shift
    env "${envs[@]}" "$Q" "$1" "$2" "$3" > "$OUT/$label.quantize.log" 2>&1 || { log "- $label: quantize FAILED"; tail -20 "$OUT/$label.quantize.log"; failed=1; return 1; }
    { grep -E 'mix \(|policy \(' "$OUT/$label.quantize.log" || true; } >> "$OUT/summary.md"
    return 0
}

log "## inpaint quantizer policy byte check"
M=cstr/crispembed-moebius-private
for f in moebius-f32 moebius-q4_k-mix moebius-q4_k sdxl-vae-f32 sdxl-vae-q8_0; do fetch $M gguf/$f.gguf > /dev/null || exit 1; done
quant moebius-mix CRISPEMBED_QUANTIZE_INPAINT_MIX=1 -- "$G/gguf/moebius-f32.gguf" "$S/moebius-mix.gguf" q4_k && \
    compare moebius-q4_k-mix "$S/moebius-mix.gguf" "$G/gguf/moebius-q4_k-mix.gguf"
mix_q8=$(grep -c $'\tQ8_0$' "$OUT/moebius-q4_k-mix.produced.types" 2>/dev/null)
log "  attention tensors at Q8_0 in the produced mix file: $mix_q8"
rm -f "$S/moebius-mix.gguf"
quant moebius-plain CRISPEMBED_QUANTIZE_INPAINT_MIX=0 -- "$G/gguf/moebius-f32.gguf" "$S/moebius-q4_k.gguf" q4_k && \
    compare moebius-q4_k-default "$S/moebius-q4_k.gguf" "$G/gguf/moebius-q4_k.gguf"
rm -f "$S/moebius-q4_k.gguf" "$G/gguf/moebius-"*.gguf
quant vae-mix CRISPEMBED_QUANTIZE_INPAINT_MIX=1 -- "$G/gguf/sdxl-vae-f32.gguf" "$S/vae-mix.gguf" q4_k && \
    compare sdxl-vae-mix "$S/vae-mix.gguf" "$G/gguf/sdxl-vae-q8_0.gguf"
quant vae-q8 CRISPEMBED_QUANTIZE_INPAINT_MIX=0 -- "$G/gguf/sdxl-vae-f32.gguf" "$S/vae-q8.gguf" q8_0 && \
    compare sdxl-vae-q8_0-default "$S/vae-q8.gguf" "$G/gguf/sdxl-vae-q8_0.gguf"
rm -f "$S/vae-"*.gguf "$G/gguf/sdxl-vae-"*.gguf

R=cstr/crispembed-migan512-private
fetch $R migan-512-places2-f32-experimental.gguf > /dev/null || exit 1
for q in f16 q8_0 q4_k; do
    fetch $R migan-512-places2-$q.gguf > /dev/null || exit 1
    quant migan512-$q CRISPEMBED_QUANTIZE_INPAINT_MIX=0 -- "$G/migan-512-places2-f32-experimental.gguf" "$S/migan512-$q.gguf" $q && \
        compare migan512-$q "$S/migan512-$q.gguf" "$G/migan-512-places2-$q.gguf"
done
# Initial run 41388773676 reported PASS without comparing anything (a
# SIGPIPE in the log filter skipped every comparison); require all seven.
[ $compared = 7 ] || { log "only $compared of 7 comparisons ran"; failed=1; }
log "result: $([ $failed = 0 ] && echo PASS || echo FAIL)"
exit $failed
