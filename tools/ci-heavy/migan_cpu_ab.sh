#!/usr/bin/env bash
# MI-GAN CPU lane of the feat/metal-migan A/B (docs/migan-metal.md). Metal runs stay on the M1.
#
#   gh workflow run heavy-cpu.yml --ref feat/metal-migan -f script=tools/ci-heavy/migan_cpu_ab.sh \
#       -f isolated_lane=migan-cpu-ab
#
# 1. Reproduce the reviewed F32 GGUF, derive F16/Q8_0/Q4_K (migan-pointwise-v1).
# 2. test-migan-gpu-ab on the eight repository-authored frozen fixtures: every
#    observed stage vs CPU/F32 under the strict 0.002/0.99999 gates (recorded
#    separately), decoded image gates, output cosine vs F32 (owner bar > 0.94).
# 3. 1920x1080 synthetic subtitle frames on CPU (warm frame time, tiles/s).
# 4. Delivery 5 (feat/issue71-five-engines-graphs 67406abe) test-migan-graph on
#    the same runner: persistent graph with and without CRISPEMBED_MIGAN_CPU_KERNELS.
set -uo pipefail
OUT=${HEAVY_OUT:?}
S=${HEAVY_SCRATCH:?}
THREADS=${MIGAN_AB_THREADS:-4}
failed=0
step() { local name=$1; shift; echo "::group::$name"
    if "$@" > "$OUT/$name.log" 2>&1; then echo "$name=pass" >> "$OUT/results.txt"; else
        echo "$name=fail($?)" >> "$OUT/results.txt"; failed=1; fi
    tail -30 "$OUT/$name.log"; echo "::endgroup::"; }
echo "source=$(git rev-parse HEAD) runner=$(uname -m) cpus=$(nproc)" > "$OUT/provenance.txt"
B=$S/build-cpu
step configure cmake -S . -B "$B" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DCRISPEMBED_NATIVE=OFF \
    -DBUILD_SHARED_LIBS=OFF -DGGML_BLAS=OFF -DGGML_METAL=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_OPENMP=OFF \
    -DGGML_LLAMAFILE=OFF -DCRISPEMBED_BUILD_SHARED=OFF
step build cmake --build "$B" --parallel 4 --target test-migan-gpu-ab crispembed-quantize test-inpaint test-migan-diff
step native-controls bash -c "'$B/test-inpaint' && '$B/test-migan-diff' --contracts && '$B/test-migan-diff' --gate-controls && '$B/test-migan-diff' --observer-retention"
M=$S/migan-inpaint-model
step prepare-model python tools/prepare_migan_inpaint.py --download --output-dir "$M"
F32=$M/migan-256-places2-f32-experimental.gguf
for t in f16 q8_0 q4_k; do
    step "quantize-$t" "$B/crispembed-quantize" "$F32" "$M/migan-256-places2-$t-experimental.gguf" "$t"
done
sha256sum "$M"/*.gguf > "$OUT/models.sha256"
FX=$S/fixtures; mkdir -p "$FX"
step fixtures python - "$FX" <<'PY'
import sys; sys.path.insert(0, "tools")
from pathlib import Path
import migan_reference_contract as c
out = Path(sys.argv[1])
for case in c.CASES:
    rgb, mask = c.fixture(case)
    (out / f"{case}-input.raw").write_bytes(rgb); (out / f"{case}-fill-mask.raw").write_bytes(mask)
    print(case, c.fixture_identity(case))
PY
step device-ab python tools/migan_device_ab.py --binary "$B/test-migan-gpu-ab" --f32 "$F32" --fixtures "$FX" \
    --candidate "cpu:f32:$F32" --candidate "cpu:f16:$M/migan-256-places2-f16-experimental.gguf" \
    --candidate "cpu:q8_0:$M/migan-256-places2-q8_0-experimental.gguf" \
    --candidate "cpu:q4_k:$M/migan-256-places2-q4_k-experimental.gguf" \
    --threads "$THREADS" --iters 3 --frame-iters 2 --output-dir "$OUT/device-ab"
# Delivery 5 graph on the same runner.
D5=$S/d5
step d5-checkout bash -c "git fetch --depth 1 origin 67406abe28764288aa05e2cc6519c6fd3a324e3b 2>/dev/null || git fetch origin feat/issue71-five-engines-graphs; \
    git worktree add --detach '$D5' 67406abe && cp -a ggml/. '$D5/ggml/'"
step d5-configure cmake -S "$D5" -B "$D5/build" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DCRISPEMBED_NATIVE=OFF \
    -DBUILD_SHARED_LIBS=OFF -DGGML_BLAS=OFF -DGGML_METAL=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_OPENMP=OFF \
    -DGGML_LLAMAFILE=OFF -DCRISPEMBED_BUILD_SHARED=OFF
step d5-build cmake --build "$D5/build" --parallel 4 --target test-migan-graph
SHA=8b82b2e82fc8e5a2e1f06827594aac1a5d66a9ca41e24199ddd969847788097f
for arm in plain kernels; do
    extra=(); [ "$arm" = kernels ] && extra=(--cpu-kernels)
    step "d5-graph-$arm" "$D5/build/test-migan-graph" --model "$F32" --checkpoint-sha256 "$SHA" --size 256 \
        --backend cpu --threads "$THREADS" --warm-runs 5 --rgb "$FX/subtitle_bar-input.raw" \
        --mask "$FX/subtitle_bar-fill-mask.raw" "${extra[@]}"
done
{ echo "### MI-GAN CPU A/B ($(git rev-parse --short HEAD))"; echo '```'; cat "$OUT/results.txt"; echo '```'; } > "$OUT/summary.md"
exit $failed
