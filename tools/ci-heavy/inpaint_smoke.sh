#!/usr/bin/env bash
# MI-GAN inpainting CLI: model-free controls plus the real-model smoke on Linux.
#
#   gh workflow run heavy-cpu.yml --ref <branch> -f script=tools/ci-heavy/inpaint_smoke.sh \
#       -f isolated_lane=inpaint
#
# Mirrors .github/workflows/inpaint-cli.yml (its model_smoke job is dispatch-only from
# main): Python contracts, native image/CLI controls, then reproduces the exact
# reviewed 25 MB model from its pinned sources (every hash checked) and runs the
# production CLI on the synthetic smoke set. Smoke only: no subtitle/video quality claim.
set -uo pipefail
OUT=${HEAVY_OUT:?}
S=${HEAVY_SCRATCH:?}
export CRISPEMBED_MIGAN=1
failed=0
step() { # name, command...
    local name=$1; shift
    echo "::group::$name"
    if "$@" > "$OUT/$name.log" 2>&1; then echo "$name=pass" >> "$OUT/results.txt"; else
        echo "$name=fail($?)" >> "$OUT/results.txt"; failed=1; fi
    tail -25 "$OUT/$name.log"
    echo "::endgroup::"
}
echo "source=$(git rev-parse HEAD)" > "$OUT/provenance.txt"
B=$S/build-inpaint
step python-contracts bash -c "python tests/test_prepare_migan_inpaint.py -v && python tests/test_inpaint_png.py -v &&
    python tests/test_migan_reference_contract.py -v && python tests/test_migan_native_driver.py -v"
step configure cmake -S . -B "$B" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DCRISPEMBED_NATIVE=OFF \
    -DBUILD_SHARED_LIBS=OFF -DGGML_BLAS=OFF -DGGML_METAL=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_OPENMP=OFF \
    -DGGML_LLAMAFILE=OFF -DCRISPEMBED_BUILD_SHARED=OFF
step build cmake --build "$B" --parallel 4 --target crispembed-cli test-inpaint test-inpaint-cli-driver test-migan-diff
step native-controls bash -c "'$B/test-inpaint' && '$B/test-migan-diff' --contracts && '$B/test-migan-diff' --gate-controls &&
    '$B/test-migan-diff' --observer-retention"
step cli-controls python tests/test_inpaint_cli.py --binary "$B/test-inpaint-cli-driver" --real-binary "$B/crispembed" -v
step prepare-model python tools/prepare_migan_inpaint.py --download --output-dir "$S/migan-inpaint-model"
cp "$S/migan-inpaint-model/replication.json" "$OUT/" 2>/dev/null
cp -R "$S/migan-inpaint-model/licenses" "$OUT/" 2>/dev/null
step model-smoke python tests/test_inpaint_model.py --binary "$B/crispembed" \
    --model "$S/migan-inpaint-model/migan-256-places2-f32-experimental.gguf" --output-dir "$OUT/inpaint-model-smoke"
{ echo "### MI-GAN inpainting CLI ($(git rev-parse --short HEAD))"; echo '```'; cat "$OUT/results.txt"; echo '```'; } \
    > "$OUT/summary.md"
exit $failed
