#!/usr/bin/env bash
# INTERNAL RESEARCH ONLY (branch feat/issue71-lama-research): MI-GAN and Big-LaMa
# inpainting unit tests, model parity and live ground-truth evaluation.
#
#   gh workflow run heavy-cpu.yml --ref feat/issue71-lama-research \
#       -f script=tools/ci-heavy/lama_research.sh -f isolated_lane=lama-research
#
# Big-LaMa's checkpoint rights are unresolved (docs/inpainting-feasibility.md): the
# checkpoint is fetched from the third-party archive the upstream README links, used
# only inside this job, and never published, registered or uploaded as an artifact.
set -uo pipefail
OUT=${HEAVY_OUT:?}
S=${HEAVY_SCRATCH:?}
ROOT=$(pwd)
ZIP_URL=https://huggingface.co/smartywu/big-lama/resolve/05cb2be7f8dbe6ca7c6e78f4fc827a4b2baaa4a9/big-lama.zip
ZIP_SHA=f1b358ca24093b93a106183b98a3dea6e8ed09f3b43ea7251eb2c81e7b4575f6
LAMA_SRC=786f5936b27fb3dacd2b1ad799e4de968ea697e7
DBNET_URL=https://huggingface.co/cstr/dbnet-ic15-GGUF/resolve/main/dbnet-ic15-q8_0.gguf
DBNET_SHA=163c3b3d1211fd47b4ea323e2e684287bdc75ae68f0ffea397ccfcdc29a74de2
export CRISPEMBED_MIGAN=1 CRISPEMBED_LAMA=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
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
B=$S/build

# ---- unit tests (model-free) ------------------------------------------------
step py-units bash -c "python tests/test_prepare_migan_inpaint.py -v && python tests/test_inpaint_png.py -v &&
    python tests/test_migan_reference_contract.py -v && python tests/test_migan_native_driver.py -v &&
    python tests/test_lama_converter.py -v && python tests/test_lama_reference.py -v"
step build bash -c "cmake -S . -B '$B' -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DCRISPEMBED_NATIVE=OFF \
    -DBUILD_SHARED_LIBS=OFF -DGGML_BLAS=OFF -DGGML_METAL=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_OPENMP=OFF \
    -DGGML_LLAMAFILE=OFF -DCRISPEMBED_BUILD_SHARED=OFF >/dev/null && cmake --build '$B' --parallel 4 --target \
    crispembed-cli test-inpaint test-inpaint-cli-driver test-migan-diff test-lama-fft test-lama test-lama-native"
step native-units bash -c "'$B/test-inpaint' && '$B/test-migan-diff' --contracts && '$B/test-migan-diff' --gate-controls &&
    '$B/test-migan-diff' --observer-retention && '$B/test-lama-fft' && '$B/test-lama'"
step cli-units python tests/test_inpaint_cli.py --binary "$B/test-inpaint-cli-driver" --real-binary "$B/crispembed" -v

# ---- MI-GAN real model -------------------------------------------------------
step migan-prepare python tools/prepare_migan_inpaint.py --download --output-dir "$S/migan"
MIGAN=$S/migan/migan-256-places2-f32-experimental.gguf
step migan-smoke python tests/test_inpaint_model.py --binary "$B/crispembed" --model "$MIGAN" \
    --output-dir "$OUT/migan-smoke"

# ---- Big-LaMa: exact checkpoint bytes, conversion, original reference, native parity
step lama-deps bash -c "python -m pip install -q numpy==2.2.6 pillow==11.3.0 PyYAML==6.0.2 &&
    python -m pip install -q torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu &&
    git clone -q https://github.com/advimman/lama.git '$S/upstream-lama' && git -C '$S/upstream-lama' checkout -q $LAMA_SRC"
step lama-fetch bash -c "curl -sfL -o '$S/big-lama.zip' '$ZIP_URL' && echo '$ZIP_SHA  $S/big-lama.zip' | sha256sum -c - &&
    mkdir -p '$S/lama-input' && cd '$S/lama-input' && unzip -q -j ../big-lama.zip '*best.ckpt' '*config.yaml' &&
    sha256sum best.ckpt config.yaml | tee '$OUT/lama-checkpoint-sha256.txt'"
CK=$(awk '/best.ckpt/{print $1}' "$OUT/lama-checkpoint-sha256.txt" 2>/dev/null)
step lama-convert python models/convert-lama-to-gguf.py --checkpoint "$S/lama-input/best.ckpt" --checkpoint-sha256 "$CK" \
    --config "$S/lama-input/config.yaml" --output "$S/lama-input/big-lama-f32.gguf"
cp "$S/lama-input/big-lama-f32.json" "$OUT/" 2>/dev/null
step lama-reference python tools/dump_lama_reference.py --source-dir "$S/upstream-lama" --checkpoint "$S/lama-input/best.ckpt" \
    --checkpoint-sha256 "$CK" --config "$S/lama-input/config.yaml" --converted-model "$S/lama-input/big-lama-f32.gguf" \
    --output "$OUT/lama-original" --threads 4
step lama-native python tools/run_lama_native.py --binary "$B/test-lama-native" --model "$S/lama-input/big-lama-f32.gguf" \
    --original-dir "$OUT/lama-original" --output-dir "$OUT/lama-native" --threads 4
step lama-cli python tests/test_inpaint_model.py --binary "$B/crispembed" --model "$S/lama-input/big-lama-f32.gguf" \
    --backend lama --reference "$OUT/lama-original" --output-dir "$OUT/lama-cli"

# ---- live: burned-in subtitles on real photos, scored against the clean originals
mkdir -p "$S/photos"
cp tests/regression/images/fox.png tests/regression/images/face.png "$S/photos/"
python - "$S/photos" <<'PY' > "$OUT/photos.json"
import hashlib, json, sys, urllib.request
from pathlib import Path
d = Path(sys.argv[1])
# Public-domain / CC0 Wikimedia Commons photographs (1280px renditions), hash recorded.
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
curl -sfL -o "$S/dbnet.gguf" "$DBNET_URL" && echo "$DBNET_SHA  $S/dbnet.gguf" | sha256sum -c - >/dev/null || rm -f "$S/dbnet.gguf"
TM=(); [ -f "$S/dbnet.gguf" ] && TM=(--text-model "$S/dbnet.gguf")
step live-migan python tools/inpaint_live_eval.py --binary "$B/crispembed" --backend migan --model "$MIGAN" \
    --photos "$S"/photos/* --output-dir "$OUT/live-migan" "${TM[@]}"
step live-lama python tools/inpaint_live_eval.py --binary "$B/crispembed" --backend lama \
    --model "$S/lama-input/big-lama-f32.gguf" --photos "$S"/photos/* --output-dir "$OUT/live-lama" "${TM[@]}"

{ echo "### Inpainting research lane ($(git rev-parse --short HEAD)) — internal only"; echo '```'
  cat "$OUT/results.txt"; echo '```'; } > "$OUT/summary.md"
exit $failed
