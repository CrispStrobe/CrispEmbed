#!/usr/bin/env bash
# Native Marker-style pipeline on hosted Linux/CPU (docs/marker-validation.md panels).
#
#   gh workflow run heavy-cpu.yml --ref <branch> -f script=tools/ci-heavy/marker_qualify.sh \
#       -f isolated_lane=marker
#
# PDFium comes from the pypdfium2 5.10.1 wheel the delivery paired (PDFium
# 151.0.7891.0); TeleOCR F16/Q8_0 are the published files at their pinned Hub
# commit, whose F16 bytes are the delivery's expected model. Runs the four native
# test binaries, the Python/CLI/HTTP integration, the compiled Rust and Dart
# consumers, and the neural panels (synthetic F16 and Q8_0, full F16, force-pdf F16).
# Functional and exact-content checks only; no Surya parity is claimed.
set -uo pipefail
OUT=${HEAVY_OUT:?}
S=${HEAVY_SCRATCH:?}
ROOT=$(pwd)
HUB=https://huggingface.co/cstr/TeleOCR-GGUF/resolve/1de0119f27f470cfc7ef0be14327f5953730c7e8
declare -A SHA=([f16]=6c52a2800e313964717cf4e5d06e0392c2b5936d8adeac423d563b8a026a2780
    [q8_0]=9de7e45ce6c13ba03d3d2879f2a1bff9f8407e12808be588cf057cd2553f224b)
DART_VERSION=3.13.7
export CRISPEMBED_MARKER=1 CRISPEMBED_TELEOCR=1 CRISPEMBED_ACCEPT_QWEN_RESEARCH_LICENSE=1 \
    CRISPEMBED_TELEOCR_GPU=0 QWEN2VL_OCR_FORCE_CPU=1 DART_SUPPRESS_ANALYTICS=1

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

echo "::group::build, PDFium and toolchains"
cmake -S . -B "$S/build" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DCRISPEMBED_NATIVE=OFF -DGGML_CUDA=OFF \
    -DCRISPEMBED_BUILD_SHARED=ON >/dev/null &&
    cmake --build "$S/build" -j4 --target crispembed-shared crispembed-cli crispembed-server crispembed-marker \
        test-marker-document test-marker-backend test-marker-api test-marker-save > "$S/build.log" 2>&1 ||
    { tail -60 "$S/build.log"; exit 4; }
B=$S/build
LIB=$(find "$B" -name 'libcrispembed.so' | head -1)
LIBDIR=$(dirname "$LIB")
bin() { find "$B" -type f -name "$1" -perm -u+x | head -1; }
export LD_LIBRARY_PATH=$LIBDIR:$B/ggml/src:${LD_LIBRARY_PATH:-} CRISPEMBED_SYS_LIB_DIR=$LIBDIR
python -m pip install -q numpy pillow pypdfium2==5.10.1 || exit 3
PDFIUM=$(python -c "import pypdfium2_raw, pathlib; print(next(pathlib.Path(pypdfium2_raw.__file__).parent.glob('libpdfium*')))")
sha256sum "$PDFIUM" | tee "$OUT/pdfium-sha256.txt"
curl -sfL -o "$S/dart.zip" \
    "https://storage.googleapis.com/dart-archive/channels/stable/release/$DART_VERSION/sdk/dartsdk-linux-x64-release.zip" &&
    unzip -q "$S/dart.zip" -d "$S" || exit 3
DART=$S/dart-sdk/bin/dart
{ cargo --version; "$DART" --version; } 2>&1 | tee "$OUT/toolchains.txt"
for v in f16 q8_0; do
    curl -sfL -o "$S/teleocr-$v.gguf" "$HUB/teleocr-$v.gguf" && echo "${SHA[$v]}  $S/teleocr-$v.gguf" | sha256sum -c - || exit 5
done
echo "::endgroup::"

PDF=tests/fixtures/marker/simpledigital.pdf
step native-document "$(bin test-marker-document)"
step native-backend "$(bin test-marker-backend)" "$PDFIUM"
step native-api "$(bin test-marker-api)" "$PDFIUM" "$PDF"
step native-save "$(bin test-marker-save)"
step integration python tests/marker_integration.py --library "$LIB" --pdfium "$PDFIUM" \
    --cli "$(bin crispembed-marker)" --server "$(bin crispembed-server)" --out "$OUT/integration"

mkdir -p "$S/rust/src" && cp tests/marker_public_bindings.rs "$S/rust/src/main.rs"
printf '[package]\nname = "marker-probe"\nversion = "0.1.0"\nedition = "2021"\n[dependencies]\ncrispembed = { path = "%s/crispembed" }\n' \
    "$ROOT" > "$S/rust/Cargo.toml"
step rust-build cargo build -q --release --manifest-path "$S/rust/Cargo.toml"
step rust "$S/rust/target/release/marker-probe" "$PDFIUM" "$PDF"
mkdir -p "$S/dart/bin" "$S/dart/lib" && cp flutter/crispembed/lib/src/*.dart "$S/dart/lib/" &&
    cp tests/marker_public_bindings.dart "$S/dart/bin/main.dart"
printf 'name: marker_binding_test\nenvironment:\n  sdk: ">=3.0.0 <4.0.0"\ndependencies:\n  ffi: 2.2.0\n' > "$S/dart/pubspec.yaml"
step dart-build bash -c "cd '$S/dart' && '$DART' pub get >/dev/null && '$DART' analyze bin lib && '$DART' compile exe bin/main.dart -o probe"
step dart "$S/dart/probe" "$LIB" "$PDFIUM" "$PDF"

# Without the TeleOCR license acknowledgement the OCR modes must refuse.
step license-gate env -u CRISPEMBED_ACCEPT_QWEN_RESEARCH_LICENSE bash -c \
    "! python tests/marker_native.py --library '$LIB' --pdfium '$PDFIUM' --model '$S/teleocr-f16.gguf' \
       --expected-model-sha256 ${SHA[f16]} --panel synthetic --threads 2 --out '$S/gate-probe'"

for run in synthetic:f16 synthetic:q8_0 full:f16 force-pdf:f16; do
    panel=${run%%:*}; v=${run##*:}
    step "neural-$panel-$v" python tests/marker_native.py --library "$LIB" --pdfium "$PDFIUM" \
        --model "$S/teleocr-$v.gguf" --expected-model-sha256 "${SHA[$v]}" --panel "$panel" --threads 2 \
        --out "$OUT/neural-$panel-$v"
done

{ echo "### Marker native pipeline ($(git rev-parse --short HEAD))"; echo '```'; cat "$OUT/results.txt"; echo '```'; } \
    > "$OUT/summary.md"
exit $failed
