#!/usr/bin/env bash
# Every public wrapper against real TeleOCR and MinerU2.5-Pro GGUFs on Linux/CPU.
#
#   gh workflow run heavy-cpu.yml --ref <branch> -f script=tools/ci-heavy/ocr_wrappers.sh \
#       -f isolated_lane=ocr-wrappers
#
# TeleOCR (published teleocr-f16 at its pinned Hub commit): tests/test_teleocr_public.py
# (C ABI, Python, orchestrator stage 24, CLI, HTTP, license/feature gates) then the Rust
# and Dart consumers tests/teleocr_public_bindings.{rs,dart} on the C anchor.
# MinerU (converted F16): Python CrispMinerU predict/native_extract/two_step_extract, the
# Rust and Dart consumers tests/mineru25_public_bindings.{rs,dart} on the Python anchor,
# CLI --ocr-pipeline --ocr-engine mineru25-pro, and HTTP /ocr/pipeline.
# Interface consistency only; reference parity is the qualification lanes' job.
set -uo pipefail
OUT=${HEAVY_OUT:?}
S=${HEAVY_SCRATCH:?}
ROOT=$(pwd)
TELE_URL=https://huggingface.co/cstr/TeleOCR-GGUF/resolve/1de0119f27f470cfc7ef0be14327f5953730c7e8/teleocr-f16.gguf
TELE_SHA=6c52a2800e313964717cf4e5d06e0392c2b5936d8adeac423d563b8a026a2780
MODEL=opendatalab/MinerU2.5-Pro-2604-1.2B
REV=d3f5e08d073c21466bbabe21c71bb1e9c2e595da
DART_VERSION=3.13.7
export CRISPEMBED_TELEOCR=1 CRISPEMBED_ACCEPT_QWEN_RESEARCH_LICENSE=1 CRISPEMBED_MINERU25_PRO=1 \
    QWEN2VL_OCR_FORCE_CPU=1 DART_SUPPRESS_ANALYTICS=1

failed=0
step() { # name, command...
    local name=$1; shift
    echo "::group::$name"
    if "$@" > "$OUT/$name.log" 2>&1; then echo "$name=pass" >> "$OUT/results.txt"; else
        echo "$name=fail($?)" >> "$OUT/results.txt"; failed=1; fi
    tail -20 "$OUT/$name.log"
    echo "::endgroup::"
}
echo "source=$(git rev-parse HEAD)" > "$OUT/provenance.txt"

echo "::group::build and toolchains"
cmake -S . -B "$S/build" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DCRISPEMBED_NATIVE=OFF -DGGML_CUDA=OFF \
    -DCRISPEMBED_BUILD_SHARED=ON >/dev/null &&
    cmake --build "$S/build" -j4 --target crispembed-shared crispembed-cli crispembed-server > "$S/build.log" 2>&1 ||
    { tail -60 "$S/build.log"; exit 4; }
B=$S/build
LIB=$(find "$B" -name 'libcrispembed.so' | head -1)
LIBDIR=$(dirname "$LIB")
CLI=$(find "$B" -type f -name crispembed -perm -u+x | head -1)
SERVER=$(find "$B" -type f -name crispembed-server -perm -u+x | head -1)
export LD_LIBRARY_PATH=$LIBDIR:$B/ggml/src:${LD_LIBRARY_PATH:-} CRISPEMBED_SYS_LIB_DIR=$LIBDIR
python -m pip install -q numpy pillow "huggingface_hub>=0.36,<2" -r tools/mineru25-blueprint-requirements.txt \
    -r models/requirements-teleocr.txt || exit 3
curl -sfL -o "$S/dart.zip" \
    "https://storage.googleapis.com/dart-archive/channels/stable/release/$DART_VERSION/sdk/dartsdk-linux-x64-release.zip" &&
    unzip -q "$S/dart.zip" -d "$S" || exit 3
DART=$S/dart-sdk/bin/dart
{ cargo --version; "$DART" --version; } 2>&1 | tee "$OUT/toolchains.txt"
echo "::endgroup::"

probe_rust() { # name source.rs
    mkdir -p "$S/$1/src" && cp "$2" "$S/$1/src/main.rs"
    cat > "$S/$1/Cargo.toml" <<EOF
[package]
name = "$1"
version = "0.1.0"
edition = "2021"
[dependencies]
crispembed = { path = "$ROOT/crispembed" }
serde_json = "1"
EOF
    cargo build -q --release --manifest-path "$S/$1/Cargo.toml"
}
probe_dart() { # name package source.dart
    mkdir -p "$S/$1/bin" "$S/$1/lib" && cp "$ROOT"/flutter/crispembed/lib/src/*.dart "$S/$1/lib/" &&
        cp "$3" "$S/$1/bin/main.dart"
    printf 'name: %s\nenvironment:\n  sdk: ">=3.0.0 <4.0.0"\ndependencies:\n  ffi: 2.2.0\n' "$2" > "$S/$1/pubspec.yaml"
    (cd "$S/$1" && "$DART" pub get >/dev/null && "$DART" analyze bin lib && "$DART" compile exe bin/main.dart -o probe)
}
rgb() { # image out.rgb -> prints "W H"
    python - "$1" "$2" <<'PY'
import sys
from PIL import Image
im = Image.open(sys.argv[1]).convert("RGB"); open(sys.argv[2], "wb").write(im.tobytes()); print(*im.size)
PY
}

echo "::group::models"
curl -sfL -o "$S/teleocr-f16.gguf" "$TELE_URL" && echo "$TELE_SHA  $S/teleocr-f16.gguf" | sha256sum -c - || exit 5
python - "$MODEL" "$REV" "$S/checkpoint" <<'PY' || exit 5
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3])
PY
python models/convert-mineru25-pro-to-gguf.py --model "$S/checkpoint" --dtype f16 --output "$S/mineru25-f16.gguf" > "$S/convert.log" 2>&1 ||
    { tail -20 "$S/convert.log"; exit 5; }
echo "::endgroup::"

# ---- TeleOCR ----------------------------------------------------------------
TIMG=tests/fixtures/teleocr/synthetic-text.png
read -r TW TH < <(rgb "$TIMG" "$S/tele.rgb")
step teleocr-public python tests/test_teleocr_public.py --lib "$LIB" --cli "$CLI" --server "$SERVER" \
    --model "$S/teleocr-f16.gguf" --image "$TIMG" --max-tokens 8 --threads 2 --report "$OUT/teleocr-public.json"
python -c "import json,sys; open(sys.argv[2],'w').write(json.load(open(sys.argv[1]))['anchor_text'])" \
    "$OUT/teleocr-public.json" "$S/tele-anchor.txt" 2>/dev/null
step teleocr-rust-build probe_rust teleocr-public-probe tests/teleocr_public_bindings.rs
step teleocr-rust "$S/teleocr-public-probe/target/release/teleocr-public-probe" "$S/teleocr-f16.gguf" "$S/tele.rgb" \
    "$TW" "$TH" "$TIMG" "$S/tele-anchor.txt" 8
step teleocr-dart-build probe_dart teleocr-dart teleocr_probe tests/teleocr_public_bindings.dart
step teleocr-dart "$S/teleocr-dart/probe" "$S/teleocr-f16.gguf" "$S/tele.rgb" "$TW" "$TH" "$TIMG" "$LIB" \
    "$S/tele-anchor.txt" 8
# License gate: without the acknowledgement the loader must refuse.
step teleocr-license-gate env -u CRISPEMBED_ACCEPT_QWEN_RESEARCH_LICENSE python -c "
import ctypes, sys
lib = ctypes.CDLL(sys.argv[1]); lib.crispembed_ocr_model_init.restype = ctypes.c_void_p
lib.crispembed_ocr_model_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
sys.exit(0 if not lib.crispembed_ocr_model_init(sys.argv[2].encode(), 1) else 1)" "$LIB" "$S/teleocr-f16.gguf"

# ---- MinerU2.5-Pro ----------------------------------------------------------
CROP=tests/regression/images/formula_quadratic.png
PAGE=tests/regression/images/cc0/receipt_example.png
read -r CW CH < <(rgb "$CROP" "$S/crop.rgb")
read -r PW PH < <(rgb "$PAGE" "$S/page.rgb")
step mineru-python env PYTHONPATH=python python - "$S/mineru25-f16.gguf" "$LIB" "$CROP" "$PAGE" "$OUT" <<'PY'
import json, sys
from PIL import Image
from crispembed import CrispMinerU
model, lib, crop, page, out = sys.argv[1:]
with CrispMinerU(model, n_threads=2, lib_path=lib) as m:
    anchor = m.predict(Image.open(crop).convert("RGB"), "\nText Recognition:", 64)
    json.dump(anchor, open(out + "/mineru-anchor.json", "w"), ensure_ascii=False)
    native = m.native_extract(Image.open(page).convert("RGB"))
    blocks = m.two_step_extract(Image.open(page).convert("RGB"))
    markdown = m.to_markdown(blocks)
report = {"anchor_text": anchor.get("text"), "anchor_eos": anchor.get("eos"), "native_blocks": len(native.get("blocks", [])),
          "native_markdown_bytes": len(native.get("markdown", "")), "helper_blocks": len(blocks), "helper_markdown": markdown[:2000]}
json.dump(report, open(out + "/mineru-python.json", "w"), ensure_ascii=False, indent=2)
print(json.dumps(report, ensure_ascii=False)[:600])
assert anchor.get("eos") and anchor.get("text") and report["native_blocks"] > 0 and report["helper_blocks"] > 0 and markdown.strip()
PY
step mineru-rust-build probe_rust mineru25-public-probe tests/mineru25_public_bindings.rs
step mineru-rust "$S/mineru25-public-probe/target/release/mineru25-public-probe" "$S/mineru25-f16.gguf" "$S/crop.rgb" \
    "$CW" "$CH" "$OUT/mineru-anchor.json" 64 "$S/page.rgb" "$PW" "$PH"
step mineru-dart-build probe_dart mineru-dart mineru_probe tests/mineru25_public_bindings.dart
step mineru-dart "$S/mineru-dart/probe" "$LIB" "$S/mineru25-f16.gguf" "$S/crop.rgb" "$CW" "$CH" \
    "$OUT/mineru-anchor.json" 64 "$S/page.rgb" "$PW" "$PH"
step mineru-cli bash -c "'$CLI' --ocr-pipeline '$PAGE' --ocr-engine mineru25-pro --ocr-rec '$S/mineru25-f16.gguf' \
    > '$OUT/mineru-cli.txt' && test -s '$OUT/mineru-cli.txt'"
step mineru-http python - "$SERVER" "$S/mineru25-f16.gguf" "$PAGE" "$OUT" <<'PY'
import json, os, subprocess, sys, time, urllib.request
server, model, page, out = sys.argv[1:]
page = os.path.abspath(page)  # the server accepts paths under --image-root only, never inline bytes
proc = subprocess.Popen([server, "--ocr-pipeline", "--ocr-engine", "mineru25-pro", "--ocr-rec", model,
                         "--image-root", os.path.dirname(page), "--port", "18093", "--host", "127.0.0.1"],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
try:
    for _ in range(600):
        try:
            opener.open("http://127.0.0.1:18093/health", timeout=2).read(); break
        except Exception:
            if proc.poll() is not None: raise RuntimeError(proc.stdout.read().decode()[-3000:])
            time.sleep(1)
    body = json.dumps({"image": page}).encode()
    reply = opener.open(urllib.request.Request("http://127.0.0.1:18093/ocr/pipeline", body,
                                               {"Content-Type": "application/json"}), timeout=1800).read().decode()
    open(out + "/mineru-http.json", "w").write(reply)
    assert len(reply) > 20, reply
    print(reply[:400])
finally:
    proc.terminate()
PY
step mineru-license-gate env -u CRISPEMBED_MINERU25_PRO python -c "
import ctypes, sys
lib = ctypes.CDLL(sys.argv[1]); lib.crispembed_mineru_init.restype = ctypes.c_void_p
lib.crispembed_mineru_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
sys.exit(0 if not lib.crispembed_mineru_init(sys.argv[2].encode(), 1) else 1)" "$LIB" "$S/mineru25-f16.gguf"

{ echo "### OCR wrappers ($(git rev-parse --short HEAD))"; echo '```'; cat "$OUT/results.txt"; echo '```'; } > "$OUT/summary.md"
exit $failed
