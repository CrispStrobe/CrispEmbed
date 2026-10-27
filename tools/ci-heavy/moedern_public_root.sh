#!/usr/bin/env bash
# Model-free proof of root-wired MoEdern public text (heavy-cpu.yml): actual root
# shared library/CLI/server with production LLAMAFILE ON, exports, ABI controls,
# observer exclusion, arithmetic flags, Python import, Rust and Dart checks.
set -euo pipefail
git fetch -q --unshallow origin 2>/dev/null || true
R="$HEAVY_OUT"; B="$HEAVY_SCRATCH/root"; mkdir -p "$R"
python tools/moedern-public/root_contract.py | tee "$R/source.json"
python tests/test_moedern_public_root.py -v 2>&1 | tee "$R/stdlib.txt"
cmake -S . -B "$B" -DCMAKE_BUILD_TYPE=Release -DCRISPEMBED_BUILD_SHARED=ON -DBUILD_SHARED_LIBS=OFF \
  -DGGML_NATIVE=OFF -DCRISPEMBED_NATIVE=OFF -DGGML_BLAS=OFF -DGGML_METAL=OFF -DGGML_CUDA=OFF \
  -DGGML_LLAMAFILE=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON > "$R/configure.txt"
cmake --build "$B" --target crispembed-shared crispembed-cli crispembed-server --parallel 2 > "$R/build.txt" 2>&1 || { tail -60 "$R/build.txt"; exit 1; }
grep -q 'GGML_LLAMAFILE:BOOL=ON' "$B/CMakeCache.txt"
nm -D --defined-only "$B/libcrispembed.so" | grep ' T crispembed_' > "$R/exports.txt"
test "$(grep -c ' crispembed_moedern_' "$R/exports.txt")" = 5
grep -q ' crispembed_init$' "$R/exports.txt"
if nm -C "$B/libcrispembed.so" | grep -E 'moedern.*(composed_test|observe)'; then echo "observer leaked"; exit 1; fi
cp "$B/compile_commands.json" "$R/"
python - "$R/compile_commands.json" <<'P'
import json,sys
rows=[r for r in json.load(open(sys.argv[1])) if r['file'].rsplit('/',1)[-1] in
      {'moedern_text_engine.cpp','moedern_text_frontend.cpp','moedern_text.cpp','moedern_text_api.cpp'}]
assert rows and all('-ffp-contract=off' in r['command'] and 'TESTING' not in r['command']
                    and '-ffast-math' not in r['command'] for r in rows), rows
print('flags ok', len(rows))
P
# Same plain-C ABI controls, now linked against the actual root library.
cc -O2 -I src tests/moedern-public/controls.c -L "$B" -lcrispembed -Wl,-rpath,"$B" -o "$HEAVY_SCRATCH/controls"
"$HEAVY_SCRATCH/controls" | tee "$R/controls.txt"
python -m pip install -q numpy==2.2.6
( cd python && CRISPEMBED_LIB_PATH="$B/libcrispembed.so" python -c "import crispembed; print('python export', crispembed.CrispMoEdern.__name__)" ) | tee "$R/python.txt"
CRISPEMBED_SYS_LIB_DIR="$B" cargo check --workspace 2>&1 | tee "$R/rust.txt"
curl -sSLo "$HEAVY_SCRATCH/dart.zip" https://storage.googleapis.com/dart-archive/channels/stable/release/3.13.5/sdk/dartsdk-linux-x64-release.zip
unzip -q "$HEAVY_SCRATCH/dart.zip" -d "$HEAVY_SCRATCH"
D="$HEAVY_SCRATCH/dt"; mkdir -p "$D/lib"; cp flutter/crispembed/lib/src/moedern_text.dart "$D/lib/"
printf 'name: moedern_audit\nenvironment:\n  sdk: ">=3.0.0 <4.0.0"\ndependencies:\n  ffi: 2.1.4\n' > "$D/pubspec.yaml"
( cd "$D" && "$HEAVY_SCRATCH/dart-sdk/bin/dart" pub get && "$HEAVY_SCRATCH/dart-sdk/bin/dart" analyze --fatal-infos ) 2>&1 | tee "$R/dart.txt"
echo "### MoEdern root-wired model-free proof passed" > "$R/summary.md"
