#!/usr/bin/env bash
# Model-free build of the merged final candidate (heavy-cpu.yml): root library,
# CLI and server with LLAMAFILE ON; SystemOne and MoEdern exports side by side;
# Python import, Rust workspace check. No model is loaded.
set -euo pipefail
R="$HEAVY_OUT"; B="$HEAVY_SCRATCH/final"; mkdir -p "$R"
cmake -S . -B "$B" -DCMAKE_BUILD_TYPE=Release -DCRISPEMBED_BUILD_SHARED=ON -DBUILD_SHARED_LIBS=OFF \
  -DGGML_NATIVE=OFF -DCRISPEMBED_NATIVE=OFF -DGGML_BLAS=OFF -DGGML_METAL=OFF -DGGML_CUDA=OFF \
  -DGGML_LLAMAFILE=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON > "$R/configure.txt"
cmake --build "$B" --target crispembed-shared crispembed-cli crispembed-server --parallel 2 > "$R/build.txt" 2>&1 || { tail -80 "$R/build.txt"; exit 1; }
nm -D --defined-only "$B/libcrispembed.so" | grep ' T crispembed_' > "$R/exports.txt"
test "$(grep -c ' crispembed_moedern_' "$R/exports.txt")" = 5
grep -q ' crispembed_systemone_run_json$' "$R/exports.txt"
if nm -C "$B/libcrispembed.so" | grep -E 'composed_test::|run_observed'; then echo "observer leaked"; exit 1; fi
"$B/crispembed" --help 2>&1 | grep -E -- '--systemone|--moedern' | tee "$R/cli-help.txt"
python -m pip install -q numpy==2.2.6
( cd python && python -c "import crispembed; print(crispembed.CrispSystemOne.__name__, crispembed.CrispMoEdern.__name__)" ) | tee "$R/python.txt"
CRISPEMBED_SYS_LIB_DIR="$B" cargo check --workspace 2>&1 | tail -3 | tee "$R/rust.txt"
echo "### Final candidate model-free build passed" > "$R/summary.md"
