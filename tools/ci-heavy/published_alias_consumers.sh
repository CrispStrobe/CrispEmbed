#!/usr/bin/env bash
# Actual consumers of the published immutable aliases (heavy-cpu.yml): fresh cache ->
# alias auto-download -> exact SHA256 -> real CLI inference -> offline reuse, identical.
set -euo pipefail
R="$HEAVY_OUT"; B="$HEAVY_SCRATCH/b"; C="$HEAVY_SCRATCH/cache"; mkdir -p "$R" "$C"
python -m pip install -q numpy==2.2.6
cmake -S . -B "$B" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DCRISPEMBED_NATIVE=OFF -DGGML_LLAMAFILE=ON > "$R/configure.txt"
cmake --build "$B" --target crispembed-cli --parallel 2 > "$R/build.txt" 2>&1 || { tail -40 "$R/build.txt"; exit 1; }
export CRISPEMBED_CACHE_DIR="$C"
export CRISPEMBED_MOEDERN_PUBLIC_TEXT=1 CRISPEMBED_MOEDERN_PACKED=1 CRISPEMBED_MOEDERN_TEXT_COMPOSED=1 CRISPEMBED_MOEDERN_TEXT_FRONTEND=1 CRISPEMBED_MOEDERN_COLBERT=1
export CRISPEMBED_JOSEV_PACKED=1 CRISPEMBED_JOSEV_COMPOSED=1 CRISPEMBED_JOSEV_BACKBONE=1 CRISPEMBED_QWEN35_OCR=1 CRISPEMBED_JOSEV_HEAD=1 CRISPEMBED_JOSEV_TEXT_FRONTEND=1 CRISPEMBED_JOSEV_ANSWER_SERIALIZER=1
check_sha() { f=$(find "$C" -name "$1" -type f | head -1); test -n "$f"; echo "$(sha256sum "$f" | cut -d' ' -f1)"; }
# MoEdern 1.5M: frozen q-de text; first IDs must equal the original panel's.
"$B/crispembed" -m moedern-1.5m --moedern query "Wie funktioniert die Photosynthese?" > "$R/moedern-online.json"
test "$(check_sha moedern-1.5m-text-f32.gguf)" = 0e2b8773abd8c78a5825280f9a73cb808b88d56f0675a8e4bb7f0509f5b6cc51
CRISPEMBED_OFFLINE=1 "$B/crispembed" -m moedern-1.5m --moedern query "Wie funktioniert die Photosynthese?" > "$R/moedern-offline.json"
cmp "$R/moedern-online.json" "$R/moedern-offline.json"
python - "$R/moedern-online.json" <<'P'
import json,sys
r=json.load(open(sys.argv[1]))
assert r['ids'][:12]==[2,30000,11842,11275,3309,1944,38,1327,7064,4124,11836,60], r['ids'][:12]
assert len(r['selected_vectors'])==32*128, len(r['selected_vectors'])
print('moedern ok')
P
# Josev: first frozen raw request; complete typed response must equal the original.
REQ=$(python -c "import json;print(json.dumps(json.load(open('tests/results/josev2b/original-input-freeze.json'))['records'][0]['request'],ensure_ascii=False,separators=(',',':')))")
"$B/crispembed" -m josev2b-systemone --systemone "$REQ" -t 4 > "$R/josev-online.json"
test "$(check_sha josev2b-systemone-f32.gguf)" = 72684e39d0eed526b40c5150c78b2d881d324c39e16af48060b3d741f526ccf2
CRISPEMBED_OFFLINE=1 "$B/crispembed" -m josev2b-systemone --systemone "$REQ" -t 4 > "$R/josev-offline.json"
cmp "$R/josev-online.json" "$R/josev-offline.json"
python - "$R/josev-online.json" <<'P'
import json,sys
got=json.load(open(sys.argv[1]));gold=json.load(open('tests/results/josev2b/original-reference.json'))['cases'][0]['official_response']
assert got==gold,(got,gold);print('josev ok')
P
echo "### Published alias consumers: fresh-cache SHA, real inference, offline reuse all passed" > "$R/summary.md"
