#!/usr/bin/env bash
# MiniCPM-OCR-1.0 quantization sweep (docs/minicpm-ocr-q4k.md).
#
# Final output = the vocabulary scores at every generated position of the five
# frozen cases, teacher-forced on the frozen ORIGINAL generated IDs, compared by
# cosine with the ORIGINAL model's scores (Transformers 5.21.0 F32, one forward per
# case over frozen prompt + frozen IDs, recomputed here and checked against the
# frozen capture: input_ids/pixel_values bit-equal, row 0 vs the frozen first-step
# lm_head, argmax vs the frozen IDs). For each named recipe in
# tools/minicpm_quant_recipes.json: build the GGUF, census its stored types, run the
# native CLI teacher-forced (CRISPEMBED_MINICPM_OCR_FORCE_IDS/_LOGITS_OUT), score; a
# case whose forced argmax diverges is also decoded free-running (FREE=all: every case).
# Portable build (GGML_NATIVE=OFF, CRISPEMBED_NATIVE=OFF) so scores do not follow the runner ISA.
#
#   gh workflow run heavy-cpu.yml --ref <branch> -f script=tools/ci-heavy/minicpm_q4k_sweep.sh \
#     -f args="q8,q4k [stage-arms] [free-arms|all]" -f isolated_lane=q4k-minicpm-ocr-a \
#     -f pip="torch==2.13.0 torchvision transformers==5.21.0 pillow==11.3.0 numpy==2.2.6 gguf==0.17.1 kaggle==2.2.5"
set -uo pipefail
ARMS=${1:?comma list of recipe names}
STAGES=${2:-}   # arms whose prefill decoder stages are compared with the frozen original arrays
FREE=${3:-}     # arms (or "all") decoded free-running on every case
OUT=${HEAVY_OUT:?}
S=${HEAVY_SCRATCH:?}
H=tools/minicpm-ocr/q4k_sweep.py
REC=tools/minicpm_quant_recipes.json
T=$(nproc)
{
    echo "build_mode=portable (explicit -DGGML_NATIVE=OFF -DCRISPEMBED_NATIVE=OFF)"
    echo "heavy_cpu_build_mode=${CRISPEMBED_BUILD_MODE:-unset}"
    echo "commit=${GITHUB_SHA:-unknown}"
    echo "cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //')"
    echo "threads=$T"
} > "$OUT/build-mode.txt"
lscpu > "$OUT/cpu.txt" 2>/dev/null
cat "$OUT/build-mode.txt"

cmake -S . -B "$S/build" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DCRISPEMBED_NATIVE=OFF -DGGML_CUDA=OFF \
    > "$OUT/cmake.log" 2>&1 &&
    cmake --build "$S/build" --target crispembed-cli crispembed-quantize -j"$T" > "$OUT/build.log" 2>&1 ||
    { tail -40 "$OUT/build.log"; exit 2; }
CLI=$(find "$S/build" -name crispembed -type f -perm -u+x | head -1)
Q=$(find "$S/build" -name crispembed-quantize -type f | head -1)
grep -E "GGML_NATIVE|CRISPEMBED_NATIVE" "$S/build/CMakeCache.txt" >> "$OUT/build-mode.txt"

python $H fetch "$S/frozen" || exit 2
REF=$S/frozen/minicpm-ocr/reference.json
python $H original "$REF" "$S/ckpt" "$S/orig" > "$OUT/original.log" 2>&1 || { tail -30 "$OUT/original.log"; exit 2; }
cp "$S/orig/original.json" "$OUT/"
python - "$REF" "$S/cases" <<'EOF'
import json, sys, pathlib
ref = json.load(open(sys.argv[1])); d = pathlib.Path(sys.argv[2]); d.mkdir(exist_ok=True)
for c in ref['cases']:
    (d/(c['id']+'.ids')).write_text(' '.join(map(str, c['ids']))+'\n')
    (d/(c['id']+'.task')).write_text(c['task'])
EOF

gguf_for() { # recipe -> path of its GGUF
    local name=$1 src
    src=$(python -c "import json;print(json.load(open('$REC'))['recipes']['$name']['source'])")
    if [ ! -f "$S/$src.gguf" ]; then
        python models/convert-minicpm-ocr-to-gguf.py "$S/ckpt" "$S/$src.gguf" --dtype "$src" > "$OUT/convert-$src.log" 2>&1 ||
            return 1
    fi
    local argv; argv=$(python $H argv "$REC" "$name")
    if [ -z "$argv" ]; then echo "$S/$src.gguf"; return 0; fi
    # shellcheck disable=SC2086
    "$Q" "$S/$src.gguf" "$S/arm-$name.gguf" $argv > "$OUT/arms/$name/quantize.log" 2>&1 || return 1
    echo "$S/arm-$name.gguf"
}

run_case() { # gguf case dir mode(forced|free|dump) [dumpdir] -> writes case files
    local g=$1 c=$2 d=$3 mode=$4 dumpdir=${5:-} start end log=$3/$2.$4.log
    local env=(CRISPEMBED_MINICPM_OCR=1 CRISPEMBED_QWEN35_OCR=1 CRISPEMBED_QWEN35_QUANT=1 CRISPEMBED_MINICPM_OCR_TIMING=1)
    [ -n "$dumpdir" ] && env+=(CRISPEMBED_MINICPM_OCR_DUMP="$dumpdir")
    [ "$mode" = forced ] && env+=(CRISPEMBED_MINICPM_OCR_FORCE_IDS="$S/cases/$c.ids" CRISPEMBED_MINICPM_OCR_LOGITS_OUT="$d/$c.f32")
    start=$(date +%s.%N)
    env "${env[@]}" "$CLI" -m "$g" --ocr "$S/frozen/minicpm-ocr/$c.png" --ocr-prompt "$(cat "$S/cases/$c.task")" \
        --ocr-max-tokens 1024 -t "$T" > "$d/$c.$mode.txt" 2> "$log"
    local rc=$?
    end=$(date +%s.%N)
    python - "$log" "$d/$c.$mode.txt" "$rc" "$start" "$end" "$d/$c.$mode.json" <<'EOF'
import json, re, sys
log, txt, rc, a, b, out = sys.argv[1:]
m = re.search(r'minicpm_ocr\.ids n=\d+ list=([0-9,]*)', open(log, errors='replace').read())
ids = [int(x) for x in m.group(1).split(',') if x] if m else []
text = open(txt, errors='replace').read()
json.dump(dict(exit=int(rc), ids=ids, text=text.rstrip('\n'), seconds=round(float(b)-float(a), 3)), open(out, 'w'))
EOF
    return $rc
}

mkdir -p "$OUT/arms"
IFS=, read -ra LIST <<< "$ARMS"
for arm in "${LIST[@]}"; do
    D=$S/run/$arm; mkdir -p "$D" "$OUT/arms/$arm"
    echo "== $arm"
    G=$(gguf_for "$arm") || { echo "$arm: build failed"; echo "{\"arm\":\"$arm\",\"failed\":\"build\"}" > "$OUT/arms/$arm/score.json"; continue; }
    python $H census "$G" > "$OUT/arms/$arm/census.json"
    sha256sum "$G" | cut -d' ' -f1 > "$OUT/arms/$arm/gguf-sha256.txt"
    ok=1
    for c in line para-en table tall tiny; do
        run_case "$G" "$c" "$D" forced || { ok=0; tail -5 "$D/$c.forced.log"; }
        cp "$D/$c.forced.json" "$D/$c.forced.log" "$OUT/arms/$arm/" 2>/dev/null
    done
    if [ $ok = 0 ]; then echo "{\"arm\":\"$arm\",\"failed\":\"forced run\"}" > "$OUT/arms/$arm/score.json"; continue; fi
    python $H score "$REF" "$S/orig" "$arm" "$D" > "$D/pre.json" || continue
    for c in line para-en table tall tiny; do
        div=$(python -c "import json;print([x for x in json.load(open('$D/pre.json'))['cases'] if x['id']=='$c'][0]['first_divergence'])")
        if [ "$div" != None ] || [ "$FREE" = all ] || [[ ",$FREE," == *",$arm,"* ]]; then
            run_case "$G" "$c" "$D" free
            cp "$D/$c.free.json" "$OUT/arms/$arm/"
        fi
    done
    if [[ ",$STAGES," == *",$arm,"* ]]; then
        for c in line para-en table tall tiny; do
            mkdir -p "$D/dump/$c"
            run_case "$G" "$c" "$D" dump "$D/dump/$c" > /dev/null 2>&1
        done
        python $H stages "$REF" "$S/frozen/minicpm-ocr" "$D/dump" > "$OUT/arms/$arm/stages.json"
    fi
    python $H score "$REF" "$S/orig" "$arm" "$D" census="$OUT/arms/$arm/census.json" \
        gguf_sha256="$(cat "$OUT/arms/$arm/gguf-sha256.txt")" > "$OUT/arms/$arm/score.json"
    python -c "import json;r=json.load(open('$OUT/arms/$arm/score.json'));print('$arm', r['census']['file_mib'], 'MiB cos_min', round(r['cos_min'],5), r['worst_case'], 'exact', r['decoded_exact_cases'])"
    rm -rf "$D"/*.f32 "$D/dump"
    case "$G" in "$S"/arm-*) rm -f "$G" ;; esac
done
{
    echo "## MiniCPM-OCR quantization sweep"
    echo
    cat "$OUT/build-mode.txt" | sed 's/^/    /'
    echo
    python $H table "$OUT"/arms/*/score.json 2>/dev/null
} > "$OUT/summary.md"
cat "$OUT/summary.md"
