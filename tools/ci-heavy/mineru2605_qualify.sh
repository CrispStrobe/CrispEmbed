#!/usr/bin/env bash
# Exact 2605 original-Python/native qualification. All model payloads/builds
# belong on hosted CI. No publication or registry update is performed.
# gh workflow run heavy-cpu.yml --ref <branch> \
#   -f script=tools/ci-heavy/mineru2605_qualify.sh \
#   -f args="receipt f32,f16,q8_0,q4_k 2048 3 1" \
#   -f build_mode=portable -f isolated_lane=mineru2605-receipt
# Args: CASE_IDS (or all), TIERS, MAX_TOKENS, REPEAT, DOCUMENT_HELPERS(0/1).
set -euo pipefail
CASE_IDS=${1:?comma-separated frozen case IDs or all}
TIERS=${2:-f32,f16,q8_0,q4_k}
MAX_TOKENS=${3:-2048}
REPEAT=${4:-3}
DOCUMENT_HELPERS=${5:-1}
OUT=${HEAVY_OUT:?}
S=${HEAVY_SCRATCH:?}
ROOT=$(pwd)
export CRISPEMBED_BUILD_MODE=portable
export CRISPEMBED_MINERU25_PRO=1 CRISPEMBED_MINERU25_PRO_2605=1 QWEN2VL_OCR_FORCE_CPU=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p "$OUT" "$S/mineru2605"
S=$S/mineru2605
python - "$CASE_IDS" "$TIERS" "$MAX_TOKENS" "$REPEAT" "$DOCUMENT_HELPERS" "$OUT" <<'PY'
import hashlib,json,pathlib,sys
protocol=json.load(open('docs/mineru25-pro-2605-protocol.json'))
ids=list(protocol['cases']) if sys.argv[1]=='all' else sys.argv[1].split(',')
tiers=sys.argv[2].split(',')
assert ids and len(set(ids))==len(ids) and set(ids)<=protocol['cases'].keys(), 'unknown/duplicate case'
assert tiers and len(set(tiers))==len(tiers) and set(tiers)<=protocol['precision_bars_exclusive'].keys(), 'unknown/duplicate precision'
assert 0<=int(sys.argv[3])<=8192 and 2<=int(sys.argv[4])<=100 and sys.argv[5] in ('0','1')
for cid in ids:
 row=protocol['cases'][cid];assert hashlib.sha256(pathlib.Path(row['image']).read_bytes()).hexdigest()==row['sha256']
pathlib.Path(sys.argv[6],'selected-cases.tsv').write_text(''.join(f"{cid}\t{protocol['cases'][cid]['image']}\t{protocol['cases'][cid]['task']}\n" for cid in ids))
pathlib.Path(sys.argv[6],'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
PY
{
    git rev-parse HEAD
    git submodule status
    printf 'mode=portable threads=4 cases=%s tiers=%s max_tokens=%s repeat=%s\n' "$CASE_IDS" "$TIERS" "$MAX_TOKENS" "$REPEAT"
    lscpu
} > "$OUT/provenance.txt"
python -m pip install -q torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -q -r tools/mineru2605-reference-requirements.txt
python -m pip freeze > "$OUT/packages.txt"
# Download and check the small helper artifact BEFORE importing its sources.
python - "$S" <<'PY'
import hashlib,json,pathlib,sys,urllib.request,zipfile
out=pathlib.Path(sys.argv[1]);manifest=json.load(open('models/mineru25-pro-2605-source.json'))
b=manifest['blueprint'];wheel=out/b['wheel_name']
with urllib.request.urlopen(b['wheel_url'],timeout=90) as r: data=r.read()
assert len(data)==b['wheel_size'] and hashlib.sha256(data).hexdigest()==b['wheel_sha256']
wheel.write_bytes(data)
with zipfile.ZipFile(wheel) as z:
 for name in z.namelist():
  p=pathlib.PurePosixPath(name)
  assert not p.is_absolute() and '..' not in p.parts
 z.extractall(out/'blueprint')
from huggingface_hub import snapshot_download
snapshot_download(manifest['repository'],revision=manifest['revision'],local_dir=out/'checkpoint',allow_patterns=list(manifest['files']))
PY
cmake -S . -B "$S/build" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DCRISPEMBED_NATIVE=OFF -DGGML_CUDA=OFF > "$OUT/cmake.log" 2>&1
cmake --build "$S/build" -j4 --target test-mineru25-run test-mineru25-diff test-mineru25-profile crispembed-quantize crispembed-shared > "$OUT/build.log" 2>&1
cp "$S/build/CMakeCache.txt" "$OUT/CMakeCache.txt"
find_bin() { find "$S/build" -type f -name "$1" -perm -u+x -print -quit; }
RUN=$(find_bin test-mineru25-run)
DIFF=$(find_bin test-mineru25-diff)
QUANT=$(find_bin crispembed-quantize)
LIB=$(find "$S/build" -type f -name 'libcrispembed.so*' -print -quit)
"$(find_bin test-mineru25-profile)" > "$OUT/profile-controls.txt" 2>&1
mkdir -p "$S/gguf"
python models/convert-mineru25-pro-to-gguf.py --checkpoint 2605 --model "$S/checkpoint" --dtype f32 --output "$S/gguf/f32.gguf" > "$OUT/conversion-f32.log" 2>&1
cp "$S/gguf/f32.gguf.json" "$OUT/conversion-f32.json"
IFS=, read -ra VARIANTS <<< "$TIERS"
for tier in "${VARIANTS[@]}"; do
    case "$tier" in
        f32) ;;
        f16)
            python models/convert-mineru25-pro-to-gguf.py --checkpoint 2605 --model "$S/checkpoint" --dtype f16 --output "$S/gguf/f16.gguf" > "$OUT/conversion-f16.log" 2>&1
            cp "$S/gguf/f16.gguf.json" "$OUT/conversion-f16.json" ;;
        q8_0|q4_k) "$QUANT" "$S/gguf/f32.gguf" "$S/gguf/$tier.gguf" "$tier" > "$OUT/quantize-$tier.log" 2>&1 ;;
    esac
    python tools/qwen2vl_quant_census.py census "$S/gguf/$tier.gguf" "$OUT/$tier-tensor-census.json"
done
status=0
while IFS=$'\t' read -r cid source task; do
    c=$OUT/$cid
    mkdir -p "$c" "$S/$cid"
    # Decode once. Both arms consume the exact same lossless RGB page bytes.
    python - "$source" "$S/$cid/page.png" "$c/image.json" <<'PY'
from PIL import Image
import hashlib,json,sys
Image.open(sys.argv[1]).convert('RGB').save(sys.argv[2])
json.dump({'source_sha256':hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest(),
           'decoded_png_sha256':hashlib.sha256(open(sys.argv[2],'rb').read()).hexdigest(),
           'policy':'both engines consume the same Pillow-decoded RGB PNG'},open(sys.argv[3],'w'),indent=2)
PY
    if ! python tools/mineru2605_validate.py reference --model "$S/checkpoint" --blueprint "$S/blueprint" \
        --image "$S/$cid/page.png" --task "$task" --threads 4 --max-tokens "$MAX_TOKENS" --repeat "$REPEAT" \
        --dtype f32 --output "$c/original.json" --dump-reference "$S/$cid/stages.gguf" --logits-dir "$S/$cid/original-logits" > "$c/original.log" 2>&1; then
        status=1
        continue
    fi
    for tier in "${VARIANTS[@]}"; do
        if ! python tools/mineru2605_qualify.py --reference "$c/original.json" --reference-stages "$S/$cid/stages.gguf" \
            --reference-logits "$S/$cid/original-logits" --native "$RUN" --diff "$DIFF" --gguf "$S/gguf/$tier.gguf" \
            --image "$S/$cid/page.png" --precision "$tier" --cmake-cache "$S/build/CMakeCache.txt" \
            --native-logits "$S/$cid/$tier-logits" --out "$c/$tier" > "$c/$tier.log" 2>&1; then
            status=1
        fi
    done
    if [[ "$DOCUMENT_HELPERS" = 1 ]]; then
        if ! python tools/mineru2605_document.py --blueprint "$S/blueprint" --gguf "$S/gguf/f32.gguf" --lib "$LIB" \
            --image "$S/$cid/page.png" --threads 4 --max-tokens "$MAX_TOKENS" --out "$c/document" > "$c/document.log" 2>&1; then
            status=1
        fi
    fi
done < "$OUT/selected-cases.tsv"
python - "$OUT" "$TIERS" "$status" <<'PY'
import json,pathlib,sys
out=pathlib.Path(sys.argv[1]);tiers=sys.argv[2].split(',');ids=[l.split('\t')[0] for l in (out/'selected-cases.tsv').read_text().splitlines()]
rows=[]
for cid in ids:
 for tier in tiers:
  p=out/cid/tier/'precision.json'
  rows.append({'case':cid,'precision':tier,'measured':p.exists(),**(json.loads(p.read_text()) if p.exists() else {})})
json.dump({'schema':'crispembed.mineru2605.lane.v1','cases':ids,'all_frozen_cases':len(ids)==8,
           'build_mode':'portable','status':int(sys.argv[3]),'rows':rows,
           'all_selected_precision_pass':bool(rows) and all(x.get('precision_pass',False) for x in rows),
           'benchmark_qualified':False,'registry_alias':None},open(out/'summary.json','w'),indent=2)
PY
exit "$status"
