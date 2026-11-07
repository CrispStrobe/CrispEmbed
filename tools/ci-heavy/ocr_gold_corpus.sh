#!/usr/bin/env bash
# OCR gold corpus v1: verify every image pin and derive the Wikisource block boxes.
#
#   gh workflow run heavy-cpu.yml -R CrispStrobe/CrispEmbed --ref <branch> \
#       -f script=tools/ci-heavy/ocr_gold_corpus.sh -f isolated_lane=ocr-gold-corpus
#
# 1. Every Wikimedia image: Commons file revision sha1 must equal the pin, then the bytes from the
#    revision-qualified URL (fallback: bare URL) must hash to the pinned sha256. A mismatch is
#    reported per page (drift) and that page gets no derived boxes.
# 2. CORD: the pinned test parquet must hash to its pin; image sha256/bytes are filled for new rows
#    and must equal the existing pins for rows reused from docs/ocr-real-scans-v2.json.
# 3. Official Tesseract (Ubuntu package; eng / deu(+frk for Fraktur) / fra) word boxes per Wikisource
#    page -> tools/ocr-gold-corpus/build_corpus.py boxes (alignment only; Tesseract text is never
#    written into the gold).
# Results: $HEAVY_OUT/ocr-gold-corpus-v1.json (candidate corpus), pins.json, tsv/, summary.md.
set -uo pipefail
OUT=${HEAVY_OUT:?}
S=${HEAVY_SCRATCH:?}
CORPUS=${1:-docs/ocr-gold-corpus-v1.json}
mkdir -p "$OUT/tsv" "$S/img"
{ echo "source=$(git rev-parse HEAD)"; python --version; } > "$OUT/provenance.txt"

echo "::group::tools"
sudo apt-get update -qq >/dev/null && sudo apt-get install -y -qq tesseract-ocr tesseract-ocr-eng tesseract-ocr-deu \
    tesseract-ocr-fra tesseract-ocr-frk > /dev/null || exit 3
python -m pip install -q pyarrow || exit 3
tesseract --version 2>&1 | head -1 | tee "$OUT/tesseract-version.txt"
TV="tesseract-$(tesseract --version 2>&1 | head -1 | awk '{print $2}')"
echo "::endgroup::"

echo "::group::image pins"
python -I tools/ocr-gold-corpus/verify_pins.py "$CORPUS" "$S/img" "$OUT/pins.json" || pins_status=$?
echo "::endgroup::"

echo "::group::cord parquet"
url=$(python -I -c "import json,sys;m=json.load(open(sys.argv[1]));print(next(p['image']['hf_parquet']['url'] for p in m['pages'] if p['source']=='cord-v2'))" "$CORPUS")
curl -fsSL -o "$S/cord-test.parquet" "$url" || exit 5
python -I tools/ocr-gold-corpus/build_corpus.py cord-pin "$CORPUS" "$S/corpus-cord.json" "$S/cord-test.parquet" || exit 6
echo "::endgroup::"

echo "::group::tesseract word boxes"
python -I - "$S/corpus-cord.json" "$OUT/pins.json" > "$S/jobs.txt" <<'PY'
import json, sys
m = json.load(open(sys.argv[1])); pins = json.load(open(sys.argv[2]))
for p in m['pages']:
    if p['source'] != 'wikisource' or not pins.get(p['id'], {}).get('ok'):
        continue
    lang = {'en': 'eng', 'fr': 'fra', 'de': 'deu'}[p['tags'][0]]
    if 'fraktur' in p['tags']:
        lang = 'deu+frk'
    print(p['id'], lang, pins[p['id']]['path'])
PY
while read -r id lang img; do
    tesseract "$img" "$OUT/tsv/$id" -l "$lang" --psm 3 tsv > /dev/null 2>&1 || echo "$id tesseract failed" | tee -a "$OUT/errors.txt"
done < "$S/jobs.txt"
echo "::endgroup::"

python -I tools/ocr-gold-corpus/build_corpus.py boxes "$S/corpus-cord.json" "$OUT/ocr-gold-corpus-v1.json" "$OUT/tsv" "$TV" || exit 7
python -I - "$OUT/ocr-gold-corpus-v1.json" "$OUT/pins.json" > "$OUT/summary.md" <<'PY'
import json, sys
m = json.load(open(sys.argv[1])); pins = json.load(open(sys.argv[2]))
print('## OCR gold corpus v1: pins and derived boxes\n')
print('| page | pin | blocks | with bbox |'); print('|---|---|---|---|')
tb = tw = 0
for p in m['pages']:
    n = len(p['blocks']); w = sum(b['bbox'] is not None for b in p['blocks']); tb += n; tw += w
    print(f"| {p['id']} | {pins.get(p['id'], {}).get('status', '-')} | {n} | {w} |")
print(f'\n{len(m["pages"])} pages, {tw}/{tb} blocks with a bbox')
PY
cat "$OUT/summary.md"
exit ${pins_status:-0}
