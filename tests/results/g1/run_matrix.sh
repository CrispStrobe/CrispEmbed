#!/bin/bash
# G1 acceptance matrix: 5-page T15 set x {metal, cpu} arms, serialized.
set -u
BIN=/Users/christianstrobele/code/CrispEmbed/.claude/worktrees/feat-smoldocling-metal/build/crispembed
MODEL=$HOME/.cache/crispembed-local/smoldocling-q8_0.gguf
IMGDIR=/Users/christianstrobele/code/CrispEmbed/tests/regression/images
OUT=$(dirname "$0")
PAGES="fox.png scan_page_pd.png cc0/commons_test_ocr_document.jpg cc0/simple_form.png cc0/receipt_historical.png"

for arm in metal cpu; do
  mkdir -p "$OUT/$arm"
  for p in $PAGES; do
    name=$(basename "$p")
    echo "=== $arm $name $(date +%H:%M:%S) loadavg=$(sysctl -n vm.loadavg)"
    CRISPEMBED_SMOLDOCLING_BENCH=1 "$BIN" -m "$MODEL" --ocr "$IMGDIR/$p" \
      --gpu-backend $arm > "$OUT/$arm/$name.txt" 2> "$OUT/$arm/$name.err"
    rc=$?
    if [ "$arm" = metal ] && ! grep -q "MTL0" "$OUT/$arm/$name.err"; then
      echo "FAIL: no MTL0 proof in $arm/$name stderr"
    fi
    echo "rc=$rc bytes=$(wc -c < "$OUT/$arm/$name.txt")"
    grep -E "smoldocling-bench" "$OUT/$arm/$name.err" | sed 's/^/  /'
  done
done
echo DONE
