#!/bin/bash
# G2b byte-identity gates for the default flip, serialized:
#  default-on runs must equal the g2 crop arms; DS2_CROP_MODE=0 must equal base arms.
set -u
WT=/Users/christianstrobele/code/CrispEmbed/.claude/worktrees/feat-ds2-crop-default
BIN=$WT/build/crispembed
MODEL=/tmp/crispembed-regression/deepseek-ocr2-q4_k-stacked.gguf
IMG=$WT/tests/regression/images
G2=$WT/tests/results/g2
OUT=$(dirname "$0")
CC0="commons_example_receipt.png commons_test_ocr_document.jpg german_official_print.jpg receipt_historical.png simple_form.png"

run() { # outdir arm env page
  local dir=$1 arm=$2 envv=$3 page=$4
  mkdir -p "$OUT/$dir"
  local stem=${page%.*}
  env $envv "$BIN" -m "$MODEL" --ocr "$IMG/cc0/$page" --gpu-backend $arm \
    > "$OUT/$dir/$stem.txt" 2> "$OUT/$dir/$stem.err"
  echo "$dir/$stem rc=$? $( [ $arm = metal ] && (grep -q MTL0 "$OUT/$dir/$stem.err" && echo mtl0=yes || echo mtl0=NO) )"
}

echo "== default-on metal cc0 vs g2/m-crop-cc0"
for p in $CC0; do run m-default-cc0 metal "DS2_TRUE=1" "$p"; done
echo "== default-on cpu cc0 vs g2/c-crop-cc0"
for p in $CC0; do run c-default-cc0 cpu "DS2_TRUE=1" "$p"; done
echo "== gate-off metal cc0 (2 pages) vs g2/m-base-cc0"
for p in receipt_historical.png german_official_print.jpg; do run m-off-cc0 metal "DS2_CROP_MODE=0" "$p"; done

echo "== byte-compare"
fail=0
for p in $CC0; do s=${p%.*}
  cmp -s "$OUT/m-default-cc0/$s.txt" "$G2/m-crop-cc0/$s.txt" && echo "OK m $s" || { echo "DIFF m $s"; fail=1; }
  cmp -s "$OUT/c-default-cc0/$s.txt" "$G2/c-crop-cc0/$s.txt" && echo "OK c $s" || { echo "DIFF c $s"; fail=1; }
done
for p in receipt_historical german_official_print; do
  cmp -s "$OUT/m-off-cc0/$p.txt" "$G2/m-base-cc0/$p.txt" && echo "OK off $p" || { echo "DIFF off $p"; fail=1; }
done
echo "IDENTITY_DONE fail=$fail"
