#!/bin/bash
# G1 interleaved timing: (metal, cpu) pairs back-to-back, loadavg-gated.
set -u
BIN=/Users/christianstrobele/code/CrispEmbed/.claude/worktrees/feat-smoldocling-metal/build/crispembed
MODEL=$HOME/.cache/crispembed-local/smoldocling-q8_0.gguf
IMGDIR=/Users/christianstrobele/code/CrispEmbed/tests/regression/images
OUT=$(dirname "$0")/timing
mkdir -p "$OUT"

run_one() { # arm page tag
  local arm=$1 page=$2 tag=$3
  local err="$OUT/$tag.err" txt="$OUT/$tag.txt"
  local la=$(sysctl -n vm.loadavg | awk '{print $2}')
  CRISPEMBED_SMOLDOCLING_BENCH=1 "$BIN" -m "$MODEL" --ocr "$IMGDIR/$page" \
      --gpu-backend $arm > "$txt" 2> "$err"
  local rc=$?
  local vis=$(grep -o 'vision_encoder+connector: [0-9]*' "$err" | grep -o '[0-9]*')
  local tot=$(grep -o 'total: [0-9.]*' "$err" | grep -o '[0-9.]*')
  local mtl=no
  grep -q MTL0 "$err" && mtl=yes
  echo "$tag arm=$arm page=$page rc=$rc loadavg=$la mtl0=$mtl vis_ms=$vis total_ms=$tot"
}

for page in fox.png scan_page_pd.png; do
  for i in 0 1 2 3; do   # pair 0 = cold, discarded in analysis
    run_one metal "$page" "$page-m$i"
    run_one cpu   "$page" "$page-c$i"
  done
done
echo TIMING_DONE
