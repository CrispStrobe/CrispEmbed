#!/bin/bash
# DS_* value-parse audit: three-spelling verification per gate, one variable
# per A/B, all runs serialized (one heavy consumer at a time).
# Judge: each run's own stdout (byte-compare vs the shared absent-arm
# baseline) + stderr markers (the DS_DBG gate-resolution line, decode-path
# lines, MTL0). fox.png engages crop mode (800x200 > 768).
set -u
WT=/Users/christianstrobele/code/CrispEmbed/.claude/worktrees/fix-ds-env-value-parse
BIN=$WT/build/crispembed
MODEL=/tmp/crispembed-regression/deepseek-ocr2-q4_k-stacked.gguf
FOX=$WT/tests/regression/images/fox.png
RECEIPT=$WT/tests/regression/images/cc0/receipt_historical.png
OUT=$(cd "$(dirname "$0")" && pwd)

run() { # name image env-assignments...
  local name=$1 img=$2; shift 2
  local t0=$SECONDS
  env "$@" "$BIN" -m "$MODEL" --ocr "$img" --gpu-backend metal \
    > "$OUT/$name.txt" 2> "$OUT/$name.err"
  local rc=$?
  local mtl0=$(grep -c MTL0 "$OUT/$name.err" || true)
  echo "RUN $name rc=$rc wall=$((SECONDS - t0))s mtl0=$mtl0"
}

# Baselines
run base            "$FOX"
run base-dbg        "$FOX" DS_DBG=1
run DS_DBG-0        "$FOX" DS_DBG=0

# Per-gate =0 / =1 arms (DS_DBG=1 held constant for the gate-resolution line;
# DS_DBG only writes stderr, stdout stays comparable to the plain baseline)
for G in DS_MMAP DS_MOE_CPU DS_SAM_CONV_CPU DS_QWEN2_ENC_FLASH DS_QWEN2_SCALAR DS_NO_KV DS_LMHEAD_CPU DS2_FORCE_CPU DS_PROFILE; do
  run "$G-0" "$FOX" DS_DBG=1 "$G=0"
  run "$G-1" "$FOX" DS_DBG=1 "$G=1"
done

# DS_LLM_FLASH only matters with f16 KV; hold DS2_KV_F16=1 constant
run kvf16-base      "$FOX" DS_DBG=1 DS2_KV_F16=1
run DS_LLM_FLASH-0  "$FOX" DS_DBG=1 DS2_KV_F16=1 DS_LLM_FLASH=0
run DS_LLM_FLASH-1  "$FOX" DS_DBG=1 DS2_KV_F16=1 DS_LLM_FLASH=1

# No-op-vs-main gate: default run must byte-match the recorded g2b default arm
# (g2b arms are raw CLI captures — no newline framing issue)
run noop-receipt    "$RECEIPT"

echo "== comparisons"
fail=0
ck() { cmp -s "$1" "$2" && echo "OK   $3" || { echo "DIFF $3"; fail=1; }; }

ck "$OUT/base-dbg.txt" "$OUT/base.txt" "DS_DBG=1 stdout == base"
ck "$OUT/DS_DBG-0.txt" "$OUT/base.txt" "DS_DBG=0 stdout == base"
grep -q '\[dbg\]' "$OUT/base.err"     && { echo "BAD  dbg lines in absent arm"; fail=1; } || echo "OK   no dbg lines when absent"
grep -q '\[dbg\]' "$OUT/DS_DBG-0.err" && { echo "BAD  dbg lines in =0 arm"; fail=1; }     || echo "OK   no dbg lines at DS_DBG=0"
grep -q '\[dbg\] gates:' "$OUT/base-dbg.err" && echo "OK   gates line at DS_DBG=1" || { echo "BAD  no gates line at DS_DBG=1"; fail=1; }

field() { # file field -> value
  grep -o "$2=[01]" "$1" | head -1 | cut -d= -f2
}
# gate:field pairs (macOS ships bash 3.2 — no associative arrays)
for pair in DS_MMAP:mmap DS_MOE_CPU:moe_cpu DS_SAM_CONV_CPU:sam_conv_cpu \
            DS_QWEN2_ENC_FLASH:qwen2_enc_flash DS_QWEN2_SCALAR:qwen2_scalar \
            DS_NO_KV:no_kv DS_LMHEAD_CPU:lmhead_cpu DS2_FORCE_CPU:force_cpu DS_PROFILE:profile; do
  G=${pair%%:*}; f=${pair##*:}
  v0=$(field "$OUT/$G-0.err" "$f"); v1=$(field "$OUT/$G-1.err" "$f")
  [ "$v0" = 0 ] && echo "OK   $G=0 parses off" || { echo "BAD  $G=0 parses $v0"; fail=1; }
  [ "$v1" = 1 ] && echo "OK   $G=1 parses on"  || { echo "BAD  $G=1 parses $v1"; fail=1; }
  ck "$OUT/$G-0.txt" "$OUT/base.txt" "$G=0 stdout == base"
done
v0=$(field "$OUT/DS_LLM_FLASH-0.err" llm_flash); v1=$(field "$OUT/DS_LLM_FLASH-1.err" llm_flash)
[ "$v0" = 0 ] && echo "OK   DS_LLM_FLASH=0 parses off" || { echo "BAD  DS_LLM_FLASH=0 parses $v0"; fail=1; }
[ "$v1" = 1 ] && echo "OK   DS_LLM_FLASH=1 parses on"  || { echo "BAD  DS_LLM_FLASH=1 parses $v1"; fail=1; }
ck "$OUT/DS_LLM_FLASH-0.txt" "$OUT/kvf16-base.txt" "DS_LLM_FLASH=0 stdout == kvf16 base"

echo "== per-gate =1 engagement markers"
grep -q 'legacy per-layer (CPU MoE' "$OUT/DS_MOE_CPU-1.err"    && echo "OK   MOE_CPU=1 blocks persistent decode" || { echo "BAD  MOE_CPU=1 marker missing"; fail=1; }
grep -q 'legacy per-layer (DS_NO_KV=1' "$OUT/DS_NO_KV-1.err"   && echo "OK   NO_KV=1 blocks persistent decode"   || { echo "BAD  NO_KV=1 marker missing"; fail=1; }
grep -q 'DS2_FORCE_CPU=1' "$OUT/DS2_FORCE_CPU-1.err"           && echo "OK   FORCE_CPU=1 announces CPU backend"  || { echo "BAD  FORCE_CPU=1 marker missing"; fail=1; }
grep -q '\[ds-profile\]' "$OUT/DS_PROFILE-1.err"               && echo "OK   PROFILE=1 prints profile"           || { echo "BAD  PROFILE=1 marker missing"; fail=1; }
grep -q 'kv=f16, flash' "$OUT/DS_LLM_FLASH-1.err"              && echo "OK   LLM_FLASH=1 decode-path says flash" || { echo "BAD  LLM_FLASH=1 marker missing"; fail=1; }
grep -q 'kv=f16, flash' "$OUT/DS_LLM_FLASH-0.err"              && { echo "BAD  LLM_FLASH=0 says flash"; fail=1; } || echo "OK   LLM_FLASH=0 decode-path no flash"
ck "$OUT/DS_MMAP-1.txt" "$OUT/base.txt" "MMAP=1 stdout == base (same weights, same compute)"

ck "$OUT/noop-receipt.txt" "$WT/tests/results/g2b/m-default-cc0/receipt_historical.txt" "default receipt == recorded g2b default arm (no-op vs main)"

echo "GATES_DONE fail=$fail"
