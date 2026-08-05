#!/bin/bash
# UOCR_* value-parse audit: three-spelling verification per boolean gate
# (absent / =0 / =1), one variable per A/B, every run serialized (one heavy
# model process at a time on this 16 GB box).
#
# Judge: each run's OWN stdout (byte-compare vs the shared absent-arm
# baseline) + its OWN stderr (the UOCR_DBG gate-resolution line, path
# markers, MTL0). Fixture: fox.png, the regression-manifest gold sample for
# `unlimited-ocr-stacked`.
#
# macOS ships bash 3.2 — no `declare -A` anywhere in here.
set -u
WT=/Users/christianstrobele/code/CrispEmbed/.claude/worktrees/feat-uocr-gates
SP=/private/tmp/claude-501/-Users-christianstrobele-code-CrispEmbed/2c8d5a96-52f6-4edb-add5-a8eaf9151c9b/scratchpad
BIN=$WT/build/crispembed
PREBIN=$SP/crispembed-prefix
MODEL=$SP/uocr-model/unlimited-ocr-q4_k-stacked.gguf
FOX=$WT/tests/regression/images/fox.png
OUT=$(cd "$(dirname "$0")" && pwd)

run() { # name binary env-assignments...
  local name=$1 bin=$2; shift 2
  local t0=$SECONDS
  env "$@" "$bin" -m "$MODEL" --ocr "$FOX" --gpu-backend metal \
    > "$OUT/$name.txt" 2> "$OUT/$name.err"
  local rc=$?
  local mtl0=$(grep -c MTL0 "$OUT/$name.err")
  echo "RUN $name rc=$rc wall=$((SECONDS - t0))s mtl0=$mtl0"
}

# ---------------------------------------------------------------------------
# Pre-fix controls (parent-commit binary): prove `=0` USED to engage.
# ---------------------------------------------------------------------------
if [ "${STAGE:-all}" = all ] || [ "${STAGE:-}" = pre ]; then
echo "== pre-fix controls (parent-commit binary)"
run pre-base        "$PREBIN"
run pre-DBG-0       "$PREBIN" UOCR_DBG=0
run pre-PD-0        "$PREBIN" UOCR_DBG=1 UOCR_PD=0
run pre-NO_KV-0     "$PREBIN" UOCR_DBG=1 UOCR_NO_KV=0
fi

# ---------------------------------------------------------------------------
# Post-fix matrix
# ---------------------------------------------------------------------------
if [ "${STAGE:-all}" = all ] || [ "${STAGE:-}" = post ]; then
echo "== post-fix baselines"
run base            "$BIN"
run base-dbg        "$BIN" UOCR_DBG=1
run UOCR_DBG-0      "$BIN" UOCR_DBG=0

echo "== post-fix per-gate arms (UOCR_DBG=1 held constant; it writes stderr only)"
for G in UOCR_MMAP UOCR_MOE_CPU UOCR_SAM_CONV_CPU UOCR_OPT_GRAPH_LN UOCR_CLIP_DBG \
         UOCR_FA_F32 UOCR_LMHEAD_CPU UOCR_NO_KV UOCR_OPT_FUSED_DECODE \
         UOCR_DECODE_TIMING UOCR_INJECT_VIS UOCR_INJECT_REF UOCR_PD; do
  run "$G-0" "$BIN" UOCR_DBG=1 "$G=0"
  run "$G-1" "$BIN" UOCR_DBG=1 "$G=1"
done

echo "== PD-conditional gates (UOCR_PD=1 held constant — they only matter when the PD graph is built)"
run pd-base           "$BIN" UOCR_DBG=1 UOCR_PD=1
for G in UOCR_OPT_PD_F32 UOCR_PD_DBG UOCR_DECODE_REBUILD; do
  run "$G-0" "$BIN" UOCR_DBG=1 UOCR_PD=1 "$G=0"
  run "$G-1" "$BIN" UOCR_DBG=1 UOCR_PD=1 "$G=1"
done
fi

# UOCR_PD_DBG also instruments the per-step REBUILD path ([rb_dbg] dumps at
# generation steps 2..3). The PD path it primarily instruments segfaults at
# gen=2 on this fixture (pre-existing, see SUMMARY.md), so the [pd_dbg] dump is
# unreachable there — these two arms prove the gate on the path that survives.
if [ "${STAGE:-all}" = all ] || [ "${STAGE:-}" = rb ]; then
echo "== UOCR_PD_DBG on the rebuild path (no UOCR_PD)"
run UOCR_PD_DBG-rb-0 "$BIN" UOCR_DBG=1 UOCR_PD_DBG=0
run UOCR_PD_DBG-rb-1 "$BIN" UOCR_DBG=1 UOCR_PD_DBG=1
fi

# UOCR_OPT_FUSED_DECODE has no print of its own, but taking the fused path sets
# did_pd=true, which skips the `if (!did_pd)` rebuild block — and that block owns
# the [rb_timing] print. So UOCR_DECODE_TIMING=1 is a probe: [rb_timing] present
# => rebuild path ran; absent => the fused path ran.
if [ "${STAGE:-all}" = all ] || [ "${STAGE:-}" = fd ]; then
echo "== UOCR_OPT_FUSED_DECODE engagement probe (UOCR_DECODE_TIMING=1 as the reporter)"
run UOCR_OPT_FUSED_DECODE-t-0 "$BIN" UOCR_DBG=1 UOCR_DECODE_TIMING=1 UOCR_OPT_FUSED_DECODE=0
run UOCR_OPT_FUSED_DECODE-t-1 "$BIN" UOCR_DBG=1 UOCR_DECODE_TIMING=1 UOCR_OPT_FUSED_DECODE=1
fi

echo "GATES_RUNS_DONE"
