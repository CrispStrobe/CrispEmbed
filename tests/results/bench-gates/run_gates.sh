#!/bin/bash
# run_gates.sh — three-spelling proof for the CRISPEMBED_*_BENCH value-parse
# sweep. macOS bash 3.2: no `declare -A`, no `readarray`.
#
# Arm naming: <tag>-{absent,zero,one}.{txt,err}
#   absent -> variable unset      (must print NO bench lines)
#   zero   -> variable set to "0" (must print NO bench lines — THE FIX)
#   one    -> variable set to "1" (must print bench lines)
# stdout must be byte-identical across all three arms in every case.
#
# The `pre-*` arms in this directory were captured on the SAME binary built
# from the parent commit (pre-fix) and show `=0` printing the bench lines.
set -u

WT=/Users/christianstrobele/code/CrispEmbed/.claude/worktrees/feat-bench-gates
BIN=$WT/build/crispembed
OUT=$WT/tests/results/bench-gates
LOCAL=$HOME/.cache/crispembed-local
LIVE=$HOME/crispembed-live-cache
IMG=$WT/tests/regression/images

run_arm() { # tag var value img_args...
    tag=$1; var=$2; val=$3; shift 3
    if [ "$val" = "-" ]; then
        env -u "$var" "$@" > "$OUT/$tag.txt" 2> "$OUT/$tag.err"
    else
        env "$var=$val" "$@" > "$OUT/$tag.txt" 2> "$OUT/$tag.err"
    fi
    echo "  $tag rc=$?"
}

# ---------------------------------------------------------------------------
# A. crispembed.cpp / CRISPEMBED_CRISPEMBED_BENCH — full three-spelling proof
# ---------------------------------------------------------------------------
M=$LOCAL/multilingual-e5-small-q8_0.gguf
echo "[A] CRISPEMBED_CRISPEMBED_BENCH (multilingual-e5-small-q8_0, metal)"
for spell in absent:- zero:0 one:1; do
    name=${spell%%:*}; val=${spell##*:}
    run_arm "post-$name" CRISPEMBED_CRISPEMBED_BENCH "$val" \
        "$BIN" -m "$M" --gpu-backend metal --json "the quick brown fox"
done

# ---------------------------------------------------------------------------
# B. cc_detect.cpp / CRISPEMBED_CC_DETECT_BENCH — model-free spot-check
# ---------------------------------------------------------------------------
echo "[B] CRISPEMBED_CC_DETECT_BENCH (model-free --cc-detect)"
for spell in absent:- zero:0 one:1; do
    name=${spell%%:*}; val=${spell##*:}
    run_arm "cc-$name" CRISPEMBED_CC_DETECT_BENCH "$val" \
        "$BIN" --gpu-backend metal --cc-detect "$IMG/scan_strip.png"
done

# ---------------------------------------------------------------------------
# C. parseq_ocr.cpp / CRISPEMBED_PARSEQ_BENCH — cached-model spot-check
# ---------------------------------------------------------------------------
P=$LIVE/parseq-q8_0.gguf
echo "[C] CRISPEMBED_PARSEQ_BENCH (parseq-q8_0, metal)"
for spell in absent:- zero:0 one:1; do
    name=${spell%%:*}; val=${spell##*:}
    run_arm "parseq-$name" CRISPEMBED_PARSEQ_BENCH "$val" \
        "$BIN" -m "$P" --gpu-backend metal --ocr "$IMG/cc0/arabic_printed_line.png"
done

# ---------------------------------------------------------------------------
# D. scan_cleanup.cpp / CRISPEMBED_SCAN_CLEANUP_BENCH — model-free spot-check
#    whose stdout is a full cleaned PNG (203485 B). The payload is reduced to
#    its sha256 so the evidence directory stays text; byte-identity across the
#    three arms is exactly what the digest proves.
# ---------------------------------------------------------------------------
echo "[D] CRISPEMBED_SCAN_CLEANUP_BENCH (model-free --cleanup-only, stdout as sha256)"
for spell in absent:- zero:0 one:1; do
    name=${spell%%:*}; val=${spell##*:}
    if [ "$val" = "-" ]; then
        env -u CRISPEMBED_SCAN_CLEANUP_BENCH "$BIN" --gpu-backend metal \
            --cleanup-only "$IMG/scan_strip.png" 2> "$OUT/cleanup-$name.err" |
            shasum -a 256 > "$OUT/cleanup-$name.txt"
    else
        env CRISPEMBED_SCAN_CLEANUP_BENCH="$val" "$BIN" --gpu-backend metal \
            --cleanup-only "$IMG/scan_strip.png" 2> "$OUT/cleanup-$name.err" |
            shasum -a 256 > "$OUT/cleanup-$name.txt"
    fi
    echo "  cleanup-$name $(cat "$OUT/cleanup-$name.txt")"
done
