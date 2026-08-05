#!/bin/bash
# compare.sh — the acceptance checks over run_gates.sh's arms.
# macOS bash 3.2: no `declare -A`.
set -u
OUT=$(cd "$(dirname "$0")" && pwd)
pass=0; fail=0
chk() { # description expected actual
    if [ "$2" = "$3" ]; then pass=$((pass+1)); echo "PASS $1 ($3)";
    else fail=$((fail+1)); echo "FAIL $1: got '$3' want '$2'"; fi
}
for tag in post cc parseq cleanup; do
    [ -f "$OUT/$tag-absent.err" ] || continue
    a=$(grep -c -- '-bench\]' "$OUT/$tag-absent.err")
    z=$(grep -c -- '-bench\]' "$OUT/$tag-zero.err")
    o=$(grep -c -- '-bench\]' "$OUT/$tag-one.err")
    chk "$tag absent prints no bench lines" 0 "$a"
    chk "$tag =0 prints no bench lines (THE FIX)" 0 "$z"
    [ "$o" -gt 0 ] && chk "$tag =1 prints bench lines" yes yes || chk "$tag =1 prints bench lines" yes no
    cmp -s "$OUT/$tag-absent.txt" "$OUT/$tag-zero.txt" && r=same || r=differ
    chk "$tag stdout absent==zero" same "$r"
    cmp -s "$OUT/$tag-absent.txt" "$OUT/$tag-one.txt" && r=same || r=differ
    chk "$tag stdout absent==one" same "$r"
done
# Pre-fix binary (parent commit, same build tree): =0 DID print the bench lines,
# and stdout is unchanged by this commit.
if [ -f "$OUT/pre-zero.err" ]; then
    p=$(grep -c -- '-bench\]' "$OUT/pre-zero.err")
    [ "$p" -gt 0 ] && chk "pre-fix =0 printed bench lines (the defect)" yes yes || chk "pre-fix =0 printed bench lines (the defect)" yes no
    for a in absent zero one; do
        cmp -s "$OUT/pre-$a.txt" "$OUT/post-$a.txt" && r=same || r=differ
        chk "no-op: pre-$a stdout == post-$a stdout" same "$r"
    done
fi
echo "---- $pass passed, $fail failed"
exit $([ "$fail" -eq 0 ] && echo 0 || echo 1)
