#!/bin/bash
# UOCR_* value-parse audit — comparison/acceptance block for run_gates.sh's
# artifacts. Judged entirely from the per-arm .txt/.err files in this
# directory. bash 3.2 compatible (no associative arrays).
set -u
WT=/Users/christianstrobele/code/CrispEmbed/.claude/worktrees/feat-uocr-gates
OUT=$(cd "$(dirname "$0")" && pwd)
fail=0

ck() { cmp -s "$1" "$2" && echo "OK   $3" || { echo "DIFF $3"; fail=1; }; }
have() { grep -q "$2" "$1" && echo "OK   $3" || { echo "BAD  $3"; fail=1; }; }
hasnt() { grep -q "$2" "$1" && { echo "BAD  $3"; fail=1; } || echo "OK   $3"; }

# value of one field on the gate-resolution line (exact key match)
field() { grep '\[dbg\] gates:' "$1" | head -1 | tr ' ' '\n' | awk -F= -v k="$2" '$1==k{print $2}'; }

echo "== A. pre-fix controls: =0 USED to engage (parent-commit binary)"
have "$OUT/pre-DBG-0.err"   '\[dbg\]'        "pre-fix UOCR_DBG=0 emitted [dbg] lines (=0 was ON)"
have "$OUT/pre-PD-0.err"    'T=1 pd=1'       "pre-fix UOCR_PD=0 took the persistent-decode path (pd=1)"
have "$OUT/pre-NO_KV-0.err" 'gen=1 n_past=0'    "pre-fix UOCR_NO_KV=0 disabled the KV cache (n_past stays 0 at gen=1)"
[ -s "$OUT/pre-base.txt" ] && echo "OK   pre-fix default run produced text" || { echo "BAD  pre-fix default run empty"; fail=1; }
[ -s "$OUT/pre-PD-0.txt" ] && { echo "BAD  pre-fix UOCR_PD=0 produced text (expected the segfault's empty stdout)"; fail=1; } \
                           || echo "OK   pre-fix UOCR_PD=0 produced EMPTY stdout (it segfaulted — the =0 inversion destroyed output)"
[ -s "$OUT/UOCR_PD-0.txt" ] && echo "OK   post-fix UOCR_PD=0 produced text (the fix)" || { echo "BAD  post-fix UOCR_PD=0 empty"; fail=1; }

echo "== B. post-fix: UOCR_DBG three spellings"
ck    "$OUT/base-dbg.txt"    "$OUT/base.txt" "UOCR_DBG=1 stdout == base"
ck    "$OUT/UOCR_DBG-0.txt"  "$OUT/base.txt" "UOCR_DBG=0 stdout == base"
hasnt "$OUT/base.err"        '\[dbg\]'       "no [dbg] lines when UOCR_DBG absent"
hasnt "$OUT/UOCR_DBG-0.err"  '\[dbg\]'       "no [dbg] lines at UOCR_DBG=0 (the fix)"
have  "$OUT/base-dbg.err"    '\[dbg\] gates:' "gate-resolution line at UOCR_DBG=1"

echo "== C. post-fix: every =0 arm parses off, and is byte-identical to base"
for pair in UOCR_MMAP:mmap UOCR_MOE_CPU:moe_cpu UOCR_SAM_CONV_CPU:sam_conv_cpu \
            UOCR_OPT_GRAPH_LN:opt_graph_ln UOCR_CLIP_DBG:clip_dbg UOCR_FA_F32:fa_f32 \
            UOCR_LMHEAD_CPU:lmhead_cpu UOCR_NO_KV:no_kv \
            UOCR_OPT_FUSED_DECODE:opt_fused_decode UOCR_DECODE_TIMING:decode_timing \
            UOCR_INJECT_VIS:inject_vis UOCR_INJECT_REF:inject_ref UOCR_PD:pd; do
  G=${pair%%:*}; f=${pair##*:}
  v0=$(field "$OUT/$G-0.err" "$f"); v1=$(field "$OUT/$G-1.err" "$f")
  [ "$v0" = 0 ] && echo "OK   $G=0 parses off" || { echo "BAD  $G=0 parses '$v0'"; fail=1; }
  [ "$v1" = 1 ] && echo "OK   $G=1 parses on"  || { echo "BAD  $G=1 parses '$v1'"; fail=1; }
  ck "$OUT/$G-0.txt" "$OUT/base.txt" "$G=0 stdout == base"
done

echo "== D. PD-conditional gates (baseline = pd-base, i.e. UOCR_PD=1 held)"
for pair in UOCR_OPT_PD_F32:opt_pd_f32 UOCR_PD_DBG:pd_dbg UOCR_DECODE_REBUILD:decode_rebuild; do
  G=${pair%%:*}; f=${pair##*:}
  v0=$(field "$OUT/$G-0.err" "$f"); v1=$(field "$OUT/$G-1.err" "$f")
  [ "$v0" = 0 ] && echo "OK   $G=0 parses off" || { echo "BAD  $G=0 parses '$v0'"; fail=1; }
  [ "$v1" = 1 ] && echo "OK   $G=1 parses on"  || { echo "BAD  $G=1 parses '$v1'"; fail=1; }
  ck "$OUT/$G-0.txt" "$OUT/pd-base.txt" "$G=0 stdout == pd-base"
done

echo "== E. =1 engagement proofs (each in that run's OWN stderr)"
have "$OUT/base-dbg.err"              '\[dbg\]'                     "UOCR_DBG=1 emits [dbg] lines"
hasnt "$OUT/UOCR_MOE_CPU-1.err"       'using prestacked MoE experts' "UOCR_MOE_CPU=1 skips the prestacked/Metal MoE path"
have "$OUT/UOCR_PD-1.err"             'T=1 pd=1'                    "UOCR_PD=1 takes the persistent-decode path"
have "$OUT/base-dbg.err"              'T=1 pd=0'                    "absent UOCR_PD -> rebuild path (pd=0)"
have "$OUT/UOCR_DECODE_TIMING-1.err"  '\[rb_timing\]'               "UOCR_DECODE_TIMING=1 prints [rb_timing]"
hasnt "$OUT/UOCR_DECODE_TIMING-0.err" '\[rb_timing\]'               "UOCR_DECODE_TIMING=0 prints no [rb_timing]"
# UOCR_PD_DBG's [pd_dbg] dump is unreachable: it fires at generation steps
# 2..3 and the PD path segfaults IN the gen=2 compute (pre-existing, also on
# the parent-commit binary). Prove the gate on its rebuild-path twin instead —
# same gate, same file, [rb_dbg] dumps at the same generation steps.
have "$OUT/UOCR_PD_DBG-rb-1.err"      '\[rb_dbg\]'                  "UOCR_PD_DBG=1 prints [rb_dbg] per-layer dumps (rebuild path)"
hasnt "$OUT/UOCR_PD_DBG-rb-0.err"     '\[rb_dbg\]'                  "UOCR_PD_DBG=0 prints none (rebuild path)"
ck "$OUT/UOCR_PD_DBG-rb-0.txt" "$OUT/base.txt" "UOCR_PD_DBG=0 (rebuild) stdout == base"
hasnt "$OUT/UOCR_PD_DBG-0.err"        '\[pd_dbg\]'                  "UOCR_PD_DBG=0 prints no [pd_dbg]"
have "$OUT/UOCR_DECODE_REBUILD-1.err" 'T=1 pd=0'                    "UOCR_DECODE_REBUILD=1 disables PD even with UOCR_PD=1"
have "$OUT/pd-base.err"               'T=1 pd=1'                    "pd-base (UOCR_PD=1) uses PD"
# Fused decode sets did_pd=true, which skips the `if (!did_pd)` rebuild block —
# and that block owns the [rb_timing] print. UOCR_DECODE_TIMING=1 reports which
# path ran.
hasnt "$OUT/UOCR_OPT_FUSED_DECODE-t-1.err" '\[rb_timing\]'          "UOCR_OPT_FUSED_DECODE=1 took the fused path (rebuild block skipped)"
have "$OUT/UOCR_OPT_FUSED_DECODE-t-0.err"  '\[rb_timing\]'          "UOCR_OPT_FUSED_DECODE=0 took the rebuild path (the fix)"
ck "$OUT/UOCR_OPT_FUSED_DECODE-t-1.txt" "$OUT/base.txt" "UOCR_OPT_FUSED_DECODE=1 stdout == base"
ck "$OUT/UOCR_OPT_FUSED_DECODE-t-0.txt" "$OUT/base.txt" "UOCR_OPT_FUSED_DECODE=0 stdout == base"
have "$OUT/UOCR_CLIP_DBG-1.err"       '\[dbg\] clip input'          "UOCR_CLIP_DBG=1 prints CLIP debug dumps"
hasnt "$OUT/UOCR_CLIP_DBG-0.err"      '\[dbg\] clip input'          "UOCR_CLIP_DBG=0 prints none"
have "$OUT/UOCR_NO_KV-1.err"          'gen=1 n_past=0'              "UOCR_NO_KV=1 disables the KV cache (n_past stays 0)"
hasnt "$OUT/UOCR_NO_KV-0.err"         'gen=1 n_past=0'              "UOCR_NO_KV=0 keeps the KV cache (the fix)"
hasnt "$OUT/UOCR_MOE_CPU-0.err"       '^unlimited_ocr: MoE'         "UOCR_MOE_CPU=0 keeps the Metal MoE path (the fix)"
have "$OUT/UOCR_MOE_CPU-0.err"        'using prestacked MoE experts' "UOCR_MOE_CPU=0 still uses prestacked/Metal MoE"

echo "== F. gold fixture: base decodes the manifest gold text"
/usr/bin/python3 - "$OUT/base.txt" "$WT/tests/regression/manifest.json" <<'PY'
import json, sys, re
out = open(sys.argv[1]).read()
man = json.load(open(sys.argv[2]))
e = [m for m in man["models"] if m["name"] == "unlimited-ocr-stacked"][0]
gold = e["expected_text"]; maxcer = e["match"]["max_cer"]
def norm(s): return re.sub(r"\s+", " ", s).strip()
a, b = norm(gold), norm(out)
prev = list(range(len(b) + 1))
for i, ca in enumerate(a, 1):
    cur = [i]
    for j, cb in enumerate(b, 1):
        cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
    prev = cur
cer = prev[-1] / max(1, len(a))
print("     gold : %r" % gold)
print("     base : %r" % out)
print("%s   base CER=%.4f (max_cer=%.2f)" % ("OK  " if cer <= maxcer else "BAD ", cer, maxcer))
sys.exit(0 if cer <= maxcer else 1)
PY
[ $? -eq 0 ] || fail=1

echo "== G. no-op vs main: default-env run identical pre-fix and post-fix"
ck "$OUT/pre-base.txt" "$OUT/base.txt" "default-env stdout pre-fix == post-fix"

echo "== H. output differences recorded verbatim (informational, not pass/fail)"
for A in UOCR_MMAP UOCR_MOE_CPU UOCR_SAM_CONV_CPU UOCR_OPT_GRAPH_LN UOCR_CLIP_DBG \
         UOCR_FA_F32 UOCR_LMHEAD_CPU UOCR_NO_KV UOCR_OPT_FUSED_DECODE \
         UOCR_DECODE_TIMING UOCR_INJECT_VIS UOCR_INJECT_REF UOCR_PD; do
  if cmp -s "$OUT/$A-1.txt" "$OUT/base.txt"; then echo "SAME $A=1 stdout == base"
  else echo "NOTE $A=1 stdout DIFFERS from base:"; diff "$OUT/base.txt" "$OUT/$A-1.txt" | sed 's/^/       /'; fi
done
for A in UOCR_OPT_PD_F32 UOCR_PD_DBG UOCR_DECODE_REBUILD; do
  if cmp -s "$OUT/$A-1.txt" "$OUT/pd-base.txt"; then echo "SAME $A=1 stdout == pd-base"
  else echo "NOTE $A=1 stdout DIFFERS from pd-base:"; diff "$OUT/pd-base.txt" "$OUT/$A-1.txt" | sed 's/^/       /'; fi
done
if cmp -s "$OUT/pd-base.txt" "$OUT/base.txt"; then echo "SAME pd-base stdout == base"
else echo "NOTE pd-base (UOCR_PD=1) stdout DIFFERS from base:"; diff "$OUT/base.txt" "$OUT/pd-base.txt" | sed 's/^/       /'; fi

echo "GATES_DONE fail=$fail"
