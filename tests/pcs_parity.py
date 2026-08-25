#!/usr/bin/env python
"""PCS parity vs its own driving code.

Third and last punctuation engine to get ground truth. Unlike the other two, the
ONNX graph exports ARGMAXED predictions rather than logits, so there is no
cosine here — the comparison is on the discrete decisions, which is the stronger
gate anyway because they are exactly what produces the text.

Two gates:

  1. ALL FOUR HEADS  per token, exact:
       post  argmax of $PCS_DUMP_LOGITS      vs the blueprint's `post_preds`
       pre   $PCS_DUMP_PRE                   vs `pre_preds`
       seg   $PCS_DUMP_SEG column 0          vs `seg_preds`
       cap   $PCS_DUMP_CAP (16 bits/token)   vs `cap_preds`
  2. DECODED TEXT    exact, and gated — unlike the other two harnesses. PCS
     emits truecasing itself rather than leaving the caller's words alone, so
     there is no deliberate case deviation to excuse: the blueprint's string and
     the runtime's should match character for character.

Gate 1 used to cover post-punc only, which localised nothing: plain q4_k gets
all 67 post-punc decisions right on this corpus and still turns "I'm OK" into
"I'm ok". That is a truecase-head regression, and with only post-punc dumped it
could be seen but not attributed. The three hooks now exist for exactly that.

⚠ `$PCS_DUMP_SEG` has TWO columns and only the first is compared here. Column 0
is `softmax(logits)[boundary] > 0.05`, the ONNX `seg_preds` output; column 1 is
the hard argmax, which the runtime feeds the truecase head as its
"is-sentence-initial" conditioning. The blueprint exports only the former, so
the latter has no reference — it is dumped so a cap mismatch can be traced to
its conditioning rather than to the cap head itself.

⚠ The cap bitstring is per CHARACTER of the token, `▁` included. `▁hello`
therefore reads `1100...`: bit 0 covers the `▁` (ignored downstream) and bit 1
capitalises the `h`. It is not one flag per token.

Usage:
    python tests/pcs_parity.py <build-dir> <pcs.gguf> <ref.txt>
"""
import os
import subprocess
import sys


def parse_ref(path):
    recs, cur, post_labels = [], None, []
    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith("#POST_LABELS "):
            post_labels = line[len("#POST_LABELS "):].split("\t")
        elif line.startswith("#LINE "):
            if cur:
                recs.append(cur)
            cur = {}
        elif line.startswith("#TEXT "):
            cur["text"] = line[6:]
        elif line.startswith("#TOKENS "):
            cur["tokens"] = line[8:].split()
        elif line.startswith("#IDS "):
            cur["ids"] = [int(x) for x in line[5:].split()]
        elif line.startswith("#POST "):
            cur["post"] = [int(x) for x in line[6:].split()]
        elif line.startswith("#PRE "):
            cur["pre"] = [int(x) for x in line[5:].split()]
        elif line.startswith("#SEG "):
            cur["seg"] = line[5:].split()
        elif line.startswith("#CAP "):
            cur["cap"] = line[5:].split()
        elif line.startswith("#PUNC "):
            cur["punc"] = line[6:]
    if cur:
        recs.append(cur)
    return recs, post_labels


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    build_dir, gguf, ref_path = sys.argv[1], sys.argv[2], sys.argv[3]
    ab = next((c for c in (os.path.join(build_dir, "punct-ab"),
                           os.path.join(build_dir, "bin", "punct-ab"))
               if os.path.exists(c)), None)
    if ab is None or not os.path.exists(gguf) or not os.path.exists(ref_path):
        print("SKIP: punct-ab, gguf or reference missing", file=sys.stderr)
        return 0

    recs, post_labels = parse_ref(ref_path)
    corpus = os.path.join(build_dir, "pcs_parity_corpus.txt")
    with open(corpus, "w") as f:
        f.write("\n".join(r["text"] for r in recs) + "\n")

    paths = {k: os.path.join(build_dir, f"pcs_parity_{k}.txt")
             for k in ("logits", "pre", "seg", "cap")}
    for pth in paths.values():
        if os.path.exists(pth):
            os.remove(pth)  # the dumps APPEND; a stale file silently misaligns
    run = subprocess.run([ab, gguf, corpus], capture_output=True, text=True,
                         env=dict(os.environ,
                                  PCS_DUMP_LOGITS=paths["logits"],
                                  PCS_DUMP_PRE=paths["pre"],
                                  PCS_DUMP_SEG=paths["seg"],
                                  PCS_DUMP_CAP=paths["cap"]))
    if run.returncode != 0:
        print(f"SKIP: rc={run.returncode}\n{run.stderr[-500:]}")
        return 0
    ours_punc = [ln for ln in run.stdout.splitlines() if ln.strip()]

    def read_lines(key):
        return [ln.strip() for ln in open(paths[key])] if os.path.exists(paths[key]) else []

    flat = [[float(x) for x in ln.split()] for ln in read_lines("logits") if ln]
    ours_pre = [ln for ln in read_lines("pre") if ln]
    ours_seg = [ln.split()[0] for ln in read_lines("seg") if ln]
    ours_cap = [ln for ln in read_lines("cap") if ln]

    fails = 0

    if not flat:
        print("post preds  : NO LOGITS DUMPED — PCS_DUMP_LOGITS produced nothing")
        fails += 1
    else:
        pos, ok, tot, diffs = 0, 0, 0, []
        aligned = True
        for ri, r in enumerate(recs):
            n = len(r["post"])
            if pos + n > len(flat):
                aligned = False
                break
            for t in range(n):
                row = flat[pos + t]
                ours = max(range(len(row)), key=lambda c: row[c])
                if ours == r["post"][t]:
                    ok += 1
                else:
                    diffs.append((ri, r["tokens"][t], r["post"][t], ours))
                tot += 1
            pos += n
        if not aligned or pos != len(flat):
            print(f"post preds  : MISALIGNED (ref {pos} tokens, ours {len(flat)})")
            fails += 1
        else:
            print(f"post preds  : {ok}/{tot} tokens agree ({100.0*ok/tot:.2f}%)")
            for ri, tokn, refp, ourp in diffs[:8]:
                rl = post_labels[refp] if refp < len(post_labels) else refp
                ol = post_labels[ourp] if ourp < len(post_labels) else ourp
                print(f"  line {ri} {tokn!r}: ref={refp}({rl}) ours={ourp}({ol})")
            if diffs:
                fails += 1

    # pre / seg / cap: flat per-token streams, same order as the logits dump.
    for key, ours, refkey, fmt in (("pre", ours_pre, "pre", str),
                                   ("seg", ours_seg, "seg", str),
                                   ("cap", ours_cap, "cap", str)):
        expect = [fmt(x) for r in recs for x in r.get(refkey, [])]
        if not ours:
            print(f"{key:<12}: NOT DUMPED — PCS_DUMP_{key.upper()} produced nothing")
            fails += 1
            continue
        if len(ours) != len(expect):
            print(f"{key:<12}: MISALIGNED (ref {len(expect)} tokens, ours {len(ours)})")
            fails += 1
            continue
        ok = sum(1 for a, b in zip(ours, expect) if a == b)
        print(f"{key:<12}: {ok}/{len(expect)} tokens agree ({100.0*ok/len(expect):.2f}%)")
        if ok != len(expect):
            shown = 0
            for i, (a, b) in enumerate(zip(ours, expect)):
                if a != b and shown < 5:
                    print(f"  token {i}: ref={b} ours={a}")
                    shown += 1
            fails += 1

    same = sum(1 for i, r in enumerate(recs)
               if i < len(ours_punc) and ours_punc[i] == r.get("punc"))
    print(f"decoded text: {same}/{len(recs)} EXACT (gated — PCS truecases itself, "
          f"so there is no deliberate case deviation to excuse)")
    for i, r in enumerate(recs):
        if i < len(ours_punc) and ours_punc[i] != r.get("punc"):
            print(f"  line {i}:\n    ref  {r.get('punc')}\n    ours {ours_punc[i]}")
    if same != len(recs):
        fails += 1

    print("PASS" if fails == 0 else f"FAIL ({fails} gate(s))")
    return fails


if __name__ == "__main__":
    sys.exit(main())
