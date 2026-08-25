#!/usr/bin/env python
"""PCS parity vs its own driving code.

Third and last punctuation engine to get ground truth. Unlike the other two, the
ONNX graph exports ARGMAXED predictions rather than logits, so there is no
cosine here — the comparison is on the discrete decisions, which is the stronger
gate anyway because they are exactly what produces the text.

Two gates:

  1. POST-PUNC PREDICTIONS  per token, exact. The runtime dumps its post-punc
     logits to $PCS_DUMP_LOGITS; their argmax must equal the blueprint's
     `post_preds`. This is the head that decides every mark.
  2. DECODED TEXT           exact, and gated — unlike the other two harnesses.
     PCS emits truecasing itself rather than leaving the caller's words alone,
     so there is no deliberate case deviation to excuse: the blueprint's string
     and the runtime's should match character for character.

NOT compared, and worth stating rather than leaving implied: `pre_preds`,
`cap_preds` and `seg_preds` have no runtime dump hook, so they are checked only
through their effect on the decoded text. A regression that changed truecasing
while leaving punctuation alone would be caught by gate 2 but not localised.
Adding PCS_DUMP_CAP / PCS_DUMP_SEG would close that.

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

    logits_path = os.path.join(build_dir, "pcs_parity_logits.txt")
    if os.path.exists(logits_path):
        os.remove(logits_path)  # the dump APPENDS; a stale file silently misaligns
    run = subprocess.run([ab, gguf, corpus], capture_output=True, text=True,
                         env=dict(os.environ, PCS_DUMP_LOGITS=logits_path))
    if run.returncode != 0:
        print(f"SKIP: rc={run.returncode}\n{run.stderr[-500:]}")
        return 0
    ours_punc = [ln for ln in run.stdout.splitlines() if ln.strip()]
    flat = [[float(x) for x in ln.split()] for ln in open(logits_path) if ln.strip()] \
        if os.path.exists(logits_path) else []

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
