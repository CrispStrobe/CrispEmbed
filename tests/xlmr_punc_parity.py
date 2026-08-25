#!/usr/bin/env python
"""XLM-R punctuation parity vs the HuggingFace blueprint.

Companion to tests/firered_punc_parity.py, for the OTHER half of
fireredpunc.cpp: the `tokenizer_type == "sentencepiece"` models
(kredor/punctuate-all, the fullstop-punc family). That half had no ground truth
at all — its empty-output bug was verified only by "the two arms now agree",
which shows consistency, not correctness.

Same layering, weakest gate last:

  1. TOKEN IDS  exact. If these differ nothing below is even the same quantity.
  2. LOGITS     per token, per class: cos and max_abs vs the f32 torch reference.
  3. PREDS      argmax agreement, then decoded text — REPORTED, not gated,
                because the runtime deliberately re-emits the user's original
                words (it post-processes OCR text) while the blueprint lowercases
                and works from token surface forms.

⚠ The reference has logits for `<s>` and `</s>`; the runtime reports only the
real tokens. The `#SPECIAL` mask says which is which, and they are dropped here
rather than inferred from the count — guessing is how a harness manufactures an
off-by-one and calls it a first divergence.

Usage:
    python tests/xlmr_punc_parity.py <build-dir> <model.gguf> <ref.txt> [min-cos]
"""
import math
import os
import subprocess
import sys


def parse_ref(path):
    recs, cur = [], None
    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith("#LINE "):
            if cur:
                recs.append(cur)
            cur = {"logits": []}
        elif line.startswith("#TEXT "):
            cur["text"] = line[6:]
        elif line.startswith("#TOKENS "):
            cur["tokens"] = line[8:].split()
        elif line.startswith("#IDS "):
            cur["ids"] = [int(x) for x in line[5:].split()]
        elif line.startswith("#SPECIAL "):
            cur["special"] = [int(x) for x in line[9:].split()]
        elif line.startswith("#LOGITS "):
            cur["logits"].append([float(x) for x in line[8:].split()])
        elif line.startswith("#PREDS "):
            cur["preds"] = [int(x) for x in line[7:].split()]
        elif line.startswith("#PUNC "):
            cur["punc"] = line[6:]
    if cur:
        recs.append(cur)
    # Drop the special positions so both sides describe the same tokens.
    for r in recs:
        keep = [i for i, sp in enumerate(r["special"]) if not sp]
        r["ids"] = [r["ids"][i] for i in keep]
        r["tokens"] = [r["tokens"][i] for i in keep]
        r["logits"] = [r["logits"][i] for i in keep]
        r["preds"] = [r["preds"][i] for i in keep]
    return recs


def cosine(a, b):
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(y * y for y in b))
    return num / (da * db) if da and db else 0.0


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    build_dir, gguf, ref_path = sys.argv[1], sys.argv[2], sys.argv[3]
    min_cos = float(sys.argv[4]) if len(sys.argv) > 4 else 0.99
    ab = next((c for c in (os.path.join(build_dir, "firered-punct-ab"),
                           os.path.join(build_dir, "bin", "firered-punct-ab"))
               if os.path.exists(c)), None)
    if ab is None or not os.path.exists(gguf) or not os.path.exists(ref_path):
        print("SKIP: binary, gguf or reference missing", file=sys.stderr)
        return 0

    recs = parse_ref(ref_path)
    corpus = os.path.join(build_dir, "xlmr_parity_corpus.txt")
    with open(corpus, "w") as f:
        f.write("\n".join(r["text"] for r in recs) + "\n")

    logits_path = os.path.join(build_dir, "xlmr_parity_logits.txt")
    env = dict(os.environ, FIREREDPUNC_DUMP_LOGITS=logits_path)

    if os.path.exists(logits_path):
        os.remove(logits_path)
    ids_run = subprocess.run([ab, gguf, corpus], capture_output=True, text=True,
                             env=dict(env, FIREREDPUNC_DUMP_IDS="1"))
    if os.path.exists(logits_path):
        os.remove(logits_path)
    txt_run = subprocess.run([ab, gguf, corpus], capture_output=True, text=True, env=env)
    for r in (ids_run, txt_run):
        if r.returncode != 0:
            print(f"SKIP: rc={r.returncode}\n{r.stderr[-500:]}")
            return 0

    ours_ids = [[int(x) for x in ln.split()] if ln.strip() else []
                for ln in ids_run.stdout.splitlines()]
    ours_punc = txt_run.stdout.splitlines()
    flat = [[float(x) for x in ln.split()] for ln in open(logits_path) if ln.strip()]

    fails = 0

    id_ok = sum(1 for i, r in enumerate(recs)
                if i < len(ours_ids) and ours_ids[i] == r["ids"])
    print(f"token ids   : {id_ok}/{len(recs)} exact vs blueprint")
    if id_ok != len(recs):
        fails += 1
        for i, r in enumerate(recs):
            if i < len(ours_ids) and ours_ids[i] != r["ids"]:
                print(f"  DIFF line {i}: {r['text'][:40]!r}")
                print(f"    ref  ({len(r['ids'])}): {r['ids'][:14]}")
                print(f"    ours ({len(ours_ids[i])}): {ours_ids[i][:14]}")
                break

    pos, cos_min, max_abs, n_tok, worst = 0, 1.0, 0.0, 0, None
    aligned = True
    for ri, r in enumerate(recs):
        n = len(r["logits"])
        if pos + n > len(flat):
            aligned = False
            break
        for t in range(n):
            c = cosine(r["logits"][t], flat[pos + t])
            if c < cos_min:
                cos_min, worst = c, (ri, t, r["tokens"][t])
            max_abs = max(max_abs, max(abs(x - y) for x, y in zip(r["logits"][t], flat[pos + t])))
            n_tok += 1
        pos += n
    if not aligned or pos != len(flat):
        print(f"logits      : MISALIGNED (ref {pos} tokens, ours {len(flat)}) — fix token ids first")
        fails += 1
    else:
        print(f"logits      : {n_tok} tokens, cos_min {cos_min:.6f}, max_abs {max_abs:.4f}"
              + (f"   worst at line {worst[0]} token {worst[1]} {worst[2]!r}" if worst else ""))
        if cos_min < min_cos:
            print(f"  GATE FAIL: cos_min < {min_cos}. Structural at f16/f32; for a "
                  "k-quant compare the f16 arm of the same graph first.")
            fails += 1

    pos, pred_ok, pred_tot, mismatches = 0, 0, 0, []
    for ri, r in enumerate(recs):
        n = len(r["preds"])
        if pos + n > len(flat):
            break
        for t in range(n):
            row = flat[pos + t]
            ours_p = max(range(len(row)), key=lambda c: row[c])
            if ours_p == r["preds"][t]:
                pred_ok += 1
            else:
                mismatches.append((ri, r["tokens"][t], r["preds"][t], ours_p,
                                   r["logits"][t], row))
            pred_tot += 1
        pos += n
    if pred_tot:
        print(f"preds       : {pred_ok}/{pred_tot} tokens agree ({100.0*pred_ok/pred_tot:.2f}%)")
    for m in mismatches[:6]:
        ri, tokn, refp, ourp, rl, ol = m
        # Print the margin. A near-tie is quantisation; a wide gap is a bug, and
        # the difference is the whole diagnosis.
        print(f"  line {ri} {tokn!r}: ref={refp} ours={ourp}  "
              f"ref margin {rl[refp]-rl[ourp]:+.3f}  ours margin {ol[ourp]-ol[refp]:+.3f}")

    same = sum(1 for i, r in enumerate(recs)
               if i < len(ours_punc) and ours_punc[i].lower() == r["punc"].lower())
    print(f"decoded text: {same}/{len(recs)} match ignoring case "
          f"(case differs BY DESIGN — the runtime re-emits the user's words)")
    for i, r in enumerate(recs):
        if i < len(ours_punc) and ours_punc[i].lower() != r["punc"].lower():
            print(f"  line {i}:\n    ref  {r['punc']}\n    ours {ours_punc[i]}")

    print("PASS" if fails == 0 else f"FAIL ({fails} gate(s))")
    return fails


if __name__ == "__main__":
    sys.exit(main())
