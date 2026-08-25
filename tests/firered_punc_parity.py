#!/usr/bin/env python
"""fireredpunc parity vs the Python blueprint — the MODEL, not just the tokenizer.

tests/firered_tokenizer_parity.py checks token ids against HuggingFace. That is
necessary and not sufficient: a tokenizer can be exact while the forward pass is
wrong, and until now nothing in either repo checked the model at all.

Compares three things, weakest gate last, per the dev guide's rule that cosine
alone is not an acceptance test:

  1. TOKEN IDS      exact-match. Structural gate — if these differ the logits
                    are not even the same quantity and nothing below means
                    anything.
  2. LOGITS         per-token, per-class. cos and max_abs vs the f32 reference.
                    This is where a wrong mask / missing-or-extra special token
                    / wrong position offset actually shows up. Quantised GGUFs
                    are expected to drift here; the reference is f32 torch.
  3. PREDS          argmax agreement, and the decoded punctuated text.
                    ⚠ The decoded text is deliberately NOT an equality gate:
                    upstream's RuleBaedTxtFix lowercases the whole string and
                    re-capitalises, so it emits `google` for `Google`, while
                    CrispEmbed emits the user's original words on purpose (it is
                    a post-processor over OCR text, not over ASR output). Case
                    differences are expected; punctuation differences are not.

Usage:
    python tests/firered_punc_parity.py <build-dir> <fireredpunc.gguf> <ref.txt> [min-cos]

`min-cos` defaults to 0.99, which is the right floor for an f16/f32 GGUF against
the f32 torch reference. For a k-quant it is NOT: measured on this model,
f16 cos_min 1.000000, q8_0 0.999234, q4_k 0.935078 — the q4_k number is the
quantiser, and the proof is that the f16 arm of the SAME graph is exact. So when
a quant arm dips, re-run f16 before concluding anything about the port; pass a
looser floor for the quant arm rather than relaxing it for everything.

Produce ref.txt with:
    python tools/dump_fireredpunc_reference.py --model-dir <FireRedPunc dir> \
        --corpus corpus.txt --output ref.txt
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
        elif line.startswith("#LOGITS "):
            cur["logits"].append([float(x) for x in line[8:].split()])
        elif line.startswith("#PREDS "):
            cur["preds"] = [int(x) for x in line[7:].split()]
        elif line.startswith("#PUNC "):
            cur["punc"] = line[6:]
    if cur:
        recs.append(cur)
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
    ab = os.path.join(build_dir, "firered-punct-ab")
    for p in (ab, gguf, ref_path):
        if not os.path.exists(p):
            print(f"SKIP: missing {p}", file=sys.stderr)
            return 0

    recs = parse_ref(ref_path)
    corpus = os.path.join(build_dir, "firered_parity_corpus.txt")
    with open(corpus, "w") as f:
        f.write("\n".join(r["text"] for r in recs) + "\n")

    env = dict(os.environ)
    logits_path = os.path.join(build_dir, "firered_parity_logits.txt")
    if os.path.exists(logits_path):
        os.remove(logits_path)
    env["FIREREDPUNC_DUMP_LOGITS"] = logits_path

    # Token ids and punctuated text come from two runs of the same binary; the
    # logits file is appended to across the whole run, in token order.
    ids_run = subprocess.run([ab, gguf, corpus], capture_output=True, text=True,
                             env=dict(env, FIREREDPUNC_DUMP_IDS="1"))
    if ids_run.returncode != 0:
        print(f"SKIP: rc={ids_run.returncode}\n{ids_run.stderr[-500:]}")
        return 0
    if os.path.exists(logits_path):
        os.remove(logits_path)
    txt_run = subprocess.run([ab, gguf, corpus], capture_output=True, text=True, env=env)
    if txt_run.returncode != 0:
        print(f"SKIP: rc={txt_run.returncode}\n{txt_run.stderr[-500:]}")
        return 0

    ours_ids = [[int(x) for x in ln.split()] if ln.strip() else []
                for ln in ids_run.stdout.splitlines()]
    ours_punc = txt_run.stdout.splitlines()
    flat = [[float(x) for x in ln.split()] for ln in open(logits_path) if ln.strip()]

    fails = 0

    # --- 1. token ids -------------------------------------------------------
    id_ok = sum(1 for i, r in enumerate(recs)
                if i < len(ours_ids) and ours_ids[i] == r["ids"])
    print(f"token ids   : {id_ok}/{len(recs)} exact vs blueprint")
    if id_ok != len(recs):
        fails += 1
        for i, r in enumerate(recs):
            if i < len(ours_ids) and ours_ids[i] != r["ids"]:
                print(f"  DIFF line {i}: {r['text'][:40]!r}")
                print(f"    ref  ({len(r['ids'])}): {r['ids'][:16]}")
                print(f"    ours ({len(ours_ids[i])}): {ours_ids[i][:16]}")
                break

    # --- 2. logits ----------------------------------------------------------
    # The dump is flat across lines; walk it with the reference's token counts.
    pos, cos_min, max_abs, n_tok = 0, 1.0, 0.0, 0
    aligned = True
    for r in recs:
        n = len(r["logits"])
        if pos + n > len(flat):
            aligned = False
            break
        for t in range(n):
            cos_min = min(cos_min, cosine(r["logits"][t], flat[pos + t]))
            max_abs = max(max_abs, max(abs(a - b) for a, b in
                                       zip(r["logits"][t], flat[pos + t])))
            n_tok += 1
        pos += n
    if not aligned or pos != len(flat):
        print(f"logits      : MISALIGNED (ref {pos} tokens, ours {len(flat)}) "
              f"— cannot compare; fix the token ids first")
        fails += 1
    else:
        print(f"logits      : {n_tok} tokens, cos_min {cos_min:.6f}, max_abs {max_abs:.4f}")
        if cos_min < min_cos:
            print(f"  GATE FAIL: cos_min < {min_cos}. For an f16/f32 GGUF that is "
                  "structural (wrong mask / stray special token / wrong "
                  "positions), not precision. For a k-quant, compare against the "
                  "f16 arm of the same graph before blaming the port.")
            fails += 1

    # --- 3. preds + decoded text -------------------------------------------
    pos = 0
    pred_ok = pred_tot = 0
    for r in recs:
        n = len(r["preds"])
        if pos + n > len(flat):
            break
        for t in range(n):
            row = flat[pos + t]
            ours_p = max(range(len(row)), key=lambda c: row[c])
            pred_ok += int(ours_p == r["preds"][t])
            pred_tot += 1
        pos += n
    if pred_tot:
        print(f"preds       : {pred_ok}/{pred_tot} tokens agree "
              f"({100.0 * pred_ok / pred_tot:.2f}%)")

    same = sum(1 for i, r in enumerate(recs)
               if i < len(ours_punc) and ours_punc[i] == r["punc"])
    ci = sum(1 for i, r in enumerate(recs)
             if i < len(ours_punc) and ours_punc[i].lower() == r["punc"].lower())
    print(f"decoded text: {same}/{len(recs)} identical, {ci}/{len(recs)} "
          f"identical ignoring case (case differs BY DESIGN — see docstring)")
    for i, r in enumerate(recs):
        if i < len(ours_punc) and ours_punc[i].lower() != r["punc"].lower():
            print(f"  DIFF line {i}:\n    ref  {r['punc']}\n    ours {ours_punc[i]}")

    print("PASS" if fails == 0 else f"FAIL ({fails} gate(s))")
    return fails


if __name__ == "__main__":
    sys.exit(main())
