#!/usr/bin/env python
"""Score punctuation restoration against GOLD text, not against another model.

The blueprint harnesses answer "does the port reproduce the reference?". They
cannot answer "which artifact produces better punctuation?", because the
reference is one of the candidates. That needs text whose punctuation is known.

Method: take genuinely punctuated prose, strip the marks and the casing to make
the input, restore it, and score the restored marks against the originals. The
gold here is the manual transcription of the CC0 OCR fixtures
(tests/regression/images/cc0/ground_truth.json) — real printed prose, not
sentences written to make a point.

Scored per mark class, and only over the classes these models can emit
(. , ? : -). Gold marks outside that set are stripped from both sides rather
than counted as misses, or the score would measure the label inventory instead
of the model.

⚠ Word alignment is asserted, not assumed. Punctuation restoration must not
change the words; if the counts differ the segment is dropped and reported,
because a silent misalignment turns into a meaningless F1.

Two metrics, because they fail differently:

  EXACT-MARK F1     did the model emit the same mark the editor did? Harsh, and
                    partly a style contest — a period where the author wrote a
                    colon scores as both a miss and a false positive, though a
                    reader would call it fine.
  BOUNDARY F1       did the model end a sentence where the editor did (any of
                    . ?)? Far less style-dependent: whether a sentence stops is
                    much closer to determined than which mark stops it. When the
                    two metrics disagree, believe this one.

    python tools/eval_punct_gold.py <build-dir> <model.gguf> [<model.gguf> ...]
    python tools/eval_punct_gold.py --gold <gold.json> <build-dir> <model> ...
"""
import json
import os
import re
import subprocess
import sys

MARKS = ".,?:-"
GOLD_JSON = "tests/regression/images/cc0/ground_truth.json"
# Prose only. Receipts and form dumps are line-oriented label soup; punctuation
# restoration on them measures nothing.
PROSE = {"commons_test_ocr_document.jpg", "german_official_print.jpg"}


def load_gold():
    d = json.load(open(GOLD_JSON))
    out = []
    for r in d["records"]:
        if r["name"] not in PROSE:
            continue
        text = re.sub(r"\s+", " ", r["text"]).strip()
        out.append((r["name"], text))
    return out


def split_words(text):
    """-> [(word_without_trailing_mark, mark_or_empty)] over the scored classes."""
    pairs = []
    for tok in text.split():
        # Strip anything that is not a scored mark from the end, then read the mark.
        core = tok
        mark = ""
        while core and core[-1] in "\"'”’»)]":       # closers ride outside the mark
            core = core[:-1]
        if core and core[-1] in MARKS:
            mark, core = core[-1], core[:-1]
        core = re.sub(r"[^\wÀ-￿'’-]+$", "", core)
        if core:
            pairs.append((core, mark))
    return pairs


def run_model(ab, gguf, lines):
    corpus = "/tmp/punct_gold_corpus.txt"
    with open(corpus, "w") as f:
        f.write("\n".join(lines) + "\n")
    r = subprocess.run([ab, gguf, corpus], capture_output=True, text=True)
    if r.returncode != 0:
        return None
    return [ln for ln in r.stdout.splitlines() if ln.strip()]


def load_gold_json(path):
    d = json.load(open(path, encoding="utf-8"))
    return [(s.get("src", "?"), s["text"]) for s in d["sentences"]]


def main():
    argv = sys.argv[1:]
    gold_json = None
    if argv and argv[0] == "--gold":
        gold_json, argv = argv[1], argv[2:]
    if len(argv) < 2:
        print(__doc__)
        return 2
    build_dir, models = argv[0], argv[1:]
    ab = next((c for c in (os.path.join(build_dir, "punct-ab"),
                           os.path.join(build_dir, "bin", "punct-ab")) if os.path.exists(c)), None)
    if ab is None:
        print("SKIP: punct-ab not found", file=sys.stderr)
        return 0

    gold = load_gold_json(gold_json) if gold_json else load_gold()
    inputs, refs, names = [], [], []
    for name, text in gold:
        pairs = split_words(text)
        if gold_json:
            # One sentence per line. Each is already inside the window, and
            # keeping them separate means a bad sentence cannot drag its
            # neighbours' alignment down with it.
            if len(pairs) < 4:
                continue
            inputs.append(" ".join(w.lower() for w, _ in pairs))
            refs.append(pairs)
            names.append(name)
        else:
            if len(pairs) < 20:
                continue
            for i in range(0, len(pairs), 120):
                chunk = pairs[i:i + 120]
                inputs.append(" ".join(w.lower() for w, _ in chunk))
                refs.append(chunk)
                names.append(f"{name}[{i}]")
    print(f"gold: {len(refs)} segments, {sum(len(c) for c in refs)} words "
          f"({sum(1 for c in refs for _, m in c if m)} marks)\n")

    scores = {}
    print(f"{'model':<34}{'markP':>7}{'markR':>7}{'markF1':>8}"
          f"{'bndP':>7}{'bndR':>7}{'bndF1':>8}{'exact':>8}  drop")
    for gguf in models:
        outs = run_model(ab, gguf, inputs)
        if outs is None or len(outs) != len(inputs):
            print(f"{os.path.basename(gguf):<34}  RUN FAILED / line-count mismatch")
            continue
        tp = fp = fn = exact = total = dropped = 0
        btp = bfp = bfn = 0
        per_sentence = []   # (tp, fp, fn) per sentence, for the paired bootstrap
        END = ".?"
        for ref, out in zip(refs, outs):
            hyp = split_words(out)
            if len(hyp) != len(ref):
                dropped += 1
                continue
            s_tp = s_fp = s_fn = 0
            for (_, gm), (_, hm) in zip(ref, hyp):
                total += 1
                if gm and hm and gm == hm:
                    tp += 1; s_tp += 1; exact += 1
                elif gm and hm:      # wrong mark: a false positive AND a miss
                    fp += 1; fn += 1; s_fp += 1; s_fn += 1
                elif hm:
                    fp += 1; s_fp += 1
                elif gm:
                    fn += 1; s_fn += 1
                else:
                    exact += 1
                # Boundary: any sentence-ending mark counts as the same event,
                # so `.` vs `?` is not punished and `.` vs `,` still is.
                gb, hb = gm in END and gm != "", hm in END and hm != ""
                if gb and hb:
                    btp += 1
                elif hb:
                    bfp += 1
                elif gb:
                    bfn += 1
            per_sentence.append((s_tp, s_fp, s_fn))
        scores[os.path.basename(gguf)] = per_sentence

        def prf(t, f, n):
            p_ = t / (t + f) if t + f else 0.0
            r__ = t / (t + n) if t + n else 0.0
            return p_, r__, (2 * p_ * r__ / (p_ + r__) if p_ + r__ else 0.0)

        p, r_, f1 = prf(tp, fp, fn)
        bp, br, bf1 = prf(btp, bfp, bfn)
        acc = exact / total if total else 0.0
        print(f"{os.path.basename(gguf):<34}{p:>7.3f}{r_:>7.3f}{f1:>8.3f}"
              f"{bp:>7.3f}{br:>7.3f}{bf1:>8.3f}{acc:>8.3f}  {dropped}")

    # Paired bootstrap over SENTENCES. Paired because every model saw the same
    # sentences: comparing two independent confidence intervals would throw away
    # that pairing and overstate the uncertainty. Resampling sentences (not
    # marks) is what respects the fact that marks within a sentence are not
    # independent draws.
    if len(scores) >= 2:
        import random
        names_ = list(scores)
        n = min(len(v) for v in scores.values())
        print("\n  paired bootstrap, 2000 resamples of the 120 sentences "
              "(markF1 difference, 95% interval):")
        for i in range(len(names_)):
            for j in range(i + 1, len(names_)):
                a, b = scores[names_[i]][:n], scores[names_[j]][:n]
                diffs = []
                rnd = random.Random(20260825)
                for _ in range(2000):
                    idx = [rnd.randrange(n) for _ in range(n)]
                    def f1_of(v):
                        t = sum(v[k][0] for k in idx)
                        f = sum(v[k][1] for k in idx)
                        m = sum(v[k][2] for k in idx)
                        p_ = t / (t + f) if t + f else 0.0
                        r__ = t / (t + m) if t + m else 0.0
                        return 2 * p_ * r__ / (p_ + r__) if p_ + r__ else 0.0
                    diffs.append(f1_of(a) - f1_of(b))
                diffs.sort()
                lo, hi = diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))]
                verdict = "SIGNIFICANT" if (lo > 0 or hi < 0) else "not distinguishable"
                print(f"    {names_[i][:26]:<27} - {names_[j][:26]:<27} "
                      f"{sum(diffs)/len(diffs):+.4f}  [{lo:+.4f}, {hi:+.4f}]  {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
