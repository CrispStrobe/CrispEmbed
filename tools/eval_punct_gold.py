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

    python tools/eval_punct_gold.py <build-dir> <model.gguf> [<model.gguf> ...]
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


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    build_dir, models = sys.argv[1], sys.argv[2:]
    ab = next((c for c in (os.path.join(build_dir, "punct-ab"),
                           os.path.join(build_dir, "bin", "punct-ab")) if os.path.exists(c)), None)
    if ab is None:
        print("SKIP: punct-ab not found", file=sys.stderr)
        return 0

    gold = load_gold()
    inputs, refs, names = [], [], []
    for name, text in gold:
        pairs = split_words(text)
        if len(pairs) < 20:
            continue
        # Chunk to keep each line inside the model's window with room to spare.
        for i in range(0, len(pairs), 120):
            chunk = pairs[i:i + 120]
            inputs.append(" ".join(w.lower() for w, _ in chunk))
            refs.append(chunk)
            names.append(f"{name}[{i}]")
    print(f"gold: {len(refs)} segments, {sum(len(c) for c in refs)} words "
          f"({sum(1 for c in refs for _, m in c if m)} marks)\n")

    print(f"{'model':<34}{'P':>8}{'R':>8}{'F1':>8}{'exact':>8}   dropped")
    for gguf in models:
        outs = run_model(ab, gguf, inputs)
        if outs is None or len(outs) != len(inputs):
            print(f"{os.path.basename(gguf):<34}  RUN FAILED / line-count mismatch")
            continue
        tp = fp = fn = exact = total = dropped = 0
        for ref, out in zip(refs, outs):
            hyp = split_words(out)
            if len(hyp) != len(ref):
                dropped += 1
                continue
            for (_, gm), (_, hm) in zip(ref, hyp):
                total += 1
                if gm and hm and gm == hm:
                    tp += 1; exact += 1
                elif gm and hm:      # wrong mark: a false positive AND a miss
                    fp += 1; fn += 1
                elif hm:
                    fp += 1
                elif gm:
                    fn += 1
                else:
                    exact += 1
        p = tp / (tp + fp) if tp + fp else 0.0
        r_ = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r_ / (p + r_) if p + r_ else 0.0
        acc = exact / total if total else 0.0
        print(f"{os.path.basename(gguf):<34}{p:>8.3f}{r_:>8.3f}{f1:>8.3f}{acc:>8.3f}   {dropped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
