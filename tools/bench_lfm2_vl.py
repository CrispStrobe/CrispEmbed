#!/usr/bin/env python3
"""LFM2.5-VL OCR bench: run the CLI over the CC0 fixtures, report CER/WER and stage timings.

    tools/bench_lfm2_vl.py --model ~/models/LFM2.5-VL-3B-Q4_K_M.gguf --label baseline
    LFM2_VL_MULTI_TILE=0 tools/bench_lfm2_vl.py --model ... --label single-tile

Every LFM2_VL_* gate in the environment reaches the child process, so an A/B is
just two invocations with different env and different --label.

Two error rates are reported per fixture, and BOTH belong in any writeup:

  CER / WER                  raw, against ground_truth.json as printed
  fmt-normalised CER / WER   after stripping markdown table scaffolding
                             (pipes, rule rows, ** markers) from both sides

The gap between them is not noise, it is the model choosing a different output
FORMAT. On commons_example_receipt the raw CER is 0.337 and the normalised one
0.092: the receipt is read almost correctly and then emitted as a pipe table,
which the plain-text ground truth does not have. Reporting only the raw number
sends you hunting for a recognition bug that is not there; reporting only the
normalised one hides a real difference in what the engine returns to a caller.

Timing caveat (dev guide rule 5): a wall-clock column from a single pass on a
loaded box is not a measurement. For a perf claim, interleave the arms, take a
median of >= 3, and check `sysctl vm.loadavg` first.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
IMAGES = os.path.join(ROOT, "tests", "regression", "images", "cc0")

DEFAULT_FIXTURES = [
    "commons_example_receipt.png",
    "simple_form.png",
    "receipt_historical.png",
    "german_official_print.jpg",
    "commons_test_ocr_document.jpg",
]

STAGE_PATTERNS = [
    ("preprocess", re.compile(r"(\d+) ms preprocess")),
    ("images", re.compile(r"patches x (\d+) image")),
    ("vision", re.compile(r"vision encoder: \d+ patches, (\d+) ms")),
    ("projector", re.compile(r"projector: \d+ tokens .* (\d+) ms")),
    ("prompt_tokens", re.compile(r"prompt: (\d+) tokens")),
    ("prefill", re.compile(r"prefill: \d+ tokens, (\d+) ms")),
    ("generate", re.compile(r"generate: \d+ tokens, (\d+) ms total")),
    ("total", re.compile(r"total pipeline: (\d+) ms")),
]


def normalise(s):
    s = unicodedata.normalize("NFKC", s)
    for a, b in (("\u2018", "'"), ("\u2019", "'"), ("\u201c", '"'), ("\u201d", '"'),
                 ("\u2014", "-"), ("\u2013", "-"), ("\u00a0", " ")):
        s = s.replace(a, b)
    s = re.sub(r"[ \t]+", " ", s)
    return "\n".join(line.strip() for line in s.strip().split("\n"))


def strip_markdown_tables(s):
    out = []
    for line in normalise(s).split("\n"):
        if re.fullmatch(r"[\|\s:\-]+", line):  # a table rule row
            continue
        line = re.sub(r"\s*\|\s*", " ", line.replace("**", "")).strip()
        if line:
            out.append(line)
    return re.sub(r"[ \t]+", " ", "\n".join(out))


def levenshtein(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def rates(hyp, ref):
    cer = levenshtein(hyp, ref) / max(1, len(ref))
    wer = levenshtein(hyp.split(), ref.split()) / max(1, len(ref.split()))
    return cer, wer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to LFM2.5-VL-3B-*.gguf (the LLM half)")
    ap.add_argument("--bin", default=os.path.join(ROOT, "build", "crispembed"),
                    help="crispembed CLI (pin an ABSOLUTE path: a stale binary from another "
                         "worktree mints false conclusions)")
    ap.add_argument("--label", default="run")
    ap.add_argument("--max-tokens", type=int, default=1024,
                    help="a cap that truncates the page turns CER into a measure of the cap")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=None, help="write the full records here as JSON")
    ap.add_argument("fixtures", nargs="*", default=None)
    args = ap.parse_args()

    fixtures = args.fixtures or DEFAULT_FIXTURES
    truth = json.load(open(os.path.join(IMAGES, "ground_truth.json")))
    refs = {r["file"]: r["text"] for r in truth["records"]}

    env = dict(os.environ, CRISPEMBED_ACCEPT_LFM_LICENSE="1")
    records, sums = [], [0.0, 0.0, 0.0, 0.0]

    for name in fixtures:
        if name not in refs:
            print(f"[{args.label}] {name}: no ground truth, skipping", flush=True)
            continue
        cmd = [args.bin, "-m", args.model, "--ocr", os.path.join(IMAGES, name),
               "--ocr-max-tokens", str(args.max_tokens), "-t", str(args.threads)]
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        wall = time.time() - t0

        stages = {}
        for key, rx in STAGE_PATTERNS:
            m = rx.search(proc.stderr)
            if m:
                stages[key] = int(m.group(1))

        if proc.returncode != 0 or not proc.stdout.strip():
            # A non-zero exit or an empty transcript is a FAILURE, never a fast run.
            print(f"[{args.label}] {name}: FAILED rc={proc.returncode} "
                  f"({wall:.1f}s) {proc.stderr.strip().splitlines()[-3:]}", flush=True)
            records.append(dict(file=name, rc=proc.returncode, wall_s=round(wall, 2)))
            continue

        ref, hyp = normalise(refs[name]), normalise(proc.stdout)
        cer, wer = rates(hyp, ref)
        cer_n, wer_n = rates(strip_markdown_tables(proc.stdout), strip_markdown_tables(refs[name]))
        sums = [sums[0] + cer, sums[1] + wer, sums[2] + cer_n, sums[3] + wer_n]

        records.append(dict(file=name, rc=0, wall_s=round(wall, 2), stages=stages,
                            chars=len(hyp), ref_chars=len(ref),
                            cer=round(cer, 4), wer=round(wer, 4),
                            cer_fmt=round(cer_n, 4), wer_fmt=round(wer_n, 4),
                            text=proc.stdout))
        trunc = " TRUNCATED" if len(hyp) < 0.9 * len(ref) else ""
        print(f"[{args.label}] {name:34s} {wall:7.1f}s chars={len(hyp):5d}/{len(ref):5d}{trunc} "
              f"CER={cer:.3f} WER={wer:.3f} | fmt CER={cer_n:.3f} WER={wer_n:.3f}  {stages}",
              flush=True)

    ok = [r for r in records if r.get("rc") == 0]
    if ok:
        n = len(ok)
        print(f"[{args.label}] {'MEAN over %d fixtures' % n:34s} "
              f"{'':7s}      {'':11s} CER={sums[0]/n:.3f} WER={sums[1]/n:.3f} | "
              f"fmt CER={sums[2]/n:.3f} WER={sums[3]/n:.3f}", flush=True)

    if args.out:
        json.dump(records, open(args.out, "w"), indent=1, ensure_ascii=False)
        print("wrote", args.out)
    return 0 if len(ok) == len(records) else 1


if __name__ == "__main__":
    sys.exit(main())
