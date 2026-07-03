#!/usr/bin/env python3
"""Regenerate the bilingual (EN+DE) imatrix calibration/eval corpora from Tatoeba.

Writes tools/kaggle/crispembed-imatrix-quant/{calib_corpus.txt,eval_corpus.txt},
which the harness auto-reads for the embed / sparse / ColBERT A/B (read_corpus()).
The A/B compares a quant vs the full-precision model on the SAME text, so no labels
are needed — only realistic, diverse, permissively-licensed text.

Source : Tatoeba (via mteb/tatoeba-bitext-mining, config deu-eng)
License: CC-BY 2.0 FR (attribution) — see CORPORA_LICENSE.md
Usage  : python tools/gen_eval_corpora.py   (needs: pip install datasets)
"""
from pathlib import Path

OUT = Path(__file__).resolve().parent / "kaggle" / "crispembed-imatrix-quant"
N_CALIB_PAIRS = 15   # -> 30 lines (EN+DE interleaved)
N_EVAL_PAIRS = 8     # -> 16 lines (disjoint)


def main():
    from datasets import load_dataset
    ds = load_dataset("mteb/tatoeba-bitext-mining", "deu-eng", split="test")
    en, de, seen = [], [], set()
    for ex in ds:
        d, e = ex["sentence1"].strip(), ex["sentence2"].strip()
        if 40 <= len(e) <= 130 and 40 <= len(d) <= 140 and e not in seen:
            seen.add(e); en.append(e); de.append(d)
        if len(en) >= N_CALIB_PAIRS + N_EVAL_PAIRS:
            break
    calib = [x for p in zip(en[:N_CALIB_PAIRS], de[:N_CALIB_PAIRS]) for x in p]
    eval_ = [x for p in zip(en[N_CALIB_PAIRS:], de[N_CALIB_PAIRS:]) for x in p]
    (OUT / "calib_corpus.txt").write_text("\n".join(calib) + "\n")
    (OUT / "eval_corpus.txt").write_text("\n".join(eval_) + "\n")
    print(f"wrote {len(calib)} calib + {len(eval_)} eval lines (EN+DE) to {OUT}")


if __name__ == "__main__":
    main()
