#!/usr/bin/env python3
"""Capture original spaCy annotations for the engine/precision A/B corpus.

Developer tooling. Runs the authenticated de_core_news_sm 3.8.0 wheel through
the original spaCy pipeline (tok2vec, tagger, morphologizer, parser,
lemmatizer, attribute_ruler) and writes per-token decoded annotations. The
native harness tests/spacy_ggml_ab.py compares every native engine/precision
against this file, never against another native run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "models"))
import spacy_bundle  # noqa: E402

FIELDS = ("tag", "pos", "morph", "lemma", "dep", "head", "sent_start")


def annotate(doc):
    return [{"text": t.text, "tag": t.tag_, "pos": t.pos_, "morph": str(t.morph), "lemma": t.lemma_,
             "dep": t.dep_, "head": t.head.i, "sent_start": bool(t.is_sent_start)} for t in doc]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import spacy
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as directory:
        nlp = spacy_bundle.load_verified_model(args.wheel, Path(directory))
        cases = [{"text": text, "tokens": annotate(nlp(text))} for text in corpus["texts"]]
    result = {"schema": "crispembed.spacy.ggml_ab_oracle.v1", "spacy": spacy.__version__,
              "wheel_sha256": hashlib.sha256(args.wheel.read_bytes()).hexdigest(),
              "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
              "fields": list(FIELDS), "cases": cases}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{len(cases)} cases, {sum(len(c['tokens']) for c in cases)} tokens")


if __name__ == "__main__":
    main()
