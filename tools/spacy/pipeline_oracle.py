#!/usr/bin/env python3
"""Original-spaCy oracle for any installed/unpacked pipeline (developer tooling).

Writes the two files tests/spacy_smoke_compare.py reads: per-token annotations
(text, tag, pos, morph, lemma, dep, head, sent_start) and entities, for each
text of a corpus JSON ({"texts": [...]}). The pipeline runs with its source
defaults (disabled components stay disabled).

    python tools/spacy/pipeline_oracle.py --model en_core_web_sm --corpus c.json \
        --oracle oracle.json --ner-oracle ner-oracle.json
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Package name or unpacked model directory")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--ner-oracle", type=Path, required=True)
    args = parser.parse_args()
    import spacy
    nlp = spacy.load(args.model)
    texts = json.loads(args.corpus.read_text(encoding="utf-8"))["texts"]
    cases, entities = [], []
    for doc in nlp.pipe(texts):
        cases.append({"text": doc.text, "tokens": [
            {"text": t.text, "tag": t.tag_, "pos": t.pos_, "morph": str(t.morph), "lemma": t.lemma_, "dep": t.dep_,
             "head": t.head.i, "sent_start": bool(t.is_sent_start)} for t in doc]})
        entities.append({"text": doc.text, "entities": [[e.start, e.end, e.label_] for e in doc.ents]})
    meta = {"spacy": spacy.__version__, "model": f"{nlp.lang}_{nlp.meta['name']}-{nlp.meta['version']}",
            "pipeline": nlp.pipe_names, "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest()}
    args.oracle.write_text(json.dumps({**meta, "cases": cases}, ensure_ascii=False) + "\n", encoding="utf-8")
    args.ner_oracle.write_text(json.dumps({**meta, "cases": entities}, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(cases)} texts, {sum(len(c['tokens']) for c in cases)} tokens, "
          f"{sum(len(e['entities']) for e in entities)} entities")


if __name__ == "__main__":
    main()
