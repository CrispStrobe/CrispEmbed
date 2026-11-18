#!/usr/bin/env python3
"""Compare spacy_smoke output (one JSON object per line) with an original-spaCy
oracle from tools/spacy/ggml_ab_oracle.py and, optionally, a NER oracle from
tests/spacy_ner_ab.py oracle. Exact decoded agreement is required."""
import argparse
import json
from pathlib import Path
import sys

FIELDS = ("text", "tag", "pos", "morph", "lemma", "dep", "head", "sent_start")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--ner-oracle", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    smoke = [json.loads(line) for line in args.smoke.read_text(encoding="utf-8").splitlines() if line.strip()]
    oracle = json.loads(args.oracle.read_text(encoding="utf-8"))["cases"]
    ner = json.loads(args.ner_oracle.read_text(encoding="utf-8"))["cases"] if args.ner_oracle else None
    differences, exact, tokens, entities_matched = [], 0, 0, 0
    if len(smoke) != len(oracle):
        differences.append({"case_count": [len(smoke), len(oracle)]})
    for index, (native, expected) in enumerate(zip(smoke, oracle)):
        ok = native["text"] == expected["text"] and len(native["tokens"]) == len(expected["tokens"])
        for n, e in zip(native["tokens"], expected["tokens"]):
            for f in FIELDS:
                if n[f] != e[f]:
                    ok = False
                    if len(differences) < 50:
                        differences.append({"case": index, "field": f, "oracle": e[f], "native": n[f]})
        if ner is not None:
            got = sorted(map(tuple, native["entities"]))
            want = sorted(map(tuple, ner[index]["entities"]))
            entities_matched += len(set(got) & set(want))
            if got != want:
                ok = False
                differences.append({"case": index, "entities": {"oracle": want, "native": got}})
        exact += ok
        tokens += len(expected["tokens"])
    report = {"cases": len(oracle), "exact_cases": exact, "tokens": tokens, "differences": differences}
    if ner is not None:
        report["entities"] = {"oracle": sum(len(c["entities"]) for c in ner), "matched": entities_matched}
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "differences"}))
    return 0 if exact == len(oracle) == len(smoke) and not differences else 1


if __name__ == "__main__":
    sys.exit(main())
