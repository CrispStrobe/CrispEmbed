#!/usr/bin/env python3
"""A/B one native spaCy engine/precision against the original spaCy oracle.

Each invocation measures exactly one configuration (environment + GGUF) in a
fresh process and reports decoded-output agreement with the oracle written by
tools/spacy/ggml_ab_oracle.py, numerical drift against an optional reference
dump of another native configuration, and warm timing. Decoded agreement is
the quality measure; numerical drift only localizes where a difference starts.

    python3 tests/spacy_ggml_ab.py --library build-spacy/libcrispembed-spacy.dylib \
        --bundle BUNDLE --oracle oracle.json --label scalar-f32 --report r.json \
        [--model q8.gguf] [--dump d.npz] [--reference-dump base.npz] [--env K=V ...]
"""
from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_spacy_text_api import NativeAPI, TRACE  # noqa: E402

FIELDS = ("tag", "pos", "morph", "lemma", "dep", "head", "sent_start")
DUMPED = ("tok2vec.output", "tagger.scores", "morphologizer.scores", "lemmatizer.scores", "parser.lower_cache")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(api, context, text, dump=None):
    stages = {}

    def callback(_user, name, values, shape, rank):
        name = name.decode("utf-8")
        if name in DUMPED:
            dims = [shape[i] for i in range(rank)]
            count = 1
            for d in dims:
                count *= d
            stages[name] = (dims, [values[i] for i in range(count)])
        return 0

    trace = TRACE(callback) if dump is not None else None
    status, document, error = api.call(context, text.encode("utf-8"), analyze=True, trace=trace)
    if status != 0:
        raise RuntimeError(f"analyze failed ({status}): {error}")
    try:
        tokens = api.tokens(document)
    finally:
        api.lib.crispembed_spacy_document_free(document)
    result = [{"text": t["text"].decode("utf-8"), "tag": t["tag"].decode("utf-8"), "pos": t["pos"].decode("utf-8"),
               "morph": t["morph"].decode("utf-8"), "lemma": t["lemma"].decode("utf-8"),
               "dep": t["dependency"].decode("utf-8"), "head": t["head"], "sent_start": bool(t["flags"] & 4)}
              for t in tokens]
    if dump is not None:
        dump.append(stages)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--model", type=Path, help="Alternative GGUF (default: bundle model.gguf)")
    parser.add_argument("--dump", type=Path, help="Write traced outputs (.npz) for later drift comparison")
    parser.add_argument("--reference-dump", type=Path, help="Compare traced outputs against this .npz")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--env", action="append", default=[], help="KEY=VALUE set before loading")
    args = parser.parse_args()
    for item in args.env:
        key, value = item.split("=", 1)
        os.environ[key] = value
    os.environ.setdefault("CRISPEMBED_SPACY", "1")
    oracle = json.loads(args.oracle.read_text(encoding="utf-8"))
    manifest = json.loads((args.bundle / "manifest.json").read_text(encoding="utf-8"))
    api = NativeAPI(args.library, args.bundle, manifest)
    model = args.model or args.bundle / "model.gguf"
    error = C.create_string_buffer(1024)
    load_start = time.perf_counter()
    context = api.lib.crispembed_spacy_create_with_resources(
        os.fsencode(model), sha256(model).encode(), os.fsencode(args.bundle / "resources.json"),
        manifest["files"]["resources.json"]["sha256"].encode(), error, len(error))
    load_ms = (time.perf_counter() - load_start) * 1e3
    if not context:
        report = {"label": args.label, "admitted": False, "error": error.value.decode("utf-8", "replace")}
        args.report.write_text(json.dumps(report, indent=1) + "\n")
        print(json.dumps(report))
        return 2
    try:
        dump = [] if (args.dump or args.reference_dump) else None
        agree = {f: 0 for f in FIELDS}
        tokens_total = token_mismatch_cases = exact_cases = 0
        differences = []
        for index, case in enumerate(oracle["cases"]):
            native = analyze(api, context, case["text"], dump)
            expected = case["tokens"]
            if [t["text"] for t in native] != [t["text"] for t in expected]:
                token_mismatch_cases += 1
                differences.append({"case": index, "tokenization": True})
                continue
            tokens_total += len(expected)
            case_exact = True
            for i, (n, e) in enumerate(zip(native, expected)):
                for f in FIELDS:
                    if n[f] == e[f]:
                        agree[f] += 1
                    else:
                        case_exact = False
                        if len(differences) < 200:
                            differences.append({"case": index, "token": i, "text": e["text"], "field": f,
                                                "oracle": e[f], "native": n[f]})
            exact_cases += case_exact
        # Warm timing: per-case analyze and one long concatenated document.
        texts = [c["text"] for c in oracle["cases"]]
        long_text = " ".join(texts)
        analyze(api, context, long_text)
        corpus_ms, long_ms = [], []
        for _ in range(args.repeats):
            start = time.perf_counter()
            for text in texts:
                analyze(api, context, text)
            corpus_ms.append((time.perf_counter() - start) * 1e3)
            start = time.perf_counter()
            long_tokens = len(analyze(api, context, long_text))
            long_ms.append((time.perf_counter() - start) * 1e3)
    finally:
        api.lib.crispembed_spacy_free(context)

    drift = None
    if dump is not None:
        import numpy as np
        arrays = {f"{i}:{name}": np.asarray(values, dtype=np.float32).reshape(dims)
                  for i, stages in enumerate(dump) for name, (dims, values) in stages.items()}
        if args.dump:
            np.savez_compressed(args.dump, **arrays)
        if args.reference_dump:
            reference = np.load(args.reference_dump)
            drift = {}
            for name in DUMPED:
                keys = [k for k in reference.files if k.split(":", 1)[1] == name]
                worst = max(float(np.max(np.abs(reference[k] - arrays[k]))) if reference[k].size else 0.0
                            for k in keys)
                drift[name] = worst
    report = {
        "label": args.label, "admitted": True, "env": args.env, "model": str(model), "model_sha256": sha256(model),
        "model_bytes": model.stat().st_size, "load_ms": round(load_ms, 3),
        "cases": len(oracle["cases"]), "tokens": tokens_total, "tokenization_mismatch_cases": token_mismatch_cases,
        "exact_cases": exact_cases,
        "agreement": {f: round(agree[f] / tokens_total, 6) if tokens_total else None for f in FIELDS},
        "disagreements": {f: tokens_total - agree[f] for f in FIELDS},
        "timing": {"corpus_ms_median": round(statistics.median(corpus_ms), 3),
                   "long_doc_tokens": long_tokens, "long_doc_ms_median": round(statistics.median(long_ms), 3),
                   "long_doc_tokens_per_s": round(long_tokens / (statistics.median(long_ms) / 1e3), 1)},
        "max_abs_drift_vs_reference": drift, "differences": differences,
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    summary = {k: report[k] for k in ("label", "model_bytes", "exact_cases", "disagreements", "timing",
                                      "max_abs_drift_vs_reference")}
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
