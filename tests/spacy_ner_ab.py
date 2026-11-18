#!/usr/bin/env python3
"""A/B the opt-in native NER component against original spaCy entities.

Two steps, each in its own process:

    # 1. original spaCy (developer reference environment)
    python tests/spacy_ner_ab.py oracle --wheel WHEEL --corpus corpus.json --output ner-oracle.json
    # 2. native (any Python with ctypes)
    python tests/spacy_ner_ab.py native --library LIB --bundle BUNDLE --ner ner.gguf \
        --oracle ner-oracle.json --report report.json [--env KEY=VALUE ...]

The oracle records the original de_core_news_sm 3.8.0 pipeline's tokens,
sentence starts and entities (exclude=["senter"], the source default). The
native report compares tokenization, sentence starts and exact entity spans and
labels case by case, and times analysis with and without NER.
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
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def oracle(args):
    sys.path.insert(0, str(ROOT / "models"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("ner_convert", ROOT / "models" / "convert-spacy-ner-to-gguf.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as directory:
        nlp = module.load_with_ner(args.wheel, Path(directory))
        cases = []
        for doc in nlp.pipe(corpus["texts"]):
            cases.append({"text": doc.text, "tokens": [t.text for t in doc],
                          "sent_starts": [bool(t.is_sent_start) for t in doc],
                          "entities": [[e.start, e.end, e.label_] for e in doc.ents]})
    import spacy
    result = {"schema": "crispembed.spacy.ner_oracle.v1", "spacy": spacy.__version__,
              "wheel_sha256": sha256(args.wheel), "corpus_sha256": sha256(args.corpus), "cases": cases}
    args.output.write_text(json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(cases)} cases, {sum(len(c['entities']) for c in cases)} entities")


class EntityView(C.Structure):
    pass


def native(args):
    sys.path.insert(0, str(ROOT / "tests"))
    for item in args.env:
        key, value = item.split("=", 1)
        os.environ[key] = value
    os.environ.setdefault("CRISPEMBED_SPACY", "1")
    os.environ.setdefault("CRISPEMBED_SPACY_NER", "1")
    from test_spacy_text_api import NativeAPI, TextView, validate_bundle
    EntityView._fields_ = [("label", TextView)] + [(n, C.c_size_t) for n in (
        "token_start", "token_end", "byte_start", "byte_end", "scalar_start", "scalar_end", "utf16_start", "utf16_end")]
    api = NativeAPI(args.library, args.bundle, validate_bundle(args.bundle))
    lib = api.lib
    lib.crispembed_spacy_attach_ner.argtypes = [C.c_void_p, C.c_char_p, C.c_char_p, C.c_char_p, C.c_size_t]
    lib.crispembed_spacy_attach_ner.restype = C.c_int
    lib.crispembed_spacy_document_entity_count.argtypes = [C.c_void_p]
    lib.crispembed_spacy_document_entity_count.restype = C.c_size_t
    lib.crispembed_spacy_document_entity.argtypes = [C.c_void_p, C.c_size_t, C.POINTER(EntityView)]
    lib.crispembed_spacy_document_entity.restype = C.c_int
    data = json.loads(args.oracle.read_text(encoding="utf-8"))

    def run_all(context, with_entities):
        results = []
        for case in data["cases"]:
            status, document, error = api.call(context, case["text"].encode("utf-8"), analyze=True)
            if status != 0:
                raise RuntimeError(f"analyze failed ({status}): {error}")
            try:
                tokens = api.tokens(document)
                entities = []
                if with_entities:
                    view = EntityView()
                    for i in range(lib.crispembed_spacy_document_entity_count(document)):
                        if lib.crispembed_spacy_document_entity(document, i, C.byref(view)) != 0:
                            raise RuntimeError("entity view failed")
                        label = C.string_at(view.label.data, view.label.length).decode("utf-8")
                        entities.append([view.token_start, view.token_end, label])
                results.append((tokens, entities))
            finally:
                lib.crispembed_spacy_document_free(document)
        return results

    plain, error = api.open()
    if not plain:
        raise RuntimeError(error)
    context, error = api.open()
    if not context:
        raise RuntimeError(error)
    try:
        message = C.create_string_buffer(1024)
        status = lib.crispembed_spacy_attach_ner(context, os.fsencode(args.ner), sha256(args.ner).encode(),
                                                 message, len(message))
        if status != 0:
            report = {"attached": False, "status": status, "error": message.value.decode("utf-8", "replace")}
            args.report.write_text(json.dumps(report, indent=1) + "\n")
            print(json.dumps(report))
            return 2
        results = run_all(context, True)
        exact = token_mismatch = sentence_mismatch = 0
        oracle_entities = native_entities = matched = 0
        differences = []
        for index, (case, (tokens, entities)) in enumerate(zip(data["cases"], results)):
            if [t["text"].decode("utf-8") for t in tokens] != case["tokens"]:
                token_mismatch += 1
                differences.append({"case": index, "tokenization": True})
                continue
            if [bool(t["flags"] & 4) for t in tokens] != case["sent_starts"]:
                sentence_mismatch += 1
            expected = {tuple(e) for e in case["entities"]}
            got = {tuple(e) for e in entities}
            oracle_entities += len(expected)
            native_entities += len(got)
            matched += len(expected & got)
            if expected == got:
                exact += 1
            elif len(differences) < 100:
                differences.append({"case": index, "text": case["text"][:200],
                                    "oracle_only": sorted(expected - got), "native_only": sorted(got - expected)})
        timing = {}
        for label, ctx, flag in (("without_ner", plain, False), ("with_ner", context, True)):
            run_all(ctx, flag)
            samples = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                run_all(ctx, flag)
                samples.append((time.perf_counter() - start) * 1e3)
            timing[label + "_corpus_ms_median"] = round(statistics.median(samples), 3)
        tokens_total = sum(len(c["tokens"]) for c in data["cases"])
        timing["tokens"] = tokens_total
        timing["ner_overhead_fraction"] = round(timing["with_ner_corpus_ms_median"] /
                                                timing["without_ner_corpus_ms_median"] - 1, 4)
    finally:
        lib.crispembed_spacy_free(context)
        lib.crispembed_spacy_free(plain)
    report = {"schema": "crispembed.spacy.ner_ab.v1", "attached": True, "ner_sha256": sha256(args.ner),
              "oracle_sha256": sha256(args.oracle), "env": args.env, "cases": len(data["cases"]),
              "exact_entity_cases": exact, "token_mismatch_cases": token_mismatch,
              "sentence_start_mismatch_cases": sentence_mismatch, "oracle_entities": oracle_entities,
              "native_entities": native_entities, "matched_entities": matched, "timing": timing,
              "differences": differences}
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "differences"}))
    return 0 if exact == len(data["cases"]) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    o = sub.add_parser("oracle")
    o.add_argument("--wheel", type=Path, required=True)
    o.add_argument("--corpus", type=Path, required=True)
    o.add_argument("--output", type=Path, required=True)
    n = sub.add_parser("native")
    n.add_argument("--library", type=Path, required=True)
    n.add_argument("--bundle", type=Path, required=True)
    n.add_argument("--ner", type=Path, required=True)
    n.add_argument("--oracle", type=Path, required=True)
    n.add_argument("--report", type=Path, required=True)
    n.add_argument("--repeats", type=int, default=3)
    n.add_argument("--env", action="append", default=[])
    args = parser.parse_args()
    return oracle(args) if args.command == "oracle" else native(args)


if __name__ == "__main__":
    raise SystemExit(main() or 0)
