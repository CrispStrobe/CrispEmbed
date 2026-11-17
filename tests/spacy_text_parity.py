#!/usr/bin/env python3
"""Qualify native raw-text analysis against the unchanged development oracle.

No reference tokens, lexical IDs, parser states or action masks are supplied to
the native analyzer. Both public text services receive only original UTF-8.
The frozen eight-case acceptance split is deliberately outside this driver.
"""
from __future__ import annotations

import argparse
import ctypes as C
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

import spacy_parity as numeric
from test_spacy_text_api import NativeAPI, TRACE

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    "src/spacy_pipeline.h", "src/spacy_pipeline.cpp", "src/spacy_numeric.h", "src/spacy_numeric.cpp",
    "src/spacy_lexical.h", "src/spacy_lexical.cpp", "src/spacy_lexical_data.h",
    "src/spacy_decode.h", "src/spacy_decode.cpp", "src/core/json.h", "src/core/json_value.h", "src/core/sha256.h",
    "src/core/gguf_loader.h", "src/core/gguf_loader.cpp", "src/core/env_gate.h",
    "tools/spacy/CMakeLists.txt", "tests/spacy_parity.py", "tests/spacy_text_parity.py",
    "tests/test_spacy_text_api.py",
]


def source_hashes():
    return {name: numeric.sha256(ROOT / name) for name in SOURCES}


def token_annotations(tokens, analyzed):
    result = []
    for index, token in enumerate(tokens):
        item = {"i": index, **{name: token[name].decode("utf-8")
                               for name in ("text", "norm", "prefix", "suffix", "shape")},
                "orth_id": str(token["orth_id"]), "norm_id": str(token["features"][0]),
                "whitespace": " " if token["flags"] & 1 else "",
                "is_space": bool(token["flags"] & 2)}
        for api_name, reference_name in [("byte", "byte"), ("scalar", "codepoint"), ("utf16", "utf16")]:
            item["start_" + reference_name] = token[api_name + "_start"]
            item["end_" + reference_name] = token[api_name + "_end"]
        if analyzed:
            item.update({name: token[name].decode("utf-8") for name in ("tag", "pos", "morph", "lemma")})
            item.update(dep=token["dependency"].decode("utf-8"), head=token["head"],
                        is_sent_start=bool(token["flags"] & 4))
        result.append(item)
    return result


def sentence_annotations(text, tokens):
    data = text.encode("utf-8")
    sentences = []
    start = 0
    for index, token in enumerate(tokens):
        if token["flags"] & 8:
            first = tokens[start]
            item = {"start_token": start, "end_token": index + 1,
                    "text": data[first["byte_start"]:token["byte_end"]].decode("utf-8")}
            for api_name, reference_name in [("byte", "byte"), ("scalar", "codepoint"), ("utf16", "utf16")]:
                item["start_" + reference_name] = first[api_name + "_start"]
                item["end_" + reference_name] = token[api_name + "_end"]
            sentences.append(item)
            start = index + 1
    numeric.require(start == len(tokens), "Native sentence ends do not cover all tokens")
    return sentences


def differences(expected, actual, path=""):
    """Keep every discrete difference, including the earliest token/field."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        result = []
        for key in sorted(set(expected) | set(actual)):
            field = f"{path}.{key}" if path else key
            if key not in expected or key not in actual:
                result.append({"field": field, "expected": expected.get(key), "actual": actual.get(key)})
            else:
                result.extend(differences(expected[key], actual[key], field))
        return result
    if isinstance(expected, list) and isinstance(actual, list):
        result = []
        if len(expected) != len(actual):
            result.append({"field": path + ".length", "expected": len(expected), "actual": len(actual)})
        for index, (left, right) in enumerate(zip(expected, actual)):
            result.extend(differences(left, right, f"{path}[{index}]"))
        return result
    return [] if expected == actual else [{"field": path, "expected": expected, "actual": actual}]


def run_case(api, context, case, payload):
    data = case["case"]["text"].encode("utf-8")
    expected = case["annotations"]
    started = time.perf_counter()
    status, lexical_document, error = api.call(context, data)
    numeric.require(status == 0 and lexical_document, f"Native tokenization failed ({status}): {error}")
    try:
        numeric.require(not api.lib.crispembed_spacy_document_is_analyzed(lexical_document),
                        "Tokenization result claims analysis")
        lexical_tokens = api.tokens(lexical_document)
    finally:
        api.lib.crispembed_spacy_document_free(lexical_document)
    lexical_seconds = time.perf_counter() - started
    lexical = token_annotations(lexical_tokens, False)
    lexical_expected = [{key: value for key, value in token.items()
                         if key not in {"tag", "pos", "morph", "lemma", "dep", "head", "is_sent_start"}}
                        for token in expected["tokens"]]
    discrete = differences(lexical_expected, lexical, "tokenization.tokens")
    discrete += differences(case["features"]["uint64_decimal"],
                            [[str(value) for value in token["features"]] for token in lexical_tokens],
                            "tokenization.features")

    captured, callback_errors = {}, []

    @TRACE
    def callback(_user, encoded_name, values, shape, rank):
        try:
            numeric.require(rank <= 4, "Invalid native trace rank")
            dims = [int(shape[index]) for index in range(rank)]
            numeric.require(all(0 <= size <= 1000000 for size in dims), "Invalid native trace dimensions")
            count = math.prod(dims)
            numeric.require(count <= 16 * 1024 * 1024 and (bool(values) or count == 0), "Invalid native trace extent")
            name = encoded_name.decode("utf-8")
            numeric.require(name not in captured, f"Repeated native trace stage: {name}")
            array = np.ctypeslib.as_array(values, shape=(count,)).copy() if count else np.empty(0, dtype=np.float32)
            captured[name] = array.reshape(dims)
            return 0
        except Exception as exception:
            callback_errors.append(str(exception))
            return 1

    started = time.perf_counter()
    status, document, error = api.call(context, data, analyze=True, trace=callback)
    numeric.require(status == 0 and document and not callback_errors,
                    f"Native analysis failed ({status}): {error}; callback={callback_errors}")
    try:
        numeric.require(api.lib.crispembed_spacy_document_is_analyzed(document) == 1,
                        "Complete analysis result lacks its capability flag")
        tokens = api.tokens(document)
        annotations = {"tokens": token_annotations(tokens, True),
                       "sentences": sentence_annotations(case["case"]["text"], tokens)}
    finally:
        api.lib.crispembed_spacy_document_free(document)
    seconds = time.perf_counter() - started
    discrete += differences(expected, annotations, "analysis")
    discrete += differences(case["features"]["uint64_decimal"],
                            [[str(value) for value in token["features"]] for token in tokens], "analysis.features")
    # Exactly the original stages, plus three explicit native state records per
    # transition. Neither missing nor extra stages can pass qualification.
    state_names = set()
    native_steps = []
    for step in case["parser_steps"]:
        prefix = f"parser.step.{step['step']}.state."
        for name in ("features", "valid_actions", "chosen_action"):
            state_names.add(prefix + name)
        if any(prefix + name not in captured for name in ("features", "valid_actions", "chosen_action")):
            continue
        actual_step = {"step": step["step"], "feature_token_ids": captured[prefix + "features"].tolist(),
                       "valid_actions": captured[prefix + "valid_actions"].tolist(),
                       "chosen_action": captured[prefix + "chosen_action"].tolist()}
        native_steps.append(actual_step)
        discrete += differences(step["feature_token_ids"], actual_step["feature_token_ids"], prefix + "features")
        discrete += differences([int(value) for value in step["valid_actions"]], actual_step["valid_actions"],
                                prefix + "valid_actions")
        discrete += differences([step["chosen_action"]], actual_step["chosen_action"], prefix + "chosen_action")
    required_stages = set(case["traces"]) | state_names
    missing, extra = sorted(required_stages - set(captured)), sorted(set(captured) - required_stages)
    stages = []
    classifier_decisions = 0
    for name, entry in sorted(case["traces"].items(), key=lambda pair: pair[1]["offset_bytes"]):
        if name not in captured:
            continue
        original = np.frombuffer(payload, dtype="<f4", count=entry["size_bytes"] // 4,
                                 offset=entry["offset_bytes"]).reshape(entry["shape"])
        stages.append(numeric.compare_stage(name, original, captured[name]))
        if name in {"tagger.scores", "morphologizer.scores", "lemmatizer.scores"}:
            expected_classes = original.argmax(axis=1).tolist()
            actual_classes = captured[name].argmax(axis=1).tolist()
            classifier_decisions += len(actual_classes)
            discrete += differences(expected_classes, actual_classes, name + ".argmax")
    return {"id": case["case"]["id"], "input_sha256": case["input_sha256"], "tokens": len(tokens),
            "sentences": len(annotations["sentences"]), "parser_steps": len(native_steps),
            "classifier_decisions": classifier_decisions,
            "tokenize_seconds": lexical_seconds, "analysis_seconds_with_trace_copy": seconds,
            "discrete_differences": discrete, "first_discrete_difference": discrete[0] if discrete else None,
            "missing_stages": missing, "extra_stages": extra, "stages": stages,
            "native_annotations": annotations, "native_parser_steps": native_steps,
            "passed": not discrete and not missing and not extra and all(stage["passed"] for stage in stages)}


def qualify(args):
    reference, manifest, bundle, verified = numeric.validate_inputs(args)
    os.environ["CRISPEMBED_SPACY"] = "1"
    api = NativeAPI(args.library, args.bundle, bundle)
    context, error = api.open()
    numeric.require(context, "Native text resources admission failed: " + error)
    cases = []
    try:
        numeric.require(api.lib.crispembed_spacy_text_abi_version() == 1 and
                        api.lib.crispembed_spacy_context_capabilities(context) == 15,
                        "Required complete native text capabilities missing")
        for item in manifest["cases"]:
            case = numeric.read_json(reference / item["id"] / "case.json")
            try:
                cases.append(run_case(api, context, case, (reference / item["id"] / "traces.f32").read_bytes()))
            except Exception as exception:
                cases.append({"id": item["id"], "passed": False, "error": str(exception)})
    finally:
        api.lib.crispembed_spacy_free(context)
    counts = {"cases": len(cases), "tokens": sum(case.get("tokens", 0) for case in cases),
              "sentences": sum(case.get("sentences", 0) for case in cases),
              "parser_steps": sum(case.get("parser_steps", 0) for case in cases),
              "classifier_decisions": sum(case.get("classifier_decisions", 0) for case in cases),
              "numeric_stages": sum(len(case.get("stages", [])) for case in cases),
              "float_elements": sum(stage["elements"] for case in cases for stage in case.get("stages", []))}
    complete = (counts["tokens"] == verified["tokens"] and counts["numeric_stages"] == verified["trace_stages"]
                and counts["parser_steps"] == verified["parser_steps"] and counts["tokens"] > 0
                and counts["classifier_decisions"] == 3 * counts["tokens"])
    return {"schema": "crispembed.spacy.raw_text_parity.v1", "split": "development",
            "scope": "Original UTF-8 through native tokenizer, numeric graph and greedy linguistic decoders",
            "reference_manifest_sha256": numeric.REFERENCE_SHA256,
            "bundle_manifest_sha256": numeric.sha256(args.bundle / "manifest.json"),
            "library_sha256": numeric.sha256(args.library), "sources_sha256": source_hashes(),
            "environment": {"system": platform.system(), "machine": platform.machine(),
                            "python": platform.python_version(), "numpy": np.__version__},
            "numerical_budget": {"absolute_tolerance": numeric.ATOL, "relative_tolerance": numeric.RTOL},
            "reference_data_supplied_to_native": False, "acceptance_inputs_executed": False,
            "counts": counts, "complete_denominators": complete, "cases": cases,
            "passed": complete and len(cases) > 0 and all(case["passed"] for case in cases)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--quds-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    numeric.require(not args.report.exists(), "Refusing to overwrite a retained qualification receipt")
    before = source_hashes()
    try:
        report = qualify(args)
        numeric.require(source_hashes() == before, "Source changed during native qualification")
    except Exception as exception:
        report = {"schema": "crispembed.spacy.raw_text_parity.v1", "passed": False, "error": str(exception),
                  "sources_sha256": before, "acceptance_inputs_executed": False}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({key: report[key] for key in ("passed", "counts", "error") if key in report}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
