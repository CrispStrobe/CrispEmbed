#!/usr/bin/env python3
"""Exact native lexical comparison against the remotely frozen development set.

Build the developer adapter from the same checkout:
  c++ -std=c++17 -O2 -Isrc tools/spacy/lexical_probe.cpp src/spacy_lexical.cpp \
      -lpcre2-8 -o /path/to/lexical-probe
The probe is a developer executable and is not installed into the runtime.
Missing fixtures, an empty census, changed manifests or any discrete difference
fail explicitly. This is lexical evidence, not trained inference or grammar.
"""
from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from types import SimpleNamespace

from spacy_parity import ROOT, read_json, require, sha256, validate_inputs


SOURCE_FILES = [
    "src/spacy_lexical.cpp", "src/spacy_lexical.h", "src/spacy_lexical_data.h", "src/spacy_numeric.h",
    "src/core/json_value.h", "src/core/json.h", "src/core/clean_exit.h",
    "tools/spacy/lexical_probe.cpp", "tools/spacy/generate_lexical_data.py",
    "tests/test_spacy_lexical.cpp", "tests/spacy_lexical_parity.py", "tests/spacy_parity.py",
]
FIELDS = ["text", "norm", "prefix", "suffix", "shape", "orth_id", "start_byte", "end_byte",
          "start_codepoint", "end_codepoint", "start_utf16", "end_utf16", "whitespace", "is_space", "features"]


def native(probe, resources, texts):
    request = "".join(json.dumps(text, ensure_ascii=False) + "\n" for text in texts)
    result = subprocess.run([str(probe.resolve()), str(resources.resolve())], input=request,
                            text=True, encoding="utf-8", capture_output=True, timeout=120)
    require(result.returncode == 0, f"Native lexical probe failed ({result.returncode}): {result.stderr}")
    # U+2028/U+2029 are valid unescaped JSON string contents, not records.
    # str.splitlines() incorrectly splits them (and other Unicode separators).
    outputs = [json.loads(line) for line in result.stdout.split("\n") if line]
    require(len(outputs) == len(texts), "Native lexical output census mismatch")
    return outputs


def compare(cases, outputs):
    results, failures = [], []
    tokens = comparisons = 0
    for case, actual in zip(cases, outputs):
        name, expected = case["id"], case["tokens"]
        before = len(failures)
        if len(actual) != len(expected):
            failures.append({"case": name, "field": "token_count", "expected": len(expected), "actual": len(actual)})
        for i, (a, e) in enumerate(zip(actual, expected)):
            require(set(a) == set(FIELDS), "Incomplete/unexpected native lexical fields")
            for field in FIELDS:
                comparisons += len(a[field]) if field == "features" else 1
                if a[field] != e[field]:
                    failures.append({"case": name, "token": i, "field": field, "expected": e[field], "actual": a[field]})
        tokens += len(expected)
        results.append({"id": name, "tokens": len(expected), "exact": before == len(failures)})
    return {"cases": len(cases), "tokens": tokens, "exact_field_comparisons": comparisons,
            "case_results": results, "failures": failures, "passed": not failures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--quds-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    require(not args.report.exists(), "Refusing to overwrite retained lexical evidence")
    reference, reference_manifest, manifest, verified = validate_inputs(SimpleNamespace(
        library=args.probe, bundle=args.bundle, quds_root=args.quds_root, reference=None))
    require(verified["cases"] == 16 and verified["tokens"] == 113, "Frozen lexical census changed")
    cases = []
    for record in reference_manifest["cases"]:
        original = read_json(reference / record["id"] / "case.json")
        tokens = []
        for i, token in enumerate(original["annotations"]["tokens"]):
            values = {field: token[field] for field in FIELDS if field != "features"}
            values["features"] = original["features"]["uint64_decimal"][i]
            tokens.append(values)
        cases.append({"id": record["id"], "text": original["case"]["text"], "tokens": tokens})
    started = time.perf_counter()
    outputs = native(args.probe, args.bundle / "resources.json", [case["text"] for case in cases])
    report = compare(cases, outputs)
    report.update({
        "schema": "crispembed-spacy-lexical-parity-v1", "split": "development",
        "scope": "Native tokenizer, lexical strings/IDs/features and original byte/scalar/UTF16 spans",
        "trained_inference": False, "full_analysis_qualified": False, "acceptance_inputs_used": False,
        "reference_manifest_sha256": sha256(reference / "manifest.json"),
        "bundle_manifest_sha256": sha256(args.bundle / "manifest.json"),
        "resources_sha256": manifest["files"]["resources.json"]["sha256"],
        "probe_sha256": sha256(args.probe),
        "source_sha256": {name: sha256(ROOT / name) for name in SOURCE_FILES},
        "python": platform.python_version(), "platform": platform.system(), "machine": platform.machine(),
        "wall_seconds_including_probe_start": time.perf_counter() - started,
        "reproduction": [
            "c++ -std=c++17 -O2 -Isrc tools/spacy/lexical_probe.cpp src/spacy_lexical.cpp -lpcre2-8 -o /path/to/lexical-probe",
            "python tests/spacy_lexical_parity.py --probe /path/to/lexical-probe --bundle /path/to/bundle --quds-root /path/to/quds --report /path/to/new-report.json",
        ],
    })
    pcre = ctypes.CDLL("libpcre2-8.so")
    version = ctypes.create_string_buffer(128)
    pcre.pcre2_config_8(11, version)
    report["pcre2"] = version.value.decode("ascii")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ["cases", "tokens", "exact_field_comparisons", "passed"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"LEXICAL_PARITY_FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)
