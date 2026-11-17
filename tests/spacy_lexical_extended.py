#!/usr/bin/env python3
"""Additional developer lexical coverage, separate from frozen acceptance inputs.

Exhaust every authenticated tokenizer exception under six fixed punctuation/
context wrappers, every reserved StringStore symbol and every exported scalar
lowercase entry with sigma context. Add explicit Unicode/URL/whitespace cases
and 250 deterministic authored mixtures. This compares tokenizer output only;
it does not execute trained neural inference or qualify grammar quality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
import sys
import tempfile

from spacy_lexical_parity import FIELDS, SOURCE_FILES, compare, native
from spacy_parity import ROOT, read_json, require, sha256


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def inputs(resources, symbols):
    cases = []
    for index, text in enumerate(sorted(resources["tokenizer"]["exceptions"])):
        contexts = [text, "(" + text + ")", "x/" + text + "?x", text + " " + text,
                    "“" + text + "”", "a-" + text + "-b"]
        for wrapper, value in enumerate(contexts):
            cases.append({"id": f"exception-{index}-{wrapper}", "family": "exception-contexts", "text": value})
    for index, text in enumerate(sorted(symbols)):
        cases.append({"id": f"symbol-{index}", "family": "reserved-symbols", "text": text})
    for codepoint in resources["unicode"]["scalar_lower"]:
        char = chr(int(codepoint))
        cases.append({"id": f"lower-{codepoint}", "family": "lowercase-sigma-contexts", "text": "A" + char + "Σ " + char + "ΣA"})
    controls = ["ΟΣ ΟΣΑ Σ ΑΣ\u0301 ΑΣ\u0301Α", "İ ı ſ K", "a\x00b", "a\r\nb", "  a  b  ",
                "a \t b\n", "a\u00a0b", "a\u2028b\u2029c", "x" * 99, "x" * 100, "👩\u200d🔬👩\u200d🔬",
                "https://é.de/abc?d=2", "https://².de", "https://\u2163.de", "https://\u00a0.de",
                "ΟΣ\u0345 ΟΣ\u0301 ΟΣ\u02bc ΟΣ\u0345Α"]
    for index, text in enumerate(controls):
        cases.append({"id": f"unicode-control-{index}", "family": "unicode-controls", "text": text})
    rng = random.Random(90210)
    alphabet = "abAZ012.!,/:-_()\r\n\t ÄÖßéΣσς\u0301\u200d\u00a0\0👩𝄞"
    for index in range(250):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 80)))
        cases.append({"id": f"authored-random-{index}", "family": "authored-mixtures", "text": text})
    return cases


def reference_tokens(nlp, text):
    from spacy.attrs import NORM, PREFIX, SUFFIX, SHAPE, SPACY, IS_SPACE
    doc = nlp.make_doc(text)
    features = doc.to_array([NORM, PREFIX, SUFFIX, SHAPE, SPACY, IS_SPACE])
    result = []
    for i, token in enumerate(doc):
        end = token.idx + len(token.text)
        result.append({
            "text": token.text, "norm": token.norm_, "prefix": token.prefix_, "suffix": token.suffix_,
            "shape": token.shape_, "orth_id": str(token.orth), "start_codepoint": token.idx, "end_codepoint": end,
            "start_byte": len(text[:token.idx].encode("utf-8")), "end_byte": len(text[:end].encode("utf-8")),
            "start_utf16": len(text[:token.idx].encode("utf-16-le")) // 2,
            "end_utf16": len(text[:end].encode("utf-16-le")) // 2,
            "whitespace": token.whitespace_, "is_space": token.is_space,
            "features": [str(value) for value in features[i]],
        })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    require(not args.report.exists(), "Refusing to overwrite retained extended evidence")
    require(args.probe.is_file(), "Required native lexical probe missing")
    sys.path.insert(0, str(ROOT / "models"))
    from spacy_bundle import load_verified_model, verify_environment, verify_wheel
    verify_environment()
    verify_wheel(args.wheel)
    from spacy.symbols import IDS
    manifest = read_json(ROOT / "docs/validation/native-spacy-20261117/bundle-manifest.json")
    require(read_json(args.bundle / "manifest.json") == manifest, "Converted bundle changed from the frozen W1 manifest")
    resources_path = args.bundle / "resources.json"
    require(sha256(resources_path) == manifest["files"]["resources.json"]["sha256"], "Resource bytes changed")
    resources = read_json(resources_path)
    cases = inputs(resources, IDS)
    require(len(cases) == 4658, "Extended corpus census changed")
    input_hash = hashlib.sha256(canonical(cases)).hexdigest()
    outputs = native(args.probe, resources_path, [case["text"] for case in cases])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="spacy-lexical-reference-", dir=args.report.parent) as temporary:
        nlp = load_verified_model(args.wheel, Path(temporary) / "model")
        for case in cases:
            case["tokens"] = reference_tokens(nlp, case["text"])
    result = compare(cases, outputs)
    families = {}
    for case, actual in zip(cases, outputs):
        family = families.setdefault(case["family"], {"cases": 0, "tokens": 0, "exact_cases": 0})
        family["cases"] += 1
        family["tokens"] += len(case["tokens"])
        family["exact_cases"] += actual == case["tokens"]
    result.pop("case_results")
    result.update({
        "schema": "crispembed-spacy-lexical-extended-v1", "split": "additional-development",
        "input_sha256": input_hash, "families": families,
        "actual_output_sha256": hashlib.sha256(canonical(outputs)).hexdigest(),
        "reference_output_sha256": hashlib.sha256(canonical([case["tokens"] for case in cases])).hexdigest(),
        "scope": "Every token field and six exact feature IDs; no neural inference",
        "acceptance_inputs_used": False, "trained_inference": False, "full_analysis_qualified": False,
        "source_wheel_sha256": sha256(args.wheel), "resources_sha256": sha256(resources_path),
        "probe_sha256": sha256(args.probe),
        "source_sha256": {name: sha256(ROOT / name) for name in SOURCE_FILES + ["tests/spacy_lexical_extended.py", "models/spacy_bundle.py"]},
        "python": platform.python_version(), "platform": platform.system(), "machine": platform.machine(),
        "initial_setup_failure": {
            "native_exit_code": 0, "comparison_completed": False, "runtime_failure": False,
            "error": "str.splitlines() split raw U+2028/U+2029 within valid JSON strings before comparison",
            "repair": "Split only the literal LF record delimiter; preserve Unicode separators within strings",
        },
        "reproduction": "python tests/spacy_lexical_extended.py --probe /path/to/lexical-probe --bundle /path/to/bundle --wheel /path/to/de_core_news_sm-3.8.0-py3-none-any.whl --report /path/to/new-extended-report.json",
    })
    args.report.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ["cases", "tokens", "exact_field_comparisons", "passed"]}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
