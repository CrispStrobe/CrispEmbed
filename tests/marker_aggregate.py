#!/usr/bin/env python3
"""Rehash the retained Marker receipts and summarize their distinct scopes.

This does not run models, alter gold, rewrite raw captures, or promote any
historical numerical failure. Run from the repository after the declared panels.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "tests/results/marker"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(relative):
    return json.loads((RESULTS / relative).read_text(encoding="utf-8"))


def main():
    public = read("native-final-public/summary.json")
    independent = read("native-reference-final/receipt.json")
    original = read("upstream-model-free-attempt-2/receipt.json")
    units = read("build/unit-final.json")
    consumers = read("build/compiled-consumers-final.json")
    sanitizer = read("sanitizers-final/receipt.json")
    clients = read("client-review-receipt.json")
    packaging = read("python-packaging/receipt.json")
    installs = read("cmake-install/receipt.json")
    fixtures = read("fixture-reproducibility.json")
    build = read("build/final-source.json")
    continuity = read("teleocr-source-continuity.json")
    static = read("build/static-final.json")
    # Readiness of the separately written independent review is mandatory.
    review = read("native-review.json")
    assert review["review_complete"] and review["review_passed"]
    assert digest(ROOT / review["reviewer_source"]) == review["reviewer_source_sha256"]

    current_source = {name: digest(ROOT / name) == expected
                      for name, expected in build["source_sha256"].items()}
    assert all(current_source.values()), "Runtime/consumer source drift after final build"
    assert all(item["identical_to_teleocr_delivery"] for item in continuity["files"])
    assert all(digest(ROOT / "tests/fixtures/marker" / item["file"]) == item["sha256"]
               for item in fixtures["files"]), "Owned fixture bytes changed"
    final_library = build["binary_sha256"]["libcrispembed.so.0.17.14"]
    assert units["library_sha256"] == consumers["library_sha256"] == final_library
    assert independent["native_library_sha256"] == final_library
    assert all(digest(ROOT / name) == expected for name, expected in sanitizer["source_sha256"].items())
    assert all(digest(ROOT / name) == expected for name, expected in packaging["source_sha256"].items())

    model_dirs = ["native-ocr-f16", "native-final-f16", "native-final-q8",
                  "native-final-q4", "native-final-force-pdf"]
    models = []
    distinct_inputs = set()
    final_source_matches = []
    for folder in model_dirs:
        report = read(folder + "/summary.json")
        assert report["model_sha256"] == report["expected_model_sha256"]
        snapshot = RESULTS / folder / "marker_native.executed.py"
        assert digest(snapshot) == report["source_sha256"]["tests/marker_native.py"]
        for case in report["cases"]:
            assert digest(ROOT / case["input"]) == case["sha256"]
            distinct_inputs.add(case["sha256"])
        if folder != "native-ocr-f16":
            assert report["library_sha256"] == final_library
            matches = {name: digest(ROOT / name) == expected
                       for name, expected in report["source_sha256"].items()}
            assert all(matches.values()), "Final model receipt source drift"
            final_source_matches.append({"panel": folder, "sources_match": matches})
        models.append({
            "receipt": folder + "/summary.json",
            "build": "initial-full-F16" if folder == "native-ocr-f16" else "final",
            "precision": report["model_file"],
            "mode": report.get("mode", "auto"),
            "model_sha256": report["model_sha256"],
            "library_sha256": report["library_sha256"],
            "threads": report["threads"],
            "max_tokens": report["max_tokens"],
            "max_layout_tokens": report["max_layout_tokens"],
            "peak_rss_kib": report["peak_rss_kib"],
            "total_seconds_including_load": report["seconds"],
            "functional_passed": report["functional_passed"],
            "quality_passed": report["quality_passed"],
            "results": [{"id": item["id"], "seconds": item["seconds"],
                         "checks": item.get("checks", {}), "quality": item.get("quality", {}),
                         "functional_passed": item["functional_passed"],
                         "quality_passed": item["quality_passed"],
                         "actual_text": item.get("actual_text"), "error": item.get("error")}
                        for item in report["results"]],
        })

    receipt_paths = [
        "native-final-public/summary.json", "native-reference-final/receipt.json",
        "upstream-model-free-attempt-2/receipt.json", "build/unit-final.json",
        "build/compiled-consumers-final.json", "sanitizers-final/receipt.json",
        "client-review-receipt.json", "python-packaging/receipt.json",
        "cmake-install/receipt.json", "fixture-reproducibility.json",
        "build/final-source.json", "teleocr-source-continuity.json",
        "build/static-final.json", "native-review.json",
        *[folder + "/summary.json" for folder in model_dirs],
    ]
    processing_passed = all([
        public["all_passed"],
        independent["summary"]["checks_passed"] == independent["summary"]["check_count"],
        units["passed"], consumers["passed"], sanitizer["passed"],
        all(item["exit_code"] == 0 for item in clients["tests"]),
        packaging["passed"], installs["passed"], fixtures["pass"], static["passed"],
    ])
    summary = {
        "schema_version": 1,
        "implementation": "Native Marker-style PDF/image document processing with TeleOCR GGUF/GGML",
        "upstream_marker_revision": "71dd5fd7e51ab24f8624b6ba2fcb6014b4e40ed4",
        "source_baseline": "4776ef977214890b416e871f6f66a6891d2f4b2d",
        "inherited_teleocr_revision": "68bd30b4db5d1dd2c899136491b14b9269948609",
        "final_library_sha256": final_library,
        "source_identity": "Final delivery SOURCE-PINS.json identifies the committed tree; file hashes bind these tests.",
        "default_off_gates": ["CRISPEMBED_MARKER", "CRISPEMBED_TELEOCR"],
        "qualification": {
            "bounded_processing_and_public_checks_passed": processing_passed,
            "declared_neural_functional_panel_passed": all(model["functional_passed"] for model in models),
            "declared_neural_text_and_cell_panel_passed": all(model["quality_passed"] for model in models),
            "strict_numerical_qualification": False,
            "original_surya_model_port_or_parity": False,
            "representative_OCR_or_layout_quality": False,
            "GPU_mobile_WASM_or_nonLinux_qualification": False,
            "release_publication_or_default_promotion": False,
        },
        "source_hashes_still_match": current_source,
        "final_model_source_checks": final_source_matches,
        "model_free": {
            "native_units": units["tests"],
            "public_Python_CLI_HTTP": {key: public[key] for key in ("passed", "total", "all_passed")},
            "compiled_consumers": consumers["results"],
            "independent_native": independent["summary"],
            "original_marker": original["summary"],
            "client_regressions": clients["tests"],
            "ASAN_UBSAN": {key: sanitizer[key] for key in ("checks", "failures", "passed", "leak_sanitizer")},
            "python_packaging_checks": len(packaging["checks"]),
            "python_packaging_passed": packaging["passed"],
            "CMake_install_file_comparisons": sum(len(item["notices_exact"]) for item in installs["results"]),
            "fixture_hash_matches": sum(item["exact"] for item in fixtures["files"]),
            "counts_overlap": True,
        },
        "neural_execution": {
            "conversion_trials": sum(len(model["results"]) for model in models),
            "distinct_owned_inputs": len(distinct_inputs),
            "independent_saved_output_review_checks": review["counts"]["independent_review_checks"],
            "independent_saved_output_review_passed": review["review_passed"],
            "panels": models,
            "generated_token_ID_or_EOS_parity_measured_by_this_runner": False,
            "quality_metric": "Whitespace-collapse-only visible text CER/WER; table cells additionally checked. Exact code indentation is separately checked in the digital panel.",
            "layout_limits": "Synthetic lines are heading-labeled; table title is caption-labeled. Table row/column content passes but header semantics and label/box-IoU accuracy are not qualified.",
        },
        "retained_initial_evidence": [
            "native-initial/summary.json", "native-reference-attempt-1/receipt.json",
            "upstream-model-free-attempt-1/receipt.json", "native-ocr-f16/summary.json",
            "format-review-before/", "client-review-before/", "sanitizers/receipt.json",
        ],
        "inherited_numerical_evidence": "../teleocr/summary.json",
        "receipt_sha256": {path: digest(RESULTS / path) for path in receipt_paths},
        "aggregator_sha256": digest(Path(__file__)),
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"qualification": summary["qualification"],
                      "conversion_trials": summary["neural_execution"]["conversion_trials"],
                      "distinct_owned_inputs": len(distinct_inputs)}, indent=2))
    return 0 if (processing_passed
                 and summary["qualification"]["declared_neural_functional_panel_passed"]
                 and summary["qualification"]["declared_neural_text_and_cell_panel_passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
