#!/usr/bin/env python3
"""Recheck the frozen TeleOCR report matrix without running model inference.

Usage: python summarize-matrix.executed.py RESULT_DIRECTORY SYNTHETIC_GOLD_JSON
Writes summary.json only when all 24 final reports and their receipts exist.
The original reports, gates, prompts, caps and outputs are never modified.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


CASES = ("text", "formula", "table", "code", "layout", "synthetic")
RUNS = {
    "f32": ("native-f32",),
    "f16": ("native-f16-small", "native-f16-large"),
    "q8_0": ("native-q8-small", "native-q8-large"),
    "q4_k": ("native-q4-small", "native-q4-large"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def distance(a, b):
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    row = list(range(len(b) + 1))
    for i, aa in enumerate(a, 1):
        new = [i]
        for j, bb in enumerate(b, 1):
            new.append(min(new[-1] + 1, row[j] + 1, row[j - 1] + (aa != bb)))
        row = new
    return row[-1]


def errors(expected: str, actual: str):
    normalized_expected = " ".join(expected.split())
    normalized_actual = " ".join(actual.split())
    out = {}
    for label, a, b in (
        ("raw_cer", expected, actual),
        ("whitespace_normalized_cer", normalized_expected, normalized_actual),
        ("wer", expected.split(), actual.split()),
    ):
        edits = distance(a, b)
        out[label] = {"edits": edits, "reference_units": len(a),
                      "rate": edits / len(a) if a else (0.0 if not b else None)}
    return out


def main():
    root = Path(sys.argv[1])
    gold = read(Path(sys.argv[2]))
    references = {case: read(root / "reference" / f"{case}.json") for case in CASES}
    rows = []
    formats = {}
    for precision, run_names in RUNS.items():
        found = {}
        for name in run_names:
            receipt = read(root / name / "summary.json")
            assert receipt["backend"] == "cpu" and receipt["threads"] == 2
            for record in receipt["cases"]:
                case = record["case"]
                assert case not in found, (precision, case, "duplicate result")
                found[case] = (name, receipt, record)
        assert set(found) == set(CASES), (precision, "missing or extra case", sorted(found))
        format_rows = []
        model_hashes = set()
        for case in CASES:
            name, receipt, record = found[case]
            relative = Path(name) / record["report"]
            path = root / relative
            assert sha256(path) == record["report_sha256"], (path, "report hash mismatch")
            report = read(path)
            ref = references[case]
            assert report["threads"] == ref["environment"]["threads"] == 2
            for key in ("numerical_pass", "decoded_pass", "coverage_complete", "confidence_pass",
                        "reached_eos", "tokens", "seconds"):
                assert record[key] == report[key], (path, "receipt flag or count mismatch", key)
            if report["confidence_pass"]:
                assert 0 < report["confidence_count"] == report["tokens"]
            assert record["reference_sha256"] == ref["reference_sha256"]
            assert record["image_sha256"] == ref["image_sha256"]
            assert record["max_new_tokens"] == ref["environment"]["max_new_tokens"] == 1024
            model_hashes.add(receipt["model_sha256"])
            stages = {s["name"]: s for s in report["stages"]}
            assert len(stages) == len(report["stages"]), (path, "duplicate stage")
            coverage = set(stages) == set(ref["stages"]) and len(stages) == 75
            assert coverage == report["coverage_complete"], (path, "coverage flag mismatch")
            ids_match = report["generated_ids"] == ref["generated_ids"]
            different_ids = [i for i, (a, b) in enumerate(zip(ref["generated_ids"], report["generated_ids"]))
                             if a != b]
            mismatch_count = len(different_ids) + abs(len(ref["generated_ids"]) - len(report["generated_ids"]))
            first_difference = None
            if not ids_match:
                index = different_ids[0] if different_ids else min(len(ref["generated_ids"]), len(report["generated_ids"]))
                first_difference = {"index": index,
                                    "reference": ref["generated_ids"][index] if index < len(ref["generated_ids"]) else None,
                                    "native": report["generated_ids"][index] if index < len(report["generated_ids"]) else None}
            text_match = report["text"] == ref["text"]
            eos = bool(report["generated_ids"]) and report["generated_ids"][-1] in (151645, 151643)
            assert eos == report["reached_eos"]
            assert report["tokens"] == len(report["generated_ids"])
            first_logits = stages["logits_step0"]
            raw_argmax_match = first_logits["argmax_native"] == first_logits["argmax_ref"]
            assert report["decoded_pass"] == (ids_match and text_match and eos)
            assert report["numerical_pass"] == (coverage and not report["error"]
                                                and all(s["pass"] for s in report["stages"]))
            assert report["pass"] == (report["numerical_pass"] and report["decoded_pass"]
                                      and coverage and report["confidence_pass"])
            assert record["exit_code"] == (0 if report["pass"] else 1)
            exact_inputs = all(stages[s]["max_abs"] == 0 for s in
                               ("input_patches", "image_grid", "prompt_ids", "mrope_positions"))
            row = {
                "precision": precision, "case": case, "report": relative.as_posix(),
                "report_sha256": sha256(path),
                "reference_report_sha256": sha256(root / "reference" / f"{case}.json"),
                "model_sha256": receipt["model_sha256"],
                "binary_sha256": receipt["binary_sha256"],
                "original_tokens": ref["tokens"], "native_tokens": report["tokens"],
                "generated_ids_exact": ids_match, "raw_text_exact": text_match,
                "generated_id_mismatch_count": mismatch_count,
                "first_different_generated_id": first_difference,
                "first_logits_argmax_exact": raw_argmax_match,
                "decoded_pass": report["decoded_pass"], "reached_eos": eos,
                "hit_cap_without_eos": report["tokens"] >= record["max_new_tokens"] and not eos,
                "original_characters": len(ref["text"]), "native_characters": len(report["text"]),
                "original_lines": len(ref["text"].splitlines()),
                "native_lines": len(report["text"].splitlines()),
                "errors_vs_original": errors(ref["text"], report["text"]),
                "coverage_complete": coverage, "stage_count": len(stages),
                "strict_stage_passes": sum(s["pass"] for s in report["stages"]),
                "strict_stage_failures": sum(not s["pass"] for s in report["stages"]),
                "numerical_pass": report["numerical_pass"],
                "max_stage_abs": max(s["max_abs"] for s in report["stages"]),
                "max_float_stage_abs": max(s["max_abs"] for s in report["stages"] if s["name"] not in
                                           ("generated_ids", "image_grid", "prompt_ids", "mrope_positions")),
                "minimum_stage_row_cosine": min(s["min_cos"] for s in report["stages"]),
                "first_logits_max_abs": first_logits["max_abs"],
                "input_patches_grid_prompt_mrope_exact": exact_inputs,
                "confidence_pass": report["confidence_pass"],
                "confidence_count": report["confidence_count"],
                "native_seconds": report["seconds"], "original_seconds": ref["seconds"],
                "strict_overall_pass": report["pass"],
            }
            if case == "synthetic":
                expected = gold["gold_text"]
                row["synthetic_owned_gold"] = errors(expected, report["text"])
                row["synthetic_owned_gold_text_exact"] = expected == report["text"]
            rows.append(row)
            format_rows.append(row)
        assert len(model_hashes) == 1, (precision, "mixed model hashes")
        formats[precision] = {
            "cases": len(format_rows), "model_sha256": next(iter(model_hashes)),
            "decoded_exact_cases": sum(r["decoded_pass"] for r in format_rows),
            "eos_cases": sum(r["reached_eos"] for r in format_rows),
            "native_tokens": sum(r["native_tokens"] for r in format_rows),
            "strict_numerical_pass_cases": sum(r["numerical_pass"] for r in format_rows),
            "strict_stage_passes": sum(r["strict_stage_passes"] for r in format_rows),
            "strict_stage_failures": sum(r["strict_stage_failures"] for r in format_rows),
            "max_stage_abs": max(r["max_stage_abs"] for r in format_rows),
            "max_float_stage_abs": max(r["max_float_stage_abs"] for r in format_rows),
            "minimum_stage_row_cosine": min(r["minimum_stage_row_cosine"] for r in format_rows),
        }
    result = {
        "schema_version": 1,
        "kind": "teleocr-final-original-model-cpu-matrix",
        "status": "Experimental and default-off; strict numerical qualification remains false.",
        "source_model": "XingChen-AGI/TeleOCR",
        "source_revision": "b3f6a0c084db72f971e19a779d2fd8a6bc9e3b8d",
        "source_module_sha256": references["text"]["environment"]["module_sha256"],
        "reference_environment": references["text"]["environment"],
        "backend": "Linux x86_64 CPU", "threads_per_forward": 2,
        "timing_caveat": "Runs overlapped with bounded validation work. Timings are diagnostic, not a controlled speed comparison.",
        "reference_cases": len(CASES),
        "reference_generated_ids": sum(r["tokens"] for r in references.values()),
        "native_cases": len(rows), "stages_per_case": 75,
        "decoded_exact_cases": sum(r["decoded_pass"] for r in rows),
        "all_inputs_exact": all(r["input_patches_grid_prompt_mrope_exact"] for r in rows),
        "all_coverage_complete": all(r["coverage_complete"] for r in rows),
        "all_confidences_valid": all(r["confidence_pass"] for r in rows),
        "all_reached_eos": all(r["reached_eos"] for r in rows),
        "strict_numerical_qualification_pass": all(r["numerical_pass"] for r in rows),
        "frozen_thresholds": {"input_patches_max_abs": 0.000002,
                              "other_float_max_abs": 0.002,
                              "other_float_min_row_cosine": 0.99999,
                              "integer_arrays": "exact",
                              "generated_ids_and_text_and_first_logits_argmax": "exact"},
        "formats": formats, "cases": rows,
        "summary_producer_sha256": sha256(Path(__file__)),
    }
    (root / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("native_cases", "decoded_exact_cases", "all_reached_eos",
                                           "strict_numerical_qualification_pass", "formats")}, indent=2))


if __name__ == "__main__":
    main()
