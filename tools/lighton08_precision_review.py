#!/usr/bin/env python3
"""Re-read retained full-vocabulary logits under the newer owner precision bars.

No inference, changed budget, revised historical verdict or current-runtime
qualification is performed. Missing positions or a changed source panel fail.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "tests/results/qwen35-ocr"
MODEL = "lightonai/LightOnOCR-3-0.8B"
REVISION = "36d8636925327c671bd8e479c517a004f7566bd9"
CASES = {"historical-receipt": 473, "two-column-page": 683, "inventory-table": 248}
BARS = {"f16": 0.98, "q8": 0.97, "q4": 0.94, "q8-q6": None}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def case_review(case_id, arm, reference, profile):
    count = CASES[case_id]
    ids = reference["generated_ids"]
    require(len(ids) == count and reference["max_new_tokens"] == 1024, "changed frozen reference/cap")
    decode = arm["decode"]
    teacher = decode["teacher_forced"]
    require(len(teacher) == count and all(row["step"] == i and row["reference"] == ids[i]
            for i, row in enumerate(teacher)), "missing/changed teacher-forced positions")
    scores = [row for row in decode["metrics"] if row["stage"].startswith("generated.scores.")]
    require([row["stage"] for row in scores] == [f"generated.scores.{i}" for i in range(count)],
            "missing/reordered/duplicate full-vocabulary logits")
    require(all(row["elements"] == 248320 and math.isfinite(row["cos_min"])
                and -1 <= row["cos_min"] <= 1 for row in scores), "invalid vocabulary/cosine")
    worst = min(scores, key=lambda row: row["cos_min"])
    intermediate = [row for row in arm["prefill"]["metrics"]
                    if not row["stage"].endswith("logits")]
    require(bool(intermediate) and all(math.isfinite(row["cos_min"]) for row in intermediate),
            "missing/nonfinite intermediate diagnostics")
    stage = min(intermediate, key=lambda row: row["cos_min"])
    threshold = BARS[profile]
    return {"case": case_id, "teacher_forced_positions": count,
            "minimum_final_logit_cosine": worst["cos_min"], "worst_logit_stage": worst["stage"],
            "owner_bar": threshold, "exceeds_bar_on_recorded_build":
                None if threshold is None else worst["cos_min"] > threshold,
            "minimum_prefill_intermediate_cosine": stage["cos_min"], "worst_prefill_stage": stage["stage"],
            "strict_prefill_failures_retained": len(arm["prefill"]["qualified_failed_stages"]),
            "ids_exact": decode["ids_exact"], "text_exact": decode["text_exact"],
            "complete_document": decode["complete_document"],
            "first_id_difference": decode["first_id_difference"],
            "native_quality": arm["native_quality"]}


def review():
    receipts = []
    groups = {key: [] for key in BARS}
    def read(name):
        path = RESULTS / name
        payload = path.read_bytes()
        record = json.loads(payload)
        require(record["model"] == MODEL and record["revision"] == REVISION, "changed checkpoint identity")
        receipts.append({"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(payload).hexdigest(),
                         "source_commit": record["source_commit"], "workflow": record["workflow"],
                         "build_record": record.get("build"), "historical_result": record["result"]})
        return record
    original = read("quality.json")
    require({case["id"] for case in original["cases"]} == set(CASES), "changed F16 panel")
    for case in original["cases"]:
        groups["f16"].append(case_review(case["id"], case, case["reference"], "f16"))
    for profile in ("q8", "q8-q6", "q4"):
        for case_id in CASES:
            record = read(f"quality-quant-{profile}-{case_id}.json")
            require(record["profile"] == profile and record["case"]["id"] == case_id,
                    "changed recipe/case identity")
            require(len(record["arms"]) == 2 and record["arms"][0]["name"] == "f16-baseline",
                    "changed candidate/control panel")
            candidate = record["arms"][1]
            require(candidate["frozen_quantized_weights_exact"] is True
                    and candidate["protected_tensor_bytes_exact"] is True, "unverified candidate bytes")
            groups[profile].append(case_review(case_id, candidate, record["reference"], profile))
    return {"schema": "crispembed-lighton08-retained-precision-review-v1", "model": MODEL, "revision": REVISION,
            "policy": "docs/precision-acceptance.md (1/3 November 2026)", "receipts": receipts,
            "recipes": [{"profile": profile, "minimum_final_logit_cosine": min(c["minimum_final_logit_cosine"] for c in cases),
                         "owner_bar": BARS[profile], "all_recorded_cases_exceed_bar":
                             None if BARS[profile] is None else all(c["exceeds_bar_on_recorded_build"] for c in cases),
                         "cases": cases} for profile, cases in groups.items()],
            "model_execution": False, "historical_verdicts_changed": False,
            "current_runtime_qualified": False, "publication_qualified": False,
            "limitations": ["Recorded logits use original teacher-forced prefixes, not native free-generation prefixes.",
                            "F16 records GGML_NATIVE=false; quantized receipts have no explicit portable-build record.",
                            "Current source and all precision policy build flags require a fresh pinned replay.",
                            "The owner policy gives no separate Q6 mixed bar; do not label Q8/Q6 as Q8_0.",
                            "Keep text/structure/completion regressions and all original strict failures."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = review()
        text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
        else:
            print(text, end="")
        return 0
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(json.dumps({"error": str(exc), "current_runtime_qualified": False}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
