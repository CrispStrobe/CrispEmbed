#!/usr/bin/env python3
"""Hosted test-only ggml parity prototype, using immutable original reference code."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from ultrabert_contract import sha256

FROZEN_REFERENCE_SOURCE = "0dddbd2b61740c0f25482b2afb428b815b5c85ac"


def parse_log(log):
    stages, exact, totals = {}, {}, None
    for line in log.splitlines():
        fields = line.split("\t")
        if fields[0] == "STAGE":
            if len(fields) != 9 or fields[1] in stages:
                raise ValueError("Invalid/duplicate native stage")
            values = [float(v) for v in fields[2:8]]
            if any(math.isfinite(v) and v < 0 for v in values[1:]):
                raise ValueError("Negative error/norm in native log")
            passed = all(math.isfinite(v) for v in values) and values[0] >= 0.99999 and values[1] <= 0.002
            if int(fields[8]) != int(passed):
                raise ValueError("Native stage verdict mismatch")
            stages[fields[1]] = dict(zip(("cos_min", "max_abs", "mean_abs", "rms", "native_norm", "reference_norm"),
                                        [v if math.isfinite(v) else None for v in values]))
            stages[fields[1]]["pass"] = passed
        elif fields[0] == "EXACT":
            if len(fields) != 3 or fields[1] in exact or fields[2] not in ("0", "1"):
                raise ValueError("Invalid/duplicate exact control")
            exact[fields[1]] = fields[2] == "1"
        elif fields[0] == "TOTAL":
            if len(fields) != 4 or totals is not None:
                raise ValueError("Invalid/duplicate native total")
            totals = [int(v) for v in fields[1:]]
    failures = sum(not row["pass"] for row in stages.values()) + sum(not v for v in exact.values())
    if len(stages) != 292 or len(exact) != 50 or totals != [292, 50, failures]:
        raise ValueError("Incomplete native comparisons")
    return dict(stages=stages, exact=exact, failures=failures)



def verify_frozen_reference(baseline, current):
    if (baseline.get("source_commit") != FROZEN_REFERENCE_SOURCE
            or baseline.get("status") != "completed conversion/reference audit; native parity not evaluated"
            or not baseline.get("reference") or not baseline.get("conversions")):
        raise ValueError("Incomplete/wrong-source frozen reference report")
    if (baseline["reference"] != current.get("reference")
            or baseline["conversions"] != current.get("conversions")):
        raise ValueError("Frozen original arrays/conversion records drifted")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--frozen-report", type=Path,
                        help="Reviewed complete original report at the immutable pre-port source")
    args = parser.parse_args()
    if not args.build_only and (args.frozen_report is None or not args.frozen_report.is_file()):
        raise ValueError("Review and freeze the completed original report before native model comparisons")
    output = Path(os.environ["HEAVY_OUT"])
    scratch = Path(os.environ["HEAVY_SCRATCH"]) / "ultrabert"
    output.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "tests/test_ultrabert_contract.py", "-v"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "tests/test_ultrabert_native_log.py", "-v"], cwd=ROOT, check=True)
    # Actions checkout is shallow: obtain the immutable original before comparing
    # bytes. This cannot silently accept a changed dumper as a new blueprint.
    subprocess.run(["git", "fetch", "--no-tags", "--depth=1", "origin", FROZEN_REFERENCE_SOURCE], cwd=ROOT, check=True)
    # Freeze the original and conversion code before the first native output.
    for name in ("tools/ultrabert_contract.py", "models/convert-ultrabert-to-gguf.py",
                 "tools/dump_ultrabert_reference.py", "tools/ci-heavy/ultrabert_reference.py",
                 "tests/results/ultrabert/source-audit.json", "tests/results/ultrabert/embed-weight-schema.json"):
        old = subprocess.check_output(["git", "show", FROZEN_REFERENCE_SOURCE + ":" + name], cwd=ROOT)
        if hashlib.sha256(old).hexdigest() != sha256(ROOT / name):
            raise ValueError("Original reference/conversion source drift: " + name)
    build = scratch / "native-build"
    subprocess.run(["cmake", "-S", str(ROOT), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
                    "-DGGML_NATIVE=OFF", "-DGGML_BLAS=OFF", "-DGGML_METAL=OFF",
                    "-DCRISPEMBED_BUILD_SHARED=OFF"], check=True)
    subprocess.run(["cmake", "--build", str(build), "--target", "test-ultrabert-diff", "-j2"], check=True)
    binary = next(p for p in (build / "test-ultrabert-diff", build / "bin/test-ultrabert-diff") if p.is_file())
    env = dict(os.environ, CRISPEMBED_ULTRABERT="0")
    control = subprocess.run([str(binary), "absent.gguf", "absent-ref.gguf"], env=env, capture_output=True, text=True)
    if control.returncode != 2:
        raise ValueError("Literal-zero opt-in control failed")
    default_env = dict(os.environ)
    default_env.pop("CRISPEMBED_ULTRABERT", None)
    default_control = subprocess.run([str(binary), "absent.gguf", "absent-ref.gguf"],
                                     env=default_env, capture_output=True, text=True)
    if default_control.returncode != 2:
        raise ValueError("Default-off opt-in control failed")
    report = {"source_commit": os.environ["GITHUB_SHA"], "frozen_reference_source": FROZEN_REFERENCE_SOURCE,
              "native_cpp_sha256": sha256(ROOT / "tests/test_ultrabert_diff.cpp"),
              "driver_sha256": sha256(__file__), "build_passed": True,
              "literal_zero_gate_off": True, "default_gate_off": True, "arms": [],
              "scope": "test-only CPU graph over original prepared IDs; not tokenizer/public API or task quality",
              "status": "build-only; model parity not evaluated"}
    target = output / "ultrabert-native.json"
    target.write_text(json.dumps(report, indent=2) + "\n")
    if args.build_only:
        return
    import ultrabert_reference
    ultrabert_reference.main()
    baseline = json.loads(args.frozen_report.read_text())
    current = json.loads((output / "ultrabert-reference-audit.json").read_text())
    verify_frozen_reference(baseline, current)
    report["frozen_report_sha256"] = sha256(args.frozen_report)
    report["original_and_conversion_replay_exact"] = True
    reference = current["reference"]
    env["CRISPEMBED_ULTRABERT"] = "1"
    for dtype in ("f32", "f16"):
        for case in reference["cases"]:
            proc = subprocess.run([str(binary), str(scratch / (dtype + ".gguf")),
                                   str(output / "reference" / case["archive"])],
                                  env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            log = output / (dtype + "-" + case["case"]["id"] + ".log")
            log.write_text(proc.stdout)
            if proc.returncode not in (0, 1):
                raise ValueError("Native structural/compute error; see " + log.name)
            audit = parse_log(proc.stdout)
            if proc.returncode != int(audit["failures"] > 0):
                raise ValueError("Native exit code disagrees with retained failures")
            ordered = ["input_ids", "attention_mask"] + [name for name in case["stages"]
                       if (name in audit["stages"] or name in audit["exact"])
                       and name not in ("input_ids", "attention_mask")]
            if len(ordered) != 342:
                raise ValueError("Compared stages differ from original reference")
            # Original capture order, never alphabetical map order, defines the
            # earliest difference. Input IDs here are supplied, not retokenized.
            first = next((name for name in ordered if not (audit["stages"][name]["pass"]
                         if name in audit["stages"] else audit["exact"][name])), None)
            audit.update(dtype=dtype, case=case["case"]["id"], first_failed_stage=first,
                         reference_archive_sha256=case["archive_sha256"], log=log.name,
                         log_sha256=sha256(log), compared_order=ordered,
                         uncaptured_native_stages=[name for name in case["stages"] if name not in ordered])
            report["arms"].append(audit)
            target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    report["status"] = "completed prepared-ID native comparisons; all failures retained"
    report["failures"] = sum(arm["failures"] for arm in report["arms"])
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if report["failures"]:
        raise AssertionError("UltraBERT strict native comparisons failed; see retained report/logs")


if __name__ == "__main__":
    main()
