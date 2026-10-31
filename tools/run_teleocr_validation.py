#!/usr/bin/env python3
"""Run the native TeleOCR differential executable against frozen references.

Each case retains its log and machine-readable numerical/decoded result even
when another case fails. The reference prompt, image, generation cap, and
comparison budgets are never changed by this runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", default=["text", "formula", "table", "code", "layout", "synthetic"])
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--dump-native-stages", action="store_true",
                        help="Retain raw F32 vision blocks 16/17 and decoder blocks 1/2 for arithmetic diagnosis")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("output directory is not empty; retain previous failures and choose a new directory")
    if len(set(args.cases)) != len(args.cases):
        parser.error("duplicate case name")
    for case in args.cases:
        if not case or any(ch not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for ch in case):
            parser.error("case names must contain lowercase letters, digits, hyphens or underscores")
    binary = args.binary.resolve(strict=True)
    model = args.model.resolve(strict=True)
    refs = args.reference_dir.resolve(strict=True)
    args.out.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, CRISPEMBED_TELEOCR="1", CRISPEMBED_ACCEPT_QWEN_RESEARCH_LICENSE="1", CRISPEMBED_TELEOCR_GPU="0",
                       QWEN2VL_OCR_FORCE_CPU="1")
    summary = {
        "model": model.name, "model_sha256": sha256(model),
        "binary": binary.name, "binary_sha256": sha256(binary),
        "runner_sha256": sha256(Path(__file__)), "threads": args.threads,
        "backend": "cpu", "cases": [], "pass": False,
    }
    for case in args.cases:
        metadata_path = refs / f"{case}.json"
        reference = refs / f"{case}.gguf"
        image = refs / f"{case}.png"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        reference_hash = sha256(reference)
        image_hash = sha256(image)
        if reference_hash != metadata["reference_sha256"] or image_hash != metadata["image_sha256"]:
            raise ValueError(f"Frozen reference or image hash mismatch: {case}")
        cap = metadata["environment"]["max_new_tokens"]
        if not isinstance(cap, int) or cap < 1:
            raise ValueError(f"Invalid frozen token cap: {case}")
        report_path = args.out / f"{case}.json"
        log_path = args.out / f"{case}.log"
        command = [str(binary), str(model), str(reference), str(image),
                   str(report_path.resolve()), str(args.threads), str(cap)]
        case_environment = dict(environment)
        case_environment.pop("CRISPEMBED_TELEOCR_DUMP_STAGES", None)
        if args.dump_native_stages:
            dump_dir = args.out / f"{case}-native-stages"
            dump_dir.mkdir()
            case_environment["CRISPEMBED_TELEOCR_DUMP_STAGES"] = str(dump_dir.resolve())
        print(f"Starting {model.name}: {case}, frozen cap={cap}", flush=True)
        start = time.monotonic()
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(command, env=case_environment, stdout=log, stderr=subprocess.STDOUT)
        record = {
            "case": case, "exit_code": result.returncode,
            "elapsed_seconds": time.monotonic() - start,
            "reference_sha256": reference_hash, "image_sha256": image_hash,
            "max_new_tokens": cap, "reference_reached_eos": metadata["reached_eos"],
            "reference_tokens": metadata["tokens"], "log": log_path.name,
            "report": report_path.name, "pass": False,
        }
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            record.update({key: report[key] for key in (
                "numerical_pass", "decoded_pass", "coverage_complete", "confidence_pass",
                "reached_eos", "tokens", "seconds"
            )})
            record["pass"] = result.returncode == 0 and report["pass"]
            record["failed_stages"] = [stage["name"] for stage in report["stages"] if not stage["pass"]]
            record["report_sha256"] = sha256(report_path)
        else:
            record["error"] = "Native executable did not produce a result report; inspect the retained log"
        summary["cases"].append(record)
        summary["pass"] = len(summary["cases"]) == len(args.cases) and all(r["pass"] for r in summary["cases"])
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(record), flush=True)
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
