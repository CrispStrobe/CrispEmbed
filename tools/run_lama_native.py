#!/usr/bin/env python3
"""Run native LaMa only after complete original capture and compare every case."""

import argparse
import json
import os
from pathlib import Path
import subprocess

from compare_lama_reference import compare, validate_original_report
from lama_reference_contract import CASES, fixture_identity, sha256


def run(binary, model, original_dir, output_dir, threads=2):
    binary, model = Path(binary).resolve(), Path(model).resolve()
    original_dir, output_dir = Path(original_dir).resolve(), Path(output_dir).resolve()
    if not binary.is_file() or not model.is_file():
        raise ValueError("native executable and converted model must exist")
    if not 1 <= threads <= 64:
        raise ValueError("threads must be in 1..64")
    validate_original_report(original_dir, sha256(model))
    output_dir.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ, CRISPEMBED_LAMA="1")
    for name in CASES:
        identity = fixture_identity(name)
        command = [str(binary), "--model", str(model), "--input-dir", str(original_dir / name),
                   "--output-dir", str(output_dir / name), "--width", str(identity["width"]),
                   "--height", str(identity["height"]), "--threads", str(threads)]
        try:
            completed = subprocess.run(command, env=environment, text=True, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, timeout=1200)
        except subprocess.TimeoutExpired as exc:
            def decoded(value):
                return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else (value or "")
            (output_dir / (name + ".log")).write_text(
                decoded(exc.stdout) + decoded(exc.stderr) + "\nnative case exceeded 1200 seconds\n", encoding="utf-8")
            raise RuntimeError(f"native LaMa exceeded 1200 seconds for {name}") from exc
        (output_dir / (name + ".log")).write_text(completed.stdout + completed.stderr, encoding="utf-8")
        if completed.returncode:
            raise RuntimeError(f"native LaMa failed for {name}; see {name}.log (exit {completed.returncode})")
        print(f"captured native LaMa case {name}", flush=True)
    report = compare(original_dir, output_dir)
    report["native_binary_sha256"] = sha256(binary)
    report["driver_sha256"] = sha256(__file__)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    report = run(args.binary, args.model, args.original_dir, args.output_dir, args.threads)
    print(json.dumps({"passed": report["passed"], "cases": len(report["cases"])}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
