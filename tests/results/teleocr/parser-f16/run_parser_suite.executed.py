#!/usr/bin/env python3
"""Run the two native parser cases after the public suite's RAM window."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--public-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    public = args.public_dir.resolve(strict=True)
    workspace = public.parents[1]
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        ap.error("output directory is not empty")
    previous = json.loads((public / "suite.json").read_text())
    model = workspace / "teleocr-models" / previous["model"]
    library = workspace / "build-teleocr/libcrispembed.so"
    if sha256(model) != previous["model_sha256"] or sha256(library) != previous["binary_sha256"]["library"]:
        raise ValueError("Native model/library changed since the public interface run")
    out.mkdir(parents=True, exist_ok=True)
    source = out / "source-snapshot"
    shutil.copytree(public / "source-snapshot", source, ignore=shutil.ignore_patterns("__pycache__"))
    for relative, digest in previous["source_sha256"].items():
        if sha256(source / relative) != digest:
            raise ValueError(f"Source snapshot changed: {relative}")
    shutil.copyfile(__file__, out / "run_parser_suite.executed.py")
    for name in ("synthetic-text.png", "table.png"):
        shutil.copyfile(public / name, out / name)
    results = []
    summary = {"kind": "teleocr-native-parser-suite", "independent_model_parity": False,
               "model": model.name, "model_sha256": previous["model_sha256"],
               "library_sha256": previous["binary_sha256"]["library"],
               "source_sha256": previous["source_sha256"], "runner_sha256": sha256(__file__),
               "max_tokens": 1024, "threads": 2, "backend": "cpu", "cases": results, "passed": False}
    save(out / "suite.json", summary)
    environment = dict(os.environ, CRISPEMBED_TELEOCR="1", CRISPEMBED_TELEOCR_GPU="0",
                       QWEN2VL_OCR_FORCE_CPU="1", PYTHONUNBUFFERED="1")
    environment.pop("CRISPEMBED_TELEOCR_DUMP_STAGES", None)
    for mode, name in (("detection", "synthetic-text.png"), ("segmentation", "table.png")):
        command = [sys.executable, str(source / "tests/test_teleocr_parser_native.py"),
                   "--lib", str(library), "--model", str(model), "--image", str(out / name),
                   "--mode", mode, "--out", str(out / mode), "--max-tokens", "1024", "--threads", "2"]
        record = {"mode": mode, "command": command, "image_sha256": sha256(out / name), "passed": False}
        results.append(record)
        save(out / "suite.json", summary)
        print(f"Starting native parser: {mode}", flush=True)
        started = time.monotonic()
        try:
            with (out / f"{mode}.stdout.log").open("w", encoding="utf-8") as stdout, \
                 (out / f"{mode}.stderr.log").open("w", encoding="utf-8") as stderr:
                result = subprocess.run(command, cwd=source, env=environment, stdout=stdout, stderr=stderr, timeout=5400)
            record["exit_code"] = result.returncode
            report = out / mode / "report.json"
            if report.exists():
                parsed = json.loads(report.read_text())
                record.update(report=str(report.relative_to(out)), report_sha256=sha256(report),
                              generated_tokens=parsed.get("generated_tokens"), error=parsed.get("error"))
                record["passed"] = result.returncode == 0 and parsed["passed"]
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        record["elapsed_seconds"] = time.monotonic() - started
        summary["passed"] = len(results) == 2 and all(r["passed"] for r in results)
        save(out / "suite.json", summary)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
