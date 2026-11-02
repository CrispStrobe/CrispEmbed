#!/usr/bin/env python3
"""Run the prepared native public/binding/parser suite sequentially on CPU.

Keep this executed source, source-snapshot/, commands, all logs and reports.
The expected binding text comes from the bounded C API anchor and is not an
independent reference-model result. Never overwrite an earlier result set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from PIL import Image


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--model", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--skip-parser", action="store_true")
    args = ap.parse_args()
    workspace = args.workspace.resolve(strict=True)
    repo = workspace / "CrispEmbed-TeleOCR"
    build = workspace / "build-teleocr"
    bindings = workspace / "teleocr-bindings-check"
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        ap.error("output directory is not empty; retain previous evidence and use a new directory")
    model = (args.model or workspace / "teleocr-models/teleocr-f16.gguf").resolve(strict=True)
    library = build / "libcrispembed.so"
    binaries = {
        "library": library, "cli": build / "crispembed", "server": build / "crispembed-server",
        "rust": bindings / "rust/target/release/teleocr-public-probe",
        "dart": bindings / "dart/teleocr-public-probe",
    }
    compilation = json.loads((bindings / "build-checks.json").read_text())
    for relative, expected in compilation["source_sha256"].items():
        if digest(repo / relative) != expected:
            raise ValueError(f"Binding source changed after its retained build: {relative}")
    for language in ("rust", "dart"):
        if digest(binaries[language]) != compilation[language]["consumer_sha256"]:
            raise ValueError(f"{language} binary differs from the retained successful build")

    out.mkdir(parents=True, exist_ok=True)
    snapshot = out / "source-snapshot"
    snapshot.mkdir()
    sources = list(compilation["source_sha256"])
    sources += [str(path.relative_to(repo)) for path in (repo / "python/crispembed").glob("*.py")]
    sources += [
        "tests/test_teleocr_public.py", "tests/test_teleocr_parser_native.py", "tools/teleocr_parse.py",
        "src/crispembed.cpp", "src/crispembed.h", "src/ocr_orchestrator.cpp", "src/ocr_orchestrator.h",
        "src/qwen2vl_ocr.cpp", "src/qwen2vl_ocr.h", "examples/cli/main.cpp", "examples/server/server.cpp",
        "crispembed/Cargo.toml", "crispembed-sys/Cargo.toml", "crispembed-sys/build.rs",
        "LICENSE", "NOTICE.teleocr", "models/licenses/apache-2.0/LICENSE", "docs/teleocr-license.md",
    ]
    source_hashes = {}
    for relative in dict.fromkeys(sources):
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / relative, destination)
        source_hashes[relative] = digest(destination)
    shutil.copyfile(__file__, out / "run_public_suite.executed.py")
    shutil.copyfile(bindings / "build-checks.json", out / "binding-build-checks.json")
    table = out / "table.png"
    synthetic = out / "synthetic-text.png"
    shutil.copyfile(workspace / "validation/reference/table.png", table)
    shutil.copyfile(repo / "tests/fixtures/teleocr/synthetic-text.png", synthetic)
    with Image.open(table) as input_image:
        rgb = input_image.convert("RGB")
        width, height = rgb.size
        (out / "table.rgb").write_bytes(rgb.tobytes())

    phases = []
    summary = {
        "kind": "teleocr-public-and-parser-validation", "independent_model_parity": False,
        "backend": "cpu", "model": model.name, "model_sha256": digest(model),
        "binary_sha256": {name: digest(path) for name, path in binaries.items()},
        "source_sha256": source_hashes, "runner_sha256": digest(__file__),
        "image_sha256": {path.name: digest(path) for path in (table, synthetic, out / "table.rgb")},
        "public_max_tokens": 8, "parser_max_tokens": 1024, "threads": 2,
        "requested_scope": "public-interfaces" if args.skip_parser else "public-interfaces-and-parser",
        "phases": phases, "passed": False,
    }
    write_json(out / "suite.json", summary)
    environment = dict(os.environ, CRISPEMBED_TELEOCR="1", CRISPEMBED_TELEOCR_GPU="0",
                       QWEN2VL_OCR_FORCE_CPU="1", PYTHONUNBUFFERED="1", DART_SUPPRESS_ANALYTICS="1")
    environment["LD_LIBRARY_PATH"] = str(build) + ":" + str(build / "ggml/src") + ":" + environment.get("LD_LIBRARY_PATH", "")
    environment.pop("CRISPEMBED_TELEOCR_DUMP_STAGES", None)

    def run(name, command):
        record = {"name": name, "command": [str(value) for value in command], "passed": False}
        phases.append(record)
        write_json(out / "suite.json", summary)
        print(f"Starting {name}", flush=True)
        started = time.monotonic()
        try:
            with (out / f"{name}.stdout.log").open("w", encoding="utf-8") as stdout, \
                 (out / f"{name}.stderr.log").open("w", encoding="utf-8") as stderr:
                result = subprocess.run(record["command"], cwd=snapshot, env=environment,
                                        stdout=stdout, stderr=stderr, timeout=5400)
            record["exit_code"] = result.returncode
            record["passed"] = result.returncode == 0
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        record["elapsed_seconds"] = time.monotonic() - started
        write_json(out / "suite.json", summary)
        print(json.dumps(record), flush=True)
        return record["passed"]

    public_report = out / "public.json"
    run("public", [sys.executable, snapshot / "tests/test_teleocr_public.py", "--lib", library,
                   "--cli", binaries["cli"], "--server", binaries["server"], "--model", model,
                   "--image", table, "--max-tokens", "8", "--threads", "2", "--report", public_report])
    public = json.loads(public_report.read_text()) if public_report.exists() else {}
    anchor = public.get("anchor_text")
    if isinstance(anchor, str) and anchor:
        expected = out / "anchor.txt"
        expected.write_text(anchor, encoding="utf-8")
        summary["anchor_text"] = anchor
        summary["anchor_sha256"] = digest(expected)
        summary["anchor_has_non_ascii"] = any(ord(character) > 127 for character in anchor)
        phases.append({"name": "multilingual-anchor", "passed": summary["anchor_has_non_ascii"]})
        common = [model, out / "table.rgb", str(width), str(height), table]
        run("rust", [binaries["rust"], *common, expected, "8"])
        run("dart", [binaries["dart"], *common, library, expected, "8"])
    else:
        phases.extend({"name": name, "skipped": "No successful C API anchor was produced", "passed": False}
                      for name in ("rust", "dart"))
    if not args.skip_parser:
        for mode, image in (("detection", synthetic), ("segmentation", table)):
            run("parser-" + mode, [sys.executable, snapshot / "tests/test_teleocr_parser_native.py",
                                   "--lib", library, "--model", model, "--image", image, "--mode", mode,
                                   "--out", out / ("parser-" + mode), "--max-tokens", "1024", "--threads", "2"])
    else:
        phases.append({"name": "parser", "skipped": "Explicit --skip-parser request"})
    summary["passed"] = all(record.get("passed", False) for record in phases if "skipped" not in record)
    write_json(out / "suite.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
