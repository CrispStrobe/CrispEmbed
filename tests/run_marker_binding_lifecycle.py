#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build and execute real Rust/Dart Marker consumers against owned native fixtures.

Hosted Linux only. Reuses an existing native build; no model or real PDFium is
downloaded. Package resolution is needed to compile the isolated consumers.
Each probe runs in a fresh process with explicit gates. Crashes, timeouts and
missing result records fail the report rather than counting as skipped checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ("artifacts-off", "incremental-off", "enabled")
SOURCE_FILES = (
    "tests/run_marker_binding_lifecycle.py",
    "tests/marker_public_bindings.rs",
    "tests/marker_public_bindings.dart",
    "tests/marker_pdfium_fixture.cpp",
    "crispembed/src/marker.rs",
    "crispembed/src/lib.rs",
    "crispembed-sys/src/lib.rs",
    "crispembed-sys/build.rs",
    "flutter/crispembed/lib/src/marker.dart",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--pdfium", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not sys.platform.startswith("linux"):
        parser.error("this hosted consumer runner supports Linux")
    library, fixture, report_path = args.lib.resolve(), args.pdfium.resolve(), args.report.resolve()
    if not library.is_file() or not fixture.is_file():
        parser.error("native library and owned PDFium fixture must already be built")
    log_dir = report_path.with_suffix("")
    if log_dir == report_path or (log_dir.exists() and any(log_dir.iterdir())):
        parser.error("report must have an extension and an absent or empty companion log directory")
    log_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "crispembed.marker-binding-lifecycle/1",
        "scope": "owned ABI fixture; no real PDF rendering or model inference",
        "library_sha256": sha256(library),
        "pdfium_fixture_sha256": sha256(fixture),
        "source_sha256": {name: sha256(ROOT / name) for name in SOURCE_FILES},
        "commands": [],
        "probes": [],
        "errors": [],
        "passed": False,
    }
    environment = os.environ.copy()
    environment.update({
        "CRISPEMBED_SYS_LIB_DIR": str(library.parent),
        "CRISPEMBED_MARKER": "1",
        "CRISPEMBED_MARKER_SOURCE_LINKS": "1",
        "CRISPEMBED_MARKER_REFERENCES": "1",
        "CRISPEMBED_MARKER_MULTILINGUAL_NORMALIZE": "0",
        "CRISPEMBED_MARKER_REPETITION_CHECK": "0",
        "CRISPEMBED_TELEOCR": "0",
        "CRISPEMBED_ACCEPT_QWEN_RESEARCH_LICENSE": "0",
        "DART_SUPPRESS_ANALYTICS": "1",
    })
    # The source build keeps ggml siblings below ggml/src; installed bundles
    # keep them alongside the main library. Preserve the caller's other paths.
    runtime_paths = [library.parent, library.parent / "ggml/src"]
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(
        [str(path) for path in runtime_paths] + [environment.get("LD_LIBRARY_PATH", "")]
    )

    def run(label, command, *, cwd, env=environment, timeout=180):
        log = log_dir / (label + ".log")
        record = {"name": label, "log": log.name, "exit_code": None, "timed_out": False}
        try:
            with log.open("wb") as output:
                process = subprocess.Popen(command, cwd=cwd, env=env, stdout=output,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    record["exit_code"] = process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    record["timed_out"] = True
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        except OSError as error:
            record["error"] = type(error).__name__
        record["passed"] = record["exit_code"] == 0 and not record["timed_out"]
        report["commands"].append(record)
        text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
        print(f"{label}: {'pass' if record['passed'] else 'FAIL'}", flush=True)
        if not record["passed"]:
            print(text[-8000:], flush=True)
        return record["passed"], text

    def probe(client, executable, prefix, scenario, cwd):
        env = environment.copy()
        env["CRISPEMBED_MARKER_ARTIFACTS"] = "0" if scenario == "artifacts-off" else "1"
        env["CRISPEMBED_MARKER_INCREMENTAL"] = "0" if scenario == "incremental-off" else "1"
        passed, output = run(client.lower() + "-" + scenario,
                             [str(executable), "--lifecycle", *prefix, scenario], cwd=cwd, env=env)
        records = []
        for line in output.splitlines():
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if isinstance(value, dict) and value.get("client") == client:
                records.append(value)
        result = records[0] if len(records) == 1 else {}
        total = result.get("total")
        checks = result.get("checks")
        valid = (len(records) == 1 and result.get("scenario") == scenario
                 and type(total) is int and total > 0
                 and type(result.get("passed")) is int and result["passed"] == total
                 and isinstance(checks, list) and len(checks) == total
                 and all(isinstance(row, dict) and isinstance(row.get("name"), str)
                         and row["name"] and row.get("passed") is True for row in checks)
                 and result.get("model_weights_used") is False
                 and result.get("pdf_rendering_qualified") is False)
        report["probes"].append({"client": client, "scenario": scenario,
                                 "passed": passed and valid, "result": result})

    try:
        for tool in ("cargo", "rustc", "dart"):
            if not run(tool + "-version", [tool, "--version"], cwd=ROOT, timeout=30)[0]:
                raise RuntimeError(f"required tool unavailable: {tool}")
        with tempfile.TemporaryDirectory(prefix="marker-binding-lifecycle-") as directory:
            work = Path(directory)
            rust, dart = work / "rust", work / "dart"
            (rust / "src").mkdir(parents=True)
            shutil.copyfile(ROOT / "tests/marker_public_bindings.rs", rust / "src/main.rs")
            (rust / "Cargo.toml").write_text(
                '[package]\nname = "marker-probe"\nversion = "0.1.0"\nedition = "2021"\n'
                '[dependencies]\ncrispembed = { path = ' + json.dumps(str(ROOT / "crispembed")) + ' }\n'
                'serde_json = "=1.0.145"\n', encoding="utf-8")
            # Avoid an inherited CARGO_TARGET_DIR hiding the probe binary.
            environment["CARGO_TARGET_DIR"] = str(rust / "target")
            rust_ok = run("rust-build", ["cargo", "build", "--release", "--manifest-path",
                                        str(rust / "Cargo.toml")], cwd=rust, timeout=600)[0]
            if (rust / "Cargo.lock").is_file():
                shutil.copyfile(rust / "Cargo.lock", log_dir / "Cargo.lock")
            if rust_ok:
                for scenario in SCENARIOS:
                    probe("Rust", rust / "target/release/marker-probe", [str(fixture)], scenario, rust)

            (dart / "bin").mkdir(parents=True)
            (dart / "lib").mkdir()
            # Compile the actual shipped Marker wrapper, not a handwritten FFI replica.
            shutil.copyfile(ROOT / "flutter/crispembed/lib/src/marker.dart", dart / "lib/marker.dart")
            shutil.copyfile(ROOT / "tests/marker_public_bindings.dart", dart / "bin/main.dart")
            (dart / "pubspec.yaml").write_text(
                'name: marker_binding_test\nenvironment:\n  sdk: ">=3.0.0 <4.0.0"\n'
                'dependencies:\n  ffi: 2.2.0\n', encoding="utf-8")
            dart_ok = run("dart-resolve", ["dart", "pub", "get"], cwd=dart, timeout=300)[0]
            if (dart / "pubspec.lock").is_file():
                shutil.copyfile(dart / "pubspec.lock", log_dir / "pubspec.lock")
            if dart_ok:
                dart_ok = run("dart-analyze", ["dart", "analyze", "bin", "lib"], cwd=dart)[0]
            if dart_ok:
                dart_ok = run("dart-build", ["dart", "compile", "exe", "bin/main.dart", "-o", "probe"],
                              cwd=dart, timeout=300)[0]
            if dart_ok:
                for scenario in SCENARIOS:
                    probe("Dart", dart / "probe", [str(library), str(fixture)], scenario, dart)
    except Exception as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
    expected = {(client, scenario) for client in ("Rust", "Dart") for scenario in SCENARIOS}
    observed = {(row["client"], row["scenario"]) for row in report["probes"]}
    report["passed"] = (not report["errors"] and observed == expected
                        and all(row["passed"] for row in report["commands"])
                        and all(row["passed"] for row in report["probes"]))
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"kind": report["schema"], "passed": report["passed"],
                      "probes": len(report["probes"]), "errors": report["errors"]}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
