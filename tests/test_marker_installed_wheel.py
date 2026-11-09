#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build and install a Linux wheel, then exercise its real native dependency closure.

Run in an equipped Python environment with setuptools, wheel, pip, numpy and
pypdfium2. The consumer venv inherits only the interpreter's system packages;
the newly built CrispEmbed wheel is installed there offline with --no-deps.
The native prefix must be a relocatable CMake install with sibling GGML libs.
No model weights, publication, package downloads or release tag are used.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native-prefix", "pdfium", "work-dir", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("this installed-wheel probe is scoped to Linux")
    prefix, pdfium = args.native_prefix.resolve(strict=True), args.pdfium.resolve(strict=True)
    work, output = args.work_dir.resolve(), args.output.resolve()
    for directory in (work, output):
        if directory.exists():
            parser.error("work/output must be new directories so every attempt is retained")
        directory.mkdir(parents=True)
    env = os.environ.copy()
    for key in ("LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONPATH", "PYTHONHOME"):
        env.pop(key, None)
    env.update(PIP_NO_INDEX="1", PIP_DISABLE_PIP_VERSION_CHECK="1", PYTHONNOUSERSITE="1",
               CRISPEMBED_OFFLINE="1", CRISPEMBED_ACCEPT_QWEN_RESEARCH_LICENSE="0",
               CRISPEMBED_ACCEPT_CC_BY_NC_LICENSE="0")
    checks, commands = [], []
    replacements = {str(work): "<WORK>", str(output): "<OUTPUT>", str(ROOT): "<REPOSITORY>",
                    str(prefix): "<NATIVE_PREFIX>", str(pdfium): "<PDFIUM>",
                    str(Path(sys.executable).resolve().parent.parent): "<PYTHON_RUNTIME>"}

    def sanitize(value):
        for original, replacement in sorted(replacements.items(), key=lambda item: -len(item[0])):
            value = value.replace(original, replacement)
        return value

    def check(name, passed):
        checks.append({"name": name, "passed": bool(passed)})
        if not passed:
            raise AssertionError(name)

    def run(label, command, cwd=work, timeout=120):
        result = subprocess.run(command, cwd=cwd, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
        (output / (label + ".log")).write_text(sanitize(result.stdout), encoding="utf-8")
        commands.append({"label": label, "command": [sanitize(str(part)) for part in command],
                         "returncode": result.returncode})
        check(label + " exits successfully", result.returncode == 0)
        return result.stdout.strip()

    source_paths = ["tests/test_marker_installed_wheel.py", "tests/test_marker_real_pdf.py",
                    "python/setup.py", "python/pyproject.toml", "python/MANIFEST.in",
                    "python/crispembed/_binding.py", "python/crispembed/_marker.py"]
    source_hashes = {path: sha(ROOT / path) for path in source_paths}
    receipt = {"kind": "marker-installed-linux-wheel", "source_sha256": source_hashes,
               "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "python_version": sys.version.split()[0], "checks": checks, "commands": commands,
               "packages": {name: importlib.metadata.version(name)
                            for name in ("setuptools", "wheel", "numpy", "pypdfium2")},
               "scope": "Locally built Linux wheel and real PDF lifecycle; no manylinux, release, OCR or model qualification.",
               "environment": {"library_path_overrides_removed": True, "isolated_consumer_python": True,
                               "venv_system_packages": True, "network_or_publication_used": False},
               "pdfium_sha256": sha(pdfium)}
    try:
        checkout = work / "checkout"
        staged = checkout / "python"
        shutil.copytree(ROOT / "python", staged, ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", "build", "dist", "*.egg-info", "*.so", "*.so.*", "*.dylib", "*.dll", "*.pyd"))
        notices = {"NOTICE.marker": ROOT / "NOTICE.marker", "NOTICE.teleocr": ROOT / "NOTICE.teleocr",
                   "Apache-2.0.txt": ROOT / "models/licenses/apache-2.0/LICENSE"}
        for name, source in notices.items():
            target = checkout / name if name != "Apache-2.0.txt" else checkout / "models/licenses/apache-2.0/LICENSE"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        shutil.copyfile(ROOT / "POLICY.md", checkout / "POLICY.md")
        native = {path.name: path for pattern in ("libcrispembed.so*", "libggml*.so*")
                  for path in (prefix / "lib").glob(pattern) if path.is_file()}
        check("prefix contains CrispEmbed and all three GGML families",
              all(name in native for name in ("libcrispembed.so", "libggml.so.0", "libggml-cpu.so.0", "libggml-base.so.0")))
        for name, path in native.items():
            shutil.copyfile(path, staged / "crispembed" / name)
        receipt["native_files_sha256"] = {name: sha(path) for name, path in sorted(native.items())}
        receipt["notices_sha256"] = {name: sha(path) for name, path in notices.items()}
        run("build-wheel", [sys.executable, "setup.py", "bdist_wheel", "--dist-dir", str(work / "wheelhouse")], cwd=staged)
        wheels = list((work / "wheelhouse").glob("*.whl"))
        check("one platform wheel produced", len(wheels) == 1 and "linux" in wheels[0].name)
        wheel = wheels[0]
        receipt["wheel"] = {"filename": wheel.name, "bytes": wheel.stat().st_size, "sha256": sha(wheel)}
        with zipfile.ZipFile(wheel) as archive:
            for name, path in {**native, **notices}.items():
                check("wheel exact bytes: " + name, archive.read("crispembed/" + name) == path.read_bytes())
        venv = work / "consumer-venv"
        run("create-venv", [sys.executable, "-m", "venv", "--system-site-packages", "--without-pip", str(venv)])
        consumer_python = venv / "bin/python"
        # CI installs numpy/pypdfium2 with `pip --user` when the system
        # site-packages is read-only; neither --system-site-packages nor the
        # consumer's `python -I` exposes the user site. Point the venv at the
        # directories the host actually imports them from (appended after the
        # venv's own site-packages, so the installed wheel still wins).
        host_dirs = sorted({str(Path(__import__(name).__file__).resolve().parent.parent)
                            for name in ("numpy", "pypdfium2")})
        consumer_site = next((venv / "lib").glob("python*/site-packages"))
        (consumer_site / "host-dependencies.pth").write_text("\n".join(host_dirs) + "\n", encoding="utf-8")
        receipt["environment"]["host_dependency_dirs"] = host_dirs
        run("install-wheel", [sys.executable, "-m", "pip", "--python", str(consumer_python), "install",
                              "--no-index", "--no-deps", str(wheel)])
        consumer = work / "consumer"
        consumer.mkdir()
        package = Path(run("installed-location", [str(consumer_python), "-I", "-c",
                           "from pathlib import Path; import crispembed; print(Path(crispembed.__file__).resolve().parent)"], cwd=consumer))
        check("binding actually installed in the consumer venv", package.is_relative_to(venv))
        for name, path in {**native, **notices}.items():
            check("installed exact bytes: " + name, sha(package / name) == sha(path))
        lifecycle = output / "real-pdf"
        run("installed-real-pdf", [str(consumer_python), "-I", "-B", str(ROOT / "tests/test_marker_real_pdf.py"),
                                   "--installed-package", "--lib", str(package / "libcrispembed.so"),
                                   "--pdfium", str(pdfium), "--out", str(lifecycle)], cwd=consumer, timeout=150)
        report = json.loads((lifecycle / "receipt.json").read_text())
        check("actual installed lifecycle and loaded dependency provenance pass",
              report["passed"] and report["installed_package"] and report["total_count"] == 51)
        receipt["lifecycle"] = {"passed": report["passed"], "checks": report["total_count"],
                                "receipt_sha256": sha(lifecycle / "receipt.json")}
        check("packaging and probe sources unchanged during execution",
              source_hashes == {path: sha(ROOT / path) for path in source_paths})
    except Exception as exc:
        receipt["error"] = sanitize(f"{type(exc).__name__}: {exc}")
    receipt["passed"] = "error" not in receipt and bool(checks) and all(row["passed"] for row in checks)
    receipt["passed_count"] = sum(row["passed"] for row in checks)
    receipt["total_count"] = len(checks)
    receipt["logs_sha256"] = {path.name: sha(path) for path in output.glob("*.log")}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"Installed wheel: {receipt['passed_count']}/{len(checks)} checks, passed={receipt['passed']}")
    if "error" in receipt:
        print(receipt["error"], file=sys.stderr)
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
