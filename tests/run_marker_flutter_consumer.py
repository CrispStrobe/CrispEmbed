#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Validate the actual Flutter package, retained Pub archive and Linux consumer.

Requires a Flutter SDK, Linux desktop build prerequisites, a relocatable native
install prefix and real PDFium. Native libraries are staged after Pub archive
readback; the public package intentionally does not contain platform binaries.

    xvfb-run -a python tests/run_marker_flutter_consumer.py \
        --native-prefix build/install --pdfium /path/to/libpdfium.so \
        --out build/flutter-consumer --desktop

With --desktop, the caller supplies a working X11 or Wayland display. Without
it, the report explicitly describes headless Flutter tests and a built bundle.
This tool never publishes: Pub's --to-archive writes the validated bytes locally.
No model weights are used. Keep every output directory to preserve failures.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import time


ROOT = Path(__file__).resolve().parents[1]
NOTICE_NAMES = ("NOTICE.marker", "NOTICE.teleocr", "Apache-2.0.txt")
CONSUMER_TEST = """// SPDX-License-Identifier: MIT
import 'dart:convert';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:marker_flutter_consumer/main.dart';

void main() {
  testWidgets('actual package, asset, PDF and lifecycle consumer', (tester) async {
    final result = await tester.runAsync(() => runMarkerFlutterProbe());
    expect(result, isNotNull);
    expect(result!['passed'], isTrue);
    expect(result['passed_count'], greaterThan(0));
    expect(result['passed_count'], result['total_count']);
    final status = ValueNotifier('PASS: ${result['passed_count']} checks');
    await tester.pumpWidget(MarkerProbeView(status: status));
    expect(find.text('PASS: ${result['passed_count']} checks'), findsOneWidget);
    await tester.runAsync(() => File(Platform.environment['MARKER_FLUTTER_REPORT']!)
        .writeAsString('${jsonEncode(result)}\\n'));
  });
}
"""


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flutter", default="flutter")
    parser.add_argument("--native-prefix", type=Path, required=True)
    parser.add_argument("--pdfium", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--desktop", action="store_true")
    args = parser.parse_args()
    if not sys.platform.startswith("linux"):
        parser.error("this consumer checks Linux; other platform qualification is separate")
    args.native_prefix = args.native_prefix.resolve(strict=True)
    args.pdfium = args.pdfium.resolve(strict=True)
    args.out = args.out.resolve()
    if args.out.exists():
        parser.error("--out must be new so previous evidence is never overwritten")
    flutter = shutil.which(args.flutter)
    if not flutter:
        parser.error("--flutter must identify an installed Flutter SDK")
    args.out.mkdir(parents=True)
    logs = args.out / "logs"
    logs.mkdir()
    for name in ("tmp", "config", "cache", "tools"):
        (args.out / name).mkdir()
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    env.update(TMPDIR=str(args.out / "tmp"), XDG_CONFIG_HOME=str(args.out / "config"),
               XDG_CACHE_HOME=str(args.out / "cache"), FLUTTER_SUPPRESS_ANALYTICS="true",
               DART_SUPPRESS_ANALYTICS="true", CI="true", GSETTINGS_BACKEND="memory",
               CMAKE_BUILD_PARALLEL_LEVEL="2")
    # Flutter invokes Ninja directly; CMAKE_BUILD_PARALLEL_LEVEL alone does not
    # bound that invocation. Keep the wrapper private to this validation process.
    ninja = shutil.which("ninja")
    if not ninja:
        parser.error("Ninja is required by the Linux Flutter toolchain")
    wrapper = args.out / "tools/ninja"
    wrapper.write_text("#!/bin/sh\nexec " + shlex.quote(ninja) + ' -j2 "$@"\n')
    wrapper.chmod(0o755)
    env["PATH"] = str(wrapper.parent) + os.pathsep + env["PATH"]
    commands, checks = [], []
    report = {"schema": "crispembed.marker-flutter-package/1", "passed": False,
              "desktop_requested": args.desktop, "desktop_executed": False,
              "model_weights_used": False, "package_published": False,
              "native_files_inside_pub_archive": False,
              "source_commit": subprocess.check_output(
                  ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "supervisor_sha256": sha(Path(__file__)),
              "consumer_source_sha256": sha(ROOT / "tests/marker_flutter_consumer.dart"),
              "pdfium_sha256": sha(args.pdfium), "commands": commands, "checks": checks}

    def check(value, name):
        checks.append({"name": name, "passed": bool(value)})
        if not value:
            raise AssertionError(name)

    def run(label, command, cwd, timeout=240, child_env=None):
        log = logs / (label + ".log")
        start = time.monotonic()
        with log.open("w") as stream:
            try:
                completed = subprocess.run(command, cwd=cwd, env=child_env or env,
                                           stdin=subprocess.DEVNULL, stdout=stream,
                                           stderr=subprocess.STDOUT, timeout=timeout)
                code = completed.returncode
            except subprocess.TimeoutExpired:
                stream.write("\nSUPERVISOR: command timed out\n")
                code = 124
        commands.append({"name": label, "command": command, "exit_code": code,
                         "elapsed_seconds": round(time.monotonic() - start, 3),
                         "log": str(log.relative_to(args.out)), "log_sha256": sha(log)})
        if code:
            raise RuntimeError(f"{label} exited {code}; see {log.name}")
        return log.read_text()

    def flutter_run(label, arguments, cwd, timeout=240, child_env=None):
        return run(label, [flutter, "--no-version-check", *arguments], cwd, timeout, child_env)

    def verify_probe(path, name):
        result = json.loads(path.read_text())
        check(result.get("passed") is True and result.get("passed_count", 0) > 0 and
              result["passed_count"] == result["total_count"], name)
        return result

    try:
        report["flutter"] = json.loads(flutter_run("flutter-version", ["--version", "--machine"], args.out))
        package = args.out / "package"
        names = subprocess.check_output(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "flutter/crispembed"],
            cwd=ROOT).decode().split("\0")
        source_hashes = {}
        for name in names:
            if not name:
                continue
            relative = Path(name).relative_to("flutter/crispembed")
            target = package / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
            source_hashes[str(relative)] = sha(target)
        report["package_source_sha256"] = source_hashes
        analysis_options = (package / "analysis_options.yaml").read_bytes()
        flutter_run("package-resolve", ["pub", "get"], package)
        # Flutter's migration adds excludes; analyze/publish the declared source.
        (package / "analysis_options.yaml").write_bytes(analysis_options)
        flutter_run("package-analyze", ["analyze", "--no-pub"], package)
        flutter_run("package-test", ["test", "--no-pub", "--concurrency=2", "--reporter", "expanded"], package)
        archive = args.out / "crispembed.tar.gz"
        flutter_run("package-archive", ["pub", "publish", "--to-archive", str(archive)], package)
        extracted = args.out / "package-readback"
        extracted.mkdir()
        with tarfile.open(archive, "r:gz") as stream:
            for member in stream.getmembers():
                path = PurePosixPath(member.name)
                check(not path.is_absolute() and ".." not in path.parts and
                      not member.issym() and not member.islnk(), f"safe archive member {member.name}")
            stream.extractall(extracted, filter="data")
        members = {str(path.relative_to(extracted)): sha(path)
                   for path in extracted.rglob("*") if path.is_file()}
        check(all(source_hashes.get(name) == digest for name, digest in members.items()),
              "all retained Pub archive files equal staged source bytes")
        for prefix in ("", "ios/Notices/", "macos/Notices/"):
            for name in NOTICE_NAMES:
                check(members.get(prefix + name) == source_hashes[name], f"exact archive notice {prefix}{name}")
        report["archive"] = {"sha256": sha(archive), "bytes": archive.stat().st_size,
                             "members_sha256": members}
        native = sorted((args.native_prefix / "lib").glob("*.so*"))
        check(any(path.name == "libcrispembed.so" for path in native), "native prefix contains libcrispembed.so")
        native_hashes = {path.name: sha(path) for path in native}
        staging = extracted / "linux/lib"
        staging.mkdir(exist_ok=True)
        for path in native:
            shutil.copy2(path, staging / path.name)
        report["staged_native_sha256"] = native_hashes
        consumer = args.out / "consumer"
        flutter_run("consumer-create", ["create", "--template", "app", "--platforms", "linux",
                    "--project-name", "marker_flutter_consumer", "--empty", "--no-pub", str(consumer)], args.out)
        (consumer / "pubspec.yaml").write_text("""name: marker_flutter_consumer
publish_to: none
version: 1.0.0
environment:
  sdk: '>=3.13.7 <4.0.0'
dependencies:
  flutter:
    sdk: flutter
  crispembed:
    path: ../package-readback
  crypto: 3.0.7
dev_dependencies:
  flutter_test:
    sdk: flutter
  lints: 5.1.1
flutter:
  uses-material-design: true
  assets:
    - assets/owned.pdf
""")
        (consumer / "analysis_options.yaml").write_text("include: package:lints/core.yaml\n")
        shutil.copy2(ROOT / "tests/marker_flutter_consumer.dart", consumer / "lib/main.dart")
        (consumer / "test").mkdir(exist_ok=True)
        (consumer / "test/consumer_test.dart").write_text(CONSUMER_TEST)
        specification = importlib.util.spec_from_file_location("owned_marker_pdf", ROOT / "tests/test_marker_real_pdf.py")
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        (consumer / "assets").mkdir()
        (consumer / "assets/owned.pdf").write_bytes(module.owned_pdf())
        flutter_run("consumer-resolve", ["pub", "get"], consumer)
        flutter_run("consumer-analyze", ["analyze", "--no-pub"], consumer)
        flutter_run("consumer-build", ["build", "linux", "--release", "--no-pub"], consumer, 600)
        bundle = args.out / "installed-app"
        shutil.copytree(consumer / "build/linux/x64/release/bundle", bundle, symlinks=True)
        for name, digest in native_hashes.items():
            check((bundle / "lib" / name).is_file() and sha(bundle / "lib" / name) == digest,
                  f"installed app retains exact native dependency {name}")
        shutil.copy2(args.pdfium, bundle / "lib/libpdfium.so")
        runpath = run("native-runpath", ["readelf", "-d", str(bundle / "lib/libcrispembed.so")], args.out)
        values = re.findall(r"\((?:RUNPATH|RPATH)\).*?\[(.*?)\]", runpath)
        check(any("$ORIGIN" in value for value in values) and
              not any(part.startswith("/") for value in values for part in value.split(":")),
              "installed native library resolves dependencies relative to its bundle")
        dependencies = run("installed-native-dependencies", ["ldd", str(bundle / "lib/libcrispembed.so")], args.out)
        check("not found" not in dependencies, "installed native dependency closure resolves with LD_LIBRARY_PATH unset")
        runtime_env = env.copy()
        for name in ("MARKER", "MARKER_ARTIFACTS", "MARKER_INCREMENTAL", "MARKER_SOURCE_LINKS", "MARKER_REFERENCES"):
            runtime_env["CRISPEMBED_" + name] = "1"
        for name in ("MARKER_MULTILINGUAL_NORMALIZE", "MARKER_REPETITION_CHECK", "TELEOCR",
                     "ACCEPT_QWEN_RESEARCH_LICENSE", "ACCEPT_CC_BY_NC_LICENSE"):
            runtime_env["CRISPEMBED_" + name] = "0"
        runtime_env.update(CRISPEMBED_OFFLINE="1", MARKER_FLUTTER_NATIVE_LIBRARY=str(bundle / "lib/libcrispembed.so"),
                           MARKER_FLUTTER_PDFIUM_LIBRARY=str(bundle / "lib/libpdfium.so"),
                           MARKER_FLUTTER_WORK=str(args.out / "headless-work"),
                           MARKER_FLUTTER_REPORT=str(args.out / "headless-result.json"))
        flutter_run("consumer-test", ["test", "--no-pub", "--concurrency=2", "--reporter", "expanded"],
                    consumer, child_env=runtime_env)
        report["headless"] = verify_probe(args.out / "headless-result.json", "actual Flutter headless consumer checks pass")
        if args.desktop:
            check(bool(runtime_env.get("DISPLAY") or runtime_env.get("WAYLAND_DISPLAY")), "desktop display is supplied")
            runtime_env.update(MARKER_FLUTTER_WORK=str(args.out / "desktop-work"),
                               MARKER_FLUTTER_REPORT=str(args.out / "desktop-result.json"))
            run("consumer-desktop", [str(bundle / "marker_flutter_consumer")], args.out, 90, runtime_env)
            report["desktop"] = verify_probe(args.out / "desktop-result.json", "actual installed desktop consumer checks pass")
            check(report["desktop"].get("completion_frame_rendered") is True, "desktop app renders its completion frame")
            report["desktop_executed"] = True
        report["bundle_sha256"] = {str(path.relative_to(bundle)): sha(path)
                                   for path in bundle.rglob("*") if path.is_file()}
        for name, path in (("package-lock", package / "pubspec.lock"), ("consumer-lock", consumer / "pubspec.lock")):
            shutil.copy2(path, args.out / (name + ".yaml"))
        report["passed"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    (args.out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"Flutter package/consumer: {'PASS' if report['passed'] else 'FAIL'}; "
          f"{sum(row['passed'] for row in checks)}/{len(checks)} supervisor checks")
    if report.get("error"):
        print(report["error"], file=sys.stderr)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
