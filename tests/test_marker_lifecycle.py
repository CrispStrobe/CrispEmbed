#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Actual CrispMarker Python/native lifecycle integration with owned fixtures.

    python tests/test_marker_lifecycle.py --lib build/libcrispembed.so \
        --pdfium build/libmarker-pdfium-fixture.so \
        --report build/marker-python-lifecycle.json

The PDFium-shaped library is the owned test fixture, not a PDF renderer. No
model weights, external documents, network access or mocked bindings are used.
The supervising process reports native crashes/timeouts as failures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading


ROOT = Path(__file__).resolve().parents[1]


def probe(args) -> int:
    sys.path.insert(0, str(ROOT / "python"))
    from crispembed import CrispMarker

    rows = []

    def check(condition, name):
        rows.append({"name": name, "passed": bool(condition)})
        if not condition:
            raise AssertionError(name)

    def rejects(fn, kind, fragment, name):
        try:
            fn()
        except kind as error:
            check(fragment in str(error), name)
        else:
            check(False, name)

    for name in ("CRISPEMBED_MARKER", "CRISPEMBED_MARKER_ARTIFACTS", "CRISPEMBED_MARKER_INCREMENTAL",
                 "CRISPEMBED_MARKER_REFERENCES"):
        os.environ[name] = "1"
    for name in ("CRISPEMBED_MARKER_SOURCE_LINKS", "CRISPEMBED_MARKER_MULTILINGUAL_NORMALIZE",
                 "CRISPEMBED_MARKER_REPETITION_CHECK", "CRISPEMBED_TELEOCR",
                 "CRISPEMBED_ACCEPT_QWEN_RESEARCH_LICENSE"):
        os.environ[name] = "0"

    def marker(**overrides):
        options = dict(mode="text", pdfium_path=args.pdfium, lib_path=str(args.lib), extract_images=False)
        options.update(overrides)
        return CrispMarker(**options)

    error = None
    try:
        with tempfile.TemporaryDirectory(prefix="marker-python-lifecycle-") as directory:
            root = Path(directory)
            source, failing = root / "owned.pdf", root / "failing.pdf"
            source.write_bytes(b"%PDF-1.7\nMARKER_LIFECYCLE_FIXTURE\nowned source\n")
            failing.write_bytes(b"%PDF-1.7\nMARKER_LIFECYCLE_FIXTURE\nFAIL_SECOND_PAGE\n")
            raw, view = root / "raw.cemk", root / "view.cemk"
            with marker() as first:
                check(first.progress["state"] == "idle", "actual wrapper reports idle")
                os.environ["CRISPEMBED_MARKER_INCREMENTAL"] = "OfF"
                rejects(lambda: first.start(source, raw), RuntimeError, "CRISPEMBED_MARKER_INCREMENTAL",
                        "native gate failure propagates through Python")
                check(not raw.exists(), "gate failure creates no checkpoint")
                os.environ["CRISPEMBED_MARKER_INCREMENTAL"] = "1"
                rejects(lambda: first.start(source, raw, resume=1), ValueError, "bool", "resume requires a Python bool")
                state = first.start(source, raw)
                check(state["state"] == "running" and state["completed_pages"] == 0 and state["total_pages"] == 3,
                      "start returns real native progress")
                initial = raw.read_bytes()
                for invalid in (True, 0, 65, 1.5):
                    rejects(lambda value=invalid: first.step(value), ValueError, "page_budget",
                            f"Python rejects invalid budget {invalid!r}")
                check(first.progress["state"] == "running" and raw.read_bytes() == initial,
                      "Python validation preserves the active job")
                state = first.step(1)
                check(state["completed_pages"] == 1 and state["next_page"] == 1 and state["state"] == "running",
                      "bounded Python step commits one page")
                partial, committed = first.render(), raw.read_bytes()
                check("First owned page" in partial and "Second owned page" not in partial and committed != initial,
                      "Python exposes only durably completed text")
                cancellation = threading.Thread(target=first.cancel)
                cancellation.start()
                cancellation.join(timeout=5)
                check(not cancellation.is_alive() and first.progress["cancel_requested"],
                      "another Python thread requests native cancellation")
                check(first.step(64)["state"] == "cancelled" and raw.read_bytes() == committed,
                      "cancelled Python job keeps its checkpoint")
            rejects(lambda: first.step(), RuntimeError, "closed", "closed wrapper cannot access freed native context")
            first.close()

            with marker() as resumed:
                state = resumed.start(source, raw, resume=True)
                check(state["completed_pages"] == 1 and not state["cancel_requested"],
                      "fresh Python context resumes committed cursor")
                check(resumed.step(1)["completed_pages"] == 2, "resumed step preserves page budget")
                state = resumed.step(64)
                check(state["state"] == "complete" and state["completed_pages"] == 3,
                      "Python resume completes remaining pages")
                complete = resumed.render()
                pages = json.loads(resumed.render("json"))["children"]
                check([page["printed_page_label"] for page in pages] == ["iv", "章-2", "📖-3"],
                      "actual Python JSON retains BMP and astral UTF-16 labels")
                check(resumed.metadata["page_count"] == 3 and resumed.assets == {},
                      "Python metadata and assets match the text-mode result")
                complete_raw = raw.read_bytes()
                rejects(lambda: resumed.save_extraction(raw), RuntimeError, "separately",
                        "wrapper preserves the resumable checkpoint when saving views")
                check(raw.read_bytes() == complete_raw, "rejected view save is nondestructive")
                resumed.save_extraction(view)
                resumed.save(root / "document.md")
                check((root / "document.md").read_text(encoding="utf-8") == complete and
                      json.loads((root / "document_meta.json").read_text(encoding="utf-8"))["page_count"] == 3,
                      "existing Python bundle save accepts a completed lifecycle view")

            with marker(pdfium_path=None, dpi=150) as reader:
                check(reader.open_extraction(view) == complete, "processed artifact reopens without PDFium or model paths")
                rejects(lambda: reader.open_extraction(raw), RuntimeError, "matching processing settings",
                        "raw artifact rejects mismatched standalone configuration")
                check(reader.render() == complete, "failed raw open preserves prior Python view")
                corrupted = bytearray(view.read_bytes())
                corrupted[24] = ord("1") if corrupted[24] == ord("0") else ord("0")
                broken = root / "corrupted.cemk"
                broken.write_bytes(corrupted)
                rejects(lambda: reader.open_extraction(broken), RuntimeError, "integrity mismatch",
                        "artifact integrity failure reaches Python")
                check(reader.render() == complete, "corrupt open cannot replace a valid Python view")

            failed_checkpoint = root / "failed.cemk"
            with marker() as failed:
                failed.start(failing, failed_checkpoint)
                rejects(lambda: failed.step(64), RuntimeError, "page 1", "later page failure reaches Python")
                check(failed.progress["state"] == "error" and failed.progress["completed_pages"] == 1 and
                      "First owned page" in failed.render() and "Second owned page" not in failed.render(),
                      "Python can inspect completed units after failure")
                retained = failed_checkpoint.read_bytes()
            with marker() as retry:
                check(retry.start(failing, failed_checkpoint, resume=True)["completed_pages"] == 1,
                      "failed checkpoint remains resumable from Python")
                rejects(lambda: retry.step(1), RuntimeError, "page 1", "retry targets the uncompleted page")
                check(failed_checkpoint.read_bytes() == retained, "repeated page failure preserves completed units")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    report = {
        "kind": "marker-python-native-lifecycle",
        "model_weights_used": False,
        "pdf_rendering_qualified": False,
        "library_sha256": hashlib.sha256(args.lib.read_bytes()).hexdigest(),
        "fixture_library_sha256": hashlib.sha256(args.pdfium.read_bytes()).hexdigest(),
        "wrapper_sha256": hashlib.sha256((ROOT / "python/crispembed/_marker.py").read_bytes()).hexdigest(),
        "checks": rows,
        "passed": error is None and bool(rows) and all(row["passed"] for row in rows),
        "passed_count": sum(row["passed"] for row in rows),
        "total_count": len(rows),
    }
    if error:
        report["error"] = error
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--pdfium", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--probe", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.lib = args.lib.resolve(strict=True)
    args.pdfium = args.pdfium.resolve(strict=True)
    args.report = args.report.resolve()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if args.probe:
        return probe(args)
    command = [sys.executable, str(Path(__file__).resolve()), "--probe", "--lib", str(args.lib),
               "--pdfium", str(args.pdfium), "--report", str(args.report)]
    # Do not accept an old positive report if the current native probe crashes.
    args.report.unlink(missing_ok=True)
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        report = json.loads(args.report.read_text(encoding="utf-8")) if args.report.exists() else {
            "kind": "marker-python-native-lifecycle", "passed": False, "error": "probe produced no report"}
        report.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        report["passed"] = report.get("passed") is True and result.returncode == 0
    except subprocess.TimeoutExpired:
        report = {"kind": "marker-python-native-lifecycle", "passed": False, "error": "native probe timed out"}
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Marker Python lifecycle: {'PASS' if report['passed'] else 'FAIL'}")
    if report.get("error"):
        print(report["error"], file=sys.stderr)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
