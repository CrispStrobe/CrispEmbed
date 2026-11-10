#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Public Marker controls for the opt-in glyphless OCR font correction.

The explicitly selected owned PDFium ABI fixture is not a PDF parser. These
controls isolate font classification, renderer compatibility and checkpoint
identity without models. Real PDF quality is measured by marker_glyphless_ab.py.
Every invocation requires a new output directory and retains failed attempts.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
GATE = "CRISPEMBED_MARKER_GLYPHLESS_TEXT"
FORMATS = ("markdown", "html", "json", "chunks")
CASES = (
    ("exact", "GlyphLessFont", True, True),
    ("subset", "ABCDEF+GlyphLessFont", True, True),
    ("lowercase_family", "glyphlessfont", True, False),
    ("lowercase_subset", "abcdef+GlyphLessFont", True, False),
    ("short_subset", "ABCDE+GlyphLessFont", True, False),
    ("long_subset", "ABCDEFG+GlyphLessFont", True, False),
    ("digit_subset", "ABCDE1+GlyphLessFont", True, False),
    ("suffix", "GlyphLessFontExtra", True, False),
    ("prefix", "ExtraGlyphLessFont", True, False),
    ("courier", "Courier", True, False),
    ("mono_name", "FixtureMono", False, False),
    ("sans", "FixtureSans", False, False),
    ("already_proportional", "GlyphLessFont", False, False),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def walk(node):
    yield node
    for child in node.get("children") or []:
        yield from walk(child)


def physical_lines(document):
    result = Counter()
    for page in document["children"]:
        for block in walk(page):
            for line in block.get("lines") or []:
                value = dict(line)
                value["spans"] = [{k: v for k, v in span.items() if k != "monospace"}
                                  for span in line.get("spans", [])]
                record = {"page": page["id"], "page_bbox": page["bbox"], "line": value}
                result[json.dumps(record, sort_keys=True, ensure_ascii=False)] += 1
    return result


def equivalent(left, right, fmt):
    if fmt in ("json", "chunks"):
        left, right = json.loads(left), json.loads(right)
        for value in (left, right):
            value["metadata"]["configuration"].pop("elapsed_seconds", None)
    return left == right


def probe(args):
    args.out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(ROOT / "python"))
    from crispembed import CrispMarker

    checks, outputs = [], {}

    def check(name, condition):
        checks.append({"name": name, "passed": bool(condition)})

    def rejects(name, action, fragment):
        try:
            action()
        except RuntimeError as exc:
            check(name, fragment in str(exc))
        else:
            check(name, False)

    def gate(value):
        if value is None:
            os.environ.pop(GATE, None)
        else:
            os.environ[GATE] = value

    def marker(**options):
        return CrispMarker(mode="text", n_threads=1, extract_images=False,
                           lib_path=str(args.lib), **dict(pdfium_path=args.fixture, **options))

    def render(context, tag):
        values = {fmt: context.render(fmt) for fmt in FORMATS}
        for fmt, value in values.items():
            target = args.out / (tag + "." + fmt)
            target.write_text(value, encoding="utf-8")
            outputs[target.name] = sha(target)
        return values

    error = None
    try:
        for name in ("SOURCE_LINKS", "REFERENCES", "MULTILINGUAL_NORMALIZE", "REPETITION_CHECK"):
            os.environ["CRISPEMBED_MARKER_" + name] = "0"
        os.environ.update(CRISPEMBED_MARKER="1", CRISPEMBED_MARKER_ARTIFACTS="1", CRISPEMBED_MARKER_INCREMENTAL="1")
        for tag, font, fixed, corrected in CASES:
            source = args.out / (tag + ".fixture.pdf")
            source.write_text("%PDF-1.7\nMARKER_LIFECYCLE_FIXTURE\nFONT_NAME=" + font +
                              ("\nFONT_FIXED=1\n" if fixed else "\n"), encoding="ascii")
            gate("0")
            with marker() as context:
                context.convert(source)
                before = render(context, tag + "-off")
            gate("1")
            with marker() as context:
                context.convert(source)
                after = render(context, tag + "-on")
            first, second = json.loads(before["json"]), json.loads(after["json"])
            check(tag + ": physical page, line, span text/geometry and other styles are exact",
                  physical_lines(first) == physical_lines(second) and len(first["children"]) == len(second["children"]) == 3)
            blocks_before = [n for n in walk(first) if n.get("lines")]
            blocks_after = [n for n in walk(second) if n.get("lines")]
            check(tag + ": all three fixture lines remain present", len(blocks_before) == len(blocks_after) == 3)
            if corrected:
                check(tag + ": artificial Code classification is corrected to Text",
                      all(n["block_type"] == "Code" for n in blocks_before) and
                      all(n["block_type"] == "Text" for n in blocks_after))
                spans = [s for n in blocks_after for line in n["lines"] for s in line["spans"]]
                check(tag + ": artificial monospace is cleared", bool(spans) and not any(s["monospace"] for s in spans))
                for i, (a, b) in enumerate(zip(first["children"], second["children"])):
                    text = "".join(s["text"] for n in walk(a) for line in (n.get("lines") or []) for s in line["spans"])
                    warning = ("Glyphless OCR font style is unknown; artificial monospace styling ignored for " +
                               str(sum(c != " " for c in text)) + " characters")
                    check(f"{tag}: page {i} records only the declared unknown-style diagnostic",
                          Counter(b.get("warnings", [])) == Counter(a.get("warnings", [])) + Counter([warning]))
            else:
                for fmt in FORMATS:
                    check(tag + ": unrelated font preserves exact " + fmt,
                          equivalent(before[fmt], after[fmt], fmt))
            if tag == "exact":
                for value in (None, "", "0", "FaLsE", "OfF", "nO"):
                    gate(value)
                    with marker() as context:
                        context.convert(source)
                        control = render(context, "false-" + ("unset" if value is None else value or "empty"))
                    for fmt in FORMATS:
                        check(f"{tag}: {value!r} retains legacy {fmt}", equivalent(control[fmt], before[fmt], fmt))

        source, raw, view = args.out / "exact.fixture.pdf", args.out / "identity.cemk", args.out / "processed.cemk"
        gate("0")
        with marker() as context:
            context.start(source, raw)
            check("checkpoint commits one page", context.step(1)["completed_pages"] == 1)
            committed = raw.read_bytes()
            gate("1")
            rejects("active job rejects changed font gate", lambda: context.step(1), "identity changed")
            check("active gate rejection preserves checkpoint", raw.read_bytes() == committed)
        with marker() as context:
            rejects("resume rejects changed font gate", lambda: context.start(source, raw, resume=True), "does not match")
            check("rejected resume preserves checkpoint", raw.read_bytes() == committed)
        gate("FaLsE")
        with marker() as context:
            check("false alias resumes the original processing identity",
                  context.start(source, raw, resume=True)["completed_pages"] == 1)
            check("restored gate completes all three pages", context.step(64)["completed_pages"] == 3)
            completed = render(context, "completed")
            context.save_extraction(view)
        gate("1")
        with marker() as context:
            context.open_extraction(view)
            reopened = render(context, "reopened")
            for fmt in FORMATS:
                check("processed artifact preserves original " + fmt + " under changed gate", completed[fmt] == reopened[fmt])
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        for path, name in ((args.out, "<OUTPUT>"), (args.lib, "<LIBRARY>"), (args.fixture, "<FIXTURE>"), (ROOT, "<REPOSITORY>")):
            error = error.replace(str(path), name)
    receipt = {"kind": "marker-glyphless-font-controls", "probe_sha256": sha(__file__),
               "probe_source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "native_source_commit": args.native_source_commit, "native_sha256": sha(args.lib),
               "fixture_sha256": sha(args.fixture), "fixture_source_sha256": sha(ROOT / "tests/marker_pdfium_fixture.cpp"),
               "wrapper_sha256": sha(ROOT / "python/crispembed/_marker.py"),
               "scope": "Owned PDFium ABI fixture, not real PDF/OCR quality or model qualification",
               "only_renderer_comparison_exclusion": "metadata.configuration.elapsed_seconds",
               "checks": checks, "outputs": outputs, "passed_count": sum(row["passed"] for row in checks),
               "total_count": len(checks), "passed": bool(checks) and all(row["passed"] for row in checks) and error is None}
    if error:
        receipt["error"] = error
    write_json(args.out / "receipt.json", receipt)
    return 0 if receipt["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("lib", "fixture", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--native-source-commit", required=True)
    parser.add_argument("--probe", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.lib, args.fixture = (p.resolve(strict=True) for p in (args.lib, args.fixture))
    args.out = args.out.resolve()
    if args.out.exists():
        parser.error("--out must be new; preserve every attempt")
    if args.probe:
        return probe(args)
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--probe", "--lib", str(args.lib),
               "--fixture", str(args.fixture), "--out", str(args.out), "--native-source-commit", args.native_source_commit]
    timed_out = False
    try:
        process = subprocess.run(command, capture_output=True, text=True, timeout=120)
        returncode, stdout, stderr = process.returncode, process.stdout, process.stderr
    except subprocess.TimeoutExpired as exc:
        returncode, timed_out = 124, True
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout or ""
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr or ""
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "probe.stdout").write_text(stdout, encoding="utf-8")
    (args.out / "probe.stderr").write_text(stderr, encoding="utf-8")
    try:
        receipt = json.loads((args.out / "receipt.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        receipt = {"kind": "marker-glyphless-font-controls", "passed": False, "error": "worker produced no valid receipt"}
    receipt["supervisor"] = {"returncode": returncode, "timed_out": timed_out,
                             "stdout_sha256": sha(args.out / "probe.stdout"), "stderr_sha256": sha(args.out / "probe.stderr")}
    receipt["passed"] = receipt.get("passed") is True and returncode == 0 and not timed_out
    write_json(args.out / "receipt.json", receipt)
    print(f"Marker glyphless controls: {'PASS' if receipt['passed'] else 'FAIL'} "
          f"({receipt.get('passed_count', 0)}/{receipt.get('total_count', 0)} checks)")
    for row in receipt.get("checks", []):
        if not row["passed"]:
            print("FAIL: " + row["name"])
    if receipt.get("error"):
        print(receipt["error"], file=sys.stderr)
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
