#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Read a prior owned processed artifact and reject its raw resume in a new build.

This probe uses only retained owned fixtures and explicit libraries; no models.
Its caller must supervise native exit/timeout and retain stdout/stderr.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "lib", "pdfium", "prior", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.repo, args.lib, args.pdfium, args.prior = (
        p.resolve(strict=True) for p in (args.repo, args.lib, args.pdfium, args.prior))
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.repo / "python"))
    from crispembed import CrispMarker

    previous = json.loads((args.prior / "receipt.json").read_text())
    source, raw, view = (args.prior / name for name in ("owned-three-page.pdf", "resume.cemk", "processed.cemk"))
    before = {p.name: sha(p) for p in (source, raw, view)}
    rows, rendered_hashes = [], {}

    def check(name, condition):
        rows.append({"name": name, "passed": bool(condition)})
        if not condition:
            raise AssertionError(name)

    os.environ.update(CRISPEMBED_MARKER="1", CRISPEMBED_OFFLINE="1", CRISPEMBED_TELEOCR="0",
                      CRISPEMBED_ACCEPT_QWEN_RESEARCH_LICENSE="0", CRISPEMBED_ACCEPT_CC_BY_NC_LICENSE="0")
    for suffix in ("INCREMENTAL", "SOURCE_LINKS", "REFERENCES", "MULTILINGUAL_NORMALIZE", "REPETITION_CHECK"):
        os.environ["CRISPEMBED_MARKER_" + suffix] = "0"
    os.environ["CRISPEMBED_MARKER_ARTIFACTS"] = "1"
    error, diagnostic = None, None
    try:
        check("native build differs from retained artifact producer", sha(args.lib) != previous["library_sha256"])
        check("source bytes and PDFium binary match prior producer",
              sha(source) == previous["fixture_sha256"] and sha(args.pdfium) == previous["pdfium_sha256"])
        with CrispMarker(mode="text", lib_path=str(args.lib), pdfium_path=args.out / "nonexistent-pdfium.so",
                         n_threads=1, extract_images=False, dpi=150) as reader:
            reader.open_extraction(view)
            for fmt in ("markdown", "html", "json", "chunks"):
                expected_file = args.prior / f"incremental.{fmt}"
                check(f"prior {fmt} reference matches recorded producer hash",
                      sha(expected_file) == previous["output_sha256"][expected_file.name])
                actual = reader.render(fmt)
                check(f"new build reopens prior processed {fmt} byte-exact", actual == expected_file.read_text())
                target = args.out / f"reopened.{fmt}"
                target.write_text(actual, encoding="utf-8")
                rendered_hashes[target.name] = sha(target)
            old = json.loads((args.prior / "incremental.json").read_text())["metadata"]
            check("old provenance remains unchanged without viewer metadata stamping",
                  reader.metadata == old and all(key not in old["configuration"] for key in
                  ("layout_backend", "marker_source_revision", "surya_model_parity", "text_quality_check")))
        for suffix in ("INCREMENTAL", "SOURCE_LINKS", "REFERENCES"):
            os.environ["CRISPEMBED_MARKER_" + suffix] = "1"
        with CrispMarker(mode="text", lib_path=str(args.lib), pdfium_path=args.pdfium,
                         n_threads=1, extract_images=False) as resumer:
            try:
                resumer.start(source, raw, resume=True)
            except RuntimeError as exc:
                diagnostic = str(exc)
                check("new build rejects prior raw checkpoint identity",
                      "checkpoint input, model, processing identity or selection does not match" in diagnostic)
            else:
                check("new build rejects prior raw checkpoint identity", False)
            check("rejected raw resume installs no active job", resumer.progress["state"] == "idle")
        check("all old source and artifact bytes remain unchanged",
              {p.name: sha(p) for p in (source, raw, view)} == before)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        for path, replacement in ((args.out, "<OUTPUT>"), (args.prior, "<PRIOR>"), (args.repo, "<CHECKOUT>"),
                                  (args.lib, "<NATIVE_LIBRARY>"), (args.pdfium, "<PDFIUM_LIBRARY>")):
            error = error.replace(str(path), replacement)
    report = {
        "kind": "marker-cross-build-artifact-readback", "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=args.repo, text=True).strip(),
        "prior_library_sha256": previous["library_sha256"], "library_sha256": sha(args.lib),
        "pdfium_sha256": sha(args.pdfium), "probe_sha256": sha(__file__), "prior_input_sha256": before,
        "rendered_sha256": rendered_hashes, "model_weights_used": False,
        "general_pdf_or_ocr_quality_qualified": False, "raw_rejection_diagnostic": diagnostic,
        "checks": rows, "passed_count": sum(row["passed"] for row in rows), "total_count": len(rows),
        "passed": error is None and bool(rows) and all(row["passed"] for row in rows),
    }
    if error:
        report["error"] = error
    (args.out / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{report['passed_count']}/{report['total_count']} cross-build artifact checks passed")
    if error:
        print(error, file=sys.stderr)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
