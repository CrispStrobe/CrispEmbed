#!/usr/bin/env python3
"""Validate actual C/Rust streamed PDFs and the bounded >128 MiB fixture.

Independent reader and archival checks stay distinct from native writer error
controls. The large artifact is retained remotely; resource usage describes
this hosted fixture, not a device memory guarantee or 5000-page qualification.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import fitz
from PIL import Image
from pypdf import PdfReader

import check_ocr_pdfa
import test_ocr_render_artifacts as fixtures


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--large", type=Path, required=True)
    parser.add_argument("--verapdf", required=True)
    args = parser.parse_args()
    root = args.artifacts
    reports = root / "stream-verapdf"; reports.mkdir(exist_ok=True)
    names = ["archival-" + name + ".pdf" for name in ("rgb", "gray", "alpha", "jpeg", "text")]
    names += ["archival-streamed-" + name + ".pdf" for name in ("rgb", "gray", "alpha", "jpeg", "text")]
    names += ["archival-rust-short.pdf", "archival-rust-file.pdf"]
    paths = [root / name for name in names] + [args.large / "archival-large-stream.pdf"]
    receipt = {"source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "scope": "Qualified scalar Unicode, archival resources and synchronous bounded writer; bidi remains separate",
               "validator": subprocess.check_output([args.verapdf, "--version"], text=True).strip(),
               "positive": [], "negative": [], "missing": [], "errors": [], "reader_checks": []}
    destination = root / "stream-validation.json"

    def save():
        destination.write_text(json.dumps(receipt, ensure_ascii=False, indent=2))

    case = fixtures.ArtifactTests(); case.root = root
    for path in paths:
        if not path.exists():
            receipt["missing"].append(path.name); save(); continue
        try:
            result = check_ocr_pdfa.validate(args.verapdf, path, reports / path.stem)
            result.update(bytes=path.stat().st_size, sha256=file_hash(path))
            receipt["positive"].append(result)
        except Exception as error:
            receipt["errors"].append({"file": path.name, "phase": "archival", "error": str(error)})
        save()

    for name, source, count, first in (("archival-rust-short.pdf", root / "rgb.png", 1, "Rust short writes café 中文 😀"),
            ("archival-rust-file.pdf", root / "rgb.png", 2, "Rust file page one café 中文 😀"),
            ("archival-large-stream.pdf", args.large / "source.png", 11, "Large streamed page 1 of 11 complete")):
        path = (args.large if count == 11 else root) / name
        if not path.exists():
            continue
        result = {"file": name, "expected_pages": count, "probes": [], "images": []}
        receipt["reader_checks"].append(result)
        try:
            expected_image = Image.open(source).convert("RGB")
            expected_pixels = expected_image.tobytes()
            expected_hash = hashlib.sha256(expected_pixels).hexdigest()
            reader = PdfReader(path, strict=True)
            assert len(reader.pages) == count
            with fitz.open(path) as document:
                assert not document.is_repaired and len(document) == count
                for index in range(count):
                    expected = (f"Large streamed page {index + 1} of 11 complete" if count == 11 else
                                "Rust file page two complete" if index == 1 else first)
                    assert expected in document[index].get_text()
                    assert expected in reader.pages[index].extract_text()
                    points = (512, 512) if count == 11 else (180, 135) if index == 1 else (160, 120)
                    assert tuple(document[index].rect) == (0, 0, *points)
                for index in sorted({0, count - 1}):
                    image = reader.pages[index]["/Resources"]["/XObject"]["/Im1"].get_object()
                    actual = image.get_data()
                    assert actual == expected_pixels, "Decoded page image changed"
                    scale = expected_image.width / document[index].rect.width
                    pixels = document[index].get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
                    assert (pixels.width, pixels.height) == expected_image.size
                    assert pixels.samples == expected_pixels, "Rendered page changed"
                    result["images"].append({"page_index": index, "decoded_and_rendered_sha256": expected_hash})
            for index in sorted({0, count - 1}):
                expected = (f"Large streamed page {index + 1} of 11 complete" if count == 11 else
                            "Rust file page two complete" if index == 1 else first)
                for engine in ("PyMuPDF", "Poppler", "pypdf", "PDFium"):
                    probe = case.probe_pdf(path, engine, page_index=index)
                    result["probes"].append(probe)
                    assert probe.get("exit_code") == 0 and expected in probe.get("extracted", ""), probe
            if count == 11:
                assert 128 * 1024 * 1024 < path.stat().st_size < 256 * 1024 * 1024
                result["output_limit"] = 256 * 1024 * 1024
                result["exceeds_legacy_buffer_ceiling"] = True
            result["passed"] = True
        except Exception as error:
            result["passed"] = False
            receipt["errors"].append({"file": name, "phase": "preservation", "error": str(error)})
        save()

    control = root / "archival-streamed-rgb.pdf"
    if control.exists():
        data = control.read_bytes()
        for name, old, new in (("stream-no-intent", b"/OutputIntents", b"/InvalidIntent"),
                               ("stream-no-xmp", b"/Metadata", b"/NoXmpTag")):
            assert len(old) == len(new) and old in data
            path = reports / (name + ".pdf")
            path.write_bytes(data.replace(old, new, 1))
            try:
                receipt["negative"].append(check_ocr_pdfa.validate(args.verapdf, path, reports / name))
            except Exception as error:
                receipt["errors"].append({"file": name, "phase": "negative", "error": str(error)})
            save()
    failures = [item for item in receipt["positive"] if not item["compliant"] or item["exit_code"] != 0]
    false_passes = [item for item in receipt["negative"] if item["compliant"] or not item["failed_rules"]]
    save()
    assert not (failures or false_passes or receipt["missing"] or receipt["errors"]), receipt
    assert len(receipt["positive"]) == 13 and len(receipt["negative"]) == 2
    assert len(receipt["reader_checks"]) == 3 and all(item["passed"] for item in receipt["reader_checks"])
    print("13 PDF/A-2b positives, 2 rejected controls, complete Rust/image/page/large-file preservation passed")


if __name__ == "__main__":
    main()
