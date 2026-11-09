#!/usr/bin/env python3
"""Independently inspect tiny files produced by the real Windows C++ paths."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
from pypdf import PdfReader


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(artifacts, binaries):
    expected = bytes(
        component
        for y in range(3)
        for x in range(4)
        for component in (x * 60 + y * 10, y * 80 + x * 5, 240 - x * 40 - y * 15)
    )
    outcomes = []
    for name in ("production", "harness", "legacy"):
        root = artifacts / name
        native = json.loads((root / "native.json").read_text(encoding="utf-8"))
        assert native["passed"] is True, name
        assert native["actual_production_translation_units"] is (name != "harness"), name
        assert native["historical_negative_control"] is (name == "legacy"), name
        nested = root / "Prüfung-文件-📷"
        source = root / "source.png"
        copied = nested / "Seite-é-東京-😀.png"
        written = nested / "Kopie-ü-文-😀.png"
        assert source.read_bytes() == copied.read_bytes(), "original PNG bytes changed"
        images = [source, copied]
        if name == "legacy":
            assert native["windows_ansi_codepage"] != 65001, "negative control requires ANSI host"
            assert not written.exists(), "legacy path unexpectedly wrote a Unicode destination"
            assert not list(root.rglob("*.pdf")), "legacy control must not fake PDF success"
        else:
            images.append(written)
        for path in images:
            with Image.open(path) as image:
                assert image.size == (4, 3) and image.convert("RGB").tobytes() == expected, path.name
        pdf_receipts = []
        if name != "legacy":
            pdfs = [root / "ascii.pdf", nested / "Geprüft-文-😀.pdf"]
            assert len(list(root.rglob("*.pdf"))) == 2, name
            for path in pdfs:
                reader = PdfReader(path, strict=True)
                assert len(reader.pages) == 1, path.name
                page = reader.pages[0]
                assert [float(v) for v in page.mediabox] == [0, 0, 288, 216], path.name
                text = page.extract_text().strip()
                assert text == "Reviewed Unicode path", repr(text)
                resources = page["/Resources"].get_object()
                xobjects = resources["/XObject"].get_object()
                raster = [v.get_object() for v in xobjects.values() if v.get_object().get("/Subtype") == "/Image"]
                assert len(raster) == 1, path.name
                image = raster[0]
                assert (image["/Width"], image["/Height"], image["/BitsPerComponent"]) == (4, 3, 8)
                assert image["/ColorSpace"] == "/DeviceRGB"
                assert image.get_data() == expected, "PDF page image pixels differ"
                pdf_receipts.append({"path": path.relative_to(artifacts).as_posix(), "sha256": digest(path), "text": text})
        outcomes.append({"path": name, "native": native, "pdfs": pdf_receipts})
    sources = [
        "src/image_preprocess.cpp", "src/image_preprocess.h", "src/scan_cleanup.h",
        "src/core/image_out.cpp", "src/core/image_out.h", "src/core/provenance.h", "src/core/clean_exit.h",
        "src/ocr_render.cpp", "src/ocr_render.h", "tests/ocr_render_stb.cpp",
        "tests/test_ocr_windows_paths.cpp", "tests/windows_utf8_paths/CMakeLists.txt",
        "tests/check_ocr_windows_paths.py", ".github/workflows/ocr-windows-paths.yml",
        "ggml/examples/stb_image.h", "ggml/examples/stb_image_write.h",
    ]
    return {
        "schema": 1,
        "passed": True,
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "ggml_commit": subprocess.check_output(["git", "rev-parse", "HEAD:ggml"], text=True).strip(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "dependencies": {name: importlib.metadata.version(name) for name in ("Pillow", "pypdf")},
        "sources_sha256": {name: digest(Path(name)) for name in sources},
        "binaries_sha256": {name: digest(binaries / f"ocr-paths-{name}.exe") for name in ("production", "harness", "legacy")},
        "results": outcomes,
        "scope": "Actual Windows file I/O, original pixels, numeric image preprocessing and model-free PDF embedding; no OCR model or full desktop application qualification.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--binaries", required=True, type=Path)
    args = parser.parse_args()
    result = check(args.artifacts, args.binaries)
    (args.artifacts / "qualification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Windows paths: production and standalone Unicode controls, historical failure, four PDFs and exact pixels passed")
