#!/usr/bin/env python3
"""Hosted checks of the public photo API on retained photographic inputs.

This reuses repository fixtures; it does not simulate a new physical-device
capture. Only the modern printed page has independent text ground truth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import time
import traceback
import unicodedata

try:
    import cv2
except ImportError:
    if __name__ != "__main__":
        import pytest
        pytest.skip("requires python module: opencv-python (cv2)", allow_module_level=True)
    raise
import numpy as np
from PIL import Image, __version__ as pillow_version

from test_ocr_photo_quality import distance, native


INPUTS = {
    "phone_document": {
        "file": "german_official_document.jpg",
        "sha256": "0a3937d2ac3a33bf303fd77955226af73ec74e50787708e2d327c00371997377",
        "size": [960, 1280],
    },
    "printed_page": {
        "file": "commons_test_ocr_document.jpg",
        "sha256": "92d117d352ed2ebe86efc72e661407761f2747bf44fe7a292facd8d75735e4be",
        "size": [1920, 2485],
    },
}

# Manually selected page-border pixel centres, TL/TR/BR/BL, on the unchanged
# iPhone photograph. The torn/curled sheet is not a perfectly planar rectangle;
# these are editor inputs, not a claim of reconstructing its physical shape.
PHONE_CORNERS = [(135.0, 13.0), (889.0, 31.0), (862.0, 1264.0), (122.0, 1271.0)]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def recognize(source: Path, prefix: Path, truth: str) -> dict:
    started = time.monotonic()
    command = ["tesseract", str(source), str(prefix), "--psm", "3", "-l", "eng+fra", "txt", "tsv"]
    run = subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    actual = prefix.with_suffix(".txt").read_text()
    with prefix.with_suffix(".tsv").open(newline="") as stream:
        words = [r for r in csv.DictReader(stream, delimiter="\t", quoting=csv.QUOTE_NONE)
                 if r["level"] == "5" and r["text"].strip()]
    confidence = [float(r["conf"]) for r in words if float(r["conf"]) >= 0]
    expected, observed = normalized(truth), normalized(actual)
    return {
        "text": actual, "characters": len(observed), "words": len(observed.split()),
        "cer": distance(expected, observed) / len(expected),
        "wer": distance(expected.split(), observed.split()) / len(expected.split()),
        "line_count": len({(r["block_num"], r["par_num"], r["line_num"]) for r in words}),
        "mean_word_confidence": sum(confidence) / len(confidence) if confidence else None,
        "seconds": round(time.monotonic() - started, 4), "stderr": run.stderr,
        "text_sha256": digest(prefix.with_suffix(".txt")),
        "tsv_sha256": digest(prefix.with_suffix(".tsv")),
    }


def retain_png(path: Path) -> Path:
    target = path.with_suffix(".png")
    with Image.open(path) as image:
        image.save(target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.binary = args.binary.resolve()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    report = {
        "schema": "crispembed-retained-camera-fixtures-v1",
        "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "binary_sha256": digest(args.binary),
        "physical_capture_workflow_tested": False,
        "native_recognition_models_qualified": False,
        "scope": "Public production photo geometry on existing photographs, with independent OpenCV pixels and Tesseract text checks.",
        "thresholds": {"angle_delta_max_degrees": 0.8, "perspective_mae_max_255": 1.0,
                       "perspective_p99_error_max_255": 4.0, "prepared_printed_page_cer_max": 0.08,
                       "prepared_printed_page_cer_regression_max": 0.015},
        "versions": {"python": platform.python_version(), "pillow": pillow_version,
                     "numpy": np.__version__, "opencv": cv2.__version__,
                     "tesseract": subprocess.check_output(["tesseract", "--version"], text=True).splitlines()[0]},
        "inputs": {}, "perspective": {}, "deskew": [], "failures": failures,
    }
    sources = ["src/ocr_photo.cpp", "src/ocr_crop.cpp", "src/scan_cleanup.cpp",
               "src/classical_preproc.cpp", "src/morph_fast.cpp", "src/crispembed.h",
               "tests/test_ocr_photo.cpp", "tests/test_ocr_photo_quality.py",
               "tests/test_ocr_camera_fixtures.py"]
    report["source_files"] = {name: digest(root / name) for name in sources}
    try:
        directory = root / "tests/regression/images/cc0"
        manifest = json.loads((directory / "MANIFEST.json").read_text())
        truth_file = directory / "ground_truth.json"
        truth_records = json.loads(truth_file.read_text())
        truth = next(row["text"] for row in truth_records["records"]
                     if row["file"] == INPUTS["printed_page"]["file"])
        report["ground_truth"] = {"path": str(truth_file.relative_to(root)),
                                  "sha256": digest(truth_file),
                                  "record": INPUTS["printed_page"]["file"],
                                  "normalization": "Unicode NFC and whitespace only; case, accents and punctuation retained"}
        images = {}
        for key, spec in INPUTS.items():
            source = directory / spec["file"]
            assert digest(source) == spec["sha256"], f"{key}: changed fixture bytes"
            entry = next(row for row in manifest if row["file"] == spec["file"])
            assert entry["sha256"] == spec["sha256"]
            with Image.open(source) as image:
                assert list(image.size) == spec["size"]
                exif = image.getexif()
                image_exif = {"make": exif.get(271), "model": exif.get(272), "date_time": exif.get(306)}
                if key == "phone_document":
                    assert image_exif["make"] == "Apple" and image_exif["model"] == "iPhone 7"
                images[key] = image.convert("RGB")
            report["inputs"][key] = {**spec, "source_page": entry["source_page"],
                                     "license": entry["license"], "exif": image_exif,
                                     "source_unchanged_after_checks": False}

        source = args.artifacts / "phone-original.ppm"
        images["phone_document"].save(source)
        retain_png(source)
        rectified = args.artifacts / "phone-perspective.ppm"
        geometry = native(args.binary, "photo-quad", source, rectified, False, PHONE_CORNERS)
        native_pixels = np.asarray(Image.open(rectified).convert("RGB"))
        width, height = geometry["width"], geometry["height"]
        assert native_pixels.shape == (height, width, 3)
        assert width >= 740 and height >= 1230, "Perspective unexpectedly reduced source resolution"
        destination = np.float32([(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)])
        matrix = cv2.getPerspectiveTransform(np.float32(PHONE_CORNERS), destination)
        reference = cv2.warpPerspective(np.asarray(images["phone_document"]), matrix, (width, height),
                                        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        error = np.abs(native_pixels.astype(np.int16) - reference.astype(np.int16))
        perspective = {"corners": PHONE_CORNERS, "geometry": geometry,
                       "mae_255": float(error.mean()), "p99_error_255": float(np.percentile(error, 99)),
                       "max_error_255": int(error.max()), "reference": "OpenCV getPerspectiveTransform + warpPerspective INTER_LINEAR",
                       "native_png_sha256": digest(retain_png(rectified))}
        report["perspective"] = perspective
        Image.fromarray(reference).save(args.artifacts / "phone-perspective-opencv.png")
        if perspective["mae_255"] > 1.0 or perspective["p99_error_255"] > 4.0:
            failures.append("Real phone perspective disagrees with independent OpenCV pixels")
        identity = args.artifacts / "phone-identity.ppm"
        identity_geometry = native(args.binary, "photo-quad", source, identity, False,
                                   [(0, 0), (959, 0), (959, 1279), (0, 1279)])
        report["perspective"]["identity_geometry"] = identity_geometry
        report["perspective"]["identity_png_sha256"] = digest(retain_png(identity))
        report["perspective"]["identity_byte_exact"] = Image.open(identity).tobytes() == images["phone_document"].tobytes()
        if not report["perspective"]["identity_byte_exact"]:
            failures.append("Real phone full-frame identity changed decoded source pixels")

        for key, image in images.items():
            baseline_correction = None
            for angle in [0.0, -4.2, 6.7]:
                label = f"{key}-{angle:+.1f}".replace(".", "p")
                raster = image if angle == 0 else image.rotate(angle, Image.Resampling.BICUBIC, expand=True, fillcolor="white")
                source = args.artifacts / f"{label}.ppm"
                output = args.artifacts / f"{label}-prepared.ppm"
                raster.save(source)
                retain_png(source)
                geometry = native(args.binary, "photo-deskew", source, output, False)
                prepared = Image.open(output).convert("RGB")
                retain_png(output)
                correction = geometry["correction_degrees"]
                assert math.isfinite(correction)
                if baseline_correction is None:
                    baseline_correction = correction
                delta_error = abs((correction - baseline_correction) - angle)
                item = {"input": key, "added_rotation_ccw_degrees": angle,
                        "geometry": geometry, "baseline_correction_degrees": baseline_correction,
                        "correction_delta_error_degrees": delta_error,
                        "source_size": list(raster.size), "prepared_size": list(prepared.size)}
                report["deskew"].append(item)
                if delta_error > 0.8:
                    failures.append(f"{label}: correction did not recover the independently added rotation")
                if prepared.width < raster.width or prepared.height < raster.height:
                    failures.append(f"{label}: correction downscaled the input")
                if key == "printed_page":
                    raw = recognize(source, args.artifacts / f"{label}-raw-text", truth)
                    deskew = recognize(output, args.artifacts / f"{label}-deskew-text", truth)
                    whitened = args.artifacts / f"{label}-whitened.ppm"
                    cleanup = native(args.binary, "whiten", output, whitened, False)
                    retain_png(whitened)
                    combined = recognize(whitened, args.artifacts / f"{label}-whitened-text", truth)
                    item.update(raw_ocr=raw, deskew_ocr=deskew, cleanup_geometry=cleanup, prepared_ocr=combined)
                    # Keep all arms. The declared preparation recipe includes
                    # whitening; never choose a successful arm after seeing text.
                    if combined["cer"] > 0.08 or combined["cer"] > raw["cer"] + 0.015:
                        failures.append(f"{label}: prepared text exceeds the declared CER/nonregression limits")
                print(json.dumps({"case": label, "angle": correction, "delta_error": delta_error}), flush=True)
        for key, spec in INPUTS.items():
            unchanged = digest(directory / spec["file"]) == spec["sha256"]
            report["inputs"][key]["source_unchanged_after_checks"] = unchanged
            if not unchanged:
                failures.append(f"{key}: original file changed")
    except Exception:
        failures.append(traceback.format_exc())
    finally:
        report["passed"] = not failures
        (args.artifacts / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        # PNGs retain all actual pixels; PPMs are reproducible transport files.
        for ppm in args.artifacts.glob("*.ppm"):
            if ppm.with_suffix(".png").exists():
                ppm.unlink()
    for failure in failures:
        print(f"FAIL: {failure}")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
