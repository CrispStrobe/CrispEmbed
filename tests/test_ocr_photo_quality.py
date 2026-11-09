#!/usr/bin/env python3
"""Actual OCR checks on deterministic camera-like documents.

This tests production geometry with an independent Tesseract reader. It does
not qualify any native recognition model or assert physical-camera coverage.
Run with --binary build-ocr-photo/test-ocr-photo --artifacts <output directory>.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont


LINES = [
    "FIELD SERVICE INSPECTION",
    "Record 2048: Document capture validation.",
    "Customer: North River Engineering.",
    "Inspection date: 09 November 2026.",
    "Equipment: Pump station 27.",
    "Pressure: 4.8 bar. Flow: 125 litres per minute.",
    "The safety valve opened at the expected limit.",
    "All seals were clean and the housing was dry.",
    "Replace the filter after 500 operating hours.",
    "The next service requires a signed work order.",
    "Inspector: Alex Morgan. Reference: 73915.",
    "Status: Approved for scheduled operation.",
]
TRUTH = "\n".join(LINES)


def normalized(text: str) -> str:
    return " ".join(text.split())


def distance(a, b) -> int:
    previous = list(range(len(b) + 1))
    for i, item in enumerate(a, 1):
        current = [i]
        for j, other in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (item != other)))
        previous = current
    return previous[-1]


def scores(text: str) -> dict:
    gold, actual = normalized(TRUTH), normalized(text)
    return {"cer": distance(gold, actual) / len(gold),
            "wer": distance(gold.split(), actual.split()) / len(gold.split()),
            "characters": len(actual), "words": len(actual.split()), "text": text.strip()}


def ocr(path: Path) -> dict:
    started = time.monotonic()
    result = subprocess.run(["tesseract", str(path), "stdout", "--psm", "6", "-l", "eng"],
                            check=True, text=True, capture_output=True, timeout=45)
    return {**scores(result.stdout), "seconds": round(time.monotonic() - started, 4)}


def native(binary: Path, command: str, source: Path, output: Path, enabled: bool,
           corners: list[tuple[float, float]] | None = None) -> dict:
    env = {**os.environ, "CRISPEMBED_SCAN_BOUNDED_DESKEW": "1" if enabled else "0",
           "CRISPEMBED_OCR_CROP_PROJECTIVE": "1" if enabled else "0"}
    args = [str(binary), command, str(source), str(output)]
    if corners:
        args.extend(str(value) for point in corners for value in point)
    started = time.monotonic()
    result = subprocess.run(args, check=True, text=True, capture_output=True, env=env, timeout=60)
    return {**json.loads(result.stdout), "seconds": round(time.monotonic() - started, 4)}


def shadow(image: Image.Image) -> Image.Image:
    array = np.asarray(image, dtype=np.float32)
    height, width = array.shape[:2]
    x = np.linspace(0, 1, width)[None, :]
    y = np.linspace(0, 1, height)[:, None]
    illumination = .57 + .30 * x + .13 * y
    array *= illumination[:, :, None]
    # Deterministic sensor noise and JPEG compression; the pristine image and
    # exact ground truth remain available for a paired comparison.
    noise = np.random.default_rng(20261109).normal(0, 1.4, array.shape)
    result = Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8))
    encoded = io.BytesIO()
    result.save(encoded, format="JPEG", quality=82, subsampling=2)
    encoded.seek(0)
    return Image.open(encoded).convert("RGB")


def camera_projection(image: Image.Image) -> tuple[Image.Image, list[tuple[float, float]]]:
    # Pillow's transform takes camera -> page coefficients, independently
    # solved from four correspondences (the C++ path uses a unit-square map).
    width, height = image.size
    corners = [(170.0, 170.0), (1480.0, 70.0), (1240.0, 1830.0), (330.0, 1670.0)]
    paper = [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]
    matrix, rhs = [], []
    for (x, y), (u, v) in zip(corners, paper):
        matrix.extend([[x, y, 1, 0, 0, 0, -u * x, -u * y],
                       [0, 0, 0, x, y, 1, -v * x, -v * y]])
        rhs.extend([u, v])
    coefficients = np.linalg.solve(np.asarray(matrix), np.asarray(rhs))
    photo = image.transform((1650, 1950), Image.Transform.PERSPECTIVE, coefficients,
                            Image.Resampling.BICUBIC, fillcolor=(65, 72, 80))
    return shadow(photo), corners


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--font", default="DejaVuSans.ttf")
    args = parser.parse_args()
    args.binary = args.binary.resolve()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(args.font, 43)
    page = Image.new("RGB", (1400, 1800), "white")
    draw = ImageDraw.Draw(page)
    for index, line in enumerate(LINES):
        draw.text((90, 130 + index * 122), line, fill=(25, 32, 39), font=font)
    page.save(args.artifacts / "clean.ppm")
    clean = ocr(args.artifacts / "clean.ppm")
    failures = []
    if clean["cer"] > .015:
        failures.append("clean fixture reader CER exceeds 0.015")
    root = Path(__file__).resolve().parents[1]
    sources = ["src/ocr_crop.cpp", "src/ocr_crop.h", "src/scan_cleanup.cpp", "src/scan_cleanup.h",
               "src/classical_preproc.cpp", "src/morph_fast.cpp", "tests/test_ocr_photo.cpp",
               "tests/test_ocr_photo_quality.py"]
    report = {"schema": "crispembed-phone-photo-quality-v1", "synthetic_only": True,
              "source_files": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources},
              "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
              "recipe": "RGB deskew; existing grayscale background whitening for shadowed cases",
              "recognition_engine": subprocess.run(["tesseract", "--version"], check=True,
                                                   capture_output=True, text=True).stdout.splitlines()[0],
              "font": Path(font.path).name, "font_sha256": hashlib.sha256(Path(font.path).read_bytes()).hexdigest(),
              "ground_truth": TRUTH, "clean": clean, "cases": [], "failures": failures,
              "native_recognition_models_qualified": False, "physical_devices_tested": False}
    for name, angle, shaded in [("aligned", 0, False), ("clockwise", -4.2, False),
                                ("counterclockwise_shadow", 6.7, True), ("steep_shadow", -10, True)]:
        image = page.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor="white")
        if shaded:
            image = shadow(image)
        source = args.artifacts / f"{name}.ppm"
        image.save(source)
        raw = ocr(source)
        item = {"case": name, "kind": "deskew", "rotation_ccw_degrees": angle, "raw": raw, "arms": {}}
        for enabled in (False, True):
            arm = "bounded" if enabled else "legacy"
            output = args.artifacts / f"{name}-{arm}.ppm"
            geometry = native(args.binary, "deskew", source, output, enabled)
            deskew_only = ocr(output)
            # Rotating an unevenly lit photo introduces white fill around its
            # original background. The production photo recipe must flatten
            # that illumination before OCR. Retain the deskew-only diagnostic
            # so adding this required stage cannot conceal its known failure.
            if shaded:
                whitened = args.artifacts / f"{name}-{arm}-whitened.ppm"
                cleanup = native(args.binary, "whiten", output, whitened, enabled)
                result = ocr(whitened)
            else:
                cleanup, result = None, deskew_only
            item["arms"][arm] = {"geometry": geometry, "deskew_only": deskew_only,
                                 "background_whitening": cleanup, "ocr": result}
            if abs(geometry["angle"] + angle) > .55:
                failures.append(f"{name}/{arm}: angle error exceeds 0.55 degrees")
            if result["cer"] > max(.025, raw["cer"] + .01):
                failures.append(f"{name}/{arm}: prepared CER exceeds max(0.025, raw CER + 0.01)")
            if geometry["width"] < image.width or geometry["height"] < image.height:
                failures.append(f"{name}/{arm}: full-resolution geometry was downscaled")
            if angle == 0 and Image.open(output).tobytes() != image.tobytes():
                failures.append(f"{name}/{arm}: straight page was resampled")
        report["cases"].append(item)
        print(json.dumps({"case": name, "raw_cer": raw["cer"],
                          "bounded_angle": item["arms"]["bounded"]["geometry"]["angle"],
                          "bounded_cer": item["arms"]["bounded"]["ocr"]["cer"]}), flush=True)
    photo, corners = camera_projection(page)
    source = args.artifacts / "perspective-shadow.ppm"
    photo.save(source)
    item = {"case": "perspective_shadow", "kind": "projective", "corners": corners,
            "raw": ocr(source), "arms": {}}
    for enabled in (False, True):
        arm = "projective" if enabled else "legacy"
        output = args.artifacts / f"perspective-shadow-{arm}.ppm"
        geometry = native(args.binary, "quad", source, output, enabled, corners)
        item["arms"][arm] = {"geometry": geometry, "ocr": ocr(output)}
    if item["arms"]["projective"]["ocr"]["cer"] > .025:
        failures.append("projective corrected text CER exceeds 0.025")
    if item["arms"]["projective"]["ocr"]["cer"] > item["arms"]["legacy"]["ocr"]["cer"] + .005:
        failures.append("projective corrected text is worse than the legacy camera warp")
    report["cases"].append(item)
    print(json.dumps({"case": item["case"], "raw_cer": item["raw"]["cer"],
                      "projective_cer": item["arms"]["projective"]["ocr"]["cer"],
                      "legacy_cer": item["arms"]["legacy"]["ocr"]["cer"]}), flush=True)
    report["passed"] = not failures
    (args.artifacts / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    for failure in failures:
        print(f"FAIL: {failure}")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
