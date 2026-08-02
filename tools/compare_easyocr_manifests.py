#!/usr/bin/env python3
"""Compare Python and native EasyOCR postprocessing manifests."""

import argparse
import json
from pathlib import Path


def close(a, b, tolerance):
    return abs(float(a) - float(b)) <= tolerance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--native", required=True, type=Path)
    parser.add_argument("--box-tolerance", type=float, default=1.0)
    parser.add_argument("--confidence-tolerance", type=float, default=0.02)
    parser.add_argument("--ignore-detector-confidence", action="store_true",
                        help="skip detector confidence when the reference source does not expose it")
    parser.add_argument("--recognizer-crop-only", action="store_true",
                        help="compare recognizer_crop and omit the postprocess crop field")
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    native = json.loads(args.native.read_text(encoding="utf-8"))
    errors = []
    for key in ("schema", "mode"):
        if reference.get(key) != native.get(key):
            errors.append(f"{key}: {reference.get(key)!r} != {native.get(key)!r}")
    expected, actual = reference.get("records", []), native.get("records", [])
    if len(expected) != len(actual):
        errors.append(f"record count: {len(expected)} != {len(actual)}")
    for index, (want, got) in enumerate(zip(expected, actual)):
        if want.get("text") != got.get("text"):
            errors.append(f"record {index} text: {want.get('text')!r} != {got.get('text')!r}")
        if want.get("line") != got.get("line"):
            errors.append(f"record {index} line: {want.get('line')} != {got.get('line')}")
        geometry_fields = ("box", "normalized_box") if args.recognizer_crop_only else ("box", "crop", "normalized_box")
        for field in geometry_fields:
            left, right = want.get(field, []), got.get(field, [])
            if len(left) != len(right) or any(not close(a, b, args.box_tolerance) for a, b in zip(left, right)):
                errors.append(f"record {index} {field}: {left} != {right}")
        if args.recognizer_crop_only and ("recognizer_crop" not in want or "recognizer_crop" not in got):
            errors.append(f"record {index} recognizer_crop: missing from reference or native manifest")
        if "recognizer_crop" in want or "recognizer_crop" in got:
            left, right = want.get("recognizer_crop", []), got.get("recognizer_crop", [])
            if len(left) != len(right) or any(not close(a, b, args.box_tolerance) for a, b in zip(left, right)):
                errors.append(f"record {index} recognizer_crop: {left} != {right}")
        fields = ("confidence",) if args.ignore_detector_confidence else ("confidence", "detector_confidence")
        for field in fields:
            if not close(want.get(field, 0), got.get(field, 0), args.confidence_tolerance):
                errors.append(f"record {index} {field}: {want.get(field)} != {got.get(field)}")
    if errors:
        for error in errors:
            print("MISMATCH", error)
        return 1
    print(f"easyocr-manifest-compare PASS mode={native.get('mode')} records={len(actual)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
