#!/usr/bin/env python3
"""Compare native EasyOCR word manifests with Tesseract TSV geometry/order.

This is deliberately a geometry/order comparator. Tesseract text is reported,
but is not treated as a correctness oracle for the EasyOCR recognizer lane.
"""

import argparse
import csv
import json
from pathlib import Path


def load_words(path):
    rows = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row.get("level") != "5" or not row.get("text", "").strip():
                continue
            rows.append(
                {
                    "text": row["text"],
                    "line": (int(row.get("block_num", 0)), int(row.get("par_num", 0)), int(row.get("line_num", 0))),
                    "box": [float(row["left"]), float(row["top"]), float(row["width"]), float(row["height"])],
                }
            )
    return rows


def native_words(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [{"text": r.get("text", ""), "line": r.get("line", 0), "box": r.get("box", [])} for r in payload.get("records", [])]


def close(a, b, tolerance):
    return abs(float(a) - float(b)) <= tolerance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", required=True, type=Path)
    parser.add_argument("--tesseract-tsv", required=True, type=Path)
    parser.add_argument("--box-tolerance", type=float, default=3.0)
    parser.add_argument("--geometry-only", action="store_true")
    args = parser.parse_args()

    native = native_words(args.native)
    tesseract = load_words(args.tesseract_tsv)
    errors = []
    if len(native) != len(tesseract):
        errors.append(f"record count: native={len(native)} tesseract={len(tesseract)}")
    for index, (mine, ref) in enumerate(zip(native, tesseract)):
        if not args.geometry_only and mine["text"].strip() != ref["text"].strip():
            errors.append(f"record {index} text: {mine['text']!r} != {ref['text']!r}")
        if len(mine["box"]) != 4 or any(not close(a, b, args.box_tolerance) for a, b in zip(mine["box"], ref["box"])):
            errors.append(f"record {index} box: {mine['box']} != {ref['box']}")
        if index and mine["line"] < native[index - 1]["line"]:
            errors.append(f"native line order decreases at record {index}")
        if index and ref["line"] < tesseract[index - 1]["line"]:
            errors.append(f"Tesseract line order decreases at record {index}")
    if errors:
        for error in errors[:20]:
            print("MISMATCH", error)
        if len(errors) > 20:
            print(f"MISMATCH ... {len(errors) - 20} more")
        return 1
    print(f"easyocr-tsv-compare PASS records={len(native)} geometry_only={args.geometry_only}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
