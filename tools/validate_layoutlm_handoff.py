#!/usr/bin/env python3
"""Validate and serialize CrispEmbed words for LayoutLMv2/v3 processors.

LayoutLM processors with ``apply_ocr=False`` expect externally ordered words
and normalized boxes. Confidence and pixel geometry are retained as sidecar
metadata because they are useful to downstream consumers but are not
processor arguments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate(manifest: dict) -> dict:
    if manifest.get("schema") != "crispembed.easyocr.postprocess.v1":
        raise ValueError("unsupported or missing EasyOCR manifest schema")
    width, height = int(manifest["width"]), int(manifest["height"])
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    records = manifest.get("records", [])
    words, boxes, sidecar = [], [], []
    for expected, record in enumerate(records):
        if int(record.get("index", -1)) != expected:
            raise ValueError("records must be contiguous and in reading order")
        text = str(record.get("text", ""))
        if not text:
            raise ValueError(f"record {expected} has empty text")
        pixel = record.get("box")
        normalized = record.get("normalized_box")
        if not isinstance(pixel, list) or len(pixel) != 4 or not isinstance(normalized, list) or len(normalized) != 4:
            raise ValueError(f"record {expected} is missing box metadata")
        x, y, w, h = (float(value) for value in pixel)
        if w < 0 or h < 0 or x < 0 or y < 0 or x + w > width or y + h > height:
            raise ValueError(f"record {expected} pixel box is out of bounds")
        normalized_int = [int(value) for value in normalized]
        if any(value < 0 or value > 1000 for value in normalized_int) or normalized_int[2] < normalized_int[0] or normalized_int[3] < normalized_int[1]:
            raise ValueError(f"record {expected} normalized box is invalid")
        confidence = float(record.get("confidence", 0.0))
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"record {expected} confidence is outside [0,1]")
        words.append(text)
        boxes.append(normalized_int)
        sidecar.append({"text": text, "confidence": confidence, "box": [x, y, w, h], "index": expected})
    return {
        "schema": "crispembed.layoutlm.handoff.v1",
        "image": manifest.get("image", ""),
        "width": width,
        "height": height,
        "apply_ocr": False,
        "processor_args": {"words": words, "boxes": boxes},
        "sidecar": sidecar,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(json.loads(args.manifest.read_text(encoding="utf-8")))
    serialized = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
