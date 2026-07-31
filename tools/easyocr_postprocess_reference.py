#!/usr/bin/env python3
"""Create a postprocessing reference manifest from EasyOCR ``readtext`` JSON.

The input is intentionally the output of EasyOCR's Python pipeline, not a
native/C++ dump.  This keeps the harness-blind boundary independent: detector
polygons, recognized text, and recognizer confidence come from Python, while
this script freezes the exact ordering, crop, and LayoutLM normalization
contract consumed by native callers.

Input format::

  {"image": "page.png", "width": 800, "height": 600,
   "items": [{"box": [[x,y], ...], "text": "word", "confidence": 0.9}]}

``box`` may also be ``[x0, x1, y0, y1]`` or ``{"x": ..., "y": ...,
"w": ..., "h": ...}``.  ``--mode lines`` groups compatible y bands; words
are retained as individual records by ``--mode words``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def box_xywh(value):
    if isinstance(value, dict):
        return float(value["x"]), float(value["y"]), float(value["w"]), float(value["h"])
    if len(value) == 4 and all(isinstance(x, (int, float)) for x in value):
        x0, x1, y0, y1 = map(float, value)
        return x0, y0, x1 - x0, y1 - y0
    points = [(float(p[0]), float(p[1])) for p in value]
    xs, ys = zip(*points)
    return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)


def cluster(regions):
    groups = []
    for region in sorted(regions, key=lambda r: r["y"] + r["h"] / 2):
        cy = region["y"] + region["h"] / 2
        match = None
        for group in groups:
            y0 = min(r["y"] for r in group)
            y1 = max(r["y"] + r["h"] for r in group)
            if abs(cy - (y0 + y1) / 2) <= 0.5 * max(y1 - y0, region["h"]):
                match = group
                break
        if match is None:
            groups.append([region])
        else:
            match.append(region)
    groups.sort(key=lambda group: min(r["y"] for r in group))
    return groups


def normalize(value, size):
    return max(0, min(1000, int(round(1000 * value / size))))


def make_manifest(source):
    width, height = int(source["width"]), int(source["height"])
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    mode = source["mode"]
    regions = []
    for index, item in enumerate(source["items"]):
        x, y, w, h = box_xywh(item["box"])
        regions.append({
            "text": str(item.get("text", "")),
            "confidence": float(item.get("confidence", 0.0)),
            "detector_confidence": float(item.get("detector_confidence", 0.0)),
            "x": x, "y": y, "w": w, "h": h, "source_index": index,
        })

    if mode == "words":
        ordered = [r for group in cluster(regions) for r in sorted(group, key=lambda r: r["x"])]
    elif mode == "lines":
        ordered = []
        for line, group in enumerate(cluster(regions)):
            x0 = min(r["x"] for r in group)
            y0 = min(r["y"] for r in group)
            x1 = max(r["x"] + r["w"] for r in group)
            y1 = max(r["y"] + r["h"] for r in group)
            ordered.append({
                "text": " ".join(r["text"] for r in sorted(group, key=lambda r: r["x"])),
                "confidence": sum(r["confidence"] for r in group) / len(group),
                "detector_confidence": sum(r["detector_confidence"] for r in group) / len(group),
                "x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0,
                "source_indices": [r["source_index"] for r in group], "line": line,
            })
    else:
        raise ValueError(f"unsupported mode: {mode}")

    records = []
    for index, region in enumerate(ordered):
        x0 = max(0, math.floor(region["x"] - 2))
        y0 = max(0, math.floor(region["y"] - 2))
        x1 = min(width, math.ceil(region["x"] + region["w"] + 2))
        y1 = min(height, math.ceil(region["y"] + region["h"] + 2))
        records.append({
            "index": index,
            "line": int(region.get("line", next((i for i, g in enumerate(cluster(regions)) if region in g), 0))),
            "text": region["text"],
            "confidence": region["confidence"],
            "detector_confidence": region["detector_confidence"],
            "box": [region["x"], region["y"], region["w"], region["h"]],
            "crop": [x0, y0, x1 - x0, y1 - y0],
            "normalized_box": [normalize(region["x"], width), normalize(region["y"], height),
                                normalize(region["x"] + region["w"], width),
                                normalize(region["y"] + region["h"], height)],
            "source_indices": region["source_indices"] if "source_indices" in region else [region["source_index"]],
        })
    return {
        "schema": "crispembed.easyocr.postprocess.v1",
        "source": "EasyOCR Python readtext detail=1",
        "image": source.get("image", ""),
        "width": width, "height": height, "mode": mode,
        "records": records,
        "text": "\n".join(r["text"] for r in records) if mode == "lines" else " ".join(r["text"] for r in records),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("lines", "words"), required=True)
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    source["mode"] = args.mode
    manifest = make_manifest(source)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.output}; mode={args.mode}; records={len(manifest['records'])}")


if __name__ == "__main__":
    main()
