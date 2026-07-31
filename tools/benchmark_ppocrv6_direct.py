#!/usr/bin/env python3
"""Run the PP-OCRv6 direct detector/quad/orientation/recognizer harness on CC0 fixtures."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", required=True, type=Path)
    ap.add_argument("--det", required=True, type=Path)
    ap.add_argument("--rec", required=True, type=Path)
    ap.add_argument("--orientation", required=True, type=Path)
    ap.add_argument("--sources", type=Path, default=Path("tests/regression/cc0_sources.json"))
    ap.add_argument("--image-root", type=Path, default=Path("tests/regression/images/cc0"))
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--max-regions", type=int, default=0)
    args = ap.parse_args()
    rows = []
    for item in json.loads(args.sources.read_text())[:args.limit]:
        image = args.image_root / item["file"]
        if not image.exists():
            rows.append({"name": item["name"], "status": "missing", "image": str(image)})
            continue
        if image.suffix.lower() in {".tif", ".tiff"}:
            rows.append({"name": item["name"], "status": "unsupported_format", "image": str(image)})
            continue
        env = os.environ.copy()
        if args.max_regions > 0: env["PPOCRV6_DIRECT_MAX_REGIONS"] = str(args.max_regions)
        proc = subprocess.run([str(args.harness), str(args.det), str(args.rec), str(image), str(args.orientation)],
                              text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, check=False)
        detector = re.search(r"detector_regions=(\d+).*orientation=(\w+)", proc.stdout)
        summary = re.search(r"processed=(\d+) rotated_180=(\d+) elapsed_ms=([0-9.]+)", proc.stdout)
        rows.append({"name": item["name"], "status": "ok" if proc.returncode == 0 else "failed",
                     "detector_regions": int(detector.group(1)) if detector else None,
                     "orientation": detector.group(2) if detector else None,
                     "processed": int(summary.group(1)) if summary else None,
                     "rotated_180": int(summary.group(2)) if summary else None,
                     "elapsed_ms": float(summary.group(3)) if summary else None})
    print(json.dumps(rows, indent=2))
    if any(row["status"] == "failed" for row in rows): raise SystemExit(1)


if __name__ == "__main__":
    main()
