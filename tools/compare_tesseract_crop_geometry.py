#!/usr/bin/env python3
"""Compare native Tesseract crop geometry with official TSV line boxes."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import subprocess
from pathlib import Path


def official_lines(image: Path, lang: str, psm: int, tessdata_dir: Path | None) -> list[dict[str, int]]:
    command = ["tesseract", str(image), "stdout", "--psm", str(psm), "-l", lang, "tsv"]
    if tessdata_dir:
        command[3:3] = ["--tessdata-dir", str(tessdata_dir)]
    env = os.environ.copy()
    if tessdata_dir:
        env.pop("TESSDATA_PREFIX", None)
    proc = subprocess.run(command, env=env, text=True, capture_output=True, check=False, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"tesseract TSV failed with exit {proc.returncode}: {proc.stderr.strip()}")
    rows = []
    for row in csv.DictReader(io.StringIO(proc.stdout), delimiter="\t"):
        if row.get("level") == "4":
            rows.append({key: int(row[key]) for key in ("left", "top", "width", "height")})
    return rows


def native_crops(manifest: Path) -> list[dict[str, float]]:
    with manifest.open(newline="") as stream:
        return [
            {key: float(row[key]) for key in ("box_x", "box_y", "box_w", "box_h", "crop_w", "crop_h")}
            for row in csv.DictReader(stream, delimiter="\t")
        ]


def compare(native: list[dict[str, float]], official: list[dict[str, int]]) -> dict:
    count = min(len(native), len(official))
    deltas = []
    for index in range(count):
        n, o = native[index], official[index]
        deltas.append(
            {
                "index": index,
                "dx": n["box_x"] - o["left"],
                "dy": n["box_y"] - o["top"],
                "dw": n["box_w"] - o["width"],
                "dh": n["box_h"] - o["height"],
            }
        )
    summary = {}
    for key in ("dx", "dy", "dw", "dh"):
        values = [row[key] for row in deltas]
        summary[key] = {"mean": sum(values) / len(values) if values else 0.0, "max_abs": max(map(abs, values), default=0.0)}
    return {
        "native_lines": len(native),
        "official_lines": len(official),
        "count_delta": len(native) - len(official),
        "alignment": "reading-order-index",
        "alignment_valid": len(native) == len(official),
        "paired_rows": count,
        "summary": summary,
        "rows": deltas,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lang", default="frk")
    parser.add_argument("--psm", type=int, default=6)
    parser.add_argument("--tessdata-dir", type=Path)
    args = parser.parse_args()
    result = compare(native_crops(args.manifest), official_lines(args.image, args.lang, args.psm, args.tessdata_dir))
    print(json.dumps(result, indent=2))
    return 0 if result["count_delta"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
