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


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def geometry_matches(native: list[dict[str, float]], official: list[dict[str, int]]) -> list[dict[str, int]]:
    """Monotonic one-to-one matching for diagnostic geometry alignment.

    Rows can be merged or missing, so this intentionally does not force every
    native row to pair with the same-index official row. It matches each native
    row to the best not-yet-consumed official row at or below the current
    vertical position, preferring vertical overlap and then centre distance.
    """
    matches = []
    cursor = 0
    for native_index, n in enumerate(native):
        n_top, n_bottom = n["box_y"], n["box_y"] + n["box_h"]
        n_center = (n_top + n_bottom) / 2.0
        candidates = []
        for official_index in range(cursor, len(official)):
            o = official[official_index]
            o_top, o_bottom = o["top"], o["top"] + o["height"]
            o_center = (o_top + o_bottom) / 2.0
            overlap = _overlap(n_top, n_bottom, o_top, o_bottom)
            distance = abs(n_center - o_center)
            limit = max(n["box_h"], float(o["height"])) * 2.5
            if overlap > 0.0 or distance <= limit:
                candidates.append((overlap, -distance, official_index))
        if not candidates:
            continue
        _, _, official_index = max(candidates)
        matches.append({"native_index": native_index, "official_index": official_index})
        cursor = official_index + 1
    return matches


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


def compare_geometry(native: list[dict[str, float]], official: list[dict[str, int]]) -> dict:
    """Compare geometry after monotonic matching, retaining unmatched rows."""
    matches = geometry_matches(native, official)
    deltas = []
    for match in matches:
        n = native[match["native_index"]]
        o = official[match["official_index"]]
        deltas.append({
            **match,
            "dx": n["box_x"] - o["left"],
            "dy": n["box_y"] - o["top"],
            "dw": n["box_w"] - o["width"],
            "dh": n["box_h"] - o["height"],
        })
    matched_native = {row["native_index"] for row in matches}
    matched_official = {row["official_index"] for row in matches}
    merged_official_groups = []
    for native_index, n in enumerate(native):
        n_top, n_bottom = n["box_y"], n["box_y"] + n["box_h"]
        covered = []
        for official_index, o in enumerate(official):
            o_top, o_bottom = o["top"], o["top"] + o["height"]
            overlap = _overlap(n_top, n_bottom, o_top, o_bottom)
            if overlap >= float(o["height"]) * 0.5:
                covered.append(official_index)
        if len(covered) > 1:
            primary = max(covered, key=lambda index: official[index]["width"] * official[index]["height"])
            primary_box = official[primary]
            primary_top = primary_box["top"]
            primary_bottom = primary_top + primary_box["height"]
            primary_left = primary_box["left"]
            primary_right = primary_left + primary_box["width"]
            nested = []
            for index in covered:
                if index == primary:
                    continue
                box = official[index]
                if (primary_left <= box["left"] and box["left"] + box["width"] <= primary_right
                        and primary_top <= box["top"] and box["top"] + box["height"] <= primary_bottom):
                    nested.append(index)
            merged_official_groups.append({"native_index": native_index,
                                           "official_indices": covered,
                                           "primary_official_index": primary,
                                           "nested_official_indices": nested})
    summary = {}
    for key in ("dx", "dy", "dw", "dh"):
        values = [row[key] for row in deltas]
        summary[key] = {"mean": sum(values) / len(values) if values else 0.0,
                        "max_abs": max(map(abs, values), default=0.0)}
    return {
        "native_lines": len(native),
        "official_lines": len(official),
        "count_delta": len(native) - len(official),
        "alignment": "monotonic-geometry",
        "alignment_valid": len(native) == len(official),
        "matched_rows": len(matches),
        "unmatched_native": sorted(set(range(len(native))) - matched_native),
        "unmatched_official": sorted(set(range(len(official))) - matched_official),
        "merged_official_groups": merged_official_groups,
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
    parser.add_argument("--match-by-geometry", action="store_true",
                        help="match rows monotonically by vertical geometry instead of index")
    args = parser.parse_args()
    native = native_crops(args.manifest)
    official = official_lines(args.image, args.lang, args.psm, args.tessdata_dir)
    result = compare_geometry(native, official) if args.match_by_geometry else compare(native, official)
    print(json.dumps(result, indent=2))
    return 0 if result["count_delta"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
