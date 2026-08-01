#!/usr/bin/env python3
"""Compare official Tesseract line geometry with CrispEmbed page segmentation.

This intentionally compares geometry only. Recognition text and confidence are
separate acceptance gates because a different crop can change the recognizer's
decoded output even when the line ordering is correct.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path


BOX_RE = re.compile(
    r"candidate=\d+ box=(?P<x>[-0-9.]+),(?P<y>[-0-9.]+) "
    r"(?P<w>[-0-9.]+)x(?P<h>[-0-9.]+)"
)


def run(command: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, env=env, timeout=900, check=False)


def official_lines(image: Path, lang: str, psm: int) -> list[tuple[float, float, float, float]]:
    proc = run(["tesseract", str(image), "stdout", "--psm", str(psm), "-l", lang, "tsv"])
    if proc.returncode != 0:
        raise RuntimeError(f"tesseract failed ({proc.returncode}): {proc.stderr[-500:]}")
    lines = []
    for raw in proc.stdout.splitlines()[1:]:
        fields = raw.split("\t", 11)
        if len(fields) < 12 or fields[0] != "4":
            continue
        try:
            x, y, w, h = (float(value) for value in fields[6:10])
        except ValueError:
            continue
        lines.append((x, y, w, h))
    return lines


def native_lines(cli: Path, det_model: Path, rec_model: Path, image: Path, component: bool,
                 baseline: bool) -> list[tuple[float, float, float, float]]:
    env = os.environ.copy()
    env["CRISPEMBED_TESSERACT_PAGESEG_DEBUG"] = "1"
    if component:
        env["CRISPEMBED_TESSERACT_COMPONENT_PAGESEG"] = "1"
    if baseline:
        env["CRISPEMBED_TESSERACT_COMPONENT_BASELINE"] = "1"
    proc = run(
        [
            str(cli),
            "-m",
            str(rec_model),
            "--ocr-pipeline",
            str(image),
            "--ocr-engine",
            "tesseract",
            "--ocr-det",
            str(det_model),
            "--ocr-rec",
            str(rec_model),
            "--tesseract-pageseg",
        ],
        env,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"CrispEmbed failed ({proc.returncode}): {proc.stderr[-1000:]}")
    return [
        (float(match.group("x")), float(match.group("y")), float(match.group("w")), float(match.group("h")))
        for match in BOX_RE.finditer(proc.stdout + proc.stderr)
    ]


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def compare(reference: list[tuple[float, float, float, float]], mine: list[tuple[float, float, float, float]]) -> dict:
    pairs = []
    for index, ref_box in enumerate(reference):
        if index < len(mine):
            mine_box = mine[index]
            pairs.append(
                {
                    "index": index,
                    "iou": round(iou(ref_box, mine_box), 6),
                    "reference": list(ref_box),
                    "native": list(mine_box),
                    "delta": [round(mine_box[i] - ref_box[i], 3) for i in range(4)],
                }
            )
    return {
        "reference_lines": len(reference),
        "native_lines": len(mine),
        "count_delta": len(mine) - len(reference),
        "mean_indexed_iou": round(sum(pair["iou"] for pair in pairs) / len(pairs), 6) if pairs else 0.0,
        "pairs": pairs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--det-model", type=Path, required=True)
    parser.add_argument("--rec-model", type=Path, required=True)
    parser.add_argument("--cli", type=Path, default=Path("build/crispembed"))
    parser.add_argument("--lang", default="eng")
    parser.add_argument("--psm", type=int, default=3)
    parser.add_argument("--component", action="store_true")
    parser.add_argument("--baseline", action="store_true", help="use the experimental baseline-row matcher")
    parser.add_argument("--min-native-lines", type=int, help="fail if native line count is below this value")
    parser.add_argument("--min-iou", type=float, help="fail if indexed mean IoU is below this value")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    comparison = compare(
        official_lines(args.image, args.lang, args.psm),
        native_lines(args.cli, args.det_model, args.rec_model, args.image, args.component, args.baseline),
    )
    checks = {}
    if args.min_native_lines is not None:
        checks["min_native_lines"] = comparison["native_lines"] >= args.min_native_lines
    if args.min_iou is not None:
        checks["min_iou"] = comparison["mean_indexed_iou"] >= args.min_iou
    result = {
        "image": str(args.image),
        "psm": args.psm,
        "component": args.component,
        "baseline": args.baseline,
        "comparison": comparison,
        "acceptance": {"passed": all(checks.values()) if checks else None, "checks": checks},
    }
    serialized = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(serialized)
    print(serialized, end="")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
