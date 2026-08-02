#!/usr/bin/env python3
"""Measure repeated PP-OCRv6 graph runs while preserving decoded output.

The recognizer graph is width-keyed.  This benchmark deliberately compares
same-width runs with alternating widths so Metal scheduler/planning costs do
not get hidden in a single aggregate number.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path


READY = re.compile(r"persistent GGML graph ready \(([^,]+), [^,]+, (\d+)x(\d+)x(\d+)\)")


def run(exe: Path, model: Path, images: list[Path]) -> dict:
    env = os.environ.copy()
    env.update({
        "CRISPEMBED_PPOCRV6_GRAPH": "1",
        "CRISPEMBED_PPOCRV6_GRAPH_ACCEPT": "1",
    })
    command = [str(exe), str(model), *(str(image) for image in images)]
    started = time.perf_counter()
    proc = subprocess.run(command, env=env, text=True, capture_output=True, check=False)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    texts = [line[5:] for line in proc.stdout.splitlines() if line.startswith("text=")]
    shapes = [
        {"backend": backend, "x": int(x), "y": int(y), "z": int(z)}
        for backend, x, y, z in READY.findall(proc.stderr)
    ]
    return {
        "returncode": proc.returncode,
        "elapsed_ms": round(elapsed_ms, 3),
        "images": [str(image) for image in images],
        "texts": texts,
        "graph_shapes": shapes,
        "stderr_tail": proc.stderr.splitlines()[-4:],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--short", type=Path, required=True)
    parser.add_argument("--wide", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.n < 1:
        parser.error("--n must be positive")
    short = [args.short] * args.n
    wide = [args.wide] * args.n
    cases = {
        "grouped_short": short,
        "grouped_wide": wide,
        "alternating": [p for pair in zip(short, wide) for p in pair],
    }
    result = {name: run(args.exe, args.model, images) for name, images in cases.items()}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return 0 if all(row["returncode"] == 0 for row in result.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
