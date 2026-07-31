#!/usr/bin/env python3
"""Run the real-world corpus through cheap, model-free/portable smoke paths."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def timed(cmd: list[str]) -> dict:
    t = time.perf_counter()
    p = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=60)
    return {"ms": (time.perf_counter() - t) * 1000, "returncode": p.returncode,
            "stdout": p.stdout[:2000], "stderr": p.stderr[-500:]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="")
    args = ap.parse_args()
    image_dir = ROOT / "tests/regression/images/cc0"
    rows = []
    for image in sorted(image_dir.iterdir()):
        if image.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            continue
        row = {"fixture": str(image.relative_to(ROOT))}
        row["tesseract"] = timed(["tesseract", str(image), "stdout"])
        row["skew"] = timed([str(ROOT / "build/crispembed"), "-m", "tesseract-eng-f16.gguf",
                              "--find-skew", str(image)])
        row["content"] = timed([str(ROOT / "build/crispembed"), "-m", "tesseract-eng-f16.gguf",
                                 "--detect-content", str(image)])
        rows.append(row)
        print(f"{image.name:28} tesseract={row['tesseract']['returncode']} "
              f"skew={row['skew']['returncode']} content={row['content']['returncode']}", flush=True)
    result = {"fixtures": rows}
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
