#!/usr/bin/env python3
"""Capture the model-gated PP-OCRv6 pipeline sweep as stable JSON.

The native test owns model loading and the detector→quad→orientation→recognizer
handoff.  This wrapper turns its per-fixture INFO lines into benchmark rows so
latency/quality evidence is not trapped in an interactive log.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path


ROW = re.compile(r"^  INFO: (?P<fixture>.+): (?P<regions>\d+) regions, "
                 r"(?P<chars>\d+) chars \(conf=(?P<confidence>[0-9.]+)\)$")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-binary", default="build/test-ocr-orchestrator", type=Path)
    parser.add_argument("--models-dir", required=True, type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    required = (
        "PP-OCRv6_tiny_det-f16.gguf",
        "PP-OCRv6_tiny_rec-q8-head.gguf",
        "PP-LCNet_x1_0_textline_ori-f16.gguf",
    )
    missing = [name for name in required if not (args.models_dir / name).is_file()]
    if missing:
        parser.error("missing required model(s): " + ", ".join(missing))
    env = os.environ.copy()
    env["CRISPEMBED_MODELS_DIR"] = str(args.models_dir)
    proc = subprocess.run([str(args.test_binary)], capture_output=True, text=True, env=env, check=False)
    rows = []
    for line in proc.stdout.splitlines():
        match = ROW.match(line)
        if match:
            row = match.groupdict()
            row["regions"] = int(row["regions"])
            row["chars"] = int(row["chars"])
            row["confidence"] = float(row["confidence"])
            rows.append(row)
    if proc.returncode != 0:
        raise SystemExit(f"native PP-OCRv6 regression failed (exit {proc.returncode})\n{proc.stderr[-2000:]}")
    if len(rows) != 10:
        raise SystemExit(f"expected 10 PP-OCRv6 benchmark rows, got {len(rows)}")
    result = {"version": 1, "engine": "ppocrv6", "orientation": "pplcnet-0-180", "rows": rows}
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload)
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
