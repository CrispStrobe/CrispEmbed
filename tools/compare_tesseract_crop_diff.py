#!/usr/bin/env python3
"""Regenerate and diff one Tesseract line crop.

This deliberately stops at the recognizer tensor boundary. It does not call
the system Tesseract CLI, because the installed Homebrew Leptonica currently
cannot reopen local crop files even though the same CLI reads full-page
fixtures successfully.

Run with the repository's Miniconda Python:

    /Users/christianstrobele/miniconda3/bin/python \
        tools/compare_tesseract_crop_diff.py \
        --traineddata /opt/homebrew/share/tessdata/frk.traineddata \
        --model /Volumes/backups/ai/crispembed-gguf/tesseract-frk-q8_0-seeded.gguf \
        --image /tmp/crop-00.png \
        --output-ref /Volumes/backups/ai/crispembed-gguf/tesseract-line-ref.gguf
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> int:
    print("$", " ".join(command))
    completed = subprocess.run(command, check=False)
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traineddata", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="converted Tesseract GGUF")
    parser.add_argument("--image", type=Path, required=True, help="one grayscale line crop")
    parser.add_argument("--output-ref", type=Path, required=True)
    parser.add_argument("--native-diff", type=Path, default=Path("build/test-tesseract-lstm-diff"))
    args = parser.parse_args()

    for path in (args.traineddata, args.model, args.image, args.native_diff):
        if not path.exists():
            parser.error(f"missing path: {path}")
    if args.output_ref.exists():
        parser.error(f"refusing to overwrite existing reference: {args.output_ref}")
    args.output_ref.parent.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parent.parent
    dumper = repo_root / "tools" / "dump_tesseract_reference.py"
    env = os.environ.copy()
    env.pop("TESSDATA_PREFIX", None)
    dump_command = [
        sys.executable,
        str(dumper),
        "--model", str(args.traineddata),
        "--image", str(args.image),
        "--output", str(args.output_ref),
    ]
    print("$", " ".join(dump_command))
    dumped = subprocess.run(dump_command, env=env, check=False)
    if dumped.returncode != 0:
        return dumped.returncode

    return run([
        str(args.native_diff),
        str(args.model),
        str(args.output_ref),
        str(args.image),
    ])


if __name__ == "__main__":
    raise SystemExit(main())
