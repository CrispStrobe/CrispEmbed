#!/usr/bin/env python3
"""Small reproducible DBNet+TrOCR benchmark.

Reports detection recall proxy (number of regions), decoded text, and the
existing pipeline stage timings. It intentionally uses the checked-in test
binaries so model/runtime changes are measured through the real C++ path.

Example:
  python3 tests/ocr_benchmark.py \
    --models-dir /Volumes/backups/ai/crispembed-gguf \
    --image tests/regression/images/fox.png \
    --build-dir build --json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path


def run(cmd: list[str], env: dict[str, str]) -> tuple[str, str, int]:
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    return p.stdout, p.stderr, p.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default=os.environ.get("CRISPEMBED_MODELS_DIR", ""))
    ap.add_argument("--image", required=True)
    ap.add_argument("--build-dir", default="build")
    ap.add_argument("--det", default="dbnet-ic15-q4_k.gguf")
    ap.add_argument("--rec", default="trocr-small-printed-q8_0.gguf")
    ap.add_argument("--allow-dangerous-q4", action="store_true",
                    help="explicitly allow the known-bad TrOCR Q4_K recognizer")
    ap.add_argument("--prob-threshold", type=float, default=0.3)
    ap.add_argument("--box-threshold", type=float, default=0.5)
    ap.add_argument("--expect-regions", type=int, default=None)
    ap.add_argument("--expect-text", action="append", default=[],
                    help="Expected region text; repeat for each region")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if not args.models_dir:
        ap.error("--models-dir or CRISPEMBED_MODELS_DIR is required")
    model_dir = Path(args.models_dir)
    image = Path(args.image)
    build = Path(args.build_dir)
    det = model_dir / args.det
    rec = model_dir / args.rec
    for path in (det, rec, image):
        if not path.exists():
            ap.error(f"missing input: {path}")

    env = os.environ.copy()
    env["CRISPEMBED_OCR_PIPELINE_BENCH"] = "1"
    if args.allow_dangerous_q4:
        env["CRISPEMBED_DEBUG_ALLOW_OCR_Q4"] = "1"
    detect_out, detect_err, detect_rc = run(
        [str(build / "test-ocr-detect"), str(det), str(image), str(args.prob_threshold), str(args.box_threshold)], env
    )
    pipeline_out, pipeline_err, pipeline_rc = run(
        [str(build / "test-ocr-pipeline"), str(det), str(rec), str(image)], env
    )

    detected = re.search(r"Detected (\d+) text regions", detect_out)
    recognized = re.search(r"recognized (\d+)/(\d+) regions", pipeline_err)
    timings = {
        name: float(value)
        for name, value in re.findall(r"\[ocr_pipeline-bench\] ([^:]+): ([0-9.]+) ms", pipeline_err)
    }
    regions = []
    for match in re.finditer(r'\[\s*\d+\].*?"(.*?)"', pipeline_out):
        regions.append(match.group(1))

    result = {
        "image": str(image),
        "detector": str(det),
        "recognizer": str(rec),
        "prob_threshold": args.prob_threshold,
        "box_threshold": args.box_threshold,
        "detected_regions": int(detected.group(1)) if detected else None,
        "recognized_regions": int(recognized.group(1)) if recognized else None,
        "detected_input_regions": int(recognized.group(2)) if recognized else None,
        "regions": regions,
        "timings_ms": timings,
        "detect_exit": detect_rc,
        "pipeline_exit": pipeline_rc,
        "assertions": [],
    }
    if args.expect_regions is not None and result["recognized_regions"] != args.expect_regions:
        result["assertions"].append(
            f"recognized_regions={result['recognized_regions']} != expected {args.expect_regions}"
        )
    if args.expect_text and regions != args.expect_text:
        result["assertions"].append(f"regions={regions!r} != expected {args.expect_text!r}")
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"detected={result['detected_regions']} recognized={result['recognized_regions']}")
        print("timings_ms=" + json.dumps(timings, sort_keys=True))
        print("text=" + " | ".join(regions))
    return 0 if detect_rc == 0 and pipeline_rc == 0 and not result["assertions"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
