#!/usr/bin/env python3
"""Repeat the Tesseract page comparator and summarize quality/cost results."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))
    return ordered[index]


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "min": round(min(values), 3) if values else 0.0,
        "median": round(statistics.median(values), 3) if values else 0.0,
        "p90": round(percentile(values, 0.9), 3),
        "max": round(max(values), 3) if values else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--det-model", type=Path, required=True)
    parser.add_argument("--rec-model", type=Path, required=True)
    parser.add_argument("--native-test", type=Path, required=True)
    parser.add_argument("--lang", default="eng")
    parser.add_argument("--psm", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--recode-beam", type=int, default=0)
    parser.add_argument("--dawg-score", action="store_true")
    parser.add_argument("--dawg-prefix-score", action="store_true")
    parser.add_argument("--compose", action="store_true")
    parser.add_argument("--projection", action="store_true")
    parser.add_argument("--component", action="store_true")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--scratch", action="store_true", help="enable the gated activation scratch prototype")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if (args.dawg_score or args.dawg_prefix_score) and args.recode_beam <= 1:
        parser.error("DAWG scoring requires --recode-beam > 1")
    if sum((args.projection, args.component, args.baseline)) > 1:
        parser.error("segmentation policies are mutually exclusive")

    tool = Path(__file__).with_name("compare_tesseract_page_metrics.py")
    policy = "projection" if args.projection else "component" if args.component else "baseline" if args.baseline else "legacy-fallback"
    records = []
    for _ in range(args.repeats):
        command = [
            sys.executable,
            str(tool),
            "--image",
            str(args.image),
            "--det-model",
            str(args.det_model),
            "--rec-model",
            str(args.rec_model),
            "--native-test",
            str(args.native_test),
            "--lang",
            args.lang,
            "--psm",
            str(args.psm),
            "--workers",
            str(args.workers),
            "--benchmark",
        ]
        if args.recode_beam:
            command.extend(["--recode-beam", str(args.recode_beam)])
        if args.dawg_score:
            command.append("--dawg-score")
        if args.dawg_prefix_score:
            command.append("--dawg-prefix-score")
        if args.compose:
            command.append("--compose")
        if args.projection:
            command.append("--projection")
        elif args.component:
            command.append("--component")
        elif args.baseline:
            command.append("--baseline")
        env = os.environ.copy()
        env.pop("CRISPEMBED_TESSERACT_REUSE_SCRATCH", None)
        if args.scratch:
            env["CRISPEMBED_TESSERACT_REUSE_SCRATCH"] = "1"
        proc = subprocess.run(command, capture_output=True, text=True, env=env, timeout=900, check=False)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr[-1000:] or proc.stdout[-1000:] or "page comparator failed")
        records.append(json.loads(proc.stdout))

    official_ms = [record["official_tesseract"]["elapsed_ms"] for record in records]
    native_ms = [record["native_crispembed"]["elapsed_ms"] for record in records]
    stage_ms = [record["native_crispembed"]["benchmark"]["total_ms"] for record in records if record["native_crispembed"]["benchmark"]]
    recognize_ms = [record["native_crispembed"]["benchmark"]["recognize_ms"] for record in records if record["native_crispembed"]["benchmark"]]
    result = {
        "fixture": str(args.image),
        "policy": policy,
        "workers": args.workers,
        "scratch": args.scratch,
        "recode_beam": args.recode_beam,
        "dawg_score": args.dawg_score,
        "dawg_prefix_score": args.dawg_prefix_score,
        "compose": args.compose,
        "repeats": len(records),
        "provenance": records[-1]["provenance"],
        "quality": {
            "regions": [record["native_crispembed"]["regions"] for record in records],
            "cer": summarize([record["comparison"]["cer"] for record in records]),
            "wer": summarize([record["comparison"]["wer"] for record in records]),
            "confidence_delta": summarize([record["comparison"]["confidence_delta"] for record in records]),
            "output_comparison": {
                "identical": records[-1]["official_tesseract"]["text"] == records[-1]["native_crispembed"]["text"],
                "official_text": records[-1]["official_tesseract"]["text"],
                "native_text": records[-1]["native_crispembed"]["text"],
                "official_lines": records[-1]["official_tesseract"]["lines"],
                "native_regions": records[-1]["native_crispembed"]["regions"],
            },
        },
        "timing_ms": {
            "official_cli": summarize(official_ms),
            "native_subprocess": summarize(native_ms),
            "native_stage": summarize(stage_ms),
            "native_recognize": summarize(recognize_ms),
        },
        "runs": records,
    }
    serialized = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(serialized)
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
