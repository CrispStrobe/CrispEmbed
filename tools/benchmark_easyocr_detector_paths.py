#!/usr/bin/env python3
"""Run repeated EasyOCR CRAFT/DBNet detector benchmarks and write JSON.

The native probes and the Miniconda reference probes intentionally remain
separate processes.  This wrapper only joins their measured warm inference
times and decoded box counts; model loading and postprocessing are not hidden
inside a claimed graph ratio.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


CRAFT_REFERENCE = re.compile(
    r"boxes=(?P<boxes>\d+) graph_ms=(?P<ms>[0-9.]+)"
)
CRAFT_NATIVE = re.compile(
    r"boxes=(?P<boxes>\d+) graph_ms=(?P<ms>[0-9.]+)"
)
DBNET_REFERENCE = re.compile(
    r"graph_ms=(?P<ms>[0-9.]+) input=(?P<h>\d+)x(?P<w>\d+)"
)
DBNET_NATIVE = re.compile(
    r"warm_ms=(?P<ms>[0-9.]+).*warm_boxes=(?P<boxes>\d+)"
)


def run(command: list[str]) -> str:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"command failed ({completed.returncode}): {detail[-1000:]}")
    return completed.stdout.strip()


def parse_probe(text: str, pattern: re.Pattern[str], name: str) -> dict[str, object]:
    match = pattern.search(text)
    if not match:
        raise ValueError(f"could not parse {name} probe output: {text!r}")
    values: dict[str, object] = {}
    for key, value in match.groupdict().items():
        values[key] = int(value) if key in {"boxes", "h", "w"} else float(value)
    return values


def detector_record(
    name: str,
    reference: dict[str, object],
    native: dict[str, object],
    reference_command: list[str],
    native_command: list[str],
) -> dict[str, object]:
    ref_ms = float(reference["ms"])
    native_ms = float(native["ms"])
    record: dict[str, object] = {
        "detector": name,
        "reference": {**reference, "command": reference_command},
        "native": {**native, "command": native_command},
        "timing_ms": {
            "reference_warm_graph": ref_ms,
            "native_warm_graph": native_ms,
            "native_over_reference": native_ms / ref_ms if ref_ms else None,
        },
        "output": {
            # The Python DBNet timing probe intentionally does not run
            # postprocessing, so its box count is unavailable. Preserve that
            # as unknown instead of manufacturing a quality failure.
            "box_count_match": (
                reference.get("boxes") == native.get("boxes")
                if "boxes" in reference and "boxes" in native
                else None
            )
        },
    }
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--easyocr-repo", type=Path, required=True)
    parser.add_argument("--craft-checkpoint", type=Path, required=True)
    parser.add_argument("--craft-model", type=Path, required=True)
    parser.add_argument("--craft-reference", type=Path, required=True)
    parser.add_argument("--dbnet-checkpoint", type=Path, required=True)
    parser.add_argument("--dbnet-model", type=Path, required=True)
    parser.add_argument("--dbnet-reference", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--native-craft", type=Path, required=True)
    parser.add_argument("--native-dbnet", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")

    craft_reference_command = [
        str(args.python), str(Path(__file__).with_name("benchmark_easyocr_craft_reference.py")),
        "--easyocr-repo", str(args.easyocr_repo), "--checkpoint", str(args.craft_checkpoint),
        "--image", str(args.image), "--repetitions", str(args.repetitions),
    ]
    craft_native_command = [
        str(args.native_craft), str(args.craft_model), str(args.craft_reference), str(args.repetitions)
    ]
    dbnet_reference_command = [
        str(args.python), str(Path(__file__).with_name("benchmark_dbnet_reference.py")),
        "--checkpoint", str(args.dbnet_checkpoint), "--image", str(args.image),
        "--repetitions", str(args.repetitions),
    ]
    dbnet_native_command = [
        str(args.native_dbnet), str(args.dbnet_model), str(args.dbnet_reference), str(args.repetitions)
    ]

    craft_reference = parse_probe(run(craft_reference_command), CRAFT_REFERENCE, "CRAFT reference")
    craft_native = parse_probe(run(craft_native_command), CRAFT_NATIVE, "CRAFT native")
    dbnet_reference = parse_probe(run(dbnet_reference_command), DBNET_REFERENCE, "DBNet reference")
    dbnet_native = parse_probe(run(dbnet_native_command), DBNET_NATIVE, "DBNet native")
    result = {
        "schema": "crispembed.easyocr.detector-benchmark.v1",
        "image": str(args.image),
        "repetitions": args.repetitions,
        "comparison_policy": "warm graph inference only; reference and native devices are recorded separately",
        "detectors": [
            detector_record("CRAFT", craft_reference, craft_native, craft_reference_command, craft_native_command),
            detector_record("DBNet", dbnet_reference, dbnet_native, dbnet_reference_command, dbnet_native_command),
        ],
    }
    serialized = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
