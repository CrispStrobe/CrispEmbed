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
import time
from pathlib import Path


ROW = re.compile(r"^  INFO: (?P<fixture>.+): (?P<regions>\d+) regions, "
                 r"(?P<chars>\d+) chars \(conf=(?P<confidence>[0-9.]+), "
                 r"time_ms=(?P<time_ms>[0-9.]+)\)$")
STAGE_ROW = re.compile(
    r"^\[ppocrv6-stage-bench\] detect=(?P<detect>[0-9.]+) ms "
    r"crop=(?P<crop>[0-9.]+) ms orientation=(?P<orientation>[0-9.]+) ms "
    r"recognize=(?P<recognize>[0-9.]+) ms total=(?P<total>[0-9.]+) ms "
    r"boxes=(?P<boxes>\d+) results=(?P<results>\d+)$"
)


def parse_rows(text: str, variant: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        match = ROW.match(line)
        if match:
            row = match.groupdict()
            row["variant"] = variant
            row["regions"] = int(row["regions"])
            row["chars"] = int(row["chars"])
            row["confidence"] = float(row["confidence"])
            row["time_ms"] = float(row["time_ms"])
            rows.append(row)
    return rows


def parse_stage_rows(text: str) -> list[dict]:
    """Parse one per-fixture stage record from native stderr."""
    rows = []
    for line in text.splitlines():
        match = STAGE_ROW.match(line)
        if match:
            row = match.groupdict()
            for name in ("detect", "crop", "orientation", "recognize", "total"):
                row[f"{name}_ms"] = float(row.pop(name))
            row["boxes"] = int(row.pop("boxes"))
            row["results"] = int(row.pop("results"))
            rows.append(row)
    return rows


def write_json(path: Path | None, result: dict) -> None:
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload)
    else:
        print(payload, end="")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-binary", default="build/test-ocr-orchestrator", type=Path)
    default_models_dir = Path(os.environ.get("CRISPEMBED_GGUF_DIR", "/Volumes/backups/ai/crispembed-gguf"))
    parser.add_argument("--models-dir", type=Path, default=default_models_dir,
                        help="GGUF cache directory (default: CRISPEMBED_GGUF_DIR or /Volumes/backups/ai/crispembed-gguf)")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--variants", nargs="+", default=["tiny", "small", "medium"],
                        choices=["tiny", "small", "medium"])
    parser.add_argument("--timeout", type=float, default=900.0,
                        help="maximum seconds per model variant (default: 900)")
    parser.add_argument("--fixture-limit", type=int, default=10,
                        help="run only the first N live fixtures for profiling (default: 10)")
    parser.add_argument("--fixture-start", type=int, default=0,
                        help="zero-based fixture offset for targeted profiling (default: 0)")
    args = parser.parse_args()
    if args.fixture_limit < 1 or args.fixture_limit > 10:
        parser.error("--fixture-limit must be between 1 and 10")
    if args.fixture_start < 0 or args.fixture_start >= 10 or args.fixture_start + args.fixture_limit > 10:
        parser.error("--fixture-start plus --fixture-limit must select fixtures within 0..9")
    all_rows = []
    for variant in args.variants:
        required = (f"PP-OCRv6_{variant}_det-f16.gguf", f"PP-OCRv6_{variant}_rec-q8-head.gguf",
                    "PP-LCNet_x1_0_textline_ori-f16.gguf")
        missing = [name for name in required if not (args.models_dir / name).is_file()]
        if missing:
            parser.error(f"{variant}: missing required model(s): " + ", ".join(missing))
        env = os.environ.copy()
        env["CRISPEMBED_MODELS_DIR"] = str(args.models_dir)
        env["CRISPEMBED_PPOCRV6_VARIANT"] = variant
        env["CRISPEMBED_PPOCRV6_FIXTURE_LIMIT"] = str(args.fixture_limit)
        env["CRISPEMBED_PPOCRV6_FIXTURE_START"] = str(args.fixture_start)
        env["CRISPEMBED_PPOCRV6_BENCH"] = "1"
        started = time.monotonic()
        try:
            proc = subprocess.run([str(args.test_binary)], capture_output=True, text=True, env=env, check=False,
                                  timeout=args.timeout)
        except subprocess.TimeoutExpired as exc:
            elapsed = round(time.monotonic() - started, 2)
            partial = exc.stdout or ""
            partial_err = exc.stderr or ""
            if isinstance(partial, bytes):
                partial = partial.decode(errors="replace")
            if isinstance(partial_err, bytes):
                partial_err = partial_err.decode(errors="replace")
            timeout_rows = parse_rows(partial, variant)
            stage_rows = parse_stage_rows(partial_err)
            for row, stages in zip(timeout_rows, stage_rows):
                row["stages"] = stages
            for row in timeout_rows[len(stage_rows):]:
                row["stages"] = None
            timeout_result = {
                "version": 2,
                "status": "timeout",
                "engine": "ppocrv6",
                "orientation": "pplcnet-0-180",
                "variant": variant,
                "fixture_start": args.fixture_start,
                "fixture_limit": args.fixture_limit,
                "timeout_seconds": args.timeout,
                "elapsed_seconds": elapsed,
                "rows": timeout_rows,
            }
            if args.json_out:
                write_json(args.json_out, timeout_result)
            raise SystemExit(f"native PP-OCRv6 {variant} regression timed out after {elapsed}s; "
                             "use --timeout to override") from exc
        rows = parse_rows(proc.stdout, variant)
        stage_rows = parse_stage_rows(proc.stderr)
        if len(stage_rows) == len(rows):
            for row, stages in zip(rows, stage_rows):
                row["stages"] = stages
                row["stage_telemetry_status"] = "complete"
        else:
            # Keep total-output compatibility for older binaries, but make a
            # missing stage telemetry record explicit in the artifact.
            for row in rows:
                row["stages"] = None
                row["stage_telemetry_status"] = f"unavailable:{len(stage_rows)}/{len(rows)}"
        if proc.returncode != 0:
            raise SystemExit(f"native PP-OCRv6 {variant} regression failed (exit {proc.returncode})\n"
                             f"{proc.stderr[-2000:]}")
        if len(rows) != args.fixture_limit:
            raise SystemExit(f"expected {args.fixture_limit} PP-OCRv6 {variant} benchmark rows, got {len(rows)}")
        all_rows.extend(rows)
    write_json(args.json_out, {"version": 2, "status": "ok", "engine": "ppocrv6",
                               "orientation": "pplcnet-0-180", "fixture_start": args.fixture_start,
                               "fixture_limit": args.fixture_limit, "rows": all_rows})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
