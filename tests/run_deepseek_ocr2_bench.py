#!/usr/bin/env python3
"""Run the deepseek-ocr2 lane over a fixture directory, one process per page.

T14 needs two things the generic parity harness does not give it: the decoded
text of *one* lane saved per fixture so two arms can be diffed byte-for-byte,
and the `decode=` field of `[deepseek-ocr2-stage-bench]` rather than its
`total=` (the vision tower dominates `total`, and no decode change moves it).

Each page runs as a SEPARATE process on purpose.  A second inference spawned
immediately after the first in the same shell has been observed to exit at 0 s
on a GPU/resource-release race, which mints a fake "win"; the returncode and a
non-empty transcript are both checked before any timing is recorded.

    python tests/run_deepseek_ocr2_bench.py \
        --binary build/crispembed --model ~/.cache/crispembed-local/ds.gguf \
        --images ~/crispembed-ocr-synth --out tests/results/t14/legacy-synth \
        --env DS2_LEGACY_DECODE=1
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ocr_external_parity import DEEPSEEK_OCR2_DECODE_RE, REGIONS_RE, STAGE_BENCH  # noqa: E402

FIELD_RE = re.compile(r"\[deepseek-ocr2-stage-bench\] (.*)")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def parse_bench(err: str) -> dict:
    """Pull every `name=value` pair off the stage-bench line."""
    m = FIELD_RE.search(err)
    if not m:
        return {}
    out: dict[str, float | str] = {}
    for k, v in re.findall(r"(\w+)=([-\w.]+)", m.group(1)):
        try:
            out[k] = float(v)
        except ValueError:
            out[k] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True, type=Path)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--env", action="append", default=[],
                    help="KEY=VALUE applied to every run (repeatable)")
    ap.add_argument("--gpu-backend", default=None,
                    help="passed through as --gpu-backend (e.g. cpu)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--labelled-only", action="store_true",
                    help="restrict to fixtures present in the directory's "
                         "ground_truth.json (the CC0 dir holds 14 images but "
                         "only 5 are transcribed, and the rest are Fraktur / "
                         "Arabic / handwriting the English gate does not score)")
    args = ap.parse_args()

    env = os.environ.copy()
    env["CRISPEMBED_DEEPSEEK_OCR2_BENCH"] = "1"
    for kv in args.env:
        k, _, v = kv.partition("=")
        env[k] = v

    args.out.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if args.labelled_only:
        gt = args.images / "ground_truth.json"
        keep = {r["file"] for r in json.loads(gt.read_text())["records"]}
        images = [p for p in images if p.name in keep]
    if args.limit:
        images = images[:args.limit]

    rows, failures = [], []
    for img in images:
        cmd = [str(args.binary), "--ocr-pipeline", str(img),
               "--ocr-engine", "deepseek-ocr2", "--ocr-rec", str(args.model)]
        if args.gpu_backend:
            cmd += ["--gpu-backend", args.gpu_backend]
        t0 = time.perf_counter()
        p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=3600)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        text = REGIONS_RE.sub("", p.stdout).strip()
        bench = parse_bench(p.stderr)
        # A crash or an empty transcript is never timed as a win.
        ok = p.returncode == 0 and bool(text)
        if not ok:
            failures.append({"fixture": img.name, "returncode": p.returncode,
                             "stderr_tail": p.stderr[-800:]})
        (args.out / (img.stem + ".txt")).write_text(text)
        rows.append({"fixture": img.name, "ok": ok, "returncode": p.returncode,
                     "chars": len(text), "wall_ms": round(wall_ms, 1), **bench})
        print(f"{img.name:36s} ok={ok} chars={len(text):5d} "
              f"decode={bench.get('decode')} total={bench.get('total')} "
              f"path={bench.get('decode_path')} kv={bench.get('kv')}", flush=True)

    doc = {"binary": str(args.binary), "model": str(args.model),
           "images": str(args.images), "env": args.env,
           "gpu_backend": args.gpu_backend or "default",
           "rows": rows, "failures": failures}
    (args.out / "runs.json").write_text(json.dumps(doc, indent=2) + "\n")
    print(f"\n{len(rows)} pages, {len(failures)} failures -> {args.out}")
    # Referenced so the shared regexes stay the single source of truth.
    assert "deepseek-ocr2" in STAGE_BENCH and DEEPSEEK_OCR2_DECODE_RE
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
