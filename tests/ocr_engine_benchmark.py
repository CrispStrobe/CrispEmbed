#!/usr/bin/env python3
"""Manifest-driven live OCR/OMR benchmark.

This deliberately exercises the public crispembed CLI, so timings include
model loading and preprocessing as a user sees them.  Engines without a local
GGUF or a checked-in sample are reported as skipped, never silently omitted.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import time
from urllib.parse import quote
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests/regression/manifest.json"
DEFAULT_MODEL_DIR = Path("/Volumes/backups/ai/crispembed-gguf")


def artifact_filename(spec: str | dict | None) -> str | None:
    """Return the local artifact name for either legacy or structured specs."""
    if isinstance(spec, dict):
        return spec.get("file")
    return spec


def pipeline_engine(entry: dict) -> str:
    """Use the routed engine family, not a tiered manifest display name."""
    return entry.get("pipeline_engine") or entry.get("engine") or entry.get("name", "")


def normalize(s: str) -> str:
    s = s.replace("\r", "").strip()
    s = re.sub(r"(?m)^regions=\d+\s+mean_conf=[0-9.]+\s*$", "", s)
    # SmolDocling emits DocTags coordinates around text payloads.  Compare
    # payload quality here; retain the raw output so duplicate/structure
    # errors remain visible in the benchmark artifact.
    s = re.sub(r"</?text>", "", s)
    s = re.sub(r"(?m)^\d+>\d+>\d+>\d+>", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def quality(output: str, expected: str | None) -> dict:
    if not expected:
        return {"status": "unscored"}
    got, want = normalize(output), normalize(expected)
    d = distance(got, want)
    return {
        "status": "exact" if got == want else "cer",
        "exact": got == want,
        "cer": d / max(1, len(want)),
        "edit_distance": d,
    }


def runtime_failed(meta: dict, stderr: str) -> bool:
    """Treat native load/runtime diagnostics as failures even with exit 0."""
    if meta.get("timed_out") or meta.get("returncode") != 0:
        return True
    return any(marker in stderr.lower() for marker in (
        "failed to load", "load failed", "missing stem", "fatal", "error:"))


def run(cmd: list[str], timeout: float) -> tuple[dict, str, str]:
    started = time.perf_counter()
    try:
        p = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True,
                           timeout=timeout, env=os.environ.copy())
        timed_out = False
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout or b""
        stderr = e.stderr or b""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        p = subprocess.CompletedProcess(cmd, 124, stdout, stderr)
        timed_out = True
    elapsed = (time.perf_counter() - started) * 1000
    stdout = p.stdout or ""
    stderr = p.stderr or ""
    return {"ms": elapsed, "returncode": p.returncode, "timed_out": timed_out}, stdout, stderr


def download_model(entry: dict, dest: Path) -> bool:
    gguf = entry.get("gguf") or {}
    repo, filename = gguf.get("repo"), gguf.get("file")
    if not repo or not filename:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{repo}/resolve/{gguf.get('revision', 'main')}/{quote(filename)}"
    print(f"downloading {entry.get('name')} -> {dest}", flush=True)
    p = subprocess.run(["curl", "-L", "--fail", "--retry", "3", "--retry-delay", "3",
                        "-C", "-", "-o", str(dest), url], cwd=ROOT)
    return p.returncode == 0 and dest.exists() and dest.stat().st_size > 0


def model_is_complete(entry: dict, path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    expected_mb = (entry.get("gguf") or {}).get("approx_size_mb")
    return not expected_mb or path.stat().st_size >= expected_mb * 1024 * 1024 * 0.90


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=str(ROOT / "build/crispembed"))
    ap.add_argument("--gpu-backend", default="auto",
                    help="backend passed to crispembed (auto, metal, cpu, ...)")
    ap.add_argument("--mmap", action="store_true",
                    help="set UOCR_MMAP=1 for engines supporting no-copy GGUF loading")
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    ap.add_argument("--output", default="")
    ap.add_argument("--timeout", type=float, default=180)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--download-missing", action="store_true",
                    help="download manifest GGUFs from their pinned HF repo")
    ap.add_argument("--only", nargs="*", help="manifest names to run")
    args = ap.parse_args()
    if args.mmap:
        # Unlimited-OCR and future large VLM loaders consume this opt-in; it
        # is harmless for engines that do not inspect the variable.
        os.environ["UOCR_MMAP"] = "1"

    manifest = json.loads(MANIFEST.read_text())
    model_dir = Path(args.model_dir)
    rows = []
    for entry in manifest.get("models", []):
        name = entry.get("name", "")
        if args.only and name not in args.only:
            continue
        sample = entry.get("sample")
        gguf = artifact_filename(entry.get("gguf"))
        row = {"engine": name, "model": gguf, "sample": sample,
               "expected": entry.get("expected_text")}
        if not sample:
            row["status"] = "skipped-no-sample"
            rows.append(row)
            continue
        model = model_dir / gguf if gguf else None
        if model and not model_is_complete(entry, model) and args.download_missing:
            download_model(entry, model)
        image = ROOT / sample
        if not model or not model_is_complete(entry, model):
            row["status"] = "skipped-model-unavailable"
            rows.append(row)
            continue
        if not image.exists():
            row["status"] = "skipped-sample-unavailable"
            rows.append(row)
            continue

        detector = artifact_filename(entry.get("detector"))
        if detector:
            detector_path = model_dir / detector
            if not detector_path.exists():
                row["status"] = "skipped-model-unavailable"
                rows.append(row)
                continue
            command = [args.binary, "--ocr-pipeline", str(image),
                       "--ocr-engine", pipeline_engine(entry),
                       "--ocr-det", str(detector_path), "--ocr-rec", str(model)]
        else:
            command = [args.binary, "-m", str(model), "--ocr", str(image)]
        if args.gpu_backend:
            command[1:1] = ["--gpu-backend", args.gpu_backend]
        timings, outputs = [], []
        errors = []
        for _ in range(max(1, args.repeats)):
            meta, out, err = run(command, args.timeout)
            timings.append(meta["ms"])
            outputs.append(out.strip())
            errors.append(err[-1000:])
            if meta["timed_out"] or meta["returncode"] != 0:
                break
        chosen = outputs[-1] if outputs else ""
        failed = runtime_failed(meta, errors[-1] if errors else "")
        warm = timings[1:]
        warm_p95 = statistics.quantiles(warm, n=20, method="inclusive")[-1] if len(warm) >= 2 else None
        row.update({"status": "ok" if outputs and not failed else "error",
                    "cold_ms": timings[0] if timings else None,
                    "warm_median_ms": statistics.median(warm) if warm else None,
                    "warm_p95_ms": warm_p95,
                    "timings_ms": timings,
                    "runs": len(timings), "quality": quality(chosen, entry.get("expected_text")),
                    "output": chosen[:4000], "stderr_tail": errors[-1] if errors else ""})
        rows.append(row)
        q = row["quality"]
        qtext = q.get("status") if q.get("status") == "unscored" else f"{q.get('status')} cer={q.get('cer', 0):.3f}"
        print(f"{name:24} {row['status']:8} cold={row.get('cold_ms', 0):8.1f}ms "
              f"warm={row.get('warm_median_ms') or 0:8.1f}ms {qtext}", flush=True)

    result = {"platform": "local", "binary": args.binary, "model_dir": str(model_dir),
              "repeats": args.repeats, "rows": rows}
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(f"completed={sum(r.get('status') == 'ok' for r in rows)} "
          f"skipped={sum(r.get('status', '').startswith('skipped') for r in rows)} "
          f"errors={sum(r.get('status') == 'error' for r in rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
