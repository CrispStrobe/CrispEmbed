#!/usr/bin/env python3
"""Hosted-only bounded qualification. Existing heavy-cpu workflow entrypoint.

Requires a portable build containing lightonocr3-trace and a Python environment
from models/requirements-lightonocr3-reference.txt. Artifacts stay in --output.
Never uploads, publishes, changes a model alias, or modifies frozen receipts.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "tools"))
from lightonocr3_profile import source_directory, sha256
from compare_lightonocr3 import compare


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--model-dir", help="Exact verified local snapshot, or downloads the pinned checkpoint")
    p.add_argument("--formats", default="f32,f16", help="Bounded first lane; supply preconverted --model-format for quantized arms")
    p.add_argument("--model-format", action="append", default=[], metavar="FORMAT=GGUF")
    a = p.parse_args()
    if os.environ.get("GITHUB_ACTIONS") != "true" or os.environ.get("CRISPEMBED_BUILD_MODE") != "portable":
        p.error("run on hosted CI with CRISPEMBED_BUILD_MODE=portable")
    a.output = a.output.resolve()
    a.build = a.build.resolve()
    if a.output.exists() and any(a.output.iterdir()):
        p.error("output must be empty so previous evidence cannot be overwritten")
    a.output.mkdir(parents=True, exist_ok=True)
    cache = (a.build / "CMakeCache.txt").read_text()
    for name in ("GGML_NATIVE", "CRISPEMBED_NATIVE"):
        if f"{name}:BOOL=OFF" not in cache:
            p.error(f"{name} must be OFF in the actual CMakeCache")
    memory = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))) * 1024
    disk = shutil.disk_usage(a.output).free
    preflight = {"memory_available_bytes": memory, "disk_available_bytes": disk,
                 "load_average": os.getloadavg(), "build_mode": "portable"}
    (a.output / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")
    if memory < 12 * 2**30 or disk < 30 * 2**30:
        p.error("bounded original/native lane needs at least 12 GiB available RAM and 30 GiB disk")
    formats = a.formats.split(",")
    if not formats or any(f not in ("f32", "f16", "q8_0", "q4_k") for f in formats):
        p.error("unsupported precision")
    models = {}
    for item in a.model_format:
        precision, filename = item.split("=", 1)
        if precision not in formats or precision in models:
            p.error("model-format must uniquely name a selected precision")
        models[precision] = Path(filename).resolve()
    if any(f in ("q8_0", "q4_k") and f not in models for f in formats):
        p.error("quantized arms need explicit --model-format; no unmeasured quantization recipe is assumed")
    (a.output / "requirements-freeze.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
    (a.output / "build-mode.txt").write_text("portable\n")
    source = source_directory(a.model_dir or "lightonai/LightOnOCR-3-1B")
    env = dict(os.environ)
    for key in ("CRISPEMBED_MAX_PIXELS", "CRISPEMBED_LIGHTONOCR_DEBUG", "CRISPEMBED_LIGHTONOCR_NO_KV", "LOCR_DECODE_CACHE"):
        env.pop(key, None)
    env.update(CRISPEMBED_LIGHTONOCR3="1", CRISPEMBED_LIGHTONOCR_FORCE_CPU="1", OMP_NUM_THREADS="4")

    def run(command, log, extra=None):
        with log.open("w") as out:
            return subprocess.run(list(map(str, command)), cwd=ROOT, env=env | (extra or {}),
                                  stdout=out, stderr=subprocess.STDOUT, timeout=3600).returncode

    for precision in formats:
        if precision not in models:
            models[precision] = a.output / f"candidate-{precision}.gguf"
            rc = run([sys.executable, "models/convert-lightonocr-to-gguf.py", "--model", source,
                      "--profile", "lightonocr3-1b", "--dtype", precision, "--output", models[precision]],
                     a.output / f"convert-{precision}.log")
            if rc:
                raise RuntimeError(f"{precision} conversion failed; log retained")
    recipe = json.loads((ROOT / "tests/fixtures/lightonocr3/qualification.json").read_text())
    summary = {"scope": recipe["scope"], "build_mode": "portable", "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "model_sha256": {k: sha256(v) for k, v in models.items()}, "cases": [],
               "policy": "minimum per-position final-logit cosine; decoded and strict failures retained", "published": False}
    failed = False
    for case in recipe["cases"]:
        image = ROOT / case["image"]
        if sha256(image) != case["sha256"]:
            raise ValueError(f"Frozen image mismatch: {case['id']}")
        for mode in case["modes"]:
            directory = a.output / f"{case['id']}-{mode}"
            directory.mkdir()
            ref = directory / "reference"
            rc = run([sys.executable, "tools/dump_lightonocr3_reference.py", "--model-dir", source,
                      "--image", image, "--output", ref, "--mode", mode,
                      "--max-new-tokens", recipe["max_new_tokens"]], directory / "original.log")
            if rc:
                summary["cases"].append({"case": case["id"], "mode": mode, "error": "original reference failed"})
                failed = True
                continue
            meta = json.loads((ref / "reference.json").read_text())
            for precision, model in models.items():
                dirs = {arm: directory / f"{precision}-{arm}" for arm in ("forced", "autonomous", "plain")}
                statuses = {}
                for arm, output in dirs.items():
                    forced = ref / "forced-ids.txt" if arm == "forced" else "-"
                    cap = len(meta["token_ids"]) if arm == "forced" else recipe["max_new_tokens"]
                    statuses[arm] = run([a.build / "lightonocr3-trace", model, image, output, cap,
                                         forced, "plain" if arm == "plain" else "observe"],
                                        directory / f"{precision}-{arm}.log",
                                        {"CRISPEMBED_LIGHTONOCR3_GROUNDING": "1" if mode == "grounding" else "0"})
                report = compare(ref, dirs["forced"], dirs["autonomous"], dirs["plain"], precision)
                report["native_returncodes"] = statuses
                report["owner_bar_pass"] &= not any(statuses.values())
                (directory / f"{precision}-comparison.json").write_text(json.dumps(report, indent=2) + "\n")
                summary["cases"].append({"case": case["id"], "mode": mode, "precision": precision,
                                         "pass": report["owner_bar_pass"], "minimum_final_logit_cosine": report["minimum_final_logit_cosine"],
                                         "decoded": report["decoded"], "strict_failures": report["strict_failures"]})
                failed |= not report["owner_bar_pass"]
            (a.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    summary["pass"] = not failed
    (a.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
