#!/usr/bin/env python3
"""Run the MI-GAN device/precision A/B (test-migan-gpu-ab) and write receipts.

Every candidate (device x weight container) is compared with the reviewed
CPU/F32 graph on the eight frozen original fixtures (stage-level, image gates
vs CPU/F32 and vs the original composites) and on synthetic 1920x1080
subtitle frames (tile path, warm timing). Stdlib only; inputs are raw bytes
converted from the reviewed original review artifact.

  migan_device_ab.py --binary build/test-migan-gpu-ab --f32 F32.gguf \
      --fixtures DIR --candidate gpu:f32:F32.gguf --candidate cpu:q8_0:Q8.gguf ... \
      --output-dir tests/results/migan/metal-m1/RUN
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

CASES = ["center", "checkerboard", "edge_touch", "empty", "full", "one_pixel", "subtitle_bar", "thin_strip"]


def records(text, prefix):
    out = []
    for line in text.splitlines():
        if line.startswith(prefix + " "):
            out.append(json.loads(line[len(prefix) + 1:]))
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(command, env):
    done = subprocess.run(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=3600)
    return done.returncode, done.stdout, done.stderr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", required=True, type=Path)
    ap.add_argument("--f32", required=True, type=Path)
    ap.add_argument("--fixtures", required=True, type=Path)
    ap.add_argument("--candidate", action="append", required=True, help="device:label:model.gguf[:ENV=V,ENV=V]")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--frame-iters", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--frames", default="subtitle2,bar", help="comma list of synthetic 1920x1080 kinds")
    ap.add_argument("--skip-frames", action="store_true")
    ap.add_argument("--cases", default=",".join(CASES), help="comma list of frozen fixture names")
    args = ap.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    base = os.environ.copy()
    base["CRISPEMBED_MIGAN"] = "1"
    base["CRISPEMBED_MIGAN_QUANT"] = "1"
    report = {"schema": "crispembed-migan-device-ab-v1", "host": platform.platform(), "machine": platform.machine(),
              "binary_sha256": sha256(args.binary), "reference": {"path_name": args.f32.name,
                                                                   "sha256": sha256(args.f32), "device": "cpu",
                                                                   "precision": "f32"},
              "threads": args.threads, "candidates": []}
    for spec in args.candidate:
        parts = spec.split(":")
        device, label, model = parts[0], parts[1], Path(parts[2])
        extra = dict(kv.split("=", 1) for kv in parts[3].split(",")) if len(parts) > 3 and parts[3] else {}
        env = dict(base, **extra)
        entry = {"device": device, "label": label, "model_name": model.name, "model_sha256": sha256(model),
                 "model_bytes": model.stat().st_size, "env": extra, "cases": [], "frames": []}
        for case in args.cases.split(","):
            f = args.fixtures
            cmd = [str(args.binary), "stages", str(args.f32), str(model), device, str(f / f"{case}-input.raw"),
                   str(f / f"{case}-fill-mask.raw"),
                   str(f / f"{case}-composited.raw") if (f / f"{case}-composited.raw").exists() else "-",
                   str(args.iters),
                   str(args.threads)]
            code, stdout, stderr = run(cmd, env)
            log = out / f"{label}-{device}-{case}.log"
            log.write_text(stdout + "\n--- stderr ---\n" + stderr, encoding="utf-8")
            summary = records(stdout, "MIGAN_AB_SUMMARY")
            images = records(stdout, "MIGAN_AB_IMAGE")
            entry["cases"].append({"case": case, "exit_code": code, "summary": summary[0] if summary else None,
                                   "images": images, "log_sha256": sha256(log)})
            s = summary[0] if summary else {}
            print(f"{label}/{device}/{case}: exit={code} strict={s.get('strict_passed')} "
                  f"out_max={s.get('output_max_abs')} cos={s.get('output_cos_global')} "
                  f"ms={s.get('warm_ms_candidate_median')} cpu_ms={s.get('warm_ms_cpu_f32_median')}", flush=True)
        if not args.skip_frames:
            for kind in args.frames.split(","):
                raw = out / f"{label}-{device}-frame-{kind}.rgb8"
                cmd = [str(args.binary), "frame", str(args.f32), str(model), device, "1920", "1080",
                       str(args.frame_iters), str(args.threads), kind, str(raw)]
                code, stdout, stderr = run(cmd, env)
                log = out / f"{label}-{device}-frame-{kind}.log"
                log.write_text(stdout + "\n--- stderr ---\n" + stderr, encoding="utf-8")
                frame = records(stdout, "MIGAN_AB_FRAME")
                images = records(stdout, "MIGAN_AB_IMAGE")
                entry["frames"].append({"kind": kind, "exit_code": code, "frame": frame[0] if frame else None,
                                        "image": images[0] if images else None,
                                        "output_rgb8_sha256": sha256(raw) if raw.exists() else None})
                if raw.exists():
                    raw.unlink()  # 6 MB each; the hash is retained.
                print(f"{label}/{device}/frame-{kind}: exit={code} {frame[0] if frame else stderr[-300:]}",
                      flush=True)
        report["candidates"].append(entry)
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"written": str(out / "report.json")}))


if __name__ == "__main__":
    main()
