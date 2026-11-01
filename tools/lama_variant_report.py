#!/usr/bin/env python3
"""Replay a Big-LaMa precision variant through test-lama-diff and score it.

INTERNAL RESEARCH ONLY. The strict F32 protocol (tools/ci-heavy/lama_inpaint.py)
replays the converter's F32 file against the original capture. This runs the
same driver on a lama-v2 F16/Q8_0/Q4_K file (or a gated runtime mode) and
reports, per fixture: every stage against the original (the driver's own
LAMA_STAGE metrics) and against the native F32 replay's stage dumps; the
prediction (generator output) cosine; and masked-RGB PSNR/MAE against the
original composite and against native F32.

    python tools/lama_variant_report.py --binary test-lama-diff --model big-lama-q8_0.gguf \
        --reference-dir acceptance/original --f32-native-dir acceptance --output-dir q8_0-storage \
        [--label q8_0-storage] [--env CRISPEMBED_LAMA_QUANT_MATMUL=1]

Owner bar: prediction cosine > 0.94 against both the original and native F32.
The strict F32 thresholds still apply to the driver's own pass/fail line and are
reported unchanged; a variant is not expected to meet them.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np

OWNER_MIN_OUTPUT_COSINE = 0.94


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def vector_metrics(reference, value):
    reference, value = reference.astype(np.float64), value.astype(np.float64)
    error = value - reference
    denominator = np.linalg.norm(reference) * np.linalg.norm(value)
    return {"max_abs": float(np.abs(error).max()), "relative_l2": float(np.linalg.norm(error) / max(np.linalg.norm(reference), 1e-12)),
            "cosine": float(reference @ value / denominator) if denominator else (1.0 if not error.any() else 0.0)}


def rgb_metrics(reference, value, mask):
    select = np.repeat(mask != 0, 3)
    if not select.any():
        return None
    error = value[select].astype(np.float64) - reference[select].astype(np.float64)
    mse = float(np.mean(error ** 2))
    return {"masked_psnr_db": round(10 * math.log10(255 ** 2 / mse), 3) if mse else "inf",
            "masked_mae": round(float(np.abs(error).mean()), 4), "masked_max_abs": int(np.abs(error).max()),
            "known_rgb_errors": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-sha256")
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--f32-native-dir", type=Path, required=True, help="directory holding native-<case>/")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label")
    parser.add_argument("--env", action="append", default=[])
    args = parser.parse_args()
    env = dict(item.split("=", 1) for item in args.env)
    model_sha = args.model_sha256 or sha256(args.model)
    original = json.loads((args.reference_dir / "reference.json").read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ, CRISPEMBED_LAMA="1", **env)
    cases = []
    for row in original["cases"]:
        name, width, height = row["id"], row["width"], row["height"]
        target = args.output_dir / ("native-" + name)
        started = time.monotonic()
        done = subprocess.run([str(args.binary), "--model", str(args.model), "--model-sha256", model_sha,
                               "--reference", str(args.reference_dir / name / "reference.gguf"),
                               "--reference-sha256", row["reference_sha256"], "--output", str(target)],
                              env=environment, capture_output=True, text=True, timeout=3600)
        seconds = round(time.monotonic() - started, 3)
        (args.output_dir / (name + ".log")).write_text(done.stdout + done.stderr, encoding="utf-8")
        stages, final = [], None
        for line in done.stdout.splitlines():
            if line.startswith("LAMA_STAGE "):
                stages.append(json.loads(line[len("LAMA_STAGE "):]))
            elif line.startswith("LAMA_FINAL "):
                final = json.loads(line[len("LAMA_FINAL "):])
        case = {"id": name, "returncode": done.returncode, "seconds": seconds, "driver_final": final,
                "stderr_tail": done.stderr[-600:] if final is None else None}
        if final is None or not stages:
            case["owner_bar_passed"] = False
            cases.append(case)
            print(json.dumps(case), flush=True)
            continue
        f32_dir = args.f32_native_dir / ("native-" + name)
        for stage in stages:
            ours, theirs = target / (stage["stage"] + ".f32"), f32_dir / (stage["stage"] + ".f32")
            if ours.exists() and theirs.exists():
                stage["vs_f32_native"] = vector_metrics(np.fromfile(theirs, "<f4"), np.fromfile(ours, "<f4"))
        prediction = next(stage for stage in stages if stage["stage"] == "prediction")
        mask = np.fromfile(args.reference_dir / name / "fill-mask.u8", np.uint8)
        native_rgb = np.fromfile(target / "native.rgb8", np.uint8)
        case.update({
            "prediction_cosine_vs_original": prediction["cosine"],
            "prediction_max_abs_vs_original": prediction["max_abs"],
            "prediction_cosine_vs_f32_native": prediction.get("vs_f32_native", {}).get("cosine"),
            "prediction_max_abs_vs_f32_native": prediction.get("vs_f32_native", {}).get("max_abs"),
            "min_stage_cosine_vs_original": min(stage["cosine"] for stage in stages),
            "min_stage": min(stages, key=lambda stage: stage["cosine"])["stage"],
            "rgb_vs_original": rgb_metrics(np.fromfile(args.reference_dir / name / "composited.rgb8", np.uint8),
                                           native_rgb, mask),
            "rgb_vs_f32_native": rgb_metrics(np.fromfile(f32_dir / "native.rgb8", np.uint8), native_rgb, mask)
            if (f32_dir / "native.rgb8").exists() else None,
            "known_rgb_errors": final["known_errors"],
        })
        cosines = [case["prediction_cosine_vs_original"], case["prediction_cosine_vs_f32_native"]]
        case["owner_bar_passed"] = (final["known_errors"] == 0 and None not in cosines
                                    and min(cosines) > OWNER_MIN_OUTPUT_COSINE)
        (args.output_dir / (name + ".stages.json")).write_text(json.dumps(stages) + "\n", encoding="utf-8")
        cases.append(case)
        print(json.dumps({k: case[k] for k in ("id", "prediction_cosine_vs_original", "prediction_cosine_vs_f32_native",
                                               "rgb_vs_original", "seconds")}), flush=True)
    report = {"schema": "crispembed-lama-variant-v2", "label": args.label or args.model.name,
              "model": args.model.name, "model_sha256": model_sha, "model_bytes": args.model.stat().st_size,
              "env": env, "owner_min_output_cosine": OWNER_MIN_OUTPUT_COSINE,
              "owner_bar_passed": bool(cases) and all(case["owner_bar_passed"] for case in cases),
              "cases": cases, "internal_research_only": True, "published": False}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"label": report["label"], "owner_bar_passed": report["owner_bar_passed"]}))
    return 0 if report["owner_bar_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
