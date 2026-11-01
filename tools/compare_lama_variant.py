#!/usr/bin/env python3
"""Run and compare a Big-LaMa precision variant (F16/Q8_0/Q4_K, gated runtime modes).

INTERNAL RESEARCH ONLY (feat/issue71-lama-research). The strict F32 protocol in
compare_lama_reference.py binds the native run to the exact F32 model the
original capture verified, so a derived GGUF cannot pass through it. This tool
runs the same native capture driver on a variant and reports, per fixture and
stage, its error against the original-module capture and against the native
F32 capture of the same fixtures:

    python tools/compare_lama_variant.py --binary build/test-lama-native \
        --model big-lama-q8_0.gguf --original-dir lama-original \
        --f32-native-dir lama-native --output-dir lama-q8_0 [--env K=V ...]

The owner's acceptance bar is a final generator-output cosine > 0.94 against
both references (OWNER_MIN_OUTPUT_COSINE). The strict F32 budgets are reported
unchanged for information; no variant is expected to meet them.
"""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time

from compare_lama_reference import (MAX_ABSOLUTE_ERROR, MAX_RELATIVE_L2_ERROR, MIN_COSINE, local_file, metrics,
                                    read_manifest, read_stage, validate_original_report)
from lama_reference_contract import CASES, expected_stage_names, fixture, fixture_identity, sha256

OWNER_MIN_OUTPUT_COSINE = 0.94


def capture(binary, model, original_dir, output_dir, threads, env):
    output_dir.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ, CRISPEMBED_LAMA="1", **env)
    timings = {}
    for name in CASES:
        identity = fixture_identity(name)
        command = [str(binary), "--model", str(model), "--input-dir", str(original_dir / name),
                   "--output-dir", str(output_dir / name), "--width", str(identity["width"]),
                   "--height", str(identity["height"]), "--threads", str(threads)]
        started = time.monotonic()
        done = subprocess.run(command, env=environment, text=True, capture_output=True, timeout=1200)
        timings[name] = round(time.monotonic() - started, 3)
        (output_dir / (name + ".log")).write_text(done.stdout + done.stderr, encoding="utf-8")
        if done.returncode:
            raise RuntimeError(f"native LaMa variant failed for {name}; see {name}.log")
    return timings


def psnr(a, b, select):
    pairs = [(x, y) for i, (x, y) in enumerate(zip(a, b)) if select[i // 3]]
    if not pairs:
        return None
    mse = math.fsum((x - y) ** 2 for x, y in pairs) / len(pairs)
    return math.inf if mse == 0 else 10 * math.log10(255 ** 2 / mse)


def image_metrics(reference, variant, mask):
    errors = [abs(a - b) for a, b in zip(reference, variant)]
    filled = [m != 0 for m in mask]
    masked_psnr = psnr(reference, variant, filled)
    return {"max_abs": max(errors), "mean_abs": math.fsum(errors) / len(errors),
            "different_channels": sum(e != 0 for e in errors),
            "masked_psnr_db": None if masked_psnr is None else (masked_psnr if math.isfinite(masked_psnr) else "inf")}


def compare_case(name, original_dir, f32_dir, variant_dir):
    _, original_rows = read_manifest(original_dir, "crispembed.lama.reference.v1")
    _, f32_rows = read_manifest(f32_dir, "crispembed.lama.native.v1")
    variant, variant_rows = read_manifest(variant_dir, "crispembed.lama.native.v1")
    stages = []
    for stage in expected_stage_names():
        values = read_stage(variant_dir, variant_rows[stage])
        row = {"stage": stage, "shape": variant_rows[stage]["shape"],
               "vs_original": metrics(read_stage(original_dir, original_rows[stage]), values),
               "vs_f32_native": metrics(read_stage(f32_dir, f32_rows[stage]), values)}
        stages.append(row)
    out = next(row for row in stages if row["stage"] == "generator.output")
    width, height, rgb, mask = fixture(name)
    variant_rgb = local_file(variant_dir, "output.rgb").read_bytes()
    known_exact = all(variant_rgb[3 * i:3 * i + 3] == rgb[3 * i:3 * i + 3] for i, m in enumerate(mask) if not m)
    first_strict = next((row["stage"] for row in stages if not row["vs_original"]["passed"]), None)
    worst = min(stages, key=lambda row: row["vs_original"]["cosine"])
    summary = {
        "id": name, "fill_pixels": mask.count(255),
        "output_cosine_vs_original": out["vs_original"]["cosine"],
        "output_cosine_vs_f32_native": out["vs_f32_native"]["cosine"],
        "output_max_abs_vs_original": out["vs_original"]["max_abs"],
        "output_max_abs_vs_f32_native": out["vs_f32_native"]["max_abs"],
        "min_stage_cosine_vs_original": worst["vs_original"]["cosine"], "min_stage": worst["stage"],
        "max_stage_relative_l2_vs_original": max(row["vs_original"]["relative_l2"] or 0 for row in stages),
        "first_strict_budget_exceedance": first_strict,
        "image_vs_original": image_metrics(local_file(original_dir, "output.rgb").read_bytes(), variant_rgb, mask),
        "image_vs_f32_native": image_metrics(local_file(f32_dir, "output.rgb").read_bytes(), variant_rgb, mask),
        "known_pixels_exact": known_exact, "native_controls": variant.get("controls"),
        "graph_buffer_bytes": variant.get("graph_buffer_bytes"),
    }
    # The empty fixture skips nothing in the raw graph, but its composed image
    # equals the input; output cosine is still the raw generator output.
    summary["owner_bar_passed"] = (known_exact and summary["output_cosine_vs_original"] > OWNER_MIN_OUTPUT_COSINE
                                   and summary["output_cosine_vs_f32_native"] > OWNER_MIN_OUTPUT_COSINE)
    return summary, stages


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--f32-native-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default=None)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--env", action="append", default=[], help="K=V runtime gate for the variant run")
    args = parser.parse_args()
    env = dict(item.split("=", 1) for item in args.env)
    original = validate_original_report(args.original_dir)
    timings = capture(args.binary.resolve(), args.model.resolve(), args.original_dir.resolve(),
                      args.output_dir.resolve(), args.threads, env)
    cases, all_stages = [], {}
    for name in CASES:
        summary, stages = compare_case(name, args.original_dir / name, args.f32_native_dir / name,
                                       args.output_dir / name)
        summary["seconds"] = timings[name]
        cases.append(summary)
        all_stages[name] = stages
    report = {"schema": "crispembed-lama-variant-comparison-v1", "label": args.label or args.model.name,
              "model": args.model.name, "model_sha256": sha256(args.model), "model_bytes": args.model.stat().st_size,
              "env": env, "original_model_sha256": original["model_sha256"],
              "owner_min_output_cosine": OWNER_MIN_OUTPUT_COSINE,
              "strict_f32_budgets_for_information": {"max_abs": MAX_ABSOLUTE_ERROR,
                                                     "max_relative_l2": MAX_RELATIVE_L2_ERROR,
                                                     "min_cosine": MIN_COSINE},
              "owner_bar_passed": all(case["owner_bar_passed"] for case in cases), "cases": cases,
              "internal_research_only": True, "published": False}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "stages.json").write_text(json.dumps(all_stages) + "\n", encoding="utf-8")
    print(json.dumps({"label": report["label"], "owner_bar_passed": report["owner_bar_passed"],
                      "cases": [{k: c[k] for k in ("id", "output_cosine_vs_original", "output_cosine_vs_f32_native",
                                                   "min_stage_cosine_vs_original")} for c in cases]}, indent=2))
    if not report["owner_bar_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
