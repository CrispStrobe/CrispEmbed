#!/usr/bin/env python3
"""Cross-run precision table for the latent comparator (Moebius / PixelHacker).

usage: latent_precision_table.py OUT.json F32_DIR LABEL=DIR [LABEL=DIR ...]

Each DIR is a test-latent-inpaint output (report.json + CASE/generated.rgb).
Per case it reports, unchanged from the comparator, the vae.decoded float
metrics against the ORIGINAL capture and the frozen-panel stage counts; and,
computed here, generated-RGB agreement against native F32 (F32_DIR):
full-image RGB cosine, PSNR and max/mean absolute difference. The owner's
precision gate (> 0.94) is evaluated on decoded cosine vs original and on
generated-RGB cosine vs native F32. No frozen budget is changed.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np


def rgb(path):
    return np.fromfile(path, dtype=np.uint8).astype(np.float64)


def agreement(a, b):
    d = a - b
    mse = float(np.mean(d * d))
    return {"rgb_cosine": float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))),
            "psnr_db": 10 * math.log10(255 ** 2 / mse) if mse else None,
            "max_abs": int(np.max(np.abs(d))), "mean_abs": float(np.mean(np.abs(d)))}


def main(argv):
    if len(argv) < 4:
        raise SystemExit(__doc__)
    out, reference = Path(argv[1]), Path(argv[2])
    rows = []
    for spec in argv[3:]:
        label, _, directory = spec.partition("=")
        directory = Path(directory)
        report = json.loads((directory / "report.json").read_text())
        for case in report.get("cases", []):
            stages = case["stages"]
            decoded = next(s for s in stages if s["name"] == "vae.decoded")["metrics"]
            row = {"arm": label, "case": case["id"],
                   "frozen_stages_failed": sum(not s["metrics"]["pass"] for s in stages), "stages": len(stages),
                   "earliest_source_divergence": case.get("earliest_source_divergence"),
                   "decoded_vs_original": {k: decoded[k] for k in ("cosine", "max_abs", "mean_abs", "relative_l2")},
                   "generated_rgb_vs_original": {k: case["final_generated_rgb"].get(k) for k in ("max_abs", "mean_abs", "pass")},
                   "composited_rgb_vs_original": {k: case["final_composited_rgb"].get(k) for k in ("max_abs", "mean_abs", "pass")}}
            mine, base = directory / case["id"] / "generated.rgb", reference / case["id"] / "generated.rgb"
            if mine.exists() and base.exists():
                row["generated_rgb_vs_native_f32"] = agreement(rgb(mine), rgb(base))
            cosines = [row["decoded_vs_original"]["cosine"]]
            if "generated_rgb_vs_native_f32" in row:
                cosines.append(row["generated_rgb_vs_native_f32"]["rgb_cosine"])
            row["owner_cosine_gate_0_94"] = all(c > 0.94 for c in cosines)
            rows.append(row)
            f = row.get("generated_rgb_vs_native_f32", {})
            print(f"{label:12s} {case['id']:12s} dec_cos_orig={row['decoded_vs_original']['cosine']:.6f} "
                  f"rgb_cos_f32={f.get('rgb_cosine', float('nan')):.6f} psnr_f32={f.get('psnr_db')} "
                  f"frozen_failed={row['frozen_stages_failed']}/{row['stages']} gate={row['owner_cosine_gate_0_94']}")
    out.write_text(json.dumps({"schema": "crispembed-latent-precision-table-v1", "reference_native_f32": str(reference),
                               "owner_cosine_threshold": 0.94, "rows": rows}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
