#!/usr/bin/env python3
"""Compare two inpaint_live_eval.py output directories (e.g. Q8_0 vs F32).

usage: inpaint_live_compare.py REFERENCE_DIR VARIANT_DIR [OUT.json]

For every photo and mask job present in both, scores the variant's masked
region against the reference run's output (PSNR, MAE, max abs, RGB cosine) and
repeats each run's own ground-truth scores from its report.json.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def agreement(a, b, mask):
    sel = mask > 0
    x, y = a[sel].astype(np.float64).ravel(), b[sel].astype(np.float64).ravel()
    d = x - y
    mse = float(np.mean(d * d))
    return {"psnr_db": round(10 * math.log10(255 ** 2 / mse), 3) if mse else None,
            "mae": round(float(np.mean(np.abs(d))), 3), "max_abs": int(np.max(np.abs(d))) if d.size else 0,
            "rgb_cosine": round(float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y))), 6)}


def main(argv):
    if len(argv) not in (3, 4):
        raise SystemExit(__doc__)
    ref, var = Path(argv[1]), Path(argv[2])
    rr, vr = (json.loads((d / "report.json").read_text()) for d in (ref, var))
    rows = []
    for vcase in vr["cases"]:
        rcase = next((c for c in rr["cases"] if c["photo"] == vcase["photo"]), None)
        if not rcase:
            continue
        stem = Path(vcase["photo"]).stem
        mask = np.asarray(Image.open(var / f"{stem}.mask.png").convert("L"))
        for job, entry in vcase["runs"].items():
            a, b = sorted(var.glob(f"{stem}.*.{job}.png")), sorted(ref.glob(f"{stem}.*.{job}.png"))
            row = {"photo": vcase["photo"], "job": job, "variant": entry.get("inpainted"),
                   "reference": rcase["runs"].get(job, {}).get("inpainted"), "naive": vcase["naive_fill"],
                   "variant_seconds": entry.get("seconds"), "variant_peak_rss_kib": entry.get("peak_rss_kib"),
                   "outside_exact": entry.get("outside_exact")}
            if len(a) == 1 and len(b) == 1:
                row["vs_reference"] = agreement(np.asarray(Image.open(a[0]).convert("RGB")),
                                                np.asarray(Image.open(b[0]).convert("RGB")), mask)
            rows.append(row)
            print(json.dumps(row))
    if len(argv) == 4:
        Path(argv[3]).write_text(json.dumps({"reference": str(ref), "variant": str(var), "rows": rows}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
