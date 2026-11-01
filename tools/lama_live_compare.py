#!/usr/bin/env python3
"""Score Big-LaMa live-eval variants against the clean photo, native F32 and the original.

INTERNAL RESEARCH ONLY. Inputs are tools/inpaint_live_eval.py output directories
(same photos, same rendered subtitles) and a tools/lama_original_photos.py
directory. Within the fill mask (>=128, the CLI's threshold) each variant's
exact-mask output is scored by PSNR, MAE, cosine and Pearson correlation of the
RGB values against (a) the clean photo, (b) native F32 and (c) the original
PyTorch generator; outside the mask every pixel must equal the subtitled input.

    python tools/lama_live_compare.py --original-dir live-original \
        --f32-dir live-f32 --variant f16=live-f16 --variant q8_0=live-q8_0 ... --output report.json
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def load(path):
    return np.asarray(Image.open(path).convert("RGB")).astype(np.float64)


def score(result, reference, select):
    a, b = result[select].ravel(), reference[select].ravel()
    mse = float(np.mean((a - b) ** 2))
    cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))) if a.any() and b.any() else None
    ac, bc = a - a.mean(), b - b.mean()
    denom = np.linalg.norm(ac) * np.linalg.norm(bc)
    return {"psnr_db": round(10 * math.log10(255 ** 2 / mse), 3) if mse else "inf",
            "mae": round(float(np.mean(np.abs(a - b))), 4), "cosine": cosine,
            "pearson": float(ac @ bc / denom) if denom else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--original-dir", type=Path)
    parser.add_argument("--f32-dir", type=Path, required=True)
    parser.add_argument("--variant", action="append", default=[], help="LABEL=DIR")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    variants = [("f32", args.f32_dir)] + [tuple(item.split("=", 1)) for item in args.variant]
    variants = [(label, Path(directory)) for label, directory in variants]
    report = {"schema": "crispembed-lama-live-comparison-v1", "internal_research_only": True, "photos": []}
    for clean_path in sorted(args.f32_dir.glob("*.clean.png")):
        stem = clean_path.name[:-len(".clean.png")]
        clean, subbed = load(clean_path), load(args.f32_dir / f"{stem}.input.png")
        gray = np.asarray(Image.open(args.f32_dir / f"{stem}.mask.png").convert("L"))
        fill = gray >= 128
        f32 = load(args.f32_dir / f"{stem}.lama.exact-mask.png")
        original = None
        if args.original_dir and (args.original_dir / f"{stem}.original.png").exists():
            original = load(args.original_dir / f"{stem}.original.png")
        row = {"photo": stem, "size": [int(clean.shape[1]), int(clean.shape[0])], "fill_pixels": int(fill.sum()),
               "original_vs_clean": score(original, clean, fill) if original is not None else None, "variants": {}}
        for label, directory in variants:
            path = directory / f"{stem}.lama.exact-mask.png"
            if not path.exists():
                row["variants"][label] = {"missing": True}
                continue
            result = load(path)
            row["variants"][label] = {
                "vs_clean": score(result, clean, fill), "vs_f32_native": score(result, f32, fill),
                "vs_original": score(result, original, fill) if original is not None else None,
                "outside_mask_exact": bool(np.array_equal(result[~fill], subbed[~fill])),
            }
        report["photos"].append(row)
        print(json.dumps({"photo": stem, **{k: (v.get("vs_clean", {}).get("psnr_db"),
                                                (v.get("vs_original") or {}).get("psnr_db"))
                                            for k, v in row["variants"].items()}}), flush=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
