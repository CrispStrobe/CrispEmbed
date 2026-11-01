#!/usr/bin/env python3
"""Live inpainting evaluation with ground truth: burned-in subtitles on real photos.

    python tools/inpaint_live_eval.py --binary build/crispembed --backend migan256|migan512|lama [--model-sha256 SHA] \
        --model MODEL.gguf --photos a.png b.jpg ... --output-dir OUT [--text-model dbnet.gguf]

For each photo (RGB, any size) this renders a two-line subtitle (white text, black
outline) into the bottom band, writes the exact text mask, runs the production CLI
and scores the result against the clean original:
  * masked region: PSNR and mean absolute error against the clean pixels (the
    inpainter never saw them), and the same scores for a naive baseline that fills
    the mask with the band's mean colour, so a model must beat a trivial fill;
  * outside the mask: every pixel must equal the subtitled input exactly.
With --text-model the mask also comes from the DBNet text detector
(--mask-from-text) and that run is scored the same way. Writes OUT/report.json and
the image triplets for inspection. Evaluation only; no quality gate is implied.
"""
import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

LINES = ["Das ist ein Untertitel, der entfernt", "werden soll – Testzeile 42"]


def font(size):
    for name in ("DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/System/Library/Fonts/Supplemental/Arial Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def subtitle(clean):
    w, h = clean.size
    size = max(14, h // 18)
    f = font(size)
    img, mask = clean.copy(), Image.new("L", clean.size, 0)
    d, m = ImageDraw.Draw(img), ImageDraw.Draw(mask)
    stroke = max(2, size // 10)
    y = h - int(size * 2.9)
    for line in LINES:
        tw = d.textlength(line, font=f)
        x = int((w - tw) / 2)
        d.text((x, y), line, font=f, fill=(255, 255, 255), stroke_width=stroke, stroke_fill=(0, 0, 0))
        m.text((x, y), line, font=f, fill=255, stroke_width=stroke + 2, stroke_fill=255)
        y += int(size * 1.3)
    return img, mask


def scores(result, clean, mask):
    sel = mask > 0
    diff = result[sel].astype(np.float64) - clean[sel].astype(np.float64)
    mse = float(np.mean(diff ** 2))
    return {"masked_pixels": int(sel.sum()), "psnr_db": round(10 * math.log10(255 ** 2 / mse), 3) if mse else math.inf,
            "mae": round(float(np.mean(np.abs(diff))), 3)}


def run(cmd, env):
    started = time.monotonic()
    proc = subprocess.run([str(c) for c in cmd], env=env, capture_output=True, text=True, timeout=3600)
    return proc.returncode, proc.stderr[-1500:], round(time.monotonic() - started, 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", required=True, type=Path)
    ap.add_argument("--backend", required=True, choices=("migan", "migan256", "migan512", "lama"),
                    help="CLI --inpaint-engine (migan means migan256)")
    ap.add_argument("--model-sha256", help="CLI --inpaint-model-sha256 (required by lama/migan512)")
    ap.add_argument("--checkpoint-sha256", help="CLI --inpaint-checkpoint-sha256 (migan512)")
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--photos", required=True, nargs="+", type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--text-model", type=Path)
    ap.add_argument("--threads", type=int, help="CLI --threads (default: the CLI's own default)")
    args = ap.parse_args()
    import os
    env = dict(os.environ, CRISPEMBED_MIGAN="1", CRISPEMBED_LAMA="1")
    out = args.output_dir
    engine = "migan256" if args.backend == "migan" else args.backend
    pins = [*(["--inpaint-model-sha256", args.model_sha256] if args.model_sha256 else []),
            *(["--inpaint-checkpoint-sha256", args.checkpoint_sha256] if args.checkpoint_sha256 else [])]
    out.mkdir(parents=True, exist_ok=False)
    report = {"backend": args.backend, "model": args.model.name, "threads": args.threads, "cases": []}
    for photo in args.photos:
        clean = Image.open(photo).convert("RGB")
        stem = photo.stem
        subbed, mask = subtitle(clean)
        clean.save(out / f"{stem}.clean.png")
        subbed.save(out / f"{stem}.input.png")
        mask.save(out / f"{stem}.mask.png")
        c, s, mk = (np.asarray(x) for x in (clean, subbed, mask))
        band = mk > 0
        naive = s.copy()
        naive[band] = s[~band][-(s.shape[1] * 8):].reshape(-1, 3).mean(axis=0).astype(np.uint8)
        case = {"photo": photo.name, "size": list(clean.size), "naive_fill": scores(naive, c, mk), "runs": {}}
        jobs = {"exact-mask": ["--mask", out / f"{stem}.mask.png"]}
        if args.text_model:
            jobs["dbnet-mask"] = ["--mask-from-text", "--mask-text-model", args.text_model, "--mask-dilate", "6"]
        for name, mask_args in jobs.items():
            dest = out / f"{stem}.{args.backend}.{name}.png"
            rc, err, secs = run([args.binary, "--inpaint-model", args.model, "--inpaint-engine", engine, *pins,
                                 "--inpaint", out / f"{stem}.input.png", *mask_args, "--inpaint-output", dest,
                                 *(["--threads", args.threads] if args.threads else [])], env)
            entry = {"returncode": rc, "seconds": secs}
            if rc == 0 and dest.exists():
                r = np.asarray(Image.open(dest).convert("RGB"))
                entry["shape_ok"] = r.shape == s.shape
                if entry["shape_ok"]:
                    entry["inpainted"] = scores(r, c, mk)
                    entry["outside_exact"] = bool(np.array_equal(r[~band], s[~band])) if name == "exact-mask" else None
            else:
                entry["stderr_tail"] = err
            case["runs"][name] = entry
        report["cases"].append(case)
        print(json.dumps(case), flush=True)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    ok = all(r["returncode"] == 0 and r.get("shape_ok") and r.get("outside_exact") is not False
             for case in report["cases"] for r in case["runs"].values())
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
