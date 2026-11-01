#!/usr/bin/env python3
"""Live inpainting evaluation with ground truth: burned-in subtitles on real photos.

    python tools/inpaint_live_eval.py --binary build/crispembed \
        --engine migan256|migan512|lama|moebius|pixelhacker --model MODEL.gguf \
        [--model-sha256 H] [--checkpoint-sha256 H]          # migan512 / lama digests
        [--vae VAE.gguf --vae-sha256 H] [--tiles]           # moebius / pixelhacker
        [--crop 512] [--env CRISPEMBED_INPAINT_QUANT=1 ...] [--reference-run F32_OUT_DIR]
        --photos a.png b.jpg ... --output-dir OUT [--text-model dbnet.gguf]

(--backend migan|lama is the older spelling of --engine migan256|lama.)

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
--crop N centre-crops each photo to NxN first (the strict 512 profiles of
migan512/moebius/pixelhacker); --tiles instead runs latent models on the full
photo with independent 512 tiles. Each run records wall time and, where
/usr/bin/time exists, process peak RSS. --reference-run scores the masked
region of each output against the same-named output of another run (e.g. a
precision variant against F32): PSNR, MAE and RGB cosine.
"""
import argparse
import json
import math
import os
import platform
import re
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


def agreement(result, other, mask):
    sel = mask > 0
    a, b = result[sel].astype(np.float64), other[sel].astype(np.float64)
    diff = a - b
    mse = float(np.mean(diff ** 2))
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return {"psnr_db": round(10 * math.log10(255 ** 2 / mse), 3) if mse else math.inf,
            "mae": round(float(np.mean(np.abs(diff))), 3), "max_abs": int(np.max(np.abs(diff))) if diff.size else 0,
            "rgb_cosine": round(float(a @ b / (na * nb)), 6) if na and nb else None}


def timed(cmd):
    if not Path("/usr/bin/time").exists():
        return cmd, None
    return (["/usr/bin/time", "-l" if platform.system() == "Darwin" else "-v", *cmd],
            "darwin" if platform.system() == "Darwin" else "linux")


def run(cmd, env):
    cmd, kind = timed([str(c) for c in cmd])
    started = time.monotonic()
    proc = subprocess.run([str(c) for c in cmd], env=env, capture_output=True, text=True, timeout=7200)
    seconds = round(time.monotonic() - started, 3)
    rss = None
    if kind == "linux":
        found = re.search(r"Maximum resident set size \(kbytes\): (\d+)", proc.stderr)
        rss = int(found.group(1)) if found else None
    elif kind == "darwin":
        found = re.search(r"(\d+)\s+maximum resident set size", proc.stderr)
        rss = int(found.group(1)) // 1024 if found else None
    return proc.returncode, proc.stderr[-1500:], seconds, rss


def engine_args(args):
    out = ["--inpaint-engine", args.engine, "--inpaint-model", args.model]
    if args.model_sha256:
        out += ["--inpaint-model-sha256", args.model_sha256]
    if args.checkpoint_sha256:
        out += ["--inpaint-checkpoint-sha256", args.checkpoint_sha256]
    if args.vae:
        out += ["--inpaint-vae", args.vae, "--inpaint-vae-sha256", args.vae_sha256]
    if args.tiles:
        out.append("--inpaint-tiles")
    if args.seed is not None:
        out += ["--inpaint-seed", args.seed]
    if args.steps is not None:
        out += ["--inpaint-steps", args.steps]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", required=True, type=Path)
    ap.add_argument("--engine", choices=("migan256", "migan512", "lama", "moebius", "pixelhacker"))
    ap.add_argument("--backend", choices=("migan", "migan256", "migan512", "lama"),
                    help="older spelling of --engine (migan means migan256)")
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--model-sha256", default="")
    ap.add_argument("--checkpoint-sha256", default="")
    ap.add_argument("--vae", type=Path)
    ap.add_argument("--vae-sha256", default="")
    ap.add_argument("--tiles", action="store_true", help="latent models: independent 512 tiles on the full photo")
    ap.add_argument("--crop", type=int, help="centre-crop each photo to NxN before subtitling")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--label", help="run label used in output names (default: engine)")
    ap.add_argument("--env", action="append", default=[], help="extra KEY=VALUE for the CLI (precision gates)")
    ap.add_argument("--reference-run", type=Path, help="score masked outputs against this earlier run directory")
    ap.add_argument("--photos", required=True, nargs="+", type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--text-model", type=Path)
    ap.add_argument("--threads", type=int, help="CLI --threads (default: the CLI's own default)")
    args = ap.parse_args()
    if bool(args.engine) == bool(args.backend):
        ap.error("give exactly one of --engine or --backend")
    if args.backend:
        args.engine = "migan256" if args.backend == "migan" else args.backend
    if bool(args.vae) != bool(args.vae_sha256):
        ap.error("--vae and --vae-sha256 go together")
    label = args.label or args.backend or args.engine
    env = dict(os.environ, CRISPEMBED_MIGAN="1", CRISPEMBED_LAMA="1")
    env[{"migan512": "CRISPEMBED_MIGAN512", "moebius": "CRISPEMBED_MOEBIUS",
         "pixelhacker": "CRISPEMBED_PIXELHACKER"}.get(args.engine, "CRISPEMBED_MIGAN")] = "1"
    for item in args.env:
        key, sep, value = item.partition("=")
        if not sep or not key:
            ap.error("--env takes KEY=VALUE")
        env[key] = value
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    report = {"engine": args.engine, "backend": args.backend, "label": label, "model": args.model.name,
              "vae": args.vae.name if args.vae else None, "crop": args.crop, "tiles": args.tiles,
              "threads": args.threads, "env": args.env,
              "reference_run": str(args.reference_run) if args.reference_run else None, "cases": []}
    for photo in args.photos:
        clean = Image.open(photo).convert("RGB")
        resized = None
        if args.crop:
            if min(clean.size) < args.crop:
                # Upscale the short side to N (recorded) instead of dropping the photo.
                scale = args.crop / min(clean.size)
                resized = [round(clean.width * scale), round(clean.height * scale)]
                clean = clean.resize(tuple(max(args.crop, v) for v in resized), Image.LANCZOS)
            left, top = (clean.width - args.crop) // 2, (clean.height - args.crop) // 2
            clean = clean.crop((left, top, left + args.crop, top + args.crop))
        stem = photo.stem
        subbed, mask = subtitle(clean)
        clean.save(out / f"{stem}.clean.png")
        subbed.save(out / f"{stem}.input.png")
        mask.save(out / f"{stem}.mask.png")
        c, s, mk = (np.asarray(x) for x in (clean, subbed, mask))
        band = mk > 0
        naive = s.copy()
        naive[band] = s[~band][-(s.shape[1] * 8):].reshape(-1, 3).mean(axis=0).astype(np.uint8)
        case = {"photo": photo.name, "size": list(clean.size), "resized_before_crop": resized, "naive_fill": scores(naive, c, mk), "runs": {}}
        jobs = {"exact-mask": ["--mask", out / f"{stem}.mask.png"]}
        if args.text_model:
            jobs["dbnet-mask"] = ["--mask-from-text", "--mask-text-model", args.text_model, "--mask-dilate", "6"]
        for name, mask_args in jobs.items():
            dest = out / f"{stem}.{label}.{name}.png"
            rc, err, secs, rss = run([args.binary, *engine_args(args),
                                      "--inpaint", out / f"{stem}.input.png", *mask_args, "--inpaint-output", dest,
                                      *(["--threads", args.threads] if args.threads else [])], env)
            entry = {"returncode": rc, "seconds": secs, "peak_rss_kib": rss}
            if rc == 0 and dest.exists():
                r = np.asarray(Image.open(dest).convert("RGB"))
                entry["shape_ok"] = r.shape == s.shape
                if entry["shape_ok"]:
                    entry["inpainted"] = scores(r, c, mk)
                    entry["outside_exact"] = bool(np.array_equal(r[~band], s[~band])) if name == "exact-mask" else None
                    if args.reference_run:
                        found = sorted(args.reference_run.glob(f"{stem}.*.{name}.png"))
                        if len(found) == 1:
                            other = np.asarray(Image.open(found[0]).convert("RGB"))
                            if other.shape == r.shape:
                                entry["vs_reference_run"] = agreement(r, other, mk)
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
