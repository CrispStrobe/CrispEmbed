#!/usr/bin/env python3
"""Ground-truth benchmark for CrispEmbed scan_cleanup vs. unpaper (and no-cleanup).

The metric that matters for a scan preprocessor is downstream OCR. This takes a
CLEAN page, records tesseract's read of it as the ground-truth text, applies a
battery of realistic scan degradations (the artifacts each tool targets), runs
each tool, and reports OCR CER vs. the ground truth — plus a clean|degraded|
CrispEmbed|unpaper contact sheet per degradation for human + visual-model review.

Requires: miniconda python (PIL, numpy, pytesseract), a crispembed binary, and
unpaper on PATH (optional — skipped if absent).

Usage:
    PY=~/miniconda3/bin/python
    $PY tools/scan_cleanup_bench.py --image clean_page.png --bin build/crispembed \
        --out-dir bench-out
    # degradations: uneven,speckle,hspeckle,border,shadow,skew  (default: all)
    #   --degradations uneven,shadow
"""

import argparse
import os
import re
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

try:
    import pytesseract
except Exception:
    pytesseract = None


# ── degradations (clean uint8 RGB -> degraded uint8 RGB) ────────────────
def deg_uneven(a, rng):
    H, W = a.shape[:2]
    gx = np.linspace(0.5, 1.0, W)[None, :, None]
    gy = (1 - 0.15 * np.abs(np.linspace(-1, 1, H)))[:, None, None]
    return np.clip(a * gx * gy, 0, 255)


def deg_speckle(a, rng, p=0.004):
    o = a.copy()
    m = rng.rand(*a.shape[:2]) < p
    o[m] = rng.randint(0, 60, size=(int(m.sum()), 3))
    return o


def deg_hspeckle(a, rng):
    o = deg_speckle(a, rng, p=0.05)
    H, W = a.shape[:2]
    for _ in range(120):
        y, x = rng.randint(0, H - 5), rng.randint(0, W - 5)
        o[y:y + rng.randint(2, 5), x:x + rng.randint(2, 5)] = 20
    return o


def deg_border(a, rng, b=14):
    o = a.copy()
    o[:b] = 15; o[-b:] = 15; o[:, :b] = 15; o[:, -b:] = 15
    return o


def deg_shadow(a, rng):
    im = Image.fromarray(a.astype(np.uint8)); W, H = im.size; d = ImageDraw.Draw(im)
    d.polygon([(W, 0), (W, H), (int(W * .80), H), (int(W * .88), int(H * .5)), (int(W * .82), 0)],
              fill=(12, 12, 12))
    d.ellipse([20, 30, 70, 80], fill=(15, 15, 15))
    return np.asarray(im).astype(np.float32)


def deg_skew(a, rng, angle=-4):
    im = Image.fromarray(a.astype(np.uint8)).rotate(angle, expand=True, fillcolor=(250, 250, 246))
    return np.asarray(im).astype(np.float32)


def deg_haze(a, rng):
    # faint gray haze/smudge over the paper: soft light-gray veil, never near-black
    # (capped), applied once so it stays a light haze rather than a dark stain.
    H, W = a.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W]
    mask = np.zeros((H, W), np.float32)
    for _ in range(4):
        cy, cx = rng.randint(0, H), rng.randint(0, W)
        r = rng.randint(H // 8, H // 3)
        mask = np.maximum(mask, np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * r * r))))
    return a * (1 - 0.32 * mask[..., None])     # <=32% darkening -> light gray, kept once


def deg_darkvignette(a, rng):
    # heavy dark stain/vignette (compounding blotches -> near-black in places) that
    # is still HUMANLY READABLE — the do-no-harm case: cleanup must not destroy it.
    o = a.copy(); H, W = a.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W]
    for _ in range(5):
        cy, cx = rng.randint(0, H), rng.randint(0, W)
        r = rng.randint(H // 8, H // 3)
        d = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * r * r)))
        o = o * (1 - 0.45 * d[..., None])
    return o


DEGRADATIONS = {
    "uneven": deg_uneven, "speckle": deg_speckle, "hspeckle": deg_hspeckle,
    "border": deg_border, "shadow": deg_shadow, "skew": deg_skew, "haze": deg_haze,
    "darkvignette": deg_darkvignette,
}


# ── metrics / tools ─────────────────────────────────────────────────────
def cer(ref, hyp):
    a = re.sub(r"\s+", " ", ref.strip().lower())
    b = re.sub(r"\s+", " ", hyp.strip().lower())
    m, n = len(a), len(b)
    if m == 0:
        return 0.0 if n == 0 else 1.0
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return dp[n] / m


def ocr(img_path):
    return pytesseract.image_to_string(Image.open(img_path).convert("RGB"))


def run_crispembed(bin_path, in_png, out_ppm):
    env = dict(os.environ); env["CRISPEMBED_FORCE_CPU"] = "1"
    with open(out_ppm, "wb") as f:
        p = subprocess.run([bin_path, "--cleanup-only", in_png], stdout=f,
                           stderr=subprocess.DEVNULL, env=env)
    return p.returncode == 0 and os.path.getsize(out_ppm) > 0


def run_unpaper(in_png, out_ppm, extra=None):
    in_ppm = in_png.rsplit(".", 1)[0] + ".ppm"
    Image.open(in_png).convert("RGB").save(in_ppm)
    cmd = ["unpaper", "--overwrite", "--layout", "none", *(extra or []), in_ppm, out_ppm]
    p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return p.returncode == 0 and os.path.exists(out_ppm)


def contact_sheet(paths, labels, out_path, target_h=420):
    ims = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        s = target_h / im.height
        ims.append(im.resize((max(1, int(im.width * s)), target_h)))
    gap = 8
    W = sum(im.width for im in ims) + gap * (len(ims) - 1)
    sheet = Image.new("RGB", (W, target_h), (200, 0, 0))
    x = 0
    for im in ims:
        sheet.paste(im, (x, 0)); x += im.width + gap
    sheet.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="clean page image (ground truth)")
    ap.add_argument("--bin", default="build/crispembed")
    ap.add_argument("--out-dir", default="scan-bench")
    ap.add_argument("--degradations", default=",".join(DEGRADATIONS))
    ap.add_argument("--max-width", type=int, default=1000, help="downscale clean page for speed")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--gt-text", help="ground-truth text file (default: tesseract on the clean page)")
    ap.add_argument("--no-unpaper", action="store_true")
    args = ap.parse_args()
    if pytesseract is None:
        sys.exit("pytesseract required (miniconda python)")

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.RandomState(args.seed)
    clean = Image.open(args.image).convert("RGB")
    if clean.width > args.max_width:
        s = args.max_width / clean.width
        clean = clean.resize((args.max_width, int(clean.height * s)), Image.LANCZOS)
    clean_p = os.path.join(args.out_dir, "clean.png"); clean.save(clean_p)
    a = np.asarray(clean).astype(np.float32)

    gt = open(args.gt_text).read() if args.gt_text else ocr(clean_p)
    print(f"clean {clean.width}x{clean.height}  gt={len(gt)} chars  "
          f"clean-CER={cer(gt, ocr(clean_p)):.3f} (sanity)")
    have_unpaper = (not args.no_unpaper) and subprocess.run(
        ["which", "unpaper"], stdout=subprocess.DEVNULL).returncode == 0

    print(f"\n{'degradation':12} {'degraded':>9} {'CrispEmbed':>11} {'unpaper':>9}   winner")
    rows = []
    for name in args.degradations.split(","):
        name = name.strip()
        if name not in DEGRADATIONS:
            print(f"  (skip unknown '{name}')"); continue
        deg = np.clip(DEGRADATIONS[name](a, rng), 0, 255).astype(np.uint8)
        deg_p = os.path.join(args.out_dir, f"{name}_degraded.png")
        Image.fromarray(deg).save(deg_p)
        cc_ppm = os.path.join(args.out_dir, f"{name}_crispembed.ppm")
        cc_ok = run_crispembed(args.bin, deg_p, cc_ppm)
        panels, labels = [clean_p, deg_p], ["clean", "degraded"]
        c_deg = cer(gt, ocr(deg_p))
        c_cc = cer(gt, ocr(cc_ppm)) if cc_ok else float("nan")
        if cc_ok:
            Image.open(cc_ppm).save(os.path.join(args.out_dir, f"{name}_crispembed.png"))
            panels.append(cc_ppm); labels.append("crispembed")
        c_up = float("nan")
        if have_unpaper:
            up_ppm = os.path.join(args.out_dir, f"{name}_unpaper.ppm")
            if run_unpaper(deg_p, up_ppm):
                c_up = cer(gt, ocr(up_ppm))
                Image.open(up_ppm).save(os.path.join(args.out_dir, f"{name}_unpaper.png"))
                panels.append(up_ppm); labels.append("unpaper")
        contact_sheet(panels, labels, os.path.join(args.out_dir, f"sheet_{name}.png"))
        cands = {"CrispEmbed": c_cc, "unpaper": c_up}
        cands = {k: v for k, v in cands.items() if v == v}
        win = min(cands, key=cands.get) if cands else "-"
        rows.append((name, c_deg, c_cc, c_up, win))
        print(f"{name:12} {c_deg:9.3f} {c_cc:11.3f} {c_up:9.3f}   {win}")

    print(f"\nsheets + images in {args.out_dir}/ (panels: clean | degraded | crispembed | unpaper)")


if __name__ == "__main__":
    main()
