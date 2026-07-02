#!/usr/bin/env python3
"""Ground-truth quality eval for CrispEmbed restoration/SR runtimes.

Unlike the per-stage parity diff (test-*-diff, which only proves the C++ matches a
reference implementation of the *same* math), this scores the actual OUTPUT pixels
against a known-clean ground truth:

  * denoise engines (restormer, scunet, instructir, adair):
        clean --add gaussian noise(sigma)--> noisy --engine--> denoised
        report PSNR/SSIM(denoised, clean) vs PSNR/SSIM(noisy, clean)
  * super-resolution engines (esrgan, swinir, dat, hat, pan, safmn, tbsrn):
        HR clean --bicubic downscale(1/scale)--> LR --engine(xscale)--> SR
        report PSNR/SSIM(SR, HR) vs PSNR/SSIM(bicubic-upscaled LR, HR)

Ground-truth images: pass your own with --image, or use --builtin to pull a
free-licensed real photo from skimage.data (astronaut = NASA public domain,
chelsea/coffee = CC0). Writes a clean|degraded|restored contact sheet per case.

Uses the miniconda python (torch/PIL/numpy/skimage); SSIM is optional.

Examples:
    PY=~/miniconda3/bin/python
    $PY tools/eval_restoration_quality.py --engine restormer \
        --model restormer-denoise-f16.gguf --builtin astronaut --sigmas 15,25,50
    $PY tools/eval_restoration_quality.py --engine esrgan \
        --model esrgan-x4.gguf --builtin chelsea --mode sr --scale 4
"""

import argparse
import os
import subprocess
import sys

import numpy as np
from PIL import Image

try:
    from skimage.metrics import structural_similarity as _ssim
    HAVE_SSIM = True
except Exception:
    HAVE_SSIM = False


# ── engine registry ─────────────────────────────────────────────────────
# mode: "denoise" (output same size, recover clean) or "sr" (upscale by scale).
# argv(model, img): CLI args after the crispembed binary. All engines write a
# P6 PPM to stdout. env: extra {ENGINE}_FORCE_CPU-style var for --force-cpu.
ENGINES = {
    # denoisers / same-size restoration
    "restormer":  {"mode": "denoise", "argv": lambda m, i: ["--restormer", m, "--restormer-input", i],       "cpu_env": "RESTORMER_FORCE_CPU"},
    "scunet":     {"mode": "denoise", "argv": lambda m, i: ["--scunet-model", m, "--scunet-denoise", i],      "cpu_env": "SCUNET_FORCE_CPU"},
    "instructir": {"mode": "denoise", "argv": lambda m, i: ["--instructir-model", m, "--instructir", i],      "cpu_env": "INSTRUCTIR_FORCE_CPU"},
    "adair":      {"mode": "denoise", "argv": lambda m, i: ["--adair-model", m, "--adair", i],                "cpu_env": "ADAIR_FORCE_CPU"},
    # super-resolution (default scale is the model's; override with --scale)
    "esrgan":     {"mode": "sr", "scale": 4, "argv": lambda m, i: ["--esrgan-model", m, "--esrgan-sr", i],    "cpu_env": "ESRGAN_FORCE_CPU"},
    "swinir":     {"mode": "sr", "scale": 4, "argv": lambda m, i: ["--swinir-model", m, "--swinir-sr", i],    "cpu_env": "SWINIR_FORCE_CPU"},
    "dat":        {"mode": "sr", "scale": 2, "argv": lambda m, i: ["--dat-model", m, "--dat-sr", i],          "cpu_env": "DAT_SR_FORCE_CPU"},
    "hat":        {"mode": "sr", "scale": 4, "argv": lambda m, i: ["--hat-model", m, "--hat-sr", i],          "cpu_env": "HAT_FORCE_CPU"},
    "pan":        {"mode": "sr", "scale": 4, "argv": lambda m, i: ["--pan-model", m, "--pan-sr", i],          "cpu_env": "PAN_FORCE_CPU"},
    "safmn":      {"mode": "sr", "scale": 4, "argv": lambda m, i: ["--safmn-model", m, "--safmn-sr", i],      "cpu_env": "SAFMN_FORCE_CPU"},
    "tbsrn":      {"mode": "sr", "scale": 2, "argv": lambda m, i: ["--tbsrn-model", m, "--tbsrn-sr", i],      "cpu_env": "TBSRN_FORCE_CPU"},
}


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse < 1e-9 else 20.0 * np.log10(255.0 / np.sqrt(mse))


def ssim(a, b):
    if not HAVE_SSIM:
        return float("nan")
    return _ssim(a, b, channel_axis=2, data_range=255)


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


_BOOK_TEXT = (
    "It is a truth universally acknowledged, that a single man in possession of "
    "a good fortune, must be in want of a wife. However little known the feelings "
    "or views of such a man may be on his first entering a neighbourhood, this "
    "truth is so well fixed in the minds of the surrounding families, that he is "
    "considered the rightful property of some one or other of their daughters. "
    "My dear Mr. Bennet, said his lady to him one day, have you heard that "
    "Netherfield Park is let at last? Mr. Bennet replied that he had not. But it "
    "is, returned she; for Mrs. Long has just been here, and she told me all "
    "about it. Mr. Bennet made no answer. Do you not want to know who has taken "
    "it? cried his wife impatiently. You want to tell me, and I have no objection "
    "to hearing it. This was invitation enough."
)


def synth_bookpage(w=560, h=720):
    """Clean synthetic book page: justified serif text on a lightly-tinted page.
    Serves as ground truth for a scan-restoration eval (no real scan has a clean
    reference; this does). Public-domain text (Austen)."""
    from PIL import ImageDraw, ImageFont
    import glob
    page = np.full((h, w, 3), 250, np.uint8)
    page[..., 2] = 246  # faint warm paper tint
    im = Image.fromarray(page)
    d = ImageDraw.Draw(im)
    font = None
    for pat in ["/System/Library/Fonts/Supplemental/Times New Roman.ttf",
                "/System/Library/Fonts/Supplemental/Georgia.ttf",
                "/System/Library/Fonts/NewYork.ttf"]:
        for f in glob.glob(pat):
            try:
                font = ImageFont.truetype(f, 20); break
            except Exception:
                pass
        if font:
            break
    if font is None:
        font = ImageFont.load_default()
    margin, x, y, lh, maxw = 48, 48, 60, 30, w - 96
    words, line = _BOOK_TEXT.split(), ""
    for wd in words:
        t = (line + " " + wd).strip()
        if d.textlength(t, font=font) > maxw:
            d.text((x, y), line, fill=(20, 20, 22), font=font); y += lh; line = wd
        else:
            line = t
        if y > h - lh - margin:
            break
    if line and y <= h - lh - margin:
        d.text((x, y), line, fill=(20, 20, 22), font=font)
    return np.asarray(im)[..., :3].astype(np.uint8)


def builtin_image(name):
    if name == "bookpage":
        return synth_bookpage()
    from skimage import data
    fn = getattr(data, name, None)
    if fn is None:
        sys.exit(f"unknown builtin '{name}' (try: bookpage, astronaut, chelsea, coffee, cat)")
    arr = np.asarray(fn())
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, -1)
    return arr[..., :3].astype(np.uint8)


def run_engine(bin_path, argv, out_ppm, cpu_env, force_cpu):
    env = dict(os.environ)
    if force_cpu:
        env["CRISPEMBED_FORCE_CPU"] = "1"
        if cpu_env:
            env[cpu_env] = "1"
    with open(out_ppm, "wb") as f:
        p = subprocess.run([bin_path, *argv], stdout=f, stderr=subprocess.PIPE, env=env)
    if p.returncode != 0 or os.path.getsize(out_ppm) == 0:
        sys.exit(f"engine run failed (rc={p.returncode}):\n{p.stderr.decode()[-800:]}")
    return load_rgb(out_ppm)


def crop_to(*imgs):
    h = min(x.shape[0] for x in imgs)
    w = min(x.shape[1] for x in imgs)
    return [x[:h, :w] for x in imgs]


def save_sheet(path, panels, axis):
    panels = crop_to(*panels)
    Image.fromarray(np.concatenate(panels, axis=axis).astype(np.uint8)).save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True, choices=sorted(ENGINES))
    ap.add_argument("--model", required=True, help="engine GGUF")
    ap.add_argument("--image", help="clean ground-truth image (PNG/JPG)")
    ap.add_argument("--builtin", help="free-licensed skimage.data sample: astronaut|chelsea|coffee|cat")
    ap.add_argument("--mode", choices=["auto", "denoise", "sr"], default="auto")
    ap.add_argument("--sigmas", default="25", help="denoise: comma list of noise stddevs (0-255)")
    ap.add_argument("--scale", type=int, help="sr: downscale/upscale factor (default: engine's)")
    ap.add_argument("--max-size", type=int, default=256,
                    help="downscale the GT so the longer side <= this (CPU speed); 0 = full")
    ap.add_argument("--bin", default=os.environ.get("CRISPEMBED_BIN", "build/crispembed"))
    ap.add_argument("--out-dir", default="eval-out")
    ap.add_argument("--force-cpu", action="store_true", default=True)
    ap.add_argument("--gpu", dest="force_cpu", action="store_false")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    spec = ENGINES[args.engine]
    mode = spec["mode"] if args.mode == "auto" else args.mode
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    if args.builtin:
        clean = builtin_image(args.builtin)
        tag = args.builtin
    elif args.image:
        clean = load_rgb(args.image)
        tag = os.path.splitext(os.path.basename(args.image))[0]
    else:
        sys.exit("provide --image or --builtin")

    if args.max_size and max(clean.shape[:2]) > args.max_size:
        im = Image.fromarray(clean)
        s = args.max_size / max(clean.shape[:2])
        clean = np.asarray(im.resize((int(clean.shape[1] * s), int(clean.shape[0] * s)),
                                     Image.LANCZOS))
    # keep dims divisible by 8 (tiling / pixel-shuffle friendly)
    H, W = (clean.shape[0] // 8) * 8, (clean.shape[1] // 8) * 8
    clean = clean[:H, :W]
    Image.fromarray(clean).save(os.path.join(args.out_dir, f"{tag}_clean.png"))
    print(f"engine={args.engine} mode={mode} image={tag} size={W}x{H} "
          f"ssim={'on' if HAVE_SSIM else 'off (pip install scikit-image)'}")

    rows = []
    if mode == "denoise":
        for sig in [int(s) for s in args.sigmas.split(",")]:
            f = clean.astype(np.float32) / 255.0
            noisy = np.clip(f + rng.normal(0, sig / 255.0, f.shape), 0, 1)
            noisy = (noisy * 255.0).round().astype(np.uint8)
            npath = os.path.join(args.out_dir, f"{tag}_noisy{sig}.png")
            Image.fromarray(noisy).save(npath)
            out = run_engine(args.bin, spec["argv"](args.model, npath),
                             os.path.join(args.out_dir, f"{tag}_den{sig}.ppm"),
                             spec.get("cpu_env"), args.force_cpu)
            c, no, de = crop_to(clean, noisy, out)
            rows.append((f"sigma={sig}", psnr(no, c), ssim(no, c), psnr(de, c), ssim(de, c)))
            save_sheet(os.path.join(args.out_dir, f"sheet_{tag}_{args.engine}_s{sig}.png"),
                       [c, no, de], axis=1 if W >= H else 0)
    else:  # sr
        scale = args.scale or spec.get("scale", 4)
        lr = Image.fromarray(clean).resize((W // scale, H // scale), Image.BICUBIC)
        lpath = os.path.join(args.out_dir, f"{tag}_lr.png")
        lr.save(lpath)
        bicubic = np.asarray(lr.resize((W, H), Image.BICUBIC))
        out = run_engine(args.bin, spec["argv"](args.model, lpath),
                         os.path.join(args.out_dir, f"{tag}_sr.ppm"),
                         spec.get("cpu_env"), args.force_cpu)
        c, bi, sr = crop_to(clean, bicubic, out)
        rows.append((f"x{scale}", psnr(bi, c), ssim(bi, c), psnr(sr, c), ssim(sr, c)))
        save_sheet(os.path.join(args.out_dir, f"sheet_{tag}_{args.engine}_x{scale}.png"),
                   [c, bi, sr], axis=1 if W >= H else 0)

    base_lbl = "noisy" if mode == "denoise" else "bicubic"
    print(f"\n{'case':10} {base_lbl+' PSNR/SSIM':22} {'RESTORED PSNR/SSIM':22} gain")
    for name, pb, sb, pr, sr_ in rows:
        print(f"{name:10} {pb:6.2f}dB/{sb:.3f}        {pr:6.2f}dB/{sr_:.3f}        {pr-pb:+.2f}dB")
    print(f"\ncontact sheets + images in {args.out_dir}/ "
          f"(panels: clean | {base_lbl} | restored)")


if __name__ == "__main__":
    main()
