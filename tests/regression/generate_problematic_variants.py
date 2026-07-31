#!/usr/bin/env python3
"""Create deterministic, provenance-tracked real-image OCR stress variants.

The source images remain CC0/public-domain fixtures.  Derived files are not
claims about model quality; they are controlled inputs for preprocessing and
orientation regression.  Every manifest row stores the parent checksum,
recipe, seed, and derived checksum.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


ROOT = Path(__file__).resolve().parents[2]
CC0 = ROOT / "tests/regression/images/cc0"
DEFAULT_OUT = ROOT / "tests/regression/images/derived"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def gradient(im: Image.Image) -> Image.Image:
    gray = ImageOps.grayscale(im)
    w, h = gray.size
    px = gray.load()
    out = Image.new("L", (w, h))
    dst = out.load()
    for y in range(h):
        factor = 0.58 + 0.42 * y / max(1, h - 1)
        for x in range(w):
            dst[x, y] = max(0, min(255, int(px[x, y] * factor)))
    return Image.merge("RGB", (out, out, out))


def speckle(im: Image.Image, seed: int) -> Image.Image:
    rng = random.Random(seed)
    out = im.convert("RGB").copy()
    px = out.load()
    w, h = out.size
    count = max(1, w * h // 160)
    for _ in range(count):
        x, y = rng.randrange(w), rng.randrange(h)
        v = 0 if rng.random() < 0.55 else 255
        px[x, y] = (v, v, v)
    return out


def perspective(im: Image.Image) -> Image.Image:
    w, h = im.size
    # A mild reproducible keystone distortion; coefficients are fixed and
    # intentionally small enough that text remains recoverable.
    return im.transform((w, h), Image.Transform.QUAD,
                        (int(.03*w), int(.02*h), int(.97*w), 0,
                         w, h, 0, int(.98*h)), Image.Resampling.BICUBIC)


def variants(im: Image.Image, seed: int) -> dict[str, Image.Image]:
    return {
        "skew_m4": im.rotate(-4, expand=True, fillcolor="white"),
        "skew_p4": im.rotate(4, expand=True, fillcolor="white"),
        "skew_m8": im.rotate(-8, expand=True, fillcolor="white"),
        "skew_p8": im.rotate(8, expand=True, fillcolor="white"),
        "rotate_90": im.rotate(90, expand=True, fillcolor="white"),
        "rotate_180": im.rotate(180, expand=True, fillcolor="white"),
        "rotate_270": im.rotate(270, expand=True, fillcolor="white"),
        "dark_border": ImageOps.expand(im, border=max(8, min(im.size)//40), fill=(25, 25, 25)),
        "uneven_illumination": gradient(im),
        "speckle": speckle(im, seed),
        "low_dpi": im.resize((max(1, im.width//3), max(1, im.height//3)), Image.Resampling.BOX),
        "jpeg_damage": im.convert("RGB"),
        "perspective": perspective(im),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", default=[], help="CC0 filename; repeat")
    ap.add_argument("--output-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--seed", type=int, default=20260731)
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--jpeg-quality", type=int, default=28)
    args = ap.parse_args()
    sources = [CC0 / x for x in args.source] if args.source else sorted(CC0.iterdir())
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for source in sources:
        if source.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
            continue
        parent = digest(source)
        image = Image.open(source).convert("RGB")
        for name, derived in variants(image, args.seed).items():
            if args.variants and name not in args.variants:
                continue
            path = outdir / f"{source.stem}__{name}.png"
            recipe = {"operation": name, "seed": args.seed}
            if name == "jpeg_damage":
                # Save through JPEG and decode again before writing a stable
                # PNG fixture, so the derived checksum is platform-independent.
                tmp = outdir / f".{path.stem}.jpg"
                derived.save(tmp, format="JPEG", quality=args.jpeg_quality, optimize=False)
                derived = Image.open(tmp).convert("RGB")
                tmp.unlink()
                recipe["quality"] = args.jpeg_quality
            derived.save(path, format="PNG", optimize=False)
            rows.append({"file": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                         "parent": str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source),
                         "parent_sha256": parent, "sha256": digest(path), "recipe": recipe,
                         "license_inherited": "parent CC0/public-domain only"})
    manifest = outdir / "MANIFEST.json"
    manifest.write_text(json.dumps({"version": 1, "rows": rows}, indent=2) + "\n")
    print(f"generated={len(rows)} manifest={manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
