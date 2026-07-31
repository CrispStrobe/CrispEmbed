#!/usr/bin/env python3
"""Create deterministic robustness variants from verified PD/CC0 fixtures.

Derived images are test inputs, not new source material: each record retains
the parent SHA-256 and the exact transformation recipe.  The default set is
small enough for source control and exercises geometry, illumination, noise,
resolution, compression, orientation, and mixed-orientation handling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "tests/regression/images/cc0"
OUT_DIR = ROOT / "tests/regression/images/derived"
MANIFEST = OUT_DIR / "MANIFEST.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gradient(img: Image.Image, dark: bool) -> Image.Image:
    rgb = img.convert("RGB")
    width, height = rgb.size
    pixels = rgb.load()
    for y in range(height):
        for x in range(width):
            phase = (x / max(1, width - 1) + y / max(1, height - 1)) / 2
            factor = (0.55 + 0.45 * phase) if dark else (0.82 + 0.18 * phase)
            r, g, b = pixels[x, y]
            pixels[x, y] = tuple(min(255, int(channel * factor)) for channel in (r, g, b))
    return rgb


def speckle(img: Image.Image, seed: int) -> Image.Image:
    rng = random.Random(seed)
    out = img.convert("RGB")
    pixels = out.load()
    count = max(1, out.width * out.height // 1800)
    for _ in range(count):
        x = rng.randrange(out.width)
        y = rng.randrange(out.height)
        value = rng.choice((0, 0, 0, 255, 255, 255))
        pixels[x, y] = (value, value, value)
    return out.filter(ImageFilter.GaussianBlur(radius=0.15))


def perspective(img: Image.Image) -> Image.Image:
    # A fixed trapezoid is sufficient to stress polygon geometry while keeping
    # the generated fixture reproducible across Pillow versions.
    width, height = img.size
    inset = max(2, width // 18)
    return img.transform(
        (width, height), Image.Transform.QUAD,
        (inset, 0, width - inset, 0, width, height, 0, height),
        resample=Image.Resampling.BICUBIC,
    )


def mixed_lines(img: Image.Image) -> Image.Image:
    out = img.convert("RGB")
    band = max(1, out.height // 5)
    for index, top in enumerate(range(0, out.height, band)):
        bottom = min(out.height, top + band)
        if index % 2:
            crop = out.crop((0, top, out.width, bottom)).rotate(180, expand=False)
            out.paste(crop, (0, top))
    return out


def variants(img: Image.Image) -> list[tuple[str, Image.Image, dict]]:
    width, height = img.size
    low_w = max(32, width // 3)
    low_h = max(32, height // 3)
    return [
        ("skew-p04", img.rotate(4, expand=True, fillcolor="white"), {"op": "rotate", "degrees": 4}),
        ("skew-m04", img.rotate(-4, expand=True, fillcolor="white"), {"op": "rotate", "degrees": -4}),
        ("skew-p08", img.rotate(8, expand=True, fillcolor="white"), {"op": "rotate", "degrees": 8}),
        ("skew-m08", img.rotate(-8, expand=True, fillcolor="white"), {"op": "rotate", "degrees": -8}),
        ("dark-border", ImageOps.expand(img, border=max(8, min(width, height) // 45), fill=(35, 35, 35)),
         {"op": "border", "pixels": max(8, min(width, height) // 45), "rgb": [35, 35, 35]}),
        ("uneven-illumination", gradient(img, True), {"op": "gradient", "minimum": 0.55, "maximum": 1.0}),
        ("haze", Image.blend(img.convert("RGB"), Image.new("RGB", img.size, "white"), 0.32),
         {"op": "blend", "color": "white", "amount": 0.32}),
        ("speckle", speckle(img, 20260731), {"op": "speckle", "seed": 20260731}),
        ("low-dpi", img.resize((low_w, low_h), Image.Resampling.BOX).resize((width, height), Image.Resampling.BICUBIC),
         {"op": "resize", "downsample": [low_w, low_h], "restore": [width, height]}),
        ("jpeg-damage", Image.open(_jpeg_roundtrip(img)).convert("RGB"),
         {"op": "jpeg", "quality": 28}),
        ("rot90", img.rotate(90, expand=True), {"op": "rotate", "degrees": 90}),
        ("rot180", img.rotate(180, expand=True), {"op": "rotate", "degrees": 180}),
        ("rot270", img.rotate(270, expand=True), {"op": "rotate", "degrees": 270}),
        ("perspective", perspective(img), {"op": "perspective", "top_inset": max(2, width // 18)}),
        ("mixed-orientation", mixed_lines(img), {"op": "rotate-alternating-horizontal-bands", "band_count": 5}),
    ]


_JPEG_TEMP: Path | None = None


def _jpeg_roundtrip(img: Image.Image) -> Path:
    global _JPEG_TEMP
    if _JPEG_TEMP is None:
        _JPEG_TEMP = OUT_DIR / ".jpeg-roundtrip.jpg"
    img.convert("RGB").save(_JPEG_TEMP, format="JPEG", quality=28, optimize=False, progressive=False)
    return _JPEG_TEMP


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", help="CC0 filename; repeat to limit sources")
    ap.add_argument("--output", default=str(OUT_DIR))
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    names = args.source or ["receipt_example.png", "german_official_document.jpg", "arabic_printed_line.png"]
    records = []
    for name in names:
        source = SOURCE_DIR / name
        if not source.is_file():
            raise SystemExit(f"missing source fixture: {source}")
        parent_sha = digest(source)
        base = Image.open(source).convert("RGB")
        for suffix, image, recipe in variants(base):
            target = out / f"{source.stem}__{suffix}.png"
            image.save(target, format="PNG", optimize=False)
            records.append({
                "file": str(target.relative_to(ROOT)),
                "parent": str(source.relative_to(ROOT)),
                "parent_sha256": parent_sha,
                "sha256": digest(target),
                "recipe": recipe,
                "size": list(image.size),
            })
    if _JPEG_TEMP is not None:
        _JPEG_TEMP.unlink(missing_ok=True)
    MANIFEST_PATH = out / "MANIFEST.json"
    MANIFEST_PATH.write_text(json.dumps({"version": 1, "records": records}, indent=2) + "\n")
    print(f"generated={len(records)} manifest={MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
