#!/usr/bin/env python3
"""Deterministic synthetic OCR corpus with exact ground truth.

Real-world fixtures (``tests/regression/images/cc0``) have no transcription, so
they can only score cross-engine agreement.  This renders text we already know
into page images, which turns CER/WER into an absolute number and makes a port
bug separable from a hard input: on clean rendered text every mature engine
(Tesseract, EasyOCR, PaddleOCR) scores near zero, so a CrispEmbed engine that
does not is wrong, not merely weaker.

The corpus is generated, never checked in: the renderer is the artifact.  Same
seed and same PIL/font version reproduce the same bytes; the manifest records
the font path and a sha256 per image so a mismatch is visible rather than
silently scoring a different picture.

Usage:
  python tests/ocr_synth_corpus.py --output /tmp/ocr-synth
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

# Sentences are ASCII-only and word-shaped: CER/WER on them measures the
# recognizer, not a tokenizer's unicode normalisation.
PARAGRAPHS = [
    [
        "The quick brown fox jumps over the lazy dog.",
        "Pack my box with five dozen liquor jugs.",
        "How vexingly quick daft zebras jump!",
    ],
    [
        "Invoice number 48127 dated 14 March 2024.",
        "Subtotal 129.95 EUR, tax 24.69 EUR, total 154.64 EUR.",
        "Payment due within 30 days of receipt.",
    ],
    [
        "Chapter 7. Measurement and error.",
        "Every instrument reports a value and an uncertainty.",
        "A number without an interval is not a measurement.",
    ],
    [
        "Sphinx of black quartz, judge my vow.",
        "Waltz, bad nymph, for quick jigs vex.",
        "Five quacking zephyrs jolt my wax bed.",
    ],
]

FONTS = [
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Georgia.ttf",
    "/System/Library/Fonts/Supplemental/Courier New.ttf",
]


def render(lines: list[str], font_path: str, size: int, degrade: str, seed: int) -> Image.Image:
    font = ImageFont.truetype(font_path, size)
    pad = size
    probe = Image.new("L", (8, 8), 255)
    measure = ImageDraw.Draw(probe)
    widths, height = [], int(size * 1.6)
    for line in lines:
        box = measure.textbbox((0, 0), line, font=font)
        widths.append(box[2] - box[0])
    w = max(widths) + 2 * pad
    h = height * len(lines) + 2 * pad
    img = Image.new("L", (w, h), 255)
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        draw.text((pad, pad + i * height), line, font=font, fill=0)

    rng = random.Random(seed)
    if degrade == "blur":
        img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
    elif degrade == "noise":
        px = img.load()
        for _ in range((w * h) // 40):
            x, y = rng.randrange(w), rng.randrange(h)
            px[x, y] = rng.choice((0, 255))
    elif degrade == "lowdpi":
        img = img.resize((w // 2, h // 2), Image.LANCZOS)
    elif degrade == "skew":
        img = img.rotate(1.5, resample=Image.BICUBIC, expand=True, fillcolor=255)
    return img.convert("RGB")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=20260801)
    args = ap.parse_args()

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    fonts = [f for f in FONTS if Path(f).exists()]
    if not fonts:
        raise SystemExit(f"no usable font among {FONTS}")

    records = []
    case = 0
    for pi, lines in enumerate(PARAGRAPHS):
        for degrade in ("clean", "blur", "noise", "lowdpi", "skew"):
            font = fonts[case % len(fonts)]
            size = 28 if degrade != "lowdpi" else 44
            img = render(lines, font, size, degrade, args.seed + case)
            name = f"synth_{pi:02d}_{degrade}.png"
            img.save(out / name)
            digest = hashlib.sha256((out / name).read_bytes()).hexdigest()
            records.append({
                "name": name,
                "file": name,
                "text": "\n".join(lines),
                "degrade": degrade,
                "font": font,
                "point_size": size,
                "sha256": digest,
                "size": (out / name).stat().st_size,
            })
            case += 1

    (out / "ground_truth.json").write_text(json.dumps({
        "version": 1,
        "seed": args.seed,
        "generator": "tests/ocr_synth_corpus.py",
        "records": records,
    }, indent=2) + "\n")
    print(f"wrote {len(records)} fixtures + ground_truth.json to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
