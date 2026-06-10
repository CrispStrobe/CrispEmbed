#!/usr/bin/env python3
"""Prepare mixed CROHME + 1000 MathWriting training zip.

Uses PosFormer v1 vocabulary (110 tokens). Filters MathWriting samples
to only those expressible in this vocabulary. Rasterizes InkML strokes
to BMP images. Outputs a PosFormer-compatible zip.
"""

import os
import re
import sys
import random
import zipfile
import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

POSFORMER_DICT = Path("/mnt/volume1/PosFormer-fresh/Pos_Former/datamodule/dictionary.txt")
CROHME_ZIP = Path("/mnt/volume1/data_crohme.zip")
MATHWRITING_DIR = Path("/mnt/volume1/mathwriting/mathwriting-2024")
OUTPUT_ZIP = Path("/mnt/volume1/data_crohme_plus_mw1k.zip")

N_MATHWRITING = 1000


def load_vocab():
    tokens = set()
    commands = set()
    singles = set()
    with open(POSFORMER_DICT) as f:
        for line in f:
            t = line.strip()
            if t:
                tokens.add(t)
                if t.startswith('\\'):
                    commands.add(t)
                else:
                    singles.add(t)
    return tokens, commands, singles


def strict_tokenize(latex, commands, singles):
    """Parse raw LaTeX into v1 PosFormer tokens. Returns list or None."""
    tokens = []
    i = 0
    s = latex.strip()
    while i < len(s):
        if s[i].isspace():
            i += 1
            continue
        if s[i] == '\\':
            m = re.match(r'\\[a-zA-Z]+', s[i:])
            if m:
                cmd = m.group(0)
                if cmd in commands:
                    tokens.append(cmd)
                    i += len(cmd)
                else:
                    return None
            elif i + 1 < len(s):
                pair = s[i:i+2]
                if pair in commands or pair in singles:
                    tokens.append(pair)
                    i += 2
                else:
                    return None
            else:
                return None
        elif s[i] in singles:
            tokens.append(s[i])
            i += 1
        else:
            return None
    return tokens if len(tokens) >= 2 else None


def parse_inkml(path):
    """Parse InkML → (strokes, normalizedLabel)."""
    tree = ET.parse(path)
    root = tree.getroot()
    label = None
    for ann in root.iter():
        if ann.tag.endswith("annotation"):
            t = ann.get("type", "")
            if t == "normalizedLabel":
                label = ann.text
            elif t == "label" and label is None:
                label = ann.text
    strokes = []
    for trace in root.iter():
        if not trace.tag.endswith("trace"):
            continue
        points = []
        for pt in trace.text.strip().split(","):
            coords = pt.strip().split()
            if len(coords) >= 2:
                try:
                    points.append((float(coords[0]), float(coords[1])))
                except ValueError:
                    pass
        if points:
            strokes.append(points)
    return strokes, label


def rasterize(strokes, target_h=128, line_w=3, pad=10):
    """Strokes → grayscale PIL Image."""
    if not strokes:
        return Image.new("L", (target_h, target_h), 255)
    all_x = [p[0] for s in strokes for p in s]
    all_y = [p[1] for s in strokes for p in s]
    x0, x1 = min(all_x), max(all_x)
    y0, y1 = min(all_y), max(all_y)
    wr, hr = x1 - x0 + 1e-6, y1 - y0 + 1e-6
    scale = (target_h - 2 * pad) / hr
    img_w = max(int(wr * scale + 2 * pad), target_h // 2)
    img = Image.new("L", (img_w, target_h), 255)
    draw = ImageDraw.Draw(img)
    for stroke in strokes:
        pts = [(int((x - x0) * scale + pad), int((y - y0) * scale + pad))
               for x, y in stroke]
        if len(pts) >= 2:
            draw.line(pts, fill=0, width=line_w)
        elif pts:
            x, y = pts[0]
            r = line_w // 2
            draw.ellipse([x-r, y-r, x+r, y+r], fill=0)
    return img


def main():
    vocab, commands, singles = load_vocab()
    print(f"PosFormer v1 vocab: {len(vocab)} tokens", flush=True)

    # Step 1: Scan MathWriting for compatible samples
    print(f"Scanning MathWriting train...", flush=True)
    train_dir = MATHWRITING_DIR / "train"
    if not train_dir.exists():
        print(f"ERROR: {train_dir} not found. Extract mathwriting-2024.tgz first.")
        sys.exit(1)

    compatible = []
    total = 0
    for inkml in sorted(train_dir.glob("*.inkml")):
        total += 1
        strokes, label = parse_inkml(inkml)
        if not label or not strokes:
            continue
        tokens = strict_tokenize(label, commands, singles)
        if tokens:
            compatible.append((inkml, ' '.join(tokens)))
        if total % 50000 == 0:
            print(f"  scanned {total}, compatible {len(compatible)}", flush=True)

    print(f"Total: {total}, Compatible: {len(compatible)} "
          f"({100*len(compatible)/max(total,1):.1f}%)", flush=True)

    # Step 2: Random sample 1000
    random.seed(42)
    if len(compatible) > N_MATHWRITING:
        selected = random.sample(compatible, N_MATHWRITING)
    else:
        selected = compatible
    print(f"Selected {len(selected)} MathWriting samples", flush=True)

    # Step 3: Rasterize selected samples
    print(f"Rasterizing...", flush=True)
    mw_data = []  # (fname, bmp_bytes, caption)
    for i, (inkml_path, caption) in enumerate(selected):
        strokes, _ = parse_inkml(inkml_path)
        img = rasterize(strokes)
        fname = f"mw_{i:05d}"
        import io
        buf = io.BytesIO()
        img.save(buf, format="BMP")
        mw_data.append((fname, buf.getvalue(), caption))
        if (i + 1) % 200 == 0:
            print(f"  rasterized {i+1}/{len(selected)}", flush=True)

    # Step 4: Build mixed zip (CROHME + MathWriting)
    print(f"Building mixed zip...", flush=True)
    with zipfile.ZipFile(CROHME_ZIP) as src, \
         zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_STORED) as dst:

        # Copy all CROHME data
        for item in src.namelist():
            dst.writestr(item, src.read(item))

        # Read existing train caption
        with src.open("data/train/caption.txt") as f:
            crohme_captions = f.read().decode().strip()

        # Add MathWriting images and captions
        mw_captions = []
        for fname, bmp_bytes, caption in mw_data:
            dst.writestr(f"data/train/img/{fname}.bmp", bmp_bytes)
            mw_captions.append(f"{fname}\t{caption}")

        # Overwrite caption.txt with combined captions
        combined = crohme_captions + "\n" + "\n".join(mw_captions) + "\n"
        dst.writestr("data/train/caption.txt", combined)

    size_mb = OUTPUT_ZIP.stat().st_size / 1e6
    print(f"\nDone: {OUTPUT_ZIP} ({size_mb:.0f} MB)", flush=True)
    print(f"  CROHME: ~8835 samples", flush=True)
    print(f"  MathWriting: {len(mw_data)} samples", flush=True)
    print(f"  Total: ~{8835 + len(mw_data)} samples", flush=True)


if __name__ == "__main__":
    main()
