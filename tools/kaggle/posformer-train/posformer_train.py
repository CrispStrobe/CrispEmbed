#!/usr/bin/env python3
"""PosFormer training on MathWriting dataset — Kaggle/RunPod/local script.

Trains PosFormer (DenseNet + Transformer + ARM) on Google's MathWriting
dataset (230K handwritten math images, CC BY-NC-SA 4.0).

Usage:
  # On Kaggle (T4 GPU, free):
  #   Upload as kernel, set GPU accelerator, run.

  # On RunPod (A100):
  #   python posformer_train.py --epochs 200 --batch-size 16

  # On M1 MacBook:
  #   python posformer_train.py --device mps --epochs 200 --batch-size 8

  # Local (CPU, for testing):
  #   python posformer_train.py --device cpu --epochs 1 --batch-size 4

Steps:
  1. Download MathWriting (2.9 GB) from Google Storage
  2. Rasterize InkML strokes → grayscale PNG images
  3. Build vocabulary from normalizedLabel annotations
  4. Clone PosFormer training code from GitHub
  5. Train with PyTorch Lightning
  6. Convert best checkpoint → GGUF
  7. Upload GGUF + checkpoint to HuggingFace
"""

import argparse
import os
import sys
import subprocess
import time
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter

# ── Config ──────────────────────────────────────────────────────────────────

MATHWRITING_URL = "https://storage.googleapis.com/mathwriting_data/mathwriting-2024.tgz"
POSFORMER_REPO = "https://github.com/SJTU-DeepVisionLab/PosFormer.git"
HF_REPO = "cstr/posformer-mathwriting-GGUF"

# Detect environment
ON_KAGGLE = os.path.exists("/kaggle/working")
WORK_DIR = Path("/kaggle/working" if ON_KAGGLE else os.environ.get("WORK_DIR", "/tmp/posformer-train"))
DATA_DIR = WORK_DIR / "mathwriting-2024"
IMAGES_DIR = WORK_DIR / "mathwriting-images"
POSFORMER_DIR = WORK_DIR / "PosFormer"
OUTPUT_DIR = WORK_DIR / "output"


def sh(cmd, **kwargs):
    """Run shell command with live output."""
    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=True, **kwargs)


def download_mathwriting():
    """Download and extract MathWriting dataset."""
    tarball = WORK_DIR / "mathwriting-2024.tgz"
    if DATA_DIR.exists() and len(list(DATA_DIR.glob("train/*.inkml"))) > 1000:
        print(f"MathWriting already extracted at {DATA_DIR}")
        return

    if not tarball.exists():
        print(f"Downloading MathWriting ({MATHWRITING_URL})...")
        sh(f"wget -q --show-progress -O {tarball} {MATHWRITING_URL}")

    print("Extracting...")
    sh(f"tar xzf {tarball} -C {WORK_DIR}")
    print(f"Extracted: {len(list(DATA_DIR.glob('train/*.inkml')))} train samples")


def parse_inkml(path):
    """Parse an InkML file → (strokes, label).

    Returns:
        strokes: list of traces, each trace is list of (x, y) tuples
        label: normalized LaTeX string
    """
    tree = ET.parse(path)
    root = tree.getroot()

    # Handle namespace
    ns = {'ink': 'http://www.w3.org/2003/InkML'}

    # Get label
    label = None
    for ann in root.findall('.//ink:annotation', ns) + root.findall('.//annotation'):
        ann_type = ann.get('type', '')
        if ann_type == 'normalizedLabel':
            label = ann.text
            break
        elif ann_type == 'label' and label is None:
            label = ann.text

    if label is None:
        # Try without namespace
        for ann in root.iter():
            if ann.tag.endswith('annotation'):
                ann_type = ann.get('type', '')
                if ann_type == 'normalizedLabel':
                    label = ann.text
                    break
                elif ann_type == 'label' and label is None:
                    label = ann.text

    # Get strokes
    strokes = []
    for trace in list(root.findall('.//ink:trace', ns)) + list(root.findall('.//trace')):
        text = trace.text.strip()
        points = []
        for pt in text.split(','):
            coords = pt.strip().split()
            if len(coords) >= 2:
                try:
                    x, y = float(coords[0]), float(coords[1])
                    points.append((x, y))
                except ValueError:
                    continue
        if points:
            strokes.append(points)

    return strokes, label


def rasterize_strokes(strokes, target_h=128, line_width=3, padding=10):
    """Rasterize strokes to a grayscale numpy array.

    Returns numpy array [H, W] with values 0.0 (bg) to 1.0 (ink).
    """
    import numpy as np
    from PIL import Image, ImageDraw

    if not strokes:
        return np.zeros((target_h, target_h), dtype=np.float32)

    # Find bounding box
    all_x = [p[0] for s in strokes for p in s]
    all_y = [p[1] for s in strokes for p in s]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    w_range = max_x - min_x + 1e-6
    h_range = max_y - min_y + 1e-6

    # Scale to target height, preserve aspect ratio
    scale = (target_h - 2 * padding) / h_range
    img_w = max(int(w_range * scale + 2 * padding), target_h // 2)
    img_h = target_h

    img = Image.new('L', (img_w, img_h), 255)  # white background
    draw = ImageDraw.Draw(img)

    for stroke in strokes:
        scaled = []
        for x, y in stroke:
            sx = (x - min_x) * scale + padding
            sy = (y - min_y) * scale + padding
            scaled.append((sx, sy))
        if len(scaled) >= 2:
            draw.line(scaled, fill=0, width=line_width)  # black ink
        elif len(scaled) == 1:
            x, y = scaled[0]
            r = line_width // 2
            draw.ellipse([x-r, y-r, x+r, y+r], fill=0)

    return np.array(img, dtype=np.float32) / 255.0


def preprocess_mathwriting(split="train", max_samples=None):
    """Convert InkML files to images + captions file."""
    import numpy as np
    from PIL import Image

    inkml_dir = DATA_DIR / split
    if not inkml_dir.exists():
        print(f"Warning: {inkml_dir} not found")
        return [], []

    out_img_dir = IMAGES_DIR / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    caption_path = IMAGES_DIR / f"{split}_caption.txt"

    # Check if already done
    if caption_path.exists():
        lines = caption_path.read_text().strip().split('\n')
        if len(lines) > 1000:
            print(f"Already preprocessed {split}: {len(lines)} samples")
            return

    inkml_files = sorted(inkml_dir.glob("*.inkml"))
    if max_samples:
        inkml_files = inkml_files[:max_samples]

    print(f"Preprocessing {split}: {len(inkml_files)} InkML files...")

    captions = []
    skipped = 0
    t0 = time.time()

    for i, path in enumerate(inkml_files):
        try:
            strokes, label = parse_inkml(path)
        except Exception as e:
            skipped += 1
            continue

        if not label or not strokes:
            skipped += 1
            continue

        # Rasterize
        img_arr = rasterize_strokes(strokes)

        # Save as BMP (fast, no compression dependency)
        fname = path.stem
        img = Image.fromarray((img_arr * 255).astype(np.uint8), mode='L')
        img.save(out_img_dir / f"{fname}.bmp")

        # Tokenize label (space-separated)
        tokens = label.strip()
        captions.append(f"{fname}\t{tokens}")

        if (i + 1) % 10000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  {i+1}/{len(inkml_files)} ({rate:.0f}/s, skipped {skipped})")

    # Write captions
    with open(caption_path, 'w') as f:
        f.write('\n'.join(captions) + '\n')

    elapsed = time.time() - t0
    print(f"Done: {len(captions)} samples, {skipped} skipped, {elapsed:.1f}s")


def build_vocabulary():
    """Build vocabulary from training captions."""
    caption_path = IMAGES_DIR / "train_caption.txt"
    if not caption_path.exists():
        raise FileNotFoundError(f"No training captions at {caption_path}")

    token_counts = Counter()
    with open(caption_path) as f:
        for line in f:
            parts = line.strip().split('\t', 1)
            if len(parts) < 2:
                continue
            tokens = parts[1].split()
            token_counts.update(tokens)

    # Sort by frequency (most common first)
    sorted_tokens = [t for t, _ in token_counts.most_common()]

    dict_path = IMAGES_DIR / "dictionary.txt"
    with open(dict_path, 'w') as f:
        for token in sorted_tokens:
            f.write(token + '\n')

    print(f"Vocabulary: {len(sorted_tokens)} tokens")
    print(f"Top 20: {sorted_tokens[:20]}")
    print(f"Saved to {dict_path}")
    return sorted_tokens


def setup_posformer_training():
    """Clone PosFormer repo and patch for MathWriting."""
    if not POSFORMER_DIR.exists():
        sh(f"git clone {POSFORMER_REPO} {POSFORMER_DIR}")

    # The training script needs to be adapted to use our preprocessed data
    # PosFormer expects a zip file with images and a caption file
    print("PosFormer training code ready at", POSFORMER_DIR)


def train(args):
    """Run PosFormer training."""
    sys.path.insert(0, str(POSFORMER_DIR))

    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

    # Import PosFormer modules
    from Pos_Former.model.posformer import PosFormer

    # Determine device
    if args.device == "auto":
        if torch.cuda.is_available():
            accelerator = "gpu"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            accelerator = "mps"
        else:
            accelerator = "cpu"
    else:
        accelerator = args.device

    print(f"Training on: {accelerator}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")

    # TODO: Create a custom DataModule that loads from IMAGES_DIR
    # For now, we create a CROHME-compatible zip structure
    # that PosFormer's existing DataModule can load

    print("\n" + "="*60)
    print("Training pipeline ready. Next steps:")
    print("1. Create CROHME-compatible data loader for MathWriting images")
    print("2. Run training with PosFormer Lightning module")
    print("3. Convert best checkpoint to GGUF")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="PosFormer MathWriting training")
    parser.add_argument("--device", default="auto", choices=["auto", "gpu", "mps", "cpu"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit training samples (for testing)")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-preprocess", action="store_true")
    args = parser.parse_args()

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Download
    if not args.skip_download:
        download_mathwriting()

    # Step 2: Preprocess
    if not args.skip_preprocess:
        preprocess_mathwriting("train", max_samples=args.max_samples)
        preprocess_mathwriting("valid")
        preprocess_mathwriting("test")

    # Step 3: Build vocabulary
    vocab = build_vocabulary()

    # Step 4: Setup training code
    setup_posformer_training()

    # Step 5: Train
    train(args)


if __name__ == "__main__":
    main()
