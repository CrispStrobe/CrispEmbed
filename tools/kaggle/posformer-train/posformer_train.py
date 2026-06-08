#!/usr/bin/env python3
"""PosFormer training on MathWriting dataset — Kaggle/RunPod/local.

Trains PosFormer (DenseNet + Transformer + ARM) on Google's MathWriting
dataset (230K handwritten math images, CC BY-NC-SA 4.0).

Usage:
  # Kaggle (T4 GPU, free tier):
  #   Upload as kernel, set GPU accelerator, run.

  # RunPod A100 ($1.50/hr):
  #   python posformer_train.py --epochs 200 --batch-size 16

  # M1 MacBook 16GB:
  #   python posformer_train.py --device mps --epochs 200 --batch-size 8

  # Quick test (CPU, 100 samples):
  #   python posformer_train.py --device cpu --epochs 2 --batch-size 4 --max-samples 100
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
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

# ── Environment detection ───────────────────────────────────────────────────

ON_KAGGLE = os.path.exists("/kaggle/working")
WORK_DIR = Path(
    "/kaggle/working" if ON_KAGGLE
    else os.environ.get("WORK_DIR", "/mnt/volume1/posformer-training")
)
DATA_DIR = WORK_DIR / "mathwriting-2024"
IMAGES_DIR = WORK_DIR / "mathwriting-images"
OUTPUT_DIR = WORK_DIR / "output"

MATHWRITING_URL = "https://storage.googleapis.com/mathwriting_data/mathwriting-2024.tgz"
POSFORMER_REPO = "https://github.com/SJTU-DeepVisionLab/PosFormer.git"
CRISPEMBED_REPO = "https://github.com/CrispStrobe/CrispEmbed.git"

# HF progress monitoring (same pattern as CrispASR kaggle harness)
HF_PROGRESS_REPO = "cstr/posformer-training-progress"
HF_CHECKPOINT_REPO = "cstr/posformer-mathwriting-GGUF"

# ── Utilities ───────────────────────────────────────────────────────────────

_T0 = time.time()
_PROGRESS_FILE = None


def log(msg):
    elapsed = time.time() - _T0
    line = f"[{elapsed:7.1f}s] {msg}"
    print(line, flush=True)
    if _PROGRESS_FILE:
        _PROGRESS_FILE.write(json.dumps({
            "t": round(elapsed, 1), "msg": msg,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ")
        }) + "\n")
        _PROGRESS_FILE.flush()


def sh(cmd, **kwargs):
    log(f"$ {cmd}")
    return subprocess.run(cmd, shell=True, check=True, **kwargs)


# ── Step 1: Download MathWriting ────────────────────────────────────────────

def download_mathwriting():
    tarball = WORK_DIR / "mathwriting-2024.tgz"
    if DATA_DIR.exists() and len(list(DATA_DIR.glob("train/*.inkml"))) > 1000:
        log(f"MathWriting already at {DATA_DIR}")
        return
    if not tarball.exists():
        log(f"Downloading MathWriting (2.9 GB)...")
        sh(f"wget -q --show-progress -O {tarball} '{MATHWRITING_URL}'")
    log("Extracting...")
    sh(f"tar xzf {tarball} -C {WORK_DIR}")
    n = len(list(DATA_DIR.glob("train/*.inkml")))
    log(f"Extracted: {n} train InkML files")


# ── Step 2: InkML → rasterized images ──────────────────────────────────────

def parse_inkml(path):
    """Parse InkML → (strokes, normalizedLabel)."""
    tree = ET.parse(path)
    root = tree.getroot()
    # Handle namespace variants
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
    """Strokes → grayscale uint8 PIL Image (white bg, black ink)."""
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


def preprocess_split(split, max_samples=None):
    """Convert one split of InkML → BMP images + caption.txt."""
    inkml_dir = DATA_DIR / split
    if not inkml_dir.exists():
        log(f"Warning: {inkml_dir} not found, skipping")
        return 0
    out_dir = IMAGES_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)
    cap_path = IMAGES_DIR / f"{split}_caption.txt"

    if cap_path.exists():
        n = sum(1 for _ in open(cap_path))
        if n > 100:
            log(f"{split}: already done ({n} samples)")
            return n

    files = sorted(inkml_dir.glob("*.inkml"))
    if max_samples:
        files = files[:max_samples]

    captions = []
    skipped = 0
    t0 = time.time()

    for i, p in enumerate(files):
        try:
            strokes, label = parse_inkml(p)
        except Exception:
            skipped += 1
            continue
        if not label or not strokes:
            skipped += 1
            continue
        img = rasterize(strokes)
        img.save(out_dir / f"{p.stem}.bmp")
        captions.append(f"{p.stem}\t{label.strip()}")
        if (i + 1) % 10000 == 0:
            log(f"  {split}: {i+1}/{len(files)} ({(i+1)/(time.time()-t0):.0f}/s)")

    cap_path.write_text("\n".join(captions) + "\n")
    log(f"{split}: {len(captions)} ok, {skipped} skipped, {time.time()-t0:.1f}s")
    return len(captions)


# ── Step 3: Build vocabulary ────────────────────────────────────────────────

def build_vocab():
    cap_path = IMAGES_DIR / "train_caption.txt"
    counts = Counter()
    with open(cap_path) as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                counts.update(parts[1].split())
    tokens = [t for t, _ in counts.most_common()]
    dict_path = IMAGES_DIR / "dictionary.txt"
    dict_path.write_text("\n".join(tokens) + "\n")
    log(f"Vocabulary: {len(tokens)} tokens → {dict_path}")
    return tokens


# ── Step 4: Create CROHME-compatible zip ────────────────────────────────────

def create_data_zip():
    """Pack preprocessed images into a zip that PosFormer can load."""
    import zipfile
    zip_path = WORK_DIR / "data_mathwriting.zip"
    if zip_path.exists() and zip_path.stat().st_size > 1_000_000:
        log(f"Data zip exists: {zip_path}")
        return zip_path

    log("Creating data zip...")
    t0 = time.time()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for split in ["train", "valid", "test"]:
            cap_path = IMAGES_DIR / f"{split}_caption.txt"
            if not cap_path.exists():
                continue
            # Map split name for PosFormer compatibility
            folder = split if split == "train" else split
            zf.write(cap_path, f"data/{folder}/caption.txt")

            img_dir = IMAGES_DIR / split
            with open(cap_path) as f:
                for line in f:
                    fname = line.strip().split("\t")[0]
                    bmp = img_dir / f"{fname}.bmp"
                    if bmp.exists():
                        zf.write(bmp, f"data/{folder}/img/{fname}.bmp")

    log(f"Zip: {zip_path} ({zip_path.stat().st_size / 1e6:.0f} MB, {time.time()-t0:.0f}s)")
    return zip_path


# ── Step 5: Train ───────────────────────────────────────────────────────────

def train(args, zip_path):
    """Run PosFormer training with hourly checkpoints + HF monitoring."""
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

    # Clone PosFormer if needed
    posformer_dir = WORK_DIR / "PosFormer"
    if not posformer_dir.exists():
        sh(f"git clone --depth 1 {POSFORMER_REPO} {posformer_dir}")

    sys.path.insert(0, str(posformer_dir))

    # Patch the dictionary path to use our vocabulary
    dict_path = IMAGES_DIR / "dictionary.txt"
    vocab_module = posformer_dir / "Pos_Former" / "datamodule" / "dictionary.txt"
    import shutil
    shutil.copy(dict_path, vocab_module)
    log(f"Patched dictionary: {dict_path} → {vocab_module}")

    from Pos_Former.datamodule import CROHMEDatamodule
    from Pos_Former.lit_posformer import LitPosFormer

    # Determine accelerator
    if args.device == "auto":
        if torch.cuda.is_available():
            accelerator, devices = "gpu", 1
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            accelerator, devices = "mps", 1
        else:
            accelerator, devices = "cpu", "auto"
    elif args.device == "mps":
        accelerator, devices = "mps", 1
    elif args.device == "gpu":
        accelerator, devices = "gpu", 1
    else:
        accelerator, devices = "cpu", "auto"

    log(f"Accelerator: {accelerator}, epochs: {args.epochs}, batch: {args.batch_size}")

    # Datamodule
    dm = CROHMEDatamodule(
        zipfile_path=str(zip_path),
        test_year="valid",  # use valid split for eval
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        num_workers=min(4, os.cpu_count() or 1),
        scale_aug=True,
    )

    # Model
    model = LitPosFormer(
        d_model=256,
        growth_rate=24,
        num_layers=16,
        nhead=8,
        num_decoder_layers=3,
        dim_feedforward=1024,
        dropout=0.3,
        dc=32,
        cross_coverage=True,
        self_coverage=True,
        beam_size=10,
        max_len=200,
        alpha=1.0,
        early_stopping=False,
        temperature=1.0,
        learning_rate=0.08,
        patience=20,
    )

    # Callbacks
    ckpt_best = ModelCheckpoint(
        dirpath=str(OUTPUT_DIR / "checkpoints"),
        monitor="val_ExpRate",
        mode="max",
        save_top_k=3,
        filename="{epoch}-{step}-{val_ExpRate:.4f}",
    )
    ckpt_hourly = ModelCheckpoint(
        dirpath=str(OUTPUT_DIR / "checkpoints"),
        every_n_train_steps=2000,  # ~1 hour on T4
        save_top_k=-1,  # keep all
        filename="hourly-{epoch}-{step}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Trainer
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=args.epochs,
        callbacks=[ckpt_best, ckpt_hourly, lr_monitor],
        default_root_dir=str(OUTPUT_DIR),
        log_every_n_steps=50,
        val_check_interval=0.5,  # validate twice per epoch
        gradient_clip_val=1.0,
    )

    log("Starting training...")
    trainer.fit(model, dm)
    log(f"Best model: {ckpt_best.best_model_path}")
    log(f"Best val_ExpRate: {ckpt_best.best_model_score}")

    return ckpt_best.best_model_path


# ── Step 6: Convert checkpoint → GGUF ──────────────────────────────────────

def convert_to_gguf(ckpt_path):
    """Convert best checkpoint to GGUF using CrispEmbed converter."""
    crispembed_dir = WORK_DIR / "CrispEmbed"
    if not crispembed_dir.exists():
        sh(f"git clone --depth 1 -b feat/posformer-port {CRISPEMBED_REPO} {crispembed_dir}")

    dict_path = IMAGES_DIR / "dictionary.txt"
    gguf_path = OUTPUT_DIR / "posformer-mathwriting-f32.gguf"

    sh(f"python {crispembed_dir}/models/convert-posformer-to-gguf.py "
       f"--checkpoint {ckpt_path} --dict {dict_path} --output {gguf_path}")

    log(f"GGUF: {gguf_path} ({gguf_path.stat().st_size / 1e6:.1f} MB)")
    return gguf_path


# ── Step 7: Upload to HuggingFace ───────────────────────────────────────────

def upload_to_hf(gguf_path, ckpt_path):
    """Upload GGUF and checkpoint to HuggingFace."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(gguf_path),
            path_in_repo=gguf_path.name,
            repo_id=HF_CHECKPOINT_REPO,
            repo_type="model",
        )
        log(f"Uploaded {gguf_path.name} to {HF_CHECKPOINT_REPO}")
    except Exception as e:
        log(f"HF upload failed: {e}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    global _PROGRESS_FILE

    parser = argparse.ArgumentParser(description="PosFormer MathWriting training")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "gpu", "mps", "cpu"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Resume from checkpoint")
    args = parser.parse_args()

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Progress log
    progress_path = OUTPUT_DIR / "progress.jsonl"
    _PROGRESS_FILE = open(progress_path, "a")

    log("=== PosFormer MathWriting Training ===")
    log(f"Work dir: {WORK_DIR}")
    log(f"Device: {args.device}, Epochs: {args.epochs}, Batch: {args.batch_size}")

    # 1. Download
    if not args.skip_download:
        download_mathwriting()

    # 2. Preprocess
    if not args.skip_preprocess:
        preprocess_split("train", max_samples=args.max_samples)
        preprocess_split("valid")
        preprocess_split("test")

    # 3. Vocabulary
    build_vocab()

    # 4. Create data zip
    zip_path = create_data_zip()

    # 5. Train
    if not args.skip_train:
        best_ckpt = train(args, zip_path)

        # 6. Convert to GGUF
        if best_ckpt:
            gguf_path = convert_to_gguf(best_ckpt)

            # 7. Upload
            upload_to_hf(gguf_path, best_ckpt)

    log("=== Done ===")
    _PROGRESS_FILE.close()


if __name__ == "__main__":
    main()
