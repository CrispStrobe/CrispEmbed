#!/usr/bin/env python3
"""PosFormer training on CROHME / MathWriting — Kaggle T4 / RunPod / local.

Trains PosFormer (DenseNet + Transformer + ARM, 6.5M params) with:
  - Hourly checkpoint upload to HuggingFace (survives Kaggle crashes)
  - W&B monitoring with cross-session resume
  - Multi-session training (auto-resume from HF checkpoint)
  - Dynamic vocab (no hardcoded token indices)

Modes (set DATASET env var):
  crohme       — 200 epochs on CROHME 2014 train (~25h, 3 Kaggle sessions)
  mathwriting  — 20 epochs on MathWriting 230K (~27h, 3 sessions)
  finetune     — load MathWriting checkpoint, finetune 50 epochs on CROHME

Timing on Kaggle T4 (16 GB VRAM):
  CROHME:      ~900 batches/epoch, ~3 min/epoch, 200ep ≈ 25h
  MathWriting: ~28K batches/epoch, ~30 min/epoch, 20ep ≈ 27h
  Finetune:    same as CROHME but fewer epochs

Usage:
  # Kaggle — set env vars in kernel UI "Add-ons → Variables":
  #   DATASET=crohme  (or mathwriting, finetune)
  #   Attach dataset: chr1str/crispasr-hf-token (has HF + W&B tokens)

  # Local quick test:
  #   python posformer_train.py --dataset crohme --device cpu --epochs 2 --max-samples 100

  # RunPod A100:
  #   python posformer_train.py --dataset mathwriting --epochs 50 --batch-size 16

License: the training script itself is MIT. Trained weights inherit
CC BY-NC-SA 4.0 from the training data (CROHME, MathWriting).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

# ━━━━━━━━━━━━━━━━━━━━ Environment detection ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ON_KAGGLE = os.path.exists("/kaggle/working")
WORK = Path("/kaggle/working" if ON_KAGGLE else
            os.environ.get("WORK_DIR", "/mnt/volume1/posformer-training"))

HF_MODEL_REPO = "cstr/posformer-mathwriting-GGUF"
HF_CHECKPOINT_REPO = "cstr/posformer-training-checkpoints"
HF_PROGRESS_REPO = "cstr/posformer-training-progress"
WANDB_PROJECT = "posformer-hmer"

MATHWRITING_URL = "https://storage.googleapis.com/mathwriting_data/mathwriting-2024.tgz"
POSFORMER_REPO = "https://github.com/SJTU-DeepVisionLab/PosFormer.git"
CRISPEMBED_REPO = "https://github.com/CrispStrobe/CrispEmbed.git"

# Where CROHME data zip lives (upload to HF or Kaggle dataset).
# PosFormer's expected format: data/{train,2014}/caption.txt + img/*.bmp
CROHME_HF_REPO = "cstr/posformer-training-data"
CROHME_HF_FILE = "data_crohme.zip"

# ━━━━━━━━━━━━━━━━━━━━ Progress / logging ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_T0 = time.time()
_PROGRESS_PATH: Path = WORK / "progress.jsonl"
_HF_PUSH_INTERVAL = 30.0
_HF_LAST_PUSH = 0.0


def step(name: str, **extra) -> None:
    """Append checkpoint to local JSONL + best-effort push to HF."""
    global _HF_LAST_PUSH
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_s": round(time.time() - _T0, 2),
        "step": name, **extra,
    }
    try:
        _PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _PROGRESS_PATH.open("a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass
    print(f"[step {rec['elapsed_s']:>7.1f}s] {name}"
          + (f"  {extra}" if extra else ""), flush=True)
    # Best-effort HF push
    now = time.time()
    if (now - _HF_LAST_PUSH) >= _HF_PUSH_INTERVAL and os.environ.get("HF_TOKEN"):
        try:
            from huggingface_hub import HfApi
            run_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            HfApi(token=os.environ["HF_TOKEN"]).upload_file(
                path_or_fileobj=str(_PROGRESS_PATH),
                path_in_repo=f"runs/{run_tag}.jsonl",
                repo_id=HF_PROGRESS_REPO, repo_type="dataset",
                commit_message=f"progress @ {now - _T0:.0f}s",
            )
            _HF_LAST_PUSH = now
        except Exception:
            pass


def sh(cmd: str, **kwargs) -> subprocess.CompletedProcess:
    step(f"$ {cmd}")
    return subprocess.run(cmd, shell=True, check=True, **kwargs)


# ━━━━━━━━━━━━━━━━━━━━ Auth — CrispASR kaggle_harness ━━━━━━━━━━━━━━━━━━━━━━

# ── Clone CrispASR + import kaggle_harness ───────────────────────────────
# Same pattern as all CrispASR kernels: clone the repo, import the harness.

CRISPASR_REPO_URL = "https://github.com/CrispStrobe/CrispASR.git"
_WORK = Path("/kaggle/working" if os.path.exists("/kaggle/working")
             else os.environ.get("WORK_DIR", "/mnt/volume1/posformer-training"))
_CRISPASR = _WORK / "CrispASR"

if not _CRISPASR.exists():
    subprocess.check_call([
        "git", "clone", "--depth", "1",
        CRISPASR_REPO_URL, str(_CRISPASR),
    ])

sys.path.insert(0, str(_CRISPASR / "tools" / "kaggle"))
import kaggle_harness as kh          # noqa: E402
kh.init_progress()                   # line-buffered I/O + JSONL


def resolve_tokens() -> dict:
    """Resolve HF_TOKEN and WANDB_API_KEY via kaggle_harness 3-tier auth."""
    tokens = {}

    # HF: env → Kaggle Secret (3 retries) → dataset file
    hf = kh.resolve_hf_token()
    if hf:
        tokens["hf"] = True

    # W&B: same 3-tier pattern
    wb = (os.environ.get("WANDB_API_KEY")
          or kh.kaggle_secret("WANDB_API_KEY")
          or kh.kaggle_token_from_dataset("wandb_api_key.txt"))
    if wb:
        os.environ["WANDB_API_KEY"] = wb
        tokens["wandb"] = True
        print("W&B auth: OK", flush=True)
    else:
        print("W&B auth: none (offline mode)", flush=True)

    return tokens


# ━━━━━━━━━━━━━━━━━━━━ Data: CROHME ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def download_crohme() -> Path:
    """Download CROHME data zip from HF."""
    zip_path = WORK / "data_crohme.zip"
    if zip_path.exists() and zip_path.stat().st_size > 1_000_000:
        step("crohme.cached", size_mb=round(zip_path.stat().st_size / 1e6))
        return zip_path

    step("crohme.download")
    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id=CROHME_HF_REPO, filename=CROHME_HF_FILE,
            repo_type="dataset", local_dir=str(WORK),
        )
        if Path(downloaded) != zip_path:
            shutil.copy2(downloaded, zip_path)
    except Exception as e:
        # Fallback: check if it's a Kaggle dataset
        kaggle_path = Path("/kaggle/input/crohme-data/data_crohme.zip")
        if kaggle_path.exists():
            shutil.copy2(kaggle_path, zip_path)
        else:
            raise RuntimeError(
                f"Cannot find CROHME data. Upload data_crohme.zip to "
                f"HF dataset {CROHME_HF_REPO} or Kaggle dataset 'crohme-data'. "
                f"Original error: {e}"
            )

    step("crohme.ready", size_mb=round(zip_path.stat().st_size / 1e6))
    return zip_path


# ━━━━━━━━━━━━━━━━━━━━ Data: MathWriting ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def download_mathwriting() -> Path:
    """Download MathWriting from Google + rasterize InkML → BMP zip."""
    zip_path = WORK / "data_mathwriting.zip"
    if zip_path.exists() and zip_path.stat().st_size > 100_000_000:
        step("mathwriting.cached", size_mb=round(zip_path.stat().st_size / 1e6))
        return zip_path

    # Try pre-built zip from HF first (saves 25 min of rasterization)
    step("mathwriting.check_hf_cache")
    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id=CROHME_HF_REPO, filename="data_mathwriting.zip",
            repo_type="dataset", local_dir=str(WORK),
        )
        if Path(downloaded).stat().st_size > 100_000_000:
            if Path(downloaded) != zip_path:
                shutil.move(downloaded, zip_path)
            step("mathwriting.from_hf", size_mb=round(zip_path.stat().st_size / 1e6))
            return zip_path
    except Exception:
        pass

    # Download raw MathWriting and rasterize
    data_dir = WORK / "mathwriting-2024"
    tarball = WORK / "mathwriting-2024.tgz"

    if not data_dir.exists() or len(list(data_dir.glob("train_*.inkml"))) < 1000:
        if not tarball.exists():
            step("mathwriting.download", note="2.9 GB")
            sh(f"wget -q --show-progress -O {tarball} '{MATHWRITING_URL}'")
        step("mathwriting.extract")
        sh(f"tar xzf {tarball} -C {WORK}")

    # Rasterize all splits
    images_dir = WORK / "mathwriting-images"
    split_map = {"train": "train", "valid": "valid", "test": "test"}
    all_captions: Dict[str, List[str]] = {}

    for split_name, folder in split_map.items():
        cap_lines = _rasterize_split(data_dir, images_dir, split_name)
        if cap_lines:
            all_captions[split_name] = cap_lines

    # Pack into PosFormer-compatible zip
    step("mathwriting.pack_zip")
    _pack_zip(zip_path, images_dir, all_captions)
    step("mathwriting.ready", size_mb=round(zip_path.stat().st_size / 1e6))

    # Upload rasterized zip to HF for future runs
    if os.environ.get("HF_TOKEN"):
        step("mathwriting.upload_cache")
        try:
            from huggingface_hub import HfApi
            HfApi(token=os.environ["HF_TOKEN"]).upload_file(
                path_or_fileobj=str(zip_path),
                path_in_repo="data_mathwriting.zip",
                repo_id=CROHME_HF_REPO, repo_type="dataset",
                commit_message="Pre-rasterized MathWriting data zip",
            )
        except Exception as e:
            print(f"  HF cache upload failed (non-fatal): {e}", flush=True)

    return zip_path


def _parse_inkml(path: Path) -> Tuple[list, Optional[str]]:
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


def _rasterize(strokes: list, target_h: int = 128,
               line_w: int = 3, pad: int = 10) -> Image.Image:
    """Strokes → grayscale PIL Image (white bg, black ink)."""
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
            draw.ellipse([x - r, y - r, x + r, y + r], fill=0)
    return img


def _rasterize_split(data_dir: Path, images_dir: Path,
                     split: str, max_samples: int = 0) -> List[str]:
    """Rasterize one split, return caption lines."""
    # MathWriting stores all InkML in flat dirs named train_*.inkml etc.
    # or in subdirectories — handle both layouts
    inkml_files = sorted(data_dir.glob(f"{split}/*.inkml"))
    if not inkml_files:
        inkml_files = sorted(data_dir.glob(f"{split}_*.inkml"))
    if not inkml_files:
        inkml_files = sorted(data_dir.glob(f"**/{split}/*.inkml"))
    if not inkml_files:
        step(f"rasterize.{split}.skip", note="no inkml files found")
        return []

    out_dir = images_dir / split
    cap_path = images_dir / f"{split}_caption.txt"

    # Resume check
    if cap_path.exists():
        existing = sum(1 for _ in open(cap_path))
        if existing > 100:
            step(f"rasterize.{split}.cached", n=existing)
            return cap_path.read_text().strip().split("\n")

    out_dir.mkdir(parents=True, exist_ok=True)
    if max_samples:
        inkml_files = inkml_files[:max_samples]

    captions = []
    skipped = 0
    t0 = time.time()

    for i, p in enumerate(inkml_files):
        try:
            strokes, label = _parse_inkml(p)
        except Exception:
            skipped += 1
            continue
        if not label or not strokes:
            skipped += 1
            continue
        # Tokenize label (space-separated for PosFormer)
        label = label.strip()
        img = _rasterize(strokes)
        img.save(out_dir / f"{p.stem}.bmp")
        captions.append(f"{p.stem}\t{label}")
        if (i + 1) % 10000 == 0:
            rate = (i + 1) / (time.time() - t0)
            step(f"rasterize.{split}.progress",
                 done=i + 1, total=len(inkml_files), rate=f"{rate:.0f}/s")

    cap_path.write_text("\n".join(captions) + "\n")
    step(f"rasterize.{split}.done",
         ok=len(captions), skipped=skipped,
         time_s=round(time.time() - t0, 1))
    return captions


def _pack_zip(zip_path: Path, images_dir: Path,
              captions: Dict[str, List[str]]) -> None:
    """Pack rasterized images into PosFormer-compatible zip."""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for split, lines in captions.items():
            # Write caption.txt
            # PosFormer expects: "image_name token1 token2 ..."
            # Our format is: "image_name\ttoken1 token2 ..."
            # Convert tab → space for PosFormer compatibility
            converted = []
            for line in lines:
                parts = line.strip().split("\t", 1)
                if len(parts) == 2:
                    converted.append(f"{parts[0]} {parts[1]}")
            cap_content = "\n".join(converted) + "\n"
            zf.writestr(f"data/{split}/caption.txt", cap_content)

            img_dir = images_dir / split
            for line in lines:
                fname = line.strip().split("\t")[0]
                bmp = img_dir / f"{fname}.bmp"
                if bmp.exists():
                    zf.write(bmp, f"data/{split}/img/{fname}.bmp")


# ━━━━━━━━━━━━━━━━━━━━ Vocabulary ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_vocab_from_zip(zip_path: Path) -> Tuple[List[str], Path]:
    """Extract vocabulary from training captions in the data zip.
    Returns (token_list, dict_file_path)."""
    counts: Counter = Counter()
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.endswith("caption.txt") and "/train/" in name:
                with zf.open(name) as f:
                    for line in f:
                        parts = line.decode().strip().split()
                        if len(parts) >= 2:
                            counts.update(parts[1:])

    tokens = [t for t, _ in counts.most_common()]
    dict_path = WORK / "dictionary.txt"
    dict_path.write_text("\n".join(tokens) + "\n")
    step("vocab.built", n_tokens=len(tokens), path=str(dict_path))
    return tokens, dict_path


# ━━━━━━━━━━━━━━━━━━━━ label_make_muti — vocab-aware rewrite ━━━━━━━━━━━━━━━
#
# The original label_make_muti.py hardcodes CROHME token indices:
#   42=[, 81=], 110={, 112=}, 82=^, 83=_, 53=\frac, 74=\sqrt
# These break on any vocabulary change. We rewrite the core functions
# to look up indices from the actual vocab at init time.

def make_dynamic_label_module(vocab_obj):
    """Create a module-compatible object with vocab-aware label functions.
    Monkey-patches Pos_Former.datamodule.label_make_muti in-place."""
    import types
    import Pos_Former.datamodule.label_make_muti as lm

    w2i = vocab_obj.word2idx
    # Look up structural token indices (use -999 for missing → no match)
    IDX_LBRACKET = w2i.get("[", -999)
    IDX_RBRACKET = w2i.get("]", -999)
    IDX_LBRACE = w2i.get("{", -999)
    IDX_RBRACE = w2i.get("}", -999)
    IDX_CARET = w2i.get("^", -999)
    IDX_UNDERSCORE = w2i.get("_", -999)
    IDX_FRAC = w2i.get("\\frac", -999)
    IDX_SQRT = w2i.get("\\sqrt", -999)
    IDX_PAD = 0
    IDX_SOS = 1
    IDX_EOS = 2

    def find_end_midbracket(indices, start_i, end):
        count = 1
        i = start_i + 1
        while count > 0 and i < end:
            if indices[i] == IDX_LBRACKET:
                count += 1
            elif indices[i] == IDX_RBRACKET:
                count -= 1
            i += 1
        return i - 1 if count == 0 else 0

    def find_end_bigbracket(indices, start_i, end):
        count = 1
        i = start_i + 1
        while count > 0 and i < end:
            if indices[i] == IDX_LBRACE:
                count += 1
            elif indices[i] == IDX_RBRACE:
                count -= 1
            i += 1
        return i - 1 if count == 0 else 0

    def helper(indices, start, end, result):
        flag = [0, 1, 2, 3, 4, 5]
        special = True
        i = start + 1
        while i < end:
            if indices[i] == IDX_CARET:
                end1 = find_end_bigbracket(indices, i + 1, end)
                if special:
                    result[i].append(flag[3])
                    result[i + 1].append(flag[3])
                    result[end1].append(flag[3])
                for j in range(i + 2, end1):
                    result[j].append(flag[4])
                result = helper(indices, i + 1, end1, result)
                i = end1 + 1
            elif indices[i] == IDX_UNDERSCORE:
                end1 = find_end_bigbracket(indices, i + 1, end)
                if special:
                    result[i].append(flag[3])
                    result[i + 1].append(flag[3])
                    result[end1].append(flag[3])
                for j in range(i + 2, end1):
                    result[j].append(flag[5])
                result = helper(indices, i + 1, end1, result)
                i = end1 + 1
            elif indices[i] == IDX_FRAC:
                result[i].append(flag[3])
                end1 = find_end_bigbracket(indices, i + 1, end)
                for j in range(i + 2, end1):
                    result[j].append(flag[4])
                end2 = find_end_bigbracket(indices, end1 + 1, end)
                for j in range(end1 + 2, end2):
                    result[j].append(flag[5])
                if special:
                    result[i + 1].append(flag[3])
                    result[end1].append(flag[3])
                    result[end1 + 1].append(flag[3])
                    result[end2].append(flag[3])
                result = helper(indices, i + 1, end1, result)
                result = helper(indices, end1 + 1, end2, result)
                i = end2 + 1
            elif indices[i] == IDX_SQRT:
                result[i].append(flag[3])
                if indices[i + 1] == IDX_LBRACKET:
                    end1 = find_end_midbracket(indices, i + 1, end)
                    for j in range(i + 2, end1):
                        result[j].append(flag[4])
                    end2 = find_end_bigbracket(indices, end1 + 1, end)
                    for j in range(end1 + 2, end2):
                        result[j].append(flag[5])
                    if special:
                        result[i + 1].append(flag[3])
                        result[end1].append(flag[3])
                        result[end1 + 1].append(flag[3])
                        result[end2].append(flag[3])
                    result = helper(indices, i + 1, end1, result)
                    result = helper(indices, end1 + 1, end2, result)
                    i = end2 + 1
                else:
                    end1 = find_end_bigbracket(indices, i + 1, end)
                    for j in range(i + 2, end1):
                        result[j].append(flag[5])
                    if special:
                        result[i + 1].append(flag[3])
                        result[end1].append(flag[3])
                    result = helper(indices, i + 1, end1, result)
                    i = end1 + 1
            elif indices[i] == IDX_PAD:
                result[i].append(flag[0])
                i += 1
            elif indices[i] == IDX_SOS:
                result[i].append(flag[1])
                i += 1
            elif indices[i] == IDX_EOS:
                result[i].append(flag[2])
                i += 1
            else:
                result[i].append(flag[3])
                i += 1
        return result

    def indices2muti_label(indices):
        result = [[] for _ in range(len(indices))]
        is_reverse = False
        if indices[0] == IDX_EOS:
            indices.reverse()
            is_reverse = True
        result = helper(indices, -1, len(indices), result)
        if is_reverse:
            result.reverse()
        return result

    # Patch all the public functions
    lm.find_end_midbracket = find_end_midbracket
    lm.find_end_bigbracket = find_end_bigbracket
    lm.helper = helper
    lm.indices2muti_label = indices2muti_label

    # Rewrite tgt/out functions that call indices2muti_label
    def tgt2layernum_and_pos(tgt):
        layer_num = []
        final_pos = []
        for indices in tgt:
            layer_num_sub = []
            final_pos_sub = []
            label = indices2muti_label(indices)
            for i in range(len(label)):
                if len(label[i]) == 1:
                    layer_num_sub.append(0)
                    final_pos_sub.append(label[i][0])
                elif len(label[i]) <= 5:
                    layer_num_sub.append(len(label[i]) - 1)
                    final_pos_sub.append(label[i][-2])
                else:
                    layer_num_sub.append(4)
                    final_pos_sub.append(label[i][3])
            layer_num.append(layer_num_sub)
            final_pos.append(final_pos_sub)
        return layer_num, final_pos

    def out2layernum_and_pos(tgt):
        layer_num = []
        final_pos = []
        for indices in tgt:
            layer_num_sub = []
            final_pos_sub = []
            label = indices2muti_label(indices)
            label.append(label[0])
            label.remove(label[0])
            for i in range(len(label)):
                if len(label[i]) == 1:
                    layer_num_sub.append(0)
                    final_pos_sub.append(label[i][0])
                elif len(label[i]) <= 5:
                    layer_num_sub.append(len(label[i]) - 1)
                    final_pos_sub.append(label[i][-2])
                else:
                    layer_num_sub.append(4)
                    final_pos_sub.append(label[i][3])
            layer_num.append(layer_num_sub)
            final_pos.append(final_pos_sub)
        return layer_num, final_pos

    lm.tgt2layernum_and_pos = tgt2layernum_and_pos
    lm.out2layernum_and_pos = out2layernum_and_pos

    step("label_make_muti.patched",
         bracket=f"[={IDX_LBRACKET} ]={IDX_RBRACKET}",
         brace=f"{{={IDX_LBRACE} }}={IDX_RBRACE}",
         struct=f"^={IDX_CARET} _={IDX_UNDERSCORE} "
                f"frac={IDX_FRAC} sqrt={IDX_SQRT}")


# ━━━━━━━━━━━━━━━━━━━━ Patch LitPosFormer for device-agnostic training ━━━━━

def patch_lit_posformer():
    """Fix .cuda() calls in training/validation steps → use self.device."""
    import Pos_Former.lit_posformer as lit

    _orig_training_step = lit.LitPosFormer.training_step
    _orig_validation_step = lit.LitPosFormer.validation_step

    def _patched_training_step(self, batch, batch_idx):
        import torch
        from Pos_Former.datamodule import label_make_muti
        from Pos_Former.utils.utils import to_bi_tgt_out, ce_loss_all

        tgt, out = to_bi_tgt_out(batch.indices, self.device)
        out_hat, out_hat_layer, out_hat_pos = self(
            batch.imgs, batch.mask, tgt, self.trainer.logger)
        tgt_list = tgt.cpu().numpy().tolist()
        layer_num, final_pos = label_make_muti.out2layernum_and_pos(tgt_list)
        layer_num_tensor = torch.LongTensor(layer_num).to(self.device)
        final_pos_tensor = torch.LongTensor(final_pos).to(self.device)
        loss, layer_loss, pos_loss = ce_loss_all(
            out_hat, out, out_hat_layer, layer_num_tensor,
            out_hat_pos, final_pos_tensor)
        self.log("train_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        self.log("train_loss_pos", pos_loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        self.log("train_loss_layernum", layer_loss, on_step=False,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        loss = (loss + 0.25 * layer_loss + 0.25 * pos_loss) / 1.5
        return loss

    def _patched_validation_step(self, batch, batch_idx):
        import torch
        from Pos_Former.datamodule import label_make_muti
        from Pos_Former.utils.utils import to_bi_tgt_out, ce_loss_all

        tgt, out = to_bi_tgt_out(batch.indices, self.device)
        out_hat, out_hat_layer, out_hat_pos = self(
            batch.imgs, batch.mask, tgt, self.trainer.logger)
        tgt_list = tgt.cpu().numpy().tolist()
        layer_num, final_pos = label_make_muti.out2layernum_and_pos(tgt_list)
        layer_num_tensor = torch.LongTensor(layer_num).to(self.device)
        final_pos_tensor = torch.LongTensor(final_pos).to(self.device)
        loss, layer_loss, pos_loss = ce_loss_all(
            out_hat, out, out_hat_layer, layer_num_tensor,
            out_hat_pos, final_pos_tensor)
        self.log("val_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        self.log("val_loss_pos", pos_loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        self.log("val_loss_layernum", layer_loss, on_step=False,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        hyps = self.approximate_joint_search(batch.imgs, batch.mask)
        self.exprate_recorder([h.seq for h in hyps], batch.indices)
        self.log("val_ExpRate", self.exprate_recorder, prog_bar=True,
                 on_step=False, on_epoch=True)

    lit.LitPosFormer.training_step = _patched_training_step
    lit.LitPosFormer.validation_step = _patched_validation_step
    step("lit_posformer.patched", note="replaced .cuda() with .to(self.device)")


# ━━━━━━━━━━━━━━━━━━━━ HF checkpoint upload callback ━━━━━━━━━━━━━━━━━━━━━━━

def _make_hf_checkpoint_callback():
    """PyTorch Lightning callback: upload checkpoint to HF every hour."""
    import pytorch_lightning as pl

    class HFCheckpointCallback(pl.Callback):
        def __init__(self, upload_interval_s: float = 3600.0,
                     repo_id: str = HF_CHECKPOINT_REPO,
                     run_name: str = "default"):
            super().__init__()
            self.upload_interval_s = upload_interval_s
            self.repo_id = repo_id
            self.run_name = run_name
            self._last_upload = 0.0
            self._local_ckpt = WORK / "checkpoints" / "latest.ckpt"

        def on_train_epoch_end(self, trainer, pl_module):
            now = time.time()
            if (now - self._last_upload) < self.upload_interval_s:
                return
            self._upload(trainer, pl_module, tag="hourly")

        def on_train_end(self, trainer, pl_module):
            self._upload(trainer, pl_module, tag="final")

        def _upload(self, trainer, pl_module, tag="hourly"):
            if not os.environ.get("HF_TOKEN"):
                return
            try:
                self._local_ckpt.parent.mkdir(parents=True, exist_ok=True)
                trainer.save_checkpoint(str(self._local_ckpt))
                size_mb = self._local_ckpt.stat().st_size / 1e6

                from huggingface_hub import HfApi
                api = HfApi(token=os.environ["HF_TOKEN"])

                # Ensure repo exists
                try:
                    api.create_repo(self.repo_id, repo_type="model",
                                    exist_ok=True, private=True)
                except Exception:
                    pass

                epoch = trainer.current_epoch
                global_step = trainer.global_step

                # Upload as latest (for resume) AND as timestamped (for history)
                for remote_name in [
                    f"{self.run_name}/latest.ckpt",
                    f"{self.run_name}/epoch{epoch:04d}-step{global_step}.ckpt",
                ]:
                    api.upload_file(
                        path_or_fileobj=str(self._local_ckpt),
                        path_in_repo=remote_name,
                        repo_id=self.repo_id, repo_type="model",
                        commit_message=f"{tag} epoch={epoch} step={global_step}",
                    )

                self._last_upload = time.time()
                step(f"hf_checkpoint.{tag}",
                     epoch=epoch, step=global_step,
                     size_mb=round(size_mb, 1))
            except Exception as e:
                print(f"  HF checkpoint upload failed (non-fatal): {e}",
                      flush=True)

    return HFCheckpointCallback


# ━━━━━━━━━━━━━━━━━━━━ Resume from HF ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def download_latest_checkpoint(run_name: str) -> Optional[Path]:
    """Download latest.ckpt from HF if it exists."""
    if not os.environ.get("HF_TOKEN"):
        return None
    try:
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(
            repo_id=HF_CHECKPOINT_REPO,
            filename=f"{run_name}/latest.ckpt",
            repo_type="model",
            local_dir=str(WORK / "hf_checkpoints"),
        )
        step("resume.downloaded", path=local)
        return Path(local)
    except Exception as e:
        step("resume.none", note=str(e)[:100])
        return None


# ━━━━━━━━━━━━━━━━━━━━ Training ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train(args, zip_path: Path, dict_path: Path,
          resume_ckpt: Optional[Path] = None) -> Optional[str]:
    """Run PosFormer training. Returns best checkpoint path."""
    step("train.setup")

    # Detect GPU capability BEFORE importing torch.
    # Kaggle assigns P100 (sm_60) or T4 (sm_75) randomly.
    # Modern PyTorch only supports sm_70+. On P100, install CPU-only torch.
    gpu_usable = False
    try:
        nvsmi = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap",
                                "--format=csv,noheader"], capture_output=True, text=True)
        gpu_info = nvsmi.stdout.strip()
        step("gpu.detect", info=gpu_info)
        if gpu_info:
            cap = gpu_info.split(",")[-1].strip()
            major = int(cap.split(".")[0])
            if major >= 7:
                gpu_usable = True
                step("gpu.compatible", cap=cap)
            else:
                step("gpu.incompatible", cap=cap,
                     note="sm_60: installing CPU-only torch")
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
    except Exception as e:
        step("gpu.detect.failed", error=str(e)[:80])

    # Install deps — CPU-only torch on P100, default (CUDA) torch on T4
    if gpu_usable:
        sh("pip install -q torch torchvision pytorch_lightning opencv-python-headless wandb",
           capture_output=True)
    else:
        # Force-reinstall to avoid conflicts with Kaggle's pre-installed CUDA torchvision
        sh("pip install -q --force-reinstall torch torchvision "
           "--index-url https://download.pytorch.org/whl/cpu",
           capture_output=True)
        sh("pip install -q pytorch_lightning opencv-python-headless wandb",
           capture_output=True)

    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

    # Clone PosFormer
    posformer_dir = WORK / "PosFormer"
    if not posformer_dir.exists():
        sh(f"git clone --depth 1 {POSFORMER_REPO} {posformer_dir}")
    sys.path.insert(0, str(posformer_dir))

    # Patch dictionary
    vocab_target = posformer_dir / "Pos_Former" / "datamodule" / "dictionary.txt"
    shutil.copy(dict_path, vocab_target)
    step("train.dict_patched", src=str(dict_path), dst=str(vocab_target))

    # Import AFTER patching dictionary (vocab is loaded at import time)
    from Pos_Former.datamodule import CROHMEDatamodule, vocab
    from Pos_Former.lit_posformer import LitPosFormer

    step("train.vocab_loaded", size=len(vocab))

    # Patch label_make_muti for this vocab
    make_dynamic_label_module(vocab)

    # Patch .cuda() → .to(self.device) in training/validation steps
    patch_lit_posformer()

    # Global fix: if CUDA not available, make .cuda() a no-op
    # (catches .cuda() calls buried in model forward/utils that we can't monkey-patch individually)
    if not torch.cuda.is_available():
        _orig_cuda = torch.Tensor.cuda
        torch.Tensor.cuda = lambda self, *a, **kw: self  # type: ignore
        step("torch.cuda.patched", note=".cuda() is now a no-op (CPU mode)")

    # Determine test_year / validation split
    with zipfile.ZipFile(zip_path) as zf:
        dirs = {n.split("/")[1] for n in zf.namelist()
                if n.startswith("data/") and len(n.split("/")) > 2}
    # Prefer "valid" or "2014" for validation
    if "valid" in dirs:
        test_year = "valid"
    elif "2014" in dirs:
        test_year = "2014"
    else:
        test_year = sorted(d for d in dirs if d != "train")[0]
    step("train.splits", dirs=sorted(dirs), test_year=test_year)

    # Accelerator — GPU detection already done above (before torch import)
    if args.device == "auto":
        if gpu_usable and torch.cuda.is_available():
            accelerator, devices = "gpu", 1
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            accelerator, devices = "mps", 1
        else:
            accelerator, devices = "cpu", "auto"
    elif args.device == "mps":
        accelerator, devices = "mps", 1
    elif args.device in ("gpu", "cuda"):
        accelerator, devices = "gpu", 1
    else:
        accelerator, devices = "cpu", "auto"

    step("train.accelerator", accel=accelerator, epochs=args.epochs,
         batch=args.batch_size)

    # Datamodule
    dm = CROHMEDatamodule(
        zipfile_path=str(zip_path),
        test_year=test_year,
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
    ckpt_dir = WORK / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_best = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        monitor="val_ExpRate", mode="max", save_top_k=3,
        filename="{epoch}-{step}-{val_ExpRate:.4f}",
    )
    ckpt_periodic = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        every_n_train_steps=2000,
        save_top_k=-1,
        filename="periodic-{epoch}-{step}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    HFCheckpointCallback = _make_hf_checkpoint_callback()
    hf_callback = HFCheckpointCallback(
        upload_interval_s=3600.0,  # hourly
        run_name=args.dataset,
    )

    callbacks = [ckpt_best, ckpt_periodic, lr_monitor, hf_callback]

    # W&B logger
    logger = None
    if os.environ.get("WANDB_API_KEY"):
        try:
            from pytorch_lightning.loggers import WandbLogger
            # Fixed run_id per dataset for cross-session continuity
            run_id = f"posformer-{args.dataset}"
            logger = WandbLogger(
                project=WANDB_PROJECT,
                name=f"posformer-{args.dataset}",
                id=run_id,
                resume="allow",
                save_dir=str(WORK),
                tags=[args.dataset, accelerator],
                config={
                    "dataset": args.dataset,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "vocab_size": len(vocab),
                    "accelerator": accelerator,
                },
            )
            step("wandb.init", run_id=run_id, project=WANDB_PROJECT)
        except Exception as e:
            print(f"  W&B init failed (non-fatal): {e}", flush=True)

    # Trainer
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=args.epochs,
        callbacks=callbacks,
        logger=logger or True,
        default_root_dir=str(WORK),
        log_every_n_steps=50,
        val_check_interval=0.5,
        gradient_clip_val=1.0,
    )

    step("train.start", epochs=args.epochs)
    trainer.fit(model, dm, ckpt_path=str(resume_ckpt) if resume_ckpt else None)

    best_path = ckpt_best.best_model_path
    best_score = ckpt_best.best_model_score
    step("train.done", best_path=best_path,
         best_ExpRate=float(best_score) if best_score else None)

    # Final W&B cleanup
    if logger and hasattr(logger, 'experiment'):
        try:
            logger.experiment.finish()
        except Exception:
            pass

    return best_path


# ━━━━━━━━━━━━━━━━━━━━ GGUF conversion ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def convert_to_gguf(ckpt_path: str, dict_path: Path) -> Optional[Path]:
    """Convert best checkpoint → GGUF."""
    step("gguf.start")
    crispembed_dir = WORK / "CrispEmbed"
    if not crispembed_dir.exists():
        sh(f"git clone --depth 1 -b feat/posformer-port "
           f"{CRISPEMBED_REPO} {crispembed_dir}")

    sh("pip install -q gguf", capture_output=True)

    gguf_path = WORK / "posformer-trained-f32.gguf"
    converter = crispembed_dir / "models" / "convert-posformer-to-gguf.py"

    sh(f"python {converter} --checkpoint {ckpt_path} "
       f"--dict {dict_path} --output {gguf_path}")

    step("gguf.done", size_mb=round(gguf_path.stat().st_size / 1e6, 1))
    return gguf_path


def upload_gguf(gguf_path: Path, dataset: str) -> None:
    """Upload GGUF to HuggingFace model repo."""
    if not os.environ.get("HF_TOKEN"):
        step("gguf.upload_skip", note="no HF_TOKEN")
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ["HF_TOKEN"])
        try:
            api.create_repo(HF_MODEL_REPO, repo_type="model",
                            exist_ok=True, private=False)
        except Exception:
            pass
        remote_name = f"posformer-{dataset}-f32.gguf"
        api.upload_file(
            path_or_fileobj=str(gguf_path),
            path_in_repo=remote_name,
            repo_id=HF_MODEL_REPO, repo_type="model",
            commit_message=f"Trained on {dataset}",
        )
        step("gguf.uploaded", repo=HF_MODEL_REPO, file=remote_name)
    except Exception as e:
        step("gguf.upload_failed", error=str(e)[:200])


# ━━━━━━━━━━━━━━━━━━━━ Main ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser(
        description="PosFormer training — CROHME / MathWriting")
    parser.add_argument("--dataset", default=None,
                        choices=["crohme", "mathwriting", "finetune"],
                        help="Which dataset to train on (default: from env DATASET)")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "gpu", "cuda", "mps", "cpu"])
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epoch count (default: 200 for crohme, "
                             "20 for mathwriting, 50 for finetune)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Limit training samples (0 = all)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start fresh, ignore existing HF checkpoint")
    parser.add_argument("--no-gguf", action="store_true",
                        help="Skip GGUF conversion")
    args = parser.parse_args()

    # Dataset from env or arg
    if args.dataset is None:
        args.dataset = os.environ.get("DATASET", "crohme").lower()

    # Default epochs per dataset
    if args.epochs is None:
        args.epochs = {"crohme": 200, "mathwriting": 20, "finetune": 50
                       }.get(args.dataset, 200)

    # Init
    os.environ["PYTHONUNBUFFERED"] = "1"
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    WORK.mkdir(parents=True, exist_ok=True)

    step("init", dataset=args.dataset, epochs=args.epochs,
         batch_size=args.batch_size, device=args.device)

    # Auth
    tokens = resolve_tokens()

    # Data
    if args.dataset == "crohme":
        zip_path = download_crohme()
    elif args.dataset == "mathwriting":
        zip_path = download_mathwriting()
    elif args.dataset == "finetune":
        # Finetune: train on CROHME, but resume from MathWriting checkpoint
        zip_path = download_crohme()
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Vocabulary
    tokens_list, dict_path = build_vocab_from_zip(zip_path)

    # Resume checkpoint
    resume_ckpt = None
    if not args.no_resume:
        if args.dataset == "finetune":
            # Finetune resumes from the MathWriting checkpoint
            resume_ckpt = download_latest_checkpoint("mathwriting")
            if not resume_ckpt:
                print("WARNING: finetune mode but no MathWriting checkpoint "
                      "found on HF. Training from scratch on CROHME.",
                      flush=True)
        else:
            resume_ckpt = download_latest_checkpoint(args.dataset)

    # Train
    best_ckpt = train(args, zip_path, dict_path, resume_ckpt)

    # Convert to GGUF
    if best_ckpt and not args.no_gguf:
        gguf_path = convert_to_gguf(best_ckpt, dict_path)
        if gguf_path:
            upload_gguf(gguf_path, args.dataset)

    step("all_done")


if __name__ == "__main__":
    main()
