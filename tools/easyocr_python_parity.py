#!/usr/bin/env python3
"""Python EasyOCR ground truth for the native DBNet + EasyOCR page lane.

Three references per fixture, all from the installed `easyocr` wheel:

  readtext   `easyocr.Reader(['en'])` end to end (CRAFT detector + recognizer,
             default parameters), once with the default dynamic int8
             quantization (`quantize=True`, EasyOCR's CPU default) and once in
             F32 (`quantize=False`, the precision the GGUF was converted from).
  crops      Python's own `Reader.recognize()` on the exact crops the native
             pipeline produced (`recognizer_crop` of a test-easyocr-pipeline
             manifest), given the image path (EasyOCR's documented input; grey
             comes from `cv2.imread(..., IMREAD_GRAYSCALE)`) and given the RGB
             array (grey from `cv2.cvtColor`), both F32.
  variants   The recognizer fed by a re-implementation of `get_image_list` +
             `AlignCollate` + `recognizer_predict` with one preprocessing
             choice switched at a time, to attribute every native/Python
             difference to a stage: grey conversion (libpng vs BT.601 integer),
             resized width (`int()` vs `ceil()`), and the `contrast_ths`
             retry with `adjust_contrast`.

  python tools/easyocr_python_parity.py --image fox.png --manifest fox.json \
      --model-dir ~/.EasyOCR/model --output fox.parity.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def custom_mean(x):
    return float(x.prod() ** (2.0 / np.sqrt(len(x))))


def adjust_contrast_grey(img, target=0.4):
    high = np.percentile(img, 90)
    low = np.percentile(img, 10)
    contrast = (high - low) / np.maximum(10, high + low)
    if contrast < target:
        img = img.astype(int)
        ratio = 200.0 / np.maximum(10, high - low)
        img = (img - low + 25) * ratio
        img = np.maximum(np.full(img.shape, 0), np.minimum(np.full(img.shape, 255), img)).astype(np.uint8)
    return img


def predict(reader, resized, canvas):
    """`recognizer_predict` (greedy) on one already height-64 grey crop."""
    arr = resized.astype(np.float32) / 255.0 * 2.0 - 1.0
    h, w = arr.shape
    pad = np.zeros((1, 1, h, canvas), dtype=np.float32)
    pad[0, 0, :, :w] = arr
    if w < canvas:
        pad[0, 0, :, w:] = arr[:, w - 1:w]
    with torch.no_grad():
        preds = reader.recognizer(torch.from_numpy(pad), torch.zeros((1, canvas // 10 + 1), dtype=torch.long))
    prob = F.softmax(preds, dim=2).numpy()[0]
    idx = prob.argmax(axis=1)
    keep = np.insert(idx[1:] != idx[:-1], 0, True) & (idx != 0)
    text = "".join(reader.converter.character[i] for i in idx[keep])
    best = prob.max(axis=1)[idx != 0]
    conf = custom_mean(best) if len(best) else 0.0
    return text, conf


def variant(reader, grey_page, crop, width_rule, contrast):
    x, y, w, h = crop
    g = grey_page[y:y + h, x:x + w]
    ratio = w / h
    rw = int(64 * ratio) if width_rule == "int" else max(1, math.ceil(64 * ratio))
    resized = cv2.resize(g, (rw, 64), interpolation=cv2.INTER_LINEAR)  # PIL LANCZOS == 1 == INTER_LINEAR
    canvas = max(1, math.ceil(max(ratio, 1.0))) * 64
    # AlignCollate re-resizes to ceil(64 * rw / 64) == rw with PIL BICUBIC: an identity copy.
    text, conf = predict(reader, resized, canvas)
    retried = False
    if contrast and conf < 0.1:
        t2, c2 = predict(reader, adjust_contrast_grey(resized, target=0.5), canvas)
        retried = True
        if c2 >= conf:
            text, conf = t2, c2
    return {"text": text, "confidence": round(conf, 5), "resized_width": rw, "canvas": canvas, "retried": retried}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, help="test-easyocr-pipeline manifest with recognizer_crop per record")
    ap.add_argument("--model-dir", type=Path, default=None, help="EasyOCR model_storage_directory")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--no-readtext", action="store_true", help="skip the end-to-end readtext references")
    args = ap.parse_args()

    import easyocr

    kw = {"gpu": False, "verbose": False}
    if args.model_dir:
        kw["model_storage_directory"] = str(args.model_dir)
    out = {"easyocr": easyocr.__version__, "torch": torch.__version__, "cv2": cv2.__version__, "image": args.image.name}

    # 1. End to end, default parameters (quantize=True is EasyOCR's CPU default).
    for name, q in (() if args.no_readtext else (("readtext_default_int8", True), ("readtext_f32", False))):
        r = easyocr.Reader(["en"], quantize=q, **kw)
        res = r.readtext(str(args.image))
        out[name] = [{"box": [[int(v) for v in p] for p in b], "text": t, "confidence": round(float(c), 5)}
                     for b, t, c in res]

    if not args.manifest:
        args.output.write_text(json.dumps(out, indent=1, ensure_ascii=False))
        return 0

    reader = easyocr.Reader(["en"], quantize=False, detector=False, **kw)
    reader.recognizer.eval()
    records = json.loads(args.manifest.read_text())["records"]
    rgb = cv2.cvtColor(cv2.imread(str(args.image), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    grey_libpng = cv2.imread(str(args.image), cv2.IMREAD_GRAYSCALE)
    grey_bt601 = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    rows = []
    for rec in records:
        x, y, w, h = (int(v) for v in rec.get("recognizer_crop", rec["crop"]))
        hl = [[x, x + w, y, y + h]]
        row = {"index": rec["index"], "crop": [x, y, w, h], "native": {"text": rec["text"],
                                                                        "confidence": rec["confidence"]}}
        for name, src in (("python_path", str(args.image)), ("python_rgb_array", rgb[:, :, ::-1].copy())):
            res = reader.recognize(src, horizontal_list=hl, free_list=[], decoder="greedy", batch_size=1, detail=1)
            row[name] = {"text": res[0][1] if res else "", "confidence": round(float(res[0][2]), 5) if res else 0.0}
        for grey_name, grey in (("libpng", grey_libpng), ("bt601", grey_bt601)):
            for wr in ("int", "ceil"):
                for contrast in (True, False):
                    key = f"v_{grey_name}_{wr}_{'retry' if contrast else 'noretry'}"
                    row[key] = variant(reader, grey, (x, y, w, h), wr, contrast)
        rows.append(row)
    out["crops"] = rows
    args.output.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
