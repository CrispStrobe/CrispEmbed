#!/usr/bin/env python3
"""Score native inpainting outputs across precisions (F32/F16/Q8_0/Q4_K).

Pure measurement: compares raw float outputs and composited RGB8 images of
one precision variant against (a) the native F32 output and (b) the original
module's capture. It never changes a frozen original/native parity gate; the
owner's quantization acceptance (output cosine > 0.94 against both F32 and
the original) is reported as a separate field.

usage:
  inpaint_precision_score.py pair  A.f32 B.f32
  inpaint_precision_score.py rgb   A.rgb8 B.rgb8 MASK.u8 INPUT.rgb8
  inpaint_precision_score.py table SPEC.json OUT.json
SPEC.json = {"cases": {case: {"original_float": path, "original_rgb": path,
             "mask": path, "input": path,
             "variants": {name: {"float": path, "rgb": path}}}}}
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

OWNER_COSINE = 0.94


def load_float(path):
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False).astype(np.float64).ravel()
    return np.fromfile(path, dtype="<f4").astype(np.float64)


def float_metrics(a, b):
    if a.size != b.size:
        raise ValueError(f"element count differs: {a.size} != {b.size}")
    diff = a - b
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    cosine = float(a @ b / (na * nb)) if na > 0 and nb > 0 else (1.0 if na == nb else 0.0)
    mse = float(np.mean(diff * diff))
    # Generator outputs are in [-1, 1] (peak-to-peak 2) before RGB mapping.
    psnr = 10 * math.log10(4.0 / mse) if mse > 0 else None
    return {"count": int(a.size), "max_abs": float(np.max(np.abs(diff))), "mae": float(np.mean(np.abs(diff))),
            "relative_l2": float(np.linalg.norm(diff) / nb) if nb > 0 else None,
            "cosine": cosine, "psnr_db_range2": psnr, "finite": bool(np.isfinite(a).all())}


def rgb_metrics(actual, reference, mask, source):
    actual = np.fromfile(actual, dtype=np.uint8).astype(np.int32)
    reference = np.fromfile(reference, dtype=np.uint8).astype(np.int32)
    fill = np.repeat(np.fromfile(mask, dtype=np.uint8) == 255, 3)
    source = np.fromfile(source, dtype=np.uint8).astype(np.int32)
    if not (actual.size == reference.size == fill.size == source.size):
        raise ValueError("RGB/mask sizes differ")
    known_errors = int(np.count_nonzero(actual[~fill] != source[~fill]))
    result = {"fill_channels": int(fill.sum()), "known_errors": known_errors}
    if fill.any():
        a, r = actual[fill].astype(np.float64), reference[fill].astype(np.float64)
        d = a - r
        mse = float(np.mean(d * d))
        na, nr = float(np.linalg.norm(a)), float(np.linalg.norm(r))
        result.update({"max_abs": int(np.max(np.abs(d))), "mae": float(np.mean(np.abs(d))),
                       "psnr_db": 10 * math.log10(255.0 ** 2 / mse) if mse > 0 else None,
                       "cosine_fill_rgb": float(a @ r / (na * nr)) if na > 0 and nr > 0 else None})
    return result


def table(spec_path, out_path):
    spec = json.loads(Path(spec_path).read_text())
    rows = []
    for case, record in spec["cases"].items():
        original = load_float(record["original_float"]) if record.get("original_float") else None
        variants = record["variants"]
        f32 = load_float(variants["f32"]["float"]) if "f32" in variants else None
        for name, files in variants.items():
            value = load_float(files["float"])
            row = {"case": case, "variant": name}
            if original is not None:
                row["vs_original"] = float_metrics(value, original)
            if f32 is not None and name != "f32":
                row["vs_native_f32"] = float_metrics(value, f32)
            if files.get("rgb") and record.get("original_rgb"):
                row["rgb_vs_original"] = rgb_metrics(files["rgb"], record["original_rgb"], record["mask"], record["input"])
            if files.get("rgb") and "f32" in variants and name != "f32":
                row["rgb_vs_native_f32"] = rgb_metrics(files["rgb"], variants["f32"]["rgb"], record["mask"], record["input"])
            cosines = [row[k]["cosine"] for k in ("vs_original", "vs_native_f32") if k in row]
            row["owner_cosine_gate_0_94"] = bool(cosines) and all(c > OWNER_COSINE for c in cosines)
            for key in ("seconds", "peak_rss_kib", "gguf_bytes"):
                if key in files:
                    row[key] = files[key]
            rows.append(row)
    result = {"schema": "crispembed-inpaint-precision-score-v1", "owner_cosine_threshold": OWNER_COSINE,
              "frozen_parity_gates_unchanged": True, "rows": rows}
    Path(out_path).write_text(json.dumps(result, indent=2) + "\n")
    for row in rows:
        o = row.get("vs_original", {})
        f = row.get("vs_native_f32", {})
        print(f"{row['case']:14s} {row['variant']:8s} cos_orig={o.get('cosine', float('nan')):.6f} "
              f"max_orig={o.get('max_abs', float('nan')):.4g} cos_f32={f.get('cosine', float('nan')):.6f} "
              f"gate={row['owner_cosine_gate_0_94']}")
    return result


def main(argv):
    if len(argv) >= 2 and argv[1] == "pair" and len(argv) == 4:
        print(json.dumps(float_metrics(load_float(argv[2]), load_float(argv[3])), indent=2))
    elif len(argv) >= 2 and argv[1] == "rgb" and len(argv) == 6:
        print(json.dumps(rgb_metrics(*argv[2:6]), indent=2))
    elif len(argv) >= 2 and argv[1] == "table" and len(argv) == 4:
        table(argv[2], argv[3])
    else:
        raise SystemExit(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
