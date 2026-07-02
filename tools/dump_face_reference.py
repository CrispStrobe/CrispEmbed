#!/usr/bin/env python3
"""Dump SCRFD face-detection reference to GGUF for the crispembed face guardrail.

Independent reference: runs insightface's own SCRFD detector on the ONNX model
(det_10g.onnx) over a fixed face image, and writes the detection (bbox + confidence
+ 5 landmarks) so the C++ crispembed_detect_faces output can be compared against it.
A graph-scramble regression in the C++ SCRFD engine yields wrong/zero detections.

  detection tensor layout (per face, 15 floats):
      x, y, w, h, confidence, lm0x, lm0y, ... lm4x, lm4y

Usage:
    python tools/dump_face_reference.py \
        --onnx det_10g.onnx --image face.png --output face-ref.gguf
"""
import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import gguf
except ImportError:
    sys.exit("pip install gguf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="SCRFD det_10g.onnx")
    ap.add_argument("--image", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--det-size", type=int, default=640)
    ap.add_argument("--thresh", type=float, default=0.5)
    args = ap.parse_args()

    import cv2
    from insightface.model_zoo import SCRFD

    det = SCRFD(model_file=args.onnx)
    det.prepare(ctx_id=-1, input_size=(args.det_size, args.det_size), det_thresh=args.thresh)
    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"could not read {args.image}")
    bboxes, kpss = det.detect(img, input_size=(args.det_size, args.det_size))
    n = len(bboxes)
    print(f"n_faces={n}  image={img.shape}")

    dets = []
    for b, k in zip(bboxes, kpss):
        x1, y1, x2, y2, score = b
        row = [float(x1), float(y1), float(x2 - x1), float(y2 - y1), float(score)]
        for p in k:
            row += [float(p[0]), float(p[1])]
        dets.append(row)
        print(f"  xywh=({row[0]:.1f},{row[1]:.1f},{row[2]:.1f},{row[3]:.1f}) conf={row[4]:.4f}")
    if n == 0:
        sys.exit("no faces detected — pick a clearer face image")

    detection = np.asarray(dets, dtype=np.float32).reshape(-1)  # [n*15]

    w = gguf.GGUFWriter(args.output, "face-scrfd-ref")
    w.add_uint32("face.ref.n_faces", n)
    w.add_uint32("face.ref.det_size", args.det_size)
    w.add_tensor("detection", detection, raw_dtype=gguf.GGMLQuantizationType.F32)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"Wrote {args.output} ({Path(args.output).stat().st_size} bytes)")


if __name__ == "__main__":
    main()
