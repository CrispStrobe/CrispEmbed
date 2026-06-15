#!/usr/bin/env python3
"""Dump TPS localization network reference activations for parity testing.

Loads the PaddleOCR TPS localization net from .pdparams, runs forward pass
on a test image, and dumps per-layer intermediate activations to a GGUF
archive. The C++ test binary then compares its own activations against these.

Usage:
    python tools/dump_tps_reference.py \
        --model /tmp/rec_mv3_tps_bilstm_att_v2.0_train \
        --output /tmp/tps-ref.gguf \
        [--image test.png]

Stages captured:
    input          — preprocessed [3, H, W] float32
    conv{0-3}_out  — after Conv+BN+ReLU+Pool
    fc1_out        — after FC1+ReLU
    fc2_out        — raw control point coordinates (num_fiducial * 2)
    points_pixel   — control points in pixel space
"""

import argparse
import pickle
import struct
import sys
from pathlib import Path

import numpy as np

try:
    import gguf
except ImportError:
    sys.exit("pip install gguf")


def fold_bn(conv_w, bn_w, bn_b, bn_mean, bn_var, eps=1e-5):
    oc = conv_w.shape[0]
    inv_std = 1.0 / np.sqrt(bn_var + eps)
    scale = bn_w * inv_std
    shape = [oc] + [1] * (conv_w.ndim - 1)
    w_folded = conv_w * scale.reshape(shape)
    b_folded = -bn_mean * scale + bn_b
    return w_folded, b_folded


def conv2d(x, w, b, pad=1):
    """[IC, IH, IW] × [OC, IC, KH, KW] → [OC, OH, OW]"""
    ic, ih, iw = x.shape
    oc, _, kh, kw = w.shape
    oh = ih + 2 * pad - kh + 1
    ow = iw + 2 * pad - kw + 1
    # Pad input
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)), mode='constant')
    out = np.zeros((oc, oh, ow), dtype=np.float32)
    for o in range(oc):
        for ky in range(kh):
            for kx in range(kw):
                out[o] += np.sum(
                    w[o, :, ky, kx].reshape(-1, 1, 1) *
                    xp[:, ky:ky+oh, kx:kx+ow], axis=0)
        out[o] += b[o]
    return out


def maxpool2x2(x):
    """[C, H, W] → [C, H/2, W/2]"""
    c, h, w = x.shape
    return x.reshape(c, h//2, 2, w//2, 2).max(axis=(2, 4))


def adaptive_avg_pool_1x1(x):
    """[C, H, W] → [C]"""
    return x.mean(axis=(1, 2))


def fc(x, w, b):
    """x: [IC], w: [IC, OC] (Paddle layout), b: [OC] → [OC]"""
    return x @ w + b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="PaddleOCR model dir")
    parser.add_argument("--output", "-o", required=True, help="Output reference GGUF")
    parser.add_argument("--image", help="Test image (PNG/JPG). Default: synthetic.")
    parser.add_argument("--width", type=int, default=200)
    parser.add_argument("--height", type=int, default=64)
    args = parser.parse_args()

    # Load state dict
    pdparams = Path(args.model) / "best_accuracy.pdparams"
    if not pdparams.exists():
        pdparams = Path(args.model)
    with open(pdparams, "rb") as f:
        sd = pickle.load(f)

    prefix = "transform.loc_net."

    # Fold BN into conv weights
    convs = []
    for i in range(4):
        conv_w = np.asarray(sd[f"{prefix}loc_conv{i}.conv.weight"], dtype=np.float32)
        bn_w = np.asarray(sd[f"{prefix}loc_conv{i}.bn.weight"], dtype=np.float32)
        bn_b = np.asarray(sd[f"{prefix}loc_conv{i}.bn.bias"], dtype=np.float32)
        bn_mean = np.asarray(sd[f"{prefix}loc_conv{i}.bn._mean"], dtype=np.float32)
        bn_var = np.asarray(sd[f"{prefix}loc_conv{i}.bn._variance"], dtype=np.float32)
        w, b = fold_bn(conv_w, bn_w, bn_b, bn_mean, bn_var)
        convs.append((w, b))

    fc1_w = np.asarray(sd[f"{prefix}fc1.weight"], dtype=np.float32)  # [128, 64]
    fc1_b = np.asarray(sd[f"{prefix}fc1.bias"], dtype=np.float32)    # [64]
    fc2_w = np.asarray(sd[f"{prefix}fc2.weight"], dtype=np.float32)  # [64, 40]
    fc2_b = np.asarray(sd[f"{prefix}fc2.bias"], dtype=np.float32)    # [40]

    # Prepare input image
    if args.image:
        try:
            from PIL import Image
            img = Image.open(args.image).convert("L")
            img = img.resize((args.width, args.height))
            gray = np.array(img, dtype=np.uint8)
        except ImportError:
            sys.exit("pip install Pillow for --image support")
    else:
        # Synthetic: curved text lines
        W, H = args.width, args.height
        gray = np.full((H, W), 230, dtype=np.uint8)
        for line in range(3):
            base_y = 12 + line * 18
            for x in range(10, W - 10):
                curve = int(4.0 * np.sin(np.pi * x / W))
                for dy in range(5):
                    y = base_y + curve + dy
                    if 0 <= y < H:
                        gray[y, x] = 30

    W, H = gray.shape[1], gray.shape[0]
    print(f"Input: {W}x{H} grayscale")

    # Preprocess: gray → 3-channel [C, H, W] float in [0, 1]
    x = np.stack([gray.astype(np.float32) / 255.0] * 3, axis=0)  # [3, H, W]

    stages = {}
    stages["input"] = x.copy()

    # Forward pass through 4 conv blocks
    for i in range(4):
        w, b = convs[i]
        x = conv2d(x, w, b, pad=1)
        x = np.maximum(x, 0)  # ReLU
        if i < 3:
            x = maxpool2x2(x)
        else:
            x = adaptive_avg_pool_1x1(x)  # [128]
        stages[f"conv{i}_out"] = x.copy()
        print(f"  conv{i}_out: shape={list(x.shape)}, range=[{x.min():.4f}, {x.max():.4f}]")

    # FC1 + ReLU
    x = fc(x, fc1_w, fc1_b)
    x = np.maximum(x, 0)
    stages["fc1_out"] = x.copy()
    print(f"  fc1_out: shape={list(x.shape)}, range=[{x.min():.4f}, {x.max():.4f}]")

    # FC2
    x = fc(x, fc2_w, fc2_b)
    stages["fc2_out"] = x.copy()
    print(f"  fc2_out: shape={list(x.shape)}, range=[{x.min():.4f}, {x.max():.4f}]")

    # Convert to pixel coordinates
    num_fiducial = len(x) // 2
    points = x.reshape(num_fiducial, 2)
    px = (points[:, 0] + 1.0) * 0.5 * (W - 1)
    py = (points[:, 1] + 1.0) * 0.5 * (H - 1)
    pixel_pts = np.stack([px, py], axis=1).flatten()
    stages["points_pixel"] = pixel_pts.astype(np.float32)
    print(f"\nPredicted {num_fiducial} control points:")
    for i in range(num_fiducial):
        print(f"  [{i:2d}] ({px[i]:.1f}, {py[i]:.1f})")

    # Write reference GGUF
    writer = gguf.GGUFWriter(args.output, "tps-reference")
    writer.add_uint32("tps.ref.width", W)
    writer.add_uint32("tps.ref.height", H)
    writer.add_uint32("tps.ref.num_fiducial", num_fiducial)

    for name, arr in stages.items():
        arr = arr.astype(np.float32)
        writer.add_tensor(name, arr, raw_dtype=gguf.GGMLQuantizationType.F32)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"\nReference GGUF: {args.output} ({Path(args.output).stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
