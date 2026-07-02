#!/usr/bin/env python3
"""Dump TPS localization reference activations from the shipped GGUF weights.

Independent-numpy-forward guardrail (mirrors the SR engines' *_from_gguf dumpers).
The PaddleOCR source .pdparams (rec_mv3_tps_bilstm_att_v2.0_train) is geo-blocked on
bcebos.com, so we take the already-BN-folded weights straight from tps-loc-f32.gguf and
run the SAME pure-numpy forward as tools/dump_tps_reference.py. A graph-scramble
regression in the C++ engine (the June-2026 wave failure mode) craters the per-stage
cosine vs this independent numpy path; a conversion bug is NOT caught (weights are shared)
— acceptable, the wave was about engine graphs, not conversion.

The C++ harness is tests/test_tps_parity.cpp (already exists).

Usage:
    python tools/dump_tps_reference_from_gguf.py \
        --gguf /path/tps-loc-f32.gguf --output /tmp/tps-ref.gguf
"""
import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import gguf
except ImportError:
    sys.exit("pip install gguf")


def conv2d(x, w, b, pad=1):
    """[IC, IH, IW] x [OC, IC, KH, KW] -> [OC, OH, OW]"""
    ic, ih, iw = x.shape
    oc, _, kh, kw = w.shape
    oh = ih + 2 * pad - kh + 1
    ow = iw + 2 * pad - kw + 1
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)), mode="constant")
    out = np.zeros((oc, oh, ow), dtype=np.float32)
    for o in range(oc):
        for ky in range(kh):
            for kx in range(kw):
                out[o] += np.sum(
                    w[o, :, ky, kx].reshape(-1, 1, 1) * xp[:, ky:ky + oh, kx:kx + ow], axis=0)
        out[o] += b[o]
    return out


def maxpool2x2(x):
    c, h, w = x.shape
    return x.reshape(c, h // 2, 2, w // 2, 2).max(axis=(2, 4))


def adaptive_avg_pool_1x1(x):
    return x.mean(axis=(1, 2))


def fc(x, w, b):
    """x: [IC], w: [IC, OC], b: [OC] -> [OC]"""
    return x @ w + b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True, help="tps-loc GGUF (folded weights)")
    ap.add_argument("--output", "-o", required=True)
    ap.add_argument("--width", type=int, default=200)
    ap.add_argument("--height", type=int, default=64)
    args = ap.parse_args()

    r = gguf.GGUFReader(args.gguf)
    T = {t.name: np.asarray(t.data, dtype=np.float32) for t in r.tensors}

    convs = [(T[f"loc.conv{i}.weight"], T[f"loc.conv{i}.bias"]) for i in range(4)]
    fc1_w, fc1_b = T["loc.fc1.weight"], T["loc.fc1.bias"]
    fc2_w, fc2_b = T["loc.fc2.weight"], T["loc.fc2.bias"]

    # Same synthetic curved-text image as tps_parity's C++ side & dump_tps_reference.py.
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

    x = np.stack([gray.astype(np.float32) / 255.0] * 3, axis=0)  # [3, H, W]
    stages = {"input": x.copy()}

    for i in range(4):
        w, b = convs[i]
        x = conv2d(x, w, b, pad=1)
        x = np.maximum(x, 0)
        x = maxpool2x2(x) if i < 3 else adaptive_avg_pool_1x1(x)
        stages[f"conv{i}_out"] = x.copy()
        print(f"  conv{i}_out: {list(x.shape)} range=[{x.min():.4f},{x.max():.4f}]")

    x = np.maximum(fc(x, fc1_w, fc1_b), 0)
    stages["fc1_out"] = x.copy()
    x = fc(x, fc2_w, fc2_b)
    stages["fc2_out"] = x.copy()
    print(f"  fc1_out: {list(stages['fc1_out'].shape)}  fc2_out: {list(x.shape)}")

    num_fiducial = len(x) // 2
    pts = x.reshape(num_fiducial, 2)
    px = (pts[:, 0] + 1.0) * 0.5 * (W - 1)
    py = (pts[:, 1] + 1.0) * 0.5 * (H - 1)
    stages["points_pixel"] = np.stack([px, py], axis=1).flatten().astype(np.float32)

    w = gguf.GGUFWriter(args.output, "tps-reference")
    w.add_uint32("tps.ref.width", W)
    w.add_uint32("tps.ref.height", H)
    w.add_uint32("tps.ref.num_fiducial", num_fiducial)
    for name, arr in stages.items():
        w.add_tensor(name, arr.astype(np.float32), raw_dtype=gguf.GGMLQuantizationType.F32)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"\nReference GGUF: {args.output} ({Path(args.output).stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
