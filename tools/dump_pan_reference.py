#!/usr/bin/env python3
"""Dump PAN per-stage reference activations for parity testing.

Usage:
    PYTHONNOUSERSITE=1 python tools/dump_pan_reference.py \
        --model pan_x4.pdparams --output /tmp/pan-ref.gguf [--size 32]
"""

import argparse
import pickle
import struct
import numpy as np


def conv2d(x, weight, bias=None, padding=0):
    """Vectorized Conv2D: x=[B,IC,H,W], weight=[OC,IC,KH,KW]."""
    B, IC, H, W = x.shape
    OC, IC_W, KH, KW = weight.shape
    if padding > 0:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    _, _, PH, PW = x.shape
    OH, OW = PH - KH + 1, PW - KW + 1
    s = x.strides
    col = np.lib.stride_tricks.as_strided(
        x, shape=(B, IC, KH, KW, OH, OW),
        strides=(s[0], s[1], s[2], s[3], s[2], s[3]))
    col = col.reshape(B, IC * KH * KW, OH * OW)
    w = weight.reshape(OC, IC_W * KH * KW)
    out = np.einsum('ij,bjk->bik', w, col).reshape(B, OC, OH, OW)
    if bias is not None:
        out += bias.reshape(1, OC, 1, 1)
    return out


def leaky_relu(x, slope=0.2):
    return np.where(x > 0, x, x * slope)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def nearest_upsample(x, factor):
    """Nearest-neighbor 2D upsample: [B,C,H,W] → [B,C,H*f,W*f]."""
    return x.repeat(factor, axis=2).repeat(factor, axis=3)


def bilinear_upsample(x, factor):
    """Bilinear upsample: [B,C,H,W] → [B,C,H*f,W*f]."""
    B, C, H, W = x.shape
    OH, OW = H * factor, W * factor
    out = np.zeros((B, C, OH, OW), dtype=np.float32)
    for oy in range(OH):
        sy = (oy + 0.5) * H / OH - 0.5
        iy = int(np.floor(sy))
        fy = sy - iy
        iy0, iy1 = max(0, iy), min(H - 1, iy + 1)
        for ox in range(OW):
            sx = (ox + 0.5) * W / OW - 0.5
            ix = int(np.floor(sx))
            fx = sx - ix
            ix0, ix1 = max(0, ix), min(W - 1, ix + 1)
            out[:, :, oy, ox] = ((1 - fy) * ((1 - fx) * x[:, :, iy0, ix0] + fx * x[:, :, iy0, ix1])
                                 + fy * ((1 - fx) * x[:, :, iy1, ix0] + fx * x[:, :, iy1, ix1]))
    return out


def get(state, key):
    return state[key].astype(np.float32) if key in state else None


def pan_forward(state, x):
    """PAN forward with intermediates."""
    intermediates = {"input": x[0].copy()}

    # conv_first
    fea = conv2d(x, get(state, "conv_first.weight"), get(state, "conv_first.bias"), padding=1)
    fea_skip = fea.copy()
    intermediates["conv_first"] = fea[0].copy()

    # SCPA trunk
    nb = 0
    while f"SCPA_trunk.block{nb}.conv1_a.weight" in state:
        nb += 1

    for i in range(nb):
        p = f"SCPA_trunk.block{i}"
        residual = fea.copy()

        # Branch A: conv1_a → LReLU → k1 → LReLU
        a = conv2d(fea, get(state, f"{p}.conv1_a.weight"), padding=0)
        a = leaky_relu(a)
        a = conv2d(a, get(state, f"{p}.k1.0.weight"), padding=1)
        a = leaky_relu(a)

        # Branch B: conv1_b → LReLU → PAConv → LReLU
        b = conv2d(fea, get(state, f"{p}.conv1_b.weight"), padding=0)
        b = leaky_relu(b)
        # PAConv: k2→sigmoid→mask, k3*mask, k4
        attn = conv2d(b, get(state, f"{p}.PAConv.k2.weight"), get(state, f"{p}.PAConv.k2.bias"), padding=0)
        attn = sigmoid(attn)
        b_out = conv2d(b, get(state, f"{p}.PAConv.k3.weight"), padding=1) * attn
        b_out = conv2d(b_out, get(state, f"{p}.PAConv.k4.weight"), padding=1)
        b = leaky_relu(b_out)

        # Concat + conv3 + residual
        cat = np.concatenate([a, b], axis=1)
        fea = conv2d(cat, get(state, f"{p}.conv3.weight"), padding=0) + residual

    intermediates["scpa_trunk"] = fea[0].copy()

    # trunk_conv + skip
    fea = conv2d(fea, get(state, "trunk_conv.weight"), get(state, "trunk_conv.bias"), padding=1)
    fea = fea_skip + fea
    intermediates["after_skip"] = fea[0].copy()

    # Upsample stage 1: nearest 2× → upconv1 → PA → LReLU → HRconv1 → LReLU
    fea = nearest_upsample(fea, 2)
    fea = conv2d(fea, get(state, "upconv1.weight"), get(state, "upconv1.bias"), padding=1)
    pa = sigmoid(conv2d(fea, get(state, "att1.conv.weight"), get(state, "att1.conv.bias"), padding=0))
    fea = leaky_relu(fea * pa)
    fea = leaky_relu(conv2d(fea, get(state, "HRconv1.weight"), get(state, "HRconv1.bias"), padding=1))
    intermediates["upsample1"] = fea[0].copy()

    # Upsample stage 2 (if scale=4)
    if "upconv2.weight" in state:
        fea = nearest_upsample(fea, 2)
        fea = conv2d(fea, get(state, "upconv2.weight"), get(state, "upconv2.bias"), padding=1)
        pa2 = sigmoid(conv2d(fea, get(state, "att2.conv.weight"), get(state, "att2.conv.bias"), padding=0))
        fea = leaky_relu(fea * pa2)
        fea = leaky_relu(conv2d(fea, get(state, "HRconv2.weight"), get(state, "HRconv2.bias"), padding=1))
        intermediates["upsample2"] = fea[0].copy()
        scale = 4
    else:
        scale = 2

    # conv_last + bilinear residual
    out = conv2d(fea, get(state, "conv_last.weight"), get(state, "conv_last.bias"), padding=1)
    ilr = bilinear_upsample(x, scale)
    out = out + ilr
    intermediates["output_raw"] = out[0].copy()
    # Clamped version (what a uint8 image would look like)
    intermediates["output"] = np.clip(out[0], 0.0, 1.0).copy()

    return out, intermediates


def write_gguf(path, tensors):
    MAGIC = 0x46554747; VERSION = 3; TYPE_STRING = 8; TYPE_F32 = 0
    def ws(f, s):
        b = s.encode("utf-8"); f.write(struct.pack("<Q", len(b))); f.write(b)
    tensor_list = list(tensors.items())
    with open(path, "wb") as f:
        f.write(struct.pack("<I", MAGIC)); f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<Q", len(tensor_list))); f.write(struct.pack("<Q", 1))
        ws(f, "general.architecture"); f.write(struct.pack("<I", TYPE_STRING)); ws(f, "pan_ref")
        offset = 0
        for name, data in tensor_list:
            ws(f, name); f.write(struct.pack("<I", len(data.shape)))
            for d in data.shape: f.write(struct.pack("<Q", d))
            f.write(struct.pack("<I", TYPE_F32)); f.write(struct.pack("<Q", offset))
            offset += data.nbytes; offset = (offset + 31) & ~31
        pos = f.tell(); aligned = (pos + 31) & ~31; f.write(b"\x00" * (aligned - pos))
        for name, data in tensor_list:
            f.write(data.astype(np.float32).tobytes())
            pad = ((data.nbytes + 31) & ~31) - data.nbytes
            if pad > 0: f.write(b"\x00" * pad)
    print(f"Written {path}: {len(tensor_list)} tensors")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, default=32, help="Input size (default 32)")
    args = parser.parse_args()

    with open(args.model, "rb") as f:
        state = pickle.load(f)
    if "generator" in state:
        state = state["generator"]

    np.random.seed(42)
    inp = np.random.rand(1, 3, args.size, args.size).astype(np.float32)

    out, intermediates = pan_forward(state, inp)

    print("\nIntermediate activations:")
    for name, data in intermediates.items():
        print(f"  {name:20s}  shape={str(list(data.shape)):30s}  mean={data.mean():.6f}")

    write_gguf(args.output, intermediates)


if __name__ == "__main__":
    main()
