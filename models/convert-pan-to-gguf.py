#!/usr/bin/env python3
"""Convert PaddleGAN PAN (Pixel Attention Network) checkpoint to GGUF.

Usage:
    python convert-pan-to-gguf.py \
        --model pan_x4.pdparams --output pan-x4-f16.gguf --fp16

PAN architecture (Apache-2.0):
    conv_first(3→nf) → nb× SCPA → trunk_conv → skip
    → nearest 2× → upconv1 → PA → LReLU → HRconv1 → LReLU
    → nearest 2× → upconv2 → PA → LReLU → HRconv2 → LReLU  (if scale=4)
    → conv_last(unf→3) → + bilinear(input)

Default: nf=40, unf=24, nb=16, scale=4. ~272K params, ~0.5MB F16.
"""

import argparse
import pickle
import struct
import sys

import numpy as np

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_STRING = 8


def write_string(f, s):
    b = s.encode("utf-8")
    f.write(struct.pack("<Q", len(b)))
    f.write(b)


def write_kv_string(f, key, val):
    write_string(f, key)
    f.write(struct.pack("<I", GGUF_TYPE_STRING))
    write_string(f, val)


def write_kv_u32(f, key, val):
    write_string(f, key)
    f.write(struct.pack("<I", GGUF_TYPE_UINT32))
    f.write(struct.pack("<I", val))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    with open(args.model, "rb") as f:
        state = pickle.load(f)
    if "generator" in state:
        state = state["generator"]

    tensors = {}

    def add(name, key):
        if key not in state:
            print(f"  WARNING: missing {key}")
            return
        arr = state[key].astype(np.float32)
        if args.fp16 and arr.ndim >= 2:
            tensors[name] = (arr.astype(np.float16), GGML_TYPE_F16)
        else:
            tensors[name] = (arr.astype(np.float32), GGML_TYPE_F32)

    # Detect config
    nf = state["conv_first.weight"].shape[0]
    unf = state["upconv1.weight"].shape[0]
    nb = 0
    while f"SCPA_trunk.block{nb}.conv1_a.weight" in state:
        nb += 1
    scale = 4 if "upconv2.weight" in state else 2

    print(f"PAN: nf={nf}, unf={unf}, nb={nb}, scale={scale}")
    print(f"Total params: {sum(v.size for v in state.values() if isinstance(v, np.ndarray)):,}")

    # conv_first
    add("conv_first.weight", "conv_first.weight")
    add("conv_first.bias", "conv_first.bias")

    # SCPA blocks
    for i in range(nb):
        p = f"SCPA_trunk.block{i}"
        d = f"scpa.{i}"
        add(f"{d}.conv1_a.weight", f"{p}.conv1_a.weight")
        add(f"{d}.conv1_b.weight", f"{p}.conv1_b.weight")
        add(f"{d}.k1.weight", f"{p}.k1.0.weight")
        add(f"{d}.paconv.k2.weight", f"{p}.PAConv.k2.weight")
        add(f"{d}.paconv.k2.bias", f"{p}.PAConv.k2.bias")
        add(f"{d}.paconv.k3.weight", f"{p}.PAConv.k3.weight")
        add(f"{d}.paconv.k4.weight", f"{p}.PAConv.k4.weight")
        add(f"{d}.conv3.weight", f"{p}.conv3.weight")

    # trunk_conv
    add("trunk_conv.weight", "trunk_conv.weight")
    add("trunk_conv.bias", "trunk_conv.bias")

    # Upsample path
    add("upconv1.weight", "upconv1.weight")
    add("upconv1.bias", "upconv1.bias")
    add("att1.weight", "att1.conv.weight")
    add("att1.bias", "att1.conv.bias")
    add("hrconv1.weight", "HRconv1.weight")
    add("hrconv1.bias", "HRconv1.bias")

    if scale == 4:
        add("upconv2.weight", "upconv2.weight")
        add("upconv2.bias", "upconv2.bias")
        add("att2.weight", "att2.conv.weight")
        add("att2.bias", "att2.conv.bias")
        add("hrconv2.weight", "HRconv2.weight")
        add("hrconv2.bias", "HRconv2.bias")

    # conv_last
    add("conv_last.weight", "conv_last.weight")
    add("conv_last.bias", "conv_last.bias")

    print(f"GGUF tensors: {len(tensors)}")

    n_kv = 6
    with open(args.output, "wb") as f:
        f.write(struct.pack("<I", GGUF_MAGIC))
        f.write(struct.pack("<I", GGUF_VERSION))
        f.write(struct.pack("<Q", len(tensors)))
        f.write(struct.pack("<Q", n_kv))

        write_kv_string(f, "general.architecture", "pan")
        write_kv_string(f, "general.name", f"PAN-x{scale}-nf{nf}")
        write_kv_u32(f, "pan.nf", nf)
        write_kv_u32(f, "pan.unf", unf)
        write_kv_u32(f, "pan.nb", nb)
        write_kv_u32(f, "pan.scale", scale)

        offset = 0
        tensor_list = list(tensors.items())
        for name, (data, dtype_id) in tensor_list:
            write_string(f, name)
            f.write(struct.pack("<I", len(data.shape)))
            for d in data.shape:
                f.write(struct.pack("<Q", d))
            f.write(struct.pack("<I", dtype_id))
            f.write(struct.pack("<Q", offset))
            offset += data.nbytes
            offset = (offset + 31) & ~31

        pos = f.tell()
        aligned = (pos + 31) & ~31
        f.write(b"\x00" * (aligned - pos))

        for name, (data, dtype_id) in tensor_list:
            f.write(data.tobytes())
            pad = ((data.nbytes + 31) & ~31) - data.nbytes
            if pad > 0:
                f.write(b"\x00" * pad)

    size_mb = sum(d.nbytes for d, _ in tensors.values()) / 1024 / 1024
    print(f"Written: {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
