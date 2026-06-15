#!/usr/bin/env python3
"""Convert NAFNet-SR checkpoint to GGUF for CrispEmbed text super-resolution.

Usage:
    python convert-text-sr-to-gguf.py \
        --model nafnet-sr-x2.pth \
        --output text-sr-x2-f16.gguf --fp16 --upscale 2

Same NAFNet U-Net architecture as the denoising model, but the ending conv
outputs 3*r*r channels (for PixelShuffle upscaling) instead of 3, and the
global residual is bicubic-upscaled input (handled in C++ inference).

The checkpoint can be:
  - A plain state_dict (.pth)
  - A training checkpoint with 'params' or 'state_dict' key
"""

import argparse
import struct
import sys

import numpy as np
import torch

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9


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


def write_kv_i32_array(f, key, arr):
    write_string(f, key)
    f.write(struct.pack("<I", GGUF_TYPE_ARRAY))
    f.write(struct.pack("<I", GGUF_TYPE_INT32))
    f.write(struct.pack("<Q", len(arr)))
    for v in arr:
        f.write(struct.pack("<i", v))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="NAFNet-SR .pth checkpoint")
    parser.add_argument("--output", required=True, help="Output GGUF path")
    parser.add_argument("--fp16", action="store_true", help="Store weights as F16")
    parser.add_argument("--upscale", type=int, default=2, choices=[2, 4],
                        help="Upscale factor (default: 2)")
    args = parser.parse_args()

    state = torch.load(args.model, map_location="cpu", weights_only=True)
    if "params" in state:
        state = state["params"]
    elif "state_dict" in state:
        state = state["state_dict"]

    # Detect config from weight shapes
    enc_blk_nums = []
    dec_blk_nums = []
    stage = 0
    while True:
        if f"encoders.{stage}.0.conv1.weight" not in state:
            break
        blk = 0
        while f"encoders.{stage}.{blk}.conv1.weight" in state:
            blk += 1
        enc_blk_nums.append(blk)
        stage += 1

    stage = 0
    while True:
        if f"decoders.{stage}.0.conv1.weight" not in state:
            break
        blk = 0
        while f"decoders.{stage}.{blk}.conv1.weight" in state:
            blk += 1
        dec_blk_nums.append(blk)
        stage += 1

    blk = 0
    while f"middle_blks.{blk}.conv1.weight" in state:
        blk += 1
    middle_blk_num = blk

    width = state["intro.weight"].shape[0]

    # Validate ending conv: should output 3*r*r channels for SR
    ending_oc = state["ending.weight"].shape[0]
    r = args.upscale
    expected_oc = 3 * r * r
    if ending_oc != expected_oc:
        print(f"WARNING: ending.weight has {ending_oc} output channels, "
              f"expected {expected_oc} for {r}x upscale")
        if ending_oc == 3:
            print("This looks like a denoising model, not SR. "
                  "Use convert-nafnet-to-gguf.py instead.")
            sys.exit(1)

    print(f"NAFNet-SR: width={width}, upscale={r}x, "
          f"enc={enc_blk_nums}, middle={middle_blk_num}, dec={dec_blk_nums}")
    print(f"Ending conv: {width} -> {ending_oc} channels "
          f"(PixelShuffle {r}x -> 3 RGB channels)")
    print(f"Total params: {sum(p.numel() for p in state.values()):,}")

    tensors = {}

    def add(name, tensor):
        arr = tensor.float().numpy()
        if args.fp16 and arr.ndim >= 2:
            tensors[name] = (arr.astype(np.float16), GGML_TYPE_F16)
        else:
            tensors[name] = (arr.astype(np.float32), GGML_TYPE_F32)

    # Intro / ending
    add("intro.weight", state["intro.weight"])
    add("intro.bias", state["intro.bias"])
    add("ending.weight", state["ending.weight"])
    add("ending.bias", state["ending.bias"])

    # Down/up sampling
    for i in range(len(enc_blk_nums)):
        add(f"downs.{i}.weight", state[f"downs.{i}.weight"])
        add(f"downs.{i}.bias", state[f"downs.{i}.bias"])
    for i in range(len(dec_blk_nums)):
        add(f"ups.{i}.weight", state[f"ups.{i}.0.weight"])

    # NAFBlock suffixes
    block_suffixes = [
        "beta", "gamma",
        "conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias",
        "conv3.weight", "conv3.bias", "sca.1.weight", "sca.1.bias",
        "conv4.weight", "conv4.bias", "conv5.weight", "conv5.bias",
        "norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias",
    ]

    for s in range(len(enc_blk_nums)):
        for b in range(enc_blk_nums[s]):
            for suffix in block_suffixes:
                key = f"encoders.{s}.{b}.{suffix}"
                dst = f"enc.{s}.{b}.{suffix.replace('sca.1', 'sca')}"
                add(dst, state[key])

    for b in range(middle_blk_num):
        for suffix in block_suffixes:
            key = f"middle_blks.{b}.{suffix}"
            dst = f"mid.{b}.{suffix.replace('sca.1', 'sca')}"
            add(dst, state[key])

    for s in range(len(dec_blk_nums)):
        for b in range(dec_blk_nums[s]):
            for suffix in block_suffixes:
                key = f"decoders.{s}.{b}.{suffix}"
                dst = f"dec.{s}.{b}.{suffix.replace('sca.1', 'sca')}"
                add(dst, state[key])

    print(f"GGUF tensors: {len(tensors)}")

    n_kv = 8  # one more than nafnet (upscale_factor)
    n_tensors = len(tensors)

    with open(args.output, "wb") as f:
        f.write(struct.pack("<I", GGUF_MAGIC))
        f.write(struct.pack("<I", GGUF_VERSION))
        f.write(struct.pack("<Q", n_tensors))
        f.write(struct.pack("<Q", n_kv))

        write_kv_string(f, "general.architecture", "text_sr")
        write_kv_string(f, "general.name", f"NAFNet-SR-{r}x-width{width}")
        write_kv_u32(f, "text_sr.width", width)
        write_kv_u32(f, "text_sr.middle_blk_num", middle_blk_num)
        write_kv_i32_array(f, "text_sr.enc_blk_nums", enc_blk_nums)
        write_kv_i32_array(f, "text_sr.dec_blk_nums", dec_blk_nums)
        write_kv_u32(f, "text_sr.n_stages", len(enc_blk_nums))
        write_kv_u32(f, "text_sr.upscale_factor", r)

        offset = 0
        tensor_list = list(tensors.items())
        for name, (data, dtype_id) in tensor_list:
            write_string(f, name)
            n_dims = len(data.shape)
            f.write(struct.pack("<I", n_dims))
            for d in data.shape:
                f.write(struct.pack("<Q", d))
            f.write(struct.pack("<I", dtype_id))
            f.write(struct.pack("<Q", offset))
            nbytes = data.nbytes
            offset += nbytes
            offset = (offset + 31) & ~31

        pos = f.tell()
        aligned = (pos + 31) & ~31
        f.write(b"\x00" * (aligned - pos))

        for name, (data, dtype_id) in tensor_list:
            f.write(data.tobytes())
            nbytes = data.nbytes
            pad = ((nbytes + 31) & ~31) - nbytes
            if pad > 0:
                f.write(b"\x00" * pad)

    size_mb = sum(d.nbytes for d, _ in tensors.values()) / 1024 / 1024
    print(f"Written: {args.output} ({size_mb:.1f} MB tensor data)")


if __name__ == "__main__":
    main()
