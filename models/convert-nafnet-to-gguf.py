#!/usr/bin/env python3
"""Convert NAFNet checkpoint to GGUF for CrispEmbed scan cleanup (tier 2).

Usage:
    python convert-nafnet-to-gguf.py \
        --model NAFNet-SIDD-width32.pth \
        --output nafnet-sidd-w32-f16.gguf --fp16

NAFNet architecture (megvii-research/NAFNet, MIT license):
  U-Net with NAFBlocks (Non-linear Activation Free).
  NAFBlock = LN → 1x1 → 3x3 DW → SimpleGate → SCA → 1x1 → res*beta
           + LN → 1x1 → SimpleGate → 1x1 → res*gamma

  width32 SIDD config: enc=[2,2,4,8], middle=12, dec=[2,2,2,2]
  Channels: 32→64→128→256→512 (middle) → 256→128→64→32
  ~29M params, 111 MB F32, ~29 MB Q8_0.
"""

import argparse
import struct
import sys

import numpy as np
import torch

# GGUF constants
GGUF_MAGIC = 0x46554747  # "GGUF"
GGUF_VERSION = 3
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1

# GGUF metadata value types
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
    f.write(struct.pack("<I", GGUF_TYPE_INT32))  # element type
    f.write(struct.pack("<Q", len(arr)))
    for v in arr:
        f.write(struct.pack("<i", v))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="NAFNet .pth checkpoint")
    parser.add_argument("--output", required=True, help="Output GGUF path")
    parser.add_argument("--fp16", action="store_true", help="Store weights as F16")
    args = parser.parse_args()

    # Load checkpoint
    state = torch.load(args.model, map_location="cpu", weights_only=True)
    if "params" in state:
        state = state["params"]
    elif "state_dict" in state:
        state = state["state_dict"]

    # Detect config from weight shapes
    # Count encoder blocks per stage
    enc_blk_nums = []
    dec_blk_nums = []
    middle_blk_num = 0
    stage = 0
    while True:
        key = f"encoders.{stage}.0.conv1.weight"
        if key not in state:
            break
        blk = 0
        while f"encoders.{stage}.{blk}.conv1.weight" in state:
            blk += 1
        enc_blk_nums.append(blk)
        stage += 1

    stage = 0
    while True:
        key = f"decoders.{stage}.0.conv1.weight"
        if key not in state:
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
    print(f"NAFNet: width={width}, enc={enc_blk_nums}, middle={middle_blk_num}, dec={dec_blk_nums}")
    print(f"Total params: {sum(p.numel() for p in state.values()):,}")

    # Prepare tensors with clean names
    tensors = {}
    dtype = np.float16 if args.fp16 else np.float32
    ggml_type = GGML_TYPE_F16 if args.fp16 else GGML_TYPE_F32

    def add(name, tensor):
        arr = tensor.float().numpy()
        if args.fp16 and arr.ndim >= 2:
            tensors[name] = (arr.astype(np.float16), GGML_TYPE_F16)
        else:
            tensors[name] = (arr.astype(np.float32), GGML_TYPE_F32)

    # Intro / ending convolutions
    add("intro.weight", state["intro.weight"])
    add("intro.bias", state["intro.bias"])
    add("ending.weight", state["ending.weight"])
    add("ending.bias", state["ending.bias"])

    # Downsampling convolutions
    for i in range(len(enc_blk_nums)):
        add(f"downs.{i}.weight", state[f"downs.{i}.weight"])
        add(f"downs.{i}.bias", state[f"downs.{i}.bias"])

    # Upsampling 1x1 convolutions (for PixelShuffle)
    for i in range(len(dec_blk_nums)):
        add(f"ups.{i}.weight", state[f"ups.{i}.0.weight"])

    # Encoder NAFBlocks
    for s in range(len(enc_blk_nums)):
        for b in range(enc_blk_nums[s]):
            prefix_src = f"encoders.{s}.{b}"
            prefix_dst = f"enc.{s}.{b}"
            for suffix in ["beta", "gamma",
                           "conv1.weight", "conv1.bias",
                           "conv2.weight", "conv2.bias",
                           "conv3.weight", "conv3.bias",
                           "sca.1.weight", "sca.1.bias",
                           "conv4.weight", "conv4.bias",
                           "conv5.weight", "conv5.bias",
                           "norm1.weight", "norm1.bias",
                           "norm2.weight", "norm2.bias"]:
                key = f"{prefix_src}.{suffix}"
                dst = f"{prefix_dst}.{suffix.replace('sca.1', 'sca')}"
                add(dst, state[key])

    # Middle NAFBlocks
    for b in range(middle_blk_num):
        prefix_src = f"middle_blks.{b}"
        prefix_dst = f"mid.{b}"
        for suffix in ["beta", "gamma",
                       "conv1.weight", "conv1.bias",
                       "conv2.weight", "conv2.bias",
                       "conv3.weight", "conv3.bias",
                       "sca.1.weight", "sca.1.bias",
                       "conv4.weight", "conv4.bias",
                       "conv5.weight", "conv5.bias",
                       "norm1.weight", "norm1.bias",
                       "norm2.weight", "norm2.bias"]:
            key = f"{prefix_src}.{suffix}"
            dst = f"{prefix_dst}.{suffix.replace('sca.1', 'sca')}"
            add(dst, state[key])

    # Decoder NAFBlocks
    for s in range(len(dec_blk_nums)):
        for b in range(dec_blk_nums[s]):
            prefix_src = f"decoders.{s}.{b}"
            prefix_dst = f"dec.{s}.{b}"
            for suffix in ["beta", "gamma",
                           "conv1.weight", "conv1.bias",
                           "conv2.weight", "conv2.bias",
                           "conv3.weight", "conv3.bias",
                           "sca.1.weight", "sca.1.bias",
                           "conv4.weight", "conv4.bias",
                           "conv5.weight", "conv5.bias",
                           "norm1.weight", "norm1.bias",
                           "norm2.weight", "norm2.bias"]:
                key = f"{prefix_src}.{suffix}"
                dst = f"{prefix_dst}.{suffix.replace('sca.1', 'sca')}"
                add(dst, state[key])

    print(f"GGUF tensors: {len(tensors)}")

    # Write GGUF
    n_kv = 7  # metadata keys
    n_tensors = len(tensors)

    with open(args.output, "wb") as f:
        # Header
        f.write(struct.pack("<I", GGUF_MAGIC))
        f.write(struct.pack("<I", GGUF_VERSION))
        f.write(struct.pack("<Q", n_tensors))
        f.write(struct.pack("<Q", n_kv))

        # Metadata
        write_kv_string(f, "general.architecture", "nafnet")
        write_kv_string(f, "general.name", "NAFNet-SIDD-width32")
        write_kv_u32(f, "nafnet.width", width)
        write_kv_u32(f, "nafnet.middle_blk_num", middle_blk_num)
        write_kv_i32_array(f, "nafnet.enc_blk_nums", enc_blk_nums)
        write_kv_i32_array(f, "nafnet.dec_blk_nums", dec_blk_nums)
        write_kv_u32(f, "nafnet.n_stages", len(enc_blk_nums))

        # Tensor info
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
            # Align to 32 bytes
            offset = (offset + 31) & ~31

        # Align data start to 32 bytes
        pos = f.tell()
        aligned = (pos + 31) & ~31
        f.write(b"\x00" * (aligned - pos))

        # Tensor data
        for name, (data, dtype_id) in tensor_list:
            f.write(data.tobytes())
            # Pad to 32-byte alignment
            nbytes = data.nbytes
            pad = ((nbytes + 31) & ~31) - nbytes
            if pad > 0:
                f.write(b"\x00" * pad)

    size_mb = sum(d.nbytes for d, _ in tensors.values()) / 1024 / 1024
    print(f"Written: {args.output} ({size_mb:.1f} MB tensor data)")


if __name__ == "__main__":
    main()
