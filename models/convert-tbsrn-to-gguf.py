#!/usr/bin/env python3
"""Convert PaddleOCR TBSRN (Telescope) checkpoint to GGUF for CrispEmbed.

Usage:
    # From PaddleOCR sr_telescope_train.tar (extract first):
    python convert-tbsrn-to-gguf.py \
        --model sr_telescope/best_accuracy.pdparams \
        --output tbsrn-telescope-f16.gguf --fp16

TBSRN architecture (scene-text-telescope, Apache-2.0):
    Conv9(3→64)+PReLU → 5× RecurrentResidualBlock → Conv3(64→64)+BN
    → skip + UpsampleBlock(PixelShuffle 2×) → Conv9(64→3) → tanh

    RecurrentResidualBlock:
      Conv3+BN+mish+Conv3+BN → FeatureEnhancer → residual
      (GRU weights in checkpoint are dead code — TBSRN forward skips them)

    FeatureEnhancer:
      Concat(input, PE2D) → MHA(h=4, d=128) + LN + FFN(128→128) + LN → Linear(128→64)

Input:  16×64  (LR text line, RGB [-1,1])
Output: 32×128 (HR text line, RGB [-1,1], tanh)
"""

import argparse
import struct
import sys
import os

import numpy as np

# GGUF constants
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


def load_paddle_params(path):
    """Load .pdparams (pickle of numpy arrays)."""
    import pickle
    with open(path, "rb") as f:
        state = pickle.load(f)
    # Paddle state_dict values are numpy arrays
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="Path to .pdparams file (or directory with best_accuracy.pdparams)")
    parser.add_argument("--output", required=True, help="Output GGUF path")
    parser.add_argument("--fp16", action="store_true", help="Store 2D+ weights as F16")
    args = parser.parse_args()

    model_path = args.model
    if os.path.isdir(model_path):
        candidates = ["best_accuracy.pdparams", "latest.pdparams"]
        for c in candidates:
            p = os.path.join(model_path, c)
            if os.path.exists(p):
                model_path = p
                break

    print(f"Loading: {model_path}")
    state = load_paddle_params(model_path)

    # Print all keys for debugging
    print(f"Total keys in checkpoint: {len(state)}")
    for k in sorted(state.keys()):
        v = state[k]
        if isinstance(v, np.ndarray):
            print(f"  {k}: {v.shape} {v.dtype}")
        else:
            print(f"  {k}: {type(v)}")

    # TBSRN tensor name mapping: paddle key → GGUF name
    # Paddle keys have "transform." prefix. We skip: GRU weights (dead code
    # in TBSRN forward), STN/TPS weights (training only), transformer (frozen).
    tensors = {}

    def add(gguf_name, paddle_key, transpose=False, state_dict=state):
        if paddle_key not in state_dict:
            print(f"  WARNING: missing {paddle_key}, skipping")
            return False
        arr = state_dict[paddle_key]
        if not isinstance(arr, np.ndarray):
            print(f"  WARNING: {paddle_key} is not ndarray ({type(arr)}), skipping")
            return False
        arr = arr.astype(np.float32)
        # Paddle Linear stores [in, out]; C++ expects [out, in] → transpose 2D
        if transpose and arr.ndim == 2:
            arr = arr.T.copy()
        if args.fp16 and arr.ndim >= 2:
            tensors[gguf_name] = (arr.astype(np.float16), GGML_TYPE_F16)
        else:
            tensors[gguf_name] = (arr.astype(np.float32), GGML_TYPE_F32)
        return True

    # Detect prefix: PaddleOCR wraps in "transform."
    prefix = ""
    if "transform.block1.0.weight" in state:
        prefix = "transform."
    elif "block1.0.weight" not in state:
        print("ERROR: cannot find block1 weights")
        sys.exit(1)

    # block1: Conv2D(3→64, k=9, p=4) + PReLU
    add("block1.conv.weight", f"{prefix}block1.0.weight")
    add("block1.conv.bias", f"{prefix}block1.0.bias")
    add("block1.prelu.weight", f"{prefix}block1.1._weight")  # PReLU learnable slope

    # block2..block6: RecurrentResidualBlock (srb_nums=5)
    srb_nums = 5
    for i in range(srb_nums):
        blk = i + 2  # block2..block6
        ps = f"{prefix}block{blk}"
        dst = f"srb.{i}"

        # Conv1 + BN1
        add(f"{dst}.conv1.weight", f"{ps}.conv1.weight")
        add(f"{dst}.conv1.bias", f"{ps}.conv1.bias")
        add(f"{dst}.bn1.weight", f"{ps}.bn1.weight")
        add(f"{dst}.bn1.bias", f"{ps}.bn1.bias")
        add(f"{dst}.bn1.running_mean", f"{ps}.bn1._mean")
        add(f"{dst}.bn1.running_var", f"{ps}.bn1._variance")

        # Conv2 + BN2
        add(f"{dst}.conv2.weight", f"{ps}.conv2.weight")
        add(f"{dst}.conv2.bias", f"{ps}.conv2.bias")
        add(f"{dst}.bn2.weight", f"{ps}.bn2.weight")
        add(f"{dst}.bn2.bias", f"{ps}.bn2.bias")
        add(f"{dst}.bn2.running_mean", f"{ps}.bn2._mean")
        add(f"{dst}.bn2.running_var", f"{ps}.bn2._variance")

        # FeatureEnhancer: MHA + LN + FFN + LN + Linear
        fe = f"{ps}.feature_enhancer"
        fd = f"{dst}.fe"

        # MHA: 4 linear projections (Q, K, V, out) each [128, 128]
        for j in range(4):
            add(f"{fd}.mha.linear{j}.weight", f"{fe}.multihead.linears.{j}.weight", transpose=True)
            add(f"{fd}.mha.linear{j}.bias", f"{fe}.multihead.linears.{j}.bias")

        # LayerNorm after MHA
        add(f"{fd}.ln1.weight", f"{fe}.mul_layernorm1.a_2")
        add(f"{fd}.ln1.bias", f"{fe}.mul_layernorm1.b_2")

        # FFN: Linear(128→128) + ReLU + Linear(128→128)
        add(f"{fd}.ffn.w1.weight", f"{fe}.pff.w_1.weight", transpose=True)
        add(f"{fd}.ffn.w1.bias", f"{fe}.pff.w_1.bias")
        add(f"{fd}.ffn.w2.weight", f"{fe}.pff.w_2.weight", transpose=True)
        add(f"{fd}.ffn.w2.bias", f"{fe}.pff.w_2.bias")

        # LayerNorm after FFN
        add(f"{fd}.ln3.weight", f"{fe}.mul_layernorm3.a_2")
        add(f"{fd}.ln3.bias", f"{fe}.mul_layernorm3.b_2")

        # Output linear (128→64)
        add(f"{fd}.linear.weight", f"{fe}.linear.weight", transpose=True)
        add(f"{fd}.linear.bias", f"{fe}.linear.bias")

        # Skip GRU weights (gru1, gru2) — dead code in TBSRN forward

    # block7: Conv2D(64→64, k=3, p=1) + BN
    blk7 = srb_nums + 2  # = 7
    add("final_conv.weight", f"{prefix}block{blk7}.0.weight")
    add("final_conv.bias", f"{prefix}block{blk7}.0.bias")
    add("final_bn.weight", f"{prefix}block{blk7}.1.weight")
    add("final_bn.bias", f"{prefix}block{blk7}.1.bias")
    add("final_bn.running_mean", f"{prefix}block{blk7}.1._mean")
    add("final_bn.running_var", f"{prefix}block{blk7}.1._variance")

    # block8: UpsampleBlock(64, 2) + Conv9(64→3, k=9, p=4)
    blk8 = srb_nums + 3  # = 8
    # UpsampleBlock: Conv(64→256, k=3, p=1) + PixelShuffle(2) + mish
    add("upsample.conv.weight", f"{prefix}block{blk8}.0.conv.weight")
    add("upsample.conv.bias", f"{prefix}block{blk8}.0.conv.bias")
    # Final output conv: Conv(64→3, k=9, p=4)
    add("output_conv.weight", f"{prefix}block{blk8}.1.weight")
    add("output_conv.bias", f"{prefix}block{blk8}.1.bias")

    print(f"\nGGUF tensors: {len(tensors)}")
    total_params = sum(d.size for d, _ in tensors.values())
    print(f"Total params (inference): {total_params:,}")

    # Count skipped
    skipped = set()
    for k in state.keys():
        if isinstance(state[k], np.ndarray):
            # Check if any GGUF tensor maps to this
            mapped = False
            for gk, (_, _) in tensors.items():
                pass  # already mapped above
            if "gru" in k or "stn" in k or "tps" in k or "transformer" in k:
                skipped.add(k)
    if skipped:
        print(f"Skipped {len(skipped)} tensors (GRU/STN/transformer — not needed for inference)")

    # Write GGUF
    n_kv = 5
    n_tensors = len(tensors)

    with open(args.output, "wb") as f:
        f.write(struct.pack("<I", GGUF_MAGIC))
        f.write(struct.pack("<I", GGUF_VERSION))
        f.write(struct.pack("<Q", n_tensors))
        f.write(struct.pack("<Q", n_kv))

        # Metadata
        write_kv_string(f, "general.architecture", "tbsrn")
        write_kv_string(f, "general.name", "TBSRN-Telescope-PaddleOCR")
        write_kv_u32(f, "tbsrn.srb_nums", srb_nums)
        write_kv_u32(f, "tbsrn.hidden_units", 32)  # 2*hidden_units = 64 channels
        write_kv_u32(f, "tbsrn.upscale_factor", 2)

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
            offset = (offset + 31) & ~31

        # Align data start
        pos = f.tell()
        aligned = (pos + 31) & ~31
        f.write(b"\x00" * (aligned - pos))

        # Tensor data
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
