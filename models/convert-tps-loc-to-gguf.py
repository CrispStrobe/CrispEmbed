#!/usr/bin/env python3
"""Convert TPS localization network weights to GGUF.

Supports two source formats:
  1. PaddleOCR recognition model with TPS transform (--paddle)
     Reads .pdparams via pickle (no PaddlePaddle installation needed).
  2. PyTorch state dict from deep-text-recognition-benchmark (--pytorch)

The localization network is a small CNN (4 ConvBN+ReLU + 2 FC):
  Conv0: 3  → 16, 3x3, pad1 + BN + ReLU + MaxPool2x2
  Conv1: 16 → 32, 3x3, pad1 + BN + ReLU + MaxPool2x2
  Conv2: 32 → 64, 3x3, pad1 + BN + ReLU + MaxPool2x2
  Conv3: 64 → 128, 3x3, pad1 + BN + ReLU + AdaptiveAvgPool(1)
  FC1: 128 → 64 + ReLU
  FC2: 64 → num_fiducial * 2

Output: control point coordinates in [-1, 1] normalized space.

Usage:
    python models/convert-tps-loc-to-gguf.py \\
        --paddle /tmp/rec_mv3_tps_bilstm_att_v2.0_train \\
        --output tps-loc-f32.gguf

    python models/convert-tps-loc-to-gguf.py \\
        --pytorch /path/to/state_dict.pth \\
        --output tps-loc-f32.gguf --fp16
"""

import argparse
import pickle
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

try:
    import gguf
except ImportError:
    sys.exit("pip install gguf")


# ---------------------------------------------------------------------------
# BatchNorm folding
# ---------------------------------------------------------------------------
def fold_bn_into_conv(conv_w, conv_b, bn_w, bn_b, bn_mean, bn_var, eps=1e-5):
    """Fold BatchNorm into preceding Conv2d weights."""
    oc = conv_w.shape[0]
    inv_std = 1.0 / np.sqrt(bn_var + eps)
    scale = bn_w * inv_std
    shape = [oc] + [1] * (conv_w.ndim - 1)
    w_folded = conv_w * scale.reshape(shape)
    if conv_b is not None:
        b_folded = (conv_b - bn_mean) * scale + bn_b
    else:
        b_folded = -bn_mean * scale + bn_b
    return w_folded, b_folded


# ---------------------------------------------------------------------------
# PaddleOCR loader (pickle-based, no PaddlePaddle needed)
# ---------------------------------------------------------------------------
def load_paddle(model_dir):
    """Load TPS localization weights from a PaddleOCR .pdparams file.

    PaddleOCR .pdparams files are pickle-serialized dicts of numpy arrays.
    We read them directly without importing PaddlePaddle.

    Actual key format (verified against rec_mv3_tps_bilstm_att_v2.0):
      transform.loc_net.loc_conv{0-3}.conv.weight  — [OC, IC, 3, 3]
      transform.loc_net.loc_conv{0-3}.bn.weight     — [OC]
      transform.loc_net.loc_conv{0-3}.bn.bias       — [OC]
      transform.loc_net.loc_conv{0-3}.bn._mean       — [OC]
      transform.loc_net.loc_conv{0-3}.bn._variance   — [OC]
      transform.loc_net.fc1.weight                  — [128, 64]
      transform.loc_net.fc1.bias                    — [64]
      transform.loc_net.fc2.weight                  — [64, N*2]
      transform.loc_net.fc2.bias                    — [N*2]
    """
    pdparams = Path(model_dir) / "best_accuracy.pdparams"
    if not pdparams.exists():
        # Try bare .pdparams file path
        pdparams = Path(model_dir)
    if not pdparams.exists():
        sys.exit(f"Cannot find .pdparams at {model_dir}")

    with open(pdparams, "rb") as f:
        sd = pickle.load(f)

    return extract_paddle_weights(sd)


def extract_paddle_weights(sd):
    """Extract and fold BN for PaddleOCR localization net."""
    tensors = OrderedDict()

    # Auto-detect prefix by finding loc_conv0
    prefix = None
    for k in sd:
        if "loc_conv0.conv.weight" in k:
            prefix = k.rsplit("loc_conv0.conv.weight", 1)[0]
            break
    if prefix is None:
        # Try older naming: loc_conv0_weights
        for k in sd:
            if "loc_conv0_weights" in k:
                prefix = k.rsplit("loc_conv0_weights", 1)[0]
                break
    if prefix is None:
        sys.exit("Could not find loc_conv0 keys. Available keys:\n  " +
                 "\n  ".join(k for k in sorted(sd.keys()) if "loc" in k or "transform" in k))

    def get(name):
        v = sd.get(name)
        if v is None:
            sys.exit(f"Missing key: {name}")
        return np.asarray(v).astype(np.float32)

    # Try new-style keys first (transform.loc_net.loc_conv0.conv.weight)
    new_style = f"{prefix}loc_conv0.conv.weight" in sd

    for i in range(4):
        if new_style:
            conv_w = get(f"{prefix}loc_conv{i}.conv.weight")
            bn_w = get(f"{prefix}loc_conv{i}.bn.weight")
            bn_b = get(f"{prefix}loc_conv{i}.bn.bias")
            bn_mean = get(f"{prefix}loc_conv{i}.bn._mean")
            bn_var = get(f"{prefix}loc_conv{i}.bn._variance")
        else:
            # Old-style keys: loc_conv0_weights, bn_loc_conv0_scale, etc.
            conv_w = get(f"{prefix}loc_conv{i}_weights")
            bn_w = get(f"{prefix}bn_loc_conv{i}_scale")
            bn_b = get(f"{prefix}bn_loc_conv{i}_offset")
            bn_mean = get(f"{prefix}bn_loc_conv{i}_mean")
            bn_var = get(f"{prefix}bn_loc_conv{i}_variance")

        w, b = fold_bn_into_conv(conv_w, None, bn_w, bn_b, bn_mean, bn_var)
        tensors[f"loc.conv{i}.weight"] = w
        tensors[f"loc.conv{i}.bias"] = b
        print(f"  conv{i}: {list(conv_w.shape)} → folded")

    if new_style:
        tensors["loc.fc1.weight"] = get(f"{prefix}fc1.weight")
        tensors["loc.fc1.bias"] = get(f"{prefix}fc1.bias")
        tensors["loc.fc2.weight"] = get(f"{prefix}fc2.weight")
        tensors["loc.fc2.bias"] = get(f"{prefix}fc2.bias")
    else:
        tensors["loc.fc1.weight"] = get(f"{prefix}loc_fc1_w")
        tensors["loc.fc1.bias"] = get(f"{prefix}loc_fc1.b_0")
        tensors["loc.fc2.weight"] = get(f"{prefix}loc_fc2_w")
        tensors["loc.fc2.bias"] = get(f"{prefix}loc_fc2_b")

    print(f"  fc1: {list(tensors['loc.fc1.weight'].shape)}")
    print(f"  fc2: {list(tensors['loc.fc2.weight'].shape)}")

    return tensors


# ---------------------------------------------------------------------------
# PyTorch loader (deep-text-recognition-benchmark / aster.pytorch)
# ---------------------------------------------------------------------------
def load_pytorch(path):
    """Load from PyTorch state dict."""
    import torch
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if "state_dict" in sd:
        sd = sd["state_dict"]

    numpy_sd = {k: v.cpu().numpy() for k, v in sd.items()}
    tensors = OrderedDict()

    # Find the prefix ending before "loc_conv0" or "block_list.0"
    loc_prefix = None
    for k in numpy_sd:
        if "loc_conv0.conv.weight" in k:
            loc_prefix = k.rsplit("loc_conv0.conv.weight", 1)[0]
            break
    if loc_prefix is None:
        for k in numpy_sd:
            if "block_list.0.conv.weight" in k:
                loc_prefix = k.rsplit("block_list.0.conv.weight", 1)[0]
                break
    if loc_prefix is None:
        sys.exit("Could not find localization network keys. Available: " +
                 ", ".join(sorted(numpy_sd.keys())[:20]))

    # Check which key format
    block_style = f"{loc_prefix}block_list.0.conv.weight" in numpy_sd

    for i in range(4):
        if block_style:
            idx = i * 2  # block_list: 0,2,4,6 are conv; 1,3,5,7 are pools
            conv_w = numpy_sd[f"{loc_prefix}block_list.{idx}.conv.weight"]
            conv_b = numpy_sd.get(f"{loc_prefix}block_list.{idx}.conv.bias")
            bn_w = numpy_sd.get(f"{loc_prefix}block_list.{idx}.bn.weight")
            bn_b = numpy_sd.get(f"{loc_prefix}block_list.{idx}.bn.bias")
            bn_mean = numpy_sd.get(f"{loc_prefix}block_list.{idx}.bn.running_mean")
            bn_var = numpy_sd.get(f"{loc_prefix}block_list.{idx}.bn.running_var")
        else:
            conv_w = numpy_sd[f"{loc_prefix}loc_conv{i}.conv.weight"]
            conv_b = numpy_sd.get(f"{loc_prefix}loc_conv{i}.conv.bias")
            bn_w = numpy_sd.get(f"{loc_prefix}loc_conv{i}.bn.weight")
            bn_b = numpy_sd.get(f"{loc_prefix}loc_conv{i}.bn.bias")
            bn_mean = numpy_sd.get(f"{loc_prefix}loc_conv{i}.bn.running_mean")
            bn_var = numpy_sd.get(f"{loc_prefix}loc_conv{i}.bn.running_var")

        if bn_w is not None:
            w, b = fold_bn_into_conv(conv_w, conv_b, bn_w, bn_b, bn_mean, bn_var)
        else:
            w = conv_w
            b = conv_b if conv_b is not None else np.zeros(conv_w.shape[0], dtype=np.float32)

        tensors[f"loc.conv{i}.weight"] = w
        tensors[f"loc.conv{i}.bias"] = b

    tensors["loc.fc1.weight"] = numpy_sd[f"{loc_prefix}fc1.weight"]
    tensors["loc.fc1.bias"] = numpy_sd[f"{loc_prefix}fc1.bias"]
    tensors["loc.fc2.weight"] = numpy_sd[f"{loc_prefix}fc2.weight"]
    tensors["loc.fc2.bias"] = numpy_sd[f"{loc_prefix}fc2.bias"]

    return tensors


# ---------------------------------------------------------------------------
# GGUF writer
# ---------------------------------------------------------------------------
def write_gguf(tensors, output_path, use_fp16=False):
    writer = gguf.GGUFWriter(output_path, "tps-localization")

    num_fiducial = tensors["loc.fc2.bias"].shape[0] // 2
    fc_dim = tensors["loc.fc1.bias"].shape[0]

    channels = []
    for i in range(4):
        w = tensors[f"loc.conv{i}.weight"]
        channels.append(w.shape[0])

    writer.add_uint32("tps.num_fiducial", num_fiducial)
    writer.add_uint32("tps.fc_dim", fc_dim)
    writer.add_uint32("tps.num_conv", 4)
    writer.add_array("tps.channels", channels)

    dtype = gguf.GGMLQuantizationType.F16 if use_fp16 else gguf.GGMLQuantizationType.F32

    for name, arr in tensors.items():
        arr = arr.astype(np.float32)
        if ".bias" in name:
            writer.add_tensor(name, arr, raw_dtype=gguf.GGMLQuantizationType.F32)
        else:
            writer.add_tensor(name, arr, raw_dtype=dtype)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    total_params = sum(t.size for t in tensors.values())
    file_size = Path(output_path).stat().st_size
    print(f"\nWritten {output_path}: {total_params:,} params, {file_size/1024:.0f} KB")
    print(f"  Fiducial points: {num_fiducial}, FC dim: {fc_dim}")
    print(f"  Channels: {channels}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Convert TPS localization net to GGUF")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--paddle", type=str, help="PaddleOCR model directory or .pdparams file")
    group.add_argument("--pytorch", type=str, help="PyTorch state dict path")
    parser.add_argument("--output", "-o", required=True, help="Output GGUF path")
    parser.add_argument("--fp16", action="store_true", help="Store weights in F16")
    args = parser.parse_args()

    if args.paddle:
        tensors = load_paddle(args.paddle)
    else:
        tensors = load_pytorch(args.pytorch)

    write_gguf(tensors, args.output, use_fp16=args.fp16)


if __name__ == "__main__":
    main()
