#!/usr/bin/env python3
"""Convert the EasyOCR CRAFT checkpoint to a native GGUF weight archive."""

import argparse

import gguf
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "state_dict" in state:
        state = state["state_dict"]
    tensors = {}
    for name, value in state.items():
        name = name[7:] if name.startswith("module.") else name
        if torch.is_tensor(value):
            tensors[name] = value.detach().float().cpu().numpy()

    # Fold every BatchNorm into the preceding convolution. CRAFT uses the
    # same two-convolution pattern in the VGG slices and U-Net double_conv.
    bn_prefixes = sorted(name.rsplit(".", 1)[0] for name in tensors if name.endswith(".running_var"))
    for bp in bn_prefixes:
        parts = bp.split(".")
        if not parts[-1].isdigit():
            raise KeyError(f"unmapped CRAFT BatchNorm: {bp}")
        cp = ".".join(parts[:-1] + [str(int(parts[-1]) - 1)])
        wp = f"{cp}.weight"
        if wp not in tensors:
            raise KeyError(f"missing convolution for {bp}: {wp}")
        gamma = tensors[f"{bp}.weight"]
        beta = tensors[f"{bp}.bias"]
        mean = tensors[f"{bp}.running_mean"]
        var = tensors[f"{bp}.running_var"]
        scale = gamma / np.sqrt(var + 1.0e-5)
        tensors[f"{cp}.raw_weight"] = tensors[wp].copy()
        tensors[f"{cp}.raw_bias"] = tensors[f"{cp}.bias"].copy()
        tensors[f"{cp}.bn_scale"] = scale
        tensors[f"{cp}.bn_shift"] = beta - mean * scale
        tensors[wp] = tensors[wp] * scale[:, None, None, None]
        tensors[f"{cp}.bias"] = beta - mean * scale
        for suffix in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
            tensors.pop(f"{bp}.{suffix}", None)

    writer = gguf.GGUFWriter(args.output, arch="easyocr-craft")
    writer.add_string("general.name", "easyocr-craft")
    writer.add_string("general.source", "JaidedAI/EasyOCR / CRAFT")
    writer.add_string("general.license", "BSD-2-Clause")
    writer.add_uint32("easyocr.input_channels", 3)
    writer.add_uint32("easyocr.craft.num_classes", 2)
    writer.add_uint32("easyocr.craft.bn_folded", 1)
    writer.add_uint32("easyocr.craft.bn_runtime", 1)

    for name in sorted(tensors):
        data = tensors[name]
        dtype = gguf.GGMLQuantizationType.F32
        # Keep the runtime-BN source weights in F32. The folded copies remain
        # available for older consumers, while explicit BN preserves the
        # Python Conv→BatchNorm evaluation order at decoded-output thresholds.
        keep_f32 = name.endswith((".raw_weight", ".raw_bias", ".bn_scale", ".bn_shift"))
        if args.fp16 and not keep_f32 and data.ndim >= 2 and data.size >= 256:
            data = data.astype(np.float16)
            dtype = gguf.GGMLQuantizationType.F16
        writer.add_tensor(name, data, raw_dtype=dtype)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {args.output}: {len(tensors)} tensors")


if __name__ == "__main__":
    main()
