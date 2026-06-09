#!/usr/bin/env python3
"""Convert DBNet (ResNet-18 + FPNC + DBHead) text detection to GGUF.

Loads an MMOCR DBNet checkpoint (dbnet_resnet18_fpnc_1200e_icdar2015),
folds all BatchNorm into preceding Conv2d, and packs into a single GGUF
file for CrispEmbed's text detection inference.

Architecture:
  Backbone: ResNet-18 (4 stages, output channels [64, 128, 256, 512])
  Neck:     FPNC (FPN-Cat) — lateral 1×1 + smooth 3×3, concat + reduce
  Head:     DBNet probability branch — 3×3 conv + deconv + deconv → prob map

The threshold branch is NOT exported (only needed for training).
At inference, only the probability map is used for text detection.

Usage:
    pip install gguf numpy
    python models/convert-dbnet-to-gguf.py \
        --checkpoint dbnet_resnet18_fpnc_1200e_icdar2015_20220825_221614-7c0e94f2.pth \
        --output dbnet-ic15-f32.gguf

    # Then quantize:
    crispembed-quantize dbnet-ic15-f32.gguf dbnet-ic15-q4_k.gguf q4_k
"""

import argparse
import sys
from pathlib import Path

import gguf
import numpy as np


# ---------------------------------------------------------------------------
# BatchNorm folding (same as convert-hmer-to-gguf.py)
# ---------------------------------------------------------------------------

def fold_bn_into_conv(conv_w, conv_b, bn_weight, bn_bias, bn_mean, bn_var, eps=1e-5):
    """Fold BN(gamma, beta, mean, var) into conv weight + bias.

    conv_w: (out_ch, in_ch, kH, kW)
    conv_b: (out_ch,) or None
    Returns: (fused_w, fused_b)
    """
    out_ch = conv_w.shape[0]
    inv_std = bn_weight / np.sqrt(bn_var + eps)
    # Reshape for broadcasting: [OC, 1, 1, 1] for 4D conv, [OC, 1] for 2D
    shape = [out_ch] + [1] * (conv_w.ndim - 1)
    fused_w = conv_w * inv_std.reshape(shape)
    if conv_b is not None:
        fused_b = (conv_b - bn_mean) * inv_std + bn_bias
    else:
        fused_b = -bn_mean * inv_std + bn_bias
    return fused_w, fused_b


def fold_bn_into_deconv(deconv_w, deconv_b, bn_weight, bn_bias, bn_mean, bn_var, eps=1e-5):
    """Fold BN into ConvTranspose2d weight.

    deconv_w: (in_ch, out_ch, kH, kW) — note: transposed layout!
    The output channels are at dim 1, not dim 0.
    """
    out_ch = deconv_w.shape[1]
    inv_std = bn_weight / np.sqrt(bn_var + eps)
    # For ConvTranspose2d, output channels are at dim 1
    shape = [1, out_ch] + [1] * (deconv_w.ndim - 2)
    fused_w = deconv_w * inv_std.reshape(shape)
    if deconv_b is not None:
        fused_b = (deconv_b - bn_mean) * inv_std + bn_bias
    else:
        fused_b = -bn_mean * inv_std + bn_bias
    return fused_w, fused_b


# ---------------------------------------------------------------------------
# ResNet-18 BasicBlock structure
# ---------------------------------------------------------------------------

# ResNet-18 stage config: (n_blocks, out_channels, stride)
RESNET18_STAGES = [
    (2, 64,  1),   # stage 0 (layer1): 2 BasicBlocks, 64ch, stride 1
    (2, 128, 2),   # stage 1 (layer2): 2 BasicBlocks, 128ch, stride 2
    (2, 256, 2),   # stage 2 (layer3): 2 BasicBlocks, 256ch, stride 2
    (2, 512, 2),   # stage 3 (layer4): 2 BasicBlocks, 512ch, stride 2
]

FPN_IN_CHANNELS = [64, 128, 256, 512]
FPN_LATERAL_CHANNELS = 256


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Convert MMOCR DBNet to GGUF")
    p.add_argument("--checkpoint", required=True,
                   help="MMOCR checkpoint .pth file")
    p.add_argument("--output", required=True, help="Output GGUF path")
    p.add_argument("--fp16", action="store_true",
                   help="Store weights in FP16 (halves file size)")
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    # Load checkpoint (need torch for unpickling)
    import torch
    print(f"Loading checkpoint: {ckpt_path.name} ({ckpt_path.stat().st_size / 1e6:.1f} MB)")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    # MMOCR checkpoints have 'state_dict' key
    if "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt

    # Convert to numpy
    sd = {k: v.detach().float().cpu().numpy() for k, v in sd.items()}

    print(f"State dict: {len(sd)} tensors")

    # Print key prefixes for debugging
    prefixes = set()
    for k in sd:
        parts = k.split(".")
        if len(parts) >= 2:
            prefixes.add(f"{parts[0]}.{parts[1]}")
    print(f"Key prefixes: {sorted(prefixes)}")

    # -----------------------------------------------------------------------
    # Write GGUF
    # -----------------------------------------------------------------------
    writer = gguf.GGUFWriter(str(args.output), arch="dbnet")

    # Metadata
    writer.add_string("general.architecture", "dbnet")
    writer.add_string("general.name", "dbnet-resnet18-fpnc-icdar2015")
    writer.add_string("general.license", "Apache-2.0")
    writer.add_string("general.source", "open-mmlab/mmocr")

    # Architecture hyperparameters
    writer.add_string("dbnet.backbone", "resnet18")
    writer.add_uint32("dbnet.fpn_lateral_channels", FPN_LATERAL_CHANNELS)
    writer.add_uint32("dbnet.fpn_smooth_channels", 64)  # per-level after smooth conv
    writer.add_uint32("dbnet.fpn_out_channels", 256)   # 4×64 after concat
    writer.add_uint32("dbnet.head_in_channels", 256)    # concat feeds head
    writer.add_uint32("dbnet.num_stages", 4)
    writer.add_uint32("dbnet.pad_divisor", 32)

    # Preprocessing (ImageNet)
    writer.add_array("dbnet.image_mean", [123.675, 116.28, 103.53])
    writer.add_array("dbnet.image_std", [58.395, 57.12, 57.375])

    # Post-processing defaults
    writer.add_float32("dbnet.prob_threshold", 0.3)
    writer.add_float32("dbnet.box_threshold", 0.5)
    writer.add_float32("dbnet.unclip_ratio", 1.5)
    writer.add_uint32("dbnet.min_area", 10)

    dtype_np = np.float16 if args.fp16 else np.float32
    dtype_gguf = gguf.GGMLQuantizationType.F16 if args.fp16 else gguf.GGMLQuantizationType.F32

    total_params = 0
    tensor_count = 0

    def add_tensor(name, data, flatten_conv=False):
        nonlocal total_params, tensor_count
        d = data.astype(dtype_np)
        if flatten_conv and d.ndim == 4:
            d = d.reshape(d.shape[0], -1)
        total_params += d.size
        writer.add_tensor(name, d, raw_dtype=dtype_gguf)
        tensor_count += 1

    def get(key, default=None):
        """Get tensor from state dict with MMOCR prefix handling."""
        # Try exact key first
        if key in sd:
            return sd[key]
        # Try with backbone/neck/head prefixes
        for prefix in ["backbone.", "det_head.", "neck."]:
            full = prefix + key
            if full in sd:
                return sd[full]
        return default

    def require(key):
        val = get(key)
        if val is None:
            # Try listing similar keys
            similar = [k for k in sd if key.split(".")[-1] in k][:5]
            print(f"ERROR: tensor '{key}' not found. Similar: {similar}",
                  file=sys.stderr)
            sys.exit(1)
        return val

    # -----------------------------------------------------------------------
    # Backbone: ResNet-18 stem + 4 stages
    # -----------------------------------------------------------------------
    print("\n--- Backbone (ResNet-18) ---")

    # Stem: conv1 (7×7, stride 2) + bn1
    conv1_w = require("conv1.weight")      # (64, 3, 7, 7)
    bn1_w = require("bn1.weight")
    bn1_b = require("bn1.bias")
    bn1_m = require("bn1.running_mean")
    bn1_v = require("bn1.running_var")
    fused_w, fused_b = fold_bn_into_conv(conv1_w, None, bn1_w, bn1_b, bn1_m, bn1_v)
    add_tensor("det.backbone.stem.conv.weight", fused_w, flatten_conv=True)
    add_tensor("det.backbone.stem.conv.bias", fused_b)
    print(f"  stem: conv1(3→64, 7×7, s2) + bn1 → folded {list(fused_w.shape)}")

    # Stages: layer1..layer4, each has BasicBlocks
    for stage_idx, (n_blocks, out_ch, stride) in enumerate(RESNET18_STAGES):
        stage_key = f"layer{stage_idx + 1}"
        print(f"  {stage_key}: {n_blocks} blocks, {out_ch}ch, stride={stride}")

        for block_idx in range(n_blocks):
            block_key = f"{stage_key}.{block_idx}"
            gguf_prefix = f"det.backbone.stage{stage_idx}.block{block_idx}"

            # BasicBlock: conv1 + bn1 + relu + conv2 + bn2 + (downsample?)
            # conv1: 3×3 conv
            c1_w = require(f"{block_key}.conv1.weight")
            b1_w = require(f"{block_key}.bn1.weight")
            b1_b = require(f"{block_key}.bn1.bias")
            b1_m = require(f"{block_key}.bn1.running_mean")
            b1_v = require(f"{block_key}.bn1.running_var")
            f1_w, f1_b = fold_bn_into_conv(c1_w, None, b1_w, b1_b, b1_m, b1_v)
            add_tensor(f"{gguf_prefix}.conv1.weight", f1_w, flatten_conv=True)
            add_tensor(f"{gguf_prefix}.conv1.bias", f1_b)

            # conv2: 3×3 conv
            c2_w = require(f"{block_key}.conv2.weight")
            b2_w = require(f"{block_key}.bn2.weight")
            b2_b = require(f"{block_key}.bn2.bias")
            b2_m = require(f"{block_key}.bn2.running_mean")
            b2_v = require(f"{block_key}.bn2.running_var")
            f2_w, f2_b = fold_bn_into_conv(c2_w, None, b2_w, b2_b, b2_m, b2_v)
            add_tensor(f"{gguf_prefix}.conv2.weight", f2_w, flatten_conv=True)
            add_tensor(f"{gguf_prefix}.conv2.bias", f2_b)

            # Downsample (1×1 conv + BN, only on first block of stride-2 stages)
            ds_w = get(f"{block_key}.downsample.0.weight")
            if ds_w is not None:
                ds_bn_w = require(f"{block_key}.downsample.1.weight")
                ds_bn_b = require(f"{block_key}.downsample.1.bias")
                ds_bn_m = require(f"{block_key}.downsample.1.running_mean")
                ds_bn_v = require(f"{block_key}.downsample.1.running_var")
                ds_fw, ds_fb = fold_bn_into_conv(ds_w, None, ds_bn_w, ds_bn_b, ds_bn_m, ds_bn_v)
                add_tensor(f"{gguf_prefix}.downsample.weight", ds_fw, flatten_conv=True)
                add_tensor(f"{gguf_prefix}.downsample.bias", ds_fb)
                print(f"    block{block_idx}: conv1+bn, conv2+bn, downsample+bn")
            else:
                print(f"    block{block_idx}: conv1+bn, conv2+bn")

    # -----------------------------------------------------------------------
    # Neck: FPNC (FPN-Cat)
    #
    # MMOCR FPNC structure:
    #   lateral_convs[0..3]: 1×1 conv (in_ch[i] → 256)
    #   smooth_convs[0..3]:  3×3 conv (256 → 256), pad=1
    #   After top-down pathway + upsample + concat all 4 levels:
    #   output_conv (also called "out_conv"): 3×3 conv (256*4=1024 → 256)
    #
    # BatchNorm may or may not be present in neck depending on config.
    # MMOCR default FPNC has no BN in lateral/smooth convs, but output
    # conv may have BN. We handle both cases.
    # -----------------------------------------------------------------------
    print("\n--- Neck (FPNC) ---")

    for i in range(4):
        # Lateral conv: 1×1 (no bias, no BN in MMOCR FPNC default)
        lat_w = get(f"lateral_convs.{i}.conv.weight")
        if lat_w is None:
            lat_w = get(f"lateral_convs.{i}.weight")

        if lat_w is not None:
            lat_b = get(f"lateral_convs.{i}.conv.bias")
            add_tensor(f"det.neck.lateral{i}.weight", lat_w, flatten_conv=True)
            if lat_b is not None:
                add_tensor(f"det.neck.lateral{i}.bias", lat_b)
            print(f"  lateral{i}: {list(lat_w.shape)}")
        else:
            print(f"  WARNING: lateral_convs.{i} not found!")

    for i in range(4):
        # Smooth conv: 3×3 (FPNC: 256→64, no bias, no BN)
        sm_w = get(f"smooth_convs.{i}.conv.weight")
        if sm_w is None:
            sm_w = get(f"smooth_convs.{i}.weight")

        if sm_w is not None:
            sm_b = get(f"smooth_convs.{i}.conv.bias")
            add_tensor(f"det.neck.smooth{i}.weight", sm_w, flatten_conv=True)
            if sm_b is not None:
                add_tensor(f"det.neck.smooth{i}.bias", sm_b)
            print(f"  smooth{i}: {list(sm_w.shape)}")
        else:
            print(f"  WARNING: smooth_convs.{i} not found!")

    # FPNC note: no output_conv in this config. The smooth convs already
    # reduce to 64ch per level. Concat 4×64=256 feeds directly into head.
    # If an output_conv exists in other configs, handle it:
    out_w = get("output_convs.0.conv.weight")
    if out_w is not None:
        out_b = get("output_convs.0.conv.bias")
        add_tensor("det.neck.output.weight", out_w, flatten_conv=True)
        if out_b is not None:
            add_tensor("det.neck.output.bias", out_b)
        print(f"  output: {list(out_w.shape)}")
    else:
        print("  (no output_conv — FPNC concat 4×64=256 feeds head directly)")

    # -----------------------------------------------------------------------
    # Head: DBNet probability branch
    #
    # MMOCR DBHead binarize module (probability branch):
    #   conv1: 3×3 conv (256 → 64) + BN + ReLU
    #   deconv1: ConvTranspose2d (64 → 64, k=2, s=2) + BN + ReLU
    #   deconv2: ConvTranspose2d (64 → 1, k=2, s=2) → sigmoid
    #
    # We only export the probability branch. The threshold branch
    # (binarize.threshold) is identical in structure but only needed
    # for training.
    # -----------------------------------------------------------------------
    print("\n--- Head (probability branch) ---")

    # MMOCR DBHead binarize is a nn.Sequential with numeric indices:
    #   0: Conv2d(256→64, 3×3, pad=1)
    #   1: BatchNorm2d(64)
    #   2: ReLU
    #   3: ConvTranspose2d(64→64, k=2, s=2)
    #   4: BatchNorm2d(64)
    #   5: ReLU
    #   6: ConvTranspose2d(64→1, k=2, s=2)  (then sigmoid outside)

    def find_head(suffix):
        """Try MMOCR key patterns for det_head.binarize.<suffix>."""
        for prefix in ["det_head.binarize.", "binarize."]:
            k = prefix + suffix
            if k in sd:
                return sd[k]
        return get(f"binarize.{suffix}")

    # Conv1: index 0 (3×3, 256→64) + BN at index 1
    h_c1_w = find_head("0.weight")  # (64, 256, 3, 3)
    h_c1_b = find_head("0.bias")
    h_bn1_w = find_head("1.weight")
    h_bn1_b = find_head("1.bias")
    h_bn1_m = find_head("1.running_mean")
    h_bn1_v = find_head("1.running_var")

    if h_c1_w is not None and h_bn1_w is not None:
        fused_w, fused_b = fold_bn_into_conv(h_c1_w, h_c1_b, h_bn1_w, h_bn1_b, h_bn1_m, h_bn1_v)
        add_tensor("det.head.conv1.weight", fused_w, flatten_conv=True)
        add_tensor("det.head.conv1.bias", fused_b)
        print(f"  conv1: {list(h_c1_w.shape)} + BN → folded")
    elif h_c1_w is not None:
        add_tensor("det.head.conv1.weight", h_c1_w, flatten_conv=True)
        if h_c1_b is not None:
            add_tensor("det.head.conv1.bias", h_c1_b)
        print(f"  conv1: {list(h_c1_w.shape)} (no BN)")
    else:
        print("  WARNING: head conv1 not found!")

    # Deconv1: index 3 (ConvTranspose2d 64→64, k=2, s=2) + BN at index 4
    h_dc1_w = find_head("3.weight")   # (64, 64, 2, 2)
    h_dc1_b = find_head("3.bias")
    h_dc1_bn_w = find_head("4.weight")
    h_dc1_bn_b = find_head("4.bias")
    h_dc1_bn_m = find_head("4.running_mean")
    h_dc1_bn_v = find_head("4.running_var")

    if h_dc1_w is not None and h_dc1_bn_w is not None:
        fused_w, fused_b = fold_bn_into_deconv(h_dc1_w, h_dc1_b, h_dc1_bn_w, h_dc1_bn_b, h_dc1_bn_m, h_dc1_bn_v)
        add_tensor("det.head.deconv1.weight", fused_w, flatten_conv=True)
        add_tensor("det.head.deconv1.bias", fused_b)
        print(f"  deconv1: {list(h_dc1_w.shape)} + BN → folded")
    elif h_dc1_w is not None:
        add_tensor("det.head.deconv1.weight", h_dc1_w, flatten_conv=True)
        if h_dc1_b is not None:
            add_tensor("det.head.deconv1.bias", h_dc1_b)
        print(f"  deconv1: {list(h_dc1_w.shape)} (no BN)")
    else:
        print("  WARNING: head deconv1 not found!")

    # Deconv2: index 6 (ConvTranspose2d 64→1, k=2, s=2) — no BN
    h_dc2_w = find_head("6.weight")   # (64, 1, 2, 2)
    h_dc2_b = find_head("6.bias")     # (1,)

    if h_dc2_w is not None:
        add_tensor("det.head.deconv2.weight", h_dc2_w, flatten_conv=True)
        if h_dc2_b is not None:
            add_tensor("det.head.deconv2.bias", h_dc2_b)
        print(f"  deconv2: {list(h_dc2_w.shape)} (no BN, sigmoid output)")
    else:
        print("  WARNING: head deconv2 not found!")

    # -----------------------------------------------------------------------
    # Diagnostic: print any unconverted tensors
    # -----------------------------------------------------------------------
    converted_prefixes = {
        "conv1.", "bn1.", "layer1.", "layer2.", "layer3.", "layer4.",
        "lateral_convs.", "smooth_convs.", "output_convs.",
        "binarize.", "det_head.binarize.",
    }
    # Also check backbone.* prefixed keys
    backbone_prefixes = {"backbone." + p for p in converted_prefixes}
    all_prefixes = converted_prefixes | backbone_prefixes | {"neck.", "det_head."}

    skipped = []
    for k in sorted(sd):
        # Skip num_batches_tracked (BN bookkeeping, not needed)
        if "num_batches_tracked" in k:
            continue
        # Skip threshold branch (training only)
        if "threshold" in k:
            continue
        # Check if this key was handled
        handled = False
        for pfx in all_prefixes:
            if k.startswith(pfx):
                handled = True
                break
        if not handled:
            skipped.append(k)

    if skipped:
        print(f"\n--- Skipped tensors ({len(skipped)}) ---")
        for k in skipped:
            print(f"  {k}: {list(sd[k].shape)}")

    # -----------------------------------------------------------------------
    # Write
    # -----------------------------------------------------------------------
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    output_size = Path(args.output).stat().st_size / (1024 * 1024)
    print(f"\nWritten: {args.output}")
    print(f"  Tensors: {tensor_count}")
    print(f"  Parameters: {total_params:,}")
    print(f"  File size: {output_size:.1f} MB")
    print(f"  Format: {'FP16' if args.fp16 else 'FP32'}")
    print(f"  All BN layers folded into Conv/ConvTranspose")
    print(f"\nQuantize with: crispembed-quantize {args.output} dbnet-ic15-q4_k.gguf q4_k")


if __name__ == "__main__":
    main()
