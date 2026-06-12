#!/usr/bin/env python3
"""Convert surya-ocr-2 text detector (EfficientViT segformer) to GGUF.

Loads the PyTorch state dict, folds BatchNorm into preceding Conv2d,
and packs everything into a single GGUF file for CrispEmbed.

Architecture:
  Encoder: EfficientViT-Large (Stem + 4 stages)
    - Stem: Conv3x3(s2) + 1x ConvBlock residual  →  [32, H/2, W/2]
    - Stage 0: 2x FusedMBConv                     →  [64, H/4, W/4]
    - Stage 1: 2x FusedMBConv                     →  [128, H/8, W/8]
    - Stage 2: 1x MBConv + 6x MBConv (residual)   →  [256, H/16, W/16]
    - Stage 3: 1x MBConv + 6x EfficientVitBlock   →  [512, H/32, W/32]
      (each VitBlock = LiteMLA context + MBConv local, with residuals)

  Decoder: SegFormer-style decode head
    - 4x Linear projections (per-stage)
    - Bilinear upsample all to stage_0 spatial size
    - Conv1x1 fuse → BN → ReLU → Conv1x1 classifier → 2 classes

  Input:  [3, 1200, 1200] (ImageNet-normalized RGB)
  Output: [2, 300, 300] sigmoid heatmap (text line, separator)

Usage:
    python models/convert-surya-det-to-gguf.py \\
        --output /mnt/storage/gguf-models/surya-det-f32.gguf

    python models/convert-surya-det-to-gguf.py \\
        --output /mnt/storage/gguf-models/surya-det-f16.gguf --fp16
"""

import argparse
import sys
from pathlib import Path
from collections import OrderedDict

import gguf
import numpy as np
import torch


# ---------------------------------------------------------------------------
# BatchNorm folding
# ---------------------------------------------------------------------------
def fold_bn_into_conv(conv_w, conv_b, bn_w, bn_b, bn_mean, bn_var, eps=1e-5):
    """Fold BatchNorm into preceding Conv2d weights."""
    # conv_w: [OC, IC, KH, KW]  or [OC, 1, KH, KW] for depthwise
    # bn_w, bn_b, bn_mean, bn_var: [OC]
    oc = conv_w.shape[0]
    inv_std = 1.0 / np.sqrt(bn_var + eps)
    scale = bn_w * inv_std

    # Reshape scale for broadcasting with conv weights
    shape = [oc] + [1] * (conv_w.ndim - 1)
    w_folded = conv_w * scale.reshape(shape)

    if conv_b is not None:
        b_folded = (conv_b - bn_mean) * scale + bn_b
    else:
        b_folded = -bn_mean * scale + bn_b

    return w_folded, b_folded


def load_model(checkpoint=None):
    """Load surya detection model."""
    from surya.detection.loader import DetectionModelLoader
    loader = DetectionModelLoader(checkpoint=checkpoint)
    model = loader.model(device='cpu', dtype=torch.float32)
    return model


def convert_state_dict(state_dict):
    """Convert PyTorch state dict to GGUF tensors with BN folding.

    Returns OrderedDict of {gguf_name: numpy_array}.
    """
    sd = {k: v.cpu().numpy() for k, v in state_dict.items()}
    tensors = OrderedDict()

    def get(name, default=None):
        return sd.get(name, default)

    def fold_conv_bn(prefix, gguf_prefix):
        """Fold a ConvNormAct: conv.weight + norm.{weight,bias,running_mean,running_var}."""
        conv_w = get(f"{prefix}.conv.weight")
        conv_b = get(f"{prefix}.conv.bias")
        bn_w = get(f"{prefix}.norm.weight")
        bn_b = get(f"{prefix}.norm.bias")
        bn_mean = get(f"{prefix}.norm.running_mean")
        bn_var = get(f"{prefix}.norm.running_var")

        if bn_w is not None and bn_mean is not None:
            w, b = fold_bn_into_conv(conv_w, conv_b, bn_w, bn_b, bn_mean, bn_var)
        else:
            w = conv_w
            b = conv_b if conv_b is not None else np.zeros(conv_w.shape[0], dtype=np.float32)

        tensors[f"{gguf_prefix}.weight"] = w
        tensors[f"{gguf_prefix}.bias"] = b

    # === Stem ===
    # stem.in_conv (ConvNormAct: Conv3x3 s2 + BN + ReLU)
    fold_conv_bn("vit.stem.in_conv", "stem.in_conv")

    # stem.res0 (ResidualBlock with ConvBlock: conv1 + conv2)
    fold_conv_bn("vit.stem.res0.main.conv1", "stem.res0.conv1")
    fold_conv_bn("vit.stem.res0.main.conv2", "stem.res0.conv2")

    # === Stage 0: 2x FusedMBConv ===
    for blk in range(2):
        prefix = f"vit.stages.0.blocks.{blk}.main"
        gguf_pfx = f"stage0.block{blk}"
        fold_conv_bn(f"{prefix}.spatial_conv", f"{gguf_pfx}.spatial")
        fold_conv_bn(f"{prefix}.point_conv", f"{gguf_pfx}.point")

    # === Stage 1: 2x FusedMBConv ===
    for blk in range(2):
        prefix = f"vit.stages.1.blocks.{blk}.main"
        gguf_pfx = f"stage1.block{blk}"
        fold_conv_bn(f"{prefix}.spatial_conv", f"{gguf_pfx}.spatial")
        fold_conv_bn(f"{prefix}.point_conv", f"{gguf_pfx}.point")

    # === Stage 2: block0 = MBConv (no residual), blocks 1-6 = MBConv (residual) ===
    for blk in range(7):
        prefix = f"vit.stages.2.blocks.{blk}.main"
        gguf_pfx = f"stage2.block{blk}"

        # MBConv: inverted_conv (1x1) + depth_conv (3x3 dw) + point_conv (1x1)
        # inverted_conv and depth_conv have bias (no BN for fewer_norm blocks)
        inv_w = get(f"{prefix}.inverted_conv.conv.weight")
        inv_b = get(f"{prefix}.inverted_conv.conv.bias")
        inv_bn_w = get(f"{prefix}.inverted_conv.norm.weight")

        if inv_bn_w is not None:
            fold_conv_bn(f"{prefix}.inverted_conv", f"{gguf_pfx}.inverted")
        else:
            tensors[f"{gguf_pfx}.inverted.weight"] = inv_w
            tensors[f"{gguf_pfx}.inverted.bias"] = inv_b if inv_b is not None else np.zeros(inv_w.shape[0], dtype=np.float32)

        dw_w = get(f"{prefix}.depth_conv.conv.weight")
        dw_b = get(f"{prefix}.depth_conv.conv.bias")
        dw_bn_w = get(f"{prefix}.depth_conv.norm.weight")

        if dw_bn_w is not None:
            fold_conv_bn(f"{prefix}.depth_conv", f"{gguf_pfx}.depth")
        else:
            tensors[f"{gguf_pfx}.depth.weight"] = dw_w
            tensors[f"{gguf_pfx}.depth.bias"] = dw_b if dw_b is not None else np.zeros(dw_w.shape[0], dtype=np.float32)

        fold_conv_bn(f"{prefix}.point_conv", f"{gguf_pfx}.point")

    # === Stage 3: block0 = MBConv, blocks 1-6 = EfficientVitBlock ===
    # Block 0: MBConv (same as stage 2 blocks)
    prefix = "vit.stages.3.blocks.0.main"
    gguf_pfx = "stage3.block0"

    inv_w = get(f"{prefix}.inverted_conv.conv.weight")
    inv_b = get(f"{prefix}.inverted_conv.conv.bias")
    inv_bn_w = get(f"{prefix}.inverted_conv.norm.weight")

    if inv_bn_w is not None:
        fold_conv_bn(f"{prefix}.inverted_conv", f"{gguf_pfx}.inverted")
    else:
        tensors[f"{gguf_pfx}.inverted.weight"] = inv_w
        tensors[f"{gguf_pfx}.inverted.bias"] = inv_b if inv_b is not None else np.zeros(inv_w.shape[0], dtype=np.float32)

    dw_w = get(f"{prefix}.depth_conv.conv.weight")
    dw_b = get(f"{prefix}.depth_conv.conv.bias")
    dw_bn_w = get(f"{prefix}.depth_conv.norm.weight")

    if dw_bn_w is not None:
        fold_conv_bn(f"{prefix}.depth_conv", f"{gguf_pfx}.depth")
    else:
        tensors[f"{gguf_pfx}.depth.weight"] = dw_w
        tensors[f"{gguf_pfx}.depth.bias"] = dw_b if dw_b is not None else np.zeros(dw_w.shape[0], dtype=np.float32)

    fold_conv_bn(f"{prefix}.point_conv", f"{gguf_pfx}.point")

    # Blocks 1-6: EfficientVitBlock (context_module + local_module)
    for blk in range(1, 7):
        prefix = f"vit.stages.3.blocks.{blk}"
        gguf_pfx = f"stage3.block{blk}"

        # Context module: LiteMLA
        # qkv: Conv1x1 (no BN, no activation in the code)
        qkv_w = get(f"{prefix}.context_module.main.qkv.conv.weight")
        tensors[f"{gguf_pfx}.ctx.qkv.weight"] = qkv_w
        # qkv bias is optional
        qkv_b = get(f"{prefix}.context_module.main.qkv.conv.bias")
        if qkv_b is not None:
            tensors[f"{gguf_pfx}.ctx.qkv.bias"] = qkv_b

        # aggreg: depthwise conv5x5 + grouped conv1x1
        agg_dw_w = get(f"{prefix}.context_module.main.aggreg.0.0.weight")
        tensors[f"{gguf_pfx}.ctx.agg_dw.weight"] = agg_dw_w
        agg_dw_b = get(f"{prefix}.context_module.main.aggreg.0.0.bias")
        if agg_dw_b is not None:
            tensors[f"{gguf_pfx}.ctx.agg_dw.bias"] = agg_dw_b

        agg_pw_w = get(f"{prefix}.context_module.main.aggreg.0.1.weight")
        tensors[f"{gguf_pfx}.ctx.agg_pw.weight"] = agg_pw_w
        agg_pw_b = get(f"{prefix}.context_module.main.aggreg.0.1.bias")
        if agg_pw_b is not None:
            tensors[f"{gguf_pfx}.ctx.agg_pw.bias"] = agg_pw_b

        # proj: Conv1x1 + BN
        fold_conv_bn(f"{prefix}.context_module.main.proj", f"{gguf_pfx}.ctx.proj")

        # Local module: MBConv (inverted + depth + point)
        local_prefix = f"{prefix}.local_module.main"
        local_gguf = f"{gguf_pfx}.local"

        inv_w = get(f"{local_prefix}.inverted_conv.conv.weight")
        inv_b = get(f"{local_prefix}.inverted_conv.conv.bias")
        inv_bn_w = get(f"{local_prefix}.inverted_conv.norm.weight")
        if inv_bn_w is not None:
            fold_conv_bn(f"{local_prefix}.inverted_conv", f"{local_gguf}.inverted")
        else:
            tensors[f"{local_gguf}.inverted.weight"] = inv_w
            tensors[f"{local_gguf}.inverted.bias"] = inv_b if inv_b is not None else np.zeros(inv_w.shape[0], dtype=np.float32)

        dw_w = get(f"{local_prefix}.depth_conv.conv.weight")
        dw_b = get(f"{local_prefix}.depth_conv.conv.bias")
        dw_bn_w = get(f"{local_prefix}.depth_conv.norm.weight")
        if dw_bn_w is not None:
            fold_conv_bn(f"{local_prefix}.depth_conv", f"{local_gguf}.depth")
        else:
            tensors[f"{local_gguf}.depth.weight"] = dw_w
            tensors[f"{local_gguf}.depth.bias"] = dw_b if dw_b is not None else np.zeros(dw_w.shape[0], dtype=np.float32)

        fold_conv_bn(f"{local_prefix}.point_conv", f"{local_gguf}.point")

    # === Decode head ===
    # 4 linear projections
    for i in range(4):
        w = get(f"decode_head.linear_c.{i}.proj.weight")
        b = get(f"decode_head.linear_c.{i}.proj.bias")
        tensors[f"dec.proj{i}.weight"] = w
        tensors[f"dec.proj{i}.bias"] = b

    # Fuse conv1x1 + BN (fold BN)
    fuse_w = get("decode_head.linear_fuse.weight")
    fuse_b = get("decode_head.linear_fuse.bias")  # no bias in Conv1x1
    bn_w = get("decode_head.batch_norm.weight")
    bn_b = get("decode_head.batch_norm.bias")
    bn_mean = get("decode_head.batch_norm.running_mean")
    bn_var = get("decode_head.batch_norm.running_var")
    w_f, b_f = fold_bn_into_conv(fuse_w, fuse_b, bn_w, bn_b, bn_mean, bn_var)
    tensors["dec.fuse.weight"] = w_f
    tensors["dec.fuse.bias"] = b_f

    # Classifier conv1x1
    tensors["dec.classifier.weight"] = get("decode_head.classifier.weight")
    tensors["dec.classifier.bias"] = get("decode_head.classifier.bias")

    return tensors


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=None,
        help="Local checkpoint or S3 path (default: surya built-in)")
    parser.add_argument("--output", required=True, help="Output GGUF path")
    parser.add_argument("--fp16", action="store_true",
        help="Store large tensors in F16")
    args = parser.parse_args()

    print("Loading model...")
    model = load_model(args.checkpoint)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded: {total_params:,} params")

    print("Converting weights (folding BatchNorm)...")
    tensors = convert_state_dict(model.state_dict())

    # Verify no weights were missed
    orig_names = set(model.state_dict().keys())
    # BN running stats aren't parameters but are in state_dict
    accounted = set()
    for k in orig_names:
        # Skip running_mean/running_var/num_batches_tracked — folded into conv
        if any(s in k for s in ['running_mean', 'running_var', 'num_batches_tracked']):
            accounted.add(k)
            continue
        # Skip BN weight/bias — folded
        if '.norm.weight' in k or '.norm.bias' in k:
            accounted.add(k)
            continue
        if 'batch_norm' in k:
            accounted.add(k)
            continue
        # Everything else should map to a GGUF tensor
        accounted.add(k)

    missed = orig_names - accounted
    if missed:
        print(f"WARNING: {len(missed)} state_dict keys not accounted for:")
        for m in sorted(missed):
            print(f"  {m}")

    # Write GGUF
    print(f"Writing {len(tensors)} tensors to {args.output}")
    writer = gguf.GGUFWriter(args.output, "surya-det")

    # Hyperparameters
    writer.add_uint32("surya_det.num_classes", 2)
    writer.add_uint32("surya_det.input_height", 1200)
    writer.add_uint32("surya_det.input_width", 1200)
    writer.add_uint32("surya_det.stem_channels", 32)
    writer.add_array("surya_det.stage_channels", [64, 128, 256, 512])
    writer.add_array("surya_det.stage_depths", [2, 2, 7, 7])
    writer.add_uint32("surya_det.vit_blocks_start", 1)  # stage3 block1+
    writer.add_uint32("surya_det.head_dim", 32)
    writer.add_uint32("surya_det.dec_hidden", 512)
    writer.add_uint32("surya_det.dec_layer_hidden", 128)
    writer.add_description("surya-ocr-2 text detector (EfficientViT segformer)")

    # ImageNet normalization constants
    writer.add_array("surya_det.image_mean", [0.485, 0.456, 0.406])
    writer.add_array("surya_det.image_std", [0.229, 0.224, 0.225])

    total_bytes = 0
    for name, data in tensors.items():
        data = data.astype(np.float32)
        if args.fp16 and data.size > 256:
            # Store large tensors in F16
            data = data.astype(np.float16)
        print(f"  {name}: {list(data.shape)} {data.dtype}")
        writer.add_tensor(name, data)
        total_bytes += data.nbytes

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    out_size = Path(args.output).stat().st_size / 1024 / 1024
    print(f"\nDone: {args.output} ({out_size:.1f} MB, {len(tensors)} tensors)")
    print(f"Tensor data: {total_bytes / 1024 / 1024:.1f} MB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
