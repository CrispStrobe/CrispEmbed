#!/usr/bin/env python3
"""Dump per-layer reference activations for DBNet text detection.

Runs the MMOCR DBNet ResNet-18 model on a test image, captures intermediate
activations at every architectural boundary via forward hooks, and writes
to a GGUF archive for the C++ crispembed_diff harness.

Stages captured:
  input_image       (3, H, W)       F32 preprocessed input (after normalize)
  backbone_stage_K  (C_k, H_k, W_k) F32 after ResNet stage K (0..3)
  neck_lateral_K    (256, H_k, W_k) F32 after lateral conv K
  neck_smooth_K     (64, H_k, W_k)  F32 after smooth conv K
  neck_fused        (256, H/4, W/4) F32 after FPNC concat+upsample
  head_conv1        (64, H/4, W/4)  F32 after head 3×3 conv + BN + ReLU
  head_deconv1      (64, H/2, W/2)  F32 after first deconv + BN + ReLU
  prob_map          (1, H, W)       F32 probability map (pre-sigmoid logits)
  prob_map_sigmoid  (1, H, W)       F32 probability map (post-sigmoid)

Usage:
    pip install mmocr mmengine mmdet
    python tools/dump_dbnet_reference.py \
        --checkpoint /path/to/dbnet_resnet18_fpnc_1200e_icdar2015.pth \
        --image /path/to/test.jpg \
        --output /path/to/dbnet-ref.gguf

If no --image is given, generates a synthetic test image (640×640 with
text-like dark bars on light background).
"""

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np


def make_synthetic_image(h=640, w=640):
    """Create a synthetic test image with text-like features."""
    img = np.ones((h, w, 3), dtype=np.uint8) * 220  # light gray background
    # Dark horizontal bars (simulate text lines)
    for y_start in [100, 200, 300, 400, 500]:
        img[y_start:y_start+20, 80:560, :] = 30  # dark bars
    # Some vertical features
    img[80:540, 150:155, :] = 50
    img[80:540, 400:405, :] = 50
    return img


def preprocess_image(img, target_short_side=736, pad_divisor=32,
                     mean=(123.675, 116.28, 103.53),
                     std=(58.395, 57.12, 57.375)):
    """Preprocess image for DBNet: resize, normalize, pad to divisor."""
    h, w = img.shape[:2]

    # Resize keeping aspect ratio (short side = target)
    scale = target_short_side / min(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    # Simple bilinear resize
    from PIL import Image
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)
    img_resized = np.array(pil_img, dtype=np.float32)

    # Normalize: (pixel - mean) / std
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    img_norm = (img_resized - mean) / std

    # Pad to multiple of pad_divisor
    pad_h = (pad_divisor - new_h % pad_divisor) % pad_divisor
    pad_w = (pad_divisor - new_w % pad_divisor) % pad_divisor
    if pad_h > 0 or pad_w > 0:
        img_norm = np.pad(img_norm, ((0, pad_h), (0, pad_w), (0, 0)),
                          mode='constant', constant_values=0)

    # HWC → CHW, add batch dim
    img_chw = img_norm.transpose(2, 0, 1)  # (3, H, W)
    return img_chw, (new_h, new_w), scale


def dump_with_hooks(checkpoint_path, img_chw):
    """Run DBNet with forward hooks, capture all intermediates."""
    import torch

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    # We do a manual forward pass using the raw weights to avoid
    # requiring the full mmocr installation for inference.
    # This also gives us exact control over intermediates.

    captured = {}

    # Convert to torch tensors
    weights = {k: torch.tensor(v) if not isinstance(v, torch.Tensor) else v
               for k, v in sd.items()}

    def get_w(key):
        for prefix in ["backbone.", "neck.", "det_head.", ""]:
            full = prefix + key
            if full in weights:
                return weights[full]
        raise KeyError(f"Weight not found: {key}")

    def conv2d(x, w, b=None, stride=1, padding=0, groups=1):
        return torch.nn.functional.conv2d(x, w, b, stride=stride,
                                          padding=padding, groups=groups)

    def bn(x, w_key_prefix):
        """Apply batch norm using running stats."""
        gamma = get_w(f"{w_key_prefix}.weight")
        beta = get_w(f"{w_key_prefix}.bias")
        mean = get_w(f"{w_key_prefix}.running_mean")
        var = get_w(f"{w_key_prefix}.running_var")
        return torch.nn.functional.batch_norm(x, mean, var, gamma, beta,
                                              training=False)

    def conv_bn_relu(x, conv_key, bn_key, stride=1, padding=0):
        w = get_w(f"{conv_key}.weight")
        b = weights.get(f"backbone.{conv_key}.bias", None)
        x = conv2d(x, w, b, stride=stride, padding=padding)
        x = bn(x, bn_key)
        x = torch.nn.functional.relu(x, inplace=False)
        return x

    def basic_block(x, prefix, stride=1):
        """ResNet BasicBlock: conv1+bn1+relu → conv2+bn2 → (+shortcut) → relu"""
        identity = x

        out = conv2d(x, get_w(f"{prefix}.conv1.weight"), stride=stride, padding=1)
        out = bn(out, f"{prefix}.bn1")
        out = torch.relu(out)

        out = conv2d(out, get_w(f"{prefix}.conv2.weight"), stride=1, padding=1)
        out = bn(out, f"{prefix}.bn2")

        # Downsample shortcut if needed
        ds_key = f"{prefix}.downsample.0.weight"
        has_ds = any(ds_key in (p + ds_key) or (p + ds_key) in weights
                     for p in ["backbone.", ""])
        try:
            ds_w = get_w(f"{prefix}.downsample.0.weight")
            identity = conv2d(x, ds_w, stride=stride)
            identity = bn(identity, f"{prefix}.downsample.1")
        except KeyError:
            pass

        out = out + identity
        out = torch.relu(out)
        return out

    # --- Forward pass ---
    x = torch.tensor(img_chw).unsqueeze(0).float()  # (1, 3, H, W)
    captured["input_image"] = img_chw  # (3, H, W) numpy

    # Stem: conv1(7×7, s2, p3) + bn1 + relu + maxpool(3, s2, p1)
    x = conv2d(x, get_w("conv1.weight"), stride=2, padding=3)
    x = bn(x, "bn1")
    x = torch.relu(x)
    x = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)

    # Stage 0 (layer1): 2 blocks, stride 1, 64ch
    x = basic_block(x, "layer1.0", stride=1)
    x = basic_block(x, "layer1.1", stride=1)
    c2 = x  # stage 0 output
    captured["backbone_stage_0"] = c2[0].detach().numpy()
    print(f"  backbone stage 0: {list(c2.shape[1:])}")

    # Stage 1 (layer2): 2 blocks, stride 2, 128ch
    x = basic_block(x, "layer2.0", stride=2)
    x = basic_block(x, "layer2.1", stride=1)
    c3 = x
    captured["backbone_stage_1"] = c3[0].detach().numpy()
    print(f"  backbone stage 1: {list(c3.shape[1:])}")

    # Stage 2 (layer3): 2 blocks, stride 2, 256ch
    x = basic_block(x, "layer3.0", stride=2)
    x = basic_block(x, "layer3.1", stride=1)
    c4 = x
    captured["backbone_stage_2"] = c4[0].detach().numpy()
    print(f"  backbone stage 2: {list(c4.shape[1:])}")

    # Stage 3 (layer4): 2 blocks, stride 2, 512ch
    x = basic_block(x, "layer4.0", stride=2)
    x = basic_block(x, "layer4.1", stride=1)
    c5 = x
    captured["backbone_stage_3"] = c5[0].detach().numpy()
    print(f"  backbone stage 3: {list(c5.shape[1:])}")

    # --- Neck: FPNC ---
    features = [c2, c3, c4, c5]
    laterals = []
    for i, feat in enumerate(features):
        lat_w = get_w(f"lateral_convs.{i}.conv.weight")
        lat = conv2d(feat, lat_w)
        laterals.append(lat)
        captured[f"neck_lateral_{i}"] = lat[0].detach().numpy()

    # Top-down pathway (add upsampled higher level to lower level)
    for i in range(3, 0, -1):
        up = torch.nn.functional.interpolate(
            laterals[i], size=laterals[i-1].shape[2:],
            mode='bilinear', align_corners=False)
        laterals[i-1] = laterals[i-1] + up

    # Smooth convs
    smoothed = []
    for i in range(4):
        sm_w = get_w(f"smooth_convs.{i}.conv.weight")
        sm = conv2d(laterals[i], sm_w, padding=1)
        smoothed.append(sm)
        captured[f"neck_smooth_{i}"] = sm[0].detach().numpy()

    # Upsample all to same size (1/4 of input = stage 0 size) and concat
    target_size = smoothed[0].shape[2:]
    upsampled = []
    for i, sm in enumerate(smoothed):
        if sm.shape[2:] != target_size:
            sm = torch.nn.functional.interpolate(
                sm, size=target_size, mode='bilinear', align_corners=False)
        upsampled.append(sm)

    fused = torch.cat(upsampled, dim=1)  # (1, 4*64=256, H/4, W/4)
    captured["neck_fused"] = fused[0].detach().numpy()
    print(f"  neck fused: {list(fused.shape[1:])}")

    # --- Head: probability branch ---
    # conv1: 3×3 (256→64) + BN + ReLU
    h_w = get_w("binarize.0.weight")
    h_b = weights.get("det_head.binarize.0.bias", None)
    x = conv2d(fused, h_w, h_b, padding=1)
    x = bn(x, "binarize.1")
    x = torch.relu(x)
    captured["head_conv1"] = x[0].detach().numpy()
    print(f"  head conv1: {list(x.shape[1:])}")

    # deconv1: ConvTranspose2d(64→64, k=2, s=2) + BN + ReLU
    dc1_w = get_w("binarize.3.weight")
    dc1_b = get_w("binarize.3.bias")
    x = torch.nn.functional.conv_transpose2d(x, dc1_w, dc1_b, stride=2)
    x = bn(x, "binarize.4")
    x = torch.relu(x)
    captured["head_deconv1"] = x[0].detach().numpy()
    print(f"  head deconv1: {list(x.shape[1:])}")

    # deconv2: ConvTranspose2d(64→1, k=2, s=2) → sigmoid
    dc2_w = get_w("binarize.6.weight")
    dc2_b = get_w("binarize.6.bias")
    x = torch.nn.functional.conv_transpose2d(x, dc2_w, dc2_b, stride=2)
    captured["prob_map"] = x[0].detach().numpy()
    print(f"  prob_map (pre-sigmoid): {list(x.shape[1:])}")

    prob = torch.sigmoid(x)
    captured["prob_map_sigmoid"] = prob[0].detach().numpy()
    print(f"  prob_map_sigmoid: {list(prob.shape[1:])}, "
          f"range [{prob.min():.4f}, {prob.max():.4f}]")

    return captured


def write_gguf(path, tensors, metadata=None):
    """Write captured tensors to a GGUF archive."""
    import gguf

    writer = gguf.GGUFWriter(path, "dbnet-ref")

    if metadata:
        for k, v in metadata.items():
            if isinstance(v, str):
                writer.add_string(k, v)
            elif isinstance(v, float):
                writer.add_float32(k, v)

    for name, arr in tensors.items():
        if isinstance(arr, np.ndarray):
            writer.add_tensor(name, arr.astype(np.float32),
                              raw_dtype=gguf.GGMLQuantizationType.F32)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = Path(path).stat().st_size / 1024 / 1024
    print(f"\nWrote {path} ({size_mb:.1f} MB, {len(tensors)} tensors)")
    for name in sorted(tensors.keys()):
        arr = tensors[name]
        if isinstance(arr, np.ndarray):
            print(f"  {name}: {arr.shape} [{arr.min():.4f}, {arr.max():.4f}]")


def main():
    p = argparse.ArgumentParser(description="DBNet reference activation dumper")
    p.add_argument("--checkpoint", required=True,
                   help="MMOCR DBNet checkpoint .pth")
    p.add_argument("--image", default=None,
                   help="Test image (default: synthetic 640×640)")
    p.add_argument("--output", required=True, help="Output GGUF path")
    p.add_argument("--short-side", type=int, default=736,
                   help="Resize short side (default: 736)")
    args = p.parse_args()

    # Load or generate image
    if args.image:
        from PIL import Image
        img = np.array(Image.open(args.image).convert("RGB"))
        print(f"Image: {args.image} ({img.shape[1]}×{img.shape[0]})")
    else:
        img = make_synthetic_image(640, 640)
        print(f"Synthetic image: {img.shape[1]}×{img.shape[0]}")

    # Preprocess
    img_chw, (new_h, new_w), scale = preprocess_image(
        img, target_short_side=args.short_side)
    print(f"Preprocessed: {img_chw.shape} (resized to {new_w}×{new_h}, scale={scale:.3f})")

    # Run forward pass with hooks
    captured = dump_with_hooks(args.checkpoint, img_chw)

    # Write GGUF
    metadata = {
        "dbnet.ref.checkpoint": str(args.checkpoint),
        "dbnet.ref.image": args.image or "synthetic",
        "dbnet.ref.scale": scale,
    }
    write_gguf(args.output, captured, metadata)


if __name__ == "__main__":
    main()
