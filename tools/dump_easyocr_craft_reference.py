#!/usr/bin/env python3
"""Dump a Python-blueprint EasyOCR CRAFT reference archive.

The archive captures the preprocessed input, CRAFT feature map, score map, and
the leaf module outputs needed to localize a native GGML graph divergence.
"""

import argparse
import sys
from pathlib import Path

import gguf
import numpy as np


def copy_state_dict(state):
    prefix = "module."
    return {k[len(prefix) :] if k.startswith(prefix) else k: v for k, v in state.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--easyocr-repo", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--canvas-size", type=int, default=2560)
    ap.add_argument("--mag-ratio", type=float, default=1.0)
    args = ap.parse_args()

    sys.path.insert(0, args.easyocr_repo)
    import cv2
    import torch
    from PIL import Image
    from easyocr.craft import CRAFT
    from easyocr.craft_utils import adjustResultCoordinates, getDetBoxes
    from easyocr.imgproc import normalizeMeanVariance

    image = np.asarray(Image.open(args.image).convert("RGB"))
    h, w, channels = image.shape
    target = min(args.canvas_size, args.mag_ratio * max(h, w))
    ratio = target / max(h, w)
    target_h, target_w = int(h * ratio), int(w * ratio)
    resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    canvas_h = target_h + ((32 - target_h % 32) % 32)
    canvas_w = target_w + ((32 - target_w % 32) % 32)
    canvas = np.zeros((canvas_h, canvas_w, channels), dtype=np.float32)
    canvas[:target_h, :target_w] = resized
    inp_np = np.transpose(normalizeMeanVariance(canvas), (2, 0, 1))[None]
    inp = torch.from_numpy(inp_np)

    net = CRAFT()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "state_dict" in state:
        state = state["state_dict"]
    net.load_state_dict(copy_state_dict(state))
    net.eval()

    captures = {"input_image": inp_np.astype(np.float32)}
    hooks = []

    def capture(name):
        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                output = output[0]
            if torch.is_tensor(output):
                captures[name] = output.detach().cpu().float().numpy()

        return hook

    hooks.append(net.basenet.register_forward_hook(capture("basenet")))
    hooks.append(net.upconv4.register_forward_hook(capture("upconv4")))
    hooks.append(net.conv_cls.register_forward_hook(capture("scores")))
    for name, module in net.named_modules():
        if name and len(list(module.children())) == 0:
            hooks.append(module.register_forward_hook(capture("leaf_" + name.replace(".", "_"))))

    with torch.no_grad():
        scores, feature = net(inp)
    captures["feature"] = feature.detach().cpu().float().numpy()
    captures["scores"] = scores.detach().cpu().float().numpy()
    for hook in hooks:
        hook.remove()

    text = scores[0, :, :, 0].numpy()
    link = scores[0, :, :, 1].numpy()
    boxes, polys, _ = getDetBoxes(text, link, 0.7, 0.4, 0.4, False, False)
    boxes = adjustResultCoordinates(boxes, 1.0 / ratio, 1.0 / ratio)
    writer = gguf.GGUFWriter(args.output, arch="easyocr-craft-reference")
    writer.add_string("general.source", "JaidedAI/EasyOCR / CRAFT")
    writer.add_string("general.license", "BSD-2-Clause")
    writer.add_uint32("easyocr.input_height", canvas_h)
    writer.add_uint32("easyocr.input_width", canvas_w)
    writer.add_uint32("easyocr.det_boxes", len(boxes))
    writer.add_string("easyocr.decoded", str(len(boxes)))
    for name, value in captures.items():
        writer.add_tensor(name, value.astype(np.float32), raw_dtype=gguf.GGMLQuantizationType.F32)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {args.output}; input={canvas_h}x{canvas_w}; boxes={len(boxes)}; stages={len(captures)}")


if __name__ == "__main__":
    main()
