#!/usr/bin/env python3
"""Benchmark EasyOCR CRAFT inference after model/input setup."""

import argparse
import sys
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--easyocr-repo", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--repetitions", type=int, default=10)
    ap.add_argument("--canvas-size", type=int, default=2560)
    ap.add_argument("--mag-ratio", type=float, default=1.0)
    args = ap.parse_args()
    if args.repetitions <= 0:
        raise SystemExit("repetitions must be positive")

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
    inp = torch.from_numpy(np.transpose(normalizeMeanVariance(canvas), (2, 0, 1))[None])

    net = CRAFT()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "state_dict" in state:
        state = state["state_dict"]
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    net.load_state_dict(state)
    net.eval()

    def run():
        with torch.no_grad():
            scores, _ = net(inp)
        text = scores[0, :, :, 0].numpy()
        link = scores[0, :, :, 1].numpy()
        boxes, _, _ = getDetBoxes(text, link, 0.7, 0.4, 0.4, False, False)
        adjustResultCoordinates(boxes, 1.0 / ratio, 1.0 / ratio)
        return len(boxes)

    run()  # warm-up and thread-pool initialization
    elapsed = []
    boxes = -1
    for _ in range(args.repetitions):
        start = time.perf_counter()
        boxes = run()
        elapsed.append((time.perf_counter() - start) * 1000.0)
    print(
        f"easyocr-craft-reference-benchmark repetitions={args.repetitions} "
        f"boxes={boxes} graph_ms={sum(elapsed) / len(elapsed):.3f} "
        f"canvas={canvas_h}x{canvas_w}"
    )


if __name__ == "__main__":
    main()
