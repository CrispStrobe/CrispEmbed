#!/usr/bin/env python3
"""Benchmark the Python DBNet blueprint after model/input setup."""

import argparse
import contextlib
import io
import sys
import time

import numpy as np

from dump_dbnet_reference import dump_with_hooks, preprocess_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--repetitions", type=int, default=10)
    ap.add_argument("--short-side", type=int, default=736)
    args = ap.parse_args()
    if args.repetitions <= 0:
        raise SystemExit("repetitions must be positive")

    import torch
    from PIL import Image

    img = np.array(Image.open(args.image).convert("RGB"))
    img_chw, (new_h, new_w), _ = preprocess_image(img, target_short_side=args.short_side)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = state["state_dict"] if "state_dict" in state else state
    weights = {k: torch.tensor(v) if not isinstance(v, torch.Tensor) else v for k, v in sd.items()}

    def run():
        with contextlib.redirect_stdout(io.StringIO()):
            dump_with_hooks(args.checkpoint, img_chw, weights_override=weights)

    run()  # warm-up
    elapsed = []
    for _ in range(args.repetitions):
        start = time.perf_counter()
        run()
        elapsed.append((time.perf_counter() - start) * 1000.0)
    print(
        f"dbnet-reference-benchmark repetitions={args.repetitions} "
        f"graph_ms={sum(elapsed) / len(elapsed):.3f} input={new_h}x{new_w}"
    )


if __name__ == "__main__":
    main()
