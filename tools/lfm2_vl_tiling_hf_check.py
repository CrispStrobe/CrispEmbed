#!/usr/bin/env python3
"""Cross-check tools/lfm2_vl_tiling_oracle.py against the REAL HF processor.

The oracle is a verbatim extraction of transformers' tiling functions, which
makes it a good stand-in -- but it is still a transcription, and a transcription
can be wrong in exactly the way it is meant to catch. This script runs the
actual `Lfm2VlImageProcessorFast` on real images and asserts the oracle agrees
with what upstream's own code produced: the grid, the row/col info, the tile
count, the tile and thumbnail pixel shapes, and the per-image token counts.

Not part of the build. It needs torch + torchvision and a network fetch of the
model's `processor_config.json`, so it stays a developer tool: run it whenever
transformers is upgraded, then regenerate the golden header. The golden header
itself is hermetic and needs none of this.

Only the IMAGE processor is instantiated, not `Lfm2VlProcessor` -- the full
processor drags in a tokenizer class that needs transformers >= 5.0, and the
tokenizer has nothing to do with tiling. The token-markup arithmetic is taken
from `Lfm2VlProcessor._get_image_num_tokens`, which is a plain method over the
image processor's fields.

Usage:
    PYTHONPATH=<dir with torchvision> python tools/lfm2_vl_tiling_hf_check.py
"""

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def load_oracle():
    spec = importlib.util.spec_from_file_location("lfm2_oracle", REPO / "tools" / "lfm2_vl_tiling_oracle.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def hf_image_processor(model_id):
    """Instantiate Lfm2VlImageProcessorFast from the shipped processor_config."""
    from huggingface_hub import hf_hub_download
    from transformers.models.lfm2_vl.image_processing_lfm2_vl_fast import Lfm2VlImageProcessorFast

    cfg_path = hf_hub_download(model_id, "processor_config.json")
    cfg = json.load(open(cfg_path))["image_processor"]
    cfg.pop("image_processor_type", None)
    return Lfm2VlImageProcessorFast(**cfg), cfg


def token_counts(image_h, image_w, cfg):
    """Verbatim Lfm2VlProcessor._get_image_num_tokens."""
    P = cfg["encoder_patch_size"]
    ds = cfg["downsample_factor"]
    tile = cfg["tile_size"]

    thumbnail_tokens = 0
    if cfg["use_thumbnail"]:
        nph = image_h // P
        npw = image_w // P
        thumbnail_tokens = math.ceil(nph / ds) * math.ceil(npw / ds)

    npt = tile // P
    tile_tokens = math.ceil(npt / ds) ** 2
    return thumbnail_tokens, tile_tokens


def _scalar(v):
    """Unwrap however deeply reorder_images nested a single batch entry."""
    while isinstance(v, (list, tuple)):
        v = v[0]
    return v


def _pair(v):
    """Unwrap down to the innermost 2-element [height, width]."""
    while isinstance(v, (list, tuple)) and len(v) and isinstance(v[0], (list, tuple)):
        v = v[0]
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="LiquidAI/LFM2.5-VL-3B")
    args = ap.parse_args()

    import torch  # noqa: F401  (torchvision needs it loaded)
    from PIL import Image

    oracle = load_oracle()
    ip, cfg = hf_image_processor(args.model)

    print(f"transformers image processor: {type(ip).__name__}")
    print("config from the hub:", json.dumps({k: cfg[k] for k in sorted(cfg) if not isinstance(cfg[k], dict)}, indent=None))
    print()

    # Cover every case the golden header pins, plus a sweep across the sizes
    # where banker's rounding and std::round disagree.
    sizes = [(w, h) for (w, h, _) in oracle.CASES]
    sizes += [(w, h) for w in (144, 272, 400, 528, 656, 1040) for h in (650, 1000, 2000, 4000)]
    sizes += [(1024, 768), (768, 1024), (2480, 3508), (3508, 2480), (600, 600), (5000, 400)]
    sizes = sorted(set(sizes))

    hdr = f"{'W x H':>12} {'rows':>4} {'cols':>4} {'imgs':>4} {'tile px':>10} {'thumb px':>10} {'tokens':>7}  oracle"
    print(hdr)
    print("-" * len(hdr))

    n_fail = 0
    for w, h in sizes:
        # A deterministic non-uniform image; the content does not matter to the
        # layout, but a flat image would hide nothing either way.
        img = Image.new("RGB", (w, h))
        img.putdata([((x * 7) % 256, (x * 13) % 256, (x * 29) % 256) for x in range(w * h)])

        # No return_tensors: image_rows/cols/sizes stay plain Python lists.
        # reorder_images flattens differently depending on nesting, so unwrap
        # rather than assume a depth.
        out = ip(images=[[img]], return_row_col_info=True)
        rows = int(_scalar(out["image_rows"]))
        cols = int(_scalar(out["image_cols"]))
        img_h, img_w = (int(v) for v in _pair(out["image_sizes"]))
        spatial = out["spatial_shapes"]  # (n_images, 2) = (patches_h, patches_w)
        n_images = spatial.shape[0]

        thumb_tokens, tile_tokens = token_counts(int(img_h), int(img_w), cfg)
        split = rows > 1 or cols > 1
        total = (rows * cols * tile_tokens + thumb_tokens) if split else thumb_tokens

        ref = oracle.layout(w, h)

        # spatial_shapes carries each encoded image's patch grid. When split,
        # every tile is tile/P square and the last entry is the thumbnail.
        tile_px = f"{int(spatial[0][1]) * cfg['encoder_patch_size']}x{int(spatial[0][0]) * cfg['encoder_patch_size']}"
        thumb_px = (
            f"{int(spatial[-1][1]) * cfg['encoder_patch_size']}x{int(spatial[-1][0]) * cfg['encoder_patch_size']}"
            if split
            else "-"
        )

        problems = []

        def cmp(name, got, want):
            if got != want:
                problems.append(f"{name}: HF {got} != oracle {want}")

        cmp("split", bool(split), bool(ref["split"]))
        cmp("rows", int(rows), ref["rows"])
        cmp("cols", int(cols), ref["cols"])
        cmp("n_images", int(n_images), ref["n_images"])
        cmp("resized_w", int(img_w), ref["resized_w"])
        cmp("resized_h", int(img_h), ref["resized_h"])
        cmp("thumb_tokens", int(thumb_tokens), ref["thumb_tokens"])
        cmp("total_tokens", int(total), ref["total_tokens"])
        if split:
            cmp("tile_tokens", int(tile_tokens), ref["tile_tokens"])
            cmp("n_tiles", int(rows) * int(cols), ref["n_tiles"])
            # The tiles really are tile_size square, and the thumbnail really is
            # smart_resize's output -- not a rescale of the tiled image.
            cmp(
                "tile patch grid",
                (int(spatial[0][0]), int(spatial[0][1])),
                (cfg["tile_size"] // cfg["encoder_patch_size"], cfg["tile_size"] // cfg["encoder_patch_size"]),
            )
            cmp(
                "thumb patch grid",
                (int(spatial[-1][0]), int(spatial[-1][1])),
                (ref["resized_h"] // cfg["encoder_patch_size"], ref["resized_w"] // cfg["encoder_patch_size"]),
            )

        mark = "ok" if not problems else "MISMATCH"
        print(f"{w:5d} x{h:5d} {int(rows):4d} {int(cols):4d} {int(n_images):4d} {tile_px:>10} {thumb_px:>10} {int(total):7d}  {mark}")
        for p in problems:
            print(f"      {p}")
            n_fail += 1

    print()
    if n_fail:
        print(f"FAIL — {n_fail} disagreements between the HF processor and the oracle")
        return 1
    print(f"PASS — {len(sizes)} sizes, the oracle matches the real Lfm2VlImageProcessorFast exactly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
