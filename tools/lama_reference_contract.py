#!/usr/bin/env python3
"""Owned deterministic original-replay fixtures; no inference dependencies."""
import hashlib

SOURCE_COMMIT = "786f5936b27fb3dacd2b1ad799e4de968ea697e7"
CASES = {
    "subtitle_bar": (256, 256),
    "edge_touch": (257, 249),
    "center": (128, 160),
    "empty": (64, 64),
    "full": (128, 128),
    "one_pixel": (63, 65),
    "wide": (320, 96),
    "thin_strip": (256, 256),
}
# These fixed acceptance limits are declared before native checkpoint runs.
# They cannot be changed through environment variables or a reference bundle.
THRESHOLDS = {"stage_max_abs": 0.002, "stage_relative_l2": 0.0003,
              "stage_min_cosine": 0.99999, "prediction_max_abs": 0.0002,
              "masked_rgb_max_abs": 1, "masked_rgb_mean_abs": 0.1,
              "masked_rgb_min_psnr": 50.0, "known_rgb_errors": 0}


def fixture(name):
    width, height = CASES[name]
    rgb, mask = bytearray(width * height * 3), bytearray(width * height)
    for y in range(height):
        for x in range(width):
            p = y * width + x
            rgb[p * 3:p * 3 + 3] = bytes(((3 * x + y) % 256, (x + 2 * y) % 256,
                                         (x // 16 * 23 + y // 16 * 11) % 256))
            fill = {
                "subtitle_bar": width // 8 <= x < 7 * width // 8 and 13 * height // 16 <= y < 15 * height // 16,
                "edge_touch": x < width // 5 and height // 8 <= y < 7 * height // 8,
                "center": width // 3 <= x < 2 * width // 3 and height // 3 <= y < 2 * height // 3,
                "empty": False, "full": True,
                "one_pixel": x == width // 2 and y == height // 2,
                "wide": width // 3 <= x < 2 * width // 3 and height // 4 <= y < 3 * height // 4,
                "thin_strip": height - 6 <= y < height - 3,
            }[name]
            mask[p] = 255 if fill else 0
            if name == "subtitle_bar" and fill:
                glyph = x % 19 < 12 and y % 14 < 9
                rgb[p * 3:p * 3 + 3] = bytes((235, 235, 235) if glyph else (24, 24, 24))
    return bytes(rgb), bytes(mask)


def expected_stages():
    stages = {"input", "prediction", "generator.model.23", "generator.model.26", "generator.model.29", "generator.model.32"}
    for block in range(1, 5):
        stages.add(f"generator.model.{block}.local")
        if block == 4:
            stages.add(f"generator.model.{block}.global")
    for block in range(5, 23):
        base = f"generator.model.{block}"
        stages.update((base + ".local", base + ".global"))
        for conv in ("conv1", "conv2"):
            base_conv = base + "." + conv
            stages.update((base_conv + ".local", base_conv + ".global"))
            stages.update(base_conv + ".ffc.convg2g.fu." + suffix for suffix in ("rfft", "spectral", "irfft"))
    return stages


def identity(name):
    rgb, mask = fixture(name)
    width, height = CASES[name]
    return {"id": name, "width": width, "height": height,
            "rgb_sha256": hashlib.sha256(rgb).hexdigest(),
            "mask_sha256": hashlib.sha256(mask).hexdigest(), "fill_pixels": mask.count(255)}
