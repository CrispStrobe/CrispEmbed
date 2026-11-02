#!/usr/bin/env python3
"""Hosted smoke for the real, hash-pinned model and production CLI.

Authored deterministic RGB/mask fixtures check dimensions, known-pixel identity,
empty masks, boundary padding, multiple tiles, repeatability and model reuse.
They do not measure realistic subtitle quality or temporal consistency.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from test_inpaint_cli import decode_png, png_bytes


def fixture(width, height, kind):
    rgb = bytearray()
    mask = bytearray(width * height)
    for y in range(height):
        for x in range(width):
            rgb.extend(((3 * x + y) % 256, (x + 2 * y) % 256, (x // 4 + y // 3) % 256))
            if kind == "subtitle":
                fill = 48 <= x < width - 48 and height - 48 <= y < height - 32
            elif kind == "edge":
                fill = x >= width - 12 and y >= height - 15
            elif kind == "tiles":
                fill = (180 <= x < 275 and 180 <= y < 275) or (y >= height - 5 and x % 11 < 3)
            else:
                fill = False
            mask[y * width + x] = 255 if fill else 0
    return bytes(rgb), bytes(mask)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    binary, model = args.binary.resolve(), args.model.resolve()
    environment = os.environ.copy()
    environment["CRISPEMBED_MIGAN"] = "1"
    environment["CRISPEMBED_IMAGE_FORMAT"] = "png"
    environment.pop("CRISPEMBED_C2PA_KEY", None)
    environment.pop("CRISPEMBED_C2PA_CERT", None)
    cases = [("empty", 256, 256), ("subtitle", 256, 256), ("edge", 173, 91), ("tiles", 449, 385)]
    inputs = {}
    rows = []
    for name, width, height in cases:
        rgb, mask = fixture(width, height, name)
        (root / f"{name}.png").write_bytes(png_bytes(width, height, rgb))
        (root / f"{name}-mask.png").write_bytes(png_bytes(width, height, mask, 1))
        rows.append(f"{name}.png\t{name}-mask.png\t{name}-result.png")
        inputs[name] = (width, height, rgb, mask)
    rows.append("subtitle.png\tsubtitle-mask.png\tsubtitle-repeat.png")
    manifest = root / "frames.tsv"
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
    command = [str(binary), "--inpaint-model", str(model), "--inpaint-list", str(manifest), "--threads", "2"]
    started = time.monotonic()
    completed = subprocess.run(command, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               timeout=1800, check=False)
    elapsed = time.monotonic() - started
    (root / "batch-stderr.txt").write_bytes(completed.stderr)
    if completed.returncode != 0 or completed.stdout:
        raise RuntimeError(f"production batch failed ({completed.returncode}): {completed.stderr.decode(errors='replace')}")
    records = []
    for name, (width, height, rgb, mask) in inputs.items():
        data = (root / f"{name}-result.png").read_bytes()
        out_w, out_h, pixels, metadata = decode_png(data)
        if (out_w, out_h) != (width, height) or b"engine=migan-inpaint" not in metadata:
            raise AssertionError(f"{name}: dimensions or provenance mismatch")
        for index, fill in enumerate(mask):
            if not fill and pixels[3 * index:3 * index + 3] != rgb[3 * index:3 * index + 3]:
                raise AssertionError(f"{name}: known pixel {index} changed")
        if name == "empty" and pixels != rgb:
            raise AssertionError("empty mask is not byte-identical")
        records.append({"case": name, "width": width, "height": height,
                        "fill_pixels": sum(v != 0 for v in mask), "known_rgb_exact": True,
                        "png_sha256": hashlib.sha256(data).hexdigest()})
    first = decode_png((root / "subtitle-result.png").read_bytes())[2]
    repeated = decode_png((root / "subtitle-repeat.png").read_bytes())[2]
    if first != repeated:
        raise AssertionError("same-context repeat differs")
    # A fresh context must agree with the frame processed in the persistent batch.
    single = subprocess.run([str(binary), "--inpaint-model", str(model), "--inpaint", str(root / "subtitle.png"),
                             "--mask", str(root / "subtitle-mask.png"), "--threads", "2"],
                            env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            timeout=900, check=False)
    (root / "single-stderr.txt").write_bytes(single.stderr)
    if single.returncode or decode_png(single.stdout)[2] != first:
        raise AssertionError("single-image result differs from persistent batch")
    report = {"schema": "crispembed-inpaint-cli-smoke-v1", "passed": True, "cases": records,
              "same_context_repeat_exact": True, "single_and_batch_exact": True,
              "batch_wall_seconds_diagnostic": elapsed,
              "real_scene_quality_qualified": False, "temporal_quality_qualified": False}
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
