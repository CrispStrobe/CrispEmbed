#!/usr/bin/env python3
"""Which resize does MinerU2.5-Pro's pinned fast processor actually run?

The native preprocessing gate failed only for an upscaled input (200x102 ->
336x168). This compares, pixel-exactly on this x86 host, the processor's
pixel_values (un-normalized back to uint8 levels) with: torchvision uint8 AA
bicubic, uint8 non-AA bicubic, torchvision's float path, and CrispEmbed's
src/core/torch_bicubic_u8.h port. Up- and down-scaled synthetic inputs are
included. Exit 1 if the port does not match the processor on every input.

    gh workflow run heavy-cpu.yml --ref <branch> -f script=tools/ci-heavy/mineru25_resize_probe.py \
        -f pip="torch==2.9.0 torchvision==0.24.0 transformers==4.57.3 pillow==12.3.0 numpy==2.2.6"
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.transforms.v2 import functional as F

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("HEAVY_OUT", "probe-out"))
OUT.mkdir(parents=True, exist_ok=True)
DRIVER = r'''
#include "core/torch_bicubic_u8.h"
#include <cstdio>
#include <vector>
int main(int argc, char ** argv) {
    int h = atoi(argv[1]), w = atoi(argv[2]), oh = atoi(argv[3]), ow = atoi(argv[4]);
    std::vector<uint8_t> in(size_t(h) * w * 3), out;
    fread(in.data(), 1, in.size(), stdin);
    core_torch_resize::resize_rgb(in.data(), h, w, oh, ow, out);
    fwrite(out.data(), 1, out.size(), stdout);
}
'''


def port(rgb, oh, ow, exe):
    h, w, _ = rgb.shape
    done = subprocess.run([exe, str(h), str(w), str(oh), str(ow)], input=rgb.tobytes(), capture_output=True, check=True)
    return np.frombuffer(done.stdout, np.uint8).reshape(oh, ow, 3)


def torch_variants(rgb, oh, ow):
    chw = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
    out = {"tv_uint8_aa": F.resize(chw, [oh, ow], interpolation=F.InterpolationMode.BICUBIC, antialias=True),
           "tv_uint8_noaa": F.resize(chw, [oh, ow], interpolation=F.InterpolationMode.BICUBIC, antialias=False)}
    f = torch.nn.functional.interpolate(chw[None].float(), size=[oh, ow], mode="bicubic", align_corners=False,
                                        antialias=True)[0]
    out["float_aa_round"] = f.clamp(0, 255).round().to(torch.uint8)
    return {k: v.permute(1, 2, 0).numpy() for k, v in out.items()}


def processor_levels(proc, image, oh, ow):
    """Undo the fused normalize and the merge-block patch order back to HxWx3 levels."""
    pv = proc.image_processor(images=[image], return_tensors="pt")["pixel_values"].double().numpy()
    mean = np.array(proc.image_processor.image_mean) * 255
    std = np.array(proc.image_processor.image_std) * 255
    gh, gw = oh // 14, ow // 14
    rows = pv.reshape(gh // 2, gw // 2, 2, 2, 3, 2, 14, 14)  # by bx iy ix c t py px
    img = np.zeros((oh, ow, 3))
    for by in range(gh // 2):
        for bx in range(gw // 2):
            for iy in range(2):
                for ix in range(2):
                    block = rows[by, bx, iy, ix, :, 0]  # c py px (both temporal copies equal)
                    y, x = by * 28 + iy * 14, bx * 28 + ix * 14
                    img[y:y + 14, x:x + 14, :] = block.transpose(1, 2, 0) * std + mean
    return np.rint(img).astype(np.int32)


def main():
    exe = OUT / "resize_driver"
    (OUT / "driver.cpp").write_text(DRIVER)
    subprocess.run(["c++", "-std=c++17", "-O2", f"-I{ROOT / 'src'}", str(OUT / "driver.cpp"), "-o", str(exe)], check=True)
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained("opendatalab/MinerU2.5-Pro-2604-1.2B",
                                         revision="d3f5e08d073c21466bbabe21c71bb1e9c2e595da", use_fast=True)
    rng = np.random.default_rng(0)
    cases = [("table", np.array(Image.open(ROOT / "tests/regression/images/cc0/simple_table.jpg").convert("RGB"))),
             ("formula", np.array(Image.open(ROOT / "tests/regression/images/formula_quadratic.png").convert("RGB")))]
    for h, w in [(102, 200), (60, 300), (500, 900), (1200, 1700), (150, 150)]:
        cases.append((f"random_{h}x{w}", rng.integers(0, 256, (h, w, 3), dtype=np.uint8)))
    report, ok = {"torch": torch.__version__, "torchvision": torchvision.__version__,
                  "cpu_capability": torch.backends.cpu.get_cpu_capability(), "cases": {}}, True
    for name, rgb in cases:
        image = Image.fromarray(rgb)
        pv_shape = proc.image_processor(images=[image], return_tensors="pt")["image_grid_thw"][0].tolist()
        oh, ow = pv_shape[1] * 14, pv_shape[2] * 14
        target = processor_levels(proc, image, oh, ow)
        candidates = {"port": port(rgb, oh, ow, str(exe)), **torch_variants(rgb, oh, ow)}
        row = {"input": list(rgb.shape[:2]), "resized": [oh, ow]}
        for key, value in candidates.items():
            diff = np.abs(value.astype(np.int32) - target)
            row[key] = {"mismatched": int((diff > 0).sum()), "max": int(diff.max())}
        report["cases"][name] = row
        ok &= row["port"]["mismatched"] == 0
        print(name, json.dumps(row), flush=True)
    report["port_matches_processor_everywhere"] = bool(ok)
    (OUT / "resize-probe.json").write_text(json.dumps(report, indent=2))
    (OUT / "summary.md").write_text("```\n" + json.dumps(report, indent=2) + "\n```\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
