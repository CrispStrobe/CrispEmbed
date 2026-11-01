#!/usr/bin/env python3
"""Run the unchanged original Big-LaMa generator on live-eval photo inputs.

INTERNAL RESEARCH ONLY. Reuses dump_lama_reference.original (verified upstream
source, pinned dependencies, official archive) and the capture's exact
preprocessing: the original dataset's pad_img_to_modulo(8), masked RGB + mask,
clip/truncate to uint8, composite only filled pixels. Inpaints every
`<stem>.input.png` + `<stem>.mask.png` pair tools/inpaint_live_eval.py wrote; the
mask is binarized at >=128 exactly as the production CLI does. Writes
`<stem>.original.png` and report.json.
"""
import argparse
import json
from pathlib import Path
import time

from dump_lama_reference import original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-archive", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True, help="a live-eval output directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    import numpy as np
    from PIL import Image
    import torch

    torch.set_num_threads(args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    model, pad_original, versions = original(args.source_dir, args.checkpoint_archive)
    report = {"schema": "crispembed-lama-original-photos-v2", "versions": versions, "cases": []}
    for source in sorted(args.eval_dir.glob("*.input.png")):
        stem = source.name[:-len(".input.png")]
        rgb = np.asarray(Image.open(source).convert("RGB"))
        gray = np.asarray(Image.open(args.eval_dir / f"{stem}.mask.png").convert("L"))
        fill = gray >= 128
        height, width = fill.shape
        image = rgb.transpose(2, 0, 1).astype("float32") / 255
        mask = fill[None].astype("float32")
        image, mask = torch.from_numpy(pad_original(image, 8)), torch.from_numpy(pad_original(mask, 8))
        conditioned = torch.cat((image * (1 - mask), mask), dim=0).unsqueeze(0)
        started = time.monotonic()
        with torch.inference_mode():
            prediction = model(conditioned)
        seconds = round(time.monotonic() - started, 3)
        generated = np.clip(prediction[0, :, :height, :width].permute(1, 2, 0).numpy() * 255, 0, 255).astype(np.uint8)
        composed = np.where(fill[..., None], generated, rgb)
        Image.fromarray(composed).save(args.output_dir / f"{stem}.original.png")
        report["cases"].append({"stem": stem, "size": [width, height], "fill_pixels": int(fill.sum()),
                                "seconds": seconds})
        print(json.dumps(report["cases"][-1]), flush=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
