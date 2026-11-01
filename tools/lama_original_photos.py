#!/usr/bin/env python3
"""Run the unchanged original Big-LaMa generator on live-eval photo inputs.

INTERNAL RESEARCH ONLY. Reuses dump_lama_reference.load_original (hash-verified
upstream source, checkpoint and converted-model byte equality) and its exact
preprocess/decode, then inpaints every `<stem>.input.png` + `<stem>.mask.png`
pair that tools/inpaint_live_eval.py wrote. The mask is binarized at >=128,
exactly as the production CLI does. Writes `<stem>.original.png` and report.json.
"""

import argparse
import json
from pathlib import Path
import time

from dump_lama_reference import decode, load_original, preprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--converted-model", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True, help="a live-eval output directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    import numpy as np
    from PIL import Image
    import torch

    torch.set_num_threads(args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    model, _, adapter_calls, _, _ = load_original(args.source_dir, args.checkpoint, args.checkpoint_sha256,
                                                  args.config, args.converted_model)
    report = {"schema": "crispembed-lama-original-photos-v1", "torch": torch.__version__, "cases": []}
    for source in sorted(args.eval_dir.glob("*.input.png")):
        stem = source.name[:-len(".input.png")]
        image = Image.open(source).convert("RGB")
        gray = np.asarray(Image.open(args.eval_dir / f"{stem}.mask.png").convert("L"))
        mask = np.where(gray >= 128, 255, 0).astype(np.uint8)
        width, height = image.size
        rgb = image.tobytes()
        started = time.monotonic()
        with torch.inference_mode():
            raw = model(preprocess(rgb, mask.tobytes(), width, height))
        seconds = round(time.monotonic() - started, 3)
        _, composed = decode(raw, rgb, mask.tobytes(), width, height)
        Image.frombytes("RGB", (width, height), composed).save(args.output_dir / f"{stem}.original.png")
        report["cases"].append({"stem": stem, "size": [width, height], "fill_pixels": int((mask > 0).sum()),
                                "seconds": seconds})
        print(json.dumps(report["cases"][-1]), flush=True)
    if any(adapter_calls.values()):
        raise AssertionError("training-only import adapter was invoked")
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
