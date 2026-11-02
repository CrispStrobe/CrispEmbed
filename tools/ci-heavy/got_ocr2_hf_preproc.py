#!/usr/bin/env python3
"""GOT-OCR2 HF-port (stepfun-ai/GOT-OCR-2.0-hf) preprocessing capture, no model.

Records the HF port's prompt IDs, generation config and pixel tensor (default
AutoProcessor and the slow/PIL image processor) so they can be compared with
the original remote-code pipeline (got_ocr2_orig_probe.py).

    python got_ocr2_hf_preproc.py <out_dir> <case=img> [...]
"""

import hashlib
import json
import os
import sys

import numpy as np
from PIL import Image
from transformers import AutoProcessor, GenerationConfig

REPO = "stepfun-ai/GOT-OCR-2.0-hf"


def main():
    out = sys.argv[1]
    cases = [a.split("=", 1) for a in sys.argv[2:]]
    rec = {"model": REPO, "cases": {}}
    procs = {"default": AutoProcessor.from_pretrained(REPO)}
    try:
        procs["slow"] = AutoProcessor.from_pretrained(REPO, use_fast=False)
    except Exception as e:  # noqa: BLE001
        rec["slow_error"] = repr(e)
    rec["image_processor_class"] = {k: type(p.image_processor).__name__ for k, p in procs.items()}
    try:
        rec["generation_config"] = GenerationConfig.from_pretrained(REPO).to_dict()
    except Exception as e:  # noqa: BLE001
        rec["generation_config_error"] = repr(e)
    for cid, path in cases:
        image = Image.open(path).convert("RGB")
        c = {}
        for k, p in procs.items():
            inp = p(image, return_tensors="pt")
            ids = inp["input_ids"][0].tolist()
            arr = inp["pixel_values"][0].numpy().astype(np.float32)
            np.save(os.path.join(out, f"hf.{k}.{cid}.pixels.npy"), arr)
            c[k] = {"prompt_ids_no_imgpad": [i for i in ids if i != 151859], "n_prompt": len(ids),
                    "prompt_text": p.tokenizer.decode([i for i in ids if i != 151859]),
                    "pixels_shape": list(arr.shape), "pixels_sha256": hashlib.sha256(arr.tobytes()).hexdigest()}
        rec["cases"][cid] = c
    with open(os.path.join(out, "hf_preproc.json"), "w") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps({k: v for k, v in rec.items() if k != "cases"}, indent=1)[:2000])


if __name__ == "__main__":
    main()
