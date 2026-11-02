#!/usr/bin/env python3
"""GOT-OCR2 original-model probe (HF transformers port, CPU, F32).

Runs stepfun-ai/GOT-OCR-2.0-hf on one image with the plain-OCR prompt and
records, for the prefill pass:
  * max |input| / |output| of every nn.Linear and nn.Conv2d (the values a
    ggml F16-weight matmul would convert to F16 — anything above 65504 becomes
    inf there),
  * max |hidden| after every decoder layer and the vision/projector output,
and the greedy decoded token IDs and text (the frozen ground truth).

    python tools/ci-heavy/got_ocr2_hf_probe.py <image.png> <out.json> [max_new_tokens]
"""

import json
import sys
import time

import torch
from PIL import Image
from transformers import AutoProcessor, GotOcr2ForConditionalGeneration

REPO = "stepfun-ai/GOT-OCR-2.0-hf"
F16_MAX = 65504.0


def main():
    img_path, out_path = sys.argv[1], sys.argv[2]
    max_new = int(sys.argv[3]) if len(sys.argv) > 3 else 128
    torch.manual_seed(0)
    proc = AutoProcessor.from_pretrained(REPO)
    model = GotOcr2ForConditionalGeneration.from_pretrained(REPO, torch_dtype=torch.float32).eval()
    image = Image.open(img_path).convert("RGB")
    inputs = proc(image, return_tensors="pt")

    stats = {}
    phase = {"prefill": True}

    def hook(name):
        def fn(mod, inp, out):
            if not phase["prefill"]:
                return
            x = inp[0].detach()
            y = out.detach() if torch.is_tensor(out) else out[0].detach()
            s = stats.setdefault(name, {"type": type(mod).__name__, "in_max": 0.0, "out_max": 0.0})
            s["in_max"] = max(s["in_max"], float(x.abs().max()))
            s["out_max"] = max(s["out_max"], float(y.abs().max()))
            if s["in_max"] > F16_MAX:
                # where in the sequence the overflow sits (token index of the max)
                flat = x.abs().reshape(-1, x.shape[-1]).max(dim=-1).values
                s["in_overflow_rows"] = int((flat > F16_MAX).sum())
                s["in_argmax_row"] = int(flat.argmax())
        return fn

    for name, mod in model.named_modules():
        if isinstance(mod, (torch.nn.Linear, torch.nn.Conv2d)) or type(mod).__name__.endswith("DecoderLayer"):
            mod.register_forward_hook(hook(name))

    t0 = time.time()
    with torch.no_grad():
        # Prefill-only forward for the stats, then greedy generation.
        model(**inputs)
        phase["prefill"] = False
        gen = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=max_new,
                             tokenizer=proc.tokenizer, stop_strings="<|im_end|>")
    dt = time.time() - t0
    new_ids = gen[0, inputs["input_ids"].shape[1]:].tolist()
    text = proc.decode(new_ids, skip_special_tokens=True)
    over = {k: v for k, v in stats.items() if v["in_max"] > F16_MAX or v["out_max"] > F16_MAX}
    top = sorted(stats.items(), key=lambda kv: -max(kv[1]["in_max"], kv[1]["out_max"]))[:40]
    rec = {
        "model": REPO,
        "image": img_path,
        "prompt_ids": inputs["input_ids"][0].tolist(),
        "pixel_shape": list(inputs["pixel_values"].shape),
        "max_new_tokens": max_new,
        "token_ids": new_ids,
        "text": text,
        "seconds": round(dt, 2),
        "f16_max": F16_MAX,
        "modules_exceeding_f16": over,
        "top40_by_magnitude": dict(top),
        "all": stats,
    }
    with open(out_path, "w") as f:
        json.dump(rec, f, indent=1)
    print(f"text: {text!r}")
    print(f"{len(over)} modules exceed F16 range:")
    for k, v in over.items():
        print(f"  {k}: in_max={v['in_max']:.1f} out_max={v['out_max']:.1f}")
    for k, v in top[:15]:
        print(f"  top {k}: in_max={v['in_max']:.1f} out_max={v['out_max']:.1f}")


if __name__ == "__main__":
    main()
