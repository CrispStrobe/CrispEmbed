"""Hosted Surya OCR 2.5 first native A/B: original HF model vs native format-4 F32/F16.

1. Download the pinned snapshot (metadata pins verified before weights).
2. Convert F32 and F16 GGUFs with models/convert-qwen35-ocr-to-gguf.py --profile surya25
   (the converter checks the official meta-model schema against the checkpoint).
3. Render owned synthetic text crops. Sizes sit on the package's 28-px grid inside
   [1792x28 area, 1024 vision patches], so surya's LANCZOS scale_to_fit is a no-op and
   both sides consume identical pixels.
4. Original: transformers Qwen3_5ForConditionalGeneration, float32 on CPU, processor
   with the vLLM serving kwargs, greedy, block prompt. Record ids, text, stop.
5. Native: build/test-surya25-run per crop and arm; compare ids/text/stop exactly.

Prints SURYA25_NATIVE_SUMMARY with every case; writes $HEAVY_OUT/surya25-native.json.
Nothing here qualifies quality or a default: it is the first earliest-output A/B.
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "models"))
import qwen35_surya25_profile as profile  # noqa: E402

OUT = Path(os.environ.get("HEAVY_OUT", "surya25-native-out"))
SCRATCH = Path(os.environ.get("HEAVY_SCRATCH", "surya25-native-scratch"))
RUNNER = ROOT / "build" / "test-surya25-run"
CAP = int(os.environ.get("SURYA25_CAP", "256"))
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
# (name, width, height, lines). Width/height are multiples of 28; area in [50176, 262144].
CASES = [
    ("hello", 560, 112, ["Hello, world."]),
    ("invoice", 896, 84, ["Invoice No. 2026-10-31: 1,234.56 EUR"]),
    ("german", 728, 112, ["Grüße aus München, Straße 5"]),
    ("paragraph", 672, 252, ["The quick brown fox jumps", "over the lazy dog. 42%", "of tests pass (n=17)."]),
    ("bold", 504, 140, ["Total: $99.95"]),
    # 448x112: the vLLM min_pixels (3136) gives an 8x28 grid; processor defaults would give 8x32.
    ("short", 448, 112, ["OCR test 7"]),
]


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def render(name, width, height, lines):
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(FONT if name != "bold" else FONT.replace("Sans.ttf", "Sans-Bold.ttf"), 30)
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    y = (height - 38 * len(lines)) // 2
    for line in lines:
        draw.text((16, y), line, fill=(0, 0, 0), font=font)
        y += 38
    path = SCRATCH / f"{name}.png"
    image.save(path)
    return path, image


def original(root, cases):
    import torch
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    torch.manual_seed(0)
    processor = AutoProcessor.from_pretrained(root, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(root, local_files_only=True, dtype=torch.float32)
    model.eval()
    results = {}
    for name, image in cases:
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": profile.PROMPTS["block"]}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt",
                           min_pixels=profile.IMAGE_MIN_PIXELS, max_pixels=profile.IMAGE_MAX_PIXELS)
        start = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=CAP, do_sample=False, output_scores=True,
                                 return_dict_in_generate=True)
        ids = out.sequences[0, inputs["input_ids"].shape[1]:].tolist()
        conf = [float(torch.softmax(s[0].float(), -1).max()) for s in out.scores]
        results[name] = {"prompt_ids": inputs["input_ids"][0].tolist(), "grid": inputs["image_grid_thw"][0].tolist(),
                         "token_ids": ids, "text": processor.tokenizer.decode(ids, skip_special_tokens=True),
                         "stop_eos": bool(ids and ids[-1] == profile.EOS_ID), "confidences": conf,
                         "seconds": time.time() - start}
        print("ORIGINAL", name, json.dumps(results[name]["text"], ensure_ascii=False), len(ids), flush=True)
    return results


def native(model, path):
    env = dict(os.environ, CRISPEMBED_QWEN35_OCR="1", CRISPEMBED_SURYA25_FRONTEND="1",
               CRISPEMBED_SURYA25_INFERENCE="1")
    run = subprocess.run([str(RUNNER), str(model), str(path), str(CAP), "4"], capture_output=True, text=True, env=env)
    line = next((x for x in run.stdout.splitlines() if x.startswith("SURYA25_RESULT ")), None)
    if run.returncode or not line:
        return {"error": run.returncode, "stderr": run.stderr[-500:]}
    data = json.loads(line.split(" ", 1)[1])
    data["text"] = bytes.fromhex(data.pop("raw_text_utf8_hex")).decode("utf-8", "replace")
    return data


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download
    meta = snapshot_download(profile.MODEL, revision=profile.REVISION, allow_patterns=list(profile.METADATA_FILES))
    profile.verify_metadata(meta)  # before any weight download
    root = Path(snapshot_download(profile.MODEL, revision=profile.REVISION,
                                  allow_patterns=list(profile.METADATA_FILES) + ["model.safetensors"]))
    report = {"model": profile.MODEL, "revision": profile.REVISION, "cap": CAP, "arms": {}, "cases": {}}
    converter = ROOT / "models" / "convert-qwen35-ocr-to-gguf.py"
    sys.path.insert(0, str(converter.parent))
    import importlib.util
    spec = importlib.util.spec_from_file_location("convert_qwen35", converter)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    models = {}
    for dtype in ("f32", "f16"):
        target = SCRATCH / f"surya25-{dtype}.gguf"
        manifest = module.convert(root, target, profile.MODEL, profile.REVISION, dtype, (), profile="surya25")
        models[dtype] = target
        report["arms"][dtype] = {"bytes": manifest["bytes"], "export_tensor_count": manifest["export_tensor_count"],
                                 "checkpoint_tensor_count": manifest["checkpoint_tensor_count"],
                                 "gguf_sha256": sha(target)}
        print("CONVERTED", dtype, json.dumps(report["arms"][dtype]), flush=True)
    rendered = [(name, *render(name, w, h, lines)) for name, w, h, lines in CASES]
    reference = original(root, [(name, image) for name, _, image in rendered])
    for name, path, _ in rendered:
        case = {"original": reference[name], "input_sha256": sha(path)}
        for dtype, model in models.items():
            result = native(model, path)
            if "token_ids" in result:
                ids, ref = result["token_ids"], reference[name]["token_ids"]
                first = next((i for i, (a, b) in enumerate(zip(ids, ref)) if a != b),
                             None if len(ids) == len(ref) else min(len(ids), len(ref)))
                result["ids_exact"] = ids == ref
                result["text_exact"] = result["text"] == reference[name]["text"]
                result["first_divergent_step"] = first
                if ref and len(result["confidences"]) == len(reference[name]["confidences"]):
                    result["max_confidence_delta"] = max(
                        abs(a - b) for a, b in zip(result["confidences"], reference[name]["confidences"]))
            case[dtype] = result
            print("NATIVE", dtype, name, json.dumps({k: result.get(k) for k in
                  ("text", "ids_exact", "text_exact", "first_divergent_step", "stop_kind", "wall_seconds",
                   "max_confidence_delta", "error", "stderr")}, ensure_ascii=False), flush=True)
        report["cases"][name] = case
    (OUT / "surya25-native.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    summary = {name: {"original": c["original"]["text"],
                      **{d: {k: c[d].get(k) for k in ("ids_exact", "text_exact", "first_divergent_step", "error")}
                         for d in models}} for name, c in report["cases"].items()}
    print("SURYA25_NATIVE_SUMMARY", json.dumps({"arms": report["arms"], "cases": summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
