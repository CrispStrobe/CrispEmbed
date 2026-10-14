#!/usr/bin/env python3
"""Pinned original processor/tokenizer versus native frontend, metadata only."""
import importlib.util
import json
import os
import subprocess
from pathlib import Path


def main():
    import gguf
    import numpy as np
    import torch
    import transformers
    import tokenizers
    from huggingface_hub import snapshot_download
    from PIL import Image, ImageDraw
    from transformers import AutoProcessor

    if transformers.__version__ != "5.5.4" or tokenizers.__version__ != "0.22.2":
        raise ValueError("Transformers 5.5.4 and Tokenizers 0.22.2 required")
    torch.set_num_threads(4)
    out, scratch = Path(os.environ["HEAVY_OUT"]), Path(os.environ["HEAVY_SCRATCH"])
    repo = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("exporter", repo / "models/convert-qwen35-ocr-to-gguf.py")
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    root = Path(snapshot_download(exporter.LIGHTON_MODEL, revision=exporter.LIGHTON_REVISION,
                                  allow_patterns=["*.json", "*.jinja"], token=os.environ.get("HF_TOKEN") or None))
    processor = AutoProcessor.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    from transformers.image_processing_backends import TorchvisionBackend
    if not isinstance(processor.image_processor, TorchvisionBackend):
        raise ValueError("Original reference did not resolve the expected torchvision backend")
    writer = gguf.GGUFWriter(str(out / "frontend-reference.gguf"), "qwen35_frontend_reference")
    writer.add_string("general.source.huggingface.repository", exporter.LIGHTON_MODEL)
    writer.add_string("general.source.huggingface.revision", exporter.LIGHTON_REVISION)
    exporter.write_tokenizer_metadata(writer, exporter.read_tokenizer_metadata(root, 248320))
    writer.add_string("tokenizer.chat_template", (root / "chat_template.jinja").read_text())
    def save(name, values):
        array = np.ascontiguousarray(values, dtype=np.float32)
        if not np.isfinite(array).all():
            raise ValueError("Nonfinite original frontend array")
        writer.add_tensor(name, array)
    def save_bytes(name, value):
        data = list(value.encode("utf-8"))
        save(name, data or [0])
        writer.add_uint32(name + ".length", len(data))
    rng = np.random.default_rng(3508)
    images = [
        ("unchanged", 256, 256, 65536, 16777216),
        ("upscale", 170, 220, 65536, 16777216),
        ("round-tie", 80, 1024, 65536, 16777216),
        ("extreme-aspect", 16, 2048, 65536, 16777216),
        ("rectangular", 301, 211, 65536, 16777216),
        ("downscale", 800, 900, 65536, 262144),
        ("horizontal-only", 256, 270, 65536, 16777216),
        ("vertical-only", 270, 256, 65536, 16777216),
        ("tiny-both", 1, 128, 65536, 16777216),
    ]
    records = []
    writer.add_uint32("test.image_count", len(images))
    original_resize = processor.image_processor.resize
    captured = []
    def resize(*args, **kwargs):
        value = original_resize(*args, **kwargs)
        captured.append(value.detach().cpu().numpy().copy())
        return value
    processor.image_processor.resize = resize
    try:
        for index, (name, height, width, minimum, maximum) in enumerate(images):
            pixels = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
            if index == 0:
                pil = Image.new("RGB", (width, height), "white")
                ImageDraw.Draw(pil).text((30, 80), "Hello OCR\n12345", fill="black", font_size=24)
                pixels = np.array(pil)
            pil = Image.fromarray(pixels)
            prompt = ["", "grounding", "Résumé e\u0301 ١٢३ 中文\n12345"][index % 3]
            messages = [{"role": "user", "content": [{"type": "image", "image": pil}]}]
            if prompt:
                messages[0]["content"].append({"type": "text", "text": prompt})
            rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            captured.clear()
            inputs = processor(text=[rendered], images=[pil], return_tensors="pt", return_mm_token_type_ids=True,
                               images_kwargs={"size": {"shortest_edge": minimum, "longest_edge": maximum}})
            if len(captured) != 1 or captured[0].dtype != np.uint8:
                raise ValueError("Expected one original uint8 resize capture")
            resized = captured[0][0].transpose(1, 2, 0)
            label = f"image.{index}"
            save(label + ".rgb", pixels)
            save(label + ".resized", resized)
            save(label + ".limits", [minimum, maximum])
            save(label + ".patches", inputs["pixel_values"].numpy())
            save(label + ".grid", inputs["image_grid_thw"].numpy())
            save(label + ".ids", inputs["input_ids"].numpy())
            save(label + ".modalities", inputs["mm_token_type_ids"].numpy())
            save_bytes(label + ".prompt", prompt)
            save_bytes(label + ".rendered", rendered)
            records.append({"name": name, "height": height, "width": width, "limits": [minimum, maximum],
                            "resized": list(resized.shape), "tokens": int(inputs["input_ids"].shape[1])})
    finally:
        processor.image_processor.resize = original_resize
    texts = ["Hello OCR  \n12345", "e\u0301lan a\u0308bc résumé", "किरण ١٢३ 中文", "\u0301\u0308abc\n\t  Z", "\r\n \tend  ",
             "I'm I'VE he'll we'd 0123456789", "sagte „Hallo“ heute", "<think>hello</think><|im_end|>",
             "<|vision_start|><|image_pad|><|vision_end|> text", "🙂️ 👨‍👩‍👦", "क््ष e\u0301\u0302\u0308"]
    writer.add_uint32("test.text_count", len(texts))
    for index, text in enumerate(texts):
        ids = processor.tokenizer.encode(text, add_special_tokens=False)
        decoded = processor.tokenizer.decode(ids, skip_special_tokens=True)
        save_bytes(f"text.{index}.input", text)
        save(f"text.{index}.ids", ids)
        save_bytes(f"text.{index}.decoded", decoded)
    # Original Rust NFC normalizer is the authority, including composition
    # exclusions, canonical ordering and algorithmic Hangul.
    import urllib.request
    with urllib.request.urlopen("https://www.unicode.org/Public/16.0.0/ucd/UnicodeData.txt", timeout=30) as stream:
        unicode_data = stream.read()
    import hashlib
    assert hashlib.sha256(unicode_data).hexdigest() == "ff58e5823bd095166564a006e47d111130813dcf8bf234ef79fa51a870edb48f"
    normalization_cases = ["", "e\u0301", "\u0308\u0301A", "a\u0301\u0323", "\u1100\u1161\u11a8", "\u0344"]
    for line in unicode_data.decode().splitlines():
        fields = line.split(";")
        cp = int(fields[0], 16)
        if fields[5] and not fields[5].startswith("<"):
            normalization_cases += [chr(cp), "".join(chr(int(x,16)) for x in fields[5].split())]
        if int(fields[3]):
            normalization_cases.append("a\u0301" + chr(cp) + "\u0323")
    writer.add_uint32("test.nfc_count", len(normalization_cases))
    for index, text in enumerate(normalization_cases):
        save_bytes(f"nfc.{index}.input", text)
        save_bytes(f"nfc.{index}.expected", processor.tokenizer.backend_tokenizer.normalizer.normalize_str(text))
    partial_ids = [processor.tokenizer.convert_tokens_to_ids("ð")]
    save("decode.partial.ids", partial_ids)
    save_bytes("decode.partial.expected", processor.tokenizer.decode(partial_ids, skip_special_tokens=True))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    build = scratch / "frontend-build"
    subprocess.run(["cmake", "-S", str(repo), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=OFF",
                    "-DGGML_BLAS=OFF", "-DGGML_METAL=OFF", "-DCRISPEMBED_BUILD_SHARED=OFF"], check=True)
    subprocess.run(["cmake", "--build", str(build), "--target", "test-qwen35-frontend", "test-qwen-pretokenize",
                    "test-bpe-pretokenize", "-j", "4"], check=True)
    statuses = {}
    for target, args in [("test-qwen35-frontend", [str(out / "frontend-reference.gguf")]),
                         ("test-qwen-pretokenize", []), ("test-bpe-pretokenize", [])]:
        result = subprocess.run([str(build / target), *args], capture_output=True, text=True)
        (out / (target + ".log")).write_text(result.stdout + result.stderr)
        print(result.stdout + result.stderr)
        statuses[target] = result.returncode == 0
    report = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
              "model": exporter.LIGHTON_MODEL, "revision": exporter.LIGHTON_REVISION,
              "processor": type(processor).__module__ + "." + type(processor).__name__,
              "image_processor": type(processor.image_processor).__module__ + "." + type(processor.image_processor).__name__,
              "backend": "torchvision CPU uint8", "images": records, "text_cases": len(texts), "normalization_cases": len(normalization_cases), "checks": statuses,
              "scope": "metadata-only original processor/tokenizer; no model inference or OCR-quality claim",
              "tokenizers": __import__("tokenizers").__version__,
              "effective_normalizer": json.loads(processor.tokenizer.backend_tokenizer.to_str())["normalizer"],
              "effective_pretokenizer": json.loads(processor.tokenizer.backend_tokenizer.to_str())["pre_tokenizer"]}
    (out / "frontend.json").write_text(json.dumps(report, indent=2) + "\n")
    if not all(statuses.values()):
        raise ValueError("Frontend or existing pretokenizer control failed")


if __name__ == "__main__":
    main()
