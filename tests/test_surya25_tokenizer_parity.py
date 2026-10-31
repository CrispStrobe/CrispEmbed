#!/usr/bin/env python3
"""Surya OCR 2.5 tokenizer: native core_wordlevel vs original HF tokenizer.

Usage: test_surya25_tokenizer_parity.py <checkpoint_dir> <test-wordlevel-added binary> [--cases N]
       [--frontend <test-surya25-frontend binary>]

Builds the native tokenizer inputs from tokenizer.json exactly as the converter
does (models/qwen35_surya25_profile.py), encodes a deterministic corpus with
both implementations and requires identical ids for every case. It also checks
that HF decode(skip_special_tokens=True) equals plain concatenation of the
non-special token texts (the native Fuse decode contract). Prints a JSON summary.
"""

import argparse
import json
import random
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
import qwen35_surya25_profile as profile  # noqa: E402
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "convert_qwen35_ocr", Path(__file__).resolve().parents[1] / "models" / "convert-qwen35-ocr-to-gguf.py")
converter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(converter)


def scale_parity(binary, metadata, tmp):
    """Native surya_scale_to_fit vs the pinned package scale_to_fit (Pillow LANCZOS), byte for byte."""
    import numpy as np
    from PIL import Image
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "ci-heavy"))
    import surya25_metadata
    scale = surya25_metadata.surya_source()["scale_to_fit"]
    rng = np.random.default_rng(11)
    sizes = [(300, 40), (448, 112), (1792, 28), (97, 13), (1000, 30), (2550, 3300), (3300, 2550), (5000, 60),
             (60, 5000), (1, 1), (4000, 4000), (616, 84), (1201, 37)]
    results = []
    for width, height in sizes:
        rgb = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        expected = np.asarray(scale(Image.fromarray(rgb)).convert("RGB"))
        raw, out = Path(tmp, "scale-in.bin"), Path(tmp, "scale-out.bin")
        raw.write_bytes(rgb.tobytes())
        line = subprocess.run([binary, str(metadata), "--scale", str(raw), str(width), str(height), str(out)],
                              capture_output=True, text=True).stdout.split()
        entry = {"size": [width, height], "expected": [expected.shape[1], expected.shape[0]]}
        if len(line) == 3:
            h, w = int(line[0]), int(line[1])
            entry["native"] = [w, h]
            actual = np.fromfile(out, dtype=np.uint8)
            entry["bytes_exact"] = actual.size == expected.size and bool((actual == expected.reshape(-1)).all())
        results.append(entry)
    return results


def image_parity(processor, binary, metadata, tmp):
    """Native preprocess_rgb vs HF processor with the vLLM serving kwargs, on owned synthetic images."""
    import numpy as np
    from PIL import Image, ImageDraw
    rng = np.random.default_rng(31)
    # Sizes after the package's scale_to_fit (28-px grid); the last exceeds the native 1024-patch bound.
    sizes = [(448, 112), (1792, 28), (616, 84), (308, 196), (504, 504), (1624, 2100)]
    results = []
    for width, height in sizes:
        image = Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))
        draw = ImageDraw.Draw(image)
        for k in range(0, width, 37):
            draw.text((k, height // 3), "Ab", fill=(0, 0, 0))
        rgb = np.asarray(image, dtype=np.uint8)
        raw, out = Path(tmp, "image.bin"), Path(tmp, "patches.bin")
        raw.write_bytes(rgb.tobytes())
        native = subprocess.run([binary, str(metadata), "--image", str(raw), str(width), str(height), str(out)],
                                capture_output=True, text=True).stdout.split()
        hf = processor.image_processor(images=[image], return_tensors="pt", min_pixels=profile.IMAGE_MIN_PIXELS,
                                       max_pixels=profile.IMAGE_MAX_PIXELS)
        grid = hf["image_grid_thw"][0].tolist()
        reference = hf["pixel_values"].float().numpy().reshape(-1)
        entry = {"size": [width, height], "hf_grid": grid, "native": native}
        if native and native[0] != "REJECT":
            patches = np.fromfile(out, dtype=np.float32)
            entry["grid_equal"] = [int(x) for x in native[2:]] == grid
            entry["max_abs"] = float(np.abs(patches - reference).max()) if patches.size == reference.size else None
        results.append(entry)
    return results


def frontend_parity(root, binary, tmp):
    """Native format-4 frontend (real converter metadata) vs HF chat-template ids."""
    import gguf
    from transformers import AutoProcessor
    path = Path(tmp, "surya25-metadata-only.gguf")
    writer = gguf.GGUFWriter(str(path), converter.ARCH)
    writer.add_string("general.source.huggingface.repository", profile.MODEL)
    writer.add_string("general.source.huggingface.revision", profile.REVISION)
    writer.add_uint32(f"{converter.ARCH}.format_version", 4)
    converter.write_surya25_metadata(writer, root)
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
    processor = AutoProcessor.from_pretrained(root, local_files_only=True)
    prompts = list(profile.PROMPTS.values()) + ["", "Custom: Grüße <b>x</b>"]
    expected = []
    for prompt in prompts:
        text = prompt or profile.PROMPTS["block"]
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if rendered != profile.render_prompt(text):
            raise SystemExit("HF chat template rendering differs from the pinned prefix/suffix")
        expected.append(processor.tokenizer.encode(rendered, add_special_tokens=False))
    result = subprocess.run([binary, str(path)], input="\n".join(p or "-" for p in prompts) + "\n",
                            capture_output=True, text=True)
    lines = result.stdout.splitlines()
    actual = [[int(x) for x in line.split()[1:]] for line in lines if line.startswith("IDS")]
    images = image_parity(processor, binary, path, tmp)
    scales = scale_parity(binary, path, tmp)
    return {"images": images, "scale_to_fit": scales,
            "scale_to_fit_all_exact": all(x.get("bytes_exact") and x["native"] == x["expected"] for x in scales), "frontend_exit": result.returncode, "frontend_pass": lines[-1:] == ["PASS"],
            "frontend_prompts": len(prompts), "frontend_mismatches": sum(a != e for a, e in zip(actual, expected))
            + abs(len(actual) - len(expected)),
            "block_prompt_ids": expected[0]}

PACKAGE_PROMPTS = [
    "OCR this block image to HTML.",
    "OCR this image to HTML. Each block is a div with data-label and data-bbox (x0 y0 x1 y1, normalized 0-1000).",
    'Output the layout of this image as JSON. Each entry is a dict with "label", "bbox", and "count" fields. '
    'Bbox is x0 y0 x1 y1, normalized 0-1000.',
    'Output the table rows then columns as JSON. Each entry is a dict with "label" ("Row" or "Col") and "bbox" '
    '(x0 y0 x1 y1, normalized 0-1000).',
]
FIXED = [
    "", " ", "\n", "\r\n\t", "Grüße, 東京 — ½ x² <b>bold</b>\n\tTab", "<|im_end|>", ".<|im_end|>", "x<|im_end|>",
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>", "<think>\n\n</think>\n\n", "<unk>",
    '<div data-label="Text" data-bbox="12 34 567 890"><p>Hello, world.</p></div>',
    "<table><tr><td colspan=\"2\">1,234.56</td></tr></table>", "<math>\\frac{a}{b} = \\sqrt{x^2+1}</math>",
    "Straße über Äpfel — naïve café; Ελληνικά; русский; العربية; हिन्दी; 한국어; ไทย", "😀👍🏽 e\u0301 \ufeff \u200b",
    "a" * 300, "<" * 40, ".<" * 30, "10 11 100 1000 2026-10-31",
]


def write_vocab(path, data):
    with open(path, "wb") as f:
        f.write(b"WLA1" + struct.pack("<I", len(data["tokens"])))
        for token in data["tokens"]:
            raw = token.encode("utf-8")
            f.write(struct.pack("<I", len(raw)) + raw)
        for key in ["model_ids", "raw_added_ids", "normalized_added_ids", "special_ids"]:
            f.write(struct.pack("<I", len(data[key])) + struct.pack(f"<{len(data[key])}i", *data[key]))
        f.write(struct.pack("<Ii", 1, data["unk_id"]))


def corpus(data, count, seed=20261031):
    rng = random.Random(seed)
    tokens = data["tokens"]
    added = data["raw_added_ids"] + data["normalized_added_ids"]
    chars = [tokens[i] for i in data["model_ids"] if len(tokens[i]) == 1]
    cases = list(FIXED) + PACKAGE_PROMPTS + [p + "<|im_end|>" for p in PACKAGE_PROMPTS]
    for _ in range(count):
        parts = []
        for _ in range(rng.randint(1, 40)):
            kind = rng.random()
            if kind < 0.45:
                parts.append(tokens[rng.choice(added)])
            elif kind < 0.85:
                parts.append(rng.choice(chars))
            elif kind < 0.95:
                parts.append(rng.choice(" \n<>.\"=/"))
            else:  # code points absent from the vocabulary -> <unk>
                parts.append(chr(rng.choice([0x2FA1D, 0x1F9FF, 0xE000, 0x10FFFD, 0x0378 + 0])))
        cases.append("".join(parts))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("binary")
    parser.add_argument("--cases", type=int, default=20000)
    parser.add_argument("--frontend", help="test-surya25-frontend binary")
    args = parser.parse_args()
    from transformers import AutoTokenizer

    root = Path(args.checkpoint)
    data = profile.read_tokenizer(root)
    hf = AutoTokenizer.from_pretrained(root, local_files_only=True)
    if len(hf) != len(data["tokens"]):
        raise SystemExit("HF tokenizer length differs from exported vocabulary")
    cases = corpus(data, args.cases)
    with tempfile.TemporaryDirectory() as tmp:
        vocab, inputs, out = Path(tmp, "vocab.bin"), Path(tmp, "cases.bin"), Path(tmp, "out.txt")
        write_vocab(vocab, data)
        with open(inputs, "wb") as f:
            f.write(struct.pack("<I", len(cases)))
            for case in cases:
                raw = case.encode("utf-8")
                f.write(struct.pack("<I", len(raw)) + raw)
        subprocess.run([args.binary, str(vocab), str(inputs), str(out)], check=True)
        native = out.read_text().split("\n")[:-1]
        front = frontend_parity(root, args.frontend, tmp) if args.frontend else {}
    mismatches = []
    for case, line in zip(cases, native):
        expected = hf.encode(case, add_special_tokens=False)
        actual = None if line == "REJECT" else [int(x) for x in line.split()] if line else []
        if actual != expected:
            mismatches.append({"case": case[:200], "hf": expected[:80], "native": None if actual is None else actual[:80]})
    special = set(data["special_ids"])
    rng = random.Random(7)
    decode_mismatch = 0
    for _ in range(2000):
        ids = [rng.randrange(len(data["tokens"])) for _ in range(rng.randint(1, 60))]
        fused = "".join(data["tokens"][i] for i in ids if i not in special)
        if hf.decode(ids, skip_special_tokens=True) != fused:
            decode_mismatch += 1
    summary = {"cases": len(cases), "native_lines": len(native), "encode_mismatches": len(mismatches),
               "decode_mismatches": decode_mismatch, "first_mismatches": mismatches[:5],
               "tokenizer_json_sha256": data["tokenizer_json_sha256"], **front}
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    front_ok = not front or (front["frontend_exit"] == 0 and front["frontend_pass"] and not front["frontend_mismatches"]
                             and front["scale_to_fit_all_exact"])
    return 0 if not mismatches and not decode_mismatch and len(native) == len(cases) and front_ok else 1


if __name__ == "__main__":
    sys.exit(main())
