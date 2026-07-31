#!/usr/bin/env python3
"""Dump an EasyOCR recognizer reference archive for crispembed-diff.

This deliberately calls EasyOCR's own ``model(image, text)`` path rather than
reimplementing the network in NumPy.  The captured tensors are the contract a
ggml implementation must match.

Usage:
  python tools/dump_easyocr_reference.py --easyocr-repo /path/EasyOCR \
      --checkpoint english_g2.pth --charset english.txt --image crop.png \
      --output easyocr-english-g2-ref.gguf --generation 2
"""

import argparse
import sys
from pathlib import Path

import gguf
import numpy as np


def read_charset(path):
    text = Path(path).read_text(encoding="utf-8")
    lines = text.splitlines()
    return "".join(lines) if lines and all(len(x) <= 1 for x in lines) else text.rstrip("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--easyocr-repo", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--charset", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--generation", type=int, choices=(1, 2), required=True)
    ap.add_argument("--width", type=int, default=200)
    args = ap.parse_args()

    sys.path.insert(0, args.easyocr_repo)
    import torch
    from PIL import Image
    from easyocr.model import model as gen1_model
    from easyocr.model import vgg_model as gen2_model

    chars = read_charset(args.charset)
    output_channel = 512 if args.generation == 1 else 256
    hidden_size = 512 if args.generation == 1 else 256
    pkg = gen1_model if args.generation == 1 else gen2_model
    net = pkg.Model(input_channel=1, output_channel=output_channel,
                    hidden_size=hidden_size, num_class=len(chars) + 1)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "state_dict" in state:
        state = state["state_dict"]
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    net.load_state_dict(state)
    net.eval()

    image = Image.open(args.image).convert("L")
    img_h = 64
    resized_w = min(args.width, max(1, int(np.ceil(img_h * image.width / image.height))))
    image = image.resize((resized_w, img_h), Image.Resampling.BICUBIC)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    padded = np.zeros((1, img_h, args.width), dtype=np.float32)
    padded[:, :, :resized_w] = arr
    if resized_w < args.width:
        padded[:, :, resized_w:] = arr[None, :, resized_w - 1:resized_w]
    inp = torch.from_numpy(padded[None])
    text = torch.zeros((1, args.width // 10 + 1), dtype=torch.long)

    captures = {"input_image": inp.detach().numpy().astype(np.float32)}
    hooks = []

    def capture(name):
        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                output = output[0]
            if torch.is_tensor(output):
                captures[name] = output.detach().cpu().float().numpy()
        return hook

    hooks.append(net.FeatureExtraction.register_forward_hook(capture("features")))
    for name, module in net.FeatureExtraction.named_modules():
        if name and len(list(module.children())) == 0:
            hooks.append(module.register_forward_hook(capture("cnn_" + name.replace(".", "_"))))
    hooks.append(net.AdaptiveAvgPool.register_forward_hook(capture("sequence_input")))
    hooks.append(net.SequenceModeling[0].register_forward_hook(capture("bilstm_0")))
    hooks.append(net.SequenceModeling[1].register_forward_hook(capture("bilstm_1")))
    hooks.append(net.Prediction.register_forward_hook(capture("logits")))
    with torch.no_grad():
        logits = net(inp, text)
    for hook in hooks:
        hook.remove()

    # CTC greedy decode: blank=0, classes 1..N map to charset entries.
    best = logits.softmax(2).argmax(2)[0].tolist()
    decoded = []
    prev = 0
    for token in best:
        if token != 0 and token != prev:
            decoded.append(chars[token - 1])
        prev = token

    writer = gguf.GGUFWriter(args.output, arch="easyocr-reference")
    writer.add_string("general.source", "JaidedAI/EasyOCR")
    writer.add_string("general.license", "Apache-2.0")
    writer.add_uint32("easyocr.generation", args.generation)
    writer.add_uint32("easyocr.input_height", img_h)
    writer.add_uint32("easyocr.input_width", args.width)
    writer.add_array("tokenizer.tokens", ["<blank>"] + list(chars))
    writer.add_string("easyocr.decoded", "".join(decoded))
    for name, value in captures.items():
        writer.add_tensor(name, value.astype(np.float32), raw_dtype=gguf.GGMLQuantizationType.F32)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {args.output}; decoded={''.join(decoded)!r}; stages={len(captures)}")


if __name__ == "__main__":
    main()
