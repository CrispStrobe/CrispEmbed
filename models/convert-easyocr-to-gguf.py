#!/usr/bin/env python3
"""Convert an EasyOCR CRNN checkpoint to GGUF.

EasyOCR checkpoints are ordinary PyTorch state dictionaries.  This converter
keeps the module names (after removing the optional ``module.`` wrapper) and
stores the recognizer configuration/charset in GGUF metadata.  The first
supported family is the stock VGG-BiLSTM-CTC recognizer (``english_g2``,
``latin_g2``, etc.); the same file format is deliberately usable for the
ResNet Generation-1 recognizers.

Usage:
  python models/convert-easyocr-to-gguf.py --checkpoint english_g2.pth \
      --charset english.txt --output easyocr-english-g2-f16.gguf --fp16
"""

import argparse
import sys
from pathlib import Path

import gguf
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--charset", required=True,
                    help="UTF-8 file containing one output character per line, "
                         "or a single string file")
    ap.add_argument("--output", required=True)
    ap.add_argument("--network", choices=("vgg", "resnet"), default="vgg")
    ap.add_argument("--img-height", type=int, default=64)
    ap.add_argument("--img-width", type=int, default=200)
    ap.add_argument("--hidden-size", type=int, default=256)
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    import torch

    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if not isinstance(sd, dict):
        raise TypeError("checkpoint is not a state dictionary")

    # DataParallel checkpoints use module.<name>; EasyOCR's loader removes it.
    clean = {}
    for name, value in sd.items():
        if name.startswith("module."):
            name = name[7:]
        if not hasattr(value, "detach"):
            continue
        clean[name] = value.detach().float().cpu().numpy()

    # EasyOCR inference is always eval-mode. Fold BatchNorm layers into their
    # preceding convolutions so the runtime graph only needs conv → ReLU. This
    # keeps both VGG and ResNet tensors in native [out, in, kh, kw] layout.
    if args.network == "vgg":
        for conv_idx, bn_idx in ((11, 12), (14, 15)):
            cp = f"FeatureExtraction.ConvNet.{conv_idx}"
            bp = f"FeatureExtraction.ConvNet.{bn_idx}"
            if f"{cp}.weight" in clean and f"{bp}.weight" in clean:
                w = clean[f"{cp}.weight"]
                gamma = clean[f"{bp}.weight"]
                beta = clean[f"{bp}.bias"]
                mean = clean[f"{bp}.running_mean"]
                var = clean[f"{bp}.running_var"]
                scale = gamma / np.sqrt(var + 1.0e-5)
                clean[f"{cp}.weight"] = w * scale[:, None, None, None]
                clean[f"{cp}.bias"] = beta - mean * scale
                for suffix in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
                    clean.pop(f"{bp}.{suffix}", None)
    else:
        bn_prefixes = sorted({name.rsplit(".", 1)[0] for name in clean if name.endswith(".running_var")})
        for bp in bn_prefixes:
            if bp.endswith(".bn0_1") or bp.endswith(".bn0_2"):
                cp = bp.rsplit(".", 1)[0] + ".conv" + bp[-3:]
            elif bp.endswith(".bn1") or bp.endswith(".bn2"):
                cp = bp.rsplit(".", 1)[0] + ".conv" + bp[-1]
            elif bp.endswith(".downsample.1"):
                cp = bp.rsplit(".", 1)[0] + ".0"
            else:
                raise KeyError(f"unmapped ResNet BatchNorm: {bp}")
            wp = f"{cp}.weight"
            if wp not in clean:
                raise KeyError(f"missing convolution for {bp}: {wp}")
            gamma = clean[f"{bp}.weight"]
            beta = clean[f"{bp}.bias"]
            mean = clean[f"{bp}.running_mean"]
            var = clean[f"{bp}.running_var"]
            scale = gamma / np.sqrt(var + 1.0e-5)
            clean[wp] = clean[wp] * scale[:, None, None, None]
            clean[f"{cp}.bias"] = beta - mean * scale
            for suffix in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
                clean.pop(f"{bp}.{suffix}", None)

    chars = Path(args.charset).read_text(encoding="utf-8")
    if "\n" in chars:
        lines = chars.splitlines()
        # Accept either one character per line or a normal EasyOCR charset file.
        charset = "".join(lines) if all(len(x) <= 1 for x in lines) else chars.rstrip("\n")
    else:
        charset = chars

    # Prediction is CTC: class 0 is blank and characters start at class 1.
    prediction = clean.get("Prediction.weight")
    if prediction is None:
        raise KeyError("Prediction.weight")
    num_class = int(prediction.shape[0])
    expected = len(charset) + 1
    if num_class != expected:
        raise ValueError(f"charset has {len(charset)} chars but checkpoint has "
                         f"{num_class} classes (expected {expected})")

    feature_key = "FeatureExtraction.ConvNet.0.weight" if args.network == "vgg" else "FeatureExtraction.ConvNet.conv0_1.weight"
    feature = clean.get(feature_key)
    if feature is None:
        raise KeyError(f"{feature_key} (unsupported EasyOCR feature extractor)")
    output_key = "FeatureExtraction.ConvNet.11.weight" if args.network == "vgg" else "FeatureExtraction.ConvNet.conv4_2.weight"
    output_channel = int(clean[output_key].shape[0])
    sequence = clean["SequenceModeling.0.rnn.weight_ih_l0"]
    hidden_size = int(sequence.shape[0] // 4)
    input_channel = int(feature.shape[1])

    writer = gguf.GGUFWriter(args.output, arch="easyocr")
    writer.add_string("general.name", f"easyocr-{args.network}-bilstm-ctc")
    writer.add_string("general.license", "Apache-2.0")
    writer.add_string("general.source", "JaidedAI/EasyOCR")
    writer.add_uint32("easyocr.network", 0 if args.network == "vgg" else 1)
    writer.add_uint32("easyocr.input_channels", input_channel)
    writer.add_uint32("easyocr.input_height", args.img_height)
    writer.add_uint32("easyocr.input_width", args.img_width)
    writer.add_uint32("easyocr.output_channels", output_channel)
    writer.add_uint32("easyocr.hidden_size", hidden_size)
    writer.add_uint32("easyocr.num_classes", num_class)
    writer.add_array("tokenizer.tokens", ["<blank>"] + list(charset))

    for name in sorted(clean):
        data = clean[name]
        dtype = gguf.GGMLQuantizationType.F32
        if args.fp16 and data.ndim >= 2 and data.size >= 256:
            data = data.astype(np.float16)
            dtype = gguf.GGMLQuantizationType.F16
        writer.add_tensor(name, data, raw_dtype=dtype)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {args.output}: {len(clean)} tensors, {num_class} CTC classes")


if __name__ == "__main__":
    main()
