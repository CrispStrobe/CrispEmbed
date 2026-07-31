#!/usr/bin/env python3
"""Paddle-free PP-LCNet orientation reference for native parity checks."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def load_converter():
    path = Path(__file__).parents[1] / "models" / "convert-pplcnet-orientation-to-gguf.py"
    spec = importlib.util.spec_from_file_location("pplcnet_converter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preprocess(path: Path | None):
    if path is None:
        image = np.full((80, 160, 3), 255, dtype=np.uint8)
    else:
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    h, w = image.shape[:2]
    yy = np.minimum(h - 1, np.maximum(0, np.floor((np.arange(80) + 0.5) * h / 80))).astype(int)
    xx = np.minimum(w - 1, np.maximum(0, np.floor((np.arange(160) + 0.5) * w / 160))).astype(int)
    image = image[yy[:, None], xx[None, :]].astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return torch.from_numpy(((image - mean) / std).transpose(2, 0, 1)[None])


def conv(x, w, b, stride=1, groups=1):
    return F.conv2d(x, w, b, stride=stride, padding=w.shape[-1] // 2, groups=groups)


def bn(x, p, state):
    def find(suffix):
        return state[next(n for n in state if n.startswith(p + suffix))]
    mean, var, scale, bias = (find(k) for k in (".w_1", ".w_2", ".w_0", ".b_0"))
    shape = (1, -1, 1, 1)
    return (x - mean.reshape(shape)) * (scale / torch.sqrt(var + 1e-5)).reshape(shape) + bias.reshape(shape)


def cbn(x, conv_id, bn_id, in_ch, out_ch, k, stride, groups, state):
    p = f"conv2d_{conv_id}"
    w = state[next(n for n in state if n.startswith(p + ".w_0"))].reshape(out_ch, in_ch // groups, k, k)
    bname = next((n for n in state if n.startswith(p + ".b_0")), None)
    b = state[bname] if bname else None
    x = conv(x, w, b, stride, groups)
    x = bn(x, f"batch_norm2d_{bn_id}", state)
    return F.hardswish(x)


def run(model_dir: Path, image: Path | None):
    converter = load_converter()
    names = converter.graph_parameters(__import__("json").loads((model_dir / "inference.json").read_text())["program"])
    records = converter.read_records(model_dir / "inference.pdiparams", names)
    state = {name: torch.from_numpy(values.reshape(shape).astype(np.float32)) for name, shape, values in records}
    x = preprocess(image)
    x = cbn(x, 0, 0, 3, 16, 3, 2, 1, state)
    specs = [
        (1, 2, 1, 16, 32, 3, 1, False, 0, 0), (3, 4, 2, 32, 64, 3, 2, False, 0, 0),
        (5, 6, 3, 64, 64, 3, 1, False, 0, 0), (7, 8, 4, 64, 128, 3, 2, False, 0, 0),
        (9, 10, 5, 128, 128, 3, 1, False, 0, 0), (11, 12, 6, 128, 256, 3, 2, False, 0, 0),
        (13, 14, 13, 256, 256, 5, 1, False, 0, 0), (15, 16, 15, 256, 256, 5, 1, False, 0, 0),
        (17, 18, 17, 256, 256, 5, 1, False, 0, 0), (19, 20, 19, 256, 256, 5, 1, False, 0, 0),
        (21, 22, 21, 256, 256, 5, 1, False, 0, 0), (23, 26, 23, 256, 512, 5, 2, True, 24, 25),
        (27, 30, 25, 512, 512, 5, 1, True, 28, 29),
    ]
    for dw, pw, bn_dw, inc, outc, k, stride, use_se, se1, se2 in specs:
        residual = x
        x = cbn(x, dw, 25 if dw == 27 else dw, inc, inc, k, stride, inc, state)
        if use_se:
            gate = x.mean(dim=(2, 3), keepdim=True)
            s1 = state[next(n for n in state if n.startswith(f"conv2d_{se1}.w_0"))].reshape(inc // 4, inc, 1, 1)
            s2 = state[next(n for n in state if n.startswith(f"conv2d_{se2}.w_0"))].reshape(inc, inc // 4, 1, 1)
            gate = F.relu(F.conv2d(gate, s1))
            gate = torch.clamp((F.conv2d(gate, s2) + 3) / 6, 0, 1)
            x = x * gate
        bn_pw = pw if pw <= 23 else (24 if pw == 26 else 26)
        x = cbn(x, pw, bn_pw, inc, outc, 1, 1, 1, state)
        if inc == outc and stride == 1:
            x = x + residual
    x = x.mean(dim=(2, 3), keepdim=True)
    head = state[next(n for n in state if n.startswith("conv2d_31.w_0"))].reshape(1280, 512, 1, 1)
    x = F.hardswish(F.conv2d(x, head))[:, :, 0, 0]
    fw = state[next(n for n in state if n.startswith("linear_0.w_0"))].reshape(2, 1280)
    fb = state[next(n for n in state if n.startswith("linear_0.b_0"))]
    logits = x @ fw.T + fb
    probs = torch.softmax(logits, dim=-1)
    print("pplcnet-reference logits=%.9g,%.9g probs=%.9g,%.9g angle=%d confidence=%.9g" %
          (*logits[0].tolist(), *probs[0].tolist(), 180 if probs[0, 1] > probs[0, 0] else 0, probs.max().item()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("image", type=Path, nargs="?")
    args = ap.parse_args()
    run(args.model_dir, args.image)
