#!/usr/bin/env python3
"""Dump PP-OCRv6 recognizer intermediates into a diff-harness GGUF.

The official PaddleX blueprint is Paddle-based.  This reference runner mirrors
its inference equations with torch, loading the published safetensors directly.
It intentionally emits the input, backbone boundaries, head input and logits;
the C++ backend can compare the earliest failing boundary instead of only the
final CTC string.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.numpy import load_file

import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "ggml" / "scripts"))
import gguf


class Ref:
    def __init__(self, path: Path):
        self.d = {k: torch.from_numpy(v.astype(np.float32, copy=False)) for k, v in load_file(str(path)).items()}

    def w(self, key):
        return self.d[key]

    def conv(self, x, key, stride=1, pad=None, groups=1):
        w = self.w(key + ".weight")
        b = self.d.get(key + ".bias")
        if pad is None:
            pad = w.shape[-1] // 2
        return F.conv2d(x, w, b, stride=stride, padding=pad, groups=groups)

    def bn(self, x, key):
        return F.batch_norm(x, self.w(key + ".running_mean"), self.w(key + ".running_var"),
                            self.w(key + ".weight"), self.w(key + ".bias"), False, 0.1, 1e-5)

    def layer(self, x, key, stride=1, groups=1, activation=None):
        x = self.conv(x, key + ".convolution", stride, groups=groups)
        x = self.bn(x, key + ".normalization")
        if activation == "gelu": x = F.gelu(x)
        elif activation == "hs": x = F.hardswish(x)
        return x

    def block(self, x, si, bi, in_ch, out_ch, stride, se, stages=None):
        p = f"model.backbone.encoder.blocks.{si}.blocks.{bi}"
        token = p + ".token_conv"
        if token + ".weight" not in self.d:
            token += ".convolution"
        y = self.conv(x, token, stride, groups=in_ch)
        if token.endswith(".convolution"):
            y = self.bn(y, token[:-len(".convolution")] + ".normalization")
        if stages is not None and si == 0 and bi == 0: stages["block0_dw"] = y
        if se:
            g = y.mean(dim=(2, 3), keepdim=True)
            g = F.relu(self.conv(g, p + ".token_squeeze_excitation.convolutions.0", pad=0))
            g = torch.clamp((self.conv(g, p + ".token_squeeze_excitation.convolutions.2", pad=0) + 3) / 6, 0, 1)
            y = y * g
            if stages is not None and si == 0 and bi == 0: stages["block0_se"] = y
        z = self.layer(y, p + ".channel_conv1", activation="gelu")
        if stages is not None and si == 0 and bi == 0: stages["block0_cm1"] = z
        z = self.layer(z, p + ".channel_conv2")
        return y + z if in_ch == out_ch and stride == 1 else z


def preprocess(path: Path):
    im = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    h, w = im.shape[:2]
    rw = min(320, max(1, round(w * 48 / h)))
    yy = np.maximum(0.0, (np.arange(48, dtype=np.float32) + 0.5) * h / 48.0 - 0.5)
    xx = np.maximum(0.0, (np.arange(rw, dtype=np.float32) + 0.5) * w / rw - 0.5)
    y0 = np.floor(yy).astype(np.int32).clip(0, h - 1); y1 = np.minimum(y0 + 1, h - 1)
    x0 = np.floor(xx).astype(np.int32).clip(0, w - 1); x1 = np.minimum(x0 + 1, w - 1)
    wy = (yy - y0)[:, None, None]; wx = (xx - x0)[None, :, None]
    a = (im[y0[:, None], x0[None, :]] * (1-wy) * (1-wx) +
         im[y0[:, None], x1[None, :]] * (1-wy) * wx +
         im[y1[:, None], x0[None, :]] * wy * (1-wx) +
         im[y1[:, None], x1[None, :]] * wy * wx).astype(np.float32) / 255.0
    a = (a - 0.5) / 0.5
    out = np.zeros((48, 320, 3), dtype=np.float32)
    out[:, :rw] = a
    return torch.from_numpy(out.transpose(2, 0, 1))[None]


def dump(model_dir: Path, image: Path, output: Path):
    ref = Ref(model_dir / "model.safetensors")
    cfg = __import__("json").loads((model_dir / "config.json").read_text())
    if cfg["model_type"] != "pp_ocrv6_tiny_rec":
        raise SystemExit("reference runner currently covers tiny_rec; extend SVTR/det before dumping those variants")
    stages = {}
    x = preprocess(image)
    stages["input"] = x
    x = ref.layer(x, "model.backbone.encoder.convolution.conv1", stride=2)
    stages["stem1_pre"] = x
    x = F.gelu(x)
    stages["stem1"] = x
    x = ref.layer(x, "model.backbone.encoder.convolution.conv2", stride=2)
    stages["stem2"] = x
    configs = [[(3, 48, 48, 1, True)], [(3, 48, 48, 1, False)],
               [(3, 48, 96, 2, False), (3, 96, 96, 1, True), (3, 96, 96, 1, False)],
               [(3, 96, 160, 2, False), (3, 160, 160, 1, True), (3, 160, 160, 1, False), (3, 160, 160, 1, False)]]
    for si, blocks in enumerate(configs):
        for bi, (_, inc, outc, stride, se) in enumerate(blocks):
            x = ref.block(x, si, bi, inc, outc, stride if isinstance(stride, int) else stride[0], se, stages)
        stages[f"stage{si + 1}"] = x
    x = F.avg_pool2d(x, (3, 2))
    x = x.squeeze(2)
    x = F.hardswish(ref.bn(F.conv1d(x, ref.w("head.conv1.weight"), padding=2, groups=160), "head.norm1"))
    stages["head_conv1"] = x
    x = F.conv1d(x, ref.w("head.conv2.weight"))
    stages["head_conv2_pre"] = x
    x = ref.bn(x, "head.norm2")
    stages["head_norm2"] = x
    x = F.hardswish(x)
    x = x.transpose(1, 2)
    hidden = F.linear(x, ref.w("head.fc1.weight"), ref.w("head.fc1.bias"))
    logits = F.linear(hidden, ref.w("head.fc2.weight"), ref.w("head.fc2.bias"))
    stages["head_input"] = x
    stages["logits"] = logits
    writer = gguf.GGUFWriter(str(output), arch="ppocrv6")
    writer.add_string("general.name", cfg["model_type"])
    writer.add_string("ppocrv6.variant", "tiny")
    writer.add_string("ppocrv6.kind", "rec")
    writer.add_uint32("ppocrv6.reference", 1)
    for name, value in stages.items():
        # Store activations as one flat row.  The gguf Python writer reverses
        # multidimensional metadata axes; a flat reference keeps the C++ diff
        # harness from accidentally treating CHW pixels as RGB rows.
        writer.add_tensor("ppocrv6." + name, value[0].detach().numpy().reshape(-1).astype(np.float32))
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
    print(f"wrote {output} ({len(stages)} stages)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    dump(a.model_dir, a.image, a.output)
