#!/usr/bin/env python3
"""Dump an official PaddleX PP-OCRv6 detector output for crispembed-diff.

The model files are the released safetensors packages.  PaddleX is loaded from
--paddlex-root so this remains independent of an installed PaddleX package.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import paddle
import paddle.nn.functional as F
from PIL import Image
from safetensors.numpy import load_file

sys.path.insert(0, str(Path(__file__).parents[1] / "ggml" / "scripts"))
import gguf


def load_model(root: Path, model_dir: Path):
    inf = root / "paddlex" / "inference"
    models = inf / "models"
    packages = {
        "paddlex": root / "paddlex",
        "paddlex.inference": inf,
        "paddlex.inference.models": models,
        "paddlex.inference.models.image_classification": models / "image_classification",
        "paddlex.inference.models.image_classification.modeling": models / "image_classification" / "modeling",
        "paddlex.inference.models.text_detection": models / "text_detection",
        "paddlex.inference.models.text_detection.modeling": models / "text_detection" / "modeling",
        "paddlex.inference.models.common": inf / "common",
        "paddlex.inference.models.common.transformers": inf / "common" / "transformers",
    }
    for name, path in packages.items():
        mod = types.ModuleType(name)
        mod.__path__ = [str(path)]
        sys.modules[name] = mod
    tr = types.ModuleType("paddlex.inference.models.common.transformers.transformers")

    class PretrainedConfig:
        def __init__(self, **kwargs): self.__dict__.update(kwargs)

    class PretrainedModel(paddle.nn.Layer):
        def __init__(self, config): super().__init__(); self.config = config

    tr.PretrainedConfig = PretrainedConfig
    tr.PretrainedModel = PretrainedModel
    tr.BatchNormHFStateDictMixin = type("BatchNormHFStateDictMixin", (), {})
    sys.modules[tr.__name__] = tr
    act = types.ModuleType("paddlex.inference.models.common.transformers.activations")
    act.ACT2FN = {"relu": F.relu, "gelu": F.gelu, "silu": F.silu}
    sys.modules[act.__name__] = act

    cfg = json.loads((model_dir / "config.json").read_text())
    variant = model_dir.name.split("_")[1]
    module_name = "pp_ocrv6_medium_det" if variant == "medium" else "pp_ocrv6_small_det"
    module = importlib.import_module("paddlex.inference.models.text_detection.modeling." + module_name)
    config = module.PPOCRV6MediumDetConfig(**cfg) if variant == "medium" else module.PPOCRV6SmallDetConfig(**cfg)
    model = module.PPOCRV6MediumDet(config) if variant == "medium" else module.PPOCRV6SmallDet(config)
    source = load_file(str(model_dir / "model.safetensors"))
    values = {}
    for key in model.state_dict():
        source_key = key.replace(".normalization._mean", ".normalization.running_mean")
        source_key = source_key.replace(".normalization._variance", ".normalization.running_var")
        source_key = source_key.replace(".norm._mean", ".norm.running_mean")
        source_key = source_key.replace(".norm._variance", ".norm.running_var")
        if source_key not in source:
            raise KeyError(f"missing source tensor for {key}: {source_key}")
        values[key] = paddle.to_tensor(source[source_key], dtype="float32")
    model.set_state_dict(values)
    model.eval()
    return model, variant


def preprocess(image: Path):
    rgb = np.asarray(Image.open(image).convert("RGB"))
    h, w = rgb.shape[:2]
    rw = max(32, int(np.ceil(w / 32) * 32))
    rh = max(32, int(np.ceil(h / 32) * 32))
    resized = np.asarray(Image.fromarray(rgb).resize((rw, rh), Image.BILINEAR))[:, :, ::-1].astype("float32") / 255
    x = (resized - np.asarray([.485, .456, .406], dtype="float32")) / np.asarray([.229, .224, .225], dtype="float32")
    return x.transpose(2, 0, 1)[None], (h, w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--paddlex-root", type=Path, default=Path("/Volumes/backups/code/PaddleX"))
    args = ap.parse_args()
    model, variant = load_model(args.paddlex_root, args.model_dir)
    x, original = preprocess(args.image)
    with paddle.no_grad():
        tensor = paddle.to_tensor(x)
        stem = model.model.backbone.encoder.convolution
        embedding = stem.stem1(tensor)
        stem_values = {"stem1": embedding}
        embedding = F.pad(embedding, [0, 1, 0, 1])
        branch = stem.stem2a(embedding)
        branch = F.pad(branch, [0, 1, 0, 1])
        branch = stem.stem2b(branch)
        stem_values["stem2b"] = branch
        pooled = stem.pool(embedding)
        stem_values["stem_pooled"] = pooled
        embedding = paddle.concat([pooled, branch], axis=1)
        embedding = stem.stem3(embedding)
        stem_values["stem3"] = embedding
        embedding = stem.stem4(embedding)
        stem_values["stem4"] = embedding
        block0 = model.model.backbone.encoder.blocks[0].blocks[0]
        block = block0.token_conv(embedding)
        stem_values["block0_dw"] = block
        se_pool = block0.token_squeeze_excitation.avg_pool(block)
        se_gate = se_pool
        for layer in block0.token_squeeze_excitation.convolutions:
            se_gate = layer(se_gate)
        stem_values["block0_pool"] = se_pool
        stem_values["block0_gate"] = se_gate
        block = block * se_gate
        stem_values["block0_se"] = block
        block = block0.channel_conv1(block)
        block = block0.channel_act_fn(block)
        stem_values["block0_cm1"] = block
        block = block0.channel_conv2(block) + embedding
        stem_values["block0_out"] = block
        backbone = model.model.backbone(tensor)
        neck = model.model.neck(backbone)
        out = model.head(neck)
    out = np.asarray(out, dtype="float32")[0, 0]
    writer = gguf.GGUFWriter(str(args.output), arch="ppocrv6")
    writer.add_string("general.name", f"PP-OCRv6_{variant}_det-reference")
    writer.add_string("ppocrv6.variant", variant)
    writer.add_string("ppocrv6.kind", "det")
    writer.add_uint32("ppocrv6.reference", 1)
    writer.add_tensor("ppocrv6.input_image", x[0].reshape(-1).astype("float32"))
    for name, value in stem_values.items():
        writer.add_tensor(f"ppocrv6.{name}", value.numpy().reshape(-1).astype("float32"))
    for i, value in enumerate(backbone):
        writer.add_tensor(f"ppocrv6.backbone_stage{i}", value.numpy().reshape(-1).astype("float32"))
    writer.add_tensor("ppocrv6.neck_output", neck.numpy().reshape(-1).astype("float32"))
    writer.add_tensor("ppocrv6.prob_map_sigmoid", out.reshape(-1))
    writer.add_uint32("ppocrv6.input_height", x.shape[2])
    writer.add_uint32("ppocrv6.input_width", x.shape[3])
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
    print(f"wrote {args.output}: input={x.shape[1:]} output={out.shape} original={original}")


if __name__ == "__main__": main()
