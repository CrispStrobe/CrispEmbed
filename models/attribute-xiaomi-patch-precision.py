#!/usr/bin/env python3
"""Attribute the first Xiaomi patch-embedding difference to reference precision.

Reads the retained frozen inputs and the hosted native outputs, then rebuilds the
same projection under explicit operand/output precision controls. This is an
attribution probe: it does not run a model, a native binary, or the OCR pipeline.

usage: python3 models/attribute-xiaomi-patch-precision.py OUT.json
Needs numpy and gguf; both are already required by the exporter.
"""
import hashlib
import json
from pathlib import Path
import sys

import gguf
import numpy as np

REFERENCE = Path("/Volumes/backups/ai/crispembed-gguf/xiaomi-ocr-0/reference-v3/qianziwen-prefill.gguf")
MODEL = Path("/Volumes/backups/ai/crispembed-gguf/xiaomi-ocr-0/export-control-v1/"
             "xiaomi-ocr-0-export/xiaomi-ocr-0-f32-experimental.gguf")
EVIDENCE = Path("/Volumes/backups/ai/crispembed-gguf/xiaomi-ocr-0/native-patch-v1")
REFERENCE_SHA = "2a58fc3fc2a4ae2d603ff600371cf71fa1f54e40ebd260d7680bfa4c1f868334"
MODEL_SHA = "1759b8da2a6fa2651de9cacdb5e90fe0551d7ab8b1fe783198f7339389451235"
PATCHES, WIDTH, CHANNELS = 1984, 768, 1536


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tensor(reader, name, shape):
    found = [tensor for tensor in reader.tensors if tensor.name == name]
    if len(found) != 1:
        raise ValueError(f"tensor {name}: expected exactly one, found {len(found)}")
    values = np.array(found[0].data, dtype=np.float32)
    if values.size != int(np.prod(shape)):
        raise ValueError(f"tensor {name}: {values.size} values do not match {shape}")
    return values.reshape(shape)


def bf16(values):
    """Round F32 to BF16 with round-to-nearest-even, staying in F32 storage."""
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32)


def metrics(actual, oracle):
    a, o = actual.astype(np.float64), oracle.astype(np.float64)
    delta = a - o
    cos = np.min((a * o).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(o, axis=1)))
    global_cos = float((a * o).sum() / (np.linalg.norm(a) * np.linalg.norm(o)))
    return {"max_abs": float(np.abs(delta).max()), "mean_abs": float(np.abs(delta).mean()),
            "rms": float(np.sqrt(np.square(delta).mean())), "cos_min": float(cos),
            "cos_global": global_cos, "exact_fraction": float(np.mean(delta == 0)),
            "actual_norm": float(np.linalg.norm(a)), "reference_norm": float(np.linalg.norm(o))}


def project(pixels, weight, bias):
    return pixels.astype(np.float64) @ weight.astype(np.float64).T + bias.astype(np.float64)


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    if digest(REFERENCE) != REFERENCE_SHA or digest(MODEL) != MODEL_SHA:
        raise ValueError("retained artifact digest differs from the pinned value")
    ref = gguf.GGUFReader(str(REFERENCE))
    pixels = tensor(ref, "input.pixel_values", (PATCHES, CHANNELS))
    oracle = tensor(ref, "prefill.vision.patch_embed", (PATCHES, WIDTH))
    model = gguf.GGUFReader(str(MODEL))
    weight = tensor(model, "v.patch_embed.proj.weight", (WIDTH, CHANNELS))
    bias = tensor(model, "v.patch_embed.proj.bias", (WIDTH,))
    cpu_path = EVIDENCE / "cpu.log"
    hosted = json.loads((EVIDENCE / "report.json").read_text())
    cpu_metric = next((line for line in cpu_path.read_text().splitlines()
                       if line.startswith("PATCH_METRIC")), "")
    fields = dict(item.split("=", 1) for item in cpu_metric.split() if "=" in item)
    hosted_native = {"max_abs": float(fields["max_abs"]), "rms": float(fields["rms"]),
                     "mean_abs": float(fields["mean_abs"]), "cos_min": float(fields["cos_min"]),
                     "reference_norm": float(fields["reference_norm"]),
                     "strict_pass": fields["pass"] == "1"}

    # Controlled rebuilds of the same projection. Native F32 output is measured
    # through the hosted CPU log; the precision controls are recomputed here.
    f32 = project(pixels, weight, bias).astype(np.float32)
    bf16_operands = project(bf16(pixels), bf16(weight), bf16(bias)).astype(np.float32)
    bf16_both = bf16(bf16_operands)
    result = {
        "scope": "attribution only; retained frozen artifacts are not executed natively here",
        "reference_sha256": REFERENCE_SHA, "model_sha256": MODEL_SHA,
        "hosted_report_sha256": digest(EVIDENCE / "report.json"),
        "controls": {
            "f32_operands_and_output": metrics(f32, oracle),
            "bf16_operands_f32_output": metrics(bf16_operands, oracle),
            "bf16_operands_and_output": metrics(bf16_both, oracle),
        },
        "hosted_native_f32_vs_original": hosted_native,
        "hosted": {"cpu_vs_original": hosted["cpu_vs_original_strict_exit"],
                   "cuda_vs_original": hosted["cuda_vs_original_strict_exit"],
                   "cpu_vs_cuda": hosted["cpu_vs_cuda_f32"]},
        "cpu_metric_line": cpu_metric,
    }
    Path(sys.argv[1]).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["controls"], indent=2))


if __name__ == "__main__":
    main()
