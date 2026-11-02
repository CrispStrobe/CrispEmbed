# SPDX-License-Identifier: Apache-2.0
# TeleOCR diagnostic support for CrispEmbed. Model equations and modules:
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
# See NOTICE.teleocr and models/licenses/apache-2.0/LICENSE.
"""Shared I/O and provenance for the bounded TeleOCR numerical probes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import gguf
import numpy as np
from safetensors import safe_open
import torch
import transformers


REVISION = "b3f6a0c084db72f971e19a779d2fd8a6bc9e3b8d"
MODULE_SHA256 = "7adac1cc17009f9f1b8e0c58284d7c0b9989a590f017b9e5964cc42270207922"
CONFIG_SHA256 = "92c5e15f1768dcb54bceae170043fbfefd248508159700b2179824b88c33ae75"
MAX_ABS = 0.002
MIN_COS = 0.99999


def arguments(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Pinned original checkpoint, config.json and modeling_naviocr.py")
    parser.add_argument("--reference-dir", type=Path, required=True,
                        help="Frozen case stage directory, for example reference/text-stages")
    parser.add_argument("--native-dir", type=Path, required=True,
                        help="Native case stage directory from --dump-native-stages")
    parser.add_argument("--out", type=Path, required=True, help="New or empty diagnostic output directory")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    for name in ("model_dir", "reference_dir", "native_dir"):
        value = getattr(args, name).resolve(strict=True)
        if not value.is_dir():
            parser.error(f"--{name.replace('_', '-')} must be a directory")
        setattr(args, name, value)
    if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
        parser.error("output must be new or empty; retain previous diagnostic failures")
    args.out.mkdir(parents=True, exist_ok=True)
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(args: argparse.Namespace, name: str):
    if transformers.__version__ != "4.57.1" or torch.__version__.split("+")[0] != "2.8.0":
        raise ValueError("These recorded probes require transformers==4.57.1 and torch==2.8.0")
    module_path = args.model_dir / "modeling_naviocr.py"
    config_path = args.model_dir / "config.json"
    if sha256(module_path) != MODULE_SHA256 or sha256(config_path) != CONFIG_SHA256:
        raise ValueError("Refusing to execute a model module/configuration that differs from the pinned source")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load the pinned TeleOCR model module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return module, config


def load_block_state(model_dir: Path, prefix: str):
    # Only the selected block is materialized. The full model is never constructed.
    with safe_open(model_dir / "model.safetensors", framework="pt") as source:
        state = {key[len(prefix):]: source.get_tensor(key).float()
                 for key in source.keys() if key.startswith(prefix)}
    hashes = {prefix + key: hashlib.sha256(value.numpy().tobytes()).hexdigest()
              for key, value in state.items()}
    return state, hashes


def reference_array(directory: Path, name: str) -> torch.Tensor:
    array = np.load(directory / f"{name}.npy", allow_pickle=False)
    if array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError(f"{name}: expected a finite frozen float32 array")
    return torch.from_numpy(array)


def native_array(directory: Path, name: str, shape) -> torch.Tensor:
    path = directory / f"{name}.f32"
    if path.stat().st_size != int(np.prod(shape)) * 4:
        raise ValueError(f"{name}: native dump size does not match the frozen shape")
    array = np.fromfile(path, dtype=np.float32).reshape(shape)
    if not np.isfinite(array).all():
        raise ValueError(f"{name}: nonfinite native dump")
    return torch.from_numpy(array)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_probe(out: Path, arrays: dict[str, torch.Tensor]) -> None:
    writer = gguf.GGUFWriter(out / "probe.gguf", "teleocr_probe")
    try:
        for name, value in arrays.items():
            array = value.detach().numpy()
            writer.add_tensor(name, array)
            np.save(out / f"{name}.npy", array)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        writer.close()


def comparison(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    if actual.shape != expected.shape or not actual.numel():
        raise ValueError("Cannot compare empty arrays or different shapes")
    a, b = actual.double(), expected.double()
    difference = (a - b).abs()
    index = tuple(int(i) for i in np.unravel_index(int(difference.argmax()), difference.shape))
    rows_a, rows_b = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
    aa, bb = rows_a.square().sum(-1), rows_b.square().sum(-1)
    denominator = torch.sqrt(aa * bb)
    cosines = torch.where(denominator > 0, (rows_a * rows_b).sum(-1) / denominator,
                          torch.where((aa == 0) & (bb == 0), 1.0, 0.0))
    maximum, cosine = float(difference[index]), float(cosines.min())
    return {
        "max_abs": maximum, "index": list(index),
        "actual": float(a[index]), "reference": float(b[index]),
        "count_over_0_002": int(torch.count_nonzero(difference > MAX_ABS)),
        "min_cos": cosine,
        "pass": bool(torch.isfinite(a).all() and torch.isfinite(b).all()
                     and maximum <= MAX_ABS and cosine >= MIN_COS),
    }


def provenance(args: argparse.Namespace, script: Path, hashes: dict,
               reference_names: tuple[str, ...], native_names: tuple[str, ...]) -> None:
    write_json(args.out / "provenance.json", {
        "expected_model_revision": REVISION,
        "module_sha256": MODULE_SHA256, "config_sha256": CONFIG_SHA256,
        "selected_weights_float32_sha256": hashes,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "numpy": np.__version__, "threads": args.threads,
        "script_sha256": sha256(script), "common_sha256": sha256(Path(__file__)),
        "reference_sha256": {name: sha256(args.reference_dir / f"{name}.npy") for name in reference_names},
        "native_sha256": {name: sha256(args.native_dir / f"{name}.f32") for name in native_names},
        "max_abs_budget": MAX_ABS, "min_cos_budget": MIN_COS,
        "scope": "single-block diagnosis; does not replace full-model qualification",
    })
