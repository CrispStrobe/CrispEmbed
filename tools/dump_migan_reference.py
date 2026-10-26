#!/usr/bin/env python3
"""Capture the unchanged, hash-verified upstream MI-GAN module on hosted CPU.

No graph reimplementation is used as the reference. Float arrays and decoded
images are diagnostic artifacts, separate from the production-weight GGUF.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import time

from migan_reference_contract import (CASES, FULL_STAGE_CASES, fixture, fixture_identity,
                                      load_contract, sha256, validate_input,
                                      validate_state_shapes, write_f32_gguf)


def load_original(source_dir, checkpoint, checkpoint_sha256):
    import torch

    contract = load_contract()
    source_dir = Path(source_dir)
    for name, record in contract["original_source"]["files"].items():
        path = source_dir / name
        if sha256(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise ValueError(f"original source identity differs: {name}")
    checkpoint = Path(checkpoint)
    if sha256(checkpoint) != checkpoint_sha256:
        raise ValueError("checkpoint SHA-256 differs")
    if checkpoint.stat().st_size != contract["checkpoint"]["declared_bytes"]:
        raise ValueError("checkpoint byte count differs")
    with checkpoint.open("rb") as stream:
        if stream.read(4) != b"PK\x03\x04":
            raise ValueError("checkpoint is not the expected torch ZIP payload")
    spec = importlib.util.spec_from_file_location("crispembed_original_migan", source_dir / "lib/model_zoo/migan_inference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(0)  # Initial random noise buffers must all be overwritten.
    model = module.Generator(resolution=256).cpu().float().eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("checkpoint must be a plain state dictionary")
    validate_state_shapes({name: tensor.shape for name, tensor in state.items()})
    for name, tensor in state.items():
        if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
            raise ValueError(f"checkpoint must contain finite F32 tensors: {name}")
    model.load_state_dict(state, strict=True)
    for name, loaded in model.state_dict().items():
        if not torch.equal(loaded, state[name]):
            raise AssertionError(f"loaded state differs: {name}")
    return model, state


def inventory(state):
    import numpy as np

    rows = []
    for name, tensor in sorted(state.items()):
        arr = tensor.detach().cpu().numpy().astype("<f4", copy=False)
        rows.append({"name": name, "shape": list(arr.shape), "dtype": "float32",
                     "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
                     "min": float(arr.min()), "max": float(arr.max()),
                     "l2": float(np.linalg.norm(arr.astype(np.float64))),
                     "nonzero": int(np.count_nonzero(arr)),
                     "role": "buffer" if name.endswith(("noise_const", "filter_const")) else "parameter"})
    return rows


def preprocess(rgb, fill_mask):
    import numpy as np
    import torch

    validate_input(rgb, fill_mask)
    image = torch.tensor(np.frombuffer(rgb, dtype=np.uint8).copy().reshape(256, 256, 3)).float() * 2 / 255 - 1
    # Invert exactly once at the public white=fill boundary.
    known = torch.tensor((255 - np.frombuffer(fill_mask, dtype=np.uint8)).copy().reshape(256, 256, 1) // 255).float()
    image = image.permute(2, 0, 1).unsqueeze(0)
    known = known.permute(2, 0, 1).unsqueeze(0)
    return torch.cat([known - 0.5, image * known], dim=1), known


def decode(raw, rgb, fill_mask):
    import numpy as np
    import torch

    generated = ((raw[0] * 0.5 + 0.5).clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    original = np.frombuffer(rgb, dtype=np.uint8).reshape(256, 256, 3)
    fill = np.frombuffer(fill_mask, dtype=np.uint8).reshape(256, 256, 1) // 255
    composed = original * (1 - fill) + generated * fill
    if not np.array_equal(composed[np.repeat(fill == 0, 3, axis=2)], original[np.repeat(fill == 0, 3, axis=2)]):
        raise AssertionError("known pixels changed")
    return generated, composed


def capture_case(model, name, directory):
    import numpy as np
    from PIL import Image
    import torch

    directory.mkdir(parents=True, exist_ok=False)
    rgb, mask = fixture(name)
    x, known = preprocess(rgb, mask)
    Image.frombytes("RGB", (256, 256), rgb).save(directory / "input.png")
    Image.frombytes("L", (256, 256), mask).save(directory / "fill-mask.png")
    records = []

    def save(stage, value):
        if isinstance(value, torch.Tensor):
            arr = value.detach().cpu().numpy().astype("<f4", copy=True)
            if not np.isfinite(arr).all():
                raise AssertionError(f"nonfinite original stage: {stage}")
            path = directory / f"{stage}.npy"
            np.save(path, arr, allow_pickle=False)
            records.append({"stage": stage, "file": path.name, "shape": list(arr.shape),
                            "sha256": sha256(path), "min": float(arr.min()), "max": float(arr.max()),
                            "l2": float(np.linalg.norm(arr.astype(np.float64)))})
        elif isinstance(value, (tuple, list)):
            for index, child in enumerate(value):
                save(f"{stage}.{index}", child)
        elif isinstance(value, dict):
            for key, child in value.items():
                save(f"{stage}.{key}", child)

    with torch.inference_mode():
        save("generator.input", x)
        save("known_mask", known)
        start = time.perf_counter()
        baseline = model(x).clone()
        elapsed = time.perf_counter() - start
        if list(baseline.shape) != [1, 3, 256, 256] or baseline.dtype != torch.float32:
            raise AssertionError("original output contract differs")
        handles = []
        if name in FULL_STAGE_CASES:
            for stage, layer in model.named_modules():
                if not stage:
                    continue
                # Every real convolution, separable layer and encoder/synthesis
                # block; snapshots precede later in-place activations/additions.
                if layer.__class__.__name__ in ("Conv2d", "SeparableConv2d", "EncoderBlock", "SynthesisBlock", "SynthesisBlockFirst", "Downsample2d", "Upsample2d"):
                    if layer.__class__.__name__ != "Conv2d":
                        handles.append(layer.register_forward_pre_hook(lambda mod, args, stage=stage: save(stage + ".input", args)))
                    handles.append(layer.register_forward_hook(lambda mod, args, out, stage=stage: save(stage + ".output", out)))
            try:
                observed = model(x)
            finally:
                for handle in handles:
                    handle.remove()
            if not torch.equal(baseline, observed):
                raise AssertionError("hooks changed original output")
        save("generator.output", baseline)
    generated, composed = decode(baseline, rgb, mask)
    Image.fromarray(generated).save(directory / "generated.png")
    Image.fromarray(composed).save(directory / "composited.png")
    np.save(directory / "composited.npy", composed, allow_pickle=False)
    # A diagnostic GGUF only. This is never an input to weight conversion.
    write_f32_gguf(directory / "reference.gguf",
                   ((row["stage"], row["shape"], np.load(directory / row["file"], allow_pickle=False).tobytes()) for row in records),
                   {"general.architecture": "migan-reference", "migan.image_size": 256})
    return {**fixture_identity(name), "stages": records, "hook_neutrality_bit_exact": name in FULL_STAGE_CASES,
            "generator_seconds": elapsed, "known_pixels_bit_exact": True,
            "generated_png_sha256": sha256(directory / "generated.png"),
            "composited_png_sha256": sha256(directory / "composited.png"),
            "reference_gguf_sha256": sha256(directory / "reference.gguf")}


def run(source_dir, checkpoint, checkpoint_sha256, output, threads=4):
    import numpy as np
    from PIL import __version__ as pillow_version
    import torch

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if load_contract()["fixtures"] != [fixture_identity(name) for name in CASES]:
        raise ValueError("fixture bytes differ from the frozen protocol")
    model, state = load_original(source_dir, checkpoint, checkpoint_sha256)
    report = {"schema": "crispembed-migan-original-v1", "checkpoint_sha256": checkpoint_sha256,
              "source_commit": load_contract()["original_source"]["commit"],
              "producer": {"github_sha": os.environ.get("GITHUB_SHA"), "github_run_id": os.environ.get("GITHUB_RUN_ID"),
                           "dumper_sha256": sha256(__file__),
                           "helper_sha256": sha256(Path(__file__).with_name("migan_reference_contract.py"))},
              "contract_sha256": sha256(Path(__file__).resolve().parents[1] / "docs/migan-reference-contract.json"),
              "versions": {"python": platform.python_version(), "torch": torch.__version__,
                           "numpy": np.__version__, "pillow": pillow_version},
              "backend": {"platform": platform.platform(), "processor": platform.processor(),
                          "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
                          "mkldnn_enabled": torch.backends.mkldnn.enabled, "torch_config": torch.__config__.show()},
              "state": inventory(state), "cases": [], "native_qualified": False, "representative_quality_qualified": False}
    (output / "tensor-inventory.json").write_text(json.dumps(report["state"], indent=2) + "\n")
    for name in CASES:
        report["cases"].append(capture_case(model, name, output / name))
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"captured {name}", flush=True)
    # Contract-level negative paths are exercised without preprocessing coercion.
    rgb, mask = fixture("empty")
    rejected = []
    for label, args in (("nonbinary", (rgb, bytes([127]) + mask[1:])),
                        ("mask_size", (rgb, mask[:-1])), ("rgb_channels", (rgb[:-1], mask)),
                        ("wrong_resolution", (rgb, mask, 512, 256))):
        try:
            validate_input(*args)
        except ValueError:
            rejected.append(label)
        else:
            raise AssertionError(f"negative control accepted: {label}")
    report["rejected_controls"] = rejected
    report["complete"] = True
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    run(args.source_dir, args.checkpoint, args.checkpoint_sha256, args.output, args.threads)


if __name__ == "__main__":
    main()
