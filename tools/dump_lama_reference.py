#!/usr/bin/env python3
"""Capture the unchanged, hash-verified upstream Big-LaMa generator on hosted CPU.

Requires PyTorch, NumPy, Pillow and PyYAML for YAML config. No native inference
is used as ground truth. Small fixtures diagnose numerical parity only; they do
not establish representative subtitle, perceptual or video quality.
"""

import argparse
import hashlib
import importlib
import importlib.machinery
import json
import os
from pathlib import Path
import platform
import sys
import time
import types

from lama_reference_contract import (CASES, INERT_METADATA_TYPES, PROFILE, SOURCE_COMMIT, expected_stage_names,
                                     expected_state_shapes, extract_tensors, fixture, fixture_identity,
                                     load_config_with_sha256, model_metadata, read_f32_gguf, sha256,
                                     validate_input, validate_metadata_graph, validate_reference_torch_version,
                                     validate_sha256, verify_source,
                                     _stream_sha256)


def load_original_module(source_dir):
    """Import original source without executing its unrelated model factories.

    Two training-only imports are fail-closed adapters: the disabled spatial
    wrapper's kornia.rotate and utils.seed_everything. Their original consuming
    modules are unchanged. Invoking either aborts; receipts assert zero calls.
    Namespace routing skips package factory __init__ only, never generator code.
    """
    source_dir = Path(source_dir).resolve()
    source_files = verify_source(source_dir)
    prefixes = ("saicinpainting", "kornia", "pytorch_lightning")
    if any(any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes) for name in sys.modules):
        raise ValueError("run reference capture in a fresh Python process; source dependency modules are already loaded")
    adapter_calls = {"kornia.geometry.transform.rotate": 0, "pytorch_lightning.seed_everything": 0}

    def fail_closed(name):
        def reject(*args, **kwargs):
            adapter_calls[name] += 1
            raise RuntimeError(f"unsupported training-only dependency was invoked: {name}")
        return reject

    def namespace(name, directory=None):
        module = types.ModuleType(name)
        module.__package__ = name
        module.__path__ = [str(directory)] if directory is not None else []
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        sys.modules[name] = module
        return module

    for name in ("saicinpainting", "saicinpainting.training", "saicinpainting.training.modules"):
        namespace(name, source_dir / name.replace(".", "/"))
    for name in ("kornia", "kornia.geometry", "kornia.geometry.transform"):
        namespace(name)
    sys.modules["kornia.geometry.transform"].rotate = fail_closed("kornia.geometry.transform.rotate")
    namespace("pytorch_lightning").seed_everything = fail_closed("pytorch_lightning.seed_everything")
    module = importlib.import_module("saicinpainting.training.modules.ffc")
    for name, imported in tuple(sys.modules.items()):
        if not name.startswith("saicinpainting.") or not getattr(imported, "__file__", None):
            continue
        relative = Path(imported.__file__).resolve().relative_to(source_dir).as_posix()
        if relative not in source_files:
            raise ValueError(f"unverified source dependency was imported: {relative}")
    return module, adapter_calls


def load_original(source_dir, checkpoint, checkpoint_sha256, config, converted_model):
    import numpy as np
    import torch

    validate_reference_torch_version(torch.__version__)
    checkpoint = Path(checkpoint)
    validate_sha256(checkpoint_sha256, "checkpoint SHA-256")
    if sha256(checkpoint) != checkpoint_sha256:
        raise ValueError("checkpoint SHA-256 differs")
    generator_config, config_sha256 = load_config_with_sha256(config)
    module, adapter_calls = load_original_module(source_dir)
    # Bound the pickle program/memo, reject every unknown global and validate
    # inert metadata before invoking PyTorch's independent tensor loader.
    checkpoint_provenance = {}
    parsed_tensors, counters = extract_tensors(checkpoint, checkpoint_sha256, provenance=checkpoint_provenance)
    del parsed_tensors, counters
    globals_by_name = {".".join(name): cls for name, cls in INERT_METADATA_TYPES.items()}
    safe_classes = [globals_by_name[name] for name in checkpoint_provenance["inert_metadata_globals"]]
    with checkpoint.open("rb") as checkpoint_stream:
        if _stream_sha256(checkpoint_stream) != checkpoint_sha256:
            raise ValueError("checkpoint changed before the original loader read it")
        checkpoint_stream.seek(0)
        with torch.serialization.safe_globals(safe_classes):
            state = torch.load(checkpoint_stream, map_location="cpu", weights_only=True)
        checkpoint_stream.seek(0)
        if _stream_sha256(checkpoint_stream) != checkpoint_sha256:
            raise ValueError("checkpoint changed while the original loader was reading it")
    validate_metadata_graph(state, tensor_type=torch.Tensor)
    if not isinstance(state, dict):
        raise ValueError("checkpoint must contain a state dictionary")
    state = state.get("state_dict", state)
    if not isinstance(state, dict) or any(not isinstance(name, str) for name in state):
        raise ValueError("checkpoint state must have string keys")
    if any(name.startswith("generator.") for name in state):
        state = {name[len("generator."):]: tensor for name, tensor in state.items() if name.startswith("generator.")}
    expected = expected_state_shapes(True)
    if set(state) != set(expected):
        raise ValueError("original generator tensor inventory differs from supported profile")
    for name, shape in expected.items():
        tensor = state[name]
        counter = name.endswith(".num_batches_tracked")
        if (not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape
                or tensor.dtype != (torch.int64 if counter else torch.float32)):
            raise ValueError(f"original tensor shape/type differs: {name}")
        if (counter and tensor.item() < 0) or (not counter and not torch.isfinite(tensor).all().item()):
            raise ValueError(f"original tensor is invalid: {name}")
        if name.endswith(".running_var") and (tensor < 0).any().item():
            raise ValueError(f"original BatchNorm variance is negative: {name}")
    torch.manual_seed(0)
    kwargs = {key: value for key, value in generator_config.items() if key != "kind"}
    model = module.FFCResNetGenerator(**kwargs).cpu().float().eval()
    model.load_state_dict(state, strict=True)
    for name, tensor in model.state_dict().items():
        if not torch.equal(tensor, state[name]):
            raise AssertionError(f"original loaded tensor differs: {name}")

    actual_metadata, actual = read_f32_gguf(converted_model)
    if actual_metadata != model_metadata(checkpoint_sha256, config_sha256) or set(actual) != set(expected_state_shapes()):
        raise ValueError("converted GGUF identity/schema differs from original checkpoint")
    inventory = []
    for name in sorted(expected_state_shapes()):
        tensor = state[name]
        data = tensor.detach().cpu().numpy().astype("<f4", copy=False).tobytes()
        if actual[name] != (tuple(tensor.shape), data):
            raise ValueError(f"converted GGUF bytes differ from independent PyTorch checkpoint loader: {name}")
        inventory.append({"name": name, "shape": list(tensor.shape), "bytes": len(data),
                          "sha256": hashlib.sha256(data).hexdigest(),
                          "min": float(tensor.min()), "max": float(tensor.max()),
                          "l2": float(np.linalg.norm(tensor.numpy().astype(np.float64)))})
    return model, inventory, adapter_calls, checkpoint_provenance, config_sha256


def preprocess(rgb, mask, width, height):
    import numpy as np
    import torch

    validate_input(rgb, mask, width, height)
    image = np.frombuffer(rgb, dtype=np.uint8).reshape(height, width, 3).transpose(2, 0, 1).astype("float32") / 255
    fill = np.frombuffer(mask, dtype=np.uint8).reshape(1, height, width).astype("float32") / 255
    padded_width, padded_height = (width + 7) // 8 * 8, (height + 7) // 8 * 8
    padding = ((0, 0), (0, padded_height - height), (0, padded_width - width))
    image = torch.from_numpy(np.pad(image, padding, mode="symmetric")).unsqueeze(0)
    fill = torch.from_numpy(np.pad(fill, padding, mode="symmetric")).unsqueeze(0)
    # Original DefaultInpaintingTrainingModule.forward: image*(1-mask),mask.
    return torch.cat([image * (1 - fill), fill], dim=1)


def decode(raw, rgb, mask, width, height):
    import numpy as np

    generated = np.clip(raw[0].detach().cpu().numpy().transpose(1, 2, 0)[:height, :width] * 255, 0, 255).astype("uint8")
    original = np.frombuffer(rgb, dtype=np.uint8).reshape(height, width, 3)
    fill = np.frombuffer(mask, dtype=np.uint8).reshape(height, width) != 0
    composed = original.copy()
    composed[fill] = generated[fill]
    if not np.array_equal(composed[~fill], original[~fill]):
        raise AssertionError("known pixels changed during original composition")
    return generated.tobytes(), composed.tobytes()


def capture_case(model, name, directory):
    import numpy as np
    from PIL import Image
    import torch

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    width, height, rgb, mask = fixture(name)
    x = preprocess(rgb, mask, width, height)
    (directory / "input.rgb").write_bytes(rgb)
    (directory / "mask.gray").write_bytes(mask)
    Image.frombytes("RGB", (width, height), rgb).save(directory / "input.png")
    Image.frombytes("L", (width, height), mask).save(directory / "mask.png")
    records = []
    names_seen = set()

    def save(stage, tensor):
        if stage in names_seen:
            raise AssertionError(f"duplicate original stage: {stage}")
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32 or tensor.ndim != 4:
            raise AssertionError(f"invalid original stage tensor: {stage}")
        # Clone before subsequent in-place activations change their inputs.
        arr = tensor.detach().cpu().numpy().astype("<f4", copy=True)
        if not np.isfinite(arr).all():
            raise AssertionError(f"nonfinite original stage: {stage}")
        path = directory / (stage + ".f32")
        path.write_bytes(arr.tobytes())
        records.append({"stage": stage, "file": path.name, "shape": list(arr.shape),
                        "sha256": sha256(path), "min": float(arr.min()), "max": float(arr.max()),
                        "l2": float(np.linalg.norm(arr.astype(np.float64)))})
        names_seen.add(stage)

    def save_tuple(stage, value):
        if not isinstance(value, tuple) or len(value) != 2:
            raise AssertionError(f"expected original FFC tuple: {stage}")
        save(stage + ".local", value[0])
        if isinstance(value[1], torch.Tensor):
            save(stage + ".global", value[1])

    handles = []
    with torch.inference_mode():
        start = time.perf_counter()
        baseline = model(x).clone()
        elapsed = time.perf_counter() - start
        save("generator.input", x)
        # The original constructor reuses one ReLU object for layers26/29/32.
        # Register once per identity and label each invocation in sequence.
        required = set(expected_stage_names())
        shared_ids = {id(model.model[index]) for index in (26, 29, 32)}
        for stage, layer in model.named_modules():
            if not stage or id(layer) in shared_ids:
                continue
            if stage + ".output.local" in required:
                handles.append(layer.register_forward_hook(
                    lambda mod, inputs, out, stage=stage: save_tuple(stage + ".output", out)))
            elif stage + ".output" in required:
                handles.append(layer.register_forward_hook(
                    lambda mod, inputs, out, stage=stage: save(stage + ".output", out)))
            if stage + ".input" in required:
                handles.append(layer.register_forward_pre_hook(
                    lambda mod, inputs, stage=stage: save(stage + ".input", inputs[0])))
            if stage + ".rfft" in required:
                handles.append(layer.conv_layer.register_forward_pre_hook(
                    lambda mod, inputs, stage=stage: save(stage + ".rfft", inputs[0])))
        sequenced = {}
        for index in (26, 29, 32):
            layer = model.model[index]
            sequenced.setdefault(id(layer), (layer, []))[1].append(index)
        for layer, indices in sequenced.values():
            cursor = [0]

            def save_sequential(mod, inputs, out, indices=indices, cursor=cursor):
                if cursor[0] >= len(indices):
                    raise AssertionError("unexpected reused-module invocation")
                save(f"model.{indices[cursor[0]]}.output", out)
                cursor[0] += 1

            handles.append(layer.register_forward_hook(save_sequential))
        try:
            observed = model(x).clone()
        finally:
            for handle in handles:
                handle.remove()
        if not torch.equal(baseline, observed):
            raise AssertionError("original observer hooks changed final output")
        save("generator.output", baseline)
    if [row["stage"] for row in records] != expected_stage_names():
        raise AssertionError("original stage inventory/execution order differs from the declared trace contract")
    generated, composed = decode(baseline, rgb, mask, width, height)
    (directory / "generated.rgb").write_bytes(generated)
    (directory / "output.rgb").write_bytes(composed)
    Image.frombytes("RGB", (width, height), generated).save(directory / "generated.png")
    Image.frombytes("RGB", (width, height), composed).save(directory / "output.png")
    manifest = {"schema": "crispembed.lama.reference.v1", **fixture_identity(name), "records": records,
                "generator_seconds": elapsed, "hook_neutrality_bit_exact": True,
                "known_pixels_exact": True, "output_sha256": sha256(directory / "output.rgb"),
                "generated_sha256": sha256(directory / "generated.rgb")}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {**fixture_identity(name), "manifest_sha256": sha256(directory / "manifest.json"),
            "stages": len(records), "hook_neutrality_bit_exact": True, "known_pixels_exact": True}


def run(source_dir, checkpoint, checkpoint_sha256, config, converted_model, output, threads=2):
    import numpy as np
    from PIL import __version__ as pillow_version
    import torch

    validate_reference_torch_version(torch.__version__)
    if not 1 <= threads <= 64:
        raise ValueError("threads must be in 1..64")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    model, inventory, adapter_calls, checkpoint_provenance, config_sha256 = load_original(
        source_dir, checkpoint, checkpoint_sha256, config, converted_model)
    report = {
        "schema": "crispembed-lama-original-v1", "profile": PROFILE, "complete": False,
        "source_commit": SOURCE_COMMIT, "source_files": verify_source(source_dir),
        "checkpoint_sha256": checkpoint_sha256, "config_sha256": config_sha256,
        "model_sha256": sha256(converted_model), "all_converted_tensor_bytes_exact": True,
        "state": inventory, "cases": [], "native_qualified": False, "representative_quality_qualified": False,
        "import_adapters": {"namespace_factory_initializers_skipped": True, "fail_closed_calls": adapter_calls},
        "checkpoint_metadata": checkpoint_provenance,
        "producer": {"github_sha": os.environ.get("GITHUB_SHA"), "github_run_id": os.environ.get("GITHUB_RUN_ID"),
                     "dumper_sha256": sha256(__file__), "contract_sha256": sha256(Path(__file__).with_name("lama_reference_contract.py"))},
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "pillow": pillow_version},
        "backend": {"platform": platform.platform(), "threads": torch.get_num_threads(),
                    "interop_threads": torch.get_num_interop_threads(), "mkldnn_enabled": torch.backends.mkldnn.enabled,
                    "torch_config": torch.__config__.show()},
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name in CASES:
        report["cases"].append(capture_case(model, name, output / name))
        if any(adapter_calls.values()):
            raise AssertionError("training-only import adapter was unexpectedly invoked")
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"captured original LaMa case {name}", flush=True)
    report["complete"] = True
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--converted-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    run(args.source_dir, args.checkpoint, args.checkpoint_sha256, args.config, args.converted_model, args.output, args.threads)


if __name__ == "__main__":
    main()
