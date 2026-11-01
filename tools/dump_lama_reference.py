#!/usr/bin/env python3
"""Execute the pinned, unmodified Big-LaMa generator on a hosted CPU worker.

Uses the inert checkpoint reader; never unpickles executable checkpoint code.
Every original stage is captured with hooks after an unobserved baseline, whose
output must remain bit-identical. No native result is used to create a reference.
"""
import argparse
import importlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time
import types

from lama_contract import Checkpoint, load_contract, verify_source
from lama_reference_contract import CASES, SOURCE_COMMIT, THRESHOLDS, expected_stages, fixture, identity
from migan_reference_contract import sha256, write_f32_gguf

DEPENDENCIES = {"torch": "2.8.0", "numpy": "2.1.3", "Pillow": "11.3.0",
                "omegaconf": "2.3.0", "kornia": "0.8.1", "pytorch-lightning": "2.5.3"}


def original(source_dir, checkpoint_archive):
    contract = load_contract()
    verify_source(source_dir, contract)
    versions = {}
    for package, expected in DEPENDENCIES.items():
        actual = importlib.metadata.version(package)
        if actual.split("+")[0] != expected:
            raise ValueError(f"original dependency version differs: {package}: {actual} != {expected}")
        versions[package] = actual
    import numpy as np
    import torch
    from omegaconf import OmegaConf

    source_dir = Path(source_dir).resolve()
    # These parent package shells only select original module search paths.
    # Every numerical module below is executed unchanged from verified source;
    # unrelated training/discriminator factory __init__ code is not needed.
    for name in ("saicinpainting", "saicinpainting.training", "saicinpainting.training.modules"):
        if name in sys.modules:
            raise RuntimeError("original LaMa packages were already imported")
        package = types.ModuleType(name)
        package.__path__ = [str(source_dir.joinpath(*name.split(".")))]
        sys.modules[name] = package
    module = importlib.import_module("saicinpainting.training.modules.ffc")
    config = OmegaConf.load(source_dir / "configs/training/big-lama.yaml")
    kwargs = OmegaConf.to_container(config.generator, resolve=True)
    if kwargs.pop("kind") != "ffc_resnet":
        raise ValueError("original generator config differs")
    torch.manual_seed(0)
    model = module.FFCResNetGenerator(**kwargs).cpu().float().eval()
    destination = model.state_dict()
    with Checkpoint(checkpoint_archive, contract) as checkpoint:
        if set(destination) != {name.removeprefix("generator.") for name in checkpoint.state}:
            raise ValueError("original constructor/checkpoint generator keys differ")
        with torch.no_grad():
            for name, description in checkpoint.state.items():
                target = destination[name.removeprefix("generator.")]
                if list(target.shape) != list(description.shape):
                    raise ValueError(f"original tensor shape differs: {name}")
                if name.endswith(".num_batches_tracked"):
                    data = torch.tensor(checkpoint.read_counter(name), dtype=torch.int64)
                else:
                    raw = b"".join(checkpoint.f32_blocks(name))
                    data = torch.from_numpy(np.frombuffer(raw, dtype="<f4").copy().reshape(description.shape))
                if target.dtype != data.dtype or not torch.isfinite(data).all():
                    raise ValueError(f"original tensor type/value differs: {name}")
                target.copy_(data)
                if not torch.equal(target, data):
                    raise AssertionError(f"original state copy changed data: {name}")
    data_path = source_dir / "saicinpainting/evaluation/data.py"
    spec = importlib.util.spec_from_file_location("crispembed_original_lama_data", data_path)
    data_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data_module)
    return model, data_module.pad_img_to_modulo, versions


def capture(model, pad_original, name, root, archive_sha256, contract_sha256):
    import numpy as np
    from PIL import Image
    import torch

    width, height = CASES[name]
    rgb, mask_bytes = fixture(name)
    directory = root / name
    directory.mkdir()
    (directory / "input.rgb8").write_bytes(rgb)
    (directory / "fill-mask.u8").write_bytes(mask_bytes)
    Image.frombytes("RGB", (width, height), rgb).save(directory / "input.png")
    Image.frombytes("L", (width, height), mask_bytes).save(directory / "fill-mask.png")
    # Same original dataset normalization/padding and DefaultInpaintingTrainingModel
    # masked-RGB+mask boundary; the generator itself is the original module.
    image = np.frombuffer(rgb, dtype=np.uint8).reshape(height, width, 3).transpose(2, 0, 1).astype("float32") / 255
    mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(1, height, width).astype("float32") / 255
    image, mask = torch.from_numpy(pad_original(image, 8)), torch.from_numpy(pad_original(mask, 8))
    conditioned = torch.cat((image * (1 - mask), mask), dim=0).unsqueeze(0)
    records = {}

    def save(stage, value):
        if stage in records:
            raise AssertionError(f"duplicate original stage: {stage}")
        if not isinstance(value, torch.Tensor) or value.ndim != 4 or value.shape[0] != 1:
            raise AssertionError(f"original stage is not batch-one CHW: {stage}")
        array = value[0].detach().cpu().numpy().astype("<f4", copy=True)
        if not np.isfinite(array).all():
            raise AssertionError(f"nonfinite original stage: {stage}")
        path = directory / (stage + ".npy")
        np.save(path, array, allow_pickle=False)
        records[stage] = {"stage": stage, "file": path.name, "shape": list(array.shape),
                          "sha256": sha256(path), "min": float(array.min()), "max": float(array.max())}

    def pair_hook(stage):
        def hook(_module, _args, output):
            save(stage + ".local", output[0])
            if isinstance(output[1], torch.Tensor):
                save(stage + ".global", output[1])
        return hook

    handles = []
    with torch.inference_mode():
        started = time.perf_counter()
        baseline = model(conditioned).clone()
        seconds = time.perf_counter() - started
        save("input", conditioned)
        for layer_name, layer in model.named_modules():
            stage = "generator." + layer_name
            kind = layer.__class__.__name__
            if kind in ("FFC_BN_ACT", "FFCResnetBlock"):
                handles.append(layer.register_forward_hook(pair_hook(stage)))
            elif kind == "FourierUnit":
                handles.append(layer.conv_layer.register_forward_pre_hook(
                    lambda _m, args, stage=stage: save(stage + ".rfft", args[0])))
                handles.append(layer.relu.register_forward_hook(
                    lambda _m, _args, out, stage=stage: save(stage + ".spectral", out)))
                handles.append(layer.register_forward_hook(
                    lambda _m, _args, out, stage=stage: save(stage + ".irfft", out)))
            elif layer_name == "model.23":
                handles.append(layer.register_forward_hook(
                    lambda _m, _args, out, stage=stage: save(stage, out)))
        # The upstream constructor inserts the SAME up_activation instance
        # three times. named_modules deduplicates it; one hook labels its three
        # actual invocations in execution order instead of registering copies.
        up_stages = iter(("generator.model.26", "generator.model.29", "generator.model.32"))
        handles.append(model.model[26].register_forward_hook(
            lambda _m, _args, out: save(next(up_stages), out)))
        try:
            observed = model(conditioned)
        finally:
            for handle in handles:
                handle.remove()
        if not torch.equal(baseline.contiguous().view(torch.uint8), observed.contiguous().view(torch.uint8)):
            raise AssertionError("reference hooks changed original output")
        save("prediction", baseline)
    if set(records) != expected_stages():
        raise AssertionError(f"original stage coverage differs: missing={sorted(expected_stages()-set(records))}, extra={sorted(set(records)-expected_stages())}")
    prediction = baseline[0, :, :height, :width].permute(1, 2, 0).cpu().numpy()
    generated = np.clip(prediction * 255, 0, 255).astype(np.uint8)
    source = np.frombuffer(rgb, dtype=np.uint8).reshape(height, width, 3)
    fill = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(height, width) != 0
    composed = np.where(fill[..., None], generated, source)
    (directory / "composited.rgb8").write_bytes(composed.tobytes())
    Image.fromarray(generated).save(directory / "generated.png")
    Image.fromarray(composed).save(directory / "composited.png")
    metadata = {
        "general.architecture": "big_lama-reference", "lama.reference.schema": 1,
        "lama.source.commit": SOURCE_COMMIT, "lama.contract.sha256": contract_sha256,
        "lama.checkpoint.sha256": archive_sha256, "lama.checkpoint.container": "official-zip",
        "lama.reference.case": name, "lama.reference.width": width, "lama.reference.height": height,
        "lama.reference.stage_count": len(records), "lama.reference.precision": "float32",
        "lama.reference.hook_neutrality": "bit-exact", "lama.reference.rgb.sha256": sha256(directory / "input.rgb8"),
        "lama.reference.mask.sha256": sha256(directory / "fill-mask.u8"),
        "lama.reference.composited.sha256": sha256(directory / "composited.rgb8"),
    }
    write_f32_gguf(directory / "reference.gguf", (
        (stage, row["shape"], np.load(directory / row["file"], allow_pickle=False).tobytes())
        for stage, row in sorted(records.items())), metadata)
    return {**identity(name), "reference_sha256": sha256(directory / "reference.gguf"),
            "generator_seconds": seconds, "hook_neutrality_bit_exact": True, "stages": list(records.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), default=list(CASES))
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.threads <= 64 or len(args.cases) != len(set(args.cases)):
        parser.error("output must be new, threads in[1,64], and cases unique")
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    model, pad_original, versions = original(args.source_dir, args.checkpoint_archive)
    args.output.mkdir(parents=True)
    archive_hash = sha256(args.checkpoint_archive)
    contract_hash = sha256(Path(__file__).resolve().parents[1] / "docs/lama-source-contract.json")
    report = {"schema": 1, "model": "big_lama", "source_commit": SOURCE_COMMIT,
              "source_contract_sha256": contract_hash, "checkpoint_archive_sha256": archive_hash,
              "dependencies": versions, "platform": platform.platform(), "thresholds": THRESHOLDS,
              "native_qualified": False, "cases": []}
    for name in args.cases:
        report["cases"].append(capture(model, pad_original, name, args.output, archive_hash, contract_hash))
        (args.output / "reference.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"captured {name}: {len(report['cases'][-1]['stages'])} original stages", flush=True)


if __name__ == "__main__":
    main()
