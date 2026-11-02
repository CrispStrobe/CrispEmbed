#!/usr/bin/env python3
"""Capture original MI-GAN on two byte-pinned public photographic fixtures.

This supplemental lane leaves the eight frozen synthetic fixtures unchanged.
It imports the same hash-verified upstream module as the original capture and
retains exact RGB/mask bytes for native replay. Timings exclude stage observers.
"""

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import platform
import statistics
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    {
        "name": "formula_photo_bar",
        "path": "tests/regression/images/cc0/public_domain_formula_photo.jpg",
        "sha256": "1d2fe4b325bd2c73093901d4ccf52a670544649884fa8e7913817d099f7fe1eb",
        "source_name": "public_domain_formula_photo",
        "mask": "bar",
    },
    {
        "name": "document_photo_center",
        "path": "tests/regression/images/cc0/german_official_document.jpg",
        "sha256": "0a3937d2ac3a33bf303fd77955226af73ec74e50787708e2d327c00371997377",
        "source_name": "german_official_document",
        "mask": "center",
    },
)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def mask_bytes(size, kind):
    """Exact integer geometry: no drawing-library antialiasing or coercion."""
    if size not in (256, 512) or kind not in ("bar", "center"):
        raise ValueError("unknown fixed photo fixture geometry")
    x0, x1 = (size // 8, size * 7 // 8) if kind == "bar" else (size * 3 // 8, size * 5 // 8)
    y0, y1 = (size * 3 // 4, size * 7 // 8) if kind == "bar" else (size * 3 // 8, size * 5 // 8)
    return bytes(255 if x0 <= x < x1 and y0 <= y < y1 else 0
                 for y in range(size) for x in range(size))


def prepare(size, output):
    from PIL import Image, __version__ as pillow_version

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    provenance_file = ROOT / "tests/regression/cc0_sources.json"
    provenance = {row["name"]: row for row in json.loads(provenance_file.read_text())}
    report = {"schema": "crispembed-migan-photo-fixtures-v1", "size": size,
              "pillow": pillow_version, "provenance_manifest_sha256": digest(provenance_file),
              "transform": "RGB decode, centered integer square crop; no resize or filtering",
              "cases": []}
    for source in SOURCES:
        path = ROOT / source["path"]
        if digest(path) != source["sha256"]:
            raise ValueError("public fixture changed: " + source["path"])
        origin = provenance[source["source_name"]]
        if origin["license"] != "Public-Domain":
            raise ValueError("public fixture licence record changed")
        directory = output / source["name"]
        directory.mkdir()
        with Image.open(path) as original:
            width, height = original.size
            if min(width, height) < size:
                raise ValueError("photo is smaller than the fixed crop")
            left, top = (width - size) // 2, (height - size) // 2
            box = (left, top, left + size, top + size)
            crop = original.convert("RGB").crop(box)
            rgb = crop.tobytes()
            crop.save(directory / "input.png")
        mask = mask_bytes(size, source["mask"])
        Image.frombytes("L", (size, size), mask).save(directory / "fill-mask.png")
        (directory / "input.rgb8").write_bytes(rgb)
        (directory / "fill-mask.u8").write_bytes(mask)
        report["cases"].append({"name": source["name"], "source": source,
                                "source_page": origin["source_page"], "license": origin["license"],
                                "source_geometry": [width, height], "crop_xyxy": list(box),
                                "rgb_sha256": digest(directory / "input.rgb8"),
                                "mask_sha256": digest(directory / "fill-mask.u8"),
                                "fill_pixels": mask.count(255), "known_pixels": mask.count(0)})
    write_json(output / "fixtures.json", report)
    return report


def load_fixtures(path):
    from PIL import Image

    path = Path(path)
    report = json.loads(path.read_text())
    if (report.get("schema") != "crispembed-migan-photo-fixtures-v1"
            or report.get("size") not in (256, 512)
            or [case.get("name") for case in report.get("cases", [])] != [row["name"] for row in SOURCES]):
        raise ValueError("unexpected photo fixture manifest")
    size = report["size"]
    provenance_file = ROOT / "tests/regression/cc0_sources.json"
    if report.get("provenance_manifest_sha256") != digest(provenance_file):
        raise ValueError("photo provenance manifest changed")
    provenance = {row["name"]: row for row in json.loads(provenance_file.read_text())}
    for source, case in zip(SOURCES, report["cases"]):
        if case.get("source") != source:
            raise ValueError("photo source identity differs")
        origin = provenance[source["source_name"]]
        if case.get("source_page") != origin["source_page"] or case.get("license") != origin["license"]:
            raise ValueError("photo provenance fields differ")
        directory = path.parent / case["name"]
        for filename, field, count in (("input.rgb8", "rgb_sha256", size * size * 3),
                                       ("fill-mask.u8", "mask_sha256", size * size)):
            file = directory / filename
            if file.stat().st_size != count or digest(file) != case[field]:
                raise ValueError("photo fixture payload changed: " + case["name"])
        if (directory / "fill-mask.u8").read_bytes() != mask_bytes(size, source["mask"]):
            raise ValueError("photo mask differs from its declared geometry")
        original_path = ROOT / source["path"]
        if digest(original_path) != source["sha256"]:
            raise ValueError("original public photo bytes changed")
        with Image.open(original_path) as original:
            width, height = original.size
            left, top = (width - size) // 2, (height - size) // 2
            box = (left, top, left + size, top + size)
            rgb = original.convert("RGB").crop(box).tobytes()
        mask = mask_bytes(size, source["mask"])
        if (case.get("source_geometry") != [width, height] or case.get("crop_xyxy") != list(box)
                or hashlib.sha256(rgb).hexdigest() != case["rgb_sha256"]
                or case.get("fill_pixels") != mask.count(255) or case.get("known_pixels") != mask.count(0)):
            raise ValueError("photo crop differs from its pinned original")
    return report


def capture(fixtures_path, source_dir, checkpoint, checkpoint_sha256, output, threads, warm_runs, stages):
    import numpy as np
    from PIL import Image, __version__ as pillow_version
    import torch

    if not 1 <= threads <= 64 or not 1 <= warm_runs <= 30:
        raise ValueError("threads must be 1..64 and warm runs 1..30")
    fixtures_path = Path(fixtures_path)
    fixtures = load_fixtures(fixtures_path)
    size = fixtures["size"]
    module = importlib.import_module("dump_migan_reference" if size == 256 else "dump_migan512_reference")
    helper = importlib.import_module("migan_reference_contract" if size == 256 else "migan512_reference_contract")
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    model, _ = module.load_original(source_dir, checkpoint, checkpoint_sha256)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "crispembed-migan-photo-original-v1", "size": size,
              "source_commit": helper.load_contract()["original_source"]["commit"],
              "checkpoint_sha256": checkpoint_sha256, "fixtures_sha256": digest(fixtures_path),
              "dumper_sha256": digest(__file__), "source_dumper_sha256": digest(module.__file__),
              "versions": {"python": platform.python_version(), "torch": torch.__version__,
                           "numpy": np.__version__, "pillow": pillow_version},
              "backend": {"device": "cpu", "threads": torch.get_num_threads(),
                          "interop_threads": torch.get_num_interop_threads(),
                          "mkldnn_enabled": torch.backends.mkldnn.enabled,
                          "torch_config": torch.__config__.show()},
              "frozen_synthetic_contract_changed": False, "native_qualified": False,
              "representative_quality_qualified": False, "complete": False, "cases": []}
    write_json(output / "report.json", report)
    for case_index, case in enumerate(fixtures["cases"]):
        directory = output / case["name"]
        directory.mkdir()
        source = fixtures_path.parent / case["name"]
        rgb = (source / "input.rgb8").read_bytes()
        mask = (source / "fill-mask.u8").read_bytes()
        (directory / "input.rgb8").write_bytes(rgb)
        (directory / "fill-mask.u8").write_bytes(mask)
        Image.frombytes("RGB", (size, size), rgb).save(directory / "input.png")
        Image.frombytes("L", (size, size), mask).save(directory / "fill-mask.png")
        x, known = module.preprocess(rgb, mask)
        rows = []

        def save(stage, value):
            if isinstance(value, torch.Tensor):
                array = value.detach().cpu().numpy().astype("<f4", copy=True)
                if not np.isfinite(array).all():
                    raise ValueError("nonfinite original photo stage: " + stage)
                file = directory / (stage + ".npy")
                np.save(file, array, allow_pickle=False)
                rows.append({"stage": stage, "file": file.name, "shape": list(array.shape),
                             "sha256": digest(file), "min": float(array.min()), "max": float(array.max())})
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
            first = model(x)
            cold_ms = (time.perf_counter() - start) * 1000
            baseline = first.clone()
            if tuple(baseline.shape) != (1, 3, size, size) or baseline.dtype != torch.float32:
                raise ValueError("original photo output shape/type differs")
            samples = []
            for _ in range(warm_runs):
                start = time.perf_counter()
                repeat = model(x)
                samples.append((time.perf_counter() - start) * 1000)
                if not torch.equal(baseline, repeat):
                    raise ValueError("original photo warm output is not bit-exact")
            observed = stages and case_index == 0
            handles = []
            if observed:
                for stage, layer in model.named_modules():
                    if stage and layer.__class__.__name__ in (
                            "Conv2d", "SeparableConv2d", "EncoderBlock", "SynthesisBlock",
                            "SynthesisBlockFirst", "Downsample2d", "Upsample2d"):
                        if layer.__class__.__name__ != "Conv2d":
                            handles.append(layer.register_forward_pre_hook(
                                lambda mod, args, stage=stage: save(stage + ".input", args)))
                        handles.append(layer.register_forward_hook(
                            lambda mod, args, out, stage=stage: save(stage + ".output", out)))
                try:
                    hooked = model(x)
                finally:
                    for handle in handles:
                        handle.remove()
                if not torch.equal(baseline, hooked):
                    raise ValueError("photo observers changed original output")
            save("generator.output", baseline)
        generated, composed = module.decode(baseline, rgb, mask)
        Image.fromarray(generated).save(directory / "generated.png")
        Image.fromarray(composed).save(directory / "composited.png")
        (directory / "composited.rgb8").write_bytes(composed.tobytes())
        helper.write_f32_gguf(directory / "reference.gguf",
                              ((row["stage"], row["shape"],
                                np.load(directory / row["file"], allow_pickle=False).tobytes()) for row in rows),
                              {"general.architecture": f"migan{size}-photo-reference", "migan.image_size": size})
        report["cases"].append({**case, "stages": rows, "hook_neutrality_bit_exact": observed,
                                "repeat_bit_exact": True, "known_pixels_bit_exact": True,
                                "timing_ms": {"first": cold_ms, "warm": samples,
                                              "warm_median": statistics.median(samples),
                                              "warm_min": min(samples), "warm_max": max(samples)},
                                "reference_gguf_sha256": digest(directory / "reference.gguf"),
                                "composited_rgb8_sha256": digest(directory / "composited.rgb8")})
        write_json(output / "report.json", report)
        print(json.dumps({"case": case["name"], "timing_ms": report["cases"][-1]["timing_ms"]}), flush=True)
    report["complete"] = True
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    prep = phases.add_parser("prepare", help="Decode only two pinned images; never import torch")
    prep.add_argument("--size", type=int, choices=(256, 512), required=True)
    prep.add_argument("--output", type=Path, required=True)
    run = phases.add_parser("capture", help="Run the verified original generator and retain references")
    run.add_argument("--fixtures", type=Path, required=True)
    run.add_argument("--source-dir", type=Path, required=True)
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--checkpoint-sha256", required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--threads", type=int, default=4)
    run.add_argument("--warm-runs", type=int, default=5)
    run.add_argument("--stages", action="store_true", help="Retain all original stages on the first photo")
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.size, args.output)
    else:
        capture(args.fixtures, args.source_dir, args.checkpoint, args.checkpoint_sha256,
                args.output, args.threads, args.warm_runs, args.stages)


if __name__ == "__main__":
    main()
