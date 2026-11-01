#!/usr/bin/env python3
"""Benchmark the original Big-LaMa image pipeline on byte-pinned photo crops.

This supplemental final-image lane reuses the validated MI-GAN photo fixtures.
The frozen eight-case, 227-stage LaMa reference protocol stays unchanged.
Timings include RGB/mask preprocessing, the original generator and composition;
they exclude file I/O, model loading and stage hooks.
"""
import argparse
import json
from pathlib import Path
import platform
import statistics
import time

from dump_lama_reference import original
from dump_migan_photo_reference import digest, load_fixtures, write_json
from lama_contract import CONTRACT
from lama_reference_contract import SOURCE_COMMIT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-archive", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warm-runs", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.threads <= 64 or not 1 <= args.warm_runs <= 20:
        parser.error("output must be new, threads in[1,64], warm runs in[1,20]")
    import numpy as np
    from PIL import Image
    import torch

    fixtures = [(path, load_fixtures(path)) for path in args.fixtures]
    if len({report["size"] for _, report in fixtures}) != len(fixtures):
        parser.error("fixture resolutions must be unique")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    model, pad_original, dependencies = original(args.source_dir, args.checkpoint_archive)
    load_seconds = time.perf_counter() - started
    args.output.mkdir(parents=True)
    report = {"schema": "crispembed-lama-photo-original-v1", "source_commit": SOURCE_COMMIT,
              "checkpoint_archive_sha256": digest(args.checkpoint_archive),
              "source_contract_sha256": digest(CONTRACT),
              "dumper_sha256": digest(__file__),
              "source_dumper_sha256": digest(Path(__file__).with_name("dump_lama_reference.py")),
              "fixture_loader_sha256": digest(Path(__file__).with_name("dump_migan_photo_reference.py")),
              "dependencies": dependencies, "platform": platform.platform(), "threads": args.threads,
              "backend": {"device": "cpu", "threads": torch.get_num_threads(),
                          "interop_threads": torch.get_num_interop_threads(),
                          "mkldnn_enabled": torch.backends.mkldnn.enabled,
                          "torch_config": torch.__config__.show()},
              "model_load_seconds": load_seconds, "timing_scope": "image-preprocess+generator+RGB-composition",
              "observers": False, "native_quality_qualified": False, "cases": [], "complete": False}
    write_json(args.output / "reference.json", report)
    for fixture_path, manifest in fixtures:
        size = manifest["size"]
        for case in manifest["cases"]:
            source = fixture_path.parent / case["name"]
            rgb = (source / "input.rgb8").read_bytes()
            mask = (source / "fill-mask.u8").read_bytes()
            directory = args.output / (str(size) + "-" + case["name"])
            directory.mkdir()
            samples, baseline, prediction = [], None, None
            with torch.inference_mode():
                for iteration in range(args.warm_runs + 1):
                    start = time.perf_counter()
                    image = np.frombuffer(rgb, dtype=np.uint8).reshape(size, size, 3)
                    fill = np.frombuffer(mask, dtype=np.uint8).reshape(size, size) == 255
                    normalized = image.transpose(2, 0, 1).astype("float32") / 255
                    mask_float = fill[None].astype("float32")
                    x = torch.from_numpy(pad_original(normalized, 8))
                    m = torch.from_numpy(pad_original(mask_float, 8))
                    conditioned = torch.cat((x * (1 - m), m), dim=0).unsqueeze(0)
                    output = model(conditioned)[0, :, :size, :size].permute(1, 2, 0).cpu().numpy()
                    if not np.isfinite(output).all():
                        raise AssertionError("original prediction contains nonfinite values")
                    generated = np.clip(output * 255, 0, 255).astype(np.uint8)
                    composed = np.where(fill[..., None], generated, image).tobytes()
                    duration = (time.perf_counter() - start) * 1000
                    if baseline is None:
                        cold_ms, baseline, prediction = duration, composed, output.copy()
                    else:
                        if composed != baseline or not np.array_equal(output, prediction):
                            raise AssertionError("original repeat changed RGB or generated floats")
                        samples.append(duration)
            (directory / "composited.rgb8").write_bytes(baseline)
            np.save(directory / "prediction.npy", prediction.transpose(2, 0, 1).astype("<f4"), allow_pickle=False)
            Image.frombytes("RGB", (size, size), baseline).save(directory / "composited.png")
            original_rgb = np.frombuffer(rgb, dtype=np.uint8).reshape(size, size, 3)
            composed_array = np.frombuffer(baseline, dtype=np.uint8).reshape(size, size, 3)
            keep = np.frombuffer(mask, dtype=np.uint8).reshape(size, size) == 0
            if not np.array_equal(original_rgb[keep], composed_array[keep]):
                raise AssertionError("original compositor changed known pixels")
            row = {"case": case["name"], "size": size, "fixture_manifest_sha256": digest(fixture_path),
                   "rgb_sha256": case["rgb_sha256"], "mask_sha256": case["mask_sha256"],
                   "cold_image_ms": cold_ms, "warm_samples_ms": samples,
                   "warm_median_ms": statistics.median(samples), "repeat_float_and_rgb_bit_exact": True,
                   "known_pixel_errors": 0, "output_sha256": digest(directory / "composited.rgb8"),
                   "prediction_sha256": digest(directory / "prediction.npy"), "directory": directory.name}
            report["cases"].append(row)
            write_json(args.output / "reference.json", report)
            print(json.dumps(row, allow_nan=False), flush=True)
    report["complete"] = True
    write_json(args.output / "reference.json", report)


if __name__ == "__main__":
    main()
