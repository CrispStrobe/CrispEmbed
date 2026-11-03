#!/usr/bin/env python3
"""Original-processor controls for the bounded Nemotron image frontend.

Hosted capture imports only the hash-pinned image processor, never a model or
weights. Native comparison consumes the exact resized uint8 pixels observed at
the original processor boundary. This establishes no native resize, learned
position, neural-model, OCR, benchmark or publication qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

REPOSITORY = "nvidia/Nemotron-3.5-Parse"
REVISION = "4035ffb1d8f8bafcdb867dfaf6f7514f101464f9"
GATE = "CRISPEMBED_NEMOTRON_PARSE_GEOMETRY"
SOURCES = {
    "hf_nemotron_parse_processor.py": "983fa8aeb0d872da01296035b30d60c84fd857879a37c939c5c7203bad3c8b81",
    "preprocessor_config.json": "16d54d73fc265b31f3771caf39770dc85213cae7594823ba7a6ac3f085315ada",
    "LICENSE": "3375384ee1050a54d69c98c74d6499841166cd7db823ed90358f430df8c39856",
}
PACKAGES = {"torch": "2.11.0", "torchvision": "0.26.0", "transformers": "5.14.1",
            "numpy": "2.2.6", "Pillow": "12.3.0"}
# height, width, channels BEFORE resize. Freeze these before observing results.
CASES = {
    "portrait_rgb": (321, 187, 3),
    "landscape_rgb": (127, 451, 3),
    "nonmultiple_rgb": (17, 65, 3),
    "narrow_rgb": (65, 1, 3),
    "narrow_gray": (1, 65, 1),
    "gray_nonmultiple": (31, 77, 1),
    "single_pixel_gray": (1, 1, 1),
    "already_aligned_rgb": (32, 128, 3),
    "downscale_portrait_rgb": (2051, 833, 3),
    "downscale_landscape_gray": (513, 1901, 1),
    "downscale_both_rgb": (2301, 2191, 3),
}
NEGATIVE_CASES = {"zero_width_after_resize": (2050, 1), "zero_height_after_resize": (1, 1665)}
MAX_ABS = 1e-6
MIN_COSINE = .999999999


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def load_json(path):
    def invalid_constant(value):
        raise ValueError("nonfinite JSON constant: " + value)
    def finite_float(value):
        result = float(value)
        require(math.isfinite(result), "nonfinite JSON number")
        return result
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), parse_constant=invalid_constant,
                      parse_float=finite_float, object_pairs_hook=unique_pairs)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def empty_directory(path):
    path = Path(path)
    require(not path.exists() or (path.is_dir() and not any(path.iterdir())), "output directory must be empty")
    path.mkdir(parents=True, exist_ok=True)
    return path


def verified_sources(directory):
    directory = Path(directory)
    for name, expected in SOURCES.items():
        require(digest(directory / name) == expected, "pinned processor source drift: " + name)
    return dict(SOURCES)


def package_versions():
    return {name: importlib.metadata.version(name) for name in PACKAGES}


def validate_packages(packages):
    require(isinstance(packages, dict) and set(packages) == set(PACKAGES), "reference package inventory is incomplete")
    for name, version in PACKAGES.items():
        require(isinstance(packages[name], str) and packages[name].split("+", 1)[0] == version,
                "reference package version differs: " + name)


def pattern(height, width, channels):
    import numpy as np
    y = np.arange(height, dtype=np.uint32)[:, None]
    x = np.arange(width, dtype=np.uint32)[None, :]
    gray = ((31 * y + 7 * x + (y * x) % 251) % 256).astype(np.uint8)
    if channels == 1:
        return gray
    return np.stack([((gray.astype(np.uint32) + c * 53) % 256).astype(np.uint8) for c in range(3)], axis=-1)


def capture(source_directory, out):
    source_hashes = verified_sources(source_directory)
    versions = package_versions()
    validate_packages(versions)
    out = empty_directory(out)
    import numpy as np
    import torch
    from PIL import Image
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    source = Path(source_directory) / "hf_nemotron_parse_processor.py"
    spec = importlib.util.spec_from_file_location("crispembed_original_nemotron_processor", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    config = load_json(Path(source_directory) / "preprocessor_config.json")
    # Instantiate directly. The upstream from_dict fallback must not silently
    # replace an invalid reference config with its defaults.
    processor = module.NemotronParseImageProcessor(**config)
    require(processor.navit_packing and processor.do_normalize and
            processor.navit_padding_mode == "top_left_white" and
            processor.navit_coordinate_frame == "content_normalized" and
            processor.final_size == (2048, 1664), "original processor config did not take effect")
    rows = []
    for case_id, (height, width, channels) in CASES.items():
        original_pixels = pattern(height, width, channels)
        image = Image.fromarray(original_pixels)
        resized_observations = []
        resize = processor._resize_with_aspect_ratio
        def observed_resize(pixels):
            result = resize(pixels)
            resized_observations.append(result.copy())
            return result
        processor._resize_with_aspect_ratio = observed_resize
        try:
            with torch.inference_mode():
                reference = processor.preprocess(image, return_tensors="np")
        finally:
            processor._resize_with_aspect_ratio = resize
        require(len(resized_observations) == 1, "expected exactly one original resize boundary")
        resized_rgb = resized_observations[0]
        content = reference["content_sizes"].tolist()
        canvas = reference["image_sizes"].tolist()
        require(content == [list(resized_rgb.shape[:2])] and len(canvas) == 1,
                "original content metadata differs from observed pixels")
        inputs = resized_rgb
        if channels == 1:
            require(np.array_equal(inputs[:, :, 0], inputs[:, :, 1]) and
                    np.array_equal(inputs[:, :, 1], inputs[:, :, 2]), "grayscale conversion was not channel replication")
            inputs = inputs[:, :, 0]
        inputs = np.ascontiguousarray(inputs, dtype=np.uint8)
        patches = np.ascontiguousarray(reference["pixel_values"], dtype="<f4")
        require(patches.ndim == 2 and patches.shape[1] == 768 and bool(np.isfinite(patches).all()),
                "unexpected original patch array")
        input_path, patch_path = out / f"{case_id}.input.u8", out / f"{case_id}.reference.f32"
        inputs.tofile(input_path)
        patches.tofile(patch_path)
        rows.append({"id": case_id, "original_shape": [height, width, channels],
                     "original_pixels_sha256": hashlib.sha256(original_pixels.tobytes()).hexdigest(),
                     "content_sizes": content[0], "image_sizes": canvas[0],
                     "patch_shape": list(patches.shape), "input_bytes": inputs.nbytes,
                     "input_sha256": digest(input_path), "reference_bytes": patches.nbytes,
                     "reference_sha256": digest(patch_path)})
    negatives = []
    for case_id, (height, width) in NEGATIVE_CASES.items():
        try:
            processor.preprocess(Image.fromarray(pattern(height, width, 1)), return_tensors="np")
        except (ValueError, RuntimeError) as error:
            negatives.append({"id": case_id, "original_shape": [height, width],
                              "rejected": True, "error_type": type(error).__name__, "error": str(error)})
        else:
            raise ValueError("original unexpectedly accepted the frozen zero-dimension control: " + case_id)
    receipt = {"schema": "crispembed.nemotron-parse.frontend-reference.v1", "repo": REPOSITORY,
               "revision": REVISION, "source_sha256": source_hashes, "packages": versions,
               "reference": "unmodified original processor; observation copies only the resize return value",
               "case_order": list(CASES), "cases": rows, "negative_cases": negatives,
               "max_abs_budget": MAX_ABS, "min_cosine_budget": MIN_COSINE,
               "native_resize_qualified": False, "model_execution": False, "model_qualified": False}
    write_json(out / "reference.json", receipt)
    return receipt


def checked_pair(value, field):
    require(isinstance(value, list) and len(value) == 2 and
            all(type(x) is int and x > 0 for x in value), "invalid " + field)
    return value


def validate_reference(receipt):
    require(isinstance(receipt, dict) and receipt.get("schema") == "crispembed.nemotron-parse.frontend-reference.v1"
            and receipt.get("repo") == REPOSITORY and receipt.get("revision") == REVISION and
            receipt.get("source_sha256") == SOURCES, "reference source identity differs")
    require(receipt.get("native_resize_qualified") is False and receipt.get("model_execution") is False and
            receipt.get("model_qualified") is False, "reference claims an unsupported qualification")
    validate_packages(receipt.get("packages", {}))
    require(receipt.get("case_order") == list(CASES), "reference case order differs")
    rows = receipt.get("cases")
    require(isinstance(rows, list) and len(rows) == len(CASES), "reference cases are incomplete")
    for row, (case_id, shape) in zip(rows, CASES.items()):
        require(isinstance(row, dict) and row.get("id") == case_id and row.get("original_shape") == list(shape) and
                all(type(x) is int for x in row["original_shape"]),
                "reference case identity differs")
        ch, cw = checked_pair(row.get("content_sizes"), "content sizes")
        ph, pw = checked_pair(row.get("image_sizes"), "image sizes")
        require(ch <= 2048 and cw <= 1664 and ch <= shape[0] and cw <= shape[1] and
                ph == (ch + 15) // 16 * 16 and pw == (cw + 63) // 64 * 64,
                "invalid content/padded geometry")
        require(row.get("patch_shape") == [ph // 16 * (pw // 16), 768], "invalid patch shape")
        require(type(row.get("input_bytes")) is int and row["input_bytes"] == ch * cw * shape[2] and
                type(row.get("reference_bytes")) is int and row["reference_bytes"] == ph * pw * 3 * 4,
                "invalid reference byte count")
        for key in ("input_sha256", "reference_sha256", "original_pixels_sha256"):
            value = row.get(key)
            require(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
                    "invalid reference hash")
    negatives = receipt.get("negative_cases")
    require(isinstance(negatives, list) and len(negatives) == len(NEGATIVE_CASES), "negative controls are incomplete")
    for row, (case_id, shape) in zip(negatives, NEGATIVE_CASES.items()):
        require(isinstance(row, dict) and row.get("id") == case_id and row.get("original_shape") == list(shape)
                and row.get("rejected") is True, "negative control identity/outcome differs")
    require(receipt.get("max_abs_budget") == MAX_ABS and receipt.get("min_cosine_budget") == MIN_COSINE,
            "reference budgets differ from the frozen protocol")


def array_metrics(reference, native):
    import numpy as np
    require(reference.ndim == native.ndim == 1 and reference.size == native.size and reference.size > 0,
            "packed array sizes differ or are empty")
    require(bool(np.isfinite(reference).all() and np.isfinite(native).all()), "nonfinite packed values")
    ref, candidate = reference.astype(np.float64), native.astype(np.float64)
    difference = np.abs(ref - candidate)
    ref_norm, native_norm = float(np.linalg.norm(ref)), float(np.linalg.norm(candidate))
    require(ref_norm > 0 and native_norm > 0, "zero packed-array norm")
    cosine = float(ref @ candidate / (ref_norm * native_norm))
    maximum = float(difference.max())
    require(math.isfinite(cosine) and math.isfinite(maximum), "nonfinite comparison metrics")
    changed = np.flatnonzero(reference != native)
    first = int(changed[0]) if changed.size else None
    return {"elements": int(ref.size), "finite": True, "max_abs": maximum,
            "mean_abs": float(difference.mean()), "rms_error": float(np.sqrt(np.mean(difference ** 2))),
            "reference_max_abs": float(np.max(np.abs(ref))), "reference_norm": ref_norm,
            "native_norm": native_norm, "cosine": cosine, "changed_values": int(changed.size),
            "first_changed_index": first,
            "first_changed_reference": float(reference[first]) if first is not None else None,
            "first_changed_native": float(native[first]) if first is not None else None,
            "packing_match": maximum <= MAX_ABS and cosine >= MIN_COSINE}


def compare(native, reference_directory, out):
    reference_directory = Path(reference_directory)
    reference_path = reference_directory / "reference.json"
    receipt = load_json(reference_path)
    validate_reference(receipt)
    out = empty_directory(out)
    native = Path(native).resolve()
    env = {**os.environ, GATE: "1"}
    def run(arguments, label, environment=env):
        result = subprocess.run([str(native), *map(str, arguments)], env=environment,
                                capture_output=True, text=True)
        (out / f"{label}.stdout.json").write_text(result.stdout)
        (out / f"{label}.stderr.log").write_text(result.stderr)
        return result
    # Require a real successful self-control process, not a truthy string/status.
    check = run(["self"], "self")
    require(check.returncode == 0 and json.loads(check.stdout).get("self_controls_pass") is True,
            "native self-controls failed")
    controls = []
    for index, value in enumerate((None, "", "0")):
        off = dict(env)
        if value is None:
            off.pop(GATE, None)
        else:
            off[GATE] = value
        result = run(["plan", 17, 65], f"gate-off-{index}", off)
        passed = result.returncode != 0 and json.loads(result.stdout).get("ok") is False
        controls.append({"kind": "gate_off", "value": value, "pass": passed})
    for case_id, (height, width) in NEGATIVE_CASES.items():
        result = run(["plan", height, width], case_id)
        controls.append({"kind": case_id, "pass": result.returncode != 0 and json.loads(result.stdout).get("ok") is False})
    for index, args in enumerate((["plan", 0, 1], ["plan", -1, 10], ["plan", "1garbage", 1],
                                   ["plan", 1, 2147483648])):
        result = run(args, f"invalid-plan-{index}")
        controls.append({"kind": "invalid_native_dimensions", "pass": result.returncode != 0 and json.loads(result.stdout).get("ok") is False})
    require(all(c["pass"] for c in controls), "native negative controls failed")
    import numpy as np
    rows = []
    for row in receipt["cases"]:
        case_id = row["id"]
        input_path = reference_directory / f"{case_id}.input.u8"
        gold_path = reference_directory / f"{case_id}.reference.f32"
        require(input_path.stat().st_size == row["input_bytes"] and digest(input_path) == row["input_sha256"],
                "original resized input bytes changed: " + case_id)
        require(gold_path.stat().st_size == row["reference_bytes"] and digest(gold_path) == row["reference_sha256"],
                "original packed reference bytes changed: " + case_id)
        output_path = out / f"{case_id}.native.f32"
        height, width, channels = row["original_shape"]
        result = run(["pack", height, width, channels, input_path, output_path], case_id)
        require(result.returncode == 0, "native pack failed: " + case_id)
        meta = json.loads(result.stdout)
        ph, pw = row["image_sizes"]
        geometry_match = (meta.get("ok") is True and meta.get("content_sizes") == row["content_sizes"] and
                          meta.get("image_sizes") == row["image_sizes"] and meta.get("patch_grid") == [ph // 16, pw // 16] and
                          meta.get("patch_count") == row["patch_shape"][0] and meta.get("elements") == row["reference_bytes"] // 4 and
                          meta.get("resized") is (row["content_sizes"] != [height, width]))
        require(output_path.stat().st_size == row["reference_bytes"], "native packed byte count differs")
        metrics = array_metrics(np.fromfile(gold_path, dtype="<f4"), np.fromfile(output_path, dtype="<f4"))
        rows.append({"id": case_id, "geometry_match": geometry_match, **metrics,
                     "input_sha256": row["input_sha256"], "reference_sha256": row["reference_sha256"],
                     "native_sha256": digest(output_path), "native_metadata": meta})
        write_json(out / "partial.json", {"cases": rows, "complete": False, "model_qualified": False})
    complete = len(rows) == len(CASES) and [r["id"] for r in rows] == list(CASES)
    report = {"schema": "crispembed.nemotron-parse.frontend-comparison.v1", "repo": REPOSITORY,
              "revision": REVISION, "reference_sha256": digest(reference_path), "source_sha256": SOURCES,
              "native_executable_sha256": digest(native), "reference_packages": receipt["packages"],
              "scope": "original size decision and normalized packing of ORIGINAL-resized RGB/grayscale uint8 pixels",
              "max_abs_budget": MAX_ABS, "min_cosine_budget": MIN_COSINE, "negative_controls": controls,
              "cases": rows, "complete": complete,
              "frontend_parity_pass": complete and all(r["geometry_match"] and r["packing_match"] for r in rows),
              "native_resize_qualified": False, "position_interpolation_qualified": False,
              "model_execution": False, "model_qualified": False, "publication_qualified": False}
    write_json(out / "comparison.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    ref = commands.add_parser("capture", help="hosted original image processor only; no model or tokenizer")
    ref.add_argument("--source-directory", type=Path, required=True)
    ref.add_argument("--out", type=Path, required=True)
    candidate = commands.add_parser("compare", help="compare a hosted native frontend probe")
    candidate.add_argument("--native", type=Path, required=True)
    candidate.add_argument("--reference-directory", type=Path, required=True)
    candidate.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "capture":
            result = capture(args.source_directory, args.out)
            print(json.dumps({"captured_cases": len(result["cases"]), "model_execution": False}))
            return 0
        result = compare(args.native, args.reference_directory, args.out)
        print(json.dumps({"frontend_parity_pass": result["frontend_parity_pass"], "cases": len(result["cases"]),
                          "native_resize_qualified": False, "model_qualified": False}))
        return 0 if result["frontend_parity_pass"] else 1
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as error:
        print(json.dumps({"error": str(error), "frontend_parity_pass": False, "model_qualified": False}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
