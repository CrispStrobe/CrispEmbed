#!/usr/bin/env python3
"""Strict original/native Big-LaMa trace comparison without framework imports.

Budgets are declared before any model run. Every required stage is compared in
the original execution order. Metrics include magnitude, absolute/relative error
and cosine; a component cosine alone never establishes decoded-image parity.
"""

import argparse
import array
import json
import math
from pathlib import Path
import sys

from lama_reference_contract import (CASES, PROFILE, SOURCE_COMMIT, SOURCE_FILES, expected_stage_names,
                                     expected_state_shapes, fixture, fixture_identity, sha256,
                                     validate_reference_torch_version, validate_sha256)

MAX_ABSOLUTE_ERROR = 0.002
MAX_RELATIVE_L2_ERROR = 0.0002
MIN_COSINE = 0.99999
MAX_RGB8_ERROR = 1
MAX_STAGE_BYTES = 512 * 1024 * 1024
NATIVE_CONTROLS = ("input_exact", "observer_matches_unobserved", "repeated_graph_exact",
                   "fresh_context_exact", "process_matches_raw", "known_pixels_exact")


def metrics(original, native):
    if not original or len(original) != len(native):
        raise ValueError("stage vector lengths differ or are empty")
    if any(not math.isfinite(x) for x in original) or any(not math.isfinite(x) for x in native):
        raise ValueError("nonfinite stage vector")
    errors = [abs(a - b) for a, b in zip(original, native)]
    original_squared = math.fsum(x * x for x in original)
    native_squared = math.fsum(x * x for x in native)
    error_squared = math.fsum(x * x for x in errors)
    dot = math.fsum(a * b for a, b in zip(original, native))
    if original_squared and native_squared:
        cosine = max(-1.0, min(1.0, dot / math.sqrt(original_squared * native_squared)))
    else:
        cosine = 1.0 if original_squared == native_squared else 0.0
    relative = math.sqrt(error_squared / original_squared) if original_squared else (0.0 if error_squared == 0 else None)
    passed = (max(errors) <= MAX_ABSOLUTE_ERROR and
              (relative is not None and relative <= MAX_RELATIVE_L2_ERROR) and cosine >= MIN_COSINE)
    return {"count": len(original), "max_abs": max(errors), "mean_abs": math.fsum(errors) / len(errors),
            "rmse": math.sqrt(error_squared / len(errors)), "relative_l2": relative, "cosine": cosine,
            "original_max_abs": max(abs(x) for x in original), "native_max_abs": max(abs(x) for x in native),
            "original_l2": math.sqrt(original_squared), "native_l2": math.sqrt(native_squared), "passed": passed}


def local_file(directory, filename):
    if not isinstance(filename, str) or Path(filename).name != filename or filename in ("", ".", ".."):
        raise ValueError("stage file must be a plain local filename")
    path = Path(directory) / filename
    if path.is_symlink() or path.resolve().parent != Path(directory).resolve():
        raise ValueError("stage file must remain inside its case directory")
    return path


def read_stage(directory, row):
    shape = row.get("shape")
    if (not isinstance(shape, list) or len(shape) != 4
            or any(type(n) is not int or n <= 0 for n in shape)
            or math.prod(shape) * 4 > MAX_STAGE_BYTES):
        raise ValueError("invalid or oversized stage shape")
    path = local_file(directory, row.get("file"))
    if path.stat().st_size != math.prod(shape) * 4 or sha256(path) != row.get("sha256"):
        raise ValueError(f"stage byte count or SHA-256 differs: {row.get('stage')}")
    values = array.array("f")
    values.frombytes(path.read_bytes())
    if sys.byteorder != "little":
        values.byteswap()
    return values


def read_manifest(directory, schema):
    path = Path(directory) / "manifest.json"
    if path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("case manifest exceeds 4 MiB")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != schema:
        raise ValueError("case manifest schema differs")
    rows = manifest.get("records")
    if not isinstance(rows, list) or len(rows) > 2000:
        raise ValueError("invalid stage record inventory")
    by_name = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("stage"), str) or row["stage"] in by_name:
            raise ValueError("invalid or duplicate stage name")
        by_name[row["stage"]] = row
    missing = set(expected_stage_names()) - set(by_name)
    if missing:
        raise ValueError("required stage records missing: " + ", ".join(sorted(missing)))
    if schema == "crispembed.lama.reference.v1" and [row["stage"] for row in rows] != expected_stage_names():
        raise ValueError("original trace inventory/order differs")
    return manifest, by_name


def compare_case(name, original_dir, native_dir, model_sha256):
    original, original_rows = read_manifest(original_dir, "crispembed.lama.reference.v1")
    native, native_rows = read_manifest(native_dir, "crispembed.lama.native.v1")
    identity = fixture_identity(name)
    for key, value in identity.items():
        if original.get(key) != value:
            raise ValueError(f"original fixture identity differs: {key}")
    for key in ("width", "height", "padded_width", "padded_height"):
        if native.get(key) != identity[key]:
            raise ValueError(f"native dimensions differ: {key}")
    if original.get("hook_neutrality_bit_exact") is not True or original.get("known_pixels_exact") is not True:
        raise ValueError("original neutrality/composition controls failed")
    if native.get("model_sha256") != model_sha256:
        raise ValueError("native model SHA-256 differs")
    controls = native.get("controls", {})
    if any(controls.get(key) is not True for key in NATIVE_CONTROLS):
        raise ValueError("native observation/reuse/composition controls failed or missing")
    compared = []
    first_failure = None
    for stage in expected_stage_names():
        reference_row, native_row = original_rows[stage], native_rows[stage]
        if reference_row["shape"] != native_row["shape"]:
            raise ValueError(f"native stage shape differs: {stage}")
        reference_values, native_values = read_stage(original_dir, reference_row), read_stage(native_dir, native_row)
        row = {"stage": stage, "shape": reference_row["shape"], **metrics(reference_values, native_values)}
        if stage == "generator.input" and reference_values != native_values:
            row["passed"] = False
            row["input_bit_exact"] = False
        if not row["passed"] and first_failure is None:
            first_failure = row
        compared.append(row)

    width, height, rgb, mask = fixture(name)
    expected_bytes = width * height * 3
    reference_path, native_path = local_file(original_dir, "output.rgb"), local_file(native_dir, "output.rgb")
    if reference_path.stat().st_size != expected_bytes or native_path.stat().st_size != expected_bytes:
        raise ValueError("decoded RGB byte count differs")
    if sha256(reference_path) != original.get("output_sha256"):
        raise ValueError("original decoded RGB identity differs")
    if sha256(native_path) != native.get("output_sha256"):
        raise ValueError("native decoded RGB identity differs")
    reference_rgb, native_rgb = reference_path.read_bytes(), native_path.read_bytes()
    known_exact = all(native_rgb[3 * i:3 * i + 3] == rgb[3 * i:3 * i + 3] for i, m in enumerate(mask) if not m)
    reference_known_exact = all(reference_rgb[3 * i:3 * i + 3] == rgb[3 * i:3 * i + 3] for i, m in enumerate(mask) if not m)
    errors = [abs(a - b) for a, b in zip(reference_rgb, native_rgb)]
    image = {"known_pixels_exact": known_exact, "reference_known_pixels_exact": reference_known_exact,
             "max_abs": max(errors), "mean_abs": math.fsum(errors) / len(errors),
             "different_channels": sum(x != 0 for x in errors),
             "different_pixels": sum(reference_rgb[i:i + 3] != native_rgb[i:i + 3] for i in range(0, expected_bytes, 3)),
             "original_sha256": sha256(reference_path), "native_sha256": sha256(native_path)}
    image["passed"] = known_exact and reference_known_exact and image["max_abs"] <= MAX_RGB8_ERROR
    return {"id": name, "passed": first_failure is None and image["passed"], "first_divergence": first_failure,
            "stages": compared, "image": image, "native_controls": controls,
            "graph_buffer_bytes": native.get("graph_buffer_bytes")}


def validate_original_report(original_dir, expected_model_sha256=None):
    original_dir = Path(original_dir)
    report = json.loads((original_dir / "report.json").read_text(encoding="utf-8"))
    if (report.get("schema") != "crispembed-lama-original-v1" or report.get("complete") is not True
            or report.get("profile") != PROFILE or report.get("source_commit") != SOURCE_COMMIT
            or report.get("all_converted_tensor_bytes_exact") is not True):
        raise ValueError("complete original-module capture must precede native replay")
    if expected_model_sha256 is not None and report.get("model_sha256") != expected_model_sha256:
        raise ValueError("model differs from independently captured original checkpoint")
    validate_reference_torch_version(report.get("versions", {}).get("torch"))
    for key in ("model_sha256", "checkpoint_sha256", "config_sha256"):
        validate_sha256(report.get(key), key)
    if report.get("source_files") != SOURCE_FILES:
        raise ValueError("original source identity inventory differs")
    state, expected_shapes = report.get("state"), expected_state_shapes()
    if not isinstance(state, list) or len(state) != len(expected_shapes):
        raise ValueError("original state inventory differs")
    seen = set()
    for row in state:
        name = row.get("name")
        if name in seen or name not in expected_shapes or tuple(row.get("shape", ())) != expected_shapes[name]:
            raise ValueError("original state tensor shape/name differs")
        if row.get("bytes") != math.prod(expected_shapes[name]) * 4:
            raise ValueError("original state tensor byte count differs")
        validate_sha256(row.get("sha256"), "original tensor SHA-256")
        seen.add(name)
    producer = report.get("producer", {})
    if (producer.get("contract_sha256") != sha256(Path(__file__).with_name("lama_reference_contract.py"))
            or producer.get("dumper_sha256") != sha256(Path(__file__).with_name("dump_lama_reference.py"))):
        raise ValueError("original source protocol/helper identities differ")
    expected_calls = {"kornia.geometry.transform.rotate": 0, "pytorch_lightning.seed_everything": 0}
    if report.get("import_adapters", {}).get("fail_closed_calls") != expected_calls:
        raise ValueError("original import-only adapters were invoked or not accounted for")
    cases = report.get("cases", [])
    if [row.get("id") for row in cases] != list(CASES):
        raise ValueError("original cases incomplete or reordered")
    for row in cases:
        identity = fixture_identity(row["id"])
        if any(row.get(key) != value for key, value in identity.items()):
            raise ValueError("original fixture identity differs")
        directory = original_dir / row["id"]
        if sha256(directory / "manifest.json") != row.get("manifest_sha256"):
            raise ValueError("original case manifest SHA-256 differs")
        if sha256(directory / "input.rgb") != identity["rgb_sha256"] or sha256(directory / "mask.gray") != identity["mask_sha256"]:
            raise ValueError("original raw fixture bytes differ")
        manifest, records = read_manifest(directory, "crispembed.lama.reference.v1")
        if manifest.get("hook_neutrality_bit_exact") is not True or manifest.get("known_pixels_exact") is not True:
            raise ValueError("original observation/composition controls failed")
        for stage in expected_stage_names():
            if any(not math.isfinite(value) for value in read_stage(directory, records[stage])):
                raise ValueError(f"original stage is nonfinite: {stage}")
        output = local_file(directory, "output.rgb")
        if output.stat().st_size != identity["width"] * identity["height"] * 3 or sha256(output) != manifest.get("output_sha256"):
            raise ValueError("original output identity differs")
    return report


def compare(original_dir, native_dir):
    original = validate_original_report(original_dir)
    cases = [compare_case(name, Path(original_dir) / name, Path(native_dir) / name, original["model_sha256"]) for name in CASES]
    passed = all(row["passed"] for row in cases)
    return {"schema": "crispembed-lama-native-comparison-v1", "profile": PROFILE,
            "source_commit": SOURCE_COMMIT, "checkpoint_sha256": original["checkpoint_sha256"],
            "model_sha256": original["model_sha256"], "original_report_sha256": sha256(Path(original_dir) / "report.json"),
            "budgets": {"max_abs": MAX_ABSOLUTE_ERROR, "max_relative_l2": MAX_RELATIVE_L2_ERROR,
                        "min_cosine": MIN_COSINE, "max_rgb8_abs": MAX_RGB8_ERROR},
            "cases": cases, "passed": passed, "numerical_fixture_parity": passed,
            "representative_quality_qualified": False, "video_qualified": False, "published": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.original_dir, args.native_dir)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "cases": len(report["cases"]), "report": str(args.output)}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
