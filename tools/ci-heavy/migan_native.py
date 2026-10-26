#!/usr/bin/env python3
"""Hosted build-only or strictly reviewed original-first MI-GAN native replay.

No Torch import, original recapture or weight reconversion occurs here. Replay
requires explicit original transport/report/conversion/checkpoint/model digests.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import stat
import struct
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from migan_reference_contract import (CASES, FULL_STAGE_CASES, fixture, fixture_identity, load_contract,
                                      read_f32_gguf, sha256, validate_state_shapes, write_f32_gguf)

GGML = "890278a8342c620197c90e702e1188bcab94f510"
REPOSITORY = "CrispStrobe/CrispEmbed"
MODEL = "migan-256-places2-f32-experimental.gguf"
CONVERSION = "migan-256-places2-f32-experimental.json"
# Reviewed rejected observer run: all ordinary final payloads passed original
# gates. Retention-only repair must not change any of those arithmetic bytes.
INITIAL_NATIVE_RUN = "40336246545"
INITIAL_NATIVE_REVIEW_ARTIFACT = "12660335674"
INITIAL_NATIVE_REVIEW_ZIP_SHA256 = "b1ede41be759a91135319a48dbb14c8cd09a2ce0e9dab3f6501d34fb8b03773c"
INITIAL_ORDINARY_FINAL_SHA256 = {
    "subtitle_bar": "02bfba9cc6e2f6453e04a38b8ec054b5f84bf6bbaca804a23ded1ea0207403e2",
    "edge_touch": "948b94ba0c86b63c8849ec39c3d5723f61ceeec56a282503927d2cb152293080",
    "empty": "d784f29d420be1b4ca97ba554fe26d56a1da88a0685166f1d9cf30712d3945f0",
    "full": "26b2e50e8216c450335d4643958452871c3ad4591b8c3ddfa3e1d1f79280ec70",
    "center": "003d207e11c18634ccc88a4c3510057b1429e835c86c7b3baf5cb0e7f0b6c69f",
    "one_pixel": "812d9dfca333a132e9a2fada748843193a9d897078e5977fcf31c32a31d7b47c",
    "checkerboard": "24f9131313ecc4fcc1c0048bcf3a6b0d1336b9b27fca9ff4a60d17daafb13fae",
    "thin_strip": "90c0bdadf2bd3670d7cd905579cbbbec2f6c638c07a2ad2b872aeb2ef1b8bf4f",
}
SOURCE_FILES = ("CMakeLists.txt", "src/migan.cpp", "src/migan.h", "tests/migan_test_observer.h",
                "tests/test_migan_diff.cpp", "src/crispembed_diff.h", "src/core/gguf_loader.cpp",
                "src/core/gguf_loader.h", "src/core/ggml_cpu_backend.h", "src/core/env_gate.h",
                "src/core/clean_exit.h", "tools/ci-heavy/migan_native.py", "tools/migan_reference_contract.py",
                "tools/dump_migan_reference.py", "models/convert-migan-to-gguf.py",
                "docs/migan-reference-contract.json", ".github/workflows/migan-native.yml")


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("nonfinite JSON number: " + value)

    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def valid_digest(value, length=64):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{" + str(length) + "}", value) is not None


def validate_inputs(args):
    if args.phase not in ("build-only", "replay"):
        raise ValueError("unknown MI-GAN native phase")
    if args.phase == "build-only":
        return
    for key in ("reference_run", "reference_artifact_id"):
        if not re.fullmatch(r"[1-9][0-9]*", getattr(args, key)):
            raise ValueError("reviewed positive numeric " + key + " is required")
    for key in ("reference_archive_sha256", "reference_report_sha256", "conversion_report_sha256",
                "checkpoint_sha256", "model_gguf_sha256"):
        if not valid_digest(getattr(args, key)):
            raise ValueError("reviewed lowercase SHA-256 required: " + key)
    if not valid_digest(args.reference_head, 40):
        raise ValueError("reviewed original source commit required")
    if args.reference_workflow not in (".github/workflows/migan-reference.yml", ".github/workflows/ovisocr2-metadata.yml"):
        raise ValueError("unexpected original workflow path")


def parse_log(contents, expected_stages):
    records = {name: [] for name in ("MIGAN_CONTRACT", "MIGAN_GATES", "MIGAN_DIFF", "MIGAN_IMAGE", "MIGAN_SUMMARY")}
    for line in contents.splitlines():
        if not line.startswith("MIGAN_"):
            continue
        tag, separator, payload = line.partition(" ")
        if not separator or tag not in records:
            raise ValueError("unexpected native record")
        row = strict_json(payload)
        if type(row) is not dict:
            raise ValueError("native record must be an object")
        records[tag].append(row)
    if any(len(records[tag]) != 1 for tag in records if tag != "MIGAN_DIFF"):
        raise ValueError("missing/duplicate native control or summary")
    if records["MIGAN_CONTRACT"][0] != {"passed": True}:
        raise ValueError("RGB/mask contract controls failed")
    if records["MIGAN_GATES"][0] != {"unset_rejected": True, "empty_rejected": True, "literal_zero_rejected": True}:
        raise ValueError("default/empty/literal-zero controls failed")
    stages = records["MIGAN_DIFF"]
    if [row.get("stage") for row in stages] != list(expected_stages) or len(set(expected_stages)) != len(expected_stages):
        raise ValueError("native stage order/set differs from frozen original")
    for row in stages:
        if type(row.get("count")) is not int or row["count"] <= 0:
            raise ValueError("invalid native stage size")
        for key in ("max_abs", "mae", "rms", "cos_min", "cos_global", "native_l2", "original_l2"):
            value = row.get(key)
            if type(value) not in (float, int) or not math.isfinite(value) or (key not in ("cos_min", "cos_global") and value < 0):
                raise ValueError("invalid native metric")
            if key in ("cos_min", "cos_global") and not -1.000001 <= value <= 1.000001:
                raise ValueError("cosine outside numerical range")
        if any(type(row.get(key)) is not bool for key in ("finite", "shape_match", "passed")):
            raise ValueError("invalid native stage flags")
        passed = (row["finite"] and row["shape_match"] and row["cos_min"] >= 0.99999
                  and row["cos_global"] >= 0.99999 and row["max_abs"] <= 0.002)
        if row["passed"] != passed:
            raise ValueError("native float verdict contradicts frozen gates")
    image = records["MIGAN_IMAGE"][0]
    count, known_errors = image.get("fill_channels"), image.get("known_errors")
    if type(count) is not int or count < 0 or type(known_errors) is not int or known_errors < 0:
        raise ValueError("invalid image counts")
    if count == 0:
        if any(image.get(key) is not None for key in ("max_abs", "mae", "mse", "psnr_db")):
            raise ValueError("empty-mask metrics must be null")
        image_pass = known_errors == 0
    else:
        for key in ("max_abs", "mae", "mse"):
            if type(image.get(key)) not in (int, float) or not math.isfinite(image[key]) or image[key] < 0:
                raise ValueError("invalid RGB8 metric")
        psnr = image.get("psnr_db")
        if image["mse"] == 0:
            if psnr is not None or image.get("exact_fill") is not True:
                raise ValueError("exact image must declare infinite PSNR via null/zero MSE")
            psnr = math.inf
        elif type(psnr) not in (int, float) or not math.isfinite(psnr):
            raise ValueError("invalid PSNR")
        image_pass = known_errors == 0 and image["max_abs"] <= 1 and image["mae"] <= 0.1 and psnr >= 50
    if type(image.get("passed")) is not bool or image["passed"] != image_pass:
        raise ValueError("native image verdict contradicts frozen gates")
    summary = records["MIGAN_SUMMARY"][0]
    failures = [row["stage"] for row in stages if not row["passed"]]
    if (summary.get("compared") != len(stages) or summary.get("expected") != len(stages)
            or summary.get("failed_float_stages") != len(failures)
            or summary.get("first_failure") != (failures[0] if failures else "")
            or type(summary.get("observer_neutrality_bit_exact")) is not bool
            or summary.get("image_passed") != image_pass):
        raise ValueError("native summary differs from individual records")
    if (type(summary.get("observed_graph_buffer_bytes")) is not int or summary["observed_graph_buffer_bytes"] < 0
            or type(summary.get("generator_seconds")) not in (float, int)
            or not math.isfinite(summary["generator_seconds"]) or summary["generator_seconds"] < 0):
        raise ValueError("invalid native resource diagnostic")
    passed = not failures and image_pass and summary["observer_neutrality_bit_exact"]
    if type(summary.get("passed")) is not bool or summary["passed"] != passed:
        raise ValueError("native summary verdict differs")
    return {"stages": stages, "image": image, "summary": summary}


def safe_extract(archive, destination, max_bytes=8 * 1024**3):
    destination = Path(destination)
    with zipfile.ZipFile(archive) as source:
        members = source.infolist()
        names = set()
        total = 0
        for member in members:
            path = PurePosixPath(member.filename)
            mode = member.external_attr >> 16
            if (not member.filename or "\\" in member.filename or ":" in member.filename or path.is_absolute()
                    or ".." in path.parts or member.filename in names or stat.S_ISLNK(mode)):
                raise ValueError("unsafe/duplicate artifact ZIP path")
            total += member.file_size
            if total > max_bytes:
                raise ValueError("artifact exceeds hosted extraction budget")
            names.add(member.filename)
        destination.mkdir(parents=True, exist_ok=False)
        source.extractall(destination)


def api(path):
    return strict_json(subprocess.check_output(["gh", "api", "repos/" + REPOSITORY + path], text=True))


def original_artifact(args, scratch, report):
    run = api("/actions/runs/" + args.reference_run)
    artifact = api("/actions/artifacts/" + args.reference_artifact_id)
    if (run.get("status") != "completed" or run.get("conclusion") != "success"
            or run.get("head_sha") != args.reference_head or run.get("event") != "workflow_dispatch"
            or run.get("path") != args.reference_workflow):
        raise ValueError("original run is not the reviewed successful source/workflow")
    if (str(artifact.get("id")) != args.reference_artifact_id or artifact.get("expired") is not False
            or artifact.get("name") != "migan-original-raw-" + args.reference_run
            or str(artifact.get("workflow_run", {}).get("id")) != args.reference_run
            or artifact.get("workflow_run", {}).get("head_sha") != args.reference_head
            or artifact.get("digest") != "sha256:" + args.reference_archive_sha256):
        raise ValueError("original artifact identity/service digest differs")
    archive = scratch / "original.zip"
    with archive.open("xb") as stream:
        subprocess.run(["gh", "api", "repos/" + REPOSITORY + "/actions/artifacts/" + args.reference_artifact_id + "/zip"], stdout=stream, check=True)
    if sha256(archive) != args.reference_archive_sha256:
        raise ValueError("downloaded original ZIP differs from reviewed digest")
    extracted = scratch / "original-artifact"
    safe_extract(archive, extracted)
    report["original_transport"] = {"run_id": args.reference_run, "source_commit": args.reference_head,
                                    "workflow_path": args.reference_workflow, "artifact_id": args.reference_artifact_id,
                                    "artifact_name": artifact["name"], "archive_sha256": sha256(archive),
                                    "archive_bytes": archive.stat().st_size, "expires_at": artifact["expires_at"]}
    archive.unlink()  # Task-owned duplicate ZIP only; extracted original remains.
    manifest = strict_json((extracted / "artifact-members.json").read_text())
    members = {}
    for row in manifest:
        name = row["path"]
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name or name in members:
            raise ValueError("invalid original member manifest")
        actual = extracted / name
        if actual.stat().st_size != row["bytes"] or sha256(actual) != row["sha256"]:
            raise ValueError("original artifact member bytes differ: " + name)
        members[name] = row
    actual_files = {str(path.relative_to(extracted)) for path in extracted.rglob("*") if path.is_file()}
    if actual_files != set(members) | {"artifact-members.json", "review/artifact-members.json"}:
        raise ValueError("original ZIP member coverage differs")
    if sha256(extracted / "artifact-members.json") != sha256(extracted / "review/artifact-members.json"):
        raise ValueError("original member manifest review copy differs")
    report["original_transport"]["verified_members"] = len(members)
    report["original_transport"]["member_manifest_sha256"] = sha256(extracted / "artifact-members.json")
    return extracted


def verify_original(directory, args):
    contract = load_contract()
    source = contract["original_source"]
    for name, row in source["files"].items():
        path = directory / "source" / name
        if path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise ValueError("original upstream source differs: " + name)
    original_path = directory / "original/report.json"
    conversion_path = directory / CONVERSION
    if sha256(original_path) != args.reference_report_sha256 or sha256(conversion_path) != args.conversion_report_sha256:
        raise ValueError("reviewed original/conversion report differs")
    original = strict_json(original_path.read_text())
    conversion = strict_json(conversion_path.read_text())
    identity = strict_json((directory / "checkpoint-identity.json").read_text())
    if (identity.get("schema") != "crispembed-migan-checkpoint-first-observation-v1"
            or identity.get("sha256") != args.checkpoint_sha256 or identity.get("bytes") != contract["checkpoint"]["declared_bytes"]
            or identity.get("source_commit") != source["commit"] or identity.get("github_sha") != args.reference_head
            or identity.get("github_run_id") != args.reference_run or identity.get("upstream_sha256_declared") is not False):
        raise ValueError("original checkpoint observation differs")
    checkpoint = directory / contract["checkpoint"]["filename"]
    if checkpoint.stat().st_size != identity["bytes"] or sha256(checkpoint) != args.checkpoint_sha256:
        raise ValueError("frozen checkpoint bytes differ")
    if (original.get("schema") != "crispembed-migan-original-v1" or original.get("complete") is not True
            or original.get("source_commit") != source["commit"] or original.get("checkpoint_sha256") != args.checkpoint_sha256
            or original.get("contract_sha256") != sha256(ROOT / "docs/migan-reference-contract.json")
            or original.get("producer", {}).get("github_sha") != args.reference_head
            or original.get("producer", {}).get("github_run_id") != args.reference_run
            or original.get("producer", {}).get("dumper_sha256") != sha256(ROOT / "tools/dump_migan_reference.py")
            or original.get("producer", {}).get("helper_sha256") != sha256(ROOT / "tools/migan_reference_contract.py")
            or original.get("native_qualified") is not False or original.get("representative_quality_qualified") is not False
            or original.get("rejected_controls") != ["nonbinary", "mask_size", "rgb_channels", "wrong_resolution"]):
        raise ValueError("incomplete or changed original capture identity")
    if strict_json((directory / "original/tensor-inventory.json").read_text()) != original["state"]:
        raise ValueError("original tensor inventory copy differs")
    if (conversion.get("schema") != "crispembed-migan-conversion-v1"
            or conversion.get("checkpoint_sha256") != args.checkpoint_sha256
            or conversion.get("original_report_sha256") != args.reference_report_sha256
            or conversion.get("gguf_sha256") != args.model_gguf_sha256 or conversion.get("tensor_count") != 154
            or conversion.get("all_tensor_bytes_exact") is not True
            or conversion.get("native_qualified") is not False or conversion.get("published") is not False):
        raise ValueError("incomplete original F32 conversion receipt")
    model_path = directory / MODEL
    if sha256(model_path) != args.model_gguf_sha256 or model_path.stat().st_size != conversion["gguf_bytes"]:
        raise ValueError("frozen converted model bytes differ")
    metadata, tensors = read_f32_gguf(model_path)
    if (metadata.get("general.architecture") != "migan" or metadata.get("general.license") != "MIT"
            or metadata.get("general.file_type") != 0 or metadata.get("migan.image_size") != 256
            or metadata.get("migan.source.commit") != source["commit"]
            or metadata.get("migan.input.contract") != "RGB8;binary-mask8;255=fill;fixed256"
            or metadata.get("migan.tensor.layout") != "original contiguous OIHW; GGUF dimensions reversed; scalar=[1]"
            or metadata.get("migan.checkpoint.sha256") != args.checkpoint_sha256):
        raise ValueError("frozen model metadata differs")
    inventory = {row["name"]: row for row in original["state"]}
    if len(inventory) != 154 or set(tensors) != set(inventory):
        raise ValueError("frozen model tensor set differs")
    validate_state_shapes({name: row["shape"] for name, row in inventory.items()})
    for name, (shape, data) in tensors.items():
        row = inventory[name]
        if (shape != (tuple(row["shape"]) or (1,)) or row["dtype"] != "float32"
                or hashlib.sha256(data).hexdigest() != row["sha256"]):
            raise ValueError("frozen model original tensor mapping differs: " + name)
    if [case["id"] for case in original["cases"]] != list(CASES) or contract["fixtures"] != [fixture_identity(name) for name in CASES]:
        raise ValueError("frozen case list/fixture generator differs")
    for case in original["cases"]:
        if (any(case.get(key) != value for key, value in fixture_identity(case["id"]).items())
                or case.get("known_pixels_bit_exact") is not True
                or case.get("hook_neutrality_bit_exact") is not (case["id"] in FULL_STAGE_CASES)):
            raise ValueError("frozen case contract controls differ")
    return original, conversion, metadata, tensors


def verify_case(directory, case):
    import numpy as np
    from PIL import Image

    name = case["id"]
    original = directory / "original" / name
    rgb, mask = fixture(name)
    if (Image.open(original / "input.png").convert("RGB").tobytes() != rgb
            or Image.open(original / "fill-mask.png").convert("L").tobytes() != mask):
        raise ValueError("frozen input image/mask pixel bytes differ")
    for filename, key in (("generated.png", "generated_png_sha256"), ("composited.png", "composited_png_sha256"),
                          ("reference.gguf", "reference_gguf_sha256")):
        if sha256(original / filename) != case[key]:
            raise ValueError("frozen original output/archive bytes differ")
    _, tensors = read_f32_gguf(original / "reference.gguf")
    if list(tensors) != [row["stage"] for row in case["stages"]] or len(tensors) != len(case["stages"]):
        raise ValueError("frozen reference stage order/set differs")
    for row in case["stages"]:
        if Path(row["file"]).name != row["file"] or sha256(original / row["file"]) != row["sha256"]:
            raise ValueError("frozen original stage NPY bytes differ")
        array = np.load(original / row["file"], allow_pickle=False)
        shape, data = tensors[row["stage"]]
        if (array.dtype != np.dtype("<f4") or list(array.shape) != row["shape"] or not np.isfinite(array).all()
                or shape != tuple(row["shape"]) or data != array.tobytes()):
            raise ValueError("original NPY/GGUF stage mapping differs")
    expected_image = np.load(original / "composited.npy", allow_pickle=False)
    with Image.open(original / "composited.png") as image:
        if expected_image.dtype != np.uint8 or expected_image.shape != (256, 256, 3) or expected_image.tobytes() != image.convert("RGB").tobytes():
            raise ValueError("frozen original RGB8 NPY/PNG mapping differs")
    raw = np.load(original / "generator.output.npy", allow_pickle=False)
    if raw.shape != (1, 3, 256, 256):
        raise ValueError("frozen original generator shape differs")
    decoded = (np.clip(raw[0] * np.float32(0.5) + np.float32(0.5), 0, 1) * np.float32(255)).astype(np.uint8).transpose(1, 2, 0)
    filled = np.frombuffer(mask, dtype=np.uint8).reshape(256, 256) == 255
    input_rgb = np.frombuffer(rgb, dtype=np.uint8).reshape(256, 256, 3)
    with Image.open(original / "generated.png") as image:
        if image.convert("RGB").tobytes() != decoded.tobytes():
            raise ValueError("frozen original RGB8 decoding differs from generator floats")
    if not np.array_equal(np.where(filled[:, :, None], decoded, input_rgb), expected_image):
        raise ValueError("frozen original mask composition differs from original RGB bytes")
    return rgb, mask, expected_image.tobytes()


def independent_metrics(actual, expected):
    import numpy as np

    if actual.dtype != np.dtype("<f4") or actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError("native raw stage dtype/shape/finite contract failed")
    a, b = actual.astype(np.float64), expected.astype(np.float64)
    delta = a - b
    rows_a, rows_b = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
    dot = np.sum(rows_a * rows_b, axis=1)
    na, nb = np.sum(rows_a**2, axis=1), np.sum(rows_b**2, axis=1)
    cosine = np.zeros(len(dot))
    both = (na != 0) & (nb != 0)
    cosine[both] = dot[both] / np.sqrt(na[both] * nb[both])
    cosine[(na == 0) & (nb == 0)] = 1
    native_l2, original_l2 = float(np.linalg.norm(a.ravel())), float(np.linalg.norm(b.ravel()))
    global_cos = (float(np.sum(a * b)) / (native_l2 * original_l2) if native_l2 and original_l2
                  else (1.0 if native_l2 == original_l2 == 0 else 0.0))
    result = {"max_abs": float(np.max(np.abs(delta))), "mae": float(np.mean(np.abs(delta))),
              "rms": float(np.sqrt(np.mean(delta**2))), "cos_min": float(np.min(cosine)), "cos_global": global_cos,
              "native_l2": native_l2, "original_l2": original_l2,
              "native_min": float(a.min()), "native_max": float(a.max()),
              "original_min": float(b.min()), "original_max": float(b.max()),
              "native_rms": native_l2 / math.sqrt(a.size), "original_rms": original_l2 / math.sqrt(b.size)}
    result["passed"] = result["max_abs"] <= 0.002 and result["cos_min"] >= 0.99999 and result["cos_global"] >= 0.99999
    return result


def verify_native_case(original_directory, case, native_directory, parsed, rgb, mask, original_rgb):
    import numpy as np
    from PIL import Image

    original = original_directory / "original" / case["id"]
    if {path.stem for path in native_directory.glob("*.f32")} != {row["stage"] for row in case["stages"]}:
        raise ValueError("native raw array coverage differs")
    rows = []
    for frozen, logged in zip(case["stages"], parsed["stages"]):
        path = native_directory / (frozen["stage"] + ".f32")
        if path.stat().st_size != math.prod(frozen["shape"]) * 4 or logged["count"] != math.prod(frozen["shape"]):
            raise ValueError("native raw size differs")
        actual = np.fromfile(path, dtype="<f4").reshape(frozen["shape"])
        expected = np.load(original / frozen["file"], allow_pickle=False)
        metrics = independent_metrics(actual, expected)
        for key in ("max_abs", "mae", "rms", "cos_min", "cos_global", "native_l2", "original_l2"):
            if not math.isclose(metrics[key], logged[key], abs_tol=2e-7, rel_tol=2e-6):
                raise ValueError("native log metric differs from raw independent calculation: " + frozen["stage"] + "/" + key)
        if metrics["passed"] != logged["passed"]:
            raise ValueError("native raw/log float verdict differs")
        rows.append({"stage": frozen["stage"], "shape": frozen["shape"], "bytes": path.stat().st_size,
                     "sha256": sha256(path), **metrics})
    raw = np.fromfile(native_directory / "generator.output.f32", dtype="<f4").reshape(1, 3, 256, 256)
    ordinary = native_directory / "controls/ordinary-output.f32"
    if ordinary.stat().st_size != 3 * 256 * 256 * 4:
        raise ValueError("ordinary output byte count differs")
    neutrality = ordinary.read_bytes() == (native_directory / "generator.output.f32").read_bytes()
    if neutrality != parsed["summary"]["observer_neutrality_bit_exact"]:
        raise ValueError("ordinary/observed raw bit comparison contradicts log")
    initial_ordinary_exact = sha256(ordinary) == INITIAL_ORDINARY_FINAL_SHA256[case["id"]]
    generated = (np.clip(raw[0] * np.float32(0.5) + np.float32(0.5), 0, 1) * np.float32(255)).astype(np.uint8).transpose(1, 2, 0)
    filled = np.frombuffer(mask, dtype=np.uint8).reshape(256, 256) == 255
    input_rgb = np.frombuffer(rgb, dtype=np.uint8).reshape(256, 256, 3)
    composed = np.where(filled[:, :, None], generated, input_rgb)
    for filename, image in (("generated", generated), ("composited", composed)):
        path = native_directory / (filename + ".rgb8")
        if path.read_bytes() != image.tobytes():
            raise ValueError("native RGB8 postprocess/composition differs from actual raw output")
        Image.fromarray(image).save(native_directory / (filename + ".png"))
    expected = np.frombuffer(original_rgb, dtype=np.uint8).reshape(256, 256, 3)
    known_errors = int(np.count_nonzero(composed[~filled] != input_rgb[~filled]))
    difference = composed[filled].astype(np.int16) - expected[filled].astype(np.int16)
    count = difference.size
    image = {"fill_channels": count, "known_errors": known_errors,
             "max_abs": int(np.max(np.abs(difference))) if count else None,
             "mae": float(np.mean(np.abs(difference))) if count else None,
             "mse": float(np.mean(difference.astype(np.float64)**2)) if count else None}
    image["psnr_db"] = 10 * math.log10(255**2 / image["mse"]) if count and image["mse"] else None
    image["exact_fill"] = bool(count and image["mse"] == 0)
    image["passed"] = known_errors == 0 and (not count or (image["max_abs"] <= 1 and image["mae"] <= 0.1
                                                         and (image["mse"] == 0 or image["psnr_db"] >= 50)))
    for key in ("fill_channels", "known_errors", "max_abs", "mae", "mse", "psnr_db", "passed"):
        actual, logged = image[key], parsed["image"][key]
        if (type(actual) in (float, int) and type(logged) in (float, int)):
            if not math.isclose(actual, logged, abs_tol=1e-9, rel_tol=1e-9):
                raise ValueError("native raw/log decoded image metric differs")
        elif actual != logged:
            raise ValueError("native raw/log decoded image verdict differs")
    failures = [row["stage"] for row in rows if not row["passed"]]
    return {"id": case["id"], "raw_verified": True, "stages": rows, "image": image,
            "first_failure": failures[0] if failures else None,
            "observer_neutrality_checked": case["id"] in FULL_STAGE_CASES,
            "observer_neutrality_bit_exact": neutrality,
            "ordinary_output_sha256": sha256(ordinary),
            "initial_ordinary_final_bit_exact": initial_ordinary_exact,
            "observed_graph_buffer_bytes": parsed["summary"]["observed_graph_buffer_bytes"],
            "native_generator_seconds": parsed["summary"]["generator_seconds"],
            "passed": not failures and image["passed"] and parsed["summary"]["observer_neutrality_bit_exact"] and initial_ordinary_exact}


def command(label, args, output, report, expected=(0,), env=None, timeout=2400):
    log = output / (label + ".log")
    resources = output / (label + "-resources.json")
    timed = ["/usr/bin/time", "-f", '{"max_rss_kib":%M,"wall_seconds":%e}', "-o", str(resources), *map(str, args)]
    with log.open("w") as stream:
        result = subprocess.run(timed, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, env=env, timeout=timeout)
    text = resources.read_text()
    prefix = "Command exited with non-zero status " + str(result.returncode) + "\n"
    if text.startswith(prefix):
        text = text[len(prefix):]
    measured = strict_json(text)
    report["commands"][label] = {"exit_code": result.returncode, "log_sha256": sha256(log),
                                  "resources": measured, "command": list(map(str, args))}
    if result.returncode not in expected:
        raise ValueError(label + " failed; preserved log: " + log.name)
    return result.returncode, log.read_text()


def build(scratch, output, report):
    source = {name: sha256(ROOT / name) for name in SOURCE_FILES}
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    declared = subprocess.check_output(["git", "rev-parse", "HEAD:ggml"], cwd=ROOT, text=True).strip()
    actual = subprocess.check_output(["git", "-C", str(ROOT / "ggml"), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(ROOT / "ggml"), "status", "--porcelain"], text=True)
    if subprocess.check_output(["git", "diff", "--name-only", "HEAD", "--", *SOURCE_FILES], cwd=ROOT, text=True):
        raise ValueError("native source differs from checked-out commit")
    if declared != GGML or actual != GGML or dirty:
        raise ValueError("native ggml pin/checkout differs")
    report["producer"] = {"source_commit": head, "github_sha": os.environ.get("GITHUB_SHA"),
                          "github_run_id": os.environ.get("GITHUB_RUN_ID"), "ggml_commit": actual, "source_sha256": source,
                          "python": platform.python_version(), "platform": platform.platform(),
                          "cpu": strict_json(subprocess.check_output(["lscpu", "--json"], text=True))}
    build_dir = scratch / "build"
    command("configure", ["cmake", "-S", ROOT, "-B", build_dir, "-DCMAKE_BUILD_TYPE=Release",
                           "-DGGML_NATIVE=OFF", "-DCRISPEMBED_NATIVE=OFF", "-DGGML_BLAS=OFF", "-DGGML_CUDA=OFF",
                           "-DGGML_METAL=OFF", "-DGGML_OPENMP=OFF", "-DGGML_LLAMAFILE=OFF",
                           "-DCRISPEMBED_BUILD_SHARED=OFF"], output, report)
    command("build", ["cmake", "--build", build_dir, "--target", "crispembed-migan", "test-migan-diff", "-j2"], output, report)
    binary = next(path for path in (build_dir / "test-migan-diff", build_dir / "bin/test-migan-diff") if path.is_file())
    command("contracts", [binary, "--contracts"], output, report)
    command("gate-controls", [binary, "--gate-controls"], output, report)
    command("observer-retention", [binary, "--observer-retention"], output, report)
    if strict_json((output / "observer-retention.log").read_text().split("MIGAN_RETENTION ", 1)[1].splitlines()[0]) != {"passed": True}:
        raise ValueError("view-owner regression failed")
    if strict_json((output / "contracts.log").read_text().split("MIGAN_CONTRACT ", 1)[1].splitlines()[0]) != {"passed": True}:
        raise ValueError("build helper controls differ")
    expected_gates = {"unset_rejected": True, "empty_rejected": True, "literal_zero_rejected": True}
    if strict_json((output / "gate-controls.log").read_text().split("MIGAN_GATES ", 1)[1].splitlines()[0]) != expected_gates:
        raise ValueError("build default-off controls differ")
    symbols = {}
    artifacts = output / "build-artifacts"
    artifacts.mkdir()
    shutil.copy2(binary, artifacts / binary.name)
    for target in ("crispembed-migan", "crispembed-migan-test"):
        archive = next(build_dir.rglob("lib" + target + ".a"))
        names = subprocess.check_output(["nm", "-g", "--defined-only", "-C", str(archive)], text=True)
        (output / (target + "-symbols.txt")).write_text(names)
        symbols[target] = {"archive_sha256": sha256(archive), "observer_symbol_present": "migan::test::run_observed(" in names}
        shutil.copy2(archive, artifacts / archive.name)
    if symbols["crispembed-migan"]["observer_symbol_present"] or not symbols["crispembed-migan-test"]["observer_symbol_present"]:
        raise ValueError("test observer leaked into ordinary prototype or is missing")
    report["build"] = {"passed": True, "binary_sha256": sha256(binary), "symbols": symbols,
                       "scope": "Explicit CPU/F32 prototype/helper/gate build; no weights or model inference in build-only phase"}
    return binary


def retag_scalar_f16(path, tensors, name):
    """Produce valid F16 scalar data; its 32-byte slot/next offsets stay exact."""
    shape, original = tensors[name]
    if shape != (1,) or len(original) != 4:
        raise ValueError("negative F16 control requires a one-element tensor")
    payload = bytearray(Path(path).read_bytes())
    needle = struct.pack("<Q", len(name)) + name.encode() + struct.pack("<IQ", 1, 1)
    if payload.count(needle) != 1:
        raise ValueError("cannot uniquely locate negative dtype scalar info")
    offset = payload.index(needle) + len(needle)
    if payload[offset:offset + 4] != b"\0" * 4:
        raise ValueError("negative dtype source is not F32")
    payload[offset:offset + 4] = struct.pack("<I", 1)
    data_start = len(payload) - sum((len(data) + 31) & ~31 for _, data in tensors.values())
    tensor_offset = 0
    for key, (_, data) in tensors.items():
        if key == name:
            break
        tensor_offset += (len(data) + 31) & ~31
    start = data_start + tensor_offset
    value = struct.unpack("<f", original)[0]
    payload[start:start + 32] = struct.pack("<e", value) + b"\0" * 30
    Path(path).write_bytes(payload)


def negative_models(binary, metadata, tensors, args, scratch, output, report):
    models = scratch / "negative-models"
    models.mkdir()
    controls = []
    report["negative_models"] = controls
    for label in ("wrong_architecture", "wrong_resolution", "wrong_checkpoint", "missing_noise", "extra_probe", "wrong_shape", "nonfinite", "f16_state"):
        meta, data = dict(metadata), dict(tensors)
        if label == "wrong_architecture": meta["general.architecture"] = "not-migan"
        if label == "wrong_resolution": meta["migan.image_size"] = 512
        if label == "wrong_checkpoint": meta["migan.checkpoint.sha256"] = "0" * 64
        if label == "missing_noise": del data["synthesis.b256.conv1.noise_const"]
        if label == "extra_probe": data["reference_probe"] = ((1,), b"\0" * 4)
        name = "encoder.b256.fromrgb.weight"
        if label == "wrong_shape": data[name] = ((4, 128, 1, 1), data[name][1])
        if label == "nonfinite": data[name] = (data[name][0], bytes.fromhex("0000c07f") + data[name][1][4:])
        path = models / (label + ".gguf")
        write_f32_gguf(path, [(name, shape, payload) for name, (shape, payload) in data.items()], meta)
        if label == "f16_state":
            retag_scalar_f16(path, data, "synthesis.b8.conv1.noise_strength")
        control = {"id": label, "sha256": sha256(path), "rejected": None}
        controls.append(control)
        try:
            _, log = command("negative-" + label, [binary, "--reject", path, args.checkpoint_sha256], output, report)
        except Exception:
            shutil.copy2(path, output / ("failed-negative-" + label + ".gguf"))
            raise
        if "MIGAN_REJECT {\"rejected\":true}" not in log:
            raise ValueError("malformed model accepted: " + label)
        reason = {"wrong_architecture": "MI-GAN GGUF source/checkpoint/F32/fixed-256 metadata differs",
                  "wrong_resolution": "MI-GAN GGUF source/checkpoint/F32/fixed-256 metadata differs",
                  "wrong_checkpoint": "MI-GAN GGUF source/checkpoint/F32/fixed-256 metadata differs",
                  "missing_noise": "MI-GAN requires exactly 154 original state tensors",
                  "extra_probe": "MI-GAN requires exactly 154 original state tensors",
                  "wrong_shape": "MI-GAN state shape differs: encoder.b256.fromrgb.weight",
                  "nonfinite": "MI-GAN nonfinite checkpoint state: encoder.b256.fromrgb.weight",
                  "f16_state": "MI-GAN missing or non-F32 state: synthesis.b8.conv1.noise_strength"}[label]
        if reason not in log:
            shutil.copy2(path, output / ("failed-negative-" + label + ".gguf"))
            raise ValueError("negative model rejected at an unexpected guard: " + label)
        control["rejected"] = True
        control["guard_message"] = reason
        path.unlink()  # Reproducible task-owned negative copy; original is intact.


def save(output, scratch, report):
    text = json.dumps(report, indent=2, allow_nan=False)
    for value, replacement in ((str(ROOT), "<repo>"), (str(scratch), "<scratch>"), (str(output), "<out>")):
        text = text.replace(value, replacement)
    (output / "migan-native.json").write_text(text + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("build-only", "replay"), default="build-only")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    for name in ("reference-run", "reference-head", "reference-artifact-id", "reference-archive-sha256",
                 "reference-report-sha256", "conversion-report-sha256", "checkpoint-sha256", "model-gguf-sha256"):
        parser.add_argument("--" + name, default=os.environ.get(name.upper().replace("-", "_"), ""))
    parser.add_argument("--reference-workflow", default=os.environ.get("REFERENCE_WORKFLOW", ".github/workflows/migan-reference.yml"))
    args = parser.parse_args()
    validate_inputs(args)
    if os.environ.get("GITHUB_ACTIONS") != "true":
        raise RuntimeError("Build/model replay is hosted-only; do not run on the shared development host")
    args.output.mkdir(parents=True, exist_ok=False)
    args.scratch.mkdir(parents=True, exist_ok=False)
    report = {"schema": "crispembed-migan-native-v1", "phase": args.phase, "commands": {}, "cases": [],
              "native_qualified": False, "representative_quality_qualified": False, "published": False,
              "contract_sha256": sha256(ROOT / "docs/migan-reference-contract.json")}
    report["initial_ordinary_final_control"] = {"run_id": INITIAL_NATIVE_RUN,
        "review_artifact_id": INITIAL_NATIVE_REVIEW_ARTIFACT, "review_zip_sha256": INITIAL_NATIVE_REVIEW_ZIP_SHA256,
        "case_final_sha256": INITIAL_ORDINARY_FINAL_SHA256, "scope": "Retention-only repair preserves prior ordinary arithmetic bytes"}
    status = 2
    try:
        binary = build(args.scratch, args.output, report)
        save(args.output, args.scratch, report)
        if args.phase == "build-only":
            report["complete"] = True
            report["passed"] = True
            status = 0
        else:
            directory = original_artifact(args, args.scratch, report)
            original, conversion, metadata, tensors = verify_original(directory, args)
            report["original"] = {"report_sha256": args.reference_report_sha256, "conversion_sha256": args.conversion_report_sha256,
                                  "checkpoint_sha256": args.checkpoint_sha256, "gguf_sha256": args.model_gguf_sha256,
                                  "source_commit": original["source_commit"], "tensor_count": len(original["state"])}
            # Freeze/check every original case before the first native model
            # load, including the malformed-model loader controls.
            for case in original["cases"]:
                verify_case(directory, case)
            report["all_original_cases_verified_before_native"] = True
            receipts = args.output / "original-receipts"
            receipts.mkdir()
            for name in ("original/report.json", "original/tensor-inventory.json", CONVERSION, "checkpoint-identity.json"):
                shutil.copy2(directory / name, receipts / Path(name).name)
            for name in ("LICENSE", "LICENSE-WEIGHTS"):
                shutil.copy2(directory / "source" / name, receipts / name)
            save(args.output, args.scratch, report)
            negative_models(binary, metadata, tensors, args, args.scratch, args.output, report)
            del tensors
            for case in original["cases"]:
                rgb, mask, original_rgb = verify_case(directory, case)
                inputs = args.scratch / case["id"]
                inputs.mkdir()
                for name, payload in (("rgb.raw", rgb), ("mask.raw", mask), ("composited.raw", original_rgb)):
                    (inputs / name).write_bytes(payload)
                native = args.output / "arrays" / case["id"]
                exit_code, log = command("case-" + case["id"], [binary, directory / MODEL,
                                    directory / "original" / case["id"] / "reference.gguf", args.checkpoint_sha256,
                                    inputs / "rgb.raw", inputs / "mask.raw", inputs / "composited.raw", native],
                                    args.output, report, expected=(0, 1))
                parsed = parse_log(log, [row["stage"] for row in case["stages"]])
                if exit_code != (0 if parsed["summary"]["passed"] else 1):
                    raise ValueError("native exit code differs from logged verdict")
                result = verify_native_case(directory, case, native, parsed, rgb, mask, original_rgb)
                result["exit_code"] = exit_code
                report["cases"].append(result)
                save(args.output, args.scratch, report)
            report["complete"] = True
            report["passed"] = all(case["passed"] for case in report["cases"]) and len(report["cases"]) == 8
            report["bounded_numerical_candidate_passed"] = report["passed"]
            status = 0 if report["passed"] else 1
    except Exception as error:
        report["structural_failure"] = {"type": type(error).__name__, "message": str(error)}
        report["passed"] = False
        raise
    finally:
        save(args.output, args.scratch, report)
        review = args.output / "review"
        review.mkdir(exist_ok=True)
        shutil.copy2(args.output / "migan-native.json", review / "migan-native.json")
        for path in args.output.glob("*.log"):
            shutil.copy2(path, review / path.name)
        for path in args.output.glob("*-symbols.txt"):
            shutil.copy2(path, review / path.name)
        if (args.output / "original-receipts").exists():
            shutil.copytree(args.output / "original-receipts", review / "original-receipts", dirs_exist_ok=True)
        for path in (args.output / "arrays").glob("*/*.png"):
            target = review / path.parent.name
            target.mkdir(exist_ok=True)
            shutil.copy2(path, target / path.name)
    raise SystemExit(status)


if __name__ == "__main__":
    main()
