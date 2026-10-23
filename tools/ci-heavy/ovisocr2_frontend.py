"""Hosted OvisOCR2 frontend reference from the retained original processor arrays.

This never loads checkpoint weights or executes a model/processor forward pass.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[2]
METADATA_RUN = 39983101540
METADATA_SOURCE = "ca3d096fe673f3554e95ef8e5f534b597564233e"
CASES = (("square", 448, 448), ("odd", 449, 417), ("small-control", 96, 64))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def run_info():
    return json.loads(subprocess.check_output([
        "gh", "api", f"repos/CrispStrobe/CrispEmbed/actions/runs/{METADATA_RUN}"
    ]))


def preflight(policy, evidence):
    producer = policy._verified_evidence()
    service = run_info()
    if (service["id"] != METADATA_RUN or service["head_sha"] != METADATA_SOURCE
            or service["status"] != "completed" or service["conclusion"] != "success"
            or service["path"] != ".github/workflows/ovisocr2-metadata.yml"
            or evidence["run"] != METADATA_RUN or producer["source_commit"] != METADATA_SOURCE):
        raise ValueError("Pinned original metadata run is not qualified")
    artifact = json.loads(subprocess.check_output([
        "gh", "api", f"repos/CrispStrobe/CrispEmbed/actions/artifacts/{evidence['artifact']['id']}"
    ]))
    if (artifact["name"] != f"ovisocr2-metadata-{METADATA_RUN}"
            or artifact["expired"] or artifact["workflow_run"]["id"] != METADATA_RUN
            or artifact["expires_at"] != evidence["artifact"]["expires_at"]
            or evidence["artifact"]["zip_sha256"] != evidence["independent_review"]["artifact_zip_sha256"]):
        raise ValueError("Pinned original metadata artifact is unavailable or changed")
    return artifact


def download_artifact(artifact, expected_digest, destination):
    with destination.open("wb") as output:
        subprocess.run(["gh", "api", artifact["archive_download_url"].removeprefix("https://api.github.com/")],
                       stdout=output, check=True, timeout=300)
    if sha(destination) != expected_digest:
        raise ValueError("Original metadata artifact ZIP SHA256 differs")
    with zipfile.ZipFile(destination) as archive:
        names = archive.namelist()
        if (len(names) != len(set(names)) or any(name in (".", "..") or Path(name).name != name for name in names)
                or any(name.endswith("/") for name in names)):
            raise ValueError("Unexpected original metadata ZIP paths")
        required = {f"{label}-processor.npz" for label, _, _ in CASES}
        if not required <= set(names):
            raise ValueError("Original processor arrays missing from artifact")
        archive.extractall(destination.parent / "original")
    return destination.parent / "original"


def verified_inputs(original, producer):
    import numpy as np

    rows = {row["case"]: row for row in producer["inputs"]}
    if set(rows) != {label for label, _, _ in CASES}:
        raise ValueError("Original RGB fixture set differs")
    for label, width, height in CASES:
        row = rows[label]
        y, x = np.indices((height, width))
        rgb = np.stack(((x * 3 + y) % 256, (y * 5 + x) % 256,
                        (x + y * 7) % 256), axis=-1).astype(np.uint8)
        if (list(rgb.shape) != row["rgb_shape"]
                or hashlib.sha256(rgb.tobytes()).hexdigest() != row["rgb_sha256"]):
            raise ValueError("Original RGB recipe or hash differs: " + label)
        file = original / f"{label}-processor.npz"
        if sha(file) != row["archive_sha256"]:
            raise ValueError("Original processor archive SHA256 differs: " + label)
        with np.load(file, allow_pickle=False) as archive:
            if set(archive.files) != set(row["fields"]):
                raise ValueError("Original processor fields differ: " + label)
            arrays = {}
            for key, field in row["fields"].items():
                value = archive[key]
                if (list(value.shape) != field["shape"] or str(value.dtype) != field["dtype"]
                        or hashlib.sha256(value.tobytes()).hexdigest() != field["sha256"]):
                    raise ValueError("Original processor field differs: " + label + ":" + key)
                arrays[key] = value
        if (arrays["input_ids"].tolist() != row["ids"]
                or arrays["attention_mask"].tolist() != row["attention_mask"]
                or arrays["image_grid_thw"].tolist() != row["grid"]):
            raise ValueError("Original IDs, masks or image grid differ: " + label)
        yield label, rgb, arrays, row


def write_reference(destination, source, producer, profile, originals):
    import gguf
    import numpy as np

    exporter = module("models/convert-qwen35-ocr-to-gguf.py", "ovis_frontend_exporter")
    writer = gguf.GGUFWriter(str(destination), "qwen35_ovis_frontend_reference")
    writer.add_string("general.source.huggingface.repository", profile["source_model"])
    writer.add_string("general.source.huggingface.revision", profile["source_revision"])
    writer.add_uint32("qwen35_ocr.format_version", profile["format_version"])
    writer.add_string("qwen35_ocr.profile", profile["name"])
    writer.add_uint32("qwen35_ocr.primary_eos_id", profile["primary_eos"])
    writer.add_key_value("qwen35_ocr.stop_token_ids", profile["stop_ids"],
                         gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.INT32)
    writer.add_string("qwen35_ocr.profile.rendered_prompt", profile["rendered_prompt"])
    writer.add_string("qwen35_ocr.checkpoint_sha256", profile["checkpoint_sha256"])
    writer.add_uint32("qwen35_ocr.image_min_pixels", profile["image_min_pixels"])
    writer.add_uint32("qwen35_ocr.image_max_pixels", profile["image_max_pixels"])
    writer.add_bool("qwen35_ocr.image_max_qualified", False)
    writer.add_bool("qwen35_ocr.native_alias_qualified", False)
    writer.add_array("qwen35_ocr.missing_source_files", profile["missing_source_files"])
    writer.add_string("qwen35_ocr.resolved_generation_config", json.dumps(profile["generation"], sort_keys=True))
    writer.add_string("qwen35_ocr.resolved_processor_config", json.dumps(profile["processor"], sort_keys=True))
    writer.add_string("qwen35_ocr.metadata_qualification", json.dumps(profile["qualification"], sort_keys=True))
    for filename in ("config.json", "preprocessor_config.json", "tokenizer_config.json"):
        writer.add_string("qwen35_ocr." + filename.removesuffix(".json"), (source / filename).read_text())
    writer.add_string("tokenizer.chat_template", (source / "chat_template.jinja").read_text())
    metadata = exporter.read_tokenizer_metadata(source, 248320, profile["generation"])
    exporter.write_tokenizer_metadata(writer, metadata)
    writer.add_uint32("test.image_count", len(CASES))

    def save(key, value):
        array = np.ascontiguousarray(value, dtype=np.float32)
        if not np.isfinite(array).all():
            raise ValueError("Nonfinite original frontend field: " + key)
        writer.add_tensor(key, array)

    rendered = profile["rendered_prompt"].encode("utf-8")
    fixture_records = []
    for index, (case, rgb, arrays, row) in enumerate(originals):
        label = f"image.{index}"
        ids = arrays["input_ids"]
        modalities = arrays.get("mm_token_type_ids")
        if modalities is None:
            modalities = (ids == 248056).astype(np.int64)
        if (modalities.shape != ids.shape or not np.isin(modalities, [0, 1]).all()
                or arrays["pixel_values"].ndim != 2 or arrays["pixel_values"].shape[1] != 1536):
            raise ValueError("Original multimodal input shape differs: " + case)
        for suffix, value in (("rgb", rgb), ("patches", arrays["pixel_values"]),
                              ("grid", arrays["image_grid_thw"]), ("ids", ids),
                              ("mask", arrays["attention_mask"]), ("modalities", modalities)):
            save(label + "." + suffix, value)
        save(label + ".rendered", list(rendered) or [0])
        writer.add_uint32(label + ".rendered.length", len(rendered))
        fixture_records.append({"case": case, "rgb_sha256": row["rgb_sha256"],
                                "archive_sha256": row["archive_sha256"],
                                "field_sha256": {key: value["sha256"] for key, value in row["fields"].items()},
                                "grid": row["grid"], "ids_count": int(ids.shape[1])})
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return fixture_records


def run_logged(command, log, timeout):
    with log.open("w") as output:
        result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, timeout=timeout)
    if result.returncode:
        raise ValueError("Hosted frontend command failed: " + command[0] + " (see retained log)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    out = Path(os.environ["HEAVY_OUT"])
    scratch = Path(os.environ["HEAVY_SCRATCH"])
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    report = {"source_commit": os.environ["GITHUB_SHA"], "driver_sha256": sha(__file__),
              "status": "started", "phase": "preflight", "metadata_run": METADATA_RUN,
              "scope": "Frozen original processor arrays and tokenizer metadata versus one hosted native frontend target; no checkpoint weights or decoded model-quality claim"}

    def save():
        (out / "ovisocr2-frontend.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    try:
        policy = module("models/qwen35_ovis_profile.py", "ovis_frontend_policy")
        evidence = json.loads((ROOT / "tests/results/qwen35-ocr/ovisocr2-metadata.json").read_text())
        artifact = preflight(policy, evidence)
        report["metadata_artifact"] = {"id": artifact["id"], "sha256": evidence["artifact"]["zip_sha256"],
                                       "expires_at": artifact["expires_at"]}
        if args.preflight:
            report["status"] = "preflight passed; no model or framework loaded"
            save()
            return
        report["phase"] = "original-artifact"
        save()
        original = download_artifact(artifact, evidence["artifact"]["zip_sha256"], scratch / "ovisocr2-metadata.zip")
        report["phase"] = "source-metadata"
        save()
        from huggingface_hub import snapshot_download
        source = Path(snapshot_download(policy.MODEL, revision=policy.REVISION,
                                        allow_patterns=list(policy.metadata_files())))
        profile = policy.load_profile(source, policy.MODEL, policy.REVISION)
        report["qualification"] = profile["qualification"]
        report["phase"] = "reference"
        save()
        reference = out / "ovisocr2-frontend-reference.gguf"
        report["fixtures"] = write_reference(reference, source, evidence["producer"], profile,
                                              verified_inputs(original, evidence["producer"]))
        report["reference_sha256"] = sha(reference)
        report["reference_bytes"] = reference.stat().st_size
        report["phase"] = "native-build"
        save()
        build = scratch / "frontend-build"
        run_logged(["cmake", "-S", str(ROOT), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
                    "-DCRISPEMBED_NATIVE=OFF", "-DGGML_NATIVE=OFF", "-DGGML_METAL=OFF",
                    "-DGGML_BLAS=OFF", "-DCRISPEMBED_BUILD_SHARED=OFF"],
                   out / "configure.log", 600)
        run_logged([sys.executable, str(ROOT / "scripts/check-cpu-baseline.py"), str(build)],
                   out / "cpu-baseline.log", 120)
        run_logged(["cmake", "--build", str(build), "--target", "test-ovisocr2-frontend", "-j", "2"],
                   out / "build.log", 1800)
        report["phase"] = "native-test"
        save()
        run_logged([str(build / "test-ovisocr2-frontend"), str(reference)], out / "native-test.log", 600)
        report["status"] = "completed frozen original frontend/native comparison; model decode remains unqualified"
        report["phase"] = "complete"
        save()
    except BaseException as error:
        report["status"] = "failed frontend comparison"
        report["error_type"] = type(error).__name__
        save()
        raise


if __name__ == "__main__":
    main()
