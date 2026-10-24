#!/usr/bin/env python3
"""Hosted reconstruction of the exact LightOn format-1 wrong-profile fixture.

The receipt's full GGUF byte identity is the only success condition. This lane
does not run inference, qualify OCR quality, publish a model, or change aliases.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import urllib.request


ROOT = Path(__file__).resolve().parents[2]
RECEIPT = ROOT / "tests/results/qwen35-ocr/receipt.json"
CONVERTER = ROOT / "tools/ci-heavy/ovisocr2_lighton_converter_frozen.py"
RECEIPT_SHA = "f23d7fabc74437e2a006439b6b69bfd16c146574cafc60209789fab78decbc8a"
CONVERTER_SHA = "b98979524c65a15a95634d1832c0deb66ffaad5585c17f29c76307f0008fbfd3"
SOURCE_COMMIT = "a07e83fbd02e1c8ff73b3f4f9a224ab0cba265c3"
SOURCE_RUN = 38870347237
SOURCE_WORKFLOW = ".github/workflows/heavy-cpu.yml"
MODEL = "lightonai/LightOnOCR-3-0.8B"
REVISION = "36d8636925327c671bd8e479c517a004f7566bd9"
BASE = "Qwen/Qwen3.5-0.8B"
BASE_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
BASE_LICENSE_SHA = "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a"
README_SHA = "0df44eab78278bad416d8974e96a6a181d59bae8416fa54877bd5b022e76aaf2"
CHECKPOINT_SHA = "ae46aa6fd7136d57297ac006ed9ec345b7a2083feb2e99501bedcb014ca3efdf"
CHECKPOINT_BYTES = 1706030528
GGUF_SHA = "41409fc871e76a6bc13e9988d0efd2706e06d0c7c77e2ffb8746c28348ba7bd4"
GGUF_BYTES = 1918139328
GGUF_NAME = "lightonocr-3-0.8b-f16-experimental.gguf"
ARTIFACT_PREFIX = "lightonocr3-format1-receipt-"
VERSIONS = {"torch": "2.10.0", "torchvision": "0.25.0", "transformers": "5.5.4",
            "tokenizers": "0.22.2", "gguf": "0.19.0", "numpy": "2.5.2",
            "huggingface-hub": "1.33.0", "safetensors": "0.8.0", "Pillow": "12.1.0"}

# Immutable Hub API blob identities observed at the exact LightOn revision.
# Plain files use Git blob SHA-1; LFS files use their content SHA-256 as well.
SOURCE_FILES = {
    "README.md": (9271, "afa32cc5457d63f80c4ad410c3c11c8174ef7bb4", None),
    "chat_template.jinja": (7755, "0ef09f214eaa6d9bca297988afc1454b5827b2c7", None),
    "config.json": (2692, "ecc3812e9f54ae67145b34964e501e179cf6853b", None),
    "generation_config.json": (205, "338b88a230e884b84bcd99938df3d00bd97fe0a9", None),
    "model.safetensors": (CHECKPOINT_BYTES, "096ddc4f7c69741e355e65b2cd38d56b7086a023", CHECKPOINT_SHA),
    "preprocessor_config.json": (390, "2ea84a437d448ff71b08df68fdd949d5cc4ebb64", None),
    "processor_config.json": (1191, "33818c7f9e991ad735fd240209f4fa73e6c28c50", None),
    "tokenizer.json": (19989325, "5520bfd2dd834ce386c1312c410fa71af56db5ad",
                       "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"),
    "tokenizer_config.json": (1168, "5344df8c8031f929defb2717c227e879120c6b6d", None),
}
BASE_LICENSE_FILE = (11544, "f938136e3adacfd92be087f6e113b5d6d97f678f", None)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha(path):
    path = Path(path)
    digest = hashlib.sha1()
    digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_converter():
    if sha(CONVERTER) != CONVERTER_SHA:
        raise ValueError("Pinned historical LightOn converter source SHA differs")
    spec = importlib.util.spec_from_file_location("lighton_receipt_converter_frozen", CONVERTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.LIGHTON_MODEL != MODEL or module.LIGHTON_REVISION != REVISION:
        raise ValueError("Historical converter source model/revision differs")
    return module


def receipt_vision_recipe(receipt, converter):
    """Bind the actual 52 retained vision matrices to the old receipt."""
    recipe = converter.precision_recipe("vision-f32")
    records = receipt["weights"]["tensor_audit"]
    vision_matrices = {record["name"] for record in records
                       if record["name"].startswith("v.") and len(record["shape"]) == 2
                       and record["dtype"] == "float32"}
    if (len(recipe) != 52 or len(set(recipe)) != 52 or set(recipe) != vision_matrices
            or sum(record["dtype"] == "float32" for record in records) != 286
            or sum(record["dtype"] == "float16" for record in records) != 187
            or len({record["name"] for record in records}) != 473):
        raise ValueError("Receipt vision-F32 matrix recipe/dtype distribution differs")
    return sorted(recipe)


def require_receipt(path=RECEIPT, converter=CONVERTER):
    if sha(path) != RECEIPT_SHA or sha(converter) != CONVERTER_SHA:
        raise ValueError("Pinned LightOn receipt/converter source SHA differs")
    receipt = json.loads(Path(path).read_text())
    weights = receipt.get("weights", {})
    provenance = receipt.get("baseline_bootstrap", {}).get("provenance", [])
    if (receipt.get("source_commit") != SOURCE_COMMIT
            or receipt.get("model") != MODEL or receipt.get("revision") != REVISION
            or receipt.get("workflow") != f"https://github.com/CrispStrobe/CrispEmbed/actions/runs/{SOURCE_RUN}"
            or receipt.get("workflow_conclusion") != "failure"
            or weights.get("sha256") != GGUF_SHA or weights.get("bytes") != GGUF_BYTES
            or len(weights.get("tensor_audit", [])) != 473
            or provenance != [
                {"model": MODEL, "revision": REVISION, "license": "apache-2.0",
                 "readme_sha256": README_SHA},
                {"model": BASE, "revision": BASE_REVISION, "license": "apache-2.0",
                 "readme_sha256": "87a163af54f32fa608a0f8d3ac67945c53dd2b4a7c96740b3d7fdc28e8458864",
                 "license_sha256": BASE_LICENSE_SHA}]):
        raise ValueError("Pinned LightOn receipt weight/source/provenance differs")
    receipt_vision_recipe(receipt, frozen_converter())
    return receipt


def require_source_run(call):
    raw = call(["gh", "api", f"repos/CrispStrobe/CrispEmbed/actions/runs/{SOURCE_RUN}"])
    run = json.loads(raw)
    if (run.get("id") != SOURCE_RUN or run.get("head_sha") != SOURCE_COMMIT
            or run.get("path") != SOURCE_WORKFLOW or run.get("status") != "completed"
            or run.get("conclusion") != "failure"):
        raise ValueError("Original LightOn receipt run/source/workflow differs")
    return {"id": SOURCE_RUN, "head_sha": SOURCE_COMMIT, "path": SOURCE_WORKFLOW,
            "conclusion": "failure (numerical audit; weight identity retained)"}


def model_api(repo, revision):
    url = f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true"
    with urllib.request.urlopen(url, timeout=30) as stream:
        return json.load(stream)


def require_hub_info(info, repo, revision, files):
    if info.get("id") != repo or info.get("sha") != revision or (info.get("cardData") or {}).get("license") != "apache-2.0":
        raise ValueError("Pinned LightOn/base Hub model revision/license differs")
    siblings = {row.get("rfilename"): row for row in info.get("siblings", [])}
    for name, (size, blob, lfs_sha) in files.items():
        row = siblings.get(name, {})
        if (row.get("size") != size or row.get("blobId") != blob
                or (row.get("lfs") or {}).get("sha256") != lfs_sha):
            raise ValueError("Pinned Hub file size/blob identity differs: " + name)
    return {"model": repo, "revision": revision, "license": "apache-2.0",
            "files": {name: {"bytes": size, "git_blob_sha1": blob,
                              "lfs_sha256": lfs_sha} for name, (size, blob, lfs_sha) in files.items()}}


def verify_file(path, spec):
    size, blob, lfs_sha = spec
    if (Path(path).stat().st_size != size
            or (sha(path) != lfs_sha if lfs_sha else git_blob_sha(path) != blob)):
        raise ValueError("Pinned Hub downloaded file payload differs: " + Path(path).name)
    return sha(path)


def check_versions():
    found = {name: importlib.metadata.version(name) for name in VERSIONS}
    if any(found[name].split("+")[0] != value for name, value in VERSIONS.items()):
        raise ValueError("Pinned LightOn reconstruction package versions differ: " + repr(found))
    return found


def verify_export(model, manifest, receipt):
    keep_f32 = receipt_vision_recipe(receipt, frozen_converter())
    if (manifest.get("source_model") != MODEL or manifest.get("source_revision") != REVISION
            or manifest.get("dtype") != "f16" or manifest.get("keep_f32") != keep_f32
            or manifest.get("checkpoint_tensor_count") != 473
            or manifest.get("export_tensor_count") != 473
            or manifest.get("records") != receipt["weights"]["tensor_audit"]
            or Path(model).stat().st_size != GGUF_BYTES or manifest.get("bytes") != GGUF_BYTES
            or sha(model) != GGUF_SHA):
        raise ValueError("Reconstructed LightOn GGUF differs from full qualified receipt byte/tensor identity")
    return {"filename": Path(model).name, "bytes": GGUF_BYTES, "sha256": GGUF_SHA,
            "tensor_count": 473, "format_version": 1}


def run_main(preflight=False):
    out = Path(os.environ["HEAVY_OUT"])
    scratch = Path(os.environ["HEAVY_SCRATCH"]) / "lighton-format1-fixture"
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    report_path = out / "ovisocr2-lighton-fixture.json"
    report = {"status": "started", "phase": "preflight", "source_commit": os.environ.get("GITHUB_SHA"),
              "scope": "Exact historical vision-F32 format-1 wrong-profile fixture reconstruction; no inference, quality or publication claim",
              "receipt_sha256": RECEIPT_SHA, "converter_sha256": CONVERTER_SHA,
              "historical_converter_commit": SOURCE_COMMIT, "source_model": MODEL,
              "source_revision": REVISION, "expected_gguf_sha256": GGUF_SHA,
              "expected_gguf_bytes": GGUF_BYTES,
              "dependency_provenance": {
                  "recorded_original_workflow": ["torch", "torchvision", "transformers", "tokenizers",
                                                 "gguf", "numpy", "huggingface-hub"],
                  "new_fixed_versions_for_originally_unpinned_dependencies": ["safetensors", "Pillow"],
                  "acceptance": "Full historical GGUF and every tensor-record byte identity"}}

    def save():
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    try:
        receipt = require_receipt()
        report["precision_recipe"] = {"name": "vision-f32", "retained_f32":
                                      receipt_vision_recipe(receipt, frozen_converter()),
                                      "f32_tensor_count": 286, "f16_tensor_count": 187}
        report["receipt_run"] = require_source_run(subprocess.check_output)
        report["hub_source"] = require_hub_info(model_api(MODEL, REVISION), MODEL, REVISION, SOURCE_FILES)
        report["hub_base_license"] = require_hub_info(
            model_api(BASE, BASE_REVISION), BASE, BASE_REVISION, {"LICENSE": BASE_LICENSE_FILE})
        report["phase"] = "preflight-complete"
        save()
        if preflight:
            report["status"] = "preflight passed before framework imports and weights"
            save()
            return report

        report["versions"] = check_versions()
        from huggingface_hub import hf_hub_download, snapshot_download

        report["phase"] = "pinned-metadata-before-checkpoint"
        save()
        metadata_names = [name for name in SOURCE_FILES if name != "model.safetensors"]
        metadata = Path(snapshot_download(MODEL, revision=REVISION, allow_patterns=metadata_names))
        report["metadata_sha256"] = {name: verify_file(metadata / name, SOURCE_FILES[name])
                                     for name in metadata_names}
        if report["metadata_sha256"]["README.md"] != README_SHA:
            raise ValueError("Original LightOn README SHA differs from receipt")
        license_path = Path(hf_hub_download(BASE, "LICENSE", revision=BASE_REVISION))
        verify_file(license_path, BASE_LICENSE_FILE)
        if sha(license_path) != BASE_LICENSE_SHA:
            raise ValueError("Declared base Apache LICENSE differs from receipt")
        save()

        report["phase"] = "pinned-checkpoint-download"
        save()
        checkpoint = Path(hf_hub_download(MODEL, "model.safetensors", revision=REVISION))
        report["checkpoint_sha256"] = verify_file(checkpoint, SOURCE_FILES["model.safetensors"])
        if checkpoint.parent.resolve() != metadata.resolve():
            raise ValueError("LightOn metadata and checkpoint snapshots differ")
        save()

        report["phase"] = "original-converter-exact-byte-reconstruction"
        save()
        converter = frozen_converter()
        keep_f32 = receipt_vision_recipe(receipt, converter)
        model = scratch / GGUF_NAME
        manifest = converter.convert(metadata, model, MODEL, REVISION, dtype="f16", keep_f32=keep_f32)
        report["reconstruction"] = verify_export(model, manifest, receipt)
        manifest_path = model.with_suffix(model.suffix + ".manifest.json")
        report["reconstruction"]["converter_manifest_sha256"] = sha(manifest_path)
        shutil.copyfile(license_path, scratch / "LICENSE")
        fixture_manifest = {"purpose": "real format-1 wrong-profile control only", "quality_claim": False,
                            "receipt_sha256": RECEIPT_SHA, "historical_converter_sha256": CONVERTER_SHA,
                            "model": MODEL, "revision": REVISION, "gguf": report["reconstruction"],
                            "precision_recipe": report["precision_recipe"],
                            "license": {"file": "LICENSE", "sha256": BASE_LICENSE_SHA,
                                        "source_model": BASE, "source_revision": BASE_REVISION,
                                        "upstream_card_license": "apache-2.0"}}
        (scratch / "lighton-fixture-manifest.json").write_text(
            json.dumps(fixture_manifest, indent=2, allow_nan=False) + "\n")
        report["artifact"] = {"name": ARTIFACT_PREFIX + os.environ.get("GITHUB_RUN_ID", "unknown"),
                              "filename": GGUF_NAME, "manifest": "lighton-fixture-manifest.json",
                              "license": "LICENSE"}
        report["status"] = "exact receipt GGUF reconstructed; wrong-profile fixture only"
        report["phase"] = "complete"
        save()
        return report
    except Exception as exc:
        report["status"] = "failed LightOn wrong-profile fixture reconstruction"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    run_main(preflight=args.preflight)


if __name__ == "__main__":
    main()
