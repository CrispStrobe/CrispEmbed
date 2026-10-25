"""Conditional OvisOCR3 F16 publication; all proof gates run before auth or weights.

Preparation only until separately reviewed native, original eleven-case, F16
eleven-case, and public22 receipts exist. No quantized recipe is accepted.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import urllib.request


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "tests/results/qwen35-ocr"
MODEL = "ATH-MaaS/OvisOCR3"
REVISION = "5f6ce6932efee9ec10c30746db55bd780214b7fd"
DESTINATION = "cstr/OvisOCR3-GGUF"
FILENAME = "ovisocr3-f16.gguf"
SOURCE_CONTRACT_SHA256 = "8ca73ca6a349b67a7533c36de6c14ac066e1152f2c8d517af849b2655087a331"
LICENSE_SHA256 = "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a"
UPSTREAM_README_SHA256 = "71668558db69ac7fcd29a30ccecb4effa9df48efd2757b750742c6e22229bfc6"
ALLOWED = {FILENAME, "LICENSE", "UPSTREAM_README.md", "README.md", "manifest.json", ".gitattributes"}


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value, *, ensure_ascii=True):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=ensure_ascii, allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def receipt(name):
    path = EVIDENCE / name
    require(path.is_file(), "Reviewed OvisOCR3 receipt absent: " + name)
    return json.loads(path.read_text()), digest(path)


def hex_field(value, width):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{%d}" % width, value) is not None


def source_license_contract():
    path = EVIDENCE / "ovisocr3-source-contract.json"
    require(digest(path) == SOURCE_CONTRACT_SHA256, "Pinned OvisOCR3 source contract differs")
    source = json.loads(path.read_text())
    repo, files = source["repository"], source["source_files"]
    require(repo.get("model") == MODEL and repo.get("revision") == REVISION
            and repo.get("public") is True and repo.get("gated") is False
            and repo.get("license_card") == "apache-2.0"
            and repo.get("license_file_sha256") == LICENSE_SHA256
            and repo.get("base_declared") == "Qwen/Qwen3.5-0.8B"
            and repo.get("base_audit_revision") == "2fc06364715b967f1860aea9cf38778875588b17"
            and repo.get("base_license_file_sha256") == LICENSE_SHA256
            and repo.get("base_base_declared") == "Qwen/Qwen3.5-0.8B-Base"
            and repo.get("base_base_audit_revision") == "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
            and repo.get("base_base_license_file_sha256") ==
                "50cbab8a892c5f2993b8c7351a99182507472def3b1374558308605d99b86b32"
            and repo.get("upstream_training_base_revision_known") is False
            and files["LICENSE"] == {"bytes": 11544, "sha256": LICENSE_SHA256}
            and files["README.md"] == {"bytes": 26990, "sha256": UPSTREAM_README_SHA256},
            "Pinned OvisOCR3 Apache/base/source files differ")
    return repo


def require_receipts(destination=DESTINATION):
    """Stdlib-only and fail-closed before Hub imports, auth or model access."""
    require(destination == DESTINATION, "Unreviewed OvisOCR3 publication destination")
    source_license_contract()
    gpu, gpu_sha = receipt("ovisocr3-vllm-reference.json")
    gp, gr = gpu.get("producer", {}), gpu.get("independent_review", {})
    require(gr.get("genuine_gpu_forward_completed") is True
            and gr.get("producer_canonical_sha256") == canonical(gp, ensure_ascii=False)
            and gp.get("worker", {}).get("checkpoint_sha256") ==
                "aa2311c6b22b07cf3afe9aac6e8bcb02a2b19dd8671b2a91d0b144071577e600",
            "Reviewed original OvisOCR3 GPU proof differs")
    original, original_sha = receipt("ovisocr3-original-quality.json")
    original_driver = module("ovis3_publish_original", "tools/ci-heavy/ovisocr3_quality_f16.py")
    original_driver.require_reviewed_original_quality()
    require(original.get("independent_review", {}).get("original_eleven_complete") is True
            and original["independent_review"].get("producer_canonical_sha256") ==
                canonical(original.get("producer", {}))
            and type(original.get("run")) is int and original["run"] > 0
            and hex_field(original.get("head_sha"), 40)
            and type(original.get("artifact_id")) is int and original["artifact_id"] > 0
            and hex_field(original.get("artifact_zip_sha256"), 64),
            "Reviewed original eleven-case quality differs")
    model, model_sha = receipt("ovisocr3-native-model.json")
    public_driver = module("ovis3_publish_public", "tools/ci-heavy/ovisocr3_public_surfaces.py")
    native = public_driver.require_receipt(model)
    quality, quality_sha = receipt("ovisocr3-f16-quality.json")
    quality_pins = public_driver.require_quality_receipt(quality, native)
    require(type(quality.get("artifact_id")) is int and quality["artifact_id"] > 0
            and hex_field(quality.get("artifact_zip_sha256"), 64),
            "Reviewed F16 quality report artifact differs")
    public, public_sha = receipt("ovisocr3-public-surfaces.json")
    pp, pr = public.get("producer", {}), public.get("independent_review", {})
    require(pr.get("public_positive") is True
            and pr.get("producer_canonical_sha256") == canonical(pp)
            and pp.get("status") == "completed public audit" and pp.get("failures") == []
            and pp.get("model_prerequisite", {}).get("receipt_sha256") == model_sha
            and pp.get("quality_prerequisite", {}).get("receipt_sha256") == quality_sha
            and pp.get("artifacts", {}).get("model", {}).get("sha256") == native["model_sha256"]
            and pp.get("public_confidence_comparison", {}).get("passed") is True
            and pp.get("selected_vision_preset") == native["selected_vision_preset"]
            and type(public.get("run")) is int and public["run"] > 0
            and hex_field(public.get("head_sha"), 40)
            and type(public.get("artifact_id")) is int and public["artifact_id"] > 0
            and hex_field(public.get("artifact_zip_sha256"), 64),
            "Reviewed model-backed public22 audit differs")
    export = model["producer"]["exports"]["f16"]
    require(type(export.get("bytes")) is int and export["bytes"] > 1_000_000_000
            and native["model_sha256"] == export["model_sha256"]
            and quality_pins["model_sha256"] == export["model_sha256"],
            "Qualified F16 candidate bytes/SHA differ")
    return {"precision": "f16", "recipe": "original-f16", "destination": destination,
            "candidate_sha256": export["model_sha256"], "candidate_bytes": export["bytes"],
            "selected_vision_preset": native["selected_vision_preset"],
            "owned_png_sha256": native["image_sha256"],
            "owned_rgb_sha256": native["rgb_sha256"],
            "owned_expected_text": native["expected_text"],
            "owned_generated_ids": native["original_generated_ids"],
            "native_run": native["run"], "native_source_commit": native["source_commit"],
            "native_f16_artifact_id": native["artifact_id"],
            "native_f16_artifact_zip_sha256": native["artifact_zip_sha256"],
            "native_f16_artifact_bytes": native["artifact_bytes"],
            "native_report_artifact_id": native["report_artifact_id"],
            "native_report_artifact_zip_sha256": native["report_artifact_zip_sha256"],
            "quality_run": quality_pins["run"], "quality_source_commit": quality_pins["source_commit"],
            "quality_artifact_id": quality["artifact_id"],
            "quality_artifact_zip_sha256": quality["artifact_zip_sha256"],
            "original_run": original["run"], "original_source_commit": original["head_sha"],
            "original_artifact_id": original["artifact_id"],
            "original_artifact_zip_sha256": original["artifact_zip_sha256"],
            "public_run": public["run"], "public_source_commit": public["head_sha"],
            "public_artifact_id": public["artifact_id"],
            "public_artifact_zip_sha256": public["artifact_zip_sha256"],
            "model_producer_canonical_sha256": canonical(model["producer"]),
            "receipt_sha256": {"gpu": gpu_sha, "original_quality": original_sha,
                               "native": model_sha, "f16_quality": quality_sha,
                               "public": public_sha},
            "source_contract_sha256": SOURCE_CONTRACT_SHA256}


def artifact_service(pins, call=subprocess.check_output):
    """Check both report and large model artifacts against the live service."""
    artifacts = ((pins["native_f16_artifact_id"], pins["native_run"],
                  f'ovisocr3-native14-f16-{pins["native_run"]}', pins["native_f16_artifact_zip_sha256"],
                  pins["native_f16_artifact_bytes"]),
                 (pins["native_report_artifact_id"], pins["native_run"],
                  f'ovisocr3-native14-reports-{pins["native_run"]}',
                  pins["native_report_artifact_zip_sha256"], None),
                 (pins["original_artifact_id"], pins["original_run"],
                  f'ovisocr3-original-quality-{pins["original_run"]}',
                  pins["original_artifact_zip_sha256"], None),
                 (pins["quality_artifact_id"], pins["quality_run"],
                  f'ovisocr3-f16-quality-{pins["quality_run"]}',
                  pins["quality_artifact_zip_sha256"], None),
                 (pins["public_artifact_id"], pins["public_run"],
                  f'ovisocr3-public-surfaces-{pins["public_run"]}',
                  pins["public_artifact_zip_sha256"], None))
    result = []
    for artifact_id, run, name, expected_sha, size in artifacts:
        require(type(artifact_id) is int and artifact_id > 0 and hex_field(expected_sha, 64),
                "Reviewed artifact identity absent")
        value = json.loads(call(["gh", "api", f"repos/CrispStrobe/CrispEmbed/actions/artifacts/{artifact_id}"]))
        require(value.get("id") == artifact_id and value.get("name") == name
                and value.get("expired") is False
                and value.get("workflow_run", {}).get("id") == run
                and value.get("digest") == "sha256:" + expected_sha
                and (size is None or value.get("size_in_bytes") == size),
                "Hosted artifact service proof differs: " + name)
        result.append({"id": artifact_id, "name": name, "sha256": expected_sha})
    return result


def run_service(pins, call=subprocess.check_output):
    runs = ((pins["native_run"], pins["native_source_commit"],
             {".github/workflows/ovisocr3-native14.yml", ".github/workflows/ovisocr2-metadata.yml"}),
            (pins["original_run"], pins["original_source_commit"],
             {".github/workflows/ovisocr3-quality.yml"}),
            (pins["quality_run"], pins["quality_source_commit"],
             {".github/workflows/ovisocr3-quality-f16.yml"}),
            (pins["public_run"], pins["public_source_commit"],
             {".github/workflows/ovisocr3-public-surfaces.yml"}))
    results = []
    for run, head, paths in runs:
        require(type(run) is int and run > 0 and hex_field(head, 40),
                "Reviewed hosted run/head absent")
        value = json.loads(call(["gh", "api", f"repos/CrispStrobe/CrispEmbed/actions/runs/{run}"]))
        require(value.get("id") == run and value.get("head_sha") == head
                and value.get("status") == "completed" and value.get("conclusion") == "success"
                and value.get("path") in paths,
                "Reviewed hosted run source/workflow/status differs")
        results.append({"run": run, "head_sha": head, "path": value["path"]})
    return results


def small_source_file(name, expected_sha, expected_bytes, scratch, opener=urllib.request.urlopen):
    url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}"
    with opener(url, timeout=60) as response:
        data = response.read(expected_bytes + 1)
    require(len(data) == expected_bytes and hashlib.sha256(data).hexdigest() == expected_sha,
            "Pinned original source file differs: " + name)
    path = scratch / name
    path.write_bytes(data)
    return path


def card(pins, *, evidence_source_commit):
    require(hex_field(evidence_source_commit, 40), "Audited publication source commit required")
    template = (ROOT / "models/cards/ovisocr3-card.template.md").read_text()
    values = {"source_revision": REVISION, "model_sha256": pins["candidate_sha256"],
              "model_bytes": str(pins["candidate_bytes"]),
              "evidence_source_commit": evidence_source_commit,
              "public_source_commit": pins["public_source_commit"]}
    for key, value in values.items():
        template = template.replace("${" + key + "}", value)
    require("${" not in template, "Unfilled OvisOCR3 model card field")
    return template.encode()


def publish(pins, artifact_dir, out, scratch, *, source_commit):
    """Hosted only, after complete preflight and exact artifact service proof."""
    require(hex_field(source_commit, 40), "Publication source commit required")
    matches = list(Path(artifact_dir).rglob(FILENAME))
    require(len(matches) == 1 and matches[0].stat().st_size == pins["candidate_bytes"]
            and digest(matches[0]) == pins["candidate_sha256"],
            "Reviewed F16 GGUF missing or SHA/bytes differ")
    images = list(Path(artifact_dir).rglob("owned-readable.png"))
    require(len(images) == 1 and digest(images[0]) == pins["owned_png_sha256"],
            "Reviewed source PNG absent or altered in F16 artifact")
    scratch.mkdir(parents=True, exist_ok=True)
    license_file = small_source_file("LICENSE", LICENSE_SHA256, 11544, scratch)
    upstream = small_source_file("README.md", UPSTREAM_README_SHA256, 26990, scratch)
    upstream.rename(scratch / "UPSTREAM_README.md")
    upstream = scratch / "UPSTREAM_README.md"
    filename = FILENAME
    card_bytes = card(pins, evidence_source_commit=source_commit)
    manifest = {"source_model": MODEL, "source_revision": REVISION,
                "audited_source_commit": pins["public_source_commit"],
                "publication_source_commit": source_commit,
                "candidate": {"filename": filename, "precision": "f16", "recipe": "original-f16",
                              "sha256": pins["candidate_sha256"], "bytes": pins["candidate_bytes"]},
                "source_contract_sha256": SOURCE_CONTRACT_SHA256,
                "license_sha256": LICENSE_SHA256, "upstream_readme_sha256": UPSTREAM_README_SHA256,
                "selected_vision_preset": pins["selected_vision_preset"],
                "receipts": pins["receipt_sha256"]}
    manifest_bytes = (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode()
    # All package and credential access is below the pure and artifact gates.
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
    require(os.environ.get("HF_TOKEN"), "Dedicated OvisOCR3 writer credential absent")
    api = HfApi(token=os.environ["HF_TOKEN"])
    try:
        current = api.model_info(DESTINATION, files_metadata=True)
    except Exception as error:
        # An absent repository is expected; all other service errors remain fatal.
        from huggingface_hub.utils import RepositoryNotFoundError
        if not isinstance(error, RepositoryNotFoundError):
            raise
        api.create_repo(DESTINATION, private=False, exist_ok=False)
    else:
        require({item.rfilename for item in current.siblings} <= {".gitattributes"},
                "Destination already contains unreviewed files")
    operations = [CommitOperationAdd(path_in_repo=filename, path_or_fileobj=str(matches[0])),
                  CommitOperationAdd(path_in_repo="LICENSE", path_or_fileobj=str(license_file)),
                  CommitOperationAdd(path_in_repo="UPSTREAM_README.md", path_or_fileobj=str(upstream)),
                  CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card_bytes),
                  CommitOperationAdd(path_in_repo="manifest.json", path_or_fileobj=manifest_bytes)]
    commit = api.create_commit(repo_id=DESTINATION, operations=operations,
                               commit_message="Publish independently qualified OvisOCR3 F16 GGUF")
    require(hex_field(commit.oid, 40), "Immutable publication commit missing")
    info = api.model_info(DESTINATION, revision=commit.oid, files_metadata=True)
    names = {item.rfilename for item in info.siblings}
    model = next((item for item in info.siblings if item.rfilename == filename), None)
    require({FILENAME, "LICENSE", "UPSTREAM_README.md", "README.md", "manifest.json"} <= names
            and names <= ALLOWED and model is not None
            and model.size == pins["candidate_bytes"] and model.lfs
            and model.lfs.sha256 == pins["candidate_sha256"],
            "Immutable remote file allowlist or LFS SHA/size differs")
    cache = scratch / "empty-hf-cache"
    require(not cache.exists(), "Remote verification cache must be empty")
    cache.mkdir()
    remote_sha = {}
    for name, expected in ((filename, pins["candidate_sha256"]),
                           ("LICENSE", LICENSE_SHA256),
                           ("UPSTREAM_README.md", UPSTREAM_README_SHA256),
                           ("README.md", hashlib.sha256(card_bytes).hexdigest()),
                           ("manifest.json", hashlib.sha256(manifest_bytes).hexdigest())):
        path = Path(hf_hub_download(repo_id=DESTINATION, filename=name, revision=commit.oid,
                                    cache_dir=cache, local_files_only=False, force_download=True,
                                    token=False))
        require(digest(path) == expected and (name != filename or path.stat().st_size == pins["candidate_bytes"]),
                "Empty-cache immutable readback differs: " + name)
        remote_sha[name] = expected
    # Test-only PNG stays in the small publication report; never in the model repo.
    (out / "owned-readable.png").write_bytes(images[0].read_bytes())
    return {"repo": DESTINATION, "commit": commit.oid, "filename": filename,
            "sha256": pins["candidate_sha256"], "bytes": pins["candidate_bytes"],
            "small_files_sha256": {k: v for k, v in remote_sha.items() if k != filename},
            "repository_file_allowlist": sorted(names)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--model-run-id", required=True)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--destination", default=DESTINATION)
    args = parser.parse_args()
    out = Path(os.environ["HEAVY_OUT"])
    scratch = Path(os.environ["HEAVY_SCRATCH"]) / "ovisocr3-publish"
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "ovisocr3-publication.json"
    report = {"status": "started", "phase": "preflight", "source_commit": os.environ.get("GITHUB_SHA"),
              "destination": args.destination, "limits": [
                  "CPU F16 still-image OCR only; 1024 patch native bound; upstream 2880-square resize is not qualified.",
                  "Original T4 TRITON_ATTN FP16 and CPU original F32 are distinct baselines.",
                  "Only the reviewed F16 candidate can publish; no quantization or default alias claim."]}
    def save():
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    save()
    try:
        pins = require_receipts(args.destination)
        require(str(pins["native_run"]) == args.model_run_id, "Native artifact run differs")
        report["qualification"] = pins
        report["service_runs"] = run_service(pins)
        report["service_artifacts"] = artifact_service(pins)
        report["phase"] = "preflight-complete"
        save()
        if args.preflight:
            report["status"] = "preflight passed; no auth/packages/weights accessed"
            save()
            return
        require(args.artifact_dir is not None, "Preserved reviewed native F16 artifact required")
        report["phase"] = "publish-and-immutable-readback"
        save()
        report["remote"] = publish(pins, args.artifact_dir, out, scratch,
                                   source_commit=os.environ.get("GITHUB_SHA"))
        report["published_commit"] = report["remote"]["commit"]
        report["status"] = "published and immutable remote file reverified"
        report["phase"] = "complete"
        save()
    except BaseException as error:
        report["status"] = "failed publication or preflight"
        report["error"] = f"{type(error).__name__}: {error}"
        save()
        raise


if __name__ == "__main__":
    main()
