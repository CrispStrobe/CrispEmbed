#!/usr/bin/env python3
"""Hosted, diagnostic OvisOCR2 CPU/F32 original versus native model audit.

The GPU receipt is a prerequisite, not a numerical oracle for this CPU graph.
Large weights, NPZ and GGUF files remain in HEAVY_SCRATCH; HEAVY_OUT holds
small reports and complete command logs. This driver never promotes an alias.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "tests/results/qwen35-ocr"
GPU_RECEIPT = EVIDENCE / "ovisocr2-vllm-reference.json"
MODEL = "ATH-MaaS/OvisOCR2"
REVISION = "cbdede82e048659b83310476ff807da825f8fe78"
MIRROR = "cstr/crispembed-regression-fixtures"
MIRROR_REVISION = "50b63a86f2d706491c3cfb953da28b143324fa48"
MIRROR_FILE = "qwen35-ovisocr2/original-processor/metadata.zip"
MIRROR_BYTES = 524058
MIRROR_SHA = "e3ddf289cbac24d48d4cb4818d7cc17a1ed46be3dd2afd361c6ad747d2c7dde5"
VLLM_COMMIT = "0decac0d96c42b49572498019f0a0e3600f50398"
WORKER_SHA = "09156255c01a87a292234a5e4b90f10788e8c949e29d27cbf2b3efbc867ac72f"
SERVICES = {
    "metadata": (39983101540, "ca3d096fe673f3554e95ef8e5f534b597564233e",
                 ".github/workflows/ovisocr2-metadata.yml"),
    "export": (39992662147, "5e197baee331d9b5745c6d906c21893e8211a3fd",
               ".github/workflows/ovisocr2-metadata.yml"),
    "frontend": (40017014043, "ccb6059a6b07b1339abc4de7dd085b0f08af8493",
                 ".github/workflows/ovisocr2-metadata.yml"),
}
VERSIONS = {"torch": "2.11.0", "torchvision": "0.26.0", "transformers": "5.16.1",
            "tokenizers": "0.23.3", "numpy": "2.2.6", "gguf": "0.19.0",
            "safetensors": "0.8.0", "Pillow": "12.1.0", "huggingface-hub": "1.33.0"}
NATIVE_TARGETS = ("test-qwen35-decoder", "test-qwen35-vision", "test-qwen35-delta",
                  "test-ovisocr2-context")
NATIVE_ENV = {name: "1" for name in ("CRISPEMBED_QWEN35_OCR", "CRISPEMBED_OVISOCR2_FRONTEND",
                                     "CRISPEMBED_OVISOCR2_INFERENCE", "CRISPEMBED_QWEN35_AUDIT_ALL_LAYERS")}
NATIVE_ENV.update({name: "0" for name in (
    "CRISPEMBED_QWEN35_PRECISE_RMS", "CRISPEMBED_QWEN35_VISION_PRECISE_NORM",
    "CRISPEMBED_QWEN35_VISION_PRECISE_ATTN", "CRISPEMBED_QWEN35_VISION_CASCADE_NORM",
    "CRISPEMBED_QWEN35_IMATRIX", "CRISPEMBED_QWEN35_PARALLEL_F16",
    "CRISPEMBED_QWEN35_PROFILE", "CRISPEMBED_QWEN35_QUANT",
    "CRISPEMBED_QWEN35_TOKENIZER_MARKS")})


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def policy():
    spec = importlib.util.spec_from_file_location("ovis_model_audit_policy", ROOT / "models/qwen35_ovis_profile.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_gpu_receipt(receipt):
    """Only the independently reviewed, actually executed T4 worker unlocks CPU weights."""
    if not isinstance(receipt, dict):
        raise ValueError("Missing original vLLM GPU receipt")
    producer, review = receipt.get("producer"), receipt.get("independent_review")
    if not isinstance(producer, dict) or not isinstance(review, dict):
        raise ValueError("Original vLLM producer and independent review are required")
    worker = producer.get("worker")
    gpu_control = worker.get("gpu_control") if isinstance(worker, dict) else None
    stop_probe = worker.get("runtime_stop_probe") if isinstance(worker, dict) else None
    if (producer.get("status") != "completed bounded original vLLM T4 reference"
            or review.get("genuine_gpu_forward_completed") is not True
            or not isinstance(worker, dict)
            or worker.get("status") != "completed bounded original vLLM reference"
            or not isinstance(gpu_control, dict) or gpu_control.get("passed") is not True
            or worker.get("resolved_vllm_dtype") != "torch.float16"
            or not isinstance(stop_probe, dict) or stop_probe.get("all_stop_ids") != [248044, 248046]
            or {case.get("case") for case in worker.get("generations", [])} != {"square", "readable"}):
        raise ValueError("Original vLLM T4 forward receipt lacks positive reviewed execution")
    readable = worker.get("owned_readable", {})
    if (readable.get("case") != "readable" or
            not re.fullmatch(r"[0-9a-f]{64}", readable.get("rgb_sha256", ""))):
        raise ValueError("Original vLLM readable RGB SHA is missing")
    pins = producer.get("pins", {})
    if (pins.get("model") != MODEL or pins.get("revision") != REVISION
            or pins.get("checkpoint_sha256") != policy().CHECKPOINT_SHA256
            or pins.get("vllm_version") != "0.22.1"
            or pins.get("vllm_commit") != VLLM_COMMIT
            or pins.get("worker_sha256") != WORKER_SHA
            or producer.get("worker_sha256") != WORKER_SHA
            or pins.get("original_mirror_revision") != MIRROR_REVISION
            or pins.get("original_mirror_zip_sha256") != MIRROR_SHA):
        raise ValueError("Original vLLM receipt pins differ")
    readable_generation = next(case for case in worker["generations"] if case["case"] == "readable")
    return {"producer_status": producer["status"], "independent_review": True,
            "readable_rgb_sha256": readable["rgb_sha256"],
            "gpu_dtype": worker["resolved_vllm_dtype"],
            "worker_source_sha256": producer["worker_sha256"],
            "vllm_commit": pins["vllm_commit"],
            "readable_generation": {key: readable_generation.get(key) for key in
                                    ("token_ids", "finish_reason", "stop_reason", "raw_text",
                                     "processed_text", "bounded_max_tokens")},
            "producer_sha256": hashlib.sha256(json.dumps(producer, sort_keys=True,
                                        separators=(",", ":")).encode()).hexdigest()}


def require_public_evidence():
    metadata, export, frontend = (load_json(EVIDENCE / (name + ".json")) for name in
                                  ("ovisocr2-metadata", "ovisocr2-export", "ovisocr2-frontend"))
    producer = policy()._verified_evidence()
    if (metadata.get("run") != SERVICES["metadata"][0]
            or producer["schema"]["checkpoint_tensors"] != 473
            or export.get("run") != SERVICES["export"][0]
            or export["independent_review"].get("workflow_success_and_source_exact") is not True
            or any(export["producer"]["exports"][dtype].get("all_original_cast_payloads_exact") is not True
                   for dtype in ("f32", "f16"))
            or frontend.get("run") != SERVICES["frontend"][0]
            or frontend["independent_review"]["checks"].get("native_exit_zero") is not True):
        raise ValueError("Metadata, export, or frontend independent acceptance differs")
    return metadata, export, frontend


def service_gate(name, run_info, call):
    number, head, workflow = run_info
    result = call(["gh", "api", f"repos/CrispStrobe/CrispEmbed/actions/runs/{number}"])
    data = json.loads(result)
    if (data.get("status") != "completed" or data.get("conclusion") != "success"
            or data.get("head_sha") != head or data.get("path") != workflow):
        raise ValueError(f"{name} service run/head/workflow differs")
    return {key: data[key] for key in ("id", "status", "conclusion", "head_sha", "path")}


def check_versions():
    found = {name: importlib.metadata.version(name) for name in VERSIONS}
    wrong = {name: value for name, value in found.items()
             if value.split("+")[0] != VERSIONS[name]}
    if wrong:
        raise ValueError("Pinned original CPU package versions differ: " + repr(wrong))
    return found


def summarize_log(path):
    lines = Path(path).read_text(errors="replace").splitlines()
    metrics = [line for line in lines if re.search(
        r"\b(?:PASS|FAIL|BUDGET|DECODED|logits|cos_min|decoded|confidence|cache|argmax)\b", line)]
    failures = [line for line in lines if re.search(r"\b(?:FAIL|ERROR|Traceback|Error:)\b", line)]
    return {"bytes": Path(path).stat().st_size, "sha256": sha(path),
            "first_failure": failures[0] if failures else None, "metric_lines": metrics}


def command(label, args, out, report, *, env=None, timeout=3600):
    """Retain every command log and failure; subsequent independent arms continue."""
    log = out / (label + ".log")
    record = {"argv": [str(arg) for arg in args], "log": log.name, "exit_code": None}
    report.setdefault("commands", {})[label] = record
    with log.open("w") as stream:
        try:
            completed = subprocess.run(record["argv"], cwd=ROOT, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, timeout=timeout, check=False)
            record["exit_code"] = completed.returncode
        except (OSError, subprocess.TimeoutExpired) as exc:
            stream.write(f"\nDriver command error: {type(exc).__name__}: {exc}\n")
            record["error"] = type(exc).__name__
    record.update(summarize_log(log))
    record["passed"] = record["exit_code"] == 0 and record["bytes"] > 0 and not record["first_failure"]
    return record["passed"]


def download_metadata_and_archive(profile_module, report):
    from huggingface_hub import hf_hub_download, snapshot_download

    archive = Path(hf_hub_download(repo_id=MIRROR, repo_type="dataset", revision=MIRROR_REVISION,
                                  filename=MIRROR_FILE))
    if archive.stat().st_size != MIRROR_BYTES or sha(archive) != MIRROR_SHA:
        raise ValueError("Preserved original metadata ZIP bytes differ")
    with zipfile.ZipFile(archive) as source:
        names = source.namelist()
        if len(names) != len(set(names)) or not all(not Path(name).is_absolute() and ".." not in Path(name).parts
                                                   for name in names):
            raise ValueError("Preserved metadata archive member paths are unsafe")
    report["metadata_archive"] = {"sha256": MIRROR_SHA, "bytes": MIRROR_BYTES,
                                  "revision": MIRROR_REVISION, "members": names}
    metadata = Path(snapshot_download(MODEL, revision=REVISION,
                                      allow_patterns=list(profile_module.metadata_files())))
    profile = profile_module.load_profile(metadata, MODEL, REVISION)
    profile_module.validate_config(load_json(metadata / "config.json"), profile)
    report["metadata_source"] = {name: sha(metadata / name)
                                 for name in profile_module.metadata_files()}
    report["metadata_qualification"] = profile["qualification"]
    return metadata, profile


def replay_processor_anchors(metadata, profile, evidence, report):
    """Repeat all three original 2.14 processor fields under the primary 2.11 stack."""
    import numpy as np
    import torch
    import transformers
    import tokenizers
    from PIL import Image, ImageDraw
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    sys.path.insert(0, str(ROOT / "tools"))
    from dump_ovisocr2_reference import check_framework_sources

    processor = AutoProcessor.from_pretrained(metadata, local_files_only=True, trust_remote_code=False)
    loaded = check_framework_sources(profile, transformers, tokenizers, torch, processor,
                                     Qwen3_5ForConditionalGeneration)
    report["loaded_source_sha256"] = loaded
    rows = {row["case"]: row for row in evidence["producer"]["inputs"]}
    cases = (("square", 448, 448), ("odd", 449, 417), ("small-control", 96, 64))
    if set(rows) != {case[0] for case in cases}:
        raise ValueError("Original processor anchor case set differs")
    replay = []
    for label, width, height in cases:
        y, x = np.indices((height, width))
        rgb = np.stack(((x * 3 + y) % 256, (y * 5 + x) % 256,
                        (x + y * 7) % 256), axis=-1).astype(np.uint8)
        row = rows[label]
        rgb_sha = hashlib.sha256(rgb.tobytes()).hexdigest()
        if list(rgb.shape) != row["rgb_shape"] or rgb_sha != row["rgb_sha256"]:
            raise ValueError("Original RGB anchor differs: " + label)
        values = processor(text=[profile["rendered_prompt"]], images=[Image.fromarray(rgb)],
                           return_tensors="pt", images_kwargs={
                               "min_pixels": profile["image_min_pixels"],
                               "max_pixels": profile["image_max_pixels"]})
        fields = {}
        if set(values) != set(row["fields"]):
            raise ValueError("Original processor field set differs under Torch 2.11: " + label)
        for name, tensor in values.items():
            array = tensor.detach().cpu().numpy()
            fields[name] = {"shape": list(array.shape), "dtype": str(array.dtype),
                            "sha256": hashlib.sha256(array.tobytes()).hexdigest()}
            if fields[name] != row["fields"][name]:
                replay.append({"case": label, "rgb_sha256": rgb_sha, "fields": fields,
                               "match": False, "first_different_field": name})
                report["processor_replay"] = replay
                raise ValueError("Original processor field SHA differs under Torch 2.11: " + label + "/" + name)
        replay.append({"case": label, "rgb_sha256": rgb_sha, "fields": fields, "match": True})
        report["processor_replay"] = replay
    image = Image.new("RGB", (448, 448), "white")
    ImageDraw.Draw(image).text((30, 80), "Hello OCR\n12345", fill="black", font_size=24)
    return image


def verify_readable(image, gpu_receipt, scratch, report):
    import numpy as np

    raw = hashlib.sha256(np.asarray(image).tobytes()).hexdigest()
    path = scratch / "owned-readable.png"
    image.save(path)
    report["readable"] = {"raw_rgb_sha256": raw, "png_sha256": sha(path),
                          "gpu_raw_rgb_sha256": gpu_receipt["readable_rgb_sha256"]}
    if raw != gpu_receipt["readable_rgb_sha256"]:
        raise ValueError("CPU readable RGB differs from qualified original GPU fixture")
    return path


def verify_original_capture(path, profile, report):
    data = load_json(path)
    gate = data.get("generation_result", {})
    control = data.get("observational_control", {})
    generated_ids = data.get("generated_ids")
    if not isinstance(generated_ids, list) or len(generated_ids) != 1 or not isinstance(generated_ids[0], list):
        raise ValueError("Original generated ID shape differs")
    last = generated_ids[0][-1] if generated_ids[0] else None
    reached = last in (248044, 248046)
    if (data.get("source_model") != MODEL or data.get("source_revision") != REVISION
            or data.get("checkpoint_sha256") != profile["checkpoint_sha256"]
            or data.get("loaded_source_sha256") != profile["qualification"]["loaded_source_sha256"]
            or data.get("vision_layers") != 12 or data.get("decoder_layers") != 24
            or data.get("missing_stages") != []
            or control.get("prefill_logits_exact") is not True
            or control.get("greedy_ids_exact") is not True
            or data.get("cached_vs_full_prefix_same_argmax") is not True
            or not isinstance(data.get("cached_vs_full_prefix_max_abs"), (int, float))
            or not math.isfinite(data["cached_vs_full_prefix_max_abs"])
            or data["cached_vs_full_prefix_max_abs"] > 0.002
            or gate.get("max_new_tokens") != 12 or gate.get("eos_ids") != [248044, 248046]
            or gate.get("generated_token_count") != len(generated_ids[0])
            or gate.get("reached_eos") is not reached
            or gate.get("stop_id") != (last if reached else None)
            or gate.get("length_limited") is not (len(generated_ids[0]) == 12 and not reached)
            or gate.get("generated_id_shape") != [1, len(generated_ids[0])]):
        raise ValueError("Original F32 capture, cache or generation gate failed")
    report["original_capture"] = {"sha256": sha(path), "generation_result": gate,
        "cached_vs_full_prefix_max_abs": data["cached_vs_full_prefix_max_abs"],
        "cached_vs_full_prefix_same_argmax": True, "observational_control": control,
        "stages": data["stages"], "decoded": data["decoded"],
        "raw_decoded": data["raw_decoded"], "generated_ids": data["generated_ids"]}
    return data


def verify_export(model, manifest_path, precision, evidence):
    manifest = load_json(manifest_path)
    original = evidence["producer"]["exports"][precision]
    review = evidence["independent_review"]["exports"][precision]
    if (manifest.get("dtype") != precision or manifest.get("source_model") != MODEL
            or manifest.get("source_revision") != REVISION or manifest.get("keep_f32") != []
            or manifest.get("checkpoint_tensor_count") != 473
            or manifest.get("export_tensor_count") != 473
            or len(manifest.get("records", [])) != 473
            or sha(manifest_path) != original["manifest_sha256"]
            or sha(manifest_path) != review["manifest_sha256"]
            or model.stat().st_size != original["bytes"]
            or model.stat().st_size != review["bytes_reported"]
            or sha(model) != original["model_sha256"]
            or sha(model) != review["model_sha256_reported"]):
        raise ValueError("Actual F32/F16 export differs from qualified manifest/file SHA: " + precision)
    return {"model_sha256": original["model_sha256"], "bytes": original["bytes"],
            "manifest_sha256": original["manifest_sha256"], "tensor_count": 473}


def native_arms(precision, build, model, readable, reference):
    """Every arm is independent; the caller records all exits after any failure."""
    decoder = build / "test-qwen35-decoder"
    delta = build / "test-qwen35-delta"
    vision = build / "test-qwen35-vision"
    context = build / "test-ovisocr2-context"
    arms = {
        precision + "-delta": [delta, reference],
        precision + "-vision": [vision, model, reference],
        precision + "-decoder-prepared": [decoder, model, reference],
        precision + "-decoder-native-vision": [decoder, model, reference, "--native-vision"],
        precision + "-decoder-native-positions": [decoder, model, reference, "--native-positions"],
        precision + "-decoder-native-frontend": [decoder, model, reference, "--native-frontend"],
        precision + "-context": [context, model, readable, reference],
    }
    if precision == "f32":
        arms[precision + "-context"].append("--one-context")
    return arms


def run_native_matrix(build, models, readable, reference, out, report, save, precisions=("f32", "f16")):
    env = dict(os.environ, **NATIVE_ENV)
    report["native_environment"] = NATIVE_ENV
    for precision in precisions:
        model = models.get(precision)
        if model is None:
            report.setdefault("failures", []).append("missing-verified-" + precision + "-model")
            continue
        for label, args in native_arms(precision, build, model, readable, reference).items():
            if not Path(args[0]).is_file():
                report.setdefault("failures", []).append("missing-native-target-" + label)
                continue
            if not command(label, args, out, report, env=env, timeout=1800):
                report.setdefault("failures", []).append(label)
            save()


def run_main(preflight_only=False):
    out = Path(os.environ["HEAVY_OUT"])
    scratch = Path(os.environ["HEAVY_SCRATCH"]) / "ovis-model"
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    report_path = out / "ovisocr2-model-audit.json"
    report = {"status": "started", "phase": "preflight", "scope":
              "Original HF CPU/F32 versus separate experimental native F32/F16; "
              "vLLM T4/F16 is a distinct decoded reference, not a CPU tensor oracle",
              "failures": [], "commands": {}, "source_commit": os.environ.get("GITHUB_SHA"),
              "source_sha256": {name: sha(ROOT / name) for name in (
                  "tools/ci-heavy/ovisocr2_model_audit.py", "tools/dump_ovisocr2_reference.py",
                  "tools/dump_qwen35_ocr_reference.py", "models/qwen35_ovis_profile.py",
                  "models/convert-qwen35-ocr-to-gguf.py")}}

    def save():
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    try:
        if not GPU_RECEIPT.is_file():
            raise ValueError("Reviewed original vLLM receipt is absent; stop before heavy imports or weights")
        gpu = require_gpu_receipt(load_json(GPU_RECEIPT))
        report["gpu_prerequisite"] = {**gpu, "receipt_sha256": sha(GPU_RECEIPT)}
        metadata_evidence, export_evidence, frontend_evidence = require_public_evidence()
        report["public_evidence"] = {
            "metadata_run": metadata_evidence["run"], "export_run": export_evidence["run"],
            "frontend_run": frontend_evidence["run"],
            "checkpoint_tensors": metadata_evidence["producer"]["schema"]["checkpoint_tensors"],
            "expected_tensors": metadata_evidence["producer"]["schema"]["expected_tensors"],
            "original_framework_source_sha256": metadata_evidence["producer"]["loaded_source_sha256"],
            "metadata_producer_sha256": policy().PRODUCER_SHA256,
            "export_report_sha256": sha(EVIDENCE / "ovisocr2-export.json"),
            "frontend_report_sha256": sha(EVIDENCE / "ovisocr2-frontend.json")}
        report["service_runs"] = {name: service_gate(name, values, subprocess.check_output)
                                  for name, values in SERVICES.items()}
        report["phase"] = "preflight-complete"
        save()
        if preflight_only:
            report["status"] = "preflight passed; no framework/model loaded"
            save()
            return report

        report["versions"] = check_versions()
        report["phase"] = "metadata-archive-and-source"
        save()
        metadata, profile = download_metadata_and_archive(policy(), report)
        save()
        report["phase"] = "torch-2.11-processor-replay"
        save()
        readable_image = replay_processor_anchors(metadata, profile, metadata_evidence, report)
        readable = verify_readable(readable_image, gpu, scratch, report)
        save()

        # This is the first checkpoint download. All three original metadata
        # anchors and the reviewed GPU readable RGB identity have passed.
        from huggingface_hub import hf_hub_download
        checkpoint = Path(hf_hub_download(MODEL, "model.safetensors", revision=REVISION))
        report["checkpoint_sha256"] = policy().verify_checkpoint(checkpoint, profile)
        if checkpoint.parent.resolve() != metadata.resolve():
            raise ValueError("Metadata and checkpoint did not resolve to one pinned snapshot")
        report["phase"] = "original-f32-capture"
        save()
        reference_npz = scratch / "ovisocr2-original.npz"
        capture_args = [sys.executable, ROOT / "tools/dump_qwen35_ocr_reference.py",
                        "--profile", "ovisocr2", "--model-dir", metadata, "--image", readable,
                        "--output", reference_npz, "--max-new-tokens", "12", "--threads", "2"]
        capture_ok = command("original-f32-capture", capture_args, out, report, timeout=3600)
        save()
        reference_json = reference_npz.with_suffix(".json")
        reference_gguf = scratch / "official-delta.gguf"
        if not (capture_ok and reference_npz.is_file() and reference_json.is_file()
                and reference_gguf.is_file()):
            raise ValueError("Original capture failed or required reference artifacts are missing")
        verify_original_capture(reference_json, profile, report)
        if report["original_capture"]["decoded"]:
            report["cross_precision_decoded_observation"] = {
                "cpu_hf_f32": report["original_capture"]["decoded"][0],
                "gpu_vllm_t4_f16": gpu["readable_generation"]["raw_text"],
                "exact_raw_text_match": report["original_capture"]["decoded"][0]
                                        == gpu["readable_generation"]["raw_text"],
                "gate": "observation only; different runtime and precision"}
        report["reference_artifacts"] = {
            name: {"sha256": sha(path), "bytes": path.stat().st_size} for name, path in
            (("npz", reference_npz), ("json", reference_json), ("gguf", reference_gguf))}
        save()

        report["phase"] = "portable-native-build"
        save()
        build = scratch / "build"
        configure = ["cmake", "-S", ROOT, "-B", build, "-DCMAKE_BUILD_TYPE=Release",
                     "-DGGML_NATIVE=OFF", "-DCRISPEMBED_NATIVE=OFF", "-DGGML_METAL=OFF",
                     "-DGGML_BLAS=OFF", "-DCRISPEMBED_BUILD_SHARED=OFF"]
        if not command("cmake-configure", configure, out, report, timeout=600):
            raise ValueError("Portable native configure failed")
        if not command("cpu-baseline-check", [sys.executable, ROOT / "scripts/check-cpu-baseline.py", build],
                       out, report, timeout=120):
            raise ValueError("Portable CPU baseline checker failed before compilation")
        if not command("native-target-build", ["cmake", "--build", build, "--config", "Release",
                                               "--target", *NATIVE_TARGETS, "-j2"], out, report, timeout=1800):
            raise ValueError("Native audit target build failed")
        save()

        report["phase"] = "same-checkpoint-experimental-exports-and-native-arms"
        save()
        models = {}
        for precision in ("f32", "f16"):
            model = scratch / ("ovisocr2-" + precision + ".gguf")
            args = [sys.executable, ROOT / "models/convert-qwen35-ocr-to-gguf.py",
                    "--model", MODEL, "--revision", REVISION, "--profile", "ovisocr2",
                    "--dtype", precision, "--precision-recipe", "original-f16", "--output", model]
            label = precision + "-convert"
            converted = command(label, args, out, report, timeout=1800)
            manifest = model.with_suffix(".gguf.manifest.json")
            if converted and model.is_file() and manifest.is_file():
                try:
                    result = verify_export(model, manifest, precision, export_evidence)
                    report.setdefault("exports", {})[precision] = result
                    models[precision] = model
                    (out / manifest.name).write_bytes(manifest.read_bytes())
                except (OSError, ValueError, KeyError) as exc:
                    report["failures"].append(precision + "-export-verification: " + str(exc))
            else:
                report["failures"].append(precision + "-conversion-missing-or-failed")
            save()
            # Finish the basic F32 arm before the F16 conversion. A failed
            # native comparison does not suppress later diagnostic arms.
            run_native_matrix(build, models, readable, reference_gguf, out, report, save,
                              precisions=(precision,))
        report["status"] = "completed with failures" if report["failures"] else "completed diagnostic audit"
        save()
        if report["failures"]:
            raise ValueError("One or more original/native precision arms failed")
        return report
    except BaseException as exc:
        report["status"] = "failed hosted model audit"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        if not report["failures"]:
            report["failures"].append(report["phase"])
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true",
                        help="Verify public GPU/service/source prerequisites before heavy dependencies")
    args = parser.parse_args()
    run_main(args.preflight)


if __name__ == "__main__":
    main()
