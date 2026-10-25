"""Hosted native F16 plain-OCR eleven-case comparison; pending reviewed source pins."""

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import resource
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


original = module("ovisocr3_quality_original", ROOT / "tools/ci-heavy/ovisocr3_quality_run.py")
native_audit = module("ovisocr3_quality_native_proof", ROOT / "tools/ci-heavy/ovisocr3_native14.py")
contract = original.contract
policy = original.policy

# Set only from separately reviewed eleven-case original and 14-arm native runs.
ORIGINAL_QUALITY_RUN = 0
ORIGINAL_QUALITY_HEAD = ""
ORIGINAL_QUALITY_PRODUCER_SHA256 = ""
ORIGINAL_QUALITY_RECEIPT = ROOT / "tests/results/qwen35-ocr/ovisocr3-original-quality.json"
NATIVE_MODEL_RUN = 0
NATIVE_MODEL_HEAD = ""
NATIVE_MODEL_PRODUCER_SHA256 = ""
NATIVE_MODEL_RECEIPT = ROOT / "tests/results/qwen35-ocr/ovisocr3-native-model.json"


def sha(path):
    return policy._file_sha256(path)


def require_reviewed_original_quality():
    if (ORIGINAL_QUALITY_RUN <= 0 or len(ORIGINAL_QUALITY_HEAD) != 40
            or len(ORIGINAL_QUALITY_PRODUCER_SHA256) != 64):
        raise ValueError("Reviewed original OvisOCR3 eleven-case quality is absent")
    receipt = json.loads(ORIGINAL_QUALITY_RECEIPT.read_text())
    producer, review = receipt.get("producer"), receipt.get("independent_review")
    if not isinstance(producer, dict) or not isinstance(review, dict):
        raise ValueError("Original OvisOCR3 quality producer/review absent")
    canonical = policy._canonical_sha256(producer)
    if (receipt.get("run") != ORIGINAL_QUALITY_RUN or receipt.get("head_sha") != ORIGINAL_QUALITY_HEAD
            or canonical != ORIGINAL_QUALITY_PRODUCER_SHA256
            or review.get("producer_canonical_sha256") != canonical
            or review.get("original_eleven_complete") is not True
            or producer.get("status") != "completed original OvisOCR3 HF CPU/F32 eleven-case plain OCR quality"
            or producer.get("checkpoint_sha256") != policy.CHECKPOINT_SHA256
            or producer.get("panel_canonical_sha256") != contract.PANEL_CANONICAL_SHA256
            or {row["id"] for row in producer.get("inputs", [])} != {row["id"] for row in contract.load_panel()["cases"]}
            or set(producer.get("original_cases", {})) != {row["id"] for row in contract.load_panel()["cases"]}):
        raise ValueError("Original OvisOCR3 eleven-case quality proof differs")
    return producer


def require_reviewed_native():
    if (NATIVE_MODEL_RUN <= 0 or len(NATIVE_MODEL_HEAD) != 40
            or len(NATIVE_MODEL_PRODUCER_SHA256) != 64):
        raise ValueError("Reviewed OvisOCR3 native fourteen-arm model proof is absent")
    receipt = json.loads(NATIVE_MODEL_RECEIPT.read_text())
    producer, review = receipt.get("producer"), receipt.get("independent_review")
    if not isinstance(producer, dict) or not isinstance(review, dict):
        raise ValueError("OvisOCR3 native model producer/review absent")
    canonical = policy._canonical_sha256(producer)
    if (receipt.get("run") != NATIVE_MODEL_RUN or receipt.get("head_sha") != NATIVE_MODEL_HEAD
            or canonical != NATIVE_MODEL_PRODUCER_SHA256
            or review.get("producer_canonical_sha256") != canonical
            or review.get("all_fourteen_native_arms_positive") is not True
            or producer.get("checkpoint_sha256") != policy.CHECKPOINT_SHA256
            or not re.fullmatch(r"[0-9a-f]{64}", producer.get("exports", {}).get("f16", {}).get("model_sha256", ""))):
        raise ValueError("OvisOCR3 native fourteen-arm source/export proof differs")
    source = producer.get("native_source_sha256")
    if not isinstance(source, dict) or not source:
        raise ValueError("OvisOCR3 native source closure absent")
    if any(sha(ROOT / name) != expected for name, expected in source.items()):
        raise ValueError("OvisOCR3 native source closure changed after fourteen-arm proof")
    return producer


def parse_native_result(text, case):
    lines = [line.removeprefix("OVIS3_QUALITY_RESULT ") for line in text.splitlines()
             if line.startswith("OVIS3_QUALITY_RESULT ")]
    if len(lines) != 1:
        raise ValueError("OvisOCR3 native quality result count differs")
    result = json.loads(lines[0])
    ids, confidence = result.get("token_ids"), result.get("confidences")
    if (not isinstance(ids, list) or not 1 <= len(ids) <= case["cap"]
            or any(type(value) is not int or not 0 <= value < 248320 for value in ids)
            or not isinstance(confidence, list) or len(confidence) != len(ids)
            or any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
                   for value in confidence)
            or type(result.get("mean_confidence")) not in (int, float)
            or not math.isfinite(result["mean_confidence"])
            or abs(sum(confidence) / len(confidence) - result["mean_confidence"]) > 1e-6
            or result.get("cap") != case["cap"]
            or result.get("stop_kind") not in (1, 2)
            or (result["stop_kind"] == 1 and result.get("stop_id") != ids[-1])
            or (result["stop_kind"] == 2 and (result.get("stop_id") is not None or len(ids) != case["cap"]))):
        raise ValueError("OvisOCR3 native quality token/confidence/stop contract differs")
    raw_bytes = bytes.fromhex(result["raw_text_utf8_hex"])
    raw = raw_bytes.decode("utf-8")
    stop = result["stop_id"] if result["stop_kind"] == 1 else None
    row = {"rgb_sha256": case["rgb_sha256"], "png_sha256": case["png_sha256"],
           "cap": case["cap"], "token_ids": ids, "raw_text": raw,
           "raw_utf8_hex": result["raw_text_utf8_hex"],
           "stop_id": stop, "stop_kind": "eos" if stop is not None else "length",
           "confidences": confidence, "mean_confidence": result["mean_confidence"],
           "wall_seconds": result.get("wall_seconds"), "init_seconds": result.get("init_seconds"),
           "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    contract.score_case("positive gold", row, case["cap"])
    return row


def command(label, args, out, report, save, timeout=1800, env=None):
    log = out / (label + ".log")
    with log.open("wb") as stream:
        completed = subprocess.run([str(value) for value in args], stdout=stream,
                                   stderr=subprocess.STDOUT, timeout=timeout, env=env, check=False)
    entry = {"exit_code": completed.returncode, "log_sha256": sha(log), "argv_labels": [Path(str(args[0])).name]}
    report.setdefault("commands", {})[label] = entry
    save()
    return completed.returncode == 0, log.read_text(errors="replace")


def main(preflight=False):
    out = Path(os.environ["HEAVY_OUT"])
    scratch = Path(os.environ["HEAVY_SCRATCH"]) / "ovisocr3-quality-f16"
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    report = {"phase": "preflight", "status": "started", "source_commit": os.environ.get("GITHUB_SHA"),
              "scope": "native OvisOCR3 F16 versus frozen original eleven-case plain OCR; no public alias",
              "source_sha256": {name: sha(ROOT / name) for name in (
                  "tools/ci-heavy/ovisocr3_quality_f16.py", "tools/ci-heavy/ovisocr3_quality_contract.py",
                  "models/convert-qwen35-ocr-to-gguf.py", "tests/ovisocr3_quality_run.cpp")}}
    report_path = out / "ovisocr3-f16-quality.json"
    def save():
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    save()
    try:
        frozen = require_reviewed_original_quality()
        native = require_reviewed_native()
        metadata, service = original.reference.metadata_service()
        report["prerequisites"] = {"original_quality_canonical_sha256": policy._canonical_sha256(frozen),
                                   "native_model_canonical_sha256": policy._canonical_sha256(native),
                                   "metadata_service": service}
        save()
        if preflight:
            report["status"] = "preflight passed; no model/framework loaded"
            report["phase"] = "preflight-complete"; save()
            return report
        from huggingface_hub import snapshot_download, hf_hub_download
        report["phase"] = "same-installed-HF-dual-EOS-proof"; save()
        stop_path, stop_proof = native_audit.installed_hf_stop_probe(out)
        if sha(stop_path) != native.get("hf_stop_runtime", {}).get("proof_sha256"):
            raise ValueError("OvisOCR3 F16 installed HF stop proof differs from native fourteen-arm export")
        os.environ["CRISPEMBED_OVISOCR3_HF_STOP_PROOF"] = str(stop_path)
        report["hf_stop_runtime"] = {"proof_sha256": sha(stop_path), **stop_proof}
        save()
        report["phase"] = "same-frozen-inputs"; save()
        cases, inputs = original.prepare_inputs(scratch / "inputs")
        old = {row["id"]: row for row in frozen["inputs"]}
        if ({case["id"]: {name: case[name] for name in ("rgb_sha256", "png_sha256", "cap", "split")}
             for case in cases} !=
            {name: {field: row[field] for field in ("rgb_sha256", "png_sha256", "cap", "split")}
             for name, row in old.items()}):
            raise ValueError("OvisOCR3 F16 input bytes/splits differ from first original eleven")
        report["inputs"] = [{key: value for key, value in case.items() if key != "path"} for case in cases]
        report["input_preparation"] = inputs
        save()
        report["phase"] = "pinned-checkpoint-and-reviewed-f16-reconstruction"; save()
        directory = Path(snapshot_download(policy.MODEL, revision=policy.REVISION,
                                           allow_patterns=list(policy.metadata_files())))
        profile = policy.load_profile(directory, policy.MODEL, policy.REVISION)
        if {name: sha(directory / name) for name in policy.metadata_files()} != metadata["source_files_sha256"]:
            raise ValueError("OvisOCR3 F16 metadata source differs")
        checkpoint = Path(hf_hub_download(policy.MODEL, "model.safetensors", revision=policy.REVISION))
        report["checkpoint_sha256"] = policy.verify_checkpoint(checkpoint, profile)
        if checkpoint.parent.resolve() != directory.resolve():
            raise ValueError("OvisOCR3 F16 checkpoint/metadata snapshot differs")
        model = scratch / "ovisocr3-f16.gguf"
        converted, _ = command("f16-convert", [sys.executable, ROOT / "models/convert-qwen35-ocr-to-gguf.py",
                                                "--model", policy.MODEL, "--revision", policy.REVISION,
                                                "--profile", "ovisocr3-hf-reference", "--dtype", "f16", "--output", model],
                               out, report, save, timeout=3600)
        if not converted or not model.is_file() or sha(model) != native["exports"]["f16"]["model_sha256"]:
            raise ValueError("OvisOCR3 F16 reconstructed model differs from reviewed fourteen-arm export")
        report["f16_model_sha256"] = sha(model)
        save()
        build = scratch / "build"
        for label, args in (("configure", ["cmake", "-S", ROOT, "-B", build, "-DCMAKE_BUILD_TYPE=Release",
                                           "-DGGML_NATIVE=OFF", "-DCRISPEMBED_NATIVE=OFF", "-DGGML_METAL=OFF",
                                           "-DGGML_BLAS=OFF", "-DCRISPEMBED_BUILD_SHARED=OFF"]),
                            ("cpu-baseline", [sys.executable, ROOT / "scripts/check-cpu-baseline.py", build]),
                            ("build", ["cmake", "--build", build, "--target", "test-ovisocr3-quality-run", "-j2"])):
            okay, _ = command(label, args, out, report, save, timeout=1800)
            if not okay:
                raise ValueError("OvisOCR3 F16 portable native " + label + " failed")
        binary = build / "test-ovisocr3-quality-run"
        report["native_binary_sha256"] = sha(binary)
        report["phase"] = "native-f16-eleven-case-forward"; save()
        env = os.environ.copy()
        env.update({"CRISPEMBED_QWEN35_OCR": "1", "CRISPEMBED_OVISOCR3_FRONTEND": "1",
                    "CRISPEMBED_OVISOCR3_INFERENCE": "1", "CRISPEMBED_OVISOCR2_FRONTEND": "0",
                    "CRISPEMBED_OVISOCR2_INFERENCE": "0"})
        outputs = {}
        for case in cases:
            okay, log = command(case["id"], [binary, model, case["path"], case["cap"], 2],
                                 out, report, save, timeout=2400, env=env)
            if not okay:
                report.setdefault("failures", []).append(case["id"] + " native exit nonzero")
                save()
                continue
            try:
                outputs[case["id"]] = parse_native_result(log, case)
            except (ValueError, KeyError, TypeError) as error:
                report.setdefault("failures", []).append(case["id"] + ": " + str(error)[:300])
            report["native_cases"] = outputs
            save()
        if len(outputs) == 11:
            report["scores"] = contract.score_panel(cases, outputs)
            report["paired"] = contract.paired_screen(frozen["original_cases"], outputs)
            save()
        report["phase"] = "complete"
        report["status"] = "completed diagnostic OvisOCR3 F16 eleven-case quality"
        report["qualified"] = len(outputs) == 11 and not report.get("failures") and report["scores"]["all_complete"] and report["paired"]["all_exact"]
        save()
        return report
    except BaseException as error:
        report["status"] = "failed OvisOCR3 F16 quality execution"
        report["error_type"] = type(error).__name__
        report["error_message"] = original.reference.failure_detail(error)
        save()
        raise


if __name__ == "__main__":
    main("--preflight" in sys.argv)
