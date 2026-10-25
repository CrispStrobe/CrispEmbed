"""Reviewed OvisOCR3 HF-only original-reference export; vLLM remains unqualified."""

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import re


spec = importlib.util.spec_from_file_location("qwen35_ovis3_source_profile", Path(__file__).with_name("qwen35_ovis3_profile.py"))
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)

MODEL = source.MODEL
REVISION = source.REVISION
METADATA_PRODUCER_SHA256 = "804799bdaa944ef572ed820a3ddb71aa14251073223c30cfe8fa9041bb80d5c2"
METADATA_RECEIPT_SHA256 = "84611a0184bd3d0bc49c628524e5fdf906b1f1adb68c2cfdc3030069528f85a5"
METADATA_RECEIPT = Path(__file__).resolve().parents[1] / "tests/results/qwen35-ocr/ovisocr3-metadata.json"

# Deliberately absent until the actual original full-reference run is reviewed.
REFERENCE_RUN = 0
REFERENCE_HEAD = ""
REFERENCE_ARTIFACT_ID = 0
REFERENCE_ARTIFACT_SHA256 = ""
REFERENCE_ARTIFACT_BYTES = 0
REFERENCE_PRODUCER_SHA256 = ""
REFERENCE_RECEIPT = Path(__file__).resolve().parents[1] / "tests/results/qwen35-ocr/ovisocr3-original-reference.json"
STOP_SCOPE = "hf5.16.1-explicit-eos-list-forward; vllm-runtime-unobserved"
STOPPING_CRITERIA_SOURCE_SHA256 = "7e4ce5eeee83eab48a63a5419cd68be6f0591201fe7f9248dda77a7c3045bd7e"


def _checked_stop_proof():
    path = os.environ.get("CRISPEMBED_OVISOCR3_HF_STOP_PROOF", "")
    if not path:
        raise ValueError("OvisOCR3 installed HF EOS runtime proof absent")
    proof_path = Path(path)
    proof = json.loads(proof_path.read_text())
    if (proof.get("transformers_version") != "5.16.1"
            or proof.get("class") != "transformers.generation.stopping_criteria.EosTokenCriteria"
            or proof.get("source_sha256") != STOPPING_CRITERIA_SOURCE_SHA256
            or proof.get("eos_ids") != source.SOURCE_EOS_IDS
            or proof.get("input_last_ids") != [248044, 248046, 42]
            or proof.get("actual_shape") != [3]
            or proof.get("actual_dtype") != "torch.bool"
            or proof.get("actual_stops") != [True, True, False]
            or proof.get("expected_stops") != [True, True, False]):
        raise ValueError("OvisOCR3 installed HF EOS runtime proof differs")
    return source._file_sha256(proof_path)


def metadata_files():
    return source.metadata_files()


def _reviewed_artifacts(reference):
    review = reference.get("independent_review", {})
    large = reference.get("reference_artifact", {})
    small_id, small_sha = reference.get("artifact_id"), reference.get("artifact_zip_sha256")
    if (type(small_id) is not int or small_id < 1
            or not isinstance(small_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", small_sha)
            or review.get("artifact_zip_sha256") != small_sha
            or not isinstance(large, dict) or type(large.get("id")) is not int or large["id"] < 1
            or large["id"] == small_id
            or large.get("name") != f"ovisocr3-reference-{reference.get('run')}"
            or type(large.get("bytes")) is not int or large["bytes"] < 1
            or not isinstance(large.get("zip_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", large["zip_sha256"])
            or review.get("reference_artifact_zip_sha256") != large["zip_sha256"]):
        raise ValueError("OvisOCR3 small/large original reference artifact review differs")
    return {"id": small_id, "zip_sha256": small_sha}, large


def _reviewed_receipts():
    if (REFERENCE_RUN <= 0 or REFERENCE_ARTIFACT_ID <= 0 or REFERENCE_ARTIFACT_BYTES <= 0 or len(REFERENCE_HEAD) != 40
            or len(REFERENCE_ARTIFACT_SHA256) != 64 or len(REFERENCE_PRODUCER_SHA256) != 64):
        raise ValueError("Reviewed original OvisOCR3 HF reference proof absent")
    if source._file_sha256(METADATA_RECEIPT) != METADATA_RECEIPT_SHA256:
        raise ValueError("OvisOCR3 reviewed metadata wrapper bytes differ")
    metadata = json.loads(METADATA_RECEIPT.read_text())
    meta_producer, meta_review = metadata.get("producer"), metadata.get("independent_review")
    if (not isinstance(meta_producer, dict) or not isinstance(meta_review, dict)
            or source._canonical_sha256(meta_producer) != METADATA_PRODUCER_SHA256
            or meta_review.get("producer_canonical_sha256") != METADATA_PRODUCER_SHA256
            or meta_review.get("metadata_positive") is not True
            or metadata.get("rendered_prompt") is None
            or hashlib.sha256(metadata["rendered_prompt"].encode()).hexdigest() !=
               meta_producer.get("rendered_prompt_sha256")):
        raise ValueError("OvisOCR3 original metadata review differs")
    reference = json.loads(REFERENCE_RECEIPT.read_text())
    producer, review = reference.get("producer"), reference.get("independent_review")
    if not isinstance(producer, dict) or not isinstance(review, dict):
        raise ValueError("OvisOCR3 original HF producer/review absent")
    _, large_artifact = _reviewed_artifacts(reference)
    canonical = source._canonical_sha256(producer)
    summary = producer.get("reference_summary", {})
    generated = summary.get("generated_ids", [])
    result = summary.get("generation_result", {})
    if (reference.get("run") != REFERENCE_RUN or reference.get("head_sha") != REFERENCE_HEAD
            or large_artifact["id"] != REFERENCE_ARTIFACT_ID
            or large_artifact["zip_sha256"] != REFERENCE_ARTIFACT_SHA256
            or large_artifact["bytes"] != REFERENCE_ARTIFACT_BYTES
            or canonical != REFERENCE_PRODUCER_SHA256
            or review.get("producer_canonical_sha256") != canonical
            or review.get("original_reference_positive") is not True
            or review.get("explicit_hf_generation_and_forward_checked") is not True
            or producer.get("status") != "completed original OvisOCR3 HF CPU/F32 reference; native parity pending"
            or producer.get("checkpoint_sha256") != source.CHECKPOINT_SHA256
            or producer.get("metadata_producer_canonical_sha256") != METADATA_PRODUCER_SHA256
            or producer.get("stage_count", 0) < 888
            or not isinstance(generated, list) or len(generated) != 1
            or not isinstance(generated[0], list) or not generated[0]
            or result.get("eos_ids") != source.SOURCE_EOS_IDS
            or result.get("generated_token_count") != len(generated[0])
            or result.get("score_count") != len(generated[0])):
        raise ValueError("OvisOCR3 reviewed original HF forward/generation proof differs")
    return metadata, reference


def _expected_profile(base, metadata, reference, stop_proof_sha256):
    producer = metadata["producer"]
    return {**base,
            "qualification_kind": "hf5.16.1-original-reference",
            "processor": producer["processor"],
            "rendered_prompt": metadata["rendered_prompt"],
            "rendered_prompt_qualified": True,
            "processor_qualified": True,
            "hf_stop_policy_qualified": True,
            "hf_stop_runtime_proof_sha256": stop_proof_sha256,
            "stop_qualification_scope": STOP_SCOPE,
            "effective_stops_qualified": False,
            "stop_ids": source.SOURCE_EOS_IDS.copy(),
            "primary_eos": producer["tokenizer_eos"],
            "metadata_producer_canonical_sha256": METADATA_PRODUCER_SHA256,
            "original_reference_producer_canonical_sha256": REFERENCE_PRODUCER_SHA256,
            "source_profile": base,
            "proof_marker": "ovisocr3-hf-reference:" + REFERENCE_PRODUCER_SHA256}


def load_profile(model_dir, source_model, source_revision):
    metadata, reference = _reviewed_receipts()  # Before source/weight readers.
    stop_proof_sha256 = _checked_stop_proof()
    base = source.load_profile(model_dir, source_model, source_revision)
    if ({name: source._file_sha256(Path(model_dir) / name) for name in source.metadata_files()}
            != metadata["producer"]["source_files_sha256"]):
        raise ValueError("OvisOCR3 original metadata source differs")
    if (source._canonical_sha256(base["generation"]) != source._canonical_sha256(metadata["producer"]["resolved_generation"])
            or source._file_sha256(Path(model_dir) / "generation_config.json") != source.GENERATION_SHA256):
        raise ValueError("OvisOCR3 explicit generation source differs")
    rendered = metadata["rendered_prompt"]
    if (not isinstance(rendered, str) or hashlib.sha256(rendered.encode()).hexdigest() !=
            metadata["producer"]["rendered_prompt_sha256"]):
        raise ValueError("OvisOCR3 rendered original prompt differs")
    return _expected_profile(base, metadata, reference, stop_proof_sha256)


def _check_profile(profile):
    metadata, reference = _reviewed_receipts()
    if not isinstance(profile, dict) or not isinstance(profile.get("source_profile"), dict):
        raise ValueError("OvisOCR3 reviewed HF export profile required")
    base = profile["source_profile"]
    source._check_profile(base)
    if profile != _expected_profile(base, metadata, reference, _checked_stop_proof()):
        raise ValueError("OvisOCR3 reviewed HF export profile differs")


def validate_config(config, profile):
    _check_profile(profile)
    source.validate_config(config, profile["source_profile"])


def verify_checkpoint(path, profile):
    _check_profile(profile)
    return source.verify_checkpoint(path, profile["source_profile"])
