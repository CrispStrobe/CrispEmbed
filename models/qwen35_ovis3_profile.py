"""Pinned OvisOCR3 source-only export policy; no model imports or runtime claim."""

import hashlib
import json
from pathlib import Path


MODEL = "ATH-MaaS/OvisOCR3"
REVISION = "5f6ce6932efee9ec10c30746db55bd780214b7fd"
SOURCE_REPORT_SHA256 = "8ca73ca6a349b67a7533c36de6c14ac066e1152f2c8d517af849b2655087a331"
OVIS2_PRODUCER_SHA256 = "233dc37fdaa2ff6b4a8b3ca3832b578305b5f3b6ed4d0d88d54c73591626b372"
CONFIG_SHA256 = "aeac1785205ad86285c73072f2de8281a58bb024255ab98703ce7bab7ac54f30"
GENERATION_SHA256 = "c3ff3171724248f377efaf37197cce6f7bf1f4918a1ed3f31ef2ce53be91ecc1"
CHECKPOINT_SHA256 = "aa2311c6b22b07cf3afe9aac6e8bcb02a2b19dd8671b2a91d0b144071577e600"
CHECKPOINT_BYTES = 1706030528
SOURCE_EOS_IDS = [248044, 248046]
_MARKER = "ovisocr3-source-only:" + SOURCE_REPORT_SHA256
_REPORT = Path(__file__).resolve().parents[1] / "tests/results/qwen35-ocr/ovisocr3-source-contract.json"
_OVIS2 = Path(__file__).resolve().parents[1] / "tests/results/qwen35-ocr/ovisocr2-metadata.json"
_SHARED = frozenset({"LICENSE", "preprocessor_config.json", "tokenizer_config.json",
                     "chat_template.jinja", "tokenizer.json", "merges.txt", "vocab.json",
                     "video_preprocessor_config.json"})


def _canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                      ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _evidence():
    if _file_sha256(_REPORT) != SOURCE_REPORT_SHA256:
        raise ValueError("OvisOCR3 source audit bytes changed")
    report = json.loads(_REPORT.read_text())
    if (report["repository"]["model"] != MODEL or report["repository"]["revision"] != REVISION
            or report["config"]["canonical_sha256"] != CONFIG_SHA256
            or report["config"]["mtp_num_hidden_layers"] != 0
            or report["generation_config"]["sha256"] != GENERATION_SHA256
            or report["generation_config"]["eos_ids_declared"] != SOURCE_EOS_IDS
            or report["generation_config"]["effective_vllm_stop_set_unobserved"] is not True
            or report["checkpoint_header"]["lfs_sha256_declaration"] != CHECKPOINT_SHA256
            or report["checkpoint_header"]["file_bytes_declaration"] != CHECKPOINT_BYTES
            or report["checkpoint_header"]["tensor_count"] != 473
            or report["checkpoint_header"]["mtp_named_tensor_count"] != 0
            or report["processor_config"]["processor_config_json_present"] is not False
            or report["original_example"]["plain_ocr_prompt_sha256"] != report["prompt_files"]["prompts/plain_ocr.txt"]["sha256"]):
        raise ValueError("OvisOCR3 pinned source audit differs")
    ovis2 = json.loads(_OVIS2.read_text())
    if _canonical_sha256(ovis2["producer"]) != OVIS2_PRODUCER_SHA256:
        raise ValueError("Shared OvisOCR2 asset evidence changed")
    shared_hashes = ovis2["producer"]["checkpoint_files_sha256"]
    if set(shared_hashes) & _SHARED != _SHARED:
        raise ValueError("Shared Ovis asset hashes incomplete")
    return report, {name: shared_hashes[name] for name in _SHARED}, {
        "model": ovis2["producer"]["loaded_source_sha256"]["model"]}


def metadata_files():
    report, _, _ = _evidence()
    return tuple(sorted(set(report["source_files"]) | set(report["prompt_files"]) | _SHARED))


def _expected_profile(report, source_hashes, generation, task_prompt):
    return {
        "name": "ovisocr3", "format_version": 3,
        "source_model": MODEL, "source_revision": REVISION,
        "generation": generation, "source_eos_ids": SOURCE_EOS_IDS.copy(),
        "tokenizer_eos": 248046,
        "task": "plain_ocr", "task_prompt": task_prompt,
        "rendered_prompt_qualified": False, "effective_stops_qualified": False,
        "processor_qualified": False,
        "missing_source_files": ["processor_config.json"],
        "checkpoint_sha256": CHECKPOINT_SHA256, "checkpoint_bytes": CHECKPOINT_BYTES,
        "image_min_pixels": 448 * 448, "image_max_pixels": 2880 * 2880,
        "image_max_qualified": False, "native_alias_qualified": False,
        "qualification": {"source_report_sha256": SOURCE_REPORT_SHA256,
                          "transformers_version": "5.16.1",
                          "loaded_source_sha256": source_hashes,
                          "scope": "source-only-schema-export"},
        "proof_marker": _MARKER,
    }


def _check_generation(generation):
    if (_canonical_sha256(generation) != "60ee73ccecaa7d087b0e18e8673b26fb3d4181e19476f3445331795a8873a3b8"
            or generation.get("eos_token_id") != SOURCE_EOS_IDS
            or generation.get("do_sample") is not False):
        raise ValueError("OvisOCR3 source generation config differs")


def load_profile(model_dir, source_model, source_revision):
    """Verify all source metadata; checkpoint payload is checked separately."""
    if source_model != MODEL or source_revision != REVISION:
        raise ValueError("OvisOCR3 source identity differs")
    report, shared, source_hashes = _evidence()
    root = Path(model_dir)
    if (root / "processor_config.json").exists() or (root / "model.safetensors.index.json").exists():
        raise ValueError("OvisOCR3 source metadata/index differs")
    hashes = {name: entry["sha256"] for name, entry in report["source_files"].items()}
    hashes.update({name: entry["sha256"] for name, entry in report["prompt_files"].items()})
    for name, expected in shared.items():
        if name in hashes and hashes[name] != expected:
            raise ValueError("Shared Ovis asset audit differs: " + name)
        hashes[name] = expected
    for name, expected in hashes.items():
        if _file_sha256(root / name) != expected:
            raise ValueError("OvisOCR3 metadata source bytes changed: " + name)
    config = json.loads((root / "config.json").read_text())
    if _canonical_sha256(config) != CONFIG_SHA256:
        raise ValueError("OvisOCR3 canonical config differs")
    generation = json.loads((root / "generation_config.json").read_text())
    _check_generation(generation)
    task_prompt = (root / "prompts/plain_ocr.txt").read_text()
    return _expected_profile(report, source_hashes, generation, task_prompt)


def _check_profile(profile):
    report, _, source_hashes = _evidence()
    if not isinstance(profile, dict) or not isinstance(profile.get("generation"), dict):
        raise ValueError("OvisOCR3 pinned profile required")
    _check_generation(profile["generation"])
    prompt = profile.get("task_prompt")
    if not isinstance(prompt, str) or hashlib.sha256(prompt.encode()).hexdigest() != report["prompt_files"]["prompts/plain_ocr.txt"]["sha256"]:
        raise ValueError("OvisOCR3 plain OCR task prompt differs")
    if profile != _expected_profile(report, source_hashes, profile["generation"], prompt):
        raise ValueError("OvisOCR3 source-only profile differs")


def validate_config(config, profile):
    _check_profile(profile)
    if (_canonical_sha256(config) != CONFIG_SHA256
            or config["text_config"]["mtp_num_hidden_layers"] != 0
            or config["text_config"]["mtp_use_dedicated_embeddings"] is not False):
        raise ValueError("OvisOCR3 pinned dense config differs")


def verify_checkpoint(path, profile):
    _check_profile(profile)
    if Path(path).stat().st_size != CHECKPOINT_BYTES or _file_sha256(path) != CHECKPOINT_SHA256:
        raise ValueError("OvisOCR3 checkpoint payload differs")
    return CHECKPOINT_SHA256
