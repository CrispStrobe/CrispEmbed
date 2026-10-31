"""Pinned, evidence-driven Xiaomi-OCR-0 export policy.

Nothing here imports model code. The profile refuses to load unless an
independently produced original-capture evidence JSON exists for the pinned
revision, so the exporter cannot silently accept a different checkpoint,
prompt, stop policy or tensor census.

Environment:
  CRISPEMBED_XIAOMI_EVIDENCE   path to the original-capture evidence JSON
                               (default: tests/results/xiaomi-ocr-0/kaggle-v3-evidence.json)

The evidence JSON is produced by tools/kaggle/xiaomi-ocr-0-blueprint/blueprint.py
from the original Transformers execution. It is never shipped inside the GGUF
and never injected into native arithmetic. The captured attention control is
not native qualification.
"""

import hashlib
import json
import os
from pathlib import Path

MODEL = "xiaomi-research/Xiaomi-OCR-0"
REVISION = "bbd8cf8d21f256b3644b44cb860f3a7cecb93c30"
CHECKPOINT_BYTES = 1746937856
CHECKPOINT_SHA256 = "d4551e9f6127ac9e35cff7b4f7a72a90410e03e6c8676a36f0e2b7eaeb5d5348"
RENDERED_PROMPT_SHA256 = "4ff183b02e15704fca62adbfaafc97644cd3dc693cad720347bb0f37e25d78a3"
EVIDENCE_CANONICAL_SHA256 = "432c542fbd8a648b6fa5c6053cb096541242778deec275400bda5f472e4e3687"
MODEL_SOURCE_SHA256 = "641da34484f6b9769e28615c680cc4e3202afbf5bdbf1da7d56d5a3d678918c7"
# The three sources disagree: tokenizer eos 248046, text-config eos 248044,
# resolved generation eos 248046. The two completed greedy generations ended
# at 248046; the third hit its limit. None emitted 248044, which stays recorded
# as padding rather than a second generation stop.
REFERENCE_GENERATION_EOS = 248046
MAIN_TENSORS = 473
AUXILIARY_TENSORS = 15
_SOURCE_FILES = (
    "config.json",
    "processor_config.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "tokenizer.json",
    "README.md",
)
# Source files the checkpoint legitimately does not publish. Preserved as
# absence, never fabricated.
_MISSING_FILES = ["generation_config.json", "preprocessor_config.json"]
# Multi-token-prediction weights are an auxiliary speculative head. Ordinary
# greedy generation never consults them; the native graph has no such branch.
_EXCLUDE_PREFIXES = ["mtp."]


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def evidence_path():
    configured = os.environ.get("CRISPEMBED_XIAOMI_EVIDENCE", "")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[1] / "tests/results/xiaomi-ocr-0/kaggle-v3-evidence.json"


def _evidence():
    path = evidence_path()
    if not path.is_file():
        raise ValueError(f"Xiaomi original evidence is absent: {path.name}")
    data = json.loads(path.read_text())
    if data.get("status") != "completed original blueprint capture; no native parity claimed":
        raise ValueError("Xiaomi original capture must have completed")
    if data.get("rendered_prompt_sha256") != RENDERED_PROMPT_SHA256:
        raise ValueError("Xiaomi rendered prompt differs from the original capture")
    canonical_evidence = _canonical({k: v for k, v in data.items() if k != "evidence_canonical_sha256"})
    if (data.get("evidence_canonical_sha256") != EVIDENCE_CANONICAL_SHA256
            or canonical_evidence != EVIDENCE_CANONICAL_SHA256):
        raise ValueError("Xiaomi original evidence digest differs from the pinned capture")
    model_source = data.get("loaded_implementations", {}).get("model", {})
    alias = data.get("tied_head_observation", {})
    if (model_source.get("normalized_sha256") != MODEL_SOURCE_SHA256
            or model_source.get("class") != "transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5ForConditionalGeneration"
            or not all(alias.get(key) is True for key in ("same_parameter", "same_storage", "values_equal"))
            or data.get("original_prefill", {}).get("status") != "complete"):
        raise ValueError("Xiaomi loaded implementation, original alias, or prefill differs")
    if (data.get("source_model") != MODEL or data.get("source_revision") != REVISION
            or data.get("checkpoint_bytes") != CHECKPOINT_BYTES
            or data.get("checkpoint_sha256") != CHECKPOINT_SHA256
            or data.get("main_tensor_count") != MAIN_TENSORS
            or data.get("auxiliary_tensor_count") != AUXILIARY_TENSORS
            or data.get("transformers_version") != "5.8.1"
            or not isinstance(data.get("rendered_prompt"), str)
            or not data["rendered_prompt"]
            or data.get("rendered_prompt_sha256")
            != hashlib.sha256(data["rendered_prompt"].encode()).hexdigest()
            or data.get("task") != "document"
            or not isinstance(data.get("stop_ids"), list) or not data["stop_ids"]):
        raise ValueError("Xiaomi original evidence does not match the pinned contract")
    if (not isinstance(data.get("config_canonical_sha256"), str)
            or len(data["config_canonical_sha256"]) != 64):
        raise ValueError("Xiaomi evidence lacks the canonical config digest")
    if not all(type(x) is int and 0 <= x < 248320 for x in data["stop_ids"]):
        raise ValueError("Xiaomi evidence stop ids are invalid")
    resolved = data.get("resolved_generation", {}).get("eos_token_id")
    if isinstance(resolved, list):
        resolved_ids = list(resolved)
    elif type(resolved) is int:
        resolved_ids = [resolved]
    else:
        raise ValueError("Xiaomi evidence lacks a resolved generation EOS")
    if (len(resolved_ids) != 1 or resolved_ids != [REFERENCE_GENERATION_EOS]
            or 248044 in resolved_ids):
        raise ValueError("Xiaomi generation EOS is not the observed single stop")
    return data


def generation_stop_ids(evidence):
    """Generation stops only: the resolved EOS, never the padding id."""
    resolved = evidence["resolved_generation"]["eos_token_id"]
    return list(resolved) if isinstance(resolved, list) else [resolved]


def metadata_files():
    """Exact allowlist of audited metadata files; weights are downloaded separately."""
    return _SOURCE_FILES


def load_profile(model_dir, source_model, source_revision):
    if source_model != MODEL or source_revision != REVISION:
        raise ValueError("Xiaomi-OCR-0 source identity differs")
    evidence = _evidence()
    root = Path(model_dir)
    if (root / "model.safetensors.index.json").exists():
        raise ValueError("Pinned Xiaomi checkpoint must not use a shard index")
    for filename in _MISSING_FILES:
        if (root / filename).exists():
            raise ValueError("Xiaomi missing source file was fabricated: " + filename)
    for filename in _SOURCE_FILES:
        if not (root / filename).is_file():
            raise ValueError("Xiaomi source file is missing: " + filename)
        expected_hash = evidence.get("checkpoint_files_sha256", {}).get(filename)
        if not expected_hash or _sha256_file(root / filename) != expected_hash:
            raise ValueError("Xiaomi source file digest differs: " + filename)
    config = json.loads((root / "config.json").read_text())
    if _canonical(config) != evidence["config_canonical_sha256"]:
        raise ValueError("Xiaomi canonical config differs from the audited evidence")
    return {
        "name": "xiaomi",
        "format_version": 4,
        "source_model": MODEL,
        "source_revision": REVISION,
        "task": "document",
        "rendered_prompt": evidence["rendered_prompt"],
        "generation": {"eos_token_id": 248046},
        "stop_ids": generation_stop_ids(evidence),
        "primary_eos": 248046,
        "missing_source_files": list(_MISSING_FILES),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "checkpoint_bytes": CHECKPOINT_BYTES,
        "exclude_prefixes": list(_EXCLUDE_PREFIXES),
        "main_tensors": MAIN_TENSORS,
        "auxiliary_tensors": AUXILIARY_TENSORS,
        "image_min_pixels": 65536,
        "image_max_pixels": 16777216,
        "image_max_qualified": False,
        "native_alias_qualified": False,
        "mtp_policy": "excluded-non-speculative-main-model-only",
        "qualification": {
            "transformers_version": evidence["transformers_version"],
            "loaded_source_sha256": {"model": evidence["loaded_implementations"]["model"]["normalized_sha256"]},
            "evidence_sha256": hashlib.sha256(evidence_path().read_bytes()).hexdigest(),
            "evidence_canonical_sha256": evidence["evidence_canonical_sha256"],
        },
    }


def _check(profile):
    if not isinstance(profile, dict):
        raise ValueError("Xiaomi profile is required")
    if (profile.get("name") != "xiaomi" or profile.get("format_version") != 4
            or profile.get("source_model") != MODEL or profile.get("source_revision") != REVISION
            or profile.get("checkpoint_sha256") != CHECKPOINT_SHA256
            or profile.get("checkpoint_bytes") != CHECKPOINT_BYTES
            or profile.get("stop_ids") != generation_stop_ids(_evidence())
            or profile.get("primary_eos") != 248046
            or profile.get("exclude_prefixes") != _EXCLUDE_PREFIXES
            or profile.get("main_tensors") != MAIN_TENSORS
            or profile.get("mtp_policy") != "excluded-non-speculative-main-model-only"
            or profile.get("native_alias_qualified") is not False
            or profile.get("image_max_qualified") is not False):
        raise ValueError("Xiaomi pinned profile contract differs")


def validate_config(config, profile):
    """Allow only the audited Xiaomi configuration, including its MTP head count."""
    _check(profile)
    text = config.get("text_config", {})
    if (_canonical(config) != _evidence()["config_canonical_sha256"]
            or config.get("model_type") != "qwen3_5"
            or text.get("mtp_num_hidden_layers") != 1
            or text.get("mtp_use_dedicated_embeddings") is not False
            or config.get("tie_word_embeddings") is not True
            or config.get("vision_config", {}).get("deepstack_visual_indexes") != []):
        raise ValueError("Xiaomi pinned config differs")


def verify_checkpoint(path, profile):
    _check(profile)
    path = Path(path)
    if path.stat().st_size != CHECKPOINT_BYTES:
        raise ValueError("Xiaomi checkpoint byte count differs")
    digest = _sha256_file(path)
    if digest != CHECKPOINT_SHA256:
        raise ValueError("Xiaomi checkpoint SHA256 differs")
    return digest


def excluded_prefixes(profile):
    """Prefixes the exporter must drop from schema comparison and output."""
    _check(profile)
    return tuple(profile["exclude_prefixes"])
