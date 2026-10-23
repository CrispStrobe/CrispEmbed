"""Pinned, metadata-only OvisOCR2 export policy. No model code is imported here."""

import hashlib
import json
from pathlib import Path


MODEL = "ATH-MaaS/OvisOCR2"
REVISION = "cbdede82e048659b83310476ff807da825f8fe78"
PRODUCER_SHA256 = "233dc37fdaa2ff6b4a8b3ca3832b578305b5f3b6ed4d0d88d54c73591626b372"
CONFIG_SHA256 = "7755fa7f94c366b1bc8db482e9a3c56721093ece9ed6a8234560e443b13168d1"
CHECKPOINT_SHA256 = "9270560288656ece5cb3a6989001afcf5af8d223bceed4a423c33a008861d009"
CHECKPOINT_BYTES = 1706030496
RUN_ID = 39983101540
_PROOF_MARKER = "ovisocr2-reviewed-producer:" + PRODUCER_SHA256
_EVIDENCE = Path(__file__).resolve().parents[1] / "tests/results/qwen35-ocr/ovisocr2-metadata.json"
_SOURCE_FILES = frozenset({
    "LICENSE", "README.md", "config.json", "preprocessor_config.json",
    "tokenizer_config.json", "chat_template.jinja", "tokenizer.json",
    "merges.txt", "vocab.json", "video_preprocessor_config.json",
})
_MISSING_FILES = ["processor_config.json", "generation_config.json"]
_STOPS = [248044, 248046]


def _canonical_sha256(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_evidence():
    evidence = json.loads(_EVIDENCE.read_text())
    producer = evidence["producer"]
    if _canonical_sha256(producer) != PRODUCER_SHA256:
        raise ValueError("Reviewed OvisOCR2 producer proof changed")
    prompt=evidence.get('rendered_prompt')
    if not isinstance(prompt,str) or hashlib.sha256(prompt.encode()).hexdigest()!=producer['rendered_prompt_sha256']:
        raise ValueError('OvisOCR2 original single-image prompt differs')
    review = evidence["independent_review"]
    if (evidence["run"] != RUN_ID or review["producer_canonical_sha256"] != PRODUCER_SHA256
            or producer["phase"] != "complete" or not producer["status"].startswith("completed weight-free")
            or producer["schema"]["checkpoint_tensors"] != 473
            or producer["schema"]["expected_tensors"] != 474
            or producer["schema"]["tied_head_absent"] is not True
            or any("mtp" in name.lower() for name in producer["schema"]["official_state_dict"])
            or any(review[key] is not True for key in (
                "schema_all_473_shapes_exact", "missing_only_tied_head",
                "loaded_processor_image_tokenizer_match_upstream_tag",
                "all_three_rgb_id_grid_and_packed_field_hashes_checked",
                "default_counterfactual_fields_checked",
                "image_placeholder_counts_match_merge_geometry"))):
        raise ValueError("OvisOCR2 reviewed metadata qualification differs")
    stops = producer["effective_stops_source_probe"]
    if (producer["resolved_generation"]["eos_token_id"] != 248044
            or producer["tokenizer_eos"] != 248046
            or stops["primary_eos"] != 248046
            or stops["all_stop_ids"] != _STOPS
            or stops["additional_stop_ids"] != [248044]):
        raise ValueError("OvisOCR2 reviewed stop contract differs")
    declaration = evidence["checkpoint_lfs_declaration"]
    payload = declaration["file"]
    if (declaration["revision"] != REVISION or payload["rfilename"] != "model.safetensors"
            or payload["size"] != CHECKPOINT_BYTES or payload["lfs"]["size"] != CHECKPOINT_BYTES
            or payload["lfs"]["sha256"] != CHECKPOINT_SHA256):
        raise ValueError("OvisOCR2 checkpoint declaration differs")
    hashes = producer["checkpoint_files_sha256"]
    if set(hashes) != _SOURCE_FILES or any(len(value) != 64 for value in hashes.values()):
        raise ValueError("OvisOCR2 metadata file set differs")
    return producer


def metadata_files():
    """Exact audited metadata allowlist; includes prompt/attribution source."""
    return tuple(sorted(_SOURCE_FILES))


def load_profile(model_dir, source_model, source_revision):
    """Verify pinned source metadata and return an export-only profile."""
    if source_model != MODEL or source_revision != REVISION:
        raise ValueError("OvisOCR2 source identity differs")
    producer = _verified_evidence()
    root = Path(model_dir)
    if (root / "model.safetensors.index.json").exists():
        raise ValueError("Pinned OvisOCR2 checkpoint must not use a shard index")
    if any((root / filename).exists() for filename in _MISSING_FILES):
        raise ValueError("OvisOCR2 source-missing metadata was fabricated")
    for filename, expected in producer["checkpoint_files_sha256"].items():
        if _file_sha256(root / filename) != expected:
            raise ValueError("OvisOCR2 metadata source bytes changed: " + filename)
    config = json.loads((root / "config.json").read_text())
    if _canonical_sha256(config) != CONFIG_SHA256:
        raise ValueError("OvisOCR2 canonical config differs")
    return {
        "name": "ovisocr2", "format_version": 2,
        "source_model": MODEL, "source_revision": REVISION,
        "generation": producer["resolved_generation"],
        "processor": producer["processor"],
        "primary_eos": 248046, "stop_ids": _STOPS.copy(),
        "rendered_prompt":json.loads(_EVIDENCE.read_text())['rendered_prompt'],
        "missing_source_files": _MISSING_FILES.copy(),
        "checkpoint_sha256": CHECKPOINT_SHA256, "checkpoint_bytes": CHECKPOINT_BYTES,
        "image_min_pixels": 448 * 448, "image_max_pixels": 2880 * 2880,
        "image_max_qualified": False, "mtp_policy": "pinned-non-speculative-main-model-only",
        "native_alias_qualified": False,
        "qualification": {"run_id": RUN_ID, "source_commit": producer["source_commit"],
                          "producer_canonical_sha256": PRODUCER_SHA256,
                          "transformers_version": "5.16.1",
                          "loaded_source_sha256": producer["loaded_source_sha256"]},
        "proof_marker": _PROOF_MARKER,
    }


def _check_profile(profile):
    if not isinstance(profile, dict) or not isinstance(profile.get("qualification"), dict):
        raise ValueError("OvisOCR2 profile is required")
    if (profile.get("name") != "ovisocr2" or profile.get("format_version") != 2
            or profile.get("source_model") != MODEL or profile.get("source_revision") != REVISION
            or profile.get("proof_marker") != _PROOF_MARKER
            or profile.get("primary_eos") != 248046 or profile.get("stop_ids") != _STOPS
            or profile.get("missing_source_files") != _MISSING_FILES
            or profile.get("checkpoint_sha256") != CHECKPOINT_SHA256
            or profile.get("checkpoint_bytes") != CHECKPOINT_BYTES
            or profile.get("image_min_pixels") != 448 * 448
            or profile.get("image_max_pixels") != 2880 * 2880
            or profile.get("image_max_qualified") is not False
            or profile.get("mtp_policy") != "pinned-non-speculative-main-model-only"
            or profile.get("native_alias_qualified") is not False
            or profile["qualification"].get("producer_canonical_sha256") != PRODUCER_SHA256):
        raise ValueError("OvisOCR2 pinned profile proof or contract differs")
    producer = _verified_evidence()
    prompt=profile.get('rendered_prompt')
    if not isinstance(prompt,str) or hashlib.sha256(prompt.encode()).hexdigest()!=producer['rendered_prompt_sha256']:
        raise ValueError('OvisOCR2 profile prompt differs')
    if (profile.get("generation") != producer["resolved_generation"]
            or profile.get("processor") != producer["processor"]
            or profile["qualification"].get("run_id") != RUN_ID
            or profile["qualification"].get("source_commit") != producer["source_commit"]
            or profile["qualification"].get("transformers_version") != "5.16.1"
            or profile["qualification"].get("loaded_source_sha256") != producer["loaded_source_sha256"]):
        raise ValueError("OvisOCR2 profile differs from reviewed metadata")


def validate_config(config, profile):
    """Allow only the audited non-speculative MTP=1 source configuration."""
    _check_profile(profile)
    if (_canonical_sha256(config) != CONFIG_SHA256
            or config["model_type"] != "qwen3_5"
            or config["text_config"]["mtp_num_hidden_layers"] != 1
            or config["text_config"]["mtp_use_dedicated_embeddings"] is not False):
        raise ValueError("OvisOCR2 pinned non-speculative config differs")


def verify_checkpoint(path, profile):
    """Check complete pinned payload before any tensor reader opens it."""
    _check_profile(profile)
    path = Path(path)
    if path.stat().st_size != CHECKPOINT_BYTES:
        raise ValueError("OvisOCR2 checkpoint byte count differs")
    digest = _file_sha256(path)
    if digest != CHECKPOINT_SHA256:
        raise ValueError("OvisOCR2 checkpoint SHA256 differs")
    return digest
