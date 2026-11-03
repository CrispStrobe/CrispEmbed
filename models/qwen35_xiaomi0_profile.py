"""Exact Xiaomi-OCR-0 HF profile. Source audit is not numerical qualification."""

import hashlib
import json
from pathlib import Path

MODEL = "xiaomi-research/Xiaomi-OCR-0"
REVISION = "bbd8cf8d21f256b3644b44cb860f3a7cecb93c30"
AUDIT_SHA256 = "10956c0dfc2244b6a082ee0acaeb40de7b885d46ddfa67a5bca973556d32855f"
CHECKPOINT_SHA256 = "d4551e9f6127ac9e35cff7b4f7a72a90410e03e6c8676a36f0e2b7eaeb5d5348"
CHECKPOINT_BYTES = 1746937856
REPORT = Path(__file__).resolve().parents[1] / "tests/results/qwen35-ocr/xiaomi0-source-contract.json"
MISSING_FILES = ["generation_config.json", "preprocessor_config.json"]
MAIN_SCHEMA_SHA256 = "589578cf5234bba4a1ee2f1c20c1d660bdaf214403a3f54c50fa504348cac71f"
MAIN_SCHEMA = REPORT.parents[2] / "fixtures/xiaomi0-source/main-tensor-schema.json"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evidence():
    if file_sha256(REPORT) != AUDIT_SHA256:
        raise ValueError("Xiaomi source audit bytes changed")
    report = json.loads(REPORT.read_text())
    if (report["model"] != MODEL or report["revision"] != REVISION
            or report["checkpoint"]["sha256"] != CHECKPOINT_SHA256
            or report["checkpoint"]["bytes"] != CHECKPOINT_BYTES
            or report["checkpoint"]["tensor_count"] != 488
            or report["checkpoint"]["main_tensor_count"] != 473
            or len(report["checkpoint"]["mtp_exclusions"]) != 15
            or any(report["qualification"].values())):
        raise ValueError("Xiaomi pinned source contract differs")
    return report


def metadata_files():
    return tuple(sorted(evidence()["source_files"]))


def expected_profile():
    report = evidence()
    return {"name": "xiaomi-ocr-0", "format_version": 4,
            "source_model": MODEL, "source_revision": REVISION,
            "generation": {"eos_token_id": 248046, "pad_token_id": 248044, "do_sample": False},
            "primary_eos": 248046, "stop_ids": [248046],
            "missing_source_files": MISSING_FILES.copy(),
            "checkpoint_sha256": CHECKPOINT_SHA256, "checkpoint_bytes": CHECKPOINT_BYTES,
            "image_min_pixels": 65536, "image_max_pixels": 16777216,
            "native_patch_limit": 1024, "native_context_limit": 4096,
            "native_alias_qualified": False, "image_max_qualified": False,
            "full_document_qualified": False, "mtp_exported": False,
            "prompts": report["prompts"], "source_audit_sha256": AUDIT_SHA256,
            "qualification": {"scope": "experimental-hf-image-first-source-profile",
                              "transformers_version": "5.17.0",
                              "loaded_source_sha256": {"model": report["framework"]["source_files"][
                                  "models/qwen3_5/modeling_qwen3_5.py"]}}}


def check_profile(profile):
    if profile != expected_profile():
        raise ValueError("Xiaomi exact experimental profile required")


def load_profile(model_dir, source_model, source_revision):
    if source_model != MODEL or source_revision != REVISION:
        raise ValueError("Xiaomi source identity differs")
    root = Path(model_dir)
    if any((root / name).exists() for name in MISSING_FILES + ["model.safetensors.index.json"]):
        raise ValueError("Xiaomi absent metadata/index must not be fabricated")
    for name, record in evidence()["source_files"].items():
        if file_sha256(root / name) != record["sha256"]:
            raise ValueError("Xiaomi source bytes changed: " + name)
    return expected_profile()


def validate_config(config, profile):
    check_profile(profile)
    actual = hashlib.sha256((json.dumps(config, sort_keys=True, separators=(",", ":"))
                             .encode())).hexdigest()
    fixture = Path(__file__).resolve().parents[1] / "tests/fixtures/xiaomi0-source/config.json"
    original = json.loads(fixture.read_text())
    expected = hashlib.sha256(json.dumps(original, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if file_sha256(fixture) != evidence()["source_files"]["config.json"]["sha256"] or actual != expected:
        raise ValueError("Xiaomi dense config differs")


def verify_checkpoint(path, profile):
    check_profile(profile)
    if Path(path).stat().st_size != CHECKPOINT_BYTES or file_sha256(path) != CHECKPOINT_SHA256:
        raise ValueError("Xiaomi checkpoint payload differs")
    return CHECKPOINT_SHA256


def validate_main_schema(actual):
    """Compare every original-model parameter shape, including its tied head."""
    if file_sha256(MAIN_SCHEMA) != MAIN_SCHEMA_SHA256:
        raise ValueError("Xiaomi original checkpoint tensor-schema bytes changed")
    record = json.loads(MAIN_SCHEMA.read_text())
    if (record["model"] != MODEL or record["revision"] != REVISION
            or record["checkpoint_sha256"] != CHECKPOINT_SHA256
            or record["header_sha256"] != evidence()["checkpoint"]["header_sha256"]
            or len(record["tensors"]) != 473):
        raise ValueError("Xiaomi original checkpoint tensor-schema identity differs")
    expected = dict(record["tensors"])
    expected["lm_head.weight"] = expected["model.language_model.embed_tokens.weight"]
    if actual != expected:
        raise ValueError("Xiaomi full original-model tensor schema differs")
    return MAIN_SCHEMA_SHA256


def non_speculative_tensors(actual, profile):
    """Match the original HF class's explicit MTP ignore policy, never a wildcard export drop."""
    check_profile(profile)
    excluded = evidence()["checkpoint"]["mtp_exclusions"]
    if len(actual) != 488 or {name for name in actual if name.startswith("mtp.")} != set(excluded):
        raise ValueError("Xiaomi exact 488-tensor / 15-MTP schema differs")
    for name, record in excluded.items():
        if actual[name][1] != record["shape"]:
            raise ValueError("Xiaomi ignored MTP tensor shape differs: " + name)
    return {name: value for name, value in actual.items() if name not in excluded}


def rendered_prompt(task="document", *, order="image-first"):
    prompts = evidence()["prompts"]
    if task not in prompts or order not in ("image-first", "text-first"):
        raise ValueError("Xiaomi task or message order is outside the pinned profile")
    image = "<|vision_start|><|image_pad|><|vision_end|>"
    content = image + prompts[task] if order == "image-first" else prompts[task] + image
    return "<|im_start|>user\n" + content + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def validate_runtime_metadata(model_dir):
    """Run on a hosted producer before weights; reject tokenizer/backend/source drift."""
    import inspect
    import transformers
    from transformers import AutoConfig, AutoProcessor, GenerationConfig, Qwen3_5ForConditionalGeneration
    from transformers.tokenization_utils_tokenizers import TokenizersBackend
    from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer

    report = evidence()
    if transformers.__version__ != report["framework"]["transformers"]:
        raise ValueError("Xiaomi requires its pinned Transformers 5.17.0 blueprint")
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    classes = {"models/qwen3_5/modeling_qwen3_5.py": Qwen3_5ForConditionalGeneration,
               "models/qwen3_vl/processing_qwen3_vl.py": type(processor),
               "models/qwen2_vl/image_processing_qwen2_vl.py": type(processor.image_processor),
               "models/qwen2/tokenization_qwen2.py": Qwen2Tokenizer,
               "tokenization_utils_tokenizers.py": TokenizersBackend,
               "generation/configuration_utils.py": GenerationConfig}
    for name, cls in classes.items():
        if file_sha256(inspect.getfile(cls)) != report["framework"]["source_files"][name]:
            raise ValueError("Xiaomi loaded framework source differs: " + name)
    effective = json.loads(processor.tokenizer.backend_tokenizer.to_str())
    pattern = effective["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
    if effective["normalizer"] != {"type": "NFC"} or pattern != report["tokenizer"]["serialized_regex"]:
        raise ValueError("Xiaomi effective tokenizer differs; config/serialized regex discrepancy is not waived")
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    generation = GenerationConfig.from_model_config(config)
    if generation.eos_token_id != 248046 or generation.pad_token_id != 248044 or generation.do_sample:
        raise ValueError("Xiaomi original generation defaults differ")
    if processor.tokenizer.eos_token_id != 248046 or processor.tokenizer.pad_token_id != 248044:
        raise ValueError("Xiaomi tokenizer stop/pad differs")
    image = processor.image_processor
    if (image.size != {"shortest_edge": 65536, "longest_edge": 16777216}
            or image.patch_size != 16 or image.temporal_patch_size != 2 or image.merge_size != 2
            or image.image_mean != [0.5] * 3 or image.image_std != [0.5] * 3
            or not all((image.do_resize, image.do_rescale, image.do_normalize, image.do_convert_rgb))
            or int(image.resample) != 3):
        raise ValueError("Xiaomi original image processor differs")
    rendered = {}
    for task, prompt in report["prompts"].items():
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        rendered[task] = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if rendered[task] != rendered_prompt(task):
            raise ValueError("Xiaomi rendered prompt differs: " + task)
    return {"transformers": transformers.__version__, "effective_regex": pattern,
            "eos_token_id": generation.eos_token_id, "pad_token_id": generation.pad_token_id,
            "rendered_prompts": rendered, "mtp_policy": "exact15-excluded-nonspeculative",
            "numerical_qualification": False, "document_qualification": False}
