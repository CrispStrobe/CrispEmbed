"""Pinned LightOnOCR-3-1B input contract (no model/runtime imports).

This is a candidate conversion profile, not a qualified catalog alias. Upstream
config, processor, tokenizer and weights are verified before conversion/loading.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

MANIFEST = Path(__file__).with_name("lightonocr3-1b-source.json")
PROFILE = "lightonocr3-1b"
TRANSCRIPTION = "<|im_start|>system<|im_end|>\n<|im_start|>user\n<|image_pad|><|im_end|>\n<|im_start|>assistant\n"
GROUNDING = "<|im_start|>system<|im_end|>\n<|im_start|>user\n<|image_pad|>grounding\n<|im_end|>\n<|im_start|>assistant\n"


def manifest():
    return json.loads(MANIFEST.read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source(directory, *, weights=True):
    directory = Path(directory)
    pins = manifest()
    for name, expected in pins["files"].items():
        if name == "model.safetensors" and not weights:
            continue
        path = directory / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Pinned {PROFILE} source mismatch: {name}")
    return pins


def source_directory(model, revision=None, cache_dir=None):
    """Download only the enumerated checkpoint, never floating main/custom code."""
    pins = manifest()
    if revision not in (None, pins["revision"]):
        raise ValueError("LightOnOCR-3-1B revision must match the source manifest")
    path = Path(model)
    if not path.is_dir():
        if model != pins["model_id"]:
            raise ValueError("LightOnOCR-3-1B profile requires its exact model ID or a verified local snapshot")
        from huggingface_hub import snapshot_download
        path = Path(snapshot_download(model, revision=pins["revision"],
                                     allow_patterns=list(pins["files"]), cache_dir=cache_dir))
    verify_source(path)
    return path


def validate_config(config, processor, generation):
    """Reject unsupported architecture/processor substitutions before writing GGUF."""
    vision = config["vision_config"]
    text = config["text_config"]
    required = {
        "model_type": "mistral3", "spatial_merge_size": 2,
        "image_token_id": 151655, "eos_token_id": 151645, "pad_token_id": 151643,
        "multimodal_projector_bias": False, "projector_hidden_act": "gelu",
        "vision_feature_layer": -1,
    }
    vrequired = {"model_type": "pixtral", "hidden_size": 1024, "head_dim": 64,
                 "num_attention_heads": 16, "num_hidden_layers": 24,
                 "intermediate_size": 4096, "patch_size": 14, "image_size": 1540,
                 "hidden_act": "silu", "num_channels": 3}
    trequired = {"model_type": "qwen3", "hidden_size": 1024, "head_dim": 128,
                 "num_attention_heads": 16, "num_key_value_heads": 8,
                 "num_hidden_layers": 28, "intermediate_size": 3072,
                 "hidden_act": "silu", "vocab_size": 151936,
                 "tie_word_embeddings": True, "use_qk_norm": True,
                 "attention_bias": False, "use_sliding_window": False,
                 "rms_norm_eps": 1e-6, "max_position_embeddings": 16384}
    for obj, fields in ((config, required), (vision, vrequired), (text, trequired)):
        for key, value in fields.items():
            if obj.get(key) != value:
                raise ValueError(f"Unsupported {PROFILE} config field {key}: {obj.get(key)!r}")
    if text.get("layer_types") != ["full_attention"] * 28:
        raise ValueError("LightOnOCR-3-1B requires 28 full-attention text layers")
    for obj, theta in ((text, 1000000), (vision, 10000)):
        if obj.get("rope_theta") != theta or obj.get("rope_parameters") != {
            "rope_theta": theta, "rope_type": "default"
        }:
            raise ValueError("Unsupported rotary embedding configuration")
    ip = processor["image_processor"]
    expected = {"resample": 3, "patch_size": 14, "size": {"longest_edge": 1540},
                "do_resize": True, "do_convert_rgb": True, "do_rescale": True,
                "do_normalize": True, "rescale_factor": 1 / 255,
                "image_mean": [0.48145466, 0.4578275, 0.40821073],
                "image_std": [0.26862954, 0.26130258, 0.27577711],
                "image_processor_type": "PixtralImageProcessorFast"}
    for key, value in expected.items():
        if ip.get(key) != value:
            raise ValueError(f"Unsupported image processor field: {key}")
    if processor.get("patch_size") != 14 or processor.get("spatial_merge_size") != 2:
        raise ValueError("Unsupported processor spatial merge")
    if generation.get("eos_token_id") != [151645, 151643]:
        raise ValueError("Unexpected generation stop IDs")


def prompt_ids(tokenizer, template):
    """Use the checkpoint's actual Jinja and tokenizer, validating the image slot."""
    result = {}
    for mode, prompt in (("transcription", TRANSCRIPTION), ("grounding", GROUNDING)):
        content = [{"type": "image"}]
        if mode == "grounding":
            content.append({"type": "text", "text": "grounding"})
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], chat_template=template,
            tokenize=False, add_generation_prompt=True,
        )
        if rendered != prompt:
            raise ValueError(f"Unexpected {mode} chat-template rendering: {rendered!r}")
        ids = tokenizer.encode(rendered, add_special_tokens=False)
        if ids.count(151655) != 1:
            raise ValueError("Prompt must contain exactly one image placeholder")
        pos = ids.index(151655)
        result[mode] = (ids[:pos], ids[pos + 1:])
    return result


def add_metadata(writer, directory, tokenizer):
    directory = Path(directory)
    pins = verify_source(directory)
    cfg = json.loads((directory / "config.json").read_text())
    proc = json.loads((directory / "processor_config.json").read_text())
    gen = json.loads((directory / "generation_config.json").read_text())
    validate_config(cfg, proc, gen)
    prompts = prompt_ids(tokenizer, (directory / "chat_template.jinja").read_text())
    writer.add_string("lightonocr.profile", PROFILE)
    writer.add_uint32("lightonocr.profile_version", 1)
    writer.add_string("general.name", "LightOnOCR-3-1B (unqualified candidate)")
    writer.add_string("general.license", "apache-2.0")
    writer.add_string("lightonocr.source.model", pins["model_id"])
    writer.add_string("lightonocr.source.revision", pins["revision"])
    writer.add_string("lightonocr.source.manifest", json.dumps(pins, sort_keys=True))
    writer.add_string("lightonocr.processor.resize", "torch-bicubic-u8-antialias")
    writer.add_uint32("lightonocr.processor.patch_grid", 14)
    writer.add_array("lightonocr.stop_token_ids", gen["eos_token_id"])
    writer.add_array("lightonocr.special_token_ids", sorted(tokenizer.all_special_ids))
    writer.add_uint32("lightonocr.tokenizer.actual_vocab_size", len(tokenizer))
    for mode, (prefix, suffix) in prompts.items():
        writer.add_array(f"lightonocr.prompt.{mode}.prefix", prefix)
        writer.add_array(f"lightonocr.prompt.{mode}.suffix", suffix)
    writer.add_string("lightonocr.grounding.format", "markdown-label-xyxy-1000")
    writer.add_uint32("lightonocr.text.max_position_embeddings", cfg["text_config"]["max_position_embeddings"])
