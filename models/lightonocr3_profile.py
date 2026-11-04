"""Pinned LightOnOCR-3-1B input contract (no model/runtime imports).

This is a candidate conversion profile, not a qualified catalog alias. Upstream
config, processor, tokenizer and weights are verified before conversion/loading.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

MANIFEST = Path(__file__).with_name("lightonocr3-1b-source.json")
PROFILE = "lightonocr3-1b"
MODEL_ID = "lightonai/LightOnOCR-3-1B"
REVISION = "23d544085c886bbdace647bcfab91203299417e2"
VOCAB_SIZE = 151936
TOKENIZER_VOCAB_SIZE = 151669
STOP_IDS = (151645, 151643)
SPECIAL_IDS = tuple(range(151643, 151657))
REFERENCE_PACKAGES = {"torch": "2.13.0+cpu", "torchvision": "0.28.0+cpu", "numpy": "2.3.5",
                      "pillow": "12.0.0", "gguf": "0.19.0", "safetensors": "0.8.0",
                      "huggingface-hub": "1.31.0", "tokenizers": "0.23.1", "jinja2": "3.1.6"}
PROMPT_IDS = {
    "transcription": ([151644, 8948, 151645, 198, 151644, 872, 198], [151645, 198, 151644, 77091, 198]),
    "grounding": ([151644, 8948, 151645, 198, 151644, 872, 198], [1951, 287, 198, 151645, 198, 151644, 77091, 198]),
}
TRANSCRIPTION = "<|im_start|>system<|im_end|>\n<|im_start|>user\n<|image_pad|><|im_end|>\n<|im_start|>assistant\n"
GROUNDING = "<|im_start|>system<|im_end|>\n<|im_start|>user\n<|image_pad|>grounding\n<|im_end|>\n<|im_start|>assistant\n"


def manifest():
    pins = json.loads(MANIFEST.read_text())
    if pins.get("model_id") != MODEL_ID or pins.get("revision") != REVISION or pins.get("profile") != PROFILE:
        raise ValueError("Source manifest is not the reviewed LightOnOCR-3-1B checkpoint")
    return pins


def trace_shapes(image_size, mode, steps):
    """Expected shapes for the actual one-image, batch-one original capture."""
    if (not isinstance(image_size, list) or len(image_size) != 2
            or any(type(v) is not int or not 0 < v <= 2**31 - 1 for v in image_size)
            or mode not in PROMPT_IDS or type(steps) is not int or not 1 <= steps <= 256):
        raise ValueError("Invalid bounded trace image/mode/token count")
    width, height = image_size
    ratio = max(height / 1540, width / 1540)
    if ratio > 1:
        height, width = math.floor(height / ratio), math.floor(width / ratio)
    height, width = ((height - 1) // 14 + 1) * 14, ((width - 1) // 14 + 1) * 14
    if height < 28 or width < 28:
        raise ValueError("Image has no 2x2 patch merge")
    patches, merged = (height // 14) * (width // 14), (height // 28) * (width // 28)
    prefix, suffix = PROMPT_IDS[mode]
    tokens = len(prefix) + merged + len(suffix)
    shapes = {"pixel_values": (1, 3, height, width), "prompt_ids": (tokens,),
              "patch_conv": (patches, 1024), "vis_ln_pre": (1, patches, 1024),
              "proj_out": (merged, 1024), "decoder_embed": (1, tokens, 1024),
              "llm_final_norm": (1, tokens, 1024), "generated_ids": (steps,)}
    shapes.update({f"vis_layer_{i}": (1, patches, 1024) for i in range(24)})
    shapes.update({f"llm_layer_{i}": (1, tokens, 1024) for i in range(28)})
    shapes.update({f"logits_step{i}": (1, VOCAB_SIZE) for i in range(steps)})
    return shapes, prefix + [151655] * merged + suffix


def validate_generation_ids(ids, cap):
    if (type(cap) is not int or not 1 <= cap <= 256 or not isinstance(ids, list)
            or not 1 <= len(ids) <= cap
            or any(type(v) is not int or not 0 <= v < VOCAB_SIZE for v in ids)
            or any(v in STOP_IDS for v in ids[:-1])
            or (ids[-1] not in STOP_IDS and len(ids) != cap)):
        raise ValueError("Invalid generated IDs or termination evidence")


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
        if result[mode] != PROMPT_IDS[mode]:
            raise ValueError(f"Unexpected {mode} token IDs")
    return result


def add_metadata(writer, directory, tokenizer):
    directory = Path(directory)
    pins = verify_source(directory)
    cfg = json.loads((directory / "config.json").read_text())
    proc = json.loads((directory / "processor_config.json").read_text())
    gen = json.loads((directory / "generation_config.json").read_text())
    validate_config(cfg, proc, gen)
    prompts = prompt_ids(tokenizer, (directory / "chat_template.jinja").read_text())
    if len(tokenizer) != TOKENIZER_VOCAB_SIZE or tuple(sorted(tokenizer.all_special_ids)) != SPECIAL_IDS:
        raise ValueError("Tokenizer vocabulary/special IDs differ from the reviewed source")
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
