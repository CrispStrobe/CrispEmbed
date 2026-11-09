#!/usr/bin/env python3
"""Export dense Qwen3.5 OCR checkpoints for the pending native backend.

This is an experimental weight format, not a supported registry model. Compare
checkpoint names/shapes with the official Transformers model on meta before
writing anything. Do not silently drop unknown weights or fabricate defaults.
The source layout is preserved (including zero-centered RMSNorm weights).
"""

import argparse
import hashlib
import importlib.util
import inspect
from functools import lru_cache
import json
import math
import re
from pathlib import Path


ARCH = "qwen35_ocr"
LIGHTON_MODEL = "lightonai/LightOnOCR-3-0.8B"
LIGHTON_REVISION = "36d8636925327c671bd8e479c517a004f7566bd9"
LIGHTON4B_MODEL = "lightonai/LightOnOCR-3-4B"
LIGHTON4B_REVISION = "b71010f095dc3735de7ee6b9969bff8f8f50b39a"
LIGHTON4B_PROFILE = "lightonocr3-4b"
LIGHTON4B_FORMAT = 5  # Deliberately rejected by the current native owning context.
LIGHTON4B_VISION_RECIPE = "lightonocr3-4b-vision-f32"
# Original release code, not an observed inference runtime. Actual conversion
# must check these loaded files before constructing the independent meta schema.
LIGHTON4B_TRANSFORMERS_REVISION = "75d3bdcd4b3ba70cee21287218d9764f33da41f0"
LIGHTON4B_META_SOURCES = {
    "model": "aee59d55ee4e8ce0e50bf0e279796b85c2c66a28dfae55c3fcbb62fa9bcba048",
    "config": "cbea1f42a949edb6b9b001109df68cbf7cff04c96d3fe8a8e96602cb017b5a36",
}


@lru_cache(maxsize=1)
def _ovis_profiles():
    spec=importlib.util.spec_from_file_location('qwen35_ovis_profile',Path(__file__).with_name('qwen35_ovis_profile.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _ovis3_profile():
    spec=importlib.util.spec_from_file_location('qwen35_ovis3_profile',Path(__file__).with_name('qwen35_ovis3_profile.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _ovis3_hf_profile():
    spec=importlib.util.spec_from_file_location('qwen35_ovis3_hf_profile',Path(__file__).with_name('qwen35_ovis3_hf_profile.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _xiaomi0_profile():
    spec = importlib.util.spec_from_file_location("qwen35_xiaomi0_profile", Path(__file__).with_name("qwen35_xiaomi0_profile.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _lighton4b_profile():
    spec = importlib.util.spec_from_file_location(
        "qwen35_lighton4b_source_profile", Path(__file__).with_name("lightonocr3_4b_profile.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def precision_recipe(name):
    if name == "original-f16":
        return []
    if name == LIGHTON4B_VISION_RECIPE:
        return ["v.patch_embed.proj.weight", "v.pos_embed.weight", "v.merger.linear_fc1.weight", "v.merger.linear_fc2.weight"] + [
            f"v.blocks.{i}.{part}.weight" for i in range(24)
            for part in ["attn.qkv", "attn.proj", "mlp.linear_fc1", "mlp.linear_fc2"]
        ]
    if name != "vision-f32":
        raise ValueError(f"Unknown precision recipe: {name}")
    return ["v.patch_embed.proj.weight", "v.pos_embed.weight", "v.merger.linear_fc1.weight", "v.merger.linear_fc2.weight"] + [
        f"v.blocks.{i}.{part}.weight" for i in range(12)
        for part in ["attn.qkv", "attn.proj", "mlp.linear_fc1", "mlp.linear_fc2"]
    ]


def lighton4b_request(source_model, source_revision, profile, recipe="original-f16"):
    """Keep the explicit 4B export separate before filesystem or network access."""
    requested = profile == LIGHTON4B_PROFILE or source_model == LIGHTON4B_MODEL
    if requested and (profile != LIGHTON4B_PROFILE or source_model != LIGHTON4B_MODEL
                      or source_revision != LIGHTON4B_REVISION):
        raise ValueError("LightOnOCR-3-4B export requires its exact source/revision and --profile lightonocr3-4b")
    if requested and recipe not in ("original-f16", LIGHTON4B_VISION_RECIPE):
        raise ValueError("LightOnOCR-3-4B requires its distinct 24-layer vision recipe; historical vision-f32 is unchanged")
    if not requested and recipe == LIGHTON4B_VISION_RECIPE:
        raise ValueError("The LightOnOCR-3-4B vision recipe requires its exact export profile")
    return requested


def lighton4b_metadata_files():
    return _lighton4b_profile().METADATA_FILES + ("tokenizer.json",)


def _verify_pinned_file(path, size, digest):
    """Hash an existing exact payload with bounded reads, never download it."""
    path = Path(path)
    if type(size) is not int or size < 1 or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Invalid pinned source file identity")
    if path.stat().st_size != size:
        raise ValueError("Pinned source byte count differs: " + path.name)
    hasher = hashlib.sha256()
    remaining = size
    with path.open("rb") as stream:
        while remaining:
            chunk = stream.read(min(8 * 1024 * 1024, remaining))
            if not chunk:
                raise ValueError("Pinned source payload is truncated: " + path.name)
            hasher.update(chunk)
            remaining -= len(chunk)
        if stream.read(1):
            raise ValueError("Pinned source payload has trailing bytes: " + path.name)
    if hasher.hexdigest() != digest:
        raise ValueError("Pinned source SHA256 differs: " + path.name)
    return {"file": path.name, "bytes": size, "sha256": digest, "payload_verified": True}


def load_lighton4b_export_source(root, source_model, source_revision):
    """Verify exact small files and the full tokenizer before any weight access."""
    lighton4b_request(source_model, source_revision, LIGHTON4B_PROFILE)
    policy = _lighton4b_profile()
    root = Path(root)
    if (root / "model.safetensors.index.json").exists():
        raise ValueError("Pinned LightOnOCR-3-4B is a single file; unexpected shard index")
    audit = policy.evidence()
    raw = {name: policy._verified_bytes(root / name, audit["source_files"][name])
           for name in policy.METADATA_FILES}
    config = policy._json(raw["config.json"])
    policy.validate_config(config)
    policy.validate_generation(policy._json(raw["generation_config.json"]))
    tokenizer = _verify_pinned_file(root / "tokenizer.json", policy.TOKENIZER_BYTES, policy.TOKENIZER_SHA256)
    # This immutable source preflight still makes no framework/native claims.
    # The converter adds evidence for its actual input directory separately.
    pinned = policy.load_profile()
    pinned.update({"name": LIGHTON4B_PROFILE, "source_preflight_name": policy.PROFILE,
                   "format_version": LIGHTON4B_FORMAT, "missing_source_files": [],
                   "source_metadata_verified": {name: audit["source_files"][name]["sha256"] for name in raw}})
    pinned["tokenizer"]["payload_verified"] = tokenizer["payload_verified"]
    prefix_name = "LightOnOCR-3-4B-tokenizer.prefix.txt"
    prefix = policy._verified_bytes(policy.FIXTURE / prefix_name, {
        "bytes": policy.PREFIX_BYTES, "sha256": policy.PREFIX_SOURCES["LightOnOCR-3-4B"][2]})
    pinned["tokenizer"]["source_metadata"] = policy.tokenizer_prefix_fields(prefix, 7191)
    return pinned, config


def verify_lighton4b_checkpoint(path, config):
    """Check the acquired header first, then hash the complete existing payload."""
    policy = _lighton4b_profile()
    policy.validate_config(config)
    path = Path(path)
    if path.stat().st_size != policy.CHECKPOINT_BYTES:
        raise ValueError("LightOnOCR-3-4B checkpoint byte count differs")
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8 or int.from_bytes(prefix, "little") != policy.HEADER_BYTES:
            raise ValueError("LightOnOCR-3-4B checkpoint header length differs")
        header = stream.read(policy.HEADER_BYTES)
    census = policy.validate_header_bytes(header, config)
    proof = _verify_pinned_file(path, policy.CHECKPOINT_BYTES, policy.CHECKPOINT_SHA256)
    return {**proof, "header_sha256": policy.HEADER_SHA256, "tensor_count": census["tensor_count"]}


def lighton4b_tensor_plan(config, dtype="f16", keep_f32=(), recipe="original-f16"):
    """Derive all 723 export records without materializing a model tensor."""
    if dtype not in ("f32", "f16"):
        raise ValueError("LightOnOCR-3-4B export is F32/F16 only")
    lighton4b_request(LIGHTON4B_MODEL, LIGHTON4B_REVISION, LIGHTON4B_PROFILE, recipe)
    schema = _lighton4b_profile().expected_tensor_schema(config)
    retained = frozenset(precision_recipe(recipe)) | frozenset(keep_f32)
    exported = {tensor_name(source) for source in schema}
    if len(exported) != len(schema) or retained - exported:
        raise ValueError("LightOnOCR-3-4B export names or F32 retention tensors differ")
    records = []
    for source, source_shape in sorted(schema.items()):
        name = tensor_name(source)
        shape = ([source_shape[0], math.prod(source_shape[1:])]
                 if name == "v.patch_embed.proj.weight" else list(source_shape))
        target = ("float16" if dtype == "f16" and len(shape) == 2
                  and "norm" not in name and name not in retained else "float32")
        records.append({"source": source, "name": name, "source_shape": list(source_shape),
                        "shape": shape, "dtype": target})
    return {"format_version": LIGHTON4B_FORMAT, "dtype": dtype, "precision_recipe": recipe,
            "keep_f32": sorted(retained), "records": records,
            "checkpoint_tensor_count": len(schema), "export_tensor_count": len(records),
            "tied_head_stored_once": True, "mtp_policy": "no-mtp-tensors"}


def validate_lighton4b_meta_sources(version, source_hashes):
    if version != "5.5.4" or source_hashes != LIGHTON4B_META_SOURCES:
        raise ValueError("LightOnOCR-3-4B pinned meta-model implementation differs")
    return {"transformers_version": version, "upstream_revision": LIGHTON4B_TRANSFORMERS_REVISION,
            "loaded_source_sha256": dict(source_hashes), "scope": "meta-schema-only; no model forward"}


def validate_lighton4b_meta_schema(schema, config):
    expected = _lighton4b_profile().expected_tensor_schema(config)
    expected["lm_head.weight"] = list(expected["model.language_model.embed_tokens.weight"])
    if schema != expected:
        raise ValueError("LightOnOCR-3-4B official meta schema differs from the pinned source schema")
    return {"official_tensor_count": len(expected), "checkpoint_tensor_count": len(expected) - 1,
            "omitted_tied_alias": "lm_head.weight", "model_forward_observed": False}


def write_lighton4b_metadata(writer, pinned, config, plan, meta_validation):
    """Format 5 is an unqualified export; it cannot select an existing native path."""
    import gguf
    policy = _lighton4b_profile()
    text, vision = policy.validate_config(config)
    if (pinned["name"] != LIGHTON4B_PROFILE or pinned["source_model"] != LIGHTON4B_MODEL
            or pinned["source_revision"] != LIGHTON4B_REVISION
            or pinned["source_audit_sha256"] != policy.AUDIT_SHA256
            or plan["format_version"] != LIGHTON4B_FORMAT):
        raise ValueError("LightOnOCR-3-4B export metadata identity differs")
    writer.add_uint32("general.file_type", 0 if plan["dtype"] == "f32" else 1)
    writer.add_string(f"{ARCH}.profile", LIGHTON4B_PROFILE)
    writer.add_string(f"{ARCH}.source_audit_sha256", policy.AUDIT_SHA256)
    writer.add_string(f"{ARCH}.checkpoint_sha256", policy.CHECKPOINT_SHA256)
    writer.add_string(f"{ARCH}.tokenizer_sha256", policy.TOKENIZER_SHA256)
    writer.add_string(f"{ARCH}.tokenizer_source_metadata", json.dumps(pinned["tokenizer"]["source_metadata"], sort_keys=True))
    writer.add_string(f"{ARCH}.precision_recipe", plan["precision_recipe"])
    writer.add_string(f"{ARCH}.mtp_policy", plan["mtp_policy"])
    writer.add_string(f"{ARCH}.meta_schema_validation", json.dumps(meta_validation, sort_keys=True))
    # Standard decoder geometry keys; grouped linear-attention and vision keys
    # are explicit architecture extensions, not an assertion of native support.
    integers = {
        "context_length": text["max_position_embeddings"], "embedding_length": text["hidden_size"],
        "feed_forward_length": text["intermediate_size"], "block_count": text["num_hidden_layers"],
        "vocab_size": text["vocab_size"], "attention.head_count": text["num_attention_heads"],
        "attention.head_count_kv": text["num_key_value_heads"], "attention.key_length": text["head_dim"],
        "attention.value_length": text["head_dim"], "attention.full_attention_interval": text["full_attention_interval"],
        "rope.dimension_count": int(text["head_dim"] * text["partial_rotary_factor"]),
        "linear_attention.key_head_count": text["linear_num_key_heads"],
        "linear_attention.value_head_count": text["linear_num_value_heads"],
        "linear_attention.key_length": text["linear_key_head_dim"],
        "linear_attention.value_length": text["linear_value_head_dim"],
        "linear_attention.conv_kernel": text["linear_conv_kernel_dim"],
        "vision.block_count": vision["depth"], "vision.embedding_length": vision["hidden_size"],
        "vision.feed_forward_length": vision["intermediate_size"], "vision.attention.head_count": vision["num_heads"],
        "vision.patch_size": vision["patch_size"], "vision.temporal_patch_size": vision["temporal_patch_size"],
        "vision.spatial_merge_size": vision["spatial_merge_size"],
        "vision.position_embedding_count": vision["num_position_embeddings"],
        "vision.projection_dim": vision["out_hidden_size"], "primary_eos_id": pinned["generation"]["eos_token_id"],
        "source_text_eos_id": text["eos_token_id"], "source_top_level_eos_id": config["eos_token_id"],
        "image_min_pixels": pinned["image_processor"]["size"]["shortest_edge"],
        "image_max_pixels": pinned["image_processor"]["size"]["longest_edge"],
    }
    for key, value in integers.items():
        writer.add_uint32(f"{ARCH}.{key}", value)
    writer.add_float32(f"{ARCH}.attention.layer_norm_rms_epsilon", text["rms_norm_eps"])
    writer.add_float32(f"{ARCH}.rope.freq_base", text["rope_parameters"]["rope_theta"])
    writer.add_float32(f"{ARCH}.rope.partial_factor", text["partial_rotary_factor"])
    writer.add_bool(f"{ARCH}.rope.interleaved", text["rope_parameters"]["mrope_interleaved"])
    writer.add_array(f"{ARCH}.layer_types", text["layer_types"])
    for key, values in (("rope.dimension_sections", text["rope_parameters"]["mrope_section"]),
                        ("stop_token_ids", [pinned["generation"]["eos_token_id"]])):
        writer.add_key_value(f"{ARCH}.{key}", values, gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.INT32)
    for key in ("native_supported", "native_qualified", "export_qualified", "native_alias_qualified",
                "image_max_qualified", "full_document_qualified", "framework_runtime_observed",
                "tokenizer_runtime_qualified", "processor_qualified", "effective_stops_qualified"):
        writer.add_bool(f"{ARCH}.{key}", False)


def read_tokenizer_metadata(root, vocab_size, generation=None, allow_generation_eos_list=False):
    tokenizer = json.loads((root / "tokenizer.json").read_text())
    if tokenizer["model"]["type"] != "BPE":
        raise ValueError("Expected an audited BPE tokenizer")
    vocab = dict(tokenizer["model"]["vocab"])
    for token in tokenizer["added_tokens"]:
        vocab[token["content"]] = token["id"]
    tokens = [""] * vocab_size
    occupied = set()
    for token, index in vocab.items():
        if index < 0 or index >= len(tokens) or index in occupied:
            raise ValueError(f"Invalid/duplicate tokenizer id: {index}")
        occupied.add(index)
        tokens[index] = token
    merges = tokenizer["model"]["merges"]
    merges = [item if isinstance(item, str) else " ".join(item) for item in merges]

    if tokenizer.get("normalizer") != {"type": "NFC"} or tokenizer["model"].get("dropout") is not None:
        raise ValueError("Unimplemented tokenizer normalization/dropout")
    if any(t.get("single_word") or t.get("lstrip") or t.get("rstrip") or t.get("normalized") for t in tokenizer["added_tokens"]):
        raise ValueError("Unimplemented added-token matching policy")
    if generation is None:
        generation = json.loads((root / "generation_config.json").read_text())
    eos = generation["eos_token_id"]
    if allow_generation_eos_list:
        if (not isinstance(eos, list) or eos != [248044, 248046]
                or any(type(value) is not int or value < 0 or value >= vocab_size for value in eos)):
            raise ValueError("OvisOCR3 source EOS list differs")
        eos = eos[0]  # Legacy scalar key; full source list is written under the OvisOCR3 profile.
    if not isinstance(eos, int) or eos < 0 or eos >= vocab_size:
        raise ValueError("Only a valid single generation EOS supported")
    config = json.loads((root / "tokenizer_config.json").read_text())
    return {"tokens": tokens, "merges": merges,
            "added": [t["id"] for t in tokenizer["added_tokens"]],
            "special": [t["id"] for t in tokenizer["added_tokens"] if t["special"]],
            "generation_eos": eos, "tokenizer_eos": vocab[config["eos_token"]],
            "pad": vocab[config["pad_token"]],
            "normalizer": "NFC", "pattern": tokenizer["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]}


def write_tokenizer_metadata(writer, data):
    import gguf
    writer.add_string("tokenizer.ggml.model", "gpt2")
    writer.add_string("tokenizer.ggml.pre", "qwen35")
    writer.add_string(f"{ARCH}.normalizer", data["normalizer"])
    writer.add_string(f"{ARCH}.pretokenizer_pattern", data["pattern"])
    writer.add_array("tokenizer.ggml.tokens", data["tokens"])
    writer.add_array("tokenizer.ggml.merges", data["merges"])
    for key in ["added", "special"]:
        writer.add_key_value(f"{ARCH}.{key}_token_ids", data[key], gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.INT32)
    writer.add_uint32(f"{ARCH}.generation_eos_id", data["generation_eos"])
    writer.add_uint32("tokenizer.ggml.eos_token_id", data["tokenizer_eos"])
    writer.add_uint32("tokenizer.ggml.padding_token_id", data["pad"])


def validate_config(config, profile=None):
    if config.get("model_type") != "qwen3_5":
        raise ValueError("Only dense qwen3_5 checkpoints are supported")
    text = config["text_config"]
    vision = config["vision_config"]
    kinds = text["layer_types"]
    if len(kinds) != text["num_hidden_layers"] or not set(kinds) <= {
        "linear_attention", "full_attention"
    }:
        raise ValueError("Invalid or unsupported decoder layer schedule")
    if profile is not None:
        if profile.get("name") == LIGHTON4B_PROFILE:
            lighton4b_request(profile.get("source_model"), profile.get("source_revision"), LIGHTON4B_PROFILE)
            if (profile.get("format_version") != LIGHTON4B_FORMAT
                    or profile.get("source_audit_sha256") != _lighton4b_profile().AUDIT_SHA256):
                raise ValueError("LightOnOCR-3-4B export profile identity differs")
            _lighton4b_profile().validate_config(config)
        elif profile.get("name") == "xiaomi-ocr-0":
            _xiaomi0_profile().validate_config(config, profile)
        elif profile.get("qualification_kind") == "hf5.16.1-original-reference":
            _ovis3_hf_profile().validate_config(config,profile)
        elif profile.get("name") == "ovisocr3":
            _ovis3_profile().validate_config(config,profile)
        else:
            _ovis_profiles().validate_config(config,profile)
    if text.get("mtp_num_hidden_layers", 0) and profile is None:
        raise ValueError("MTP checkpoints need a separately audited export policy")
    if vision.get("deepstack_visual_indexes", []):
        raise ValueError("DeepStack is not part of the audited Qwen3.5 OCR format")
    if text.get("attention_bias", False) or not text.get("attn_output_gate", False):
        raise ValueError("Expected bias-free gated full attention")
    rope = text["rope_parameters"]
    rotary = int(text["head_dim"] * rope["partial_rotary_factor"])
    if not rope["mrope_interleaved"] or sum(rope["mrope_section"]) * 2 != rotary:
        raise ValueError("Inconsistent partial/interleaved MRoPE configuration")
    return text, vision


def tensor_name(name):
    if name.startswith("model.language_model."):
        result = "l." + name.removeprefix("model.language_model.")
    elif name.startswith("model.visual."):
        result = "v." + name.removeprefix("model.visual.")
    elif name == "lm_head.weight":
        result = name
    else:
        raise ValueError(f"Unaudited checkpoint tensor prefix: {name}")
    if len(result.encode("utf-8")) >= 64:
        raise ValueError(f"Tensor name exceeds native ggml limit: {result}")
    return result


def convert(model_dir, output, source_model, source_revision, dtype="f16", keep_f32=(), profile=None,
            recipe="original-f16"):
    root = Path(model_dir)
    output = Path(output)
    if not re.fullmatch(r"[0-9a-f]{40}", source_revision):
        raise ValueError("An immutable 40-hex source revision is required")
    is_lighton4b = lighton4b_request(source_model, source_revision, profile, recipe)
    if profile not in (None,"ovisocr2","ovisocr3","ovisocr3-hf-reference","xiaomi-ocr-0",LIGHTON4B_PROFILE):
        raise ValueError("Unknown export profile")
    policy=(_lighton4b_profile() if is_lighton4b else
            _xiaomi0_profile() if profile=="xiaomi-ocr-0" else
            _ovis3_hf_profile() if profile=="ovisocr3-hf-reference" else
            _ovis3_profile() if profile=="ovisocr3" else _ovis_profiles())
    if is_lighton4b:
        pinned, config = load_lighton4b_export_source(root, source_model, source_revision)
    else:
        pinned=policy.load_profile(root,source_model,source_revision) if profile else None
        config = json.loads((root / "config.json").read_text())
    text, vision = validate_config(config,pinned)
    if dtype not in {"f32", "f16"}:
        raise ValueError("Establish full-precision parity before adding quantized exports")
    export_plan = lighton4b_tensor_plan(config, dtype, keep_f32, recipe) if is_lighton4b else None
    if export_plan:
        keep_f32 = export_plan["keep_f32"]
    if pinned:
        if (root/"model.safetensors.index.json").exists():
            raise ValueError("Pinned checkpoint is a single file; unexpected shard index")
        if is_lighton4b:
            pinned["verified_checkpoint"] = verify_lighton4b_checkpoint(root / "model.safetensors", config)
            pinned["checkpoint"]["payload_verified"] = True
            pinned["scope"] = "exact source/tokenizer/checkpoint bytes; no model forward or native qualification"
        else:
            policy.verify_checkpoint(root/"model.safetensors",pinned)

    import gguf
    import numpy as np
    import torch
    from safetensors import safe_open
    import transformers
    from transformers import AutoConfig, Qwen3_5ForConditionalGeneration
    meta_validation = None
    if is_lighton4b:
        from transformers import Qwen3_5Config
        loaded_hashes = {
            key: hashlib.sha256(Path(inspect.getfile(cls)).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
            for key, cls in (("model", Qwen3_5ForConditionalGeneration), ("config", Qwen3_5Config))
        }
        meta_validation = validate_lighton4b_meta_sources(transformers.__version__, loaded_hashes)
    elif pinned:
        expected=pinned['qualification']
        implementation=Path(inspect.getfile(Qwen3_5ForConditionalGeneration)).read_bytes().replace(b'\r\n',b'\n')
        if (transformers.__version__!=expected['transformers_version']
                or hashlib.sha256(implementation).hexdigest()!=expected['loaded_source_sha256']['model']):
            raise ValueError("Pinned meta-model implementation differs from source profile")

    runtime_metadata = policy.validate_runtime_metadata(root) if profile == "xiaomi-ocr-0" else None

    # Official implementation supplies the independent expected parameter schema.
    hf_config = AutoConfig.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    if pinned: hf_config._attn_implementation="eager"
    with torch.device("meta"):
        official = Qwen3_5ForConditionalGeneration(hf_config)
    expected = {name: list(tensor.shape) for name, tensor in official.state_dict().items()}
    del official
    if is_lighton4b:
        meta_validation["schema"] = validate_lighton4b_meta_schema(expected, config)
    elif profile == "xiaomi-ocr-0":
        policy.validate_main_schema(expected)

    index = root / "model.safetensors.index.json"
    if index.exists():
        mapping = json.loads(index.read_text())["weight_map"]
        shards = sorted(set(mapping.values()))
    else:
        shards = ["model.safetensors"]
        mapping = None
    actual = {}
    for shard in shards:
        # Check index paths before opening checkpoint-controlled file names.
        if Path(shard).name != shard:
            raise ValueError("Shard paths must be plain file names")
        with safe_open(root / shard, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in actual:
                    raise ValueError(f"Duplicate tensor: {name}")
                if mapping is not None and mapping.get(name) != shard:
                    raise ValueError(f"Shard/index mismatch: {name}")
                actual[name] = (shard, list(handle.get_slice(name).get_shape()))
    if mapping is not None and set(actual) != set(mapping):
        raise ValueError("Checkpoint index has missing or extra tensors")
    checkpoint_tensor_count = len(actual)
    excluded_tensors = []
    if profile == "xiaomi-ocr-0":
        selected = policy.non_speculative_tensors(actual, pinned)
        excluded_tensors = sorted(set(actual) - set(selected))
        actual = selected
    tied = bool(config.get("tie_word_embeddings", text.get("tie_word_embeddings", False)))
    missing = set(expected) - set(actual)
    if tied and missing == {"lm_head.weight"}:
        missing.clear()
    extra = set(actual) - set(expected)
    mismatched = [name for name in actual if name in expected and actual[name][1] != expected[name]]
    if missing or extra or mismatched:
        raise ValueError(f"Tensor schema mismatch: missing={sorted(missing)}, "
                         f"extra={sorted(extra)}, shape={mismatched}")
    names = [tensor_name(name) for name in actual]
    if len(names) != len(set(names)):
        raise ValueError("Export tensor names collide")
    keep_f32 = frozenset(keep_f32)
    exported_names = set(names) - ({"lm_head.weight"} if tied else set())
    if keep_f32 - exported_names:
        raise ValueError(f"Unknown F32 retention tensors: {sorted(keep_f32 - exported_names)}")
    if tied and "lm_head.weight" in actual:
        with safe_open(root / actual["lm_head.weight"][0], framework="pt") as head_file, \
             safe_open(root / actual["model.language_model.embed_tokens.weight"][0], framework="pt") as embed_file:
            if not torch.equal(head_file.get_tensor("lm_head.weight"),
                               embed_file.get_tensor("model.language_model.embed_tokens.weight")):
                raise ValueError("Checkpoint claims tied weights but stores unequal head/embedding")

    tokenizer_data = read_tokenizer_metadata(root, text["vocab_size"],pinned["generation"] if pinned else None,
                                            allow_generation_eos_list=profile in ("ovisocr3","ovisocr3-hf-reference"))

    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_name(output.name + ".partial")
    writer = gguf.GGUFWriter(str(pending), ARCH, use_temp_file=True)
    writer.add_string("general.name", source_model)
    writer.add_string("general.source.huggingface.repository", source_model)
    writer.add_string("general.source.huggingface.revision", source_revision)
    writer.add_string(f"{ARCH}.status", "experimental-export-native-backend-pending")
    writer.add_uint32(f"{ARCH}.format_version", pinned["format_version"] if pinned else 1)
    writer.add_bool(f"{ARCH}.tie_word_embeddings", tied)
    writer.add_bool(f"{ARCH}.zero_centered_text_norm", True)
    # Preserve every config field, rather than treating nested EOS ids as synonyms.
    for filename in ["config.json", "processor_config.json", "preprocessor_config.json",
                     "tokenizer_config.json", "generation_config.json"]:
        if pinned and filename in pinned['missing_source_files']:
            continue  # Preserve absence, never fabricate an upstream source file.
        writer.add_string(f"{ARCH}.{filename.removesuffix('.json')}", (root / filename).read_text())
    writer.add_string("tokenizer.chat_template", (root / "chat_template.jinja").read_text())
    write_tokenizer_metadata(writer, tokenizer_data)
    if is_lighton4b:
        write_lighton4b_metadata(writer, pinned, config, export_plan, meta_validation)
    elif profile == "xiaomi-ocr-0":
        write_xiaomi0_metadata(writer, pinned)
    elif profile=="ovisocr2":
        writer.add_string(f"{ARCH}.profile",pinned['name'])
        writer.add_string(f"{ARCH}.profile.rendered_prompt",pinned['rendered_prompt'])
        writer.add_key_value(f"{ARCH}.stop_token_ids",pinned['stop_ids'],gguf.GGUFValueType.ARRAY,gguf.GGUFValueType.INT32)
        writer.add_uint32(f"{ARCH}.primary_eos_id",pinned['primary_eos'])
        writer.add_array(f"{ARCH}.missing_source_files",pinned['missing_source_files'])
        writer.add_string(f"{ARCH}.resolved_generation_config",json.dumps(pinned['generation'],sort_keys=True))
        writer.add_string(f"{ARCH}.resolved_processor_config",json.dumps(pinned['processor'],sort_keys=True))
        writer.add_string(f"{ARCH}.metadata_qualification",json.dumps(pinned['qualification'],sort_keys=True))
        writer.add_string(f"{ARCH}.checkpoint_sha256",pinned['checkpoint_sha256'])
        writer.add_uint32(f"{ARCH}.image_min_pixels",pinned['image_min_pixels'])
        writer.add_uint32(f"{ARCH}.image_max_pixels",pinned['image_max_pixels'])
        writer.add_bool(f"{ARCH}.image_max_qualified",False)
        writer.add_bool(f"{ARCH}.native_alias_qualified",False)
    elif profile in ("ovisocr3","ovisocr3-hf-reference"):
        writer.add_array(f"{ARCH}.missing_source_files",pinned["missing_source_files"])
        writer.add_string(f"{ARCH}.profile",pinned['name'])
        writer.add_string(f"{ARCH}.profile.task",pinned['task'])
        writer.add_string(f"{ARCH}.profile.task_prompt",pinned['task_prompt'])
        writer.add_key_value(f"{ARCH}.source_eos_token_ids",pinned['source_eos_ids'],
                             gguf.GGUFValueType.ARRAY,gguf.GGUFValueType.INT32)
        writer.add_string(f"{ARCH}.source_generation_config",json.dumps(pinned['generation'],sort_keys=True))
        writer.add_string(f"{ARCH}.source_qualification",json.dumps(pinned['qualification'],sort_keys=True))
        writer.add_string(f"{ARCH}.checkpoint_sha256",pinned['checkpoint_sha256'])
        writer.add_uint32(f"{ARCH}.image_min_pixels",pinned['image_min_pixels'])
        writer.add_uint32(f"{ARCH}.image_max_pixels",pinned['image_max_pixels'])
        writer.add_bool(f"{ARCH}.image_max_qualified",False)
        if profile=="ovisocr3-hf-reference":
            writer.add_string(f"{ARCH}.profile.rendered_prompt",pinned['rendered_prompt'])
            writer.add_key_value(f"{ARCH}.stop_token_ids",pinned['stop_ids'],
                                 gguf.GGUFValueType.ARRAY,gguf.GGUFValueType.INT32)
            writer.add_uint32(f"{ARCH}.primary_eos_id",pinned['primary_eos'])
            writer.add_string(f"{ARCH}.metadata_qualification",pinned['metadata_producer_canonical_sha256'])
            writer.add_string(f"{ARCH}.original_reference_qualification",pinned['original_reference_producer_canonical_sha256'])
            writer.add_bool(f"{ARCH}.rendered_prompt_qualified",True)
            writer.add_bool(f"{ARCH}.processor_qualified",True)
            writer.add_bool(f"{ARCH}.hf_stop_policy_qualified",True)
            writer.add_string(f"{ARCH}.hf_stop_runtime_proof_sha256",pinned['hf_stop_runtime_proof_sha256'])
            writer.add_string(f"{ARCH}.stop_qualification_scope",pinned['stop_qualification_scope'])
            writer.add_bool(f"{ARCH}.effective_stops_qualified",False)
        else:
            writer.add_bool(f"{ARCH}.rendered_prompt_qualified",False)
            writer.add_bool(f"{ARCH}.processor_qualified",False)
            writer.add_bool(f"{ARCH}.hf_stop_policy_qualified",False)
            writer.add_bool(f"{ARCH}.effective_stops_qualified",False)
        writer.add_bool(f"{ARCH}.native_alias_qualified",False)

    records = []
    planned_records = {item["source"]: item for item in export_plan["records"]} if export_plan else {}
    try:
        for shard in shards:
            with safe_open(root / shard, framework="pt", device="cpu") as handle:
                for source in sorted(handle.keys()):
                    if source in excluded_tensors:
                        continue  # Exact 15-name MTP allowlist; original HF ordinary generation ignores these.
                    if tied and source == "lm_head.weight":
                        continue  # Native head must alias the exported embedding.
                    name = tensor_name(source)
                    source_digest = None
                    if is_lighton4b:
                        original = handle.get_tensor(source)
                        if (source not in planned_records or original.dtype != torch.bfloat16
                                or list(original.shape) != planned_records[source]["source_shape"]
                                or not original.is_contiguous()):
                            raise ValueError("LightOnOCR-3-4B source tensor differs: " + source)
                        source_digest = hashlib.sha256(memoryview(original.view(torch.uint8).numpy())).hexdigest()
                        tensor = original.float().numpy()
                        del original
                    else:
                        tensor = handle.get_tensor(source).float().numpy()
                    source_shape = list(tensor.shape)
                    if name == "v.patch_embed.proj.weight":
                        # ggml supports at most four axes. HF applies this Conv3d
                        # to processor-flattened patches; preserve its C,T,H,W order.
                        tensor = tensor.reshape(tensor.shape[0], -1)
                    # Scalars, recurrent coefficients, convs and norms remain F32.
                    target = np.float16 if dtype == "f16" and name not in keep_f32 and tensor.ndim == 2 and "norm" not in name else np.float32
                    array = np.ascontiguousarray(tensor, dtype=target)
                    if not np.isfinite(array).all():
                        raise ValueError(f"Non-finite or overflowing export: {source}")
                    record = {"source": source, "name": name, "shape": list(array.shape),
                              "source_shape": source_shape, "dtype": str(array.dtype)}
                    if is_lighton4b and record != planned_records[source]:
                        raise ValueError("LightOnOCR-3-4B converted tensor differs from its plan: " + source)
                    writer.add_tensor(name, array)
                    record["sha256"] = hashlib.sha256(memoryview(array) if is_lighton4b else array.tobytes()).hexdigest()
                    if is_lighton4b:
                        record.update({"source_dtype": "BF16", "source_payload_sha256": source_digest})
                    records.append(record)
        if is_lighton4b and (len(records) != policy.TENSOR_COUNT
                            or {item["source"] for item in records} != set(planned_records)):
            raise ValueError("LightOnOCR-3-4B exported tensor census differs")
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    except BaseException:
        writer.close()
        pending.unlink(missing_ok=True)
        raise
    writer.close()
    pending.replace(output)
    manifest = {"source_model": source_model, "source_revision": source_revision,
                "architecture": ARCH, "dtype": dtype, "keep_f32": sorted(keep_f32), "tie_word_embeddings": tied,
                "checkpoint_tensor_count": checkpoint_tensor_count, "export_tensor_count": len(records),
                "records": records, "bytes": output.stat().st_size}
    if profile == "xiaomi-ocr-0":
        manifest["excluded_non_speculative_mtp_tensors"] = excluded_tensors
        manifest["runtime_metadata"] = runtime_metadata
        manifest["main_schema_sha256"] = policy.MAIN_SCHEMA_SHA256
    if is_lighton4b:
        manifest.update({"format_version": LIGHTON4B_FORMAT, "precision_recipe": recipe,
                         "meta_schema_validation": meta_validation, "source_payload_verified": True,
                         "gguf_roundtrip_verified": False, "framework_runtime_observed": False,
                         "native_supported": False, "native_qualified": False, "export_qualified": False,
                         "native_alias_qualified": False, "image_max_qualified": False,
                         "full_document_qualified": False})
    if pinned:
        manifest["profile"]={k:v for k,v in pinned.items() if k!="proof_marker"}
    output.with_suffix(output.suffix + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest



def write_xiaomi0_metadata(writer, pinned):
    """Shared with hosted metadata controls; every qualification stays explicitly false."""
    import gguf
    _xiaomi0_profile().check_profile(pinned)
    writer.add_string(f"{ARCH}.profile", pinned["name"])
    writer.add_string(f"{ARCH}.source_audit_sha256", pinned["source_audit_sha256"])
    writer.add_string(f"{ARCH}.checkpoint_sha256", pinned["checkpoint_sha256"])
    writer.add_string(f"{ARCH}.profile.prompt_order", "image-first")
    writer.add_string(f"{ARCH}.profile.default_task", "document")
    for task, prompt in pinned["prompts"].items():
        writer.add_string(f"{ARCH}.profile.prompt.{task}", prompt)
    writer.add_array(f"{ARCH}.missing_source_files", pinned["missing_source_files"])
    writer.add_key_value(f"{ARCH}.stop_token_ids", pinned["stop_ids"],
                         gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.INT32)
    writer.add_uint32(f"{ARCH}.primary_eos_id", pinned["primary_eos"])
    writer.add_uint32(f"{ARCH}.image_min_pixels", pinned["image_min_pixels"])
    writer.add_uint32(f"{ARCH}.image_max_pixels", pinned["image_max_pixels"])
    writer.add_uint32(f"{ARCH}.native_patch_limit", pinned["native_patch_limit"])
    writer.add_string(f"{ARCH}.mtp_policy", "exact15-excluded-nonspeculative")
    writer.add_string(f"{ARCH}.layout_dependency", "PP-DocLayoutV3")
    writer.add_string(f"{ARCH}.resolved_generation_config", json.dumps(pinned["generation"], sort_keys=True))
    for key in ("native_alias_qualified", "image_max_qualified", "full_document_qualified"):
        writer.add_bool(f"{ARCH}.{key}", False)


def main():
    from huggingface_hub import snapshot_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=LIGHTON_MODEL)
    parser.add_argument("--profile",choices=["ovisocr2","ovisocr3","ovisocr3-hf-reference","xiaomi-ocr-0",LIGHTON4B_PROFILE],
                        help="Pinned experimental checkpoint profile; not a published model alias")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", choices=["f32", "f16"], default="f16")
    parser.add_argument("--precision-recipe", choices=["original-f16", "vision-f32", LIGHTON4B_VISION_RECIPE], default="original-f16",
                        help="vision-f32 retains its historical 12-layer scope; lightonocr3-4b-vision-f32 is a separate unqualified 24-layer recipe")
    parser.add_argument("--keep-f32", action="append", default=[], metavar="GGUF_NAME",
                        help="Keep this exact tensor in F32; repeat for multiple tensors (unknown names reject)")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("--revision must be an immutable 40-hex commit")
    try:
        is_lighton4b = lighton4b_request(args.model, args.revision, args.profile, args.precision_recipe)
    except ValueError as error:
        parser.error(str(error))
    if is_lighton4b:
        metadata = snapshot_download(args.model, revision=args.revision,
                                     allow_patterns=list(lighton4b_metadata_files()))
        _, config = load_lighton4b_export_source(metadata, args.model, args.revision)
        lighton4b_tensor_plan(config, args.dtype, args.keep_f32, args.precision_recipe)  # Before the checkpoint download.
    elif args.profile == "xiaomi-ocr-0" or args.model == _xiaomi0_profile().MODEL:
        policy = _xiaomi0_profile()
        if args.profile != "xiaomi-ocr-0" or args.model != policy.MODEL or args.revision != policy.REVISION:
            parser.error("Xiaomi export requires its exact source/revision and --profile xiaomi-ocr-0")
        metadata = snapshot_download(args.model, revision=args.revision, allow_patterns=list(policy.metadata_files()))
        policy.load_profile(metadata, args.model, args.revision)
        policy.validate_runtime_metadata(metadata)  # Before any checkpoint download.
    elif args.profile or args.model in ("ATH-MaaS/OvisOCR2", "ATH-MaaS/OvisOCR3"):
        policy=(_ovis3_hf_profile() if args.profile=="ovisocr3-hf-reference" else
                _ovis3_profile() if args.profile=="ovisocr3" else _ovis_profiles())
        if args.profile not in ("ovisocr2","ovisocr3","ovisocr3-hf-reference") or args.model!=policy.MODEL or args.revision!=policy.REVISION:
            parser.error("Ovis export requires its exact source/revision and matching --profile")
        metadata=snapshot_download(args.model,revision=args.revision,
                                  allow_patterns=list(policy.metadata_files()))
        policy.load_profile(metadata,args.model,args.revision)  # Before any weight download.
    directory = snapshot_download(args.model, revision=args.revision,
                                  allow_patterns=(list(lighton4b_metadata_files()) + ["model.safetensors"]
                                                  if is_lighton4b else
                                                  list(_xiaomi0_profile().metadata_files()) + ["model.safetensors"]
                                                  if args.profile == "xiaomi-ocr-0" else
                                                  list(_ovis3_profile().metadata_files())+["*.safetensors"]
                                                  if args.profile in ("ovisocr3","ovisocr3-hf-reference") else
                                                  ["*.json", "*.jinja", "*.safetensors", "LICENSE*"]))
    retained = sorted(set(precision_recipe(args.precision_recipe) + args.keep_f32))
    manifest = convert(directory, args.output, args.model, args.revision, args.dtype, retained,
                       profile=args.profile, recipe=args.precision_recipe)
    print(json.dumps({key: value for key, value in manifest.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
