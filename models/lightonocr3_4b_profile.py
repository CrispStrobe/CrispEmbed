#!/usr/bin/env python3
"""Pinned, model-free LightOnOCR-3-4B source preflight.

This module verifies captured source metadata and the actual safetensors header.
It neither downloads nor opens a weight payload, imports a model framework,
writes GGUF, nor enables native inference. A successful preflight is source
evidence only; every runtime, export and publication qualification stays false.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path


MODEL = "lightonai/LightOnOCR-3-4B"
REVISION = "b71010f095dc3735de7ee6b9969bff8f8f50b39a"
PROFILE = "lightonocr3-4b-source-only"
AUDITED_NATIVE_REVISION = "2f748825ff3b56b8f2b89d156986b94218ae28c3"
FIXTURE = Path(__file__).resolve().parents[1] / "tests/fixtures/lightonocr3-4b-source"
AUDIT_SHA256 = "d7a6fc3398a35004e81271bd08ea36f0b7dce6b1285f92c1bb237cd799005a09"
CONFIG_SHA256 = "42df4dbbccb724b543e14c67f7f7b7d237c230344a1e445d60f47d9c17e686f6"
GENERATION_SHA256 = "a38e42aacc20c475d93d29479488d9512bc73281dddb7d940f58d338506d1325"
CHECKPOINT_SHA256 = "6b0dab74a2dab02d5b10ca56322255024ed8dca35c02746b42e9f296aaeb4e67"
CHECKPOINT_BYTES = 9078620536
HEADER_SHA256 = "a1e36638b195f2d5943e000fcb069b275cb22ccc28b1b6184ddee7bdea7ef081"
HEADER_BYTES = 89456
TENSOR_COUNT = 723
PARAMETER_COUNT = 4539265536
TOKENIZER_SHA256 = "87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4"
TOKENIZER_BYTES = 19989343
PREFIX_AUDIT_SHA256 = "582f45ddbd70d37ab1eba8ae715371dd274963956a659d4a704f8c8652b1e789"
PREFIX_BYTES = 65536
PREFIX_SOURCES = {
    "LightOnOCR-3-4B": (REVISION, TOKENIZER_BYTES,
                        "e8b90f9bd9bd80d18d67933ed34e562e89da5ab4f11ada2651f444aa77b9146d"),
    "LightOnOCR-3-0.8B": ("36d8636925327c671bd8e479c517a004f7566bd9", 19989325,
                          "c8313ed70439374e2d12178f4293a49f2285467cd1948841ae9ac44e7ad300aa"),
}
LICENSE_REVISION = "36755d461be079737860a5f03ae0c803501269e9"
LICENSE_SHA256 = "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
LICENSE_BLOB = "d645695673349e3947e8e5ae42332d0ac3164cd7"
METADATA_FILES = (
    "config.json", "generation_config.json", "processor_config.json",
    "preprocessor_config.json", "tokenizer_config.json", "chat_template.jinja",
)
HEADER_FILE = "model.safetensors.header.json"
SOURCE_FILES = frozenset(METADATA_FILES + (
    HEADER_FILE, "README.md", "README.no_thinking.md", "huggingface_model_api.json",
))


def _canonical_sha256(value):
    try:
        data = json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode()
    except (TypeError, ValueError) as error:
        raise ValueError("Source contract is not finite JSON") from error
    return hashlib.sha256(data).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate source JSON key: " + key)
        result[key] = value
    return result


def _json(data):
    def reject_constant(value):
        raise ValueError("Non-finite source JSON constant: " + value)
    return json.loads(data, object_pairs_hook=_unique_object, parse_constant=reject_constant)


def _verified_bytes(path, record):
    """Read only a bounded metadata asset, never a declared checkpoint payload."""
    path = Path(path)
    size = record["bytes"]
    if type(size) is not int or not 0 < size <= 32 * 1024 * 1024:
        raise ValueError("Metadata asset exceeds the preflight byte bound")
    if path.stat().st_size != size:
        raise ValueError("Source byte length differs: " + path.name)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != record["sha256"]:
        raise ValueError("Source bytes changed: " + path.name)
    return data


def evidence():
    """Verify the frozen capture and its small original files, not model weights."""
    raw = (FIXTURE / "audit_manifest.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != AUDIT_SHA256:
        raise ValueError("LightOnOCR-3-4B source audit bytes changed")
    report = _json(raw)
    if (report.get("repository") != MODEL or report.get("revision") != REVISION
            or set(report.get("source_files", {})) != SOURCE_FILES):
        raise ValueError("LightOnOCR-3-4B source audit identity differs")
    for name, record in report["source_files"].items():
        _verified_bytes(FIXTURE / name, record)
    files = report["hf_reported_files"]
    checkpoint = files["model.safetensors"]
    tokenizer = files["tokenizer.json"]
    header = report["source_files"][HEADER_FILE]
    if (checkpoint["size"] != CHECKPOINT_BYTES
            or checkpoint["lfs"]["size"] != CHECKPOINT_BYTES
            or checkpoint["lfs"]["sha256"] != CHECKPOINT_SHA256
            or tokenizer["size"] != TOKENIZER_BYTES
            or tokenizer["lfs"]["sha256"] != TOKENIZER_SHA256
            or "model.safetensors.index.json" in files
            or header["bytes"] != HEADER_BYTES or header["sha256"] != HEADER_SHA256
            or header["source_file_bytes"] != CHECKPOINT_BYTES
            or report["tensor_header_audit"] != {
                "tensor_count": TENSOR_COUNT, "bf16_parameter_count": PARAMETER_COUNT,
                "has_lm_head": False, "has_mtp": False,
                "parameter_count_includes_tied_embedding_once": True,
            }):
        raise ValueError("LightOnOCR-3-4B checkpoint declarations differ")
    return report


def validate_config(config):
    """Admit this exact dense checkpoint, not an arbitrary larger Qwen model."""
    if _canonical_sha256(config) != CONFIG_SHA256:
        raise ValueError("LightOnOCR-3-4B pinned config differs")
    return config["text_config"], config["vision_config"]


def validate_generation(generation):
    if _canonical_sha256(generation) != GENERATION_SHA256:
        raise ValueError("LightOnOCR-3-4B pinned generation config differs")
    # The nested text config declares 248044. Preserve that discrepancy; the
    # actual top-level/generation declaration is 248046. No sampling defaults
    # are invented here: temperature/do_sample are absent from this source file.
    return generation["eos_token_id"]


def tokenizer_prefix_fields(data, model_offset):
    """Derive only complete fields before `model`; never parse a full tokenizer."""
    if (len(data) != PREFIX_BYTES or type(model_offset) is not int
            or not 0 < model_offset < len(data)
            or not data[model_offset:].startswith(b'"model": {')):
        raise ValueError("Tokenizer prefix/model boundary differs")
    before_model = data[:model_offset].rstrip()
    if not before_model.endswith(b","):
        raise ValueError("Tokenizer metadata prefix is not a complete field sequence")
    # Close a derived metadata-only object in memory. The original incomplete
    # prefix is never changed or represented as a complete tokenizer document.
    fields = _json(before_model[:-1] + b"}")
    if set(fields) != {"version", "truncation", "padding", "added_tokens", "normalizer",
                       "pre_tokenizer", "post_processor", "decoder"}:
        raise ValueError("Tokenizer prefix metadata field coverage differs")
    return fields


def tokenizer_prefix_evidence(root=FIXTURE):
    """Verify exact prefix/license captures and recompute the declared comparison."""
    root = Path(root)
    raw = _verified_bytes(root / "tokenizer_prefix_audit.json",
                          {"bytes": 25244, "sha256": PREFIX_AUDIT_SHA256})
    report = _json(raw)
    if (report["schema_version"] != 1 or set(report["tokenizer_prefixes"]) != set(PREFIX_SOURCES)
            or report["existing_audit_manifest_unchanged"] != {
                "path": "tests/fixtures/lightonocr3-4b-source/audit_manifest.json",
                "sha256_before": AUDIT_SHA256, "sha256_after": AUDIT_SHA256,
            }):
        raise ValueError("Tokenizer prefix audit identity differs")
    fields, sources = {}, {}
    for name, (revision, total_bytes, digest) in PREFIX_SOURCES.items():
        item = report["tokenizer_prefixes"][name]
        filename = name + "-tokenizer.prefix.txt"
        if (item["repository"] != "lightonai/" + name or item["revision"] != revision
                or item["stored_file"] != filename or item["raw_prefix_bytes"] != PREFIX_BYTES
                or item["stored_file_bytes"] != PREFIX_BYTES or item["raw_prefix_sha256"] != digest
                or item["stored_file_sha256"] != digest
                or item["source_total_bytes_from_content_range"] != total_bytes
                or item["transport"]["http_status"] != 206
                or item["transport"]["content_range"] != "bytes 0-65535/%d" % total_bytes):
            raise ValueError("Tokenizer prefix source/range differs: " + name)
        data = _verified_bytes(root / filename, {"bytes": PREFIX_BYTES, "sha256": digest})
        fields[name] = tokenizer_prefix_fields(data, item["model_section_byte_offset_in_prefix"])
        if _canonical_sha256(fields[name]) != _canonical_sha256(item["derived_fields"]):
            raise ValueError("Derived tokenizer fields differ from original prefix: " + name)
        sources[name] = {"source_url": item["source_url"], "revision": revision,
                         "prefix_bytes": PREFIX_BYTES, "prefix_sha256": digest,
                         "declared_full_file_bytes": total_bytes, "complete_tokenizer": False}
    four, small = fields["LightOnOCR-3-4B"], fields["LightOnOCR-3-0.8B"]
    control_tokens = {"<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|vision_start|>",
                      "<|vision_end|>", "<|image_pad|>", "<think>", "</think>"}
    comparison = {
        "normalizer_equal": four["normalizer"] == small["normalizer"],
        "added_tokens_equal": four["added_tokens"] == small["added_tokens"],
        "added_token_count": len(four["added_tokens"]),
        "differing_top_level_metadata_fields": {
            key: {"LightOnOCR-3-4B": four[key], "LightOnOCR-3-0.8B": small[key]}
            for key in sorted(four) if four[key] != small[key]
        },
        "selected_control_token_ids": {item["content"]: item["id"] for item in four["added_tokens"]
                                       if item["content"] in control_tokens},
    }
    if _canonical_sha256(comparison) != _canonical_sha256(report["derived_comparison"]):
        raise ValueError("Tokenizer prefix comparison differs from captured fields")
    license_source = report["license"]
    if (license_source["repository"] != "lightonai/LightOnOCR"
            or license_source["revision"] != LICENSE_REVISION
            or license_source["stored_file"] != "LightOnOCR-Apache-2.0-LICENSE.txt"
            or license_source["bytes"] != 11358 or license_source["sha256"] != LICENSE_SHA256
            or license_source["git_blob_sha1"] != LICENSE_BLOB):
        raise ValueError("Official repository license source differs")
    license_bytes = _verified_bytes(root / license_source["stored_file"],
                                    {"bytes": 11358, "sha256": LICENSE_SHA256})
    blob = hashlib.sha1(b"blob " + str(len(license_bytes)).encode() + b"\0" + license_bytes).hexdigest()
    if blob != LICENSE_BLOB:
        raise ValueError("Official repository license Git blob differs")
    return {
        "audit_sha256": PREFIX_AUDIT_SHA256, "scope": report["scope"],
        "sources": sources, "comparison": comparison,
        "full_tokenizer_verified_by_prefix": False, "tokenizer_runtime_qualified": False,
        "official_repository_license": {
            "repository": license_source["repository"], "revision": LICENSE_REVISION,
            "source_url": license_source["source_url"], "sha256": LICENSE_SHA256,
            "git_blob_sha1": LICENSE_BLOB, "scope": license_source["scope"],
        },
    }


def expected_tensor_schema(config):
    """Derive all checkpoint shapes independently of the captured header."""
    text, vision = validate_config(config)
    hidden = text["hidden_size"]
    ffn = text["intermediate_size"]
    key_heads = text["linear_num_key_heads"]
    value_heads = text["linear_num_value_heads"]
    key_width = text["linear_key_head_dim"]
    value_width = text["linear_value_head_dim"]
    linear_channels = 2 * key_heads * key_width + value_heads * value_width
    query_width = text["num_attention_heads"] * text["head_dim"]
    kv_width = text["num_key_value_heads"] * text["head_dim"]
    schema = {
        "model.language_model.embed_tokens.weight": [text["vocab_size"], hidden],
        "model.language_model.norm.weight": [hidden],
    }
    for index, kind in enumerate(text["layer_types"]):
        prefix = "model.language_model.layers.%d." % index
        fields = {
            "input_layernorm.weight": [hidden],
            "post_attention_layernorm.weight": [hidden],
            "mlp.gate_proj.weight": [ffn, hidden],
            "mlp.up_proj.weight": [ffn, hidden],
            "mlp.down_proj.weight": [hidden, ffn],
        }
        if kind == "linear_attention":
            fields.update({
                "linear_attn.in_proj_qkv.weight": [linear_channels, hidden],
                "linear_attn.in_proj_z.weight": [value_heads * value_width, hidden],
                "linear_attn.in_proj_a.weight": [value_heads, hidden],
                "linear_attn.in_proj_b.weight": [value_heads, hidden],
                "linear_attn.out_proj.weight": [hidden, value_heads * value_width],
                "linear_attn.conv1d.weight": [linear_channels, 1, text["linear_conv_kernel_dim"]],
                "linear_attn.A_log": [value_heads],
                "linear_attn.dt_bias": [value_heads],
                "linear_attn.norm.weight": [value_width],
            })
        else:
            fields.update({
                "self_attn.q_proj.weight": [2 * query_width, hidden],
                "self_attn.k_proj.weight": [kv_width, hidden],
                "self_attn.v_proj.weight": [kv_width, hidden],
                "self_attn.o_proj.weight": [hidden, query_width],
                "self_attn.q_norm.weight": [text["head_dim"]],
                "self_attn.k_norm.weight": [text["head_dim"]],
            })
        schema.update((prefix + name, shape) for name, shape in fields.items())

    width = vision["hidden_size"]
    merged = width * vision["spatial_merge_size"] ** 2
    output = vision["out_hidden_size"]
    schema.update({
        "model.visual.patch_embed.proj.weight": [
            width, vision["in_channels"], vision["temporal_patch_size"],
            vision["patch_size"], vision["patch_size"],
        ],
        "model.visual.patch_embed.proj.bias": [width],
        "model.visual.pos_embed.weight": [vision["num_position_embeddings"], width],
        "model.visual.merger.norm.weight": [width],
        "model.visual.merger.norm.bias": [width],
        "model.visual.merger.linear_fc1.weight": [merged, merged],
        "model.visual.merger.linear_fc1.bias": [merged],
        "model.visual.merger.linear_fc2.weight": [output, merged],
        "model.visual.merger.linear_fc2.bias": [output],
    })
    for index in range(vision["depth"]):
        prefix = "model.visual.blocks.%d." % index
        fields = {
            "norm1.weight": [width], "norm1.bias": [width],
            "norm2.weight": [width], "norm2.bias": [width],
            "attn.qkv.weight": [3 * width, width], "attn.qkv.bias": [3 * width],
            "attn.proj.weight": [width, width], "attn.proj.bias": [width],
            "mlp.linear_fc1.weight": [vision["intermediate_size"], width],
            "mlp.linear_fc1.bias": [vision["intermediate_size"]],
            "mlp.linear_fc2.weight": [width, vision["intermediate_size"]],
            "mlp.linear_fc2.bias": [width],
        }
        schema.update((prefix + name, shape) for name, shape in fields.items())
    return schema


def validate_tensor_contract(header, config):
    """Check every shape, dtype and byte interval without reading tensor data."""
    expected = expected_tensor_schema(config)
    if not isinstance(header, dict) or header.get("__metadata__") != {"format": "pt"}:
        raise ValueError("Safetensors header metadata differs")
    actual = {name: item for name, item in header.items() if name != "__metadata__"}
    missing, extra = sorted(set(expected) - set(actual)), sorted(set(actual) - set(expected))
    if missing or extra:
        raise ValueError("Tensor census differs: missing=%s extra=%s" % (missing, extra))
    intervals = []
    parameters = 0
    for name, shape in expected.items():
        item = actual[name]
        if (not isinstance(item, dict) or set(item) != {"dtype", "shape", "data_offsets"}
                or item["dtype"] != "BF16" or item["shape"] != shape
                or not isinstance(item["shape"], list)
                or any(type(value) is not int or value < 1 for value in item["shape"])):
            raise ValueError("Tensor shape/dtype differs: " + name)
        offsets = item["data_offsets"]
        count = math.prod(shape)
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(type(value) is not int or value < 0 for value in offsets)
                or offsets[1] - offsets[0] != count * 2):
            raise ValueError("Tensor byte interval differs: " + name)
        intervals.append((offsets[0], offsets[1], name))
        parameters += count
    end = 0
    for start, stop, name in sorted(intervals):
        if start != end:
            raise ValueError("Tensor byte intervals overlap or leave a gap: " + name)
        end = stop
    if (len(actual) != TENSOR_COUNT or parameters != PARAMETER_COUNT
            or end + HEADER_BYTES + 8 != CHECKPOINT_BYTES):
        raise ValueError("Tensor count/parameter count/checkpoint length differs")
    return {
        "tensor_count": len(actual), "parameter_count": parameters, "dtype": "BF16",
        "payload_bytes_from_header": end, "tied_head_stored_once": True,
        "mtp_tensor_count": 0, "payload_verified": False,
    }


def validate_header_bytes(data, config):
    if len(data) != HEADER_BYTES or hashlib.sha256(data).hexdigest() != HEADER_SHA256:
        raise ValueError("LightOnOCR-3-4B original header bytes changed")
    return validate_tensor_contract(_json(data), config)


def native_blockers(config):
    """Source-derived work items, not a claim of measured native behavior."""
    text, vision = validate_config(config)
    key_heads = text["linear_num_key_heads"]
    value_heads = text["linear_num_value_heads"]
    vocabulary_elements = text["vocab_size"] * text["hidden_size"]
    source_url = f"https://github.com/CrispStrobe/CrispEmbed/blob/{AUDITED_NATIVE_REVISION}"
    return [
        {
            "id": "unequal_gdn_heads", "key_heads": key_heads, "value_heads": value_heads,
            "qk_repeat_factor": value_heads // key_heads,
            "checkpoint_qkv_channels": 2 * key_heads * text["linear_key_head_dim"]
                                       + value_heads * text["linear_value_head_dim"],
            "current_equal_head_channels": 3 * value_heads * text["linear_value_head_dim"],
            "files": [f"{source_url}/src/qwen35_linear.cpp", f"{source_url}/src/qwen35_delta.cpp",
                      f"{source_url}/src/qwen35_decoder.cpp"],
            "required": "Implement and independently compare unequal-head Q/K expansion, convolution and recurrent-cache geometry; preserve the existing equal-head path.",
        },
        {
            "id": "fixed_small_model_shapes", "tensor_count": TENSOR_COUNT,
            "decoder_layers": text["num_hidden_layers"], "decoder_width": text["hidden_size"],
            "vision_layers": vision["depth"], "vision_width": vision["hidden_size"],
            "files": [f"{source_url}/src/qwen35_ocr.cpp", f"{source_url}/src/qwen35_decoder.cpp",
                      f"{source_url}/src/qwen35_vision.cpp"],
            "required": "Add explicit pinned 4B mapping, shape validation and attention/vision parameters; current owning context admits 473 tensors and 1024-wide text only.",
        },
        {
            "id": "vocabulary_projection_limit", "head_elements": vocabulary_elements,
            "current_projection_element_limit": 512 * 1024 * 1024,
            "head_f32_bytes": vocabulary_elements * 4,
            "files": [f"{source_url}/src/qwen35_linear.cpp"],
            "required": "Implement a bounded vocabulary projection and measure memory while preserving every vocabulary logit; the head alone exceeds project_cpu's element limit.",
        },
        {
            "id": "distinct_frontend_and_recipe",
            "files": [f"{source_url}/src/qwen35_tokenizer.cpp", f"{source_url}/src/qwen35_frontend.h",
                      f"{source_url}/models/convert-qwen35-ocr-to-gguf.py"],
            "required": "Audit exact 4B tokenizer/processor/template, stop policy and source admission. Do not inherit the 0.8B tokenizer or its 12-layer vision-f32 recipe for this 24-layer vision tower.",
        },
        {
            "id": "original_native_evidence_pending",
            "required": "Pin the loaded original framework and an explicit generation/rendering recipe; capture original stages, caches, all logits and decoded output before native comparison. F32/F16, memory/long-page and later quantized/public-consumer evidence remain separate.",
        },
    ]


def load_profile(model_dir=FIXTURE, source_model=MODEL, source_revision=REVISION,
                 verify_tokenizer=False):
    """Run a source-only preflight. The full tokenizer is optional, never downloaded."""
    if source_model != MODEL or source_revision != REVISION:
        raise ValueError("LightOnOCR-3-4B source identity differs")
    report = evidence()
    prefix_evidence = tokenizer_prefix_evidence()
    root = Path(model_dir)
    if (root / "model.safetensors.index.json").exists():
        raise ValueError("Pinned LightOnOCR-3-4B is a single file, not a sharded checkpoint")
    raw = {name: _verified_bytes(root / name, report["source_files"][name])
           for name in METADATA_FILES + (HEADER_FILE,)}
    config = _json(raw["config.json"])
    generation = _json(raw["generation_config.json"])
    text, vision = validate_config(config)
    validate_generation(generation)
    census = validate_header_bytes(raw[HEADER_FILE], config)
    if verify_tokenizer:
        _verified_bytes(root / "tokenizer.json", {"bytes": TOKENIZER_BYTES, "sha256": TOKENIZER_SHA256})
    return {
        "schema": "lightonocr3-4b-source-preflight/v1", "name": PROFILE,
        "source_model": MODEL, "source_revision": REVISION,
        "source_audit_sha256": AUDIT_SHA256,
        "scope": "pinned metadata and actual header; no weight payload or model execution",
        "verified_source_files": {name: report["source_files"][name]["sha256"] for name in raw},
        "checkpoint": {
            "file": "model.safetensors", "declared_bytes": CHECKPOINT_BYTES,
            "declared_sha256": CHECKPOINT_SHA256, "header_sha256": HEADER_SHA256,
            "header_bytes": HEADER_BYTES, "tensor_contract": census, "payload_verified": False,
        },
        "generation": generation,
        "source_text_config_eos_id": text["eos_token_id"],
        "source_top_level_eos_id": config["eos_token_id"],
        "framework_version_declared": config["transformers_version"],
        "framework_runtime_observed": False,
        "native_source_review_revision": AUDITED_NATIVE_REVISION,
        "tokenizer": {
            "declared_bytes": TOKENIZER_BYTES, "declared_sha256": TOKENIZER_SHA256,
            "payload_verified": bool(verify_tokenizer), "runtime_qualified": False,
            "config": _json(raw["tokenizer_config.json"]),
            "prefix_evidence": prefix_evidence,
        },
        "image_processor": _json(raw["processor_config.json"])["image_processor"],
        "geometry": {
            "decoder_layers": text["num_hidden_layers"], "hidden_size": text["hidden_size"],
            "vision_layers": vision["depth"], "vision_hidden_size": vision["hidden_size"],
            "linear_key_heads": text["linear_num_key_heads"],
            "linear_value_heads": text["linear_num_value_heads"],
            "full_attention_query_heads": text["num_attention_heads"],
            "full_attention_kv_heads": text["num_key_value_heads"],
        },
        "native_supported": False, "native_qualified": False,
        "export_qualified": False, "native_alias_qualified": False,
        "image_max_qualified": False, "full_document_qualified": False,
        "blockers": native_blockers(config),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, default=FIXTURE,
                        help="Pinned metadata plus the extracted original header; no weights needed")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--verify-tokenizer", action="store_true",
                        help="Also hash an already-present tokenizer.json; never downloads it")
    args = parser.parse_args(argv)
    try:
        result = load_profile(args.metadata_dir, args.model, args.revision, args.verify_tokenizer)
    except (ValueError, KeyError, OSError, TypeError) as error:
        parser.exit(1, "LightOnOCR-3-4B source preflight failed: %s\n" % error)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
