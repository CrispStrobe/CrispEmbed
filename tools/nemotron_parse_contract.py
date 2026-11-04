#!/usr/bin/env python3
"""Validate the pinned Nemotron 3.5 Parse source/tensor census without weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "models/nemotron-parse-source.json"
FIXTURES = ROOT / "tests/regression/nemotron-parse"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def expected_tensors():
    out = {}
    def add(name, shape, dtype="F32"):
        out[name] = {"shape": shape, "dtype": dtype}
    def affine(name, outputs, inputs=None):
        add(name + ".weight", [outputs] if inputs is None else [outputs, inputs])
        add(name + ".bias", [outputs])
    radio = "encoder.model_encoder.radio_model."
    add(radio + "summary_idxs", [2], "I64")
    for key in ("norm_mean", "norm_std"):
        add(radio + "input_conditioner." + key, [3, 1, 1])
    patch = radio + "model.patch_generator."
    add(patch + "cls_token.token", [10, 1152])
    add(patch + "embedder.weight", [1152, 768])
    add(patch + "pos_embed", [1, 16384, 1152])
    for index in range(27):
        prefix = radio + f"model.blocks.{index}."
        affine(prefix + "norm1", 1152)
        affine(prefix + "norm2", 1152)
        affine(prefix + "attn.qkv", 3456, 1152)
        affine(prefix + "attn.proj", 1152, 1152)
        affine(prefix + "mlp.fc1", 4304, 1152)
        affine(prefix + "mlp.fc2", 1152, 4304)
    add("encoder.conv1.weight", [1024, 1152, 1])
    add("encoder.conv1.bias", [1024])
    add("encoder.conv2.weight", [1024, 1024, 1, 4])
    for index in (1, 2, 3):
        affine(f"encoder.layer_norm{index}", 1024)
    affine("encoder.sum_proj", 1024, 2304)
    add("decoder.embed_tokens.weight", [72512, 1024])
    affine("decoder.layernorm_embedding", 1024)
    affine("decoder.layer_norm", 1024)
    for index in range(10):
        prefix = f"decoder.layers.{index}."
        for key, outputs in (("q_proj", 1024), ("k_proj", 512), ("v_proj", 512), ("out_proj", 1024)):
            affine(prefix + "self_attn." + key, outputs, 1024)
        affine(prefix + "self_attn_layer_norm", 1024)
        affine(prefix + "final_layer_norm", 1024)
        affine(prefix + "fc1", 4096, 1024)
        affine(prefix + "fc2", 1024, 4096)
    # These four tensors are stored, but unused by ordinary HF greedy generation.
    affine("decoder.extra_heads.0", 1024, 1024)
    affine("decoder.extra_proj.0", 1024, 1024)
    return out


def audit_header(header, *, file_bytes, header_bytes):
    expected = expected_tensors()
    actual = {name: value for name, value in header.items() if name != "__metadata__"}
    require(set(actual) == set(expected),
            f"tensor name mismatch: missing={sorted(set(expected)-set(actual))}, extra={sorted(set(actual)-set(expected))}")
    cursor, elements = 0, 0
    for name, tensor in sorted(actual.items(), key=lambda item: item[1]["data_offsets"][0]):
        wanted = expected[name]
        require(tensor["shape"] == wanted["shape"] and tensor["dtype"] == wanted["dtype"],
                "tensor shape/dtype drift: " + name)
        count = math.prod(tensor["shape"])
        size = count * (8 if tensor["dtype"] == "I64" else 4)
        require(tensor["data_offsets"] == [cursor, cursor + size], "tensor offset/size drift: " + name)
        cursor += size
        elements += count
    require(cursor + header_bytes + 8 == file_bytes, "safetensors payload length mismatch")
    return {"tensor_count": len(actual), "stored_elements": elements,
            "parameter_elements": elements - 8, "optional_mtp_parameter_elements": 2099200,
            "ordinary_hf_parameter_elements": elements - 8 - 2099200}


def audit_config(config, processor):
    require(config["model_type"] == "nemotron_parse" and config["architectures"] == ["RadioMBartPrefixForCausalLM"],
            "wrong architecture")
    for key, expected in {"legacy_causal_vision_prefix": True, "self_attention_kv_heads": 8,
                          "cross_attention_kv_heads": 0, "max_sequence_length": 16384,
                          "num_extra_heads": 1, "navit_position_embedding_mode": "content_normalized_2d_interpolate",
                          "navit_packing": True, "vocab_size": 72512}.items():
        require(config[key] == expected, "config drift: " + key)
    for key, expected in {"decoder_layers": 10, "d_model": 1024, "decoder_ffn_dim": 4096,
                          "decoder_attention_heads": 16, "self_attention_kv_heads": 8,
                          "add_cross_attention": False, "scale_embedding": True}.items():
        require(config["decoder"][key] == expected, "decoder drift: " + key)
    require(processor["navit_packing"] is True and processor["navit_padding_mode"] == "top_left_white"
            and processor["navit_coordinate_frame"] == "content_normalized"
            and processor["final_size"] == [2048, 1664], "processor geometry drift")
    require(processor["image_mean"] == [0.48145466, 0.4578275, 0.40821073]
            and processor["image_std"] == [0.26862954, 0.26130258, 0.27577711], "normalization drift")


def audit(source_directory=None):
    source = json.loads(SOURCE.read_text())
    config = json.loads((FIXTURES / "config.json").read_text())
    processor = json.loads((FIXTURES / "preprocessor_config.json").read_text())
    audit_config(config, processor)
    header = json.loads((FIXTURES / "tensor-header.json").read_text())
    canonical = json.dumps(header, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    require(hashlib.sha256(canonical).hexdigest() == source["weights"]["header_canonical_sha256"],
            "frozen tensor header changed")
    for name in ("config.json", "preprocessor_config.json", "generation_config.json"):
        require(hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest() == source["files"][name]["sha256"],
                "frozen config bytes changed: " + name)
    verified = []
    if source_directory is not None:
        for name, wanted in source["files"].items():
            payload = (source_directory / name).read_bytes()
            require(len(payload) == wanted["bytes"] and hashlib.sha256(payload).hexdigest() == wanted["sha256"],
                    "upstream source drift: " + name)
            verified.append(name)
    report = audit_header(header, file_bytes=source["weights"]["bytes"], header_bytes=source["weights"]["header_bytes"])
    return {"schema": "crispembed-nemotron-parse-contract-v1", "repo": source["repo"],
            "revision": source["revision"], **report, "verified_source_files": verified,
            "source_blueprint_verified": source_directory is not None,
            "full_weight_hash_verified": False, "model_execution": False,
            "model_qualified": False, "publication_qualified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path,
                        help="optional small-file snapshot at the pinned revision; no weights required")
    args = parser.parse_args()
    try:
        print(json.dumps(audit(args.source_directory), indent=2))
        return 0
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(json.dumps({"error": str(exc), "model_qualified": False}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
