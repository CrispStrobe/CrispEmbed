#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 CrispStrobe
"""Convert TeleOCR safetensors to the native CrispEmbed GGUF profile.

This reads JSON and safetensors data, never imports checkpoint Python or invokes
``trust_remote_code``. NumPy and the ``gguf`` package are sufficient for local
conversion. Install ``huggingface_hub`` only when downloading from the Hub.

Examples::

    python models/convert-teleocr-to-gguf.py --model ./TeleOCR \
        --output teleocr-f16.gguf --dtype f16
    python models/convert-teleocr-to-gguf.py --model XingChen-AGI/TeleOCR \
        --revision b3f6a0c084db72f971e19a779d2fd8a6bc9e3b8d \
        --output teleocr-q8_0.gguf --dtype q8_0

Q8_0 quantizes decoder attention/FFN matrices only. Vision, the spatial merger,
patch projection and tied embedding/output matrix remain F16; vectors remain
F32. A precision recipe is not a quality claim: compare decoded output against
the F32/F16 reference before selecting a quantized model for a task.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
import tempfile
from typing import Any

import numpy as np


ARCH = "teleocr"
PROFILE_VERSION = 1
MODEL_ID = "XingChen-AGI/TeleOCR"
MODEL_REVISION = "b3f6a0c084db72f971e19a779d2fd8a6bc9e3b8d"
DEFAULT_PROMPT = "Please output the text content from the image."


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    if not isinstance(result, dict):
        raise ValueError(f"{path.name}: expected a JSON object")
    return result


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value > 0x7FFFFFFF:
        raise ValueError(f"{field}: expected a positive int32, got {value!r}")
    return value


@dataclass(frozen=True)
class Profile:
    text: dict[str, Any]
    vision: dict[str, Any]
    image: dict[str, Any]
    special: dict[str, int]


def validate_config(config: dict[str, Any], processor: dict[str, Any]) -> Profile:
    """Validate the supported Qwen2.5 vision + Qwen3-style decoder contract."""
    if config.get("model_type") != "qwen2_5_vl":
        raise ValueError("TeleOCR requires model_type=qwen2_5_vl")
    text = dict(config)
    nested = config.get("text_config", {})
    if not isinstance(nested, dict):
        raise ValueError("text_config must be an object")
    text.update(nested)
    vision = dict(config.get("vision_config", {}))
    for field in ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
                  "num_key_value_heads", "head_dim", "vocab_size", "max_position_embeddings"):
        text[field] = _positive_int(text.get(field), f"text.{field}")
    if text["num_attention_heads"] % text["num_key_value_heads"]:
        raise ValueError("attention heads must be divisible by key/value heads")
    if text.get("hidden_act", "silu") != "silu" or text.get("use_sliding_window", False):
        raise ValueError("only full-attention SiLU TeleOCR text blocks are supported")
    if text.get("tie_word_embeddings") is not True:
        raise ValueError("this TeleOCR profile requires tied word embeddings")
    rope = text.get("rope_scaling", {})
    sections = rope.get("mrope_section")
    if (not isinstance(sections, list) or len(sections) != 3
            or any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in sections)
            or 2 * sum(sections) != text["head_dim"]):
        raise ValueError("three positive mRoPE sections must cover half the attention head dimension")
    if rope.get("rope_type", rope.get("type", "default")) != "default" or rope.get("mrope_interleaved", False):
        raise ValueError("only non-interleaved, unscaled TeleOCR mRoPE is supported")
    text["rope_sections"] = sections
    for field in ("rms_norm_eps", "rope_theta"):
        value = float(text.get(field, 0))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid text.{field}")
        text[field] = value
    vision.setdefault("patch_size", vision.get("spatial_patch_size"))
    vision.setdefault("in_channels", vision.get("in_chans", 3))
    for field in ("depth", "hidden_size", "intermediate_size", "num_heads", "out_hidden_size",
                  "patch_size", "spatial_merge_size", "temporal_patch_size", "window_size", "in_channels"):
        vision[field] = _positive_int(vision.get(field), f"vision.{field}")
    if vision["in_channels"] != 3 or vision["temporal_patch_size"] != 2 or vision["spatial_merge_size"] != 2:
        raise ValueError("TeleOCR supports RGB still images, temporal duplication 2 and spatial merge 2")
    if vision.get("hidden_act", "silu") != "silu" or vision["hidden_size"] % (4 * vision["num_heads"]):
        raise ValueError("vision must use SiLU with a head dimension divisible by four")
    if vision["out_hidden_size"] != text["hidden_size"]:
        raise ValueError("vision merger output must match text hidden size")
    factor = vision["patch_size"] * vision["spatial_merge_size"]
    if vision["window_size"] % factor:
        raise ValueError("vision window_size must be divisible by patch_size * spatial_merge_size")
    fullatt = vision.get("fullatt_block_indexes")
    if (not isinstance(fullatt, list) or not fullatt or len(fullatt) != len(set(fullatt))
            or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 or i >= vision["depth"] for i in fullatt)):
        raise ValueError("invalid vision full-attention layer indexes")
    image = dict(processor)
    for field, expected in (("patch_size", vision["patch_size"]),
                            ("temporal_patch_size", vision["temporal_patch_size"]),
                            ("merge_size", vision["spatial_merge_size"])):
        if image.get(field) != expected:
            raise ValueError(f"processor.{field} disagrees with the model")
    for field in ("image_mean", "image_std"):
        vals = image.get(field)
        if not isinstance(vals, list) or len(vals) != 3 or not all(math.isfinite(float(v)) for v in vals):
            raise ValueError(f"processor.{field} must contain three finite numbers")
        image[field] = [float(v) for v in vals]
    if any(v <= 0 for v in image["image_std"]):
        raise ValueError("processor.image_std must be positive")
    for field in ("min_pixels", "max_pixels"):
        image[field] = _positive_int(image.get(field), f"processor.{field}")
    if image["min_pixels"] > image["max_pixels"] or image["max_pixels"] < factor * factor:
        raise ValueError("invalid processor pixel limits")
    special = {}
    for field in ("image_token_id", "vision_start_token_id", "vision_end_token_id", "eos_token_id"):
        value = config.get(field, text.get(field))
        if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value >= text["vocab_size"]:
            raise ValueError(f"invalid {field}")
        special[field] = value
    return Profile(text, vision, image, special)


@dataclass(frozen=True)
class TensorSpec:
    source: str
    target: str
    shape: tuple[int, ...]
    output_shape: tuple[int, ...]


def expected_tensors(profile: Profile) -> dict[str, TensorSpec]:
    """Return the exhaustive tensor schema, in checkpoint (row-major) order."""
    t, v = profile.text, profile.vision
    result: dict[str, TensorSpec] = {}

    def add(source: str, target: str, shape: tuple[int, ...], output_shape: tuple[int, ...] | None = None):
        result[source] = TensorSpec(source, target, shape, output_shape or shape)

    d, h, hd, kv, inter = (t[k] for k in ("hidden_size", "num_attention_heads", "head_dim",
                                         "num_key_value_heads", "intermediate_size"))
    add("model.embed_tokens.weight", "l.embed_tokens.weight", (t["vocab_size"], d))
    add("model.norm.weight", "l.output_norm.weight", (d,))
    for i in range(t["num_hidden_layers"]):
        src, dst = f"model.layers.{i}.", f"l.blk.{i}."
        for name, target, shape in (
            ("input_layernorm.weight", "attn_norm.weight", (d,)),
            ("post_attention_layernorm.weight", "ffn_norm.weight", (d,)),
            ("self_attn.q_proj.weight", "attn_q.weight", (h * hd, d)),
            ("self_attn.k_proj.weight", "attn_k.weight", (kv * hd, d)),
            ("self_attn.v_proj.weight", "attn_v.weight", (kv * hd, d)),
            ("self_attn.o_proj.weight", "attn_o.weight", (d, h * hd)),
            ("self_attn.q_norm.weight", "attn_q_norm.weight", (hd,)),
            ("self_attn.k_norm.weight", "attn_k_norm.weight", (hd,)),
            ("mlp.gate_proj.weight", "ffn_gate.weight", (inter, d)),
            ("mlp.up_proj.weight", "ffn_up.weight", (inter, d)),
            ("mlp.down_proj.weight", "ffn_down.weight", (d, inter)),
        ):
            add(src + name, dst + target, shape)
    vh, vi, patch, temporal = (v[k] for k in ("hidden_size", "intermediate_size", "patch_size", "temporal_patch_size"))
    add("visual.patch_embed.proj.weight", "v.patch_embed.weight", (vh, 3, temporal, patch, patch),
        (vh, 3 * temporal * patch * patch))
    for i in range(v["depth"]):
        src, dst = f"visual.blocks.{i}.", f"v.blk.{i}."
        for name, target, shape in (
            ("norm1.weight", "norm1.weight", (vh,)),
            ("norm2.weight", "norm2.weight", (vh,)),
            ("attn.qkv.weight", "attn_qkv.weight", (3 * vh, vh)),
            ("attn.qkv.bias", "attn_qkv.bias", (3 * vh,)),
            ("attn.proj.weight", "attn_proj.weight", (vh, vh)),
            ("attn.proj.bias", "attn_proj.bias", (vh,)),
            ("mlp.gate_proj.weight", "ffn_gate.weight", (vi, vh)),
            ("mlp.gate_proj.bias", "ffn_gate.bias", (vi,)),
            ("mlp.up_proj.weight", "ffn_up.weight", (vi, vh)),
            ("mlp.up_proj.bias", "ffn_up.bias", (vi,)),
            ("mlp.down_proj.weight", "ffn_down.weight", (vh, vi)),
            ("mlp.down_proj.bias", "ffn_down.bias", (vh,)),
        ):
            add(src + name, dst + target, shape)
    merged = vh * v["spatial_merge_size"] ** 2
    for name, target, shape in (
        ("ln_q.weight", "norm.weight", (vh,)),
        ("mlp.0.weight", "fc1.weight", (merged, merged)),
        ("mlp.0.bias", "fc1.bias", (merged,)),
        ("mlp.2.weight", "fc2.weight", (d, merged)),
        ("mlp.2.bias", "fc2.bias", (d,)),
    ):
        add("visual.merger." + name, "v.merger." + target, shape)
    return result


@dataclass(frozen=True)
class StoredTensor:
    path: Path
    offset: int
    nbytes: int
    dtype: str
    shape: tuple[int, ...]


DTYPES = {"BF16": ("<u2", 2), "F16": ("<f2", 2), "F32": ("<f4", 4)}


def read_safetensors_header(path: Path) -> dict[str, StoredTensor]:
    """Validate the safe, non-executable format and retain only mmap descriptors."""
    size = path.stat().st_size
    with path.open("rb") as stream:
        raw_len = stream.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"{path.name}: truncated safetensors header")
        header_len = struct.unpack("<Q", raw_len)[0]
        if header_len < 2 or header_len > min(100_000_000, size - 8):
            raise ValueError(f"{path.name}: invalid safetensors header length")
        header = json.loads(stream.read(header_len), object_pairs_hook=_unique_object)
    if not isinstance(header, dict):
        raise ValueError(f"{path.name}: invalid safetensors header")
    result, spans = {}, []
    for name, desc in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(desc, dict) or desc.get("dtype") not in DTYPES:
            raise ValueError(f"{name}: expected F32, F16 or BF16 tensor data")
        shape = desc.get("shape", [])
        if not isinstance(shape, list) or not shape:
            raise ValueError(f"{name}: empty/invalid tensor shape")
        shape = tuple(_positive_int(x, f"{name}.shape") for x in shape)
        offsets = desc.get("data_offsets")
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in offsets)):
            raise ValueError(f"{name}: invalid tensor offsets")
        start, end = offsets
        nbytes = math.prod(shape) * DTYPES[desc["dtype"]][1]
        if end - start != nbytes or end + header_len + 8 > size:
            raise ValueError(f"{name}: shape, offsets and file size disagree")
        spans.append((start, end))
        result[name] = StoredTensor(path, 8 + header_len + start, nbytes, desc["dtype"], shape)
    previous = 0
    for start, end in sorted(spans):
        if start != previous:
            raise ValueError(f"{path.name}: overlapping or non-contiguous tensor data")
        previous = end
    if previous != size - header_len - 8:
        raise ValueError(f"{path.name}: trailing or missing tensor data")
    return result


def load_tensor_index(model_dir: Path) -> dict[str, StoredTensor]:
    index_file = model_dir / "model.safetensors.index.json"
    index = read_json(index_file).get("weight_map") if index_file.exists() else None
    if index is not None and (not isinstance(index, dict) or not index):
        raise ValueError("invalid safetensors weight_map")
    shards = sorted(set(index.values())) if index else ["model.safetensors"]
    result = {}
    for shard in shards:
        if not isinstance(shard, str) or Path(shard).name != shard or not shard.endswith(".safetensors"):
            raise ValueError("safetensors shards must be filenames within the model directory")
        tensors = read_safetensors_header(model_dir / shard)
        for name, tensor in tensors.items():
            if name in result:
                raise ValueError(f"duplicate tensor across shards: {name}")
            if index and index.get(name) != shard:
                raise ValueError(f"safetensors index disagrees for {name}")
            result[name] = tensor
    if index is not None and set(result) != set(index):
        raise ValueError("safetensors index contains missing tensors")
    return result


def tensor_f32(tensor: StoredTensor) -> np.ndarray:
    mapped = np.memmap(tensor.path, mode="r", dtype=DTYPES[tensor.dtype][0],
                       offset=tensor.offset, shape=tensor.shape)
    if tensor.dtype == "BF16":
        result = (np.asarray(mapped).astype(np.uint32) << 16).view(np.float32)
    else:
        result = np.array(mapped, dtype=np.float32, copy=True)
    del mapped
    if not np.isfinite(result).all():
        raise ValueError("checkpoint contains non-finite weights")
    return result


def tensor_sha256(tensor: StoredTensor) -> str:
    digest = hashlib.sha256()
    with tensor.path.open("rb") as stream:
        stream.seek(tensor.offset)
        remaining = tensor.nbytes
        while remaining:
            data = stream.read(min(4 * 1024 * 1024, remaining))
            if not data:
                raise ValueError("tensor data changed during conversion")
            digest.update(data)
            remaining -= len(data)
    return digest.hexdigest()


def validate_tensors(profile: Profile, index: dict[str, StoredTensor]) -> list[TensorSpec]:
    expected = expected_tensors(profile)
    extra = set(index) - set(expected) - {"lm_head.weight"}
    missing = set(expected) - set(index)
    if extra or missing:
        raise ValueError(f"tensor census mismatch: missing={sorted(missing)}, unexpected={sorted(extra)}")
    for name, spec in expected.items():
        if index[name].shape != spec.shape:
            raise ValueError(f"{name}: expected {spec.shape}, got {index[name].shape}")
    if "lm_head.weight" in index:
        head, embed = index["lm_head.weight"], index["model.embed_tokens.weight"]
        if head.shape != embed.shape or head.dtype != embed.dtype or tensor_sha256(head) != tensor_sha256(embed):
            raise ValueError("tie_word_embeddings=true but lm_head and token embeddings are not byte-identical")
    return sorted(expected.values(), key=lambda s: s.target)


def tokenizer_arrays(tokenizer: dict[str, Any], profile: Profile) -> tuple[list[str], list[int], list[str]]:
    model = tokenizer.get("model", {})
    if model.get("type") != "BPE" or not isinstance(model.get("vocab"), dict):
        raise ValueError("TeleOCR requires a BPE tokenizer.json with explicit vocabulary")
    n_vocab = profile.text["vocab_size"]
    # Rows beyond the actual tokenizer vocabulary are reserved by the model;
    # empty entries retain their IDs and decode to nothing, never a fake token.
    tokens = [""] * n_vocab
    token_types = [5] * n_vocab  # UNUSED, per GGUF TokenType
    assigned: dict[int, str] = {}

    def add(token: str, token_id: int, special: bool = False):
        if (not isinstance(token, str) or isinstance(token_id, bool) or not isinstance(token_id, int)
                or token_id < 0 or token_id >= n_vocab):
            raise ValueError(f"invalid tokenizer entry: {token!r}={token_id!r}")
        if token_id in assigned and assigned[token_id] != token:
            raise ValueError(f"duplicate tokenizer ID {token_id}")
        assigned[token_id] = token
        tokens[token_id], token_types[token_id] = token, 3 if special else 1

    for token, token_id in model["vocab"].items():
        add(token, token_id)
    for item in tokenizer.get("added_tokens", []):
        add(item["content"], item["id"], bool(item.get("special")))
    for field, token in (("image_token_id", "<|image_pad|>"), ("vision_start_token_id", "<|vision_start|>"),
                         ("vision_end_token_id", "<|vision_end|>"), ("eos_token_id", "<|im_end|>")):
        if tokens[profile.special[field]] != token:
            raise ValueError(f"tokenizer {field} must name {token}")
    for required in ("<|im_start|>", "<|im_end|>", "<|endoftext|>"):
        if required not in assigned.values():
            raise ValueError(f"missing required tokenizer token {required}")
    merges = []
    for pair in model.get("merges", []):
        if isinstance(pair, list) and len(pair) == 2 and all(isinstance(x, str) for x in pair):
            pair = " ".join(pair)
        if not isinstance(pair, str) or pair.count(" ") != 1:
            raise ValueError("invalid BPE merge entry")
        left, right = pair.split(" ")
        if left not in model["vocab"] or right not in model["vocab"] or left + right not in model["vocab"]:
            raise ValueError(f"BPE merge references a missing token: {pair!r}")
        merges.append(pair)
    if not merges or len(set(merges)) != len(merges):
        raise ValueError("BPE merges must be non-empty and unique")
    return tokens, token_types, merges


def write_metadata(writer: Any, profile: Profile, config: dict[str, Any], tokenizer: dict[str, Any],
                   revision: str, dtype: str) -> None:
    import gguf

    def array(key, value, element_type):
        writer.add_key_value(key, value, gguf.GGUFValueType.ARRAY, sub_type=element_type)

    writer.add_string("general.name", "TeleOCR")
    writer.add_string("general.license", "apache-2.0")
    writer.add_string("general.url", f"https://huggingface.co/{MODEL_ID}")
    writer.add_uint32("general.file_type", {"f32": 0, "f16": 1, "q8_0": 7}[dtype])
    if dtype == "q8_0":
        writer.add_uint32("general.quantization_version", gguf.GGML_QUANT_VERSION)
    writer.add_uint32("teleocr.gguf_version", PROFILE_VERSION)
    writer.add_string("teleocr.source.repository", MODEL_ID)
    writer.add_string("teleocr.source.revision", revision)
    writer.add_string("teleocr.config", json.dumps(config, separators=(",", ":")))
    writer.add_string("teleocr.preprocessor", json.dumps(profile.image, separators=(",", ":")))
    writer.add_string("teleocr.default_prompt", DEFAULT_PROMPT)
    writer.add_float32("teleocr.repetition_penalty", 1.05)
    writer.add_string("teleocr.precision_recipe", "decoder-q8_0-vision-f16" if dtype == "q8_0" else dtype)
    for field in ("vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
                  "num_attention_heads", "num_key_value_heads", "max_position_embeddings"):
        writer.add_uint32("qwen2vl." + field, profile.text[field])
    writer.add_uint32("qwen2vl.attention.head_dim", profile.text["head_dim"])
    writer.add_bool("qwen2vl.has_qk_norm", True)
    writer.add_bool("qwen2vl.tie_word_embeddings", True)
    for field in ("rms_norm_eps", "rope_theta"):
        writer.add_float32("qwen2vl." + field, profile.text[field])
    array("qwen2vl.rope_sections", profile.text["rope_sections"], gguf.GGUFValueType.UINT32)
    for field, value in profile.special.items():
        writer.add_uint32("qwen2vl." + field, value)
    for field in ("depth", "hidden_size", "intermediate_size", "num_heads", "in_channels",
                  "spatial_merge_size", "temporal_patch_size", "out_hidden_size", "window_size"):
        writer.add_uint32("qwen2vl.vision." + field, profile.vision[field])
    writer.add_uint32("qwen2vl.vision.spatial_patch_size", profile.vision["patch_size"])
    array("qwen2vl.vision.fullatt_block_indexes", profile.vision["fullatt_block_indexes"],
          gguf.GGUFValueType.UINT32)
    for field in ("image_mean", "image_std"):
        array("qwen2vl.vision." + field, profile.image[field], gguf.GGUFValueType.FLOAT32)
    for field in ("min_pixels", "max_pixels"):
        writer.add_uint32("qwen2vl.vision." + field, profile.image[field])
    tokens, token_types, merges = tokenizer_arrays(tokenizer, profile)
    writer.add_string("tokenizer.ggml.model", "gpt2")
    writer.add_string("tokenizer.ggml.pre", "qwen2")
    array("tokenizer.ggml.tokens", tokens, gguf.GGUFValueType.STRING)
    array("tokenizer.ggml.token_type", token_types, gguf.GGUFValueType.INT32)
    array("tokenizer.ggml.merges", merges, gguf.GGUFValueType.STRING)
    writer.add_uint32("tokenizer.ggml.eos_token_id", profile.special["eos_token_id"])
    writer.add_uint32("tokenizer.ggml.padding_token_id", tokens.index("<|endoftext|>"))
    writer.add_bool("tokenizer.ggml.add_bos_token", False)
    writer.add_bool("tokenizer.ggml.add_eos_token", False)


def tensor_type(spec: TensorSpec, dtype: str) -> str:
    if len(spec.output_shape) == 1 or dtype == "f32":
        return "F32"
    if dtype == "q8_0" and spec.target.startswith("l.blk."):
        if spec.output_shape[-1] % 32:
            raise ValueError(f"{spec.source}: decoder row length cannot use Q8_0")
        return "Q8_0"
    return "F16"


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def convert(model_dir: Path, output: Path, dtype: str = "f16", revision: str = MODEL_REVISION,
            force: bool = False, manifest: Path | None = None) -> dict[str, Any]:
    import gguf

    if dtype not in ("f32", "f16", "q8_0"):
        raise ValueError(f"unsupported precision: {dtype}")
    model_dir, output = Path(model_dir), Path(output)
    manifest = Path(manifest) if manifest else Path(str(output) + ".json")
    if output.resolve() == manifest.resolve():
        raise ValueError("GGUF output and manifest must be different files")
    if not force and (output.exists() or manifest.exists()):
        raise FileExistsError("output or manifest already exists; pass --force to replace")
    config = read_json(model_dir / "config.json")
    processor = read_json(model_dir / "preprocessor_config.json")
    tokenizer = read_json(model_dir / "tokenizer.json")
    profile = validate_config(config, processor)
    tokenizer_arrays(tokenizer, profile)
    index = load_tensor_index(model_dir)
    specs = validate_tensors(profile, index)
    source_names = ["config.json", "preprocessor_config.json", "tokenizer.json"]
    source_names += sorted({tensor.path.name for tensor in index.values()})
    source_names += [name for name in ("generation_config.json", "tokenizer_config.json", "model.safetensors.index.json")
                     if (model_dir / name).is_file()]
    report: dict[str, Any] = {
        "schema_version": 1, "architecture": ARCH, "profile_version": PROFILE_VERSION,
        "source_repository": MODEL_ID, "source_revision": revision,
        "source_files": {name: file_sha256(model_dir / name) for name in source_names},
        "source_tensor_count": len(index), "output_tensor_count": len(specs),
        "tied_head_omitted": "lm_head.weight" in index, "precision": dtype,
        "tensor_types": dict(Counter(tensor_type(spec, dtype) for spec in specs)), "tensors": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=output.name + ".", suffix=".tmp", dir=output.parent)
    os.close(fd)
    writer = None
    try:
        writer = gguf.GGUFWriter(temporary, ARCH)
        write_metadata(writer, profile, config, tokenizer, revision, dtype)
        types = {}
        for spec in specs:
            dt_name = tensor_type(spec, dtype)
            dt = getattr(gguf.GGMLQuantizationType, dt_name)
            count = math.prod(spec.output_shape)
            nbytes = count // 32 * 34 if dt_name == "Q8_0" else count * (4 if dt_name == "F32" else 2)
            np_dtype = np.dtype(np.uint8 if dt_name == "Q8_0" else np.float32 if dt_name == "F32" else np.float16)
            # gguf expects the byte-array shape when tensor_dtype is uint8;
            # it expands the packed last axis back to the logical row width.
            shape = (*spec.output_shape[:-1], spec.output_shape[-1] // 32 * 34) if dt_name == "Q8_0" else spec.output_shape
            writer.add_tensor_info(spec.target, shape, np_dtype, nbytes, raw_dtype=dt)
            types[spec.target] = dt
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        for i, spec in enumerate(specs):
            array = tensor_f32(index[spec.source]).reshape(spec.output_shape)
            dt_name = tensor_type(spec, dtype)
            if dt_name == "Q8_0":
                array = gguf.quantize(array, types[spec.target])
            elif dt_name == "F16":
                array = array.astype(np.float16)
                if not np.isfinite(array).all():
                    raise ValueError(f"{spec.source}: weights overflow F16")
            array = np.ascontiguousarray(array)
            writer.write_tensor_data(array)
            report["tensors"].append({
                "source": spec.source, "target": spec.target,
                "source_shape": list(spec.shape), "shape": list(spec.output_shape),
                "source_dtype": index[spec.source].dtype, "dtype": dt_name,
                "nbytes": array.nbytes,
                "sha256": hashlib.sha256(memoryview(array).cast("B")).hexdigest(),
            })
            del array
            if (i + 1) % 50 == 0 or i + 1 == len(specs):
                print(f"Converted {i + 1}/{len(specs)} tensors", flush=True)
        writer.close()
        writer = None
        report["output_size"] = Path(temporary).stat().st_size
        report["output_sha256"] = file_sha256(Path(temporary))
        report["output_file"] = output.name
        os.replace(temporary, output)
        manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report
    finally:
        if writer is not None:
            writer.close()
        if os.path.exists(temporary):
            os.unlink(temporary)


def resolve_model(model: str, revision: str) -> Path:
    path = Path(model)
    if path.is_dir():
        return path
    if path.exists() or model.startswith((".", "/", "~")):
        raise ValueError(f"model directory not found: {model}")
    if model != MODEL_ID:
        raise ValueError(f"this converter downloads only {MODEL_ID}; use a local directory for an inspected checkpoint")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("remote downloads require an immutable 40-character revision")
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model, revision=revision, allow_patterns=[
        "config.json", "preprocessor_config.json", "tokenizer.json", "*.safetensors", "*.safetensors.index.json",
    ]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=MODEL_ID, help=f"local checkpoint directory or {MODEL_ID}")
    parser.add_argument("--revision", default=MODEL_REVISION, help="immutable source revision (recorded in GGUF)")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dtype", choices=("f32", "f16", "q8_0"), default="f16")
    parser.add_argument("--manifest", type=Path, help="tensor census and hashes; defaults to OUTPUT.json")
    parser.add_argument("--force", action="store_true", help="replace existing output and manifest")
    args = parser.parse_args()
    try:
        report = convert(resolve_model(args.model, args.revision), args.output, args.dtype,
                         args.revision, args.force, args.manifest)
    except (ValueError, OSError, ImportError) as exc:
        print(f"TeleOCR conversion failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({k: report[k] for k in ("source_tensor_count", "output_tensor_count", "tensor_types",
                                            "output_file", "output_size", "output_sha256")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
