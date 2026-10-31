#!/usr/bin/env python3
"""Convert the exact SoMarkAI/OXR-1.0 checkpoint to CrispEmbed GGUF.

No PyTorch, Transformers model, or remote Python is executed by this converter.
Source hashes, all 752 tensor shapes, the tied output matrix, and the complete
151669-entry tokenizer are checked before output is made available. F32 is the
numerical baseline; the F16 recipe retains sensitive tensors in F32 and is not
a quality-qualified quantization. The checkpoint's OXR additional terms apply.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys

MODEL_ID = "SoMarkAI/OXR-1.0"
MODEL_REVISION = "c708407f9d2b4df658f2543d9132b62bb7221df4"
BLUEPRINT_REVISION = "c1a1dcb312ee9f691404f127785510453c0d9350"
FAMILY = "oxr-1.0"
VOCAB_SIZE = 151936
TOKENIZER_SIZE = 151669
MERGE_COUNT = 151387
SPECIAL_TOKENS = dict(enumerate([
    "<|endoftext|>", "<|im_start|>", "<|im_end|>",
    "<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>",
    "<|box_end|>", "<|quad_start|>", "<|quad_end|>",
    "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>",
    "<|image_pad|>", "<|video_pad|>", "<tool_call>", "</tool_call>",
    "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>",
    "<|repo_name|>", "<|file_sep|>", "<tool_response>", "</tool_response>",
    "<think>", "</think>",
], start=151643))
QWEN_PATTERN = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_manifest() -> dict:
    return json.loads(Path(__file__).with_name("oxr-source.json").read_text(encoding="utf-8"))


def verify_source_files(directory: Path, manifest: dict | None = None) -> dict[str, str]:
    manifest = source_manifest() if manifest is None else manifest
    require_equal(manifest.get("model_id"), MODEL_ID, "manifest model")
    require_equal(manifest.get("revision"), MODEL_REVISION, "manifest revision")
    hashes = {}
    for name, expected in manifest["files"].items():
        path = directory / name
        if not path.is_file() or path.stat().st_size != expected["size"]:
            raise ValueError(f"pinned source size/missing-file mismatch: {name}")
        digest = sha256(path)
        if digest != expected["sha256"]:
            raise ValueError(f"pinned source SHA256 mismatch: {name}")
        hashes[name] = digest
    return hashes


def require_equal(actual, expected, field):
    if actual != expected:
        raise ValueError(f"unsupported {field}: {actual!r}; expected {expected!r}")


def validate_config(config: dict, processor: dict, generation: dict) -> None:
    for name, value in {"model_type": "oxr", "architectures": ["OXR"],
                        "hidden_size": 1024, "visual_vocab_size": 65536,
                        "vocab_size": VOCAB_SIZE}.items():
        require_equal(config.get(name), value, name)
    llm = config.get("llm_config", {})
    for name, value in {
        "model_type": "qwen3", "hidden_size": 1024, "intermediate_size": 3072,
        "num_hidden_layers": 28, "num_attention_heads": 16,
        "num_key_value_heads": 8, "head_dim": 128, "vocab_size": VOCAB_SIZE,
        "max_position_embeddings": 32768, "hidden_act": "silu",
        "rms_norm_eps": 1e-6, "rope_theta": 1000000,
        "tie_word_embeddings": True, "attention_bias": False,
        "attention_dropout": 0.0, "use_sliding_window": False,
        "sliding_window": None, "rope_scaling": None,
        "layer_types": ["full_attention"] * 28,
    }.items():
        require_equal(llm.get(name), value, "llm_config." + name)
    for name, value in {
        "model_type": "siglip2_navit", "hidden_size": 1152,
        "intermediate_size": 4304, "num_hidden_layers": 27,
        "num_attention_heads": 16, "num_channels": 3, "patch_size": 16,
        "temporal_patch_size": 1, "hidden_stride": 2, "image_size": 512,
        "num_patches": -1, "preserve_original_pe": True, "use_rope": True,
        "window_size": 112, "fullatt_block_indexes": None,
        "hidden_act": "gelu_pytorch_tanh", "layer_norm_eps": 1e-6,
    }.items():
        require_equal(config.get("vit_config", {}).get(name), value, "vit_config." + name)
    for name, value in {
        "image_processor_type": "SiglipImageProcessor", "do_resize": True,
        "do_rescale": True, "do_normalize": True, "resample": 2,
        "rescale_factor": 1 / 255, "image_mean": [0.5] * 3,
        "image_std": [0.5] * 3, "size": {"height": 512, "width": 512},
    }.items():
        require_equal(processor.get(name), value, "preprocessor." + name)
    require_equal(generation.get("eos_token_id"), [151643, 151645], "generation.eos_token_id")


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]


def tensor_specs() -> list[TensorSpec]:
    specs = [
        TensorSpec("llm.model.embed_tokens.weight", (VOCAB_SIZE, 1024)),
        TensorSpec("llm.lm_head.weight", (VOCAB_SIZE, 1024)),
        TensorSpec("llm.model.norm.weight", (1024,)),
    ]
    language = [
        ("input_layernorm.weight", (1024,)),
        ("post_attention_layernorm.weight", (1024,)),
        ("self_attn.q_proj.weight", (2048, 1024)),
        ("self_attn.k_proj.weight", (1024, 1024)),
        ("self_attn.v_proj.weight", (1024, 1024)),
        ("self_attn.o_proj.weight", (1024, 2048)),
        ("self_attn.q_norm.weight", (128,)),
        ("self_attn.k_norm.weight", (128,)),
        ("mlp.gate_proj.weight", (3072, 1024)),
        ("mlp.up_proj.weight", (3072, 1024)),
        ("mlp.down_proj.weight", (1024, 3072)),
    ]
    for layer in range(28):
        specs.extend(TensorSpec(f"llm.model.layers.{layer}.{name}", shape) for name, shape in language)
    base = "visual_tokenizer.vit.vision_model."
    for name, shape in [
        ("embeddings.patch_embedding.weight", (1152, 3, 16, 16)),
        ("embeddings.patch_embedding.bias", (1152,)),
        ("embeddings.position_embedding.weight", (1024, 1152)),
        ("post_layernorm.weight", (1152,)),
        ("post_layernorm.bias", (1152,)),
    ]:
        specs.append(TensorSpec(base + name, shape))
    vision = [
        ("layer_norm1.weight", (1152,)), ("layer_norm1.bias", (1152,)),
        ("layer_norm2.weight", (1152,)), ("layer_norm2.bias", (1152,)),
        ("mlp.fc1.weight", (4304, 1152)), ("mlp.fc1.bias", (4304,)),
        ("mlp.fc2.weight", (1152, 4304)), ("mlp.fc2.bias", (1152,)),
    ]
    for proj in ["q_proj", "k_proj", "v_proj", "out_proj"]:
        vision += [("self_attn." + proj + ".weight", (1152, 1152)),
                   ("self_attn." + proj + ".bias", (1152,))]
    for layer in range(27):
        specs.extend(TensorSpec(f"{base}encoder.layers.{layer}.{name}", shape) for name, shape in vision)
    specs += [TensorSpec("visual_tokenizer.head.0.weight", (65532, 4608)),
              TensorSpec("visual_tokenizer.head.1.weight", (65532,)),
              TensorSpec("visual_tokenizer.head.1.bias", (65532,)),
              TensorSpec("vte.weight", (65536, 1024))]
    return specs


def read_safetensors_header(path: Path) -> dict:
    size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("truncated safetensors header")
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size < 2 or header_size > min(size - 8, 16 * 1024 * 1024):
            raise ValueError("invalid safetensors header size")
        header = json.loads(stream.read(header_size))
    base, cursor = 8 + header_size, 0
    tensors, spans = {}, []
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        shape, dtype = entry.get("shape"), entry.get("dtype")
        if dtype not in ("BF16", "F16", "F32"):
            raise ValueError(f"unsupported dtype: {name}")
        if not isinstance(shape, list) or not shape or any(type(x) is not int or x <= 0 for x in shape):
            raise ValueError(f"invalid tensor shape: {name}")
        offsets = entry.get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2 or any(type(x) is not int for x in offsets):
            raise ValueError(f"invalid tensor offsets: {name}")
        begin, end = offsets
        if begin < 0 or end - begin != math.prod(shape) * (4 if dtype == "F32" else 2) or base + end > size:
            raise ValueError(f"invalid tensor byte range: {name}")
        spans.append((begin, end, name))
        tensors[name] = {**entry, "absolute_offset": base + begin}
    for begin, end, name in sorted(spans):
        if begin != cursor:
            raise ValueError(f"noncontiguous/overlapping safetensors data: {name}")
        cursor = end
    if cursor + base != size:
        raise ValueError("unaccounted safetensors trailing bytes")
    return tensors


def validate_tensors(tensors: dict) -> list[TensorSpec]:
    specs = tensor_specs()
    expected = {s.name for s in specs}
    if set(tensors) != expected:
        raise ValueError(f"tensor inventory mismatch: missing={sorted(expected-set(tensors))[:8]}, extra={sorted(set(tensors)-expected)[:8]}")
    for spec in specs:
        require_equal(tuple(tensors[spec.name]["shape"]), spec.shape, spec.name)
        require_equal(tensors[spec.name]["dtype"], "BF16", spec.name + " dtype")
    return specs


def verify_tied_weights(path: Path, entries: dict) -> None:
    """Both copies are shipped; verify tie identity instead of assuming loader order."""
    left, right = (entries[n] for n in ["llm.model.embed_tokens.weight", "llm.lm_head.weight"])
    require_equal(left["shape"], right["shape"], "tied embedding shape")
    require_equal(left["dtype"], right["dtype"], "tied embedding dtype")
    length = left["data_offsets"][1] - left["data_offsets"][0]
    with path.open("rb") as a, path.open("rb") as b:
        a.seek(left["absolute_offset"]); b.seek(right["absolute_offset"])
        while length:
            count = min(length, 8 * 1024 * 1024)
            x, y = a.read(count), b.read(count)
            if len(x) != count or x != y:
                raise ValueError("checkpoint has unequal tied llm_head/embed_tokens; reference loader behavior must be audited")
            length -= count


def tokenizer_payload(tokenizer: dict, config: dict):
    require_equal(tokenizer.get("normalizer"), {"type": "NFC"}, "tokenizer normalizer")
    pre = tokenizer.get("pre_tokenizer", {}).get("pretokenizers", [])
    if len(pre) != 2:
        raise ValueError("unexpected tokenizer pretokenizer chain")
    require_equal(pre[0].get("pattern"), {"Regex": QWEN_PATTERN}, "tokenizer split regex")
    require_equal(pre[1], {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False}, "tokenizer bytelevel")
    model = tokenizer.get("model", {})
    require_equal(model.get("type"), "BPE", "tokenizer model")
    vocab = dict(model.get("vocab", {}))
    require_equal(len(vocab), 151643, "base vocabulary size")
    added = tokenizer.get("added_tokens", [])
    require_equal(len(added), 26, "added-token count")
    for token in added:
        name, token_id = token["content"], token["id"]
        require_equal(SPECIAL_TOKENS.get(token_id), name, "added token ID")
        require_equal(token.get("special"), token_id <= 151656, "added token special flag")
        if name in vocab and vocab[name] != token_id:
            raise ValueError("conflicting added token")
        vocab[name] = token_id
    if len(vocab) != TOKENIZER_SIZE or set(vocab.values()) != set(range(TOKENIZER_SIZE)):
        raise ValueError("tokenizer IDs must be dense 0..151668")
    tokens, token_types = [""] * TOKENIZER_SIZE, [1] * TOKENIZER_SIZE
    for token, token_id in vocab.items():
        tokens[token_id] = token
    for token_id, token in SPECIAL_TOKENS.items():
        require_equal(tokens[token_id], token, f"token {token_id}")
        declared = config.get("added_tokens_decoder", {}).get(str(token_id), {})
        require_equal(declared.get("content"), token, "tokenizer_config content")
        require_equal(declared.get("special"), token_id <= 151656, "tokenizer_config special")
        token_types[token_id] = 3 if token_id <= 151656 else 4  # CONTROL / USER_DEFINED
    merges = []
    for pair in model.get("merges", []):
        if isinstance(pair, list) and len(pair) == 2 and all(isinstance(x, str) for x in pair):
            merges.append(" ".join(pair))
        elif isinstance(pair, str) and len(pair.split(" ")) == 2:
            merges.append(pair)
        else:
            raise ValueError("invalid BPE merge")
    require_equal(len(merges), MERGE_COUNT, "BPE merge count")
    return tokens, token_types, merges


def storage_dtype(spec: TensorSpec, requested: str) -> str:
    sensitive = len(spec.shape) == 1 or any(part in spec.name for part in (
        "norm", "embed", "vte.weight", "visual_tokenizer.head.", "llm.lm_head."))
    return "float32" if requested == "f32" or sensitive else "float16"


def read_tensor(path: Path, entry: dict, dtype: str):
    import numpy as np
    mapped = np.memmap(path, dtype={"BF16": "<u2", "F16": "<f2", "F32": "<f4"}[entry["dtype"]],
                       mode="r", offset=entry["absolute_offset"], shape=tuple(entry["shape"]))
    data = (mapped.astype(np.uint32) << 16).view(np.float32) if entry["dtype"] == "BF16" else np.array(mapped, dtype=np.float32)
    del mapped
    if not np.isfinite(data).all():
        raise ValueError("non-finite source weight")
    with np.errstate(over="raise", invalid="raise"):
        converted = data.astype(dtype, copy=False)
    if not np.isfinite(converted).all():
        raise ValueError("non-finite converted weight")
    return np.ascontiguousarray(converted)


def write_metadata(writer, config, processor, tokens, types, merges, hashes, dtype):
    writer.add_string("general.name", "SoMarkAI OXR-1.0")
    writer.add_string("general.license", "OXR License (Apache-2.0 with additional terms)")
    writer.add_string("general.license.link", f"https://huggingface.co/{MODEL_ID}/blob/{MODEL_REVISION}/LICENSE")
    for name, value in {
        "model_family": FAMILY, "source.repository": MODEL_ID, "source.revision": MODEL_REVISION,
        "blueprint.revision": BLUEPRINT_REVISION, "source.file_sha256": json.dumps(hashes, sort_keys=True),
        "conversion.recipe": "f32" if dtype == "f32" else "f16-f32-sensitive",
        "config_json": json.dumps(config, sort_keys=True), "preprocessor_json": json.dumps(processor, sort_keys=True),
        "processor": "siglip-pil-bilinear", "tokenizer.normalizer": "NFC",
        "generation.repetition_history": "prompt_and_completion",
    }.items():
        writer.add_string("oxr." + name, value)
    for key in ["hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
                "num_key_value_heads", "head_dim", "vocab_size", "max_position_embeddings"]:
        writer.add_uint32("oxr.llm." + key, config["llm_config"][key])
    for key in ["rms_norm_eps", "rope_theta"]:
        writer.add_float32("oxr.llm." + key, config["llm_config"][key])
    for key in ["hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
                "patch_size", "hidden_stride", "image_size", "window_size"]:
        writer.add_uint32("oxr.vision." + key, config["vit_config"][key])
    writer.add_float32("oxr.vision.layer_norm_eps", 1e-6)
    writer.add_float32("oxr.visual_head.layer_norm_eps", 1e-5)
    writer.add_uint32("oxr.processor.min_pixels", 200704)
    writer.add_uint32("oxr.processor.max_pixels", 921600)
    writer.add_uint32("oxr.context_length", 32768)
    writer.add_uint32("oxr.visual_vocab_size", 65536)
    writer.add_uint32("oxr.tokenizer.vocab_size", len(tokens))
    writer.add_array("oxr.generation.eos_token_ids", [151643, 151645])
    writer.add_string("tokenizer.ggml.model", "gpt2")
    writer.add_string("tokenizer.ggml.pre", "qwen2")
    writer.add_array("tokenizer.ggml.tokens", tokens)
    writer.add_array("tokenizer.ggml.token_type", types)
    writer.add_array("tokenizer.ggml.merges", merges)
    writer.add_uint32("tokenizer.ggml.eos_token_id", 151643)
    writer.add_uint32("tokenizer.ggml.padding_token_id", 151643)


def resolve_model(model: str) -> Path:
    if Path(model).is_dir():
        return Path(model)
    require_equal(model, MODEL_ID, "model repository")
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION,
                                  allow_patterns=list(source_manifest()["files"])))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=["f32", "f16"], default="f32")
    parser.add_argument("--check-only", action="store_true", help="authenticate full source including weights without writing a GGUF")
    args = parser.parse_args(argv)
    directory = resolve_model(args.model)
    hashes = verify_source_files(directory)
    payload = {n: json.loads((directory / n).read_text(encoding="utf-8")) for n in [
        "config.json", "preprocessor_config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"]}
    config, processor = payload["config.json"], payload["preprocessor_config.json"]
    validate_config(config, processor, payload["generation_config.json"])
    tokens, types, merges = tokenizer_payload(payload["tokenizer.json"], payload["tokenizer_config.json"])
    shard = directory / "model.safetensors"
    entries = read_safetensors_header(shard)
    specs = validate_tensors(entries)
    verify_tied_weights(shard, entries)
    print(f"Authenticated {len(specs)} tensors, {len(tokens)} tokens and {len(merges)} merges", file=sys.stderr)
    if args.check_only:
        return 0
    import gguf
    import numpy as np
    if args.output.exists() or Path(str(args.output) + ".json").exists():
        raise FileExistsError(f"refusing to replace output or receipt: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".partial")
    if temporary.exists():
        raise FileExistsError(f"refusing prior partial output: {temporary}")
    writer = gguf.GGUFWriter(str(temporary), "oxr")
    try:
        write_metadata(writer, config, processor, tokens, types, merges, hashes, args.dtype)
        for spec in specs:
            dtype = np.dtype(storage_dtype(spec, args.dtype))
            writer.add_tensor_info(spec.name, spec.shape, dtype, math.prod(spec.shape) * dtype.itemsize,
                                   raw_dtype=gguf.GGMLQuantizationType.F32 if dtype.itemsize == 4 else gguf.GGMLQuantizationType.F16)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        for index, spec in enumerate(specs):
            writer.write_tensor_data(read_tensor(shard, entries[spec.name], storage_dtype(spec, args.dtype)))
            if index % 64 == 0:
                print(f"Converted {index + 1}/{len(specs)}", file=sys.stderr)
        writer.close()
        os.replace(temporary, args.output)
    except BaseException:
        writer.close()
        if temporary.exists():
            temporary.unlink()
        raise
    receipt = {
        "schema": "crispembed.oxr.conversion.v1", "model": MODEL_ID, "model_revision": MODEL_REVISION,
        "source_sha256": hashes, "blueprint_revision": BLUEPRINT_REVISION,
        "recipe": args.dtype, "tensor_count": len(specs), "tokenizer_entries": len(tokens),
        "merges": len(merges), "tied_matrices_byte_identical": True,
        "output_bytes": args.output.stat().st_size, "output_sha256": sha256(args.output),
        "tensor_storage": {spec.name: storage_dtype(spec, args.dtype) for spec in specs},
        "qualification": {"model_quality_parity": False, "python_speed_parity": False,
                          "quantized_quality": False},
        "license": "OXR additional terms plus Apache-2.0; retained in checkpoint LICENSE",
    }
    Path(str(args.output) + ".json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
