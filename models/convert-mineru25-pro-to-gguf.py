#!/usr/bin/env python3
"""Convert the pinned MinerU2.5-Pro-2604-1.2B checkpoint to native CrispEmbed GGUF.

No Transformers model or remote Python code is executed. Safetensors are read one
tensor at a time; BF16 is expanded exactly to F32 before optional F16 storage.
Norms, biases and the tied embedding/output matrix stay F32 in the F16 recipe.
Quantization is deliberately excluded until decoded quality is qualified.

Install: python -m pip install numpy gguf huggingface_hub
Run: python models/convert-mineru25-pro-to-gguf.py --output mineru25-pro-f32.gguf
Use --model /path/to/snapshot to convert an already downloaded checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
from dataclasses import dataclass

MODEL_ID = "opendatalab/MinerU2.5-Pro-2604-1.2B"
MODEL_REVISION = "d3f5e08d073c21466bbabe21c71bb1e9c2e595da"
BLUEPRINT_REVISION = "c9797cde8f54cc48306196fc6aebe73ea31dd4a3"
FAMILY = "mineru2.5-pro-2604"
VOCAB_SIZE = 151936
TOKENIZER_SIZE = 151674
SPECIAL_TOKENS = dict(enumerate([
    "<|endoftext|>", "<|im_start|>", "<|im_end|>",
    "<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>",
    "<|box_end|>", "<|quad_start|>", "<|quad_end|>",
    "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>",
    "<|image_pad|>", "<|video_pad|>", "<|ref_start|>", "<|ref_end|>",
    "<|md_start|>", "<|md_end|>", "<ched>", "<ecel>", "<fcel>",
    "<lcel>", "<ucel>", "<xcel>", "<nl>", "<|rotate_up|>",
    "<|rotate_down|>", "<|rotate_left|>", "<|rotate_right|>",
    "<|txt_contd|>", "<|paratext|>",
], start=151643))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source_files(directory: Path, manifest=None) -> dict[str, str]:
    """Verify byte identity, including the HF LFS SHA256 of all model weights."""
    if manifest is None:
        manifest = json.loads(Path(__file__).with_name("mineru25-pro-source.json").read_text(encoding="utf-8"))
    hashes = {}
    for name, expected in manifest["files"].items():
        path = directory / name
        if path.stat().st_size != expected["size"]:
            raise ValueError(f"pinned source size mismatch: {name}")
        value = sha256(path)
        if value != expected["sha256"]:
            raise ValueError(f"pinned source SHA256 mismatch: {name}")
        hashes[name] = value
    return hashes


def require_equal(actual, expected, field):
    if actual != expected:
        raise ValueError(f"unsupported {field}: {actual!r}; expected {expected!r}")


def validate_config(config: dict, pp: dict, generation: dict) -> None:
    """Fail closed instead of silently exporting a different Qwen checkpoint."""
    require_equal(config.get("model_type"), "qwen2_vl", "model_type")
    text = config.get("text_config", config)
    for name, value in {
        "hidden_size": 896, "intermediate_size": 4864,
        "num_hidden_layers": 24, "num_attention_heads": 14,
        "num_key_value_heads": 2, "vocab_size": VOCAB_SIZE,
        "max_position_embeddings": 8192, "hidden_act": "silu",
        "rms_norm_eps": 1e-6, "rope_theta": 1000000.0,
        "tie_word_embeddings": True, "use_sliding_window": False,
    }.items():
        require_equal(text.get(name), value, "text_config." + name)
    require_equal(text.get("rope_scaling", {}).get("mrope_section"), [8, 12, 12], "mrope_section")
    for name, value in {
        "depth": 32, "embed_dim": 1280, "hidden_size": 896,
        "num_heads": 16, "mlp_ratio": 4, "spatial_patch_size": 14,
        "temporal_patch_size": 2, "spatial_merge_size": 2,
        "hidden_act": "quick_gelu", "in_channels": 3,
    }.items():
        require_equal(config.get("vision_config", {}).get(name), value, "vision_config." + name)
    for name, value in {"image_token_id": 151655, "video_token_id": 151656,
                        "vision_start_token_id": 151652, "vision_end_token_id": 151653}.items():
        require_equal(config.get(name), value, name)
    for name, value in {
        "min_pixels": 50176, "max_pixels": 1605632, "patch_size": 14,
        "temporal_patch_size": 2, "merge_size": 2,
        "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_std": [0.26862954, 0.26130258, 0.27577711],
        "image_processor_type": "Qwen2VLImageProcessor",
    }.items():
        require_equal(pp.get(name), value, "preprocessor." + name)
    require_equal(generation.get("eos_token_id"), [151645, 151643], "generation.eos_token_id")


@dataclass(frozen=True)
class TensorSpec:
    source: str
    target: str
    shape: tuple[int, ...]
    flatten_patch: bool = False


def tensor_specs() -> list[TensorSpec]:
    specs = [TensorSpec("visual.patch_embed.proj.weight", "v.patch_embed.weight", (1280, 3, 2, 14, 14), True)]
    vis = [
        ("norm1.weight", "norm1.weight", (1280,)),
        ("norm1.bias", "norm1.bias", (1280,)),
        ("norm2.weight", "norm2.weight", (1280,)),
        ("norm2.bias", "norm2.bias", (1280,)),
        ("attn.qkv.weight", "attn_qkv.weight", (3840, 1280)),
        ("attn.qkv.bias", "attn_qkv.bias", (3840,)),
        ("attn.proj.weight", "attn_proj.weight", (1280, 1280)),
        ("attn.proj.bias", "attn_proj.bias", (1280,)),
        ("mlp.fc1.weight", "ffn_fc1.weight", (5120, 1280)),
        ("mlp.fc1.bias", "ffn_fc1.bias", (5120,)),
        ("mlp.fc2.weight", "ffn_fc2.weight", (1280, 5120)),
        ("mlp.fc2.bias", "ffn_fc2.bias", (1280,)),
    ]
    for layer in range(32):
        specs.extend(TensorSpec(f"visual.blocks.{layer}.{src}", f"v.blk.{layer}.{dst}", shape)
                     for src, dst, shape in vis)
    for src, dst, shape in [
        ("ln_q.weight", "norm.weight", (1280,)),
        ("ln_q.bias", "norm.bias", (1280,)),
        ("mlp.0.weight", "fc1.weight", (5120, 5120)),
        ("mlp.0.bias", "fc1.bias", (5120,)),
        ("mlp.2.weight", "fc2.weight", (896, 5120)),
        ("mlp.2.bias", "fc2.bias", (896,)),
    ]:
        specs.append(TensorSpec("visual.merger." + src, "v.merger." + dst, shape))
    specs.append(TensorSpec("model.embed_tokens.weight", "l.embed_tokens.weight", (VOCAB_SIZE, 896)))
    llm = [
        ("input_layernorm.weight", "attn_norm.weight", (896,)),
        ("post_attention_layernorm.weight", "ffn_norm.weight", (896,)),
        ("self_attn.q_proj.weight", "attn_q.weight", (896, 896)),
        ("self_attn.q_proj.bias", "attn_q.bias", (896,)),
        ("self_attn.k_proj.weight", "attn_k.weight", (128, 896)),
        ("self_attn.k_proj.bias", "attn_k.bias", (128,)),
        ("self_attn.v_proj.weight", "attn_v.weight", (128, 896)),
        ("self_attn.v_proj.bias", "attn_v.bias", (128,)),
        ("self_attn.o_proj.weight", "attn_o.weight", (896, 896)),
        ("mlp.gate_proj.weight", "ffn_gate.weight", (4864, 896)),
        ("mlp.up_proj.weight", "ffn_up.weight", (4864, 896)),
        ("mlp.down_proj.weight", "ffn_down.weight", (896, 4864)),
    ]
    for layer in range(24):
        specs.extend(TensorSpec(f"model.layers.{layer}.{src}", f"l.blk.{layer}.{dst}", shape)
                     for src, dst, shape in llm)
    specs.append(TensorSpec("model.norm.weight", "l.output_norm.weight", (896,)))
    return specs


def read_safetensors_header(path: Path):
    size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("truncated safetensors header")
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size < 2 or header_size > min(size - 8, 16 * 1024 * 1024):
            raise ValueError("invalid safetensors header size")
        header = json.loads(stream.read(header_size))
    base = 8 + header_size
    spans = []
    tensors = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        shape = entry.get("shape")
        dtype = entry.get("dtype")
        if dtype not in ("BF16", "F16", "F32"):
            raise ValueError(f"unsupported tensor dtype {name}: {dtype}")
        if not isinstance(shape, list) or not shape or any(type(x) is not int or x <= 0 for x in shape):
            raise ValueError(f"invalid shape: {name}")
        begin, end = entry["data_offsets"]
        byte_count = math.prod(shape) * (4 if dtype == "F32" else 2)
        if type(begin) is not int or type(end) is not int or begin < 0 or end - begin != byte_count or base + end > size:
            raise ValueError(f"invalid byte range: {name}")
        spans.append((begin, end, name))
        tensors[name] = {**entry, "absolute_offset": base + begin}
    spans.sort()
    cursor = 0
    for begin, end, name in spans:
        if begin != cursor:
            raise ValueError(f"non-contiguous/overlapping safetensors data at {name}")
        cursor = end
    if base + cursor != size:
        raise ValueError("safetensors has unaccounted trailing bytes")
    return tensors


def validate_tensors(tensors: dict) -> list[TensorSpec]:
    specs = tensor_specs()
    expected = {spec.source for spec in specs}
    # The original checkpoint ties embeddings and ships no lm_head.weight.
    if set(tensors) != expected:
        missing, extra = sorted(expected - set(tensors)), sorted(set(tensors) - expected)
        raise ValueError(f"tensor inventory mismatch; missing={missing[:8]}, extra={extra[:8]}")
    for spec in specs:
        require_equal(tuple(tensors[spec.source]["shape"]), spec.shape, spec.source)
    return specs


def tokenizer_payload(tokenizer: dict, config: dict):
    model = tokenizer.get("model", {})
    require_equal(model.get("type"), "BPE", "tokenizer.model.type")
    vocab = dict(model.get("vocab", {}))
    for token in tokenizer.get("added_tokens", []):
        text, token_id = token["content"], token["id"]
        if text in vocab and vocab[text] != token_id:
            raise ValueError(f"conflicting added token {text}")
        vocab[text] = token_id
    if len(vocab) != TOKENIZER_SIZE or set(vocab.values()) != set(range(TOKENIZER_SIZE)):
        raise ValueError("tokenizer must have dense IDs 0..151673; do not truncate added tokens")
    tokens = [""] * TOKENIZER_SIZE
    types = [1] * TOKENIZER_SIZE  # gguf.TokenType.NORMAL
    for token, token_id in vocab.items():
        tokens[token_id] = token
    for token_id, token in SPECIAL_TOKENS.items():
        require_equal(tokens[token_id], token, f"token {token_id}")
        require_equal(config.get("added_tokens_decoder", {}).get(str(token_id), {}).get("content"), token,
                      f"tokenizer_config token {token_id}")
        types[token_id] = 3  # CONTROL, retained by MinerU's explicit decoder policy
    merges = []
    for merge in model.get("merges", []):
        if isinstance(merge, list) and len(merge) == 2 and all(isinstance(x, str) for x in merge):
            merges.append(" ".join(merge))
        elif isinstance(merge, str) and len(merge.split(" ")) == 2:
            merges.append(merge)
        else:
            raise ValueError(f"invalid BPE merge: {merge!r}")
    if not merges:
        raise ValueError("BPE merges are required; vocab-only fallback is forbidden")
    return tokens, types, merges


def write_metadata(writer, config, pp, tokens, token_types, merges, source, hashes, dtype):
    writer.add_string("general.name", "MinerU2.5-Pro-2604-1.2B")
    writer.add_string("general.license", "apache-2.0")
    writer.add_string("qwen2vl.model_family", FAMILY)
    writer.add_string("qwen2vl.mineru.processor", "qwen2vl-fast")
    writer.add_string("qwen2vl.source.repository", MODEL_ID)
    writer.add_string("qwen2vl.source.revision", source)
    writer.add_string("qwen2vl.source.expected_revision", MODEL_REVISION)
    writer.add_string("qwen2vl.source.file_sha256", json.dumps(hashes, sort_keys=True))
    writer.add_string("qwen2vl.blueprint.revision", BLUEPRINT_REVISION)
    writer.add_string("qwen2vl.conversion.recipe", "f32" if dtype == "f32" else "f16-f32-norm-bias-embedding")
    tc = config["text_config"]
    for key in ["vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
                "num_attention_heads", "num_key_value_heads", "max_position_embeddings"]:
        writer.add_uint32("qwen2vl." + key, tc[key])
    for key in ["image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"]:
        writer.add_uint32("qwen2vl." + key, config[key])
    writer.add_uint32("qwen2vl.attention.head_dim", 64)
    writer.add_float32("qwen2vl.rms_norm_eps", 1e-6)
    writer.add_float32("qwen2vl.rope_theta", 1000000.0)
    writer.add_bool("qwen2vl.tie_word_embeddings", True)
    writer.add_array("qwen2vl.rope_sections", [8, 12, 12, 0])
    for key, value in {"depth": 32, "hidden_size": 1280, "intermediate_size": 5120,
                       "num_heads": 16, "in_channels": 3, "spatial_patch_size": 14,
                       "temporal_patch_size": 2, "spatial_merge_size": 2,
                       "out_hidden_size": 896, "window_size": 0,
                       "min_pixels": pp["min_pixels"], "max_pixels": pp["max_pixels"]}.items():
        writer.add_uint32("qwen2vl.vision." + key, value)
    writer.add_float32("qwen2vl.vision.layer_norm_eps", 1e-6)
    writer.add_string("qwen2vl.vision.hidden_act", "quick_gelu")
    writer.add_string("qwen2vl.vision.merger_hidden_act", "gelu")
    writer.add_array("qwen2vl.vision.image_mean", pp["image_mean"])
    writer.add_array("qwen2vl.vision.image_std", pp["image_std"])
    writer.add_uint32("qwen2vl.generation.no_repeat_ngram_size", 100)
    writer.add_array("qwen2vl.generation.eos_token_ids", [151645, 151643])
    writer.add_string("tokenizer.ggml.model", "gpt2")
    writer.add_string("tokenizer.ggml.pre", "qwen2")
    writer.add_uint32("qwen2vl.tokenizer.vocab_size", len(tokens))
    writer.add_array("tokenizer.ggml.tokens", tokens)
    writer.add_array("tokenizer.ggml.token_type", token_types)
    writer.add_array("tokenizer.ggml.merges", merges)
    writer.add_uint32("tokenizer.ggml.eos_token_id", 151645)
    writer.add_uint32("tokenizer.ggml.padding_token_id", 151643)


def storage_dtype(spec: TensorSpec, requested: str) -> str:
    return "float32" if requested == "f32" or len(spec.shape) == 1 or "norm" in spec.target or "embed_tokens" in spec.target else "float16"


def read_tensor(path: Path, entry: dict, shape, dtype):
    import numpy as np
    src_dtype = {"BF16": "<u2", "F16": "<f2", "F32": "<f4"}[entry["dtype"]]
    mapped = np.memmap(path, dtype=src_dtype, mode="r", offset=entry["absolute_offset"], shape=tuple(entry["shape"]))
    if entry["dtype"] == "BF16":
        data = (mapped.astype(np.uint32) << 16).view(np.float32)
    else:
        data = np.array(mapped, dtype=np.float32)
    del mapped
    if not np.isfinite(data).all():
        raise ValueError("non-finite source weights")
    with np.errstate(over="raise", invalid="raise"):
        data = data.astype(dtype).reshape(shape)
    return np.ascontiguousarray(data)


def resolve_model(model: str) -> tuple[Path, str]:
    local = Path(model)
    if local.is_dir():
        # A directory label does not prove provenance. Hash every input file.
        return local, "local-unverified"
    require_equal(model, MODEL_ID, "model repository")
    from huggingface_hub import snapshot_download
    path = snapshot_download(MODEL_ID, revision=MODEL_REVISION, allow_patterns=[
        "config.json", "preprocessor_config.json", "generation_config.json",
        "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "model.safetensors",
    ])
    return Path(path), MODEL_REVISION


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=["f32", "f16"], default="f32")
    parser.add_argument("--check-only", action="store_true", help="verify source inventory/config/tokenizer without writing weights")
    args = parser.parse_args(argv)
    model_dir, source = resolve_model(args.model)
    required = ["config.json", "preprocessor_config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"]
    payload = {name: json.loads((model_dir / name).read_text(encoding="utf-8")) for name in required}
    config, pp = payload["config.json"], payload["preprocessor_config.json"]
    validate_config(config, pp, payload["generation_config.json"])
    tokens, token_types, merges = tokenizer_payload(payload["tokenizer.json"], payload["tokenizer_config.json"])
    shard = model_dir / "model.safetensors"
    entries = read_safetensors_header(shard)
    specs = validate_tensors(entries)
    print(f"Validated {len(specs)} tensors and {len(tokens)} tokenizer entries", file=sys.stderr)
    if args.check_only:
        return 0
    import gguf
    import numpy as np
    if args.output.exists():
        raise FileExistsError(f"refusing to replace {args.output}; choose a new output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    hashes = verify_source_files(model_dir)
    source = MODEL_REVISION
    temp = args.output.with_name(args.output.name + ".partial")
    if temp.exists():
        raise FileExistsError(f"refusing to overwrite prior partial output {temp}")
    writer = gguf.GGUFWriter(str(temp), "qwen2vl")
    try:
        write_metadata(writer, config, pp, tokens, token_types, merges, source, hashes, args.dtype)
        for spec in specs:
            shape = (1280, 1176) if spec.flatten_patch else spec.shape
            dtype = np.dtype(storage_dtype(spec, args.dtype))
            writer.add_tensor_info(spec.target, shape, dtype, math.prod(shape) * dtype.itemsize,
                                   raw_dtype=gguf.GGMLQuantizationType.F32 if dtype.itemsize == 4 else gguf.GGMLQuantizationType.F16)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        for index, spec in enumerate(specs):
            shape = (1280, 1176) if spec.flatten_patch else spec.shape
            data = read_tensor(shard, entries[spec.source], shape, storage_dtype(spec, args.dtype))
            writer.write_tensor_data(data)
            del data
            if index % 64 == 0:
                print(f"Converted {index + 1}/{len(specs)}", file=sys.stderr)
        writer.close()
        os.replace(temp, args.output)
    except BaseException:
        writer.close()
        # A failed conversion must never look like a usable model.
        if temp.exists():
            temp.unlink()
        raise
    receipt = {
        "model": MODEL_ID, "expected_revision": MODEL_REVISION, "source_revision": source,
        "source_sha256": hashes, "recipe": args.dtype, "tensor_count": len(specs),
        "tokenizer_size": len(tokens), "embedding_rows": VOCAB_SIZE,
        "gguf_sha256": sha256(args.output), "gguf_size_bytes": args.output.stat().st_size,
        "quality_parity_verified": False, "speed_parity_verified": False,
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
