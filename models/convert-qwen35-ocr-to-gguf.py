#!/usr/bin/env python3
"""Export dense Qwen3.5 OCR checkpoints for the pending native backend.

This is an experimental weight format, not a supported registry model. Compare
checkpoint names/shapes with the official Transformers model on meta before
writing anything. Do not silently drop unknown weights or fabricate defaults.
The source layout is preserved (including zero-centered RMSNorm weights).
"""

import argparse
import hashlib
import json
import re
from pathlib import Path


ARCH = "qwen35_ocr"
LIGHTON_MODEL = "lightonai/LightOnOCR-3-0.8B"
LIGHTON_REVISION = "36d8636925327c671bd8e479c517a004f7566bd9"


def validate_config(config):
    if config.get("model_type") != "qwen3_5":
        raise ValueError("Only dense qwen3_5 checkpoints are supported")
    text = config["text_config"]
    vision = config["vision_config"]
    kinds = text["layer_types"]
    if len(kinds) != text["num_hidden_layers"] or not set(kinds) <= {
        "linear_attention", "full_attention"
    }:
        raise ValueError("Invalid or unsupported decoder layer schedule")
    if text.get("mtp_num_hidden_layers", 0):
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


def convert(model_dir, output, source_model, source_revision, dtype="f16", keep_f32=()):
    import gguf
    import numpy as np
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig, Qwen3_5ForConditionalGeneration

    root = Path(model_dir)
    output = Path(output)
    if not re.fullmatch(r"[0-9a-f]{40}", source_revision):
        raise ValueError("An immutable 40-hex source revision is required")
    config = json.loads((root / "config.json").read_text())
    text, vision = validate_config(config)
    if dtype not in {"f32", "f16"}:
        raise ValueError("Establish full-precision parity before adding quantized exports")

    # Official implementation supplies the independent expected parameter schema.
    hf_config = AutoConfig.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    with torch.device("meta"):
        official = Qwen3_5ForConditionalGeneration(hf_config)
    expected = {name: list(tensor.shape) for name, tensor in official.state_dict().items()}
    del official

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

    tokenizer = json.loads((root / "tokenizer.json").read_text())
    if tokenizer["model"]["type"] != "BPE":
        raise ValueError("Expected an audited BPE tokenizer")
    vocab = dict(tokenizer["model"]["vocab"])
    for token in tokenizer["added_tokens"]:
        vocab[token["content"]] = token["id"]
    tokens = [""] * text["vocab_size"]
    occupied = set()
    for token, index in vocab.items():
        if index < 0 or index >= len(tokens) or index in occupied:
            raise ValueError(f"Invalid/duplicate tokenizer id: {index}")
        occupied.add(index)
        tokens[index] = token
    merges = tokenizer["model"]["merges"]
    merges = [item if isinstance(item, str) else " ".join(item) for item in merges]

    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_name(output.name + ".partial")
    writer = gguf.GGUFWriter(str(pending), ARCH, use_temp_file=True)
    writer.add_string("general.name", source_model)
    writer.add_string("general.source.huggingface.repository", source_model)
    writer.add_string("general.source.huggingface.revision", source_revision)
    writer.add_string(f"{ARCH}.status", "experimental-export-native-backend-pending")
    writer.add_uint32(f"{ARCH}.format_version", 1)
    writer.add_bool(f"{ARCH}.tie_word_embeddings", tied)
    writer.add_bool(f"{ARCH}.zero_centered_text_norm", True)
    # Preserve every config field, rather than treating nested EOS ids as synonyms.
    for filename in ["config.json", "processor_config.json", "preprocessor_config.json",
                     "tokenizer_config.json", "generation_config.json"]:
        writer.add_string(f"{ARCH}.{filename.removesuffix('.json')}", (root / filename).read_text())
    writer.add_string("tokenizer.chat_template", (root / "chat_template.jinja").read_text())
    writer.add_string("tokenizer.ggml.model", "gpt2")
    writer.add_string("tokenizer.ggml.pre", "qwen35")
    writer.add_array("tokenizer.ggml.tokens", tokens)
    writer.add_array("tokenizer.ggml.merges", merges)
    # Keep actual tokenizer EOS and model generation EOS separately.
    tok_config = json.loads((root / "tokenizer_config.json").read_text())
    writer.add_uint32("tokenizer.ggml.eos_token_id", vocab[tok_config["eos_token"]])
    writer.add_uint32("tokenizer.ggml.padding_token_id", vocab[tok_config["pad_token"]])

    records = []
    try:
        for shard in shards:
            with safe_open(root / shard, framework="pt", device="cpu") as handle:
                for source in sorted(handle.keys()):
                    if tied and source == "lm_head.weight":
                        continue  # Native head must alias the exported embedding.
                    name = tensor_name(source)
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
                    writer.add_tensor(name, array)
                    records.append({"source": source, "name": name, "shape": list(array.shape),
                                    "source_shape": source_shape,
                                    "dtype": str(array.dtype), "sha256": hashlib.sha256(array.tobytes()).hexdigest()})
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
                "checkpoint_tensor_count": len(actual), "export_tensor_count": len(records),
                "records": records, "bytes": output.stat().st_size}
    output.with_suffix(output.suffix + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    from huggingface_hub import snapshot_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=LIGHTON_MODEL)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", choices=["f32", "f16"], default="f16")
    parser.add_argument("--keep-f32", action="append", default=[], metavar="GGUF_NAME",
                        help="Keep this exact tensor in F32; repeat for multiple tensors (unknown names reject)")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("--revision must be an immutable 40-hex commit")
    directory = snapshot_download(args.model, revision=args.revision,
                                  allow_patterns=["*.json", "*.jinja", "*.safetensors", "LICENSE*"])
    manifest = convert(directory, args.output, args.model, args.revision, args.dtype, args.keep_f32)
    print(json.dumps({key: value for key, value in manifest.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
