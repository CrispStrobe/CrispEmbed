#!/usr/bin/env python3
"""Strict UltraBERT-Embed-150M conversion; not a declaration of native support.

Accepts only a local snapshot of the pinned retrieval checkpoint. Does not run
remote Python, drop heads, alias another architecture or download models.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from ultrabert_contract import (MODEL, QUERY, REVISION, expected_tensors, profile,
                                sha256, tokenizer_metadata, validate_sources,
                                validate_weight_file)


def convert(directory, output, dtype):
    directory, output = Path(directory), Path(output)
    if dtype not in ("f32", "f16"):
        raise ValueError("Unsupported precision")
    if output.exists():
        raise ValueError("Refusing to overwrite existing output")
    sources = validate_sources(directory)
    weights = validate_weight_file(directory / "model.safetensors")
    tok_text = (directory / "tokenizer.json").read_text()
    tokens, types, merges = tokenizer_metadata(json.loads(tok_text))
    config = profile()["config"]
    # Heavy libraries are imported only after the original source/payload checks.
    import gguf
    import numpy as np
    from safetensors import safe_open

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ultrabert-convert-", dir=output.parent) as staging:
        temporary = Path(staging) / "model.gguf"
        writer = gguf.GGUFWriter(str(temporary), arch="ultrabert", use_temp_file=True)
        writer.add_string("general.name", "UltraBERT-Embed-150M")
        writer.add_string("general.license", "apache-2.0")
        writer.add_string("general.source.url", "https://huggingface.co/" + MODEL)
        writer.add_string("ultrabert.source.revision", REVISION)
        writer.add_string("ultrabert.source.weight_sha256", weights["sha256"])
        writer.add_string("ultrabert.source.file_sha256", json.dumps(sources, sort_keys=True))
        writer.add_bool("ultrabert.attention.causal", False)
        for key, value in {
            "context_length": 32768, "embedding_length": 768, "block_count": 24,
            "feed_forward_length": 1024, "attention.head_count": 6,
            "attention.head_count_kv": 6, "attention.key_length": 128,
            "attention.value_length": 128, "pooling_type": 2,
        }.items():
            writer.add_uint32("ultrabert." + key, value)
        writer.add_float32("ultrabert.attention.layer_norm_rms_epsilon", config["rms_norm_eps"])
        writer.add_array("ultrabert.attention.window_radius", [w if w is not None else -1 for w in config["attention_windows"]])
        writer.add_array("ultrabert.rope.freq_base_per_layer", config["rope_theta"])
        writer.add_string("ultrabert.rope.layout", "split-half")
        writer.add_string("ultrabert.pooling", "cls")
        writer.add_bool("ultrabert.pooling.include_prompt", True)
        writer.add_bool("ultrabert.pooling.normalize_default", False)
        writer.add_string("ultrabert.prompt.query", QUERY)
        writer.add_string("ultrabert.prompt.document", "")
        writer.add_string("ultrabert.prompt.default", "")
        writer.add_string("ultrabert.tokenizer.json", tok_text)
        writer.add_string("ultrabert.tokenizer.single_template", "<|endoftext|> $A")
        writer.add_string("ultrabert.tokenizer.pair_template", "<|endoftext|> $A <|sep|> $B")
        writer.add_string("tokenizer.ggml.model", "gpt2")
        writer.add_string("tokenizer.ggml.pre", "gpt-2")
        writer.add_array("tokenizer.ggml.tokens", tokens)
        writer.add_array("tokenizer.ggml.merges", merges)
        writer.add_key_value("tokenizer.ggml.token_type", types, gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.INT32)
        for key, value in {"bos_token_id": 0, "eos_token_id": 0, "padding_token_id": 0, "separator_token_id": 15, "mask_token_id": 16}.items():
            writer.add_uint32("tokenizer.ggml." + key, value)
        writer.add_bool("tokenizer.ggml.add_bos_token", True)
        writer.add_bool("tokenizer.ggml.add_eos_token", False)
        manifest = []
        try:
            with safe_open(str(directory / "model.safetensors"), framework="np") as source:
                for name, (target, shape) in expected_tensors().items():
                    array = source.get_tensor(name)
                    if list(array.shape) != shape or array.dtype != np.float32 or not np.isfinite(array).all():
                        raise ValueError("Invalid original tensor: " + name)
                    # Retain every RMSNorm at F32. No quantization/calibration here.
                    precision = np.float32 if array.ndim == 1 or dtype == "f32" else np.float16
                    converted = np.ascontiguousarray(array, dtype=precision)
                    if not np.isfinite(converted).all():
                        raise ValueError("Non-finite converted tensor: " + name)
                    writer.add_tensor(target, converted)
                    manifest.append({"source": name, "target": target, "shape": shape,
                                     "dtype": str(converted.dtype),
                                     "max_abs_conversion_error": float(np.max(np.abs(converted.astype(np.float32) - array)))})
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
        finally:
            writer.close()
        os.replace(temporary, output)
    return {"model": MODEL, "revision": REVISION, "source_weights": weights["sha256"],
            "dtype": dtype, "bytes": output.stat().st_size, "sha256": sha256(output),
            "tensors": manifest, "native_parity": "not evaluated"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=("f32", "f16"), default="f32")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = convert(args.model_dir, args.output, args.dtype)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Converted {len(report['tensors'])} tensors; {report['bytes']} bytes; SHA256 {report['sha256']}")


if __name__ == "__main__":
    main()
