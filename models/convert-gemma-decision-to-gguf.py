#!/usr/bin/env python
"""Export the complete typed-decision model, preserving FP32 precision by default.

No inference or remote code execution is needed for conversion. Quantized variants
must be evaluated separately; head, norms, scalars and projection stay FP32.
"""

import argparse
import json
from pathlib import Path

import gguf
import numpy as np
from safetensors import safe_open


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--dtype", choices=["f32", "q8_0"], default="f32")
    args = ap.parse_args()
    config = json.loads((args.model / "config.json").read_text())
    assert config["model_type"] == "gemma_decision"
    assert config["encoder_class"] == "EmbeddingGemma2Model"
    assert config["encoder_method"] == "forward" and config["encoder_output"] == [
        "last_hidden_state"
    ]
    assert (
        config["pooling_mode"] == "mean"
        and config["pooling_include_prompt"]
        and not config["projections"]
    )
    enc = config["encoder_config"]
    assert enc["vision_config"] is None and enc["audio_config"] is None
    tc = enc["text_config"]
    assert (
        tc["hidden_size_per_layer_input"] == tc["hidden_size"]
        and tc["embedding_dim"] == config["embed_dim"]
    )
    assert config["question_types"] == ["choice", "score", "noul"]
    tok = json.loads((args.model / "tokenizer.json").read_text())
    assert tok["model"]["type"] == "BPE" and tok["model"]["byte_fallback"]
    assert tok["normalizer"] == {
        "type": "Replace",
        "pattern": {"String": " "},
        "content": "▁",
    }
    vocab = [""] * tc["vocab_size"]
    for token, idx in tok["model"]["vocab"].items():
        vocab[idx] = token
    special_tokens = []
    special_ids = []
    for t in tok["added_tokens"]:
        assert not any(t[k] for k in ("single_word", "lstrip", "rstrip", "normalized"))
        assert vocab[t["id"]] == t["content"]
        special_tokens.append(t["content"])
        special_ids.append(t["id"])
    writer = gguf.GGUFWriter(str(args.output), "gemma_decision")
    writer.add_string(
        "general.source.url",
        "https://huggingface.co/mrm8488/embeddinggemma-2-ft-typed-decision",
    )
    writer.add_string("gemma_decision.prompt", config["prompt"])
    for key, value in {
        "hidden_size": tc["hidden_size"],
        "embedding_dim": tc["embedding_dim"],
        "n_layer": tc["num_hidden_layers"],
        "n_head": tc["num_attention_heads"],
        "head_dim": tc["head_dim"],
        "n_kv_head": tc["num_key_value_heads"],
        "intermediate_size": tc["intermediate_size"],
        "vocab_size": tc["vocab_size"],
        "max_length": config["max_seq_length"],
        "sliding_window": tc["sliding_window"],
        "head_width": config["head_width"],
    }.items():
        writer.add_uint32("gemma_decision." + key, value)
    writer.add_float32("gemma_decision.rms_norm_eps", tc["rms_norm_eps"])
    for kind, value in tc["rope_parameters"].items():
        assert value["rope_type"] == "default"
        writer.add_float32("gemma_decision.rope." + kind, value["rope_theta"])
    writer.add_array("gemma_decision.layer_types", tc["layer_types"])
    hd, kv = [], []
    for i in range(tc["num_hidden_layers"]):
        override = tc["per_layer_config"].get(f"{i:02d}", {})
        hd.append(override.get("head_dim", tc["head_dim"]))
        kv.append(override.get("num_key_value_heads", tc["num_key_value_heads"]))
    writer.add_array("gemma_decision.layer_head_dim", hd)
    writer.add_array("gemma_decision.layer_kv_heads", kv)
    writer.add_array("tokenizer.ggml.tokens", vocab)
    writer.add_array(
        "tokenizer.ggml.merges",
        [" ".join(m) if isinstance(m, list) else m for m in tok["model"]["merges"]],
    )
    writer.add_array("gemma_decision.special_tokens", special_tokens)
    writer.add_array("gemma_decision.special_ids", special_ids)
    writer.add_uint32("tokenizer.ggml.bos_token_id", tc["bos_token_id"])
    writer.add_uint32("tokenizer.ggml.eos_token_id", tc["eos_token_id"])
    count = 0
    with safe_open(args.model / "model.safetensors", framework="numpy") as weights:
        for name in weights.keys():
            stored_name = name.replace("encoder.language_model.", "enc.", 1)
            assert len(stored_name.encode("utf-8")) < 64, stored_name
            data = weights.get_tensor(name).astype(np.float32)
            if data.ndim == 0:
                data = data.reshape(1)
            quantize = (
                args.dtype == "q8_0"
                and data.ndim == 2
                and name.startswith("encoder.")
                and "embedding_projection" not in name
            )
            if quantize:
                writer.add_tensor(
                    stored_name,
                    gguf.quantize(data, gguf.GGMLQuantizationType.Q8_0),
                    raw_dtype=gguf.GGMLQuantizationType.Q8_0,
                )
            else:
                writer.add_tensor(stored_name, data)
            count += 1
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Exported {count} tensors, {args.dtype}, {args.output.stat().st_size} bytes")


if __name__ == "__main__":
    main()
