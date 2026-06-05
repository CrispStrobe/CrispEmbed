#!/usr/bin/env python3
"""Convert TrOCR VisionEncoderDecoderModel (safetensors) to GGUF — NO PyTorch required.

Reads model.safetensors + config.json + tokenizer.json directly and
packs into a single GGUF file for CrispEmbed's math OCR inference.

Supports both:
  - pix2tex-style (DeiT encoder + TrOCR decoder, small/base)
  - microsoft/trocr-style (ViT encoder + TrOCR decoder, small/base/large)
  - fhswf/TrOCR_Math_handwritten (ViT-Large + TrOCR-Large)

Dependencies: safetensors, gguf, numpy (NO torch/transformers needed)

Usage:
    pip install safetensors gguf numpy
    python convert-trocr-safetensors-to-gguf.py \\
        --model-dir /path/to/model \\
        --output /path/to/output.gguf [--fp16]
"""

import argparse
import json
import sys
from pathlib import Path

import gguf
import numpy as np
from safetensors import safe_open


def main():
    p = argparse.ArgumentParser(
        description="Convert TrOCR safetensors to GGUF (torch-free)"
    )
    p.add_argument("--model-dir", required=True, help="Model directory")
    p.add_argument("--output", required=True, help="Output GGUF path")
    p.add_argument("--fp16", action="store_true", help="Store weights in FP16")
    p.add_argument("--name", default=None, help="Model name for metadata")
    args = p.parse_args()

    model_dir = Path(args.model_dir)

    # ---- Load config ----
    config_path = model_dir / "config.json"
    if not config_path.exists():
        print(f"Error: {config_path} not found", file=sys.stderr)
        return 1

    with open(config_path) as f:
        config = json.load(f)

    enc_cfg = config.get("encoder", config)
    dec_cfg = config.get("decoder", config)

    enc_layers = enc_cfg.get("num_hidden_layers", 12)
    enc_heads = enc_cfg.get("num_attention_heads", 6)
    enc_hidden = enc_cfg.get("hidden_size", 384)
    enc_intermediate = enc_cfg.get("intermediate_size", 1536)
    image_size = enc_cfg.get("image_size", 384)
    patch_size = enc_cfg.get("patch_size", 16)

    dec_layers = dec_cfg.get("decoder_layers", 6)
    dec_heads = dec_cfg.get("decoder_attention_heads", 8)
    dec_d_model = dec_cfg.get("d_model", 256)
    dec_ffn_dim = dec_cfg.get("decoder_ffn_dim", 1024)
    vocab_size = dec_cfg.get("vocab_size", 50265)
    max_seq_len = dec_cfg.get("max_position_embeddings", 512)
    cross_attn_dim = dec_cfg.get("cross_attention_hidden_size") or enc_hidden

    bos = dec_cfg.get("bos_token_id", 0)
    eos = dec_cfg.get("eos_token_id", 2)
    pad = dec_cfg.get("pad_token_id", 1)
    dec_start = dec_cfg.get("decoder_start_token_id", eos)

    print(f"Encoder: {enc_layers}L/{enc_heads}H/{enc_hidden}d, image={image_size}, patch={patch_size}")
    print(f"Decoder: {dec_layers}L/{dec_heads}H/{dec_d_model}d, vocab={vocab_size}, ffn={dec_ffn_dim}")

    # ---- Load tokenizer ----
    tokens = []
    tok_path = model_dir / "tokenizer.json"
    if tok_path.exists():
        with open(tok_path) as f:
            tok = json.load(f)
        vocab_map = tok.get("model", {}).get("vocab", {})
        if vocab_map:
            tokens = [""] * (max(vocab_map.values()) + 1)
            for word, idx in vocab_map.items():
                if idx < len(tokens):
                    tokens[idx] = word
            print(f"Tokenizer: {len(tokens)} tokens from tokenizer.json")
    else:
        # Try vocab.json
        vocab_path = model_dir / "vocab.json"
        if vocab_path.exists():
            with open(vocab_path) as f:
                vocab_map = json.load(f)
            tokens = [""] * (max(vocab_map.values()) + 1)
            for word, idx in vocab_map.items():
                if idx < len(tokens):
                    tokens[idx] = word
            print(f"Tokenizer: {len(tokens)} tokens from vocab.json")

    # ---- Load safetensors weights ----
    st_path = model_dir / "model.safetensors"
    if not st_path.exists():
        # Try pytorch_model.bin fallback
        print(f"Error: {st_path} not found (pytorch_model.bin requires torch)", file=sys.stderr)
        return 1

    print(f"Loading weights from {st_path}...")
    tensors = {}
    with safe_open(str(st_path), framework="numpy") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    print(f"Loaded {len(tensors)} tensors")

    # ---- Build name mapping ----
    # HuggingFace keys → CrispEmbed GGUF keys
    gguf_tensors = {}

    def add(gguf_name, hf_name):
        if hf_name in tensors:
            gguf_tensors[gguf_name] = tensors[hf_name]
        else:
            # Try alternative prefixes
            for prefix in ["encoder.", "decoder.model.decoder.", "decoder."]:
                alt = prefix + hf_name
                if alt in tensors:
                    gguf_tensors[gguf_name] = tensors[alt]
                    return
            # Print warning for missing
            pass

    # Encoder embeddings
    for hf_prefix in ["encoder.embeddings.", "encoder.deit.embeddings."]:
        if f"{hf_prefix}cls_token" in tensors:
            add("enc.embeddings.cls_token", f"{hf_prefix}cls_token")
            add("enc.embeddings.patch_embeddings.projection.weight",
                f"{hf_prefix}patch_embeddings.projection.weight")
            add("enc.embeddings.patch_embeddings.projection.bias",
                f"{hf_prefix}patch_embeddings.projection.bias")
            add("enc.embeddings.position_embeddings",
                f"{hf_prefix}position_embeddings")
            # Distillation token (DeiT only)
            if f"{hf_prefix}distillation_token" in tensors:
                add("enc.embeddings.distillation_token",
                    f"{hf_prefix}distillation_token")
            break

    # Encoder layers
    for hf_layer_prefix in ["encoder.encoder.layer.", "encoder.deit.encoder.layer."]:
        if f"{hf_layer_prefix}0.attention.attention.query.weight" in tensors:
            for i in range(enc_layers):
                lp = f"{hf_layer_prefix}{i}"
                gp = f"enc.encoder.layer.{i}"
                add(f"{gp}.layernorm_before.weight", f"{lp}.layernorm_before.weight")
                add(f"{gp}.layernorm_before.bias", f"{lp}.layernorm_before.bias")
                add(f"{gp}.attention.attention.query.weight", f"{lp}.attention.attention.query.weight")
                add(f"{gp}.attention.attention.query.bias", f"{lp}.attention.attention.query.bias")
                add(f"{gp}.attention.attention.key.weight", f"{lp}.attention.attention.key.weight")
                add(f"{gp}.attention.attention.key.bias", f"{lp}.attention.attention.key.bias")
                add(f"{gp}.attention.attention.value.weight", f"{lp}.attention.attention.value.weight")
                add(f"{gp}.attention.attention.value.bias", f"{lp}.attention.attention.value.bias")
                add(f"{gp}.attention.output.dense.weight", f"{lp}.attention.output.dense.weight")
                add(f"{gp}.attention.output.dense.bias", f"{lp}.attention.output.dense.bias")
                add(f"{gp}.layernorm_after.weight", f"{lp}.layernorm_after.weight")
                add(f"{gp}.layernorm_after.bias", f"{lp}.layernorm_after.bias")
                add(f"{gp}.intermediate.dense.weight", f"{lp}.intermediate.dense.weight")
                add(f"{gp}.intermediate.dense.bias", f"{lp}.intermediate.dense.bias")
                add(f"{gp}.output.dense.weight", f"{lp}.output.dense.weight")
                add(f"{gp}.output.dense.bias", f"{lp}.output.dense.bias")
            break

    # Encoder final LayerNorm
    for prefix in ["encoder.layernorm.", "encoder.deit.layernorm."]:
        if f"{prefix}weight" in tensors:
            add("enc.layernorm.weight", f"{prefix}weight")
            add("enc.layernorm.bias", f"{prefix}bias")
            break

    # Decoder embeddings
    for dp in ["decoder.model.decoder.", "decoder."]:
        if f"{dp}embed_tokens.weight" in tensors:
            add("dec.d.embed_tokens.weight", f"{dp}embed_tokens.weight")
            add("dec.d.embed_positions.weight", f"{dp}embed_positions.weight")
            add("dec.d.layernorm_embedding.weight", f"{dp}layernorm_embedding.weight")
            add("dec.d.layernorm_embedding.bias", f"{dp}layernorm_embedding.bias")
            add("dec.d.layer_norm.weight", f"{dp}layer_norm.weight")
            add("dec.d.layer_norm.bias", f"{dp}layer_norm.bias")

            # Decoder layers
            for i in range(dec_layers):
                lp = f"{dp}layers.{i}"
                gp = f"dec.d.layers.{i}"
                for suffix in [
                    "self_attn_layer_norm.weight", "self_attn_layer_norm.bias",
                    "self_attn.q_proj.weight", "self_attn.q_proj.bias",
                    "self_attn.k_proj.weight", "self_attn.k_proj.bias",
                    "self_attn.v_proj.weight", "self_attn.v_proj.bias",
                    "self_attn.out_proj.weight", "self_attn.out_proj.bias",
                    "encoder_attn_layer_norm.weight", "encoder_attn_layer_norm.bias",
                    "encoder_attn.q_proj.weight", "encoder_attn.q_proj.bias",
                    "encoder_attn.k_proj.weight", "encoder_attn.k_proj.bias",
                    "encoder_attn.v_proj.weight", "encoder_attn.v_proj.bias",
                    "encoder_attn.out_proj.weight", "encoder_attn.out_proj.bias",
                    "final_layer_norm.weight", "final_layer_norm.bias",
                    "fc1.weight", "fc1.bias",
                    "fc2.weight", "fc2.bias",
                ]:
                    add(f"{gp}.{suffix}", f"{lp}.{suffix}")
            break

    # LM head
    if "decoder.output_projection.weight" in tensors:
        add("dec.lm_head.weight", "decoder.output_projection.weight")
    elif "lm_head.weight" in tensors:
        add("dec.lm_head.weight", "lm_head.weight")
    # Bias (if present)
    for key in ["decoder.output_projection.bias", "lm_head.bias"]:
        if key in tensors:
            add("dec.lm_head.bias", key)

    print(f"Mapped {len(gguf_tensors)} tensors to GGUF names")

    # ---- Weight matrix layout ----
    # ggml_mul_mat(W, x) computes x @ W^T when W is [out_dim, in_dim].
    # HuggingFace safetensors store weights as [out_dim, in_dim] — same
    # convention as ggml. NO transpose needed (unlike ONNX models where
    # the pix2tex converter transposes because ONNX uses [in, out]).
    #
    # NOTE: The pix2tex ONNX converter (convert-pix2tex-to-gguf.py)
    # DOES transpose because ONNX MatMul uses the opposite convention.
    # This safetensors converter does NOT transpose.

    # ---- Write GGUF ----
    print(f"Writing GGUF to {args.output}...")
    writer = gguf.GGUFWriter(args.output, "math_ocr")

    # Metadata
    model_name = args.name or f"TrOCR Math OCR ({enc_layers}L enc + {dec_layers}L dec)"
    writer.add_name(model_name)
    writer.add_description("TrOCR VisionEncoderDecoderModel for math equation recognition")

    # Hyperparameters
    writer.add_uint32("encoder.num_hidden_layers", enc_layers)
    writer.add_uint32("encoder.num_attention_heads", enc_heads)
    writer.add_uint32("encoder.hidden_size", enc_hidden)
    writer.add_uint32("encoder.intermediate_size", enc_intermediate)
    writer.add_uint32("encoder.image_size", image_size)
    writer.add_uint32("encoder.patch_size", patch_size)
    writer.add_uint32("decoder.decoder_layers", dec_layers)
    writer.add_uint32("decoder.decoder_attention_heads", dec_heads)
    writer.add_uint32("decoder.d_model", dec_d_model)
    writer.add_uint32("decoder.decoder_ffn_dim", dec_ffn_dim)
    writer.add_uint32("decoder.vocab_size", vocab_size)
    writer.add_uint32("decoder.max_position_embeddings", max_seq_len)
    writer.add_uint32("decoder.cross_attention_hidden_size", cross_attn_dim)
    writer.add_uint32("decoder.bos_token_id", bos)
    writer.add_uint32("decoder.eos_token_id", eos)
    writer.add_uint32("decoder.pad_token_id", pad)
    writer.add_uint32("decoder.decoder_start_token_id", dec_start)

    # Tokenizer — use the same key as pix2tex GGUF ("tokenizer.tokens")
    # NOT add_token_list() which writes "tokenizer.ggml.tokens"
    if tokens:
        writer.add_array("tokenizer.tokens", tokens)

    # License
    writer.add_string("general.license", "AFL-3.0")

    # Tensors
    dtype = gguf.GGMLQuantizationType.F16 if args.fp16 else gguf.GGMLQuantizationType.F32
    for name, data in sorted(gguf_tensors.items()):
        if args.fp16 and data.dtype == np.float32:
            data = data.astype(np.float16)
        writer.add_tensor(name, data)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    import os
    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"Done! {args.output} ({size_mb:.0f} MB, {len(gguf_tensors)} tensors)")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
