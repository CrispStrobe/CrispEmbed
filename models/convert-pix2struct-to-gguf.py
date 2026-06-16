#!/usr/bin/env python3
"""Convert Pix2Struct to GGUF.

Usage:
    python models/convert-pix2struct-to-gguf.py \
        --model google/pix2struct-base \
        --output pix2struct-base-f32.gguf [--fp16]
"""
import argparse, sys
from collections import OrderedDict
from pathlib import Path
import numpy as np

try:
    import gguf
except ImportError:
    sys.exit("pip install gguf")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--output", "-o", required=True)
    p.add_argument("--fp16", action="store_true")
    args = p.parse_args()

    from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
    import torch

    model = Pix2StructForConditionalGeneration.from_pretrained(args.model)
    processor = Pix2StructProcessor.from_pretrained(args.model)
    config = model.config
    sd = {k: v.cpu().float().numpy() for k, v in model.state_dict().items()}

    writer = gguf.GGUFWriter(args.output, "pix2struct")

    # Hyperparameters
    enc_config = config.vision_config if hasattr(config, 'vision_config') else config.encoder
    dec_config = config.text_config if hasattr(config, 'text_config') else config.decoder
    writer.add_uint32("pix2struct.enc_layers", enc_config.num_hidden_layers)
    writer.add_uint32("pix2struct.dec_layers", dec_config.num_hidden_layers)
    writer.add_uint32("pix2struct.hidden_size", enc_config.hidden_size)
    writer.add_uint32("pix2struct.n_heads", enc_config.num_attention_heads)
    writer.add_uint32("pix2struct.d_ff", enc_config.d_ff if hasattr(enc_config, 'd_ff') else 2048)
    writer.add_uint32("pix2struct.d_kv", enc_config.d_kv if hasattr(enc_config, 'd_kv') else 64)
    writer.add_uint32("pix2struct.vocab_size", dec_config.vocab_size)
    writer.add_uint32("pix2struct.patch_size", 16)
    writer.add_uint32("pix2struct.max_patches", 2048)
    writer.add_uint32("pix2struct.rel_attn_buckets", 32)
    writer.add_uint32("pix2struct.rel_attn_max_dist", 128)

    # Tokenizer — save vocab from the processor
    tokenizer = processor.tokenizer
    tokens = [tokenizer.convert_ids_to_tokens(i) for i in range(len(tokenizer))]
    writer.add_array("tokenizer.tokens", tokens)
    writer.add_uint32("tokenizer.eos_token_id", tokenizer.eos_token_id or 1)
    writer.add_uint32("tokenizer.pad_token_id", tokenizer.pad_token_id or 0)

    dtype = gguf.GGMLQuantizationType.F16 if args.fp16 else gguf.GGMLQuantizationType.F32

    # Shorten tensor names to fit GGUF 64-char limit
    def shorten(k):
        k = k.replace("encoder_decoder_attention.attention", "xattn")
        k = k.replace("self_attention.attention", "sattn")
        k = k.replace("encoder.encoder.layer", "enc")
        k = k.replace("encoder.embeddings", "enc_emb")
        k = k.replace("decoder.layer", "dec")
        k = k.replace("pre_attention_layer_norm", "pre_attn_ln")
        k = k.replace("pre_mlp_layer_norm", "pre_mlp_ln")
        k = k.replace("DenseReluDense", "dense")
        k = k.replace("relative_attention_bias", "rel_bias")
        k = k.replace("self_attention.layer_norm", "sa_ln")
        k = k.replace("encoder_decoder_attention.layer_norm", "xa_ln")
        k = k.replace("mlp.layer_norm", "ffn_ln")
        k = k.replace("final_layer_norm", "final_ln")
        k = k.replace("decoder.embed_tokens", "dec_emb")
        k = k.replace("decoder.lm_head", "lm_head")
        k = k.replace("decoder.final_ln", "dec_final_ln")
        k = k.replace("patch_projection", "patch_proj")
        k = k.replace("row_embedder", "row_emb")
        k = k.replace("column_embedder", "col_emb")
        return k

    total = 0
    for k in sorted(sd.keys()):
        arr = sd[k].astype(np.float32)
        total += arr.size
        name = shorten(k)
        assert len(name) < 64, f"Name too long ({len(name)}): {name}"
        if "layer_norm" in k or "ln" in name or "bias" in k or "embed" in k.lower() or "lm_head" in k or arr.size < 256:
            writer.add_tensor(name, arr, raw_dtype=gguf.GGMLQuantizationType.F32)
        else:
            writer.add_tensor(name, arr, raw_dtype=dtype)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Written {args.output}: {total:,} params, {Path(args.output).stat().st_size/1024/1024:.0f} MB")
    print(f"  Encoder: {enc_config.num_hidden_layers}L, Decoder: {dec_config.num_hidden_layers}L")
    print(f"  Hidden: {enc_config.hidden_size}, Heads: {enc_config.num_attention_heads}")
    print(f"  Vocab: {dec_config.vocab_size}")


if __name__ == "__main__":
    main()
