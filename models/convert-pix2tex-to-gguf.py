#!/usr/bin/env python
"""Convert pix2text-mfr (VisionEncoderDecoderModel) to GGUF.

Downloads the ONNX encoder + decoder from breezedeus/pix2text-mfr
(MIT license), extracts the weights, and packs them into a single
GGUF file for CrispEmbed's math OCR inference.

Architecture:
  Encoder: DeiT (12 layers, 6 heads, hidden=384, patch=16, image=384×384)
  Decoder: TrOCR (6 layers, 8 heads, d_model=256, FFN=1024, vocab=1200)
  Cross-attention: encoder hidden (384) → decoder (256) via learned projection.

Usage:
    pip install onnx gguf numpy
    python models/convert-pix2tex-to-gguf.py \\
        --model-dir /mnt/storage/models/pix2tex \\
        --output /mnt/storage/models/pix2tex/pix2tex-mfr-f32.gguf
"""

import argparse
import json
import sys
from pathlib import Path

import gguf
import numpy as np
import onnx


def main():
    p = argparse.ArgumentParser(
        description="Convert pix2text-mfr ONNX to GGUF"
    )
    p.add_argument("--model-dir", required=True,
                    help="Directory with config.json, tokenizer.json, "
                         "encoder_model.onnx, decoder_model.onnx")
    p.add_argument("--output", required=True, help="Output GGUF path")
    p.add_argument("--fp16", action="store_true",
                    help="Store weights in FP16 (halves file size)")
    args = p.parse_args()

    model_dir = Path(args.model_dir)

    # Load config
    with open(model_dir / "config.json") as f:
        config = json.load(f)

    enc_cfg = config["encoder"]
    dec_cfg = config["decoder"]

    print(f"Encoder: {enc_cfg['model_type']} — "
          f"{enc_cfg['num_hidden_layers']}L, {enc_cfg['num_attention_heads']}H, "
          f"dim={enc_cfg['hidden_size']}")
    print(f"Decoder: {dec_cfg['model_type']} — "
          f"{dec_cfg['decoder_layers']}L, {dec_cfg['decoder_attention_heads']}H, "
          f"d_model={dec_cfg['d_model']}, vocab={dec_cfg['vocab_size']}")

    # Load ONNX models
    print("\nLoading encoder ONNX...")
    enc_model = onnx.load(str(model_dir / "encoder_model.onnx"))
    enc_weights = {init.name: onnx.numpy_helper.to_array(init)
                   for init in enc_model.graph.initializer}
    print(f"  {len(enc_weights)} initializers")

    print("Loading decoder ONNX...")
    dec_model = onnx.load(str(model_dir / "decoder_model.onnx"))
    dec_weights = {init.name: onnx.numpy_helper.to_array(init)
                   for init in dec_model.graph.initializer}
    print(f"  {len(dec_weights)} initializers")

    # Print sample weight names
    print("\nEncoder weights (first 10):")
    for k in sorted(enc_weights)[:10]:
        print(f"  {k}: {enc_weights[k].shape}")

    print("\nDecoder weights (first 10):")
    for k in sorted(dec_weights)[:10]:
        print(f"  {k}: {dec_weights[k].shape}")

    # Load tokenizer
    tok_data = None
    tok_path = model_dir / "tokenizer.json"
    if tok_path.exists():
        with open(tok_path) as f:
            tok_data = json.load(f)

    # Write GGUF
    writer = gguf.GGUFWriter(str(args.output), arch="pix2tex_mfr")

    # Metadata
    writer.add_string("general.architecture", "pix2tex_mfr")
    writer.add_string("general.name", "pix2text-mfr-math-ocr")
    writer.add_string("general.license", "MIT")

    # Encoder hparams
    writer.add_uint32("encoder.num_hidden_layers", enc_cfg["num_hidden_layers"])
    writer.add_uint32("encoder.num_attention_heads", enc_cfg["num_attention_heads"])
    writer.add_uint32("encoder.hidden_size", enc_cfg["hidden_size"])
    writer.add_uint32("encoder.intermediate_size", enc_cfg.get("intermediate_size", 1536))
    writer.add_uint32("encoder.image_size", enc_cfg["image_size"])
    writer.add_uint32("encoder.patch_size", enc_cfg["patch_size"])

    # Decoder hparams
    writer.add_uint32("decoder.decoder_layers", dec_cfg["decoder_layers"])
    writer.add_uint32("decoder.decoder_attention_heads", dec_cfg["decoder_attention_heads"])
    writer.add_uint32("decoder.d_model", dec_cfg["d_model"])
    writer.add_uint32("decoder.decoder_ffn_dim", dec_cfg["decoder_ffn_dim"])
    writer.add_uint32("decoder.vocab_size", dec_cfg["vocab_size"])
    writer.add_uint32("decoder.max_position_embeddings", dec_cfg["max_position_embeddings"])
    writer.add_uint32("decoder.cross_attention_hidden_size", dec_cfg["cross_attention_hidden_size"])
    writer.add_uint32("decoder.bos_token_id", dec_cfg.get("bos_token_id", 0))
    writer.add_uint32("decoder.eos_token_id", dec_cfg.get("eos_token_id", 2))
    writer.add_uint32("decoder.pad_token_id", dec_cfg.get("pad_token_id", 1))
    writer.add_uint32("decoder.decoder_start_token_id", dec_cfg.get("decoder_start_token_id", 2))

    # Tokenizer
    if tok_data:
        vocab = tok_data.get("model", {}).get("vocab", {})
        if vocab:
            # Sort by token id
            sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
            token_list = [t[0] for t in sorted_vocab]
            writer.add_array("tokenizer.tokens", token_list)
            print(f"\nTokenizer: {len(token_list)} tokens embedded")

    # Store tensors
    dtype_np = np.float16 if args.fp16 else np.float32
    dtype_gguf = gguf.GGMLQuantizationType.F16 if args.fp16 else gguf.GGMLQuantizationType.F32

    total_params = 0
    tensor_count = 0

    # Encoder tensors (prefix with "enc.")
    for name, arr in enc_weights.items():
        gguf_name = "enc." + name.replace("/", "_").replace(":", "_")
        data = arr.astype(dtype_np)
        total_params += data.size
        writer.add_tensor(gguf_name, data, raw_dtype=dtype_gguf)
        tensor_count += 1

    # Decoder tensors (prefix with "dec.")
    for name, arr in dec_weights.items():
        gguf_name = "dec." + name.replace("/", "_").replace(":", "_")
        data = arr.astype(dtype_np)
        total_params += data.size
        writer.add_tensor(gguf_name, data, raw_dtype=dtype_gguf)
        tensor_count += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    output_size = Path(args.output).stat().st_size / (1024 * 1024)
    print(f"\nWritten: {args.output}")
    print(f"  Tensors: {tensor_count}")
    print(f"  Parameters: {total_params:,}")
    print(f"  File size: {output_size:.1f} MB")
    print(f"  Format: {'FP16' if args.fp16 else 'FP32'}")
    print(f"\nQuantize with: crispembed-quantize {args.output} pix2tex-mfr-q8_0.gguf q8_0")


if __name__ == "__main__":
    main()
