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
    # ggml_mul_mat(A, B) = A^T @ B, so 2D weights need to be stored
    # transposed relative to ONNX convention so that A_stored^T = W_onnx.
    def store_tensor(writer, name, arr, dtype_np, dtype_gguf):
        nonlocal total_params, tensor_count
        data = arr.astype(dtype_np)
        # Transpose 2D weight matrices for ggml convention.
        # ggml_mul_mat(W, x) computes W_ggml^T @ x. Since ggml interprets
        # a numpy (M, N) array as ggml (N, M) due to column-major storage,
        # W_ggml = W_numpy.T, so ggml_mul_mat gives W_numpy @ x.
        # For ONNX MatMul(input, B) = input @ B, we need B.T @ x_col.
        # Storing B.T (transposed) makes ggml see (B.T).T_col = B,
        # and ggml_mul_mat gives B^T @ x = (input @ B).T. Correct!
        if data.ndim == 2 and 'position' not in name and 'embed_tokens' not in name:
            data = np.ascontiguousarray(data.T)
        total_params += data.size
        writer.add_tensor(name, data, raw_dtype=dtype_gguf)
        tensor_count += 1

    # Split encoder into named (biases, embeddings) and anonymous (MatMul weights)
    enc_named = {}
    enc_anon = []
    for name, arr in enc_weights.items():
        if name.startswith("onnx::MatMul") or name.startswith("onnx__MatMul"):
            enc_anon.append((name, arr))
        else:
            enc_named[name] = arr

    enc_anon.sort(key=lambda x: int(x[0].split("_")[-1]))

    # Write named encoder tensors (biases, embeddings, layernorms)
    for name, arr in enc_named.items():
        store_tensor(writer, "enc." + name, arr, dtype_np, dtype_gguf)

    # Rename anonymous weights to proper DeiT layer names.
    # Per layer: 4 × (H,H) = Q/K/V/Out weights, 1 × (H,I) = FFN up, 1 × (I,H) = FFN down
    n_enc_layers = enc_cfg["num_hidden_layers"]
    H_enc = enc_cfg["hidden_size"]
    I_enc = enc_cfg.get("intermediate_size", 1536)
    per_layer = len(enc_anon) // n_enc_layers if n_enc_layers > 0 else 0
    print(f"\n  Encoder: {len(enc_anon)} anonymous weights, {per_layer} per layer")

    enc_suffixes = [
        "attention.attention.query.weight",
        "attention.attention.key.weight",
        "attention.attention.value.weight",
        "attention.output.dense.weight",
        "intermediate.dense.weight",
        "output.dense.weight",
    ]
    for li in range(n_enc_layers):
        chunk = enc_anon[li * per_layer : (li + 1) * per_layer]
        for wi, (_, arr) in enumerate(chunk):
            suffix = enc_suffixes[wi] if wi < len(enc_suffixes) else f"unknown_{wi}"
            store_tensor(writer, f"enc.encoder.layer.{li}.{suffix}", arr, dtype_np, dtype_gguf)

    # Decoder: split named + anonymous, rename anonymous
    dec_named = {}
    dec_anon = []
    for name, arr in dec_weights.items():
        if name.startswith("onnx::MatMul") or name.startswith("onnx__MatMul"):
            dec_anon.append((name, arr))
        else:
            dec_named[name] = arr

    dec_anon.sort(key=lambda x: int(x[0].split("_")[-1]))

    for name, arr in dec_named.items():
        gguf_name = "dec." + name
        gguf_name = gguf_name.replace("decoder.model.decoder.", "d.")
        gguf_name = gguf_name.replace("encoder_attn_layer_norm", "xaln")
        store_tensor(writer, gguf_name, arr, dtype_np, dtype_gguf)

    # Decoder anonymous: last one is lm_head, rest are per-layer
    D = dec_cfg.get("d_model", 256)
    V = dec_cfg.get("vocab_size", 1200)
    n_dec_layers = dec_cfg.get("decoder_layers", 6)

    lm_head = None
    layer_anons = []
    for name, arr in dec_anon:
        if arr.shape == (D, V) or arr.shape == (V, D):
            lm_head = arr
        else:
            layer_anons.append(arr)

    if lm_head is not None:
        store_tensor(writer, "dec.lm_head.weight", lm_head, dtype_np, dtype_gguf)

    dec_per_layer = len(layer_anons) // n_dec_layers if n_dec_layers > 0 else 0
    print(f"  Decoder: {len(layer_anons)} anonymous weights, {dec_per_layer} per layer")

    dec_suffixes = [
        "self_attn.q_proj.weight", "self_attn.k_proj.weight",
        "self_attn.v_proj.weight", "self_attn.out_proj.weight",
        "encoder_attn.q_proj.weight", "encoder_attn.k_proj.weight",
        "encoder_attn.v_proj.weight", "encoder_attn.out_proj.weight",
        "fc1.weight", "fc2.weight",
    ]
    for li in range(n_dec_layers):
        chunk = layer_anons[li * dec_per_layer : (li + 1) * dec_per_layer]
        for wi, arr in enumerate(chunk):
            suffix = dec_suffixes[wi] if wi < len(dec_suffixes) else f"unknown_{wi}"
            store_tensor(writer, f"dec.d.layers.{li}.{suffix}", arr, dtype_np, dtype_gguf)

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
