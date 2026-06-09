#!/usr/bin/env python
"""Convert any TrOCR VisionEncoderDecoderModel to GGUF.

Supports both ONNX models (like pix2text-mfr) and PyTorch checkpoints
(like microsoft/trocr-small-handwritten). Auto-detects the format.

Usage:
    python models/convert-trocr-to-gguf.py \
        --model-dir /mnt/storage/models/trocr-small-handwritten \
        --output /mnt/storage/models/trocr-small-handwritten.gguf

    python models/convert-trocr-to-gguf.py \
        --model-dir /mnt/storage/models/pix2tex \
        --output /mnt/storage/models/pix2tex-mfr.gguf
"""

import argparse
import json
import sys
from pathlib import Path

import gguf
import numpy as np


def load_weights_pytorch(model_dir: Path):
    """Load weights from pytorch_model.bin or model.safetensors."""
    import torch

    pt_path = model_dir / "pytorch_model.bin"
    sf_path = model_dir / "model.safetensors"

    if sf_path.exists():
        from safetensors.torch import load_file
        sd = load_file(str(sf_path))
    elif pt_path.exists():
        sd = torch.load(str(pt_path), map_location="cpu")
    else:
        return None

    # Convert to numpy
    return {k: v.detach().float().cpu().numpy() for k, v in sd.items()}


def load_weights_onnx(model_dir: Path):
    """Load weights from ONNX encoder + decoder models."""
    import onnx

    weights = {}
    for name, prefix in [("encoder_model.onnx", "enc."), ("decoder_model.onnx", "dec.")]:
        path = model_dir / name
        if not path.exists():
            return None
        model = onnx.load(str(path))
        for init in model.graph.initializer:
            arr = onnx.numpy_helper.to_array(init)
            weights[prefix + init.name] = arr

    return weights


def main():
    p = argparse.ArgumentParser(
        description="Convert TrOCR VisionEncoderDecoderModel to GGUF"
    )
    p.add_argument("--model-dir", required=True, help="Model directory")
    p.add_argument("--output", required=True, help="Output GGUF path")
    p.add_argument("--fp16", action="store_true", help="FP16 weights")
    p.add_argument("--name", default=None, help="Model name for metadata")
    args = p.parse_args()

    model_dir = Path(args.model_dir)

    # Load config
    config_path = model_dir / "config.json"
    if not config_path.exists():
        print(f"ERROR: {config_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)

    enc_cfg = config.get("encoder", {})
    dec_cfg = config.get("decoder", {})

    print(f"Encoder: {enc_cfg.get('model_type', '?')} — "
          f"{enc_cfg.get('num_hidden_layers', '?')}L, "
          f"h={enc_cfg.get('hidden_size', '?')}")
    print(f"Decoder: {dec_cfg.get('model_type', '?')} — "
          f"{dec_cfg.get('decoder_layers', '?')}L, "
          f"d={dec_cfg.get('d_model', '?')}, "
          f"vocab={dec_cfg.get('vocab_size', '?')}")

    # Load weights (try ONNX first, fall back to PyTorch)
    print("\nLoading weights...")
    weights = load_weights_onnx(model_dir)
    source = "ONNX"

    if weights is None:
        weights = load_weights_pytorch(model_dir)
        source = "PyTorch"

    if weights is None:
        print("ERROR: No model weights found (tried ONNX + PyTorch)",
              file=sys.stderr)
        sys.exit(1)

    print(f"  Source: {source}")
    print(f"  Tensors: {len(weights)}")

    # Print sample keys
    for k in sorted(weights)[:8]:
        print(f"    {k}: {weights[k].shape}")
    if len(weights) > 8:
        print(f"    ... ({len(weights) - 8} more)")

    # Load tokenizer — try tokenizer.json first, fall back to SentencePiece
    tok_tokens = None
    tok_path = model_dir / "tokenizer.json"
    spm_path = model_dir / "sentencepiece.bpe.model"

    if tok_path.exists():
        with open(tok_path) as f:
            tok_data = json.load(f)
        vocab = tok_data.get("model", {}).get("vocab", {})
        if vocab:
            sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
            tok_tokens = [t[0] for t in sorted_vocab]

    if tok_tokens is None and spm_path.exists():
        # XLM-R / TrOCR uses SentencePiece BUT with a fairseq vocab offset.
        # The HF tokenizer handles this correctly. Always prefer HF tokenizer
        # to get the right token ID → string mapping.
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(str(model_dir))
            n_tokens = dec_cfg.get("vocab_size", tok.vocab_size)
            tok_tokens = []
            # Use convert_ids_to_tokens to preserve ▁ space markers
            for i in range(n_tokens):
                try:
                    piece = tok.convert_ids_to_tokens(i)
                    if piece is None:
                        piece = f"<unk_{i}>"
                    tok_tokens.append(piece)
                except Exception:
                    tok_tokens.append(f"<unk_{i}>")
            print(f"Loaded {len(tok_tokens)} tokens via AutoTokenizer (XLM-R/SentencePiece)")
        except Exception:
            # Fallback to raw SentencePiece (may have offset issues)
            try:
                import sentencepiece as spm
                sp = spm.SentencePieceProcessor(model_file=str(spm_path))
                tok_tokens = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
                print(f"Loaded {len(tok_tokens)} tokens from sentencepiece.bpe.model (raw)")
            except Exception as e:
                print(f"WARNING: cannot load tokenizer: {e}")

    # Write GGUF
    model_name = args.name or model_dir.name
    writer = gguf.GGUFWriter(str(args.output), arch="trocr")

    writer.add_string("general.architecture", "trocr")
    writer.add_string("general.name", model_name)

    # Encoder
    writer.add_uint32("encoder.num_hidden_layers",
                       enc_cfg.get("num_hidden_layers", 12))
    writer.add_uint32("encoder.num_attention_heads",
                       enc_cfg.get("num_attention_heads", 6))
    writer.add_uint32("encoder.hidden_size",
                       enc_cfg.get("hidden_size", 384))
    writer.add_uint32("encoder.intermediate_size",
                       enc_cfg.get("intermediate_size", 1536))
    writer.add_uint32("encoder.image_size",
                       enc_cfg.get("image_size", 384))
    writer.add_uint32("encoder.patch_size",
                       enc_cfg.get("patch_size", 16))

    # Decoder
    writer.add_uint32("decoder.decoder_layers",
                       dec_cfg.get("decoder_layers", 6))
    writer.add_uint32("decoder.decoder_attention_heads",
                       dec_cfg.get("decoder_attention_heads", 8))
    writer.add_uint32("decoder.d_model",
                       dec_cfg.get("d_model", 256))
    writer.add_uint32("decoder.decoder_ffn_dim",
                       dec_cfg.get("decoder_ffn_dim", 1024))
    writer.add_uint32("decoder.vocab_size",
                       dec_cfg.get("vocab_size", 1200))
    writer.add_uint32("decoder.max_position_embeddings",
                       dec_cfg.get("max_position_embeddings", 512))
    writer.add_uint32("decoder.cross_attention_hidden_size",
                       dec_cfg.get("cross_attention_hidden_size",
                                   enc_cfg.get("hidden_size", 384)))
    writer.add_uint32("decoder.bos_token_id",
                       dec_cfg.get("bos_token_id", 0))
    writer.add_uint32("decoder.eos_token_id",
                       dec_cfg.get("eos_token_id", 2))
    writer.add_uint32("decoder.pad_token_id",
                       dec_cfg.get("pad_token_id", 1))
    writer.add_uint32("decoder.decoder_start_token_id",
                       dec_cfg.get("decoder_start_token_id", 2))

    # Tokenizer
    if tok_tokens:
        writer.add_array("tokenizer.tokens", tok_tokens)
        print(f"\nTokenizer: {len(tok_tokens)} tokens embedded")
    else:
        print("\nTokenizer: not embedded (no tokenizer.json)")

    # Tensors
    dtype_np = np.float16 if args.fp16 else np.float32
    dtype_gguf = (gguf.GGMLQuantizationType.F16 if args.fp16
                  else gguf.GGMLQuantizationType.F32)

    # Map HuggingFace PyTorch key names to the convention math_ocr.cpp expects.
    # The ONNX converter produces: enc.XXX and dec.XXX with dots.
    # PyTorch safetensors has: encoder.XXX and decoder.model.decoder.XXX
    def map_tensor_name(name):
        """Map HF PyTorch tensor name to math_ocr.cpp convention."""
        # Encoder: encoder.X.Y.Z → enc.X.Y.Z
        if name.startswith("encoder."):
            return "enc." + name[len("encoder."):]
        # Decoder: decoder.model.decoder.X → dec.d.X (matches ONNX convention)
        if name.startswith("decoder.model.decoder."):
            return "dec.d." + name[len("decoder.model.decoder."):]
        # Decoder head: decoder.lm_head.X → dec.lm_head.X
        if name.startswith("decoder.lm_head."):
            return "dec." + name[len("decoder."):]
        # Decoder output_projection: decoder.output_projection.X → dec.lm_head.X
        if name.startswith("decoder.output_projection."):
            return "dec.lm_head." + name[len("decoder.output_projection."):]
        # ONNX-style (already has enc./dec. prefix) — keep dots
        if name.startswith("enc.") or name.startswith("dec."):
            return name
        # Fallback: replace dots with underscores (legacy)
        return name.replace(".", "_").replace("/", "_").replace(":", "_")

    total_params = 0
    for name, arr in weights.items():
        gguf_name = map_tensor_name(name)
        data = arr.astype(dtype_np)
        total_params += data.size
        writer.add_tensor(gguf_name, data, raw_dtype=dtype_gguf)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    output_size = Path(args.output).stat().st_size / (1024 * 1024)
    print(f"\nWritten: {args.output}")
    print(f"  Tensors: {len(weights)}")
    print(f"  Parameters: {total_params:,}")
    print(f"  File size: {output_size:.1f} MB")
    print(f"  Format: {'FP16' if args.fp16 else 'FP32'}")


if __name__ == "__main__":
    main()
