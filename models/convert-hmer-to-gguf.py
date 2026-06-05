#!/usr/bin/env python3
"""Convert Pytorch-HMER (DenseNet-121 + GRU Attention) to GGUF.

Loads the encoder and decoder checkpoints from
whywhs/Pytorch-Handwritten-Mathematical-Expression-Recognition (MIT license),
folds BatchNorm into preceding Conv2d where possible, and packs everything
into a single GGUF file for CrispEmbed's handwritten math OCR inference.

Architecture:
  Encoder: DenseNet-121 (3 dense blocks, 2-channel input, 1024-ch output)
  Decoder: 2× GRUCell + Bahdanau attention + coverage (112 LaTeX tokens)

Usage:
    pip install gguf numpy
    python models/convert-hmer-to-gguf.py \
        --model-dir /mnt/storage/Pytorch-HMER/model \
        --dict /mnt/storage/Pytorch-HMER/dictionary.txt \
        --output /mnt/storage/models/hmer-hw-f32.gguf
"""

import argparse
import sys
from pathlib import Path

import gguf
import numpy as np


# ---------------------------------------------------------------------------
# BatchNorm folding
# ---------------------------------------------------------------------------

def fold_bn_into_conv(conv_w, bn_weight, bn_bias, bn_mean, bn_var, eps=1e-5):
    """Fold BN(gamma, beta, mean, var) into conv weight + bias.

    conv_w: (out_ch, in_ch, kH, kW)
    Returns: (fused_w, fused_b)
    """
    out_ch = conv_w.shape[0]
    scale = bn_weight / np.sqrt(bn_var + eps)
    # Reshape scale for broadcasting: (out_ch, 1, 1, 1)
    fused_w = conv_w * scale.reshape(out_ch, 1, 1, 1)
    fused_b = bn_bias - bn_mean * scale
    return fused_w, fused_b


def fold_bn_standalone(bn_weight, bn_bias, bn_mean, bn_var, eps=1e-5):
    """For standalone BN (not preceded by conv), return scale + offset.

    y = (x - mean) / sqrt(var + eps) * gamma + beta
      = x * scale + offset
    where scale = gamma / sqrt(var + eps), offset = beta - mean * scale
    """
    scale = bn_weight / np.sqrt(bn_var + eps)
    offset = bn_bias - bn_mean * scale
    return scale, offset


# ---------------------------------------------------------------------------
# Dictionary loading
# ---------------------------------------------------------------------------

def load_dictionary(dict_path):
    """Load HMER dictionary.txt → list of token strings indexed by ID."""
    tokens = {}
    with open(dict_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                tok_str = parts[0]
                tok_id = int(parts[-1])
                tokens[tok_id] = tok_str
    # Build ordered list (index 0..max_id)
    max_id = max(tokens.keys())
    result = []
    for i in range(max_id + 1):
        result.append(tokens.get(i, f"<unk_{i}>"))
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Convert Pytorch-HMER to GGUF")
    p.add_argument("--model-dir", required=True,
                   help="Directory containing encoder_*.pkl and attn_decoder_*.pkl")
    p.add_argument("--dict", required=True,
                   help="Path to dictionary.txt")
    p.add_argument("--output", required=True, help="Output GGUF path")
    p.add_argument("--fp16", action="store_true",
                   help="Store weights in FP16 (halves file size)")
    args = p.parse_args()

    model_dir = Path(args.model_dir)

    # Find checkpoint files
    enc_files = sorted(model_dir.glob("encoder_*.pkl"))
    dec_files = sorted(model_dir.glob("attn_decoder_*.pkl"))
    if not enc_files:
        print("ERROR: no encoder_*.pkl found in", model_dir)
        sys.exit(1)
    if not dec_files:
        print("ERROR: no attn_decoder_*.pkl found in", model_dir)
        sys.exit(1)

    enc_path = enc_files[0]
    dec_path = dec_files[0]
    print(f"Encoder: {enc_path.name} ({enc_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Decoder: {dec_path.name} ({dec_path.stat().st_size / 1e6:.1f} MB)")

    # Load with torch (need it for unpickling)
    # Filter out local broken torch install
    import importlib
    _orig_path = sys.path[:]
    sys.path = [p for p in sys.path if '.local' not in p]
    import torch
    sys.path = _orig_path

    enc_sd = torch.load(str(enc_path), map_location="cpu", weights_only=False)
    dec_sd = torch.load(str(dec_path), map_location="cpu", weights_only=False)

    # Strip "module." prefix (DataParallel)
    def strip_prefix(sd):
        out = {}
        for k, v in sd.items():
            name = k[len("module."):] if k.startswith("module.") else k
            out[name] = v.numpy() if hasattr(v, 'numpy') else np.array(v)
        return out

    enc = strip_prefix(enc_sd)
    dec = strip_prefix(dec_sd)

    print(f"\nEncoder: {len(enc)} tensors")
    print(f"Decoder: {len(dec)} tensors")

    # Load dictionary
    tokens = load_dictionary(args.dict)
    print(f"Dictionary: {len(tokens)} tokens")

    # -----------------------------------------------------------------------
    # Write GGUF
    # -----------------------------------------------------------------------
    writer = gguf.GGUFWriter(str(args.output), arch="hmer")

    # Metadata
    writer.add_string("general.architecture", "hmer")
    writer.add_string("general.name", "hmer-handwritten-math-ocr")
    writer.add_string("general.license", "MIT")
    writer.add_string("general.source",
                      "whywhs/Pytorch-Handwritten-Mathematical-Expression-Recognition")

    # Encoder hyperparameters
    writer.add_uint32("hmer.encoder.num_init_features", 64)
    writer.add_uint32("hmer.encoder.growth_rate", 32)
    writer.add_array("hmer.encoder.block_config", [6, 12, 24])
    writer.add_uint32("hmer.encoder.input_channels", 2)
    writer.add_uint32("hmer.encoder.output_channels", 1024)

    # Decoder hyperparameters
    writer.add_uint32("hmer.decoder.hidden_size", 256)
    writer.add_uint32("hmer.decoder.output_size", 112)
    writer.add_uint32("hmer.decoder.sos_token", 111)
    writer.add_uint32("hmer.decoder.eol_token", 0)
    writer.add_uint32("hmer.decoder.max_seq_len", 48)

    # Tokenizer
    writer.add_array("tokenizer.tokens", tokens)

    dtype_np = np.float16 if args.fp16 else np.float32
    dtype_gguf = gguf.GGMLQuantizationType.F16 if args.fp16 else gguf.GGMLQuantizationType.F32

    total_params = 0
    tensor_count = 0
    skipped = []

    def add_tensor(name, data, flatten_conv=False):
        nonlocal total_params, tensor_count
        d = data.astype(dtype_np)
        # Flatten 4D conv weights to 2D (out_ch, in_ch*kH*kW) for quantization.
        # The C++ code reads them with the original shape from metadata,
        # but GGUF quantization needs 2D with ncols divisible by 32.
        if flatten_conv and d.ndim == 4:
            d = d.reshape(d.shape[0], -1)
        total_params += d.size
        writer.add_tensor(name, d, raw_dtype=dtype_gguf)
        tensor_count += 1

    # -----------------------------------------------------------------------
    # Encoder: fold BN into Conv where possible
    # -----------------------------------------------------------------------
    # Pattern: each DenseLayer has norm1→relu1→conv1→norm2→relu2→conv2
    # norm1 precedes conv1, norm2 precedes conv2.
    # BUT: BN is applied to the INPUT of conv, not the output.
    # In DenseNet, BN comes BEFORE the conv (pre-activation design).
    # So we CANNOT fold BN into the preceding conv (there is none for norm1).
    #
    # For transition layers: norm→relu→conv→pool
    # Same issue: BN comes before conv.
    #
    # For the stem: conv0_m→norm0→relu0→pool0
    # Here norm0 comes AFTER conv0_m — this one CAN be folded!
    #
    # For norm5 (final): standalone BN after last dense block.
    #
    # Revised strategy:
    #   - Fold norm0 into conv0_m (post-conv BN)
    #   - For DenseLayer norm1/norm2 and transition norms: store as
    #     precomputed scale+offset (avoids storing mean/var separately)
    #   - For norm5: store as precomputed scale+offset
    #   - dec.bn1: store as precomputed scale+offset
    # -----------------------------------------------------------------------

    print("\n--- Encoder tensors ---")

    # Stem: conv0_m + norm0 (fold)
    conv0_w = enc["features.conv0_m.weight"]  # (64, 2, 7, 7)
    bn0_w = enc["features.norm0.weight"]
    bn0_b = enc["features.norm0.bias"]
    bn0_m = enc["features.norm0.running_mean"]
    bn0_v = enc["features.norm0.running_var"]
    fused_w, fused_b = fold_bn_into_conv(conv0_w, bn0_w, bn0_b, bn0_m, bn0_v)
    add_tensor("enc.stem.conv.weight", fused_w, flatten_conv=True)
    add_tensor("enc.stem.conv.bias", fused_b)
    print(f"  enc.stem.conv: {list(fused_w.shape)} (conv0_m + norm0 folded)")

    # Dense blocks
    block_config = [6, 12, 24]
    for bi, n_layers in enumerate(block_config):
        block_name = f"denseblock{bi + 1}"
        for li in range(1, n_layers + 1):
            layer_prefix = f"features.{block_name}.denselayer{li}"
            gguf_prefix = f"enc.block{bi + 1}.layer{li}"

            # norm1 (pre-activation BN before conv1) → precompute scale+offset
            n1_w = enc[f"{layer_prefix}.norm1.weight"]
            n1_b = enc[f"{layer_prefix}.norm1.bias"]
            n1_m = enc[f"{layer_prefix}.norm1.running_mean"]
            n1_v = enc[f"{layer_prefix}.norm1.running_var"]
            s1, o1 = fold_bn_standalone(n1_w, n1_b, n1_m, n1_v)
            add_tensor(f"{gguf_prefix}.bn1.scale", s1)
            add_tensor(f"{gguf_prefix}.bn1.offset", o1)

            # conv1 (1×1 bottleneck)
            add_tensor(f"{gguf_prefix}.conv1.weight",
                       enc[f"{layer_prefix}.conv1.weight"], flatten_conv=True)

            # norm2 (pre-activation BN before conv2) → precompute scale+offset
            n2_w = enc[f"{layer_prefix}.norm2.weight"]
            n2_b = enc[f"{layer_prefix}.norm2.bias"]
            n2_m = enc[f"{layer_prefix}.norm2.running_mean"]
            n2_v = enc[f"{layer_prefix}.norm2.running_var"]
            s2, o2 = fold_bn_standalone(n2_w, n2_b, n2_m, n2_v)
            add_tensor(f"{gguf_prefix}.bn2.scale", s2)
            add_tensor(f"{gguf_prefix}.bn2.offset", o2)

            # conv2 (3×3)
            add_tensor(f"{gguf_prefix}.conv2.weight",
                       enc[f"{layer_prefix}.conv2.weight"], flatten_conv=True)

        print(f"  enc.block{bi + 1}: {n_layers} layers")

        # Transition (except after last block)
        if bi < len(block_config) - 1:
            trans_prefix = f"features.transition{bi + 1}"
            gguf_trans = f"enc.trans{bi + 1}"

            # Transition norm → precompute scale+offset
            tn_w = enc[f"{trans_prefix}.norm.weight"]
            tn_b = enc[f"{trans_prefix}.norm.bias"]
            tn_m = enc[f"{trans_prefix}.norm.running_mean"]
            tn_v = enc[f"{trans_prefix}.norm.running_var"]
            ts, to_ = fold_bn_standalone(tn_w, tn_b, tn_m, tn_v)
            add_tensor(f"{gguf_trans}.bn.scale", ts)
            add_tensor(f"{gguf_trans}.bn.offset", to_)

            # Transition conv (1×1)
            add_tensor(f"{gguf_trans}.conv.weight",
                       enc[f"{trans_prefix}.conv.weight"], flatten_conv=True)

            print(f"  enc.trans{bi + 1}: BN+Conv1x1")

    # Final norm5 → precompute scale+offset
    n5_w = enc["features.norm5.weight"]
    n5_b = enc["features.norm5.bias"]
    n5_m = enc["features.norm5.running_mean"]
    n5_v = enc["features.norm5.running_var"]
    s5, o5 = fold_bn_standalone(n5_w, n5_b, n5_m, n5_v)
    add_tensor("enc.final_bn.scale", s5)
    add_tensor("enc.final_bn.offset", o5)
    print(f"  enc.final_bn: {list(s5.shape)}")

    # Skip: classifier (not used in forward), num_batches_tracked tensors

    # -----------------------------------------------------------------------
    # Decoder tensors
    # -----------------------------------------------------------------------
    print("\n--- Decoder tensors ---")

    # Layers to include (used in forward pass)
    dec_layers = [
        "embedding.weight",
        "gru1.weight_ih", "gru1.weight_hh", "gru1.bias_ih", "gru1.bias_hh",
        "gru.weight_ih", "gru.weight_hh", "gru.bias_ih", "gru.bias_hh",
        "hidden.weight", "hidden.bias",
        "hidden2.weight", "hidden2.bias",
        "emb2.weight", "emb2.bias",
        "ua.weight", "ua.bias",
        "uf.weight", "uf.bias",
        "v.weight", "v.bias",
        "wc.weight", "wc.bias",
        "out.weight", "out.bias",
        "conv_tan.bias",
    ]

    # Unused in forward: emb, conv_et, bn, relu
    dec_skip = {"emb.", "conv_et.", "bn.", "relu."}

    for name in dec_layers:
        if name in dec:
            add_tensor(f"dec.{name}", dec[name])
            print(f"  dec.{name}: {list(dec[name].shape)}")
        else:
            print(f"  WARNING: dec.{name} not found!")

    # Decoder conv weights — flatten for quantization
    add_tensor("dec.conv1.weight", dec["conv1.weight"], flatten_conv=True)
    add_tensor("dec.conv1.bias", dec["conv1.bias"])
    add_tensor("dec.conv_tan.weight", dec["conv_tan.weight"], flatten_conv=True)

    # dec.bn1 — used in attention, cannot fold (output is added to other terms)
    # Store as precomputed scale+offset
    bn1_w = dec["bn1.weight"]
    bn1_b = dec["bn1.bias"]
    bn1_m = dec["bn1.running_mean"]
    bn1_v = dec["bn1.running_var"]
    s_bn1, o_bn1 = fold_bn_standalone(bn1_w, bn1_b, bn1_m, bn1_v)
    add_tensor("dec.bn1.scale", s_bn1)
    add_tensor("dec.bn1.offset", o_bn1)
    print(f"  dec.bn1: scale+offset {list(s_bn1.shape)} (precomputed)")

    # -----------------------------------------------------------------------
    # Write
    # -----------------------------------------------------------------------
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
    print(f"\nQuantize with: crispembed-quantize {args.output} hmer-hw-q8_0.gguf q8_0")


if __name__ == "__main__":
    main()
