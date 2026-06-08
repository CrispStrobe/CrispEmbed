#!/usr/bin/env python3
"""Convert PosFormer to GGUF.

Loads the PyTorch Lightning checkpoint from SJTU-DeepVisionLab/PosFormer
(BSD-2 license), folds BatchNorm into conv weights, and packs into GGUF
for CrispEmbed inference.

Architecture: identical to BTTR (DenseNet + Transformer decoder) plus
the Attention Refinement Module (ARM) for coverage-based attention.
The auxiliary PosDecoder branch is training-only and is discarded.

Usage:
    python models/convert-posformer-to-gguf.py \
        --checkpoint /mnt/storage/PosFormer-fresh/lightning_logs/version_0/checkpoints/best.ckpt \
        --dict /mnt/storage/PosFormer-fresh/Pos_Former/datamodule/dictionary.txt \
        --output /mnt/storage/models/posformer-hw-f32.gguf
"""

import argparse
import sys
from pathlib import Path

import gguf
import numpy as np


def fold_bn_standalone(bn_weight, bn_bias, bn_mean, bn_var, eps=1e-5):
    scale = bn_weight / np.sqrt(bn_var + eps)
    offset = bn_bias - bn_mean * scale
    return scale, offset


def fold_bn_into_conv(conv_w, bn_weight, bn_bias, bn_mean, bn_var,
                      conv_b=None, eps=1e-5):
    """Fold BN into preceding conv. conv_b may be None (bias=False)."""
    out_ch = conv_w.shape[0]
    scale = bn_weight / np.sqrt(bn_var + eps)
    fused_w = conv_w * scale.reshape(out_ch, 1, 1, 1)
    if conv_b is not None:
        fused_b = conv_b * scale + bn_bias - bn_mean * scale
    else:
        fused_b = bn_bias - bn_mean * scale
    return fused_w, fused_b


def load_dictionary(dict_path):
    tokens = ["<pad>", "<sos>", "<eos>"]
    with open(dict_path) as f:
        for line in f:
            w = line.strip()
            if w:
                tokens.append(w)
    return tokens


def main():
    p = argparse.ArgumentParser(description="Convert PosFormer to GGUF")
    p.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    p.add_argument("--dict", required=True, help="Path to dictionary.txt")
    p.add_argument("--output", required=True, help="Output GGUF path")
    p.add_argument("--fp16", action="store_true", help="Store in FP16")
    args = p.parse_args()

    # Load checkpoint — filter broken local torch installs
    sys.path = [pp for pp in sys.path if '.local' not in pp]
    for mod in list(sys.modules.keys()):
        if any(x in mod for x in ['torch', 'torchvision', 'pytorch_lightning']):
            del sys.modules[mod]
    import torch

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    hp = ckpt["hyper_parameters"]

    # Strip model. prefix, skip posdecoder (training-only auxiliary branch)
    weights = {}
    for k, v in sd.items():
        if k.startswith("model.posdecoder."):
            continue
        name = k.replace("model.", "", 1)
        weights[name] = v.numpy() if hasattr(v, 'numpy') else np.array(v)

    print(f"Hyperparameters: {hp}")
    print(f"Tensors (after filtering posdecoder): {len(weights)}")

    # Load dictionary
    tokens = load_dictionary(args.dict)
    print(f"Dictionary: {len(tokens)} tokens")

    # Write GGUF
    writer = gguf.GGUFWriter(str(args.output), arch="posformer")

    writer.add_string("general.name", "posformer-handwritten-math-ocr")
    writer.add_string("general.license", "BSD-2-Clause")
    writer.add_string("general.source", "SJTU-DeepVisionLab/PosFormer")

    # Encoder hparams (identical to BTTR)
    writer.add_uint32("posformer.encoder.growth_rate", hp["growth_rate"])
    writer.add_uint32("posformer.encoder.num_layers", hp["num_layers"])
    writer.add_uint32("posformer.encoder.input_channels", 1)

    # Decoder hparams
    writer.add_uint32("posformer.decoder.d_model", hp["d_model"])
    writer.add_uint32("posformer.decoder.nhead", hp["nhead"])
    writer.add_uint32("posformer.decoder.num_layers", hp["num_decoder_layers"])
    writer.add_uint32("posformer.decoder.dim_feedforward", hp["dim_feedforward"])
    writer.add_uint32("posformer.decoder.vocab_size", len(tokens))
    writer.add_uint32("posformer.decoder.max_len", hp["max_len"])
    writer.add_uint32("posformer.decoder.pad_token", 0)
    writer.add_uint32("posformer.decoder.sos_token", 1)
    writer.add_uint32("posformer.decoder.eos_token", 2)

    # ARM hparams
    writer.add_uint32("posformer.arm.dc", hp["dc"])
    writer.add_bool("posformer.arm.cross_coverage", hp["cross_coverage"])
    writer.add_bool("posformer.arm.self_coverage", hp["self_coverage"])

    # Tokenizer
    writer.add_array("tokenizer.tokens", tokens)

    dtype_np = np.float16 if args.fp16 else np.float32
    dtype_gguf = gguf.GGMLQuantizationType.F16 if args.fp16 else gguf.GGMLQuantizationType.F32

    total_params = 0
    tensor_count = 0

    def add_tensor(name, data, flatten_conv=False):
        nonlocal total_params, tensor_count
        d = data.astype(dtype_np)
        if flatten_conv and d.ndim == 4:
            d = d.reshape(d.shape[0], -1)
        total_params += d.size
        writer.add_tensor(name, d, raw_dtype=dtype_gguf)
        tensor_count += 1

    # -------------------------------------------------------------------
    # Encoder (identical to BTTR)
    # -------------------------------------------------------------------
    print("\n--- Encoder ---")

    # Stem: conv1 (48, 1, 7, 7) + norm1 (BN) — fold
    conv1_w = weights["encoder.model.conv1.weight"]
    n1_w = weights["encoder.model.norm1.weight"]
    n1_b = weights["encoder.model.norm1.bias"]
    n1_m = weights["encoder.model.norm1.running_mean"]
    n1_v = weights["encoder.model.norm1.running_var"]
    fused_w, fused_b = fold_bn_into_conv(conv1_w, n1_w, n1_b, n1_m, n1_v)
    add_tensor("enc.stem.conv.weight", fused_w, flatten_conv=True)
    add_tensor("enc.stem.conv.bias", fused_b)
    print(f"  stem: {list(fused_w.shape)} (conv1+norm1 folded)")

    # Dense blocks (3 blocks × 16 layers each)
    for block_idx in range(3):
        block_name = f"dense{block_idx + 1}"
        for layer_idx in range(hp["num_layers"]):
            src_prefix = f"encoder.model.{block_name}.{layer_idx}"
            dst_prefix = f"enc.block{block_idx + 1}.layer{layer_idx}"

            # conv1 + bn1 (fold)
            c1_w = weights[f"{src_prefix}.conv1.weight"]
            b1_w = weights[f"{src_prefix}.bn1.weight"]
            b1_b = weights[f"{src_prefix}.bn1.bias"]
            b1_m = weights[f"{src_prefix}.bn1.running_mean"]
            b1_v = weights[f"{src_prefix}.bn1.running_var"]
            f1_w, f1_b = fold_bn_into_conv(c1_w, b1_w, b1_b, b1_m, b1_v)
            add_tensor(f"{dst_prefix}.conv1.weight", f1_w, flatten_conv=True)
            add_tensor(f"{dst_prefix}.conv1.bias", f1_b)

            # conv2 + bn2 (fold)
            c2_w = weights[f"{src_prefix}.conv2.weight"]
            b2_w = weights[f"{src_prefix}.bn2.weight"]
            b2_b = weights[f"{src_prefix}.bn2.bias"]
            b2_m = weights[f"{src_prefix}.bn2.running_mean"]
            b2_v = weights[f"{src_prefix}.bn2.running_var"]
            f2_w, f2_b = fold_bn_into_conv(c2_w, b2_w, b2_b, b2_m, b2_v)
            add_tensor(f"{dst_prefix}.conv2.weight", f2_w, flatten_conv=True)
            add_tensor(f"{dst_prefix}.conv2.bias", f2_b)

        print(f"  block{block_idx + 1}: {hp['num_layers']} layers (BN folded)")

        # Transition (except after last block)
        if block_idx < 2:
            src_t = f"encoder.model.trans{block_idx + 1}"
            dst_t = f"enc.trans{block_idx + 1}"

            tc_w = weights[f"{src_t}.conv1.weight"]
            tb_w = weights[f"{src_t}.bn1.weight"]
            tb_b = weights[f"{src_t}.bn1.bias"]
            tb_m = weights[f"{src_t}.bn1.running_mean"]
            tb_v = weights[f"{src_t}.bn1.running_var"]
            ft_w, ft_b = fold_bn_into_conv(tc_w, tb_w, tb_b, tb_m, tb_v)
            add_tensor(f"{dst_t}.conv.weight", ft_w, flatten_conv=True)
            add_tensor(f"{dst_t}.conv.bias", ft_b)
            print(f"  trans{block_idx + 1}: BN folded")

    # Post-norm (final BN after dense3)
    pn_w = weights["encoder.model.post_norm.weight"]
    pn_b = weights["encoder.model.post_norm.bias"]
    pn_m = weights["encoder.model.post_norm.running_mean"]
    pn_v = weights["encoder.model.post_norm.running_var"]
    pn_s, pn_o = fold_bn_standalone(pn_w, pn_b, pn_m, pn_v)
    add_tensor("enc.post_norm.scale", pn_s)
    add_tensor("enc.post_norm.offset", pn_o)
    print(f"  post_norm: {list(pn_s.shape)}")

    # Feature projection: Conv1×1 (684→256) + bias
    # PosFormer uses feature_proj.weight directly (no .0. index like BTTR)
    fp_w_key = "encoder.feature_proj.0.weight" if "encoder.feature_proj.0.weight" in weights else "encoder.feature_proj.weight"
    fp_b_key = "encoder.feature_proj.0.bias" if "encoder.feature_proj.0.bias" in weights else "encoder.feature_proj.bias"
    add_tensor("enc.feature_proj.weight", weights[fp_w_key], flatten_conv=True)
    add_tensor("enc.feature_proj.bias", weights[fp_b_key])
    print(f"  feature_proj: {list(weights[fp_w_key].shape)}")

    # Encoder LayerNorm
    add_tensor("enc.norm.weight", weights["encoder.norm.weight"])
    add_tensor("enc.norm.bias", weights["encoder.norm.bias"])

    # -------------------------------------------------------------------
    # Decoder
    # -------------------------------------------------------------------
    print("\n--- Decoder ---")

    # Word embedding + LayerNorm
    add_tensor("dec.word_embed.weight", weights["decoder.word_embed.0.weight"])
    add_tensor("dec.word_embed_ln.weight", weights["decoder.word_embed.1.weight"])
    add_tensor("dec.word_embed_ln.bias", weights["decoder.word_embed.1.bias"])
    print(f"  word_embed: {list(weights['decoder.word_embed.0.weight'].shape)}")

    # Positional encoding
    add_tensor("dec.pos_enc", weights["decoder.pos_enc.pe"])
    print(f"  pos_enc: {list(weights['decoder.pos_enc.pe'].shape)}")

    # Transformer decoder layers
    n_dec = hp["num_decoder_layers"]
    for li in range(n_dec):
        src = f"decoder.model.layers.{li}"
        dst = f"dec.layers.{li}"

        # Self-attention (fused QKV)
        add_tensor(f"{dst}.self_attn.in_proj_weight",
                   weights[f"{src}.self_attn.in_proj_weight"])
        add_tensor(f"{dst}.self_attn.in_proj_bias",
                   weights[f"{src}.self_attn.in_proj_bias"])
        add_tensor(f"{dst}.self_attn.out_proj.weight",
                   weights[f"{src}.self_attn.out_proj.weight"])
        add_tensor(f"{dst}.self_attn.out_proj.bias",
                   weights[f"{src}.self_attn.out_proj.bias"])

        # Cross-attention (fused QKV)
        add_tensor(f"{dst}.cross_attn.in_proj_weight",
                   weights[f"{src}.multihead_attn.in_proj_weight"])
        add_tensor(f"{dst}.cross_attn.in_proj_bias",
                   weights[f"{src}.multihead_attn.in_proj_bias"])
        add_tensor(f"{dst}.cross_attn.out_proj.weight",
                   weights[f"{src}.multihead_attn.out_proj.weight"])
        add_tensor(f"{dst}.cross_attn.out_proj.bias",
                   weights[f"{src}.multihead_attn.out_proj.bias"])

        # FFN
        add_tensor(f"{dst}.ffn.up.weight", weights[f"{src}.linear1.weight"])
        add_tensor(f"{dst}.ffn.up.bias", weights[f"{src}.linear1.bias"])
        add_tensor(f"{dst}.ffn.down.weight", weights[f"{src}.linear2.weight"])
        add_tensor(f"{dst}.ffn.down.bias", weights[f"{src}.linear2.bias"])

        # LayerNorms (post-LN)
        add_tensor(f"{dst}.norm1.weight", weights[f"{src}.norm1.weight"])
        add_tensor(f"{dst}.norm1.bias", weights[f"{src}.norm1.bias"])
        add_tensor(f"{dst}.norm2.weight", weights[f"{src}.norm2.weight"])
        add_tensor(f"{dst}.norm2.bias", weights[f"{src}.norm2.bias"])
        add_tensor(f"{dst}.norm3.weight", weights[f"{src}.norm3.weight"])
        add_tensor(f"{dst}.norm3.bias", weights[f"{src}.norm3.bias"])

        print(f"  layer {li}: self_attn + cross_attn + FFN + 3×LN")

    # Output projection
    add_tensor("dec.proj.weight", weights["decoder.proj.weight"])
    add_tensor("dec.proj.bias", weights["decoder.proj.bias"])
    print(f"  proj: {list(weights['decoder.proj.weight'].shape)}")

    # -------------------------------------------------------------------
    # ARM (Attention Refinement Module)
    # -------------------------------------------------------------------
    print("\n--- ARM ---")

    # ARM conv: Conv2d(2*nhead=16, dc=32, 5, 5) with bias
    add_tensor("arm.conv.weight",
               weights["decoder.model.arm.conv.weight"], flatten_conv=True)
    add_tensor("arm.conv.bias",
               weights["decoder.model.arm.conv.bias"])
    print(f"  conv: {list(weights['decoder.model.arm.conv.weight'].shape)}")

    # ARM proj: Conv2d(dc=32, nhead=8, 1, 1) NO bias + MaskBatchNorm
    # Fold BN into proj (proj has no bias, BN synthesizes one)
    proj_w = weights["decoder.model.arm.proj.weight"]
    bn_w = weights["decoder.model.arm.post_norm.bn.weight"]
    bn_b = weights["decoder.model.arm.post_norm.bn.bias"]
    bn_m = weights["decoder.model.arm.post_norm.bn.running_mean"]
    bn_v = weights["decoder.model.arm.post_norm.bn.running_var"]
    fused_proj_w, fused_proj_b = fold_bn_into_conv(
        proj_w, bn_w, bn_b, bn_m, bn_v, conv_b=None)
    add_tensor("arm.proj.weight", fused_proj_w, flatten_conv=True)
    add_tensor("arm.proj.bias", fused_proj_b)
    print(f"  proj+BN: {list(proj_w.shape)} (BN folded, bias synthesized)")

    # -------------------------------------------------------------------
    # Write
    # -------------------------------------------------------------------
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


if __name__ == "__main__":
    main()
