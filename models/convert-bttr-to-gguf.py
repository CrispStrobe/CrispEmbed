#!/usr/bin/env python3
"""Convert BTTR (Bidirectionally Trained Transformer) to GGUF.

Loads the PyTorch Lightning checkpoint from Green-Wood/BTTR (MIT license),
folds BatchNorm into conv weights where possible, and packs everything
into a single GGUF file for CrispEmbed inference.

Architecture:
  Encoder: DenseNet (growth=24, 16 layers × 3 blocks, 1ch input)
           + Conv1×1 projection (684→256) + LayerNorm + 2D pos encoding
  Decoder: nn.TransformerDecoder (3 layers, 8 heads, d=256, FFN=1024)
           Post-LN, fused QKV in_proj_weight

Usage:
    python models/convert-bttr-to-gguf.py \
        --checkpoint /mnt/storage/BTTR/pretrained/pretrained-2014.ckpt \
        --dict /mnt/storage/BTTR/bttr/datamodule/dictionary.txt \
        --output /mnt/storage/models/bttr-hw-f32.gguf
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


def fold_bn_into_conv(conv_w, bn_weight, bn_bias, bn_mean, bn_var, eps=1e-5):
    out_ch = conv_w.shape[0]
    scale = bn_weight / np.sqrt(bn_var + eps)
    fused_w = conv_w * scale.reshape(out_ch, 1, 1, 1)
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
    p = argparse.ArgumentParser(description="Convert BTTR to GGUF")
    p.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    p.add_argument("--dict", required=True, help="Path to dictionary.txt")
    p.add_argument("--output", required=True, help="Output GGUF path")
    p.add_argument("--fp16", action="store_true", help="Store in FP16")
    args = p.parse_args()

    # Load checkpoint — filter broken local torch/torchvision installs
    import importlib
    sys.path = [pp for pp in sys.path if '.local' not in pp]
    # Force reimport torch from conda
    for mod in list(sys.modules.keys()):
        if any(x in mod for x in ['torch', 'torchvision', 'pytorch_lightning']):
            del sys.modules[mod]
    import torch

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    hp = ckpt["hyper_parameters"]

    # Strip bttr. prefix
    weights = {}
    for k, v in sd.items():
        name = k.replace("bttr.", "")
        weights[name] = v.numpy() if hasattr(v, 'numpy') else np.array(v)

    print(f"Hyperparameters: {hp}")
    print(f"Tensors: {len(weights)}")

    # Load dictionary
    tokens = load_dictionary(args.dict)
    print(f"Dictionary: {len(tokens)} tokens")

    # Write GGUF
    writer = gguf.GGUFWriter(str(args.output), arch="bttr")

    writer.add_string("general.architecture", "bttr")
    writer.add_string("general.name", "bttr-handwritten-math-ocr")
    writer.add_string("general.license", "MIT")
    writer.add_string("general.source", "Green-Wood/BTTR")

    # Encoder hparams
    writer.add_uint32("bttr.encoder.growth_rate", hp["growth_rate"])
    writer.add_uint32("bttr.encoder.num_layers", hp["num_layers"])
    writer.add_uint32("bttr.encoder.input_channels", 1)

    # Decoder hparams
    writer.add_uint32("bttr.decoder.d_model", hp["d_model"])
    writer.add_uint32("bttr.decoder.nhead", hp["nhead"])
    writer.add_uint32("bttr.decoder.num_layers", hp["num_decoder_layers"])
    writer.add_uint32("bttr.decoder.dim_feedforward", hp["dim_feedforward"])
    writer.add_uint32("bttr.decoder.vocab_size", len(tokens))
    writer.add_uint32("bttr.decoder.max_len", hp["max_len"])
    writer.add_uint32("bttr.decoder.pad_token", 0)
    writer.add_uint32("bttr.decoder.sos_token", 1)
    writer.add_uint32("bttr.decoder.eos_token", 2)

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
    # Encoder
    # -------------------------------------------------------------------
    print("\n--- Encoder ---")

    # Stem: conv1 (48, 1, 7, 7) + norm1 (BN)
    # norm1 is post-conv, so we can fold
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
    # BTTR DenseNet: growth=24, bn_size=4, bottleneck=True
    # Each Bottleneck: bn1→conv1(1×1)→bn2→conv2(3×3)
    # Note: BTTR's BN is POST-conv (bn1 output feeds relu→conv1, then bn2→relu→conv2)
    # Actually looking at the code: conv1(x) → bn1(result) → relu
    # So bn1 IS post-conv1. But conv1 takes raw input x, not bn'd input.
    # Wait, let me re-check the Bottleneck forward:
    #   out = F.relu(self.bn1(self.conv1(x)))   ← conv1 then bn1
    #   out = F.relu(self.bn2(self.conv2(out)))  ← conv2 then bn2
    # Both BNs are POST-conv → can be folded!

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

        print(f"  block{block_idx + 1}: {hp['num_layers']} layers (BN folded into conv)")

        # Transition (except after last block)
        if block_idx < 2:
            src_t = f"encoder.model.trans{block_idx + 1}"
            dst_t = f"enc.trans{block_idx + 1}"

            # BTTR transition: conv1(x) → bn1(result) → relu → avgpool
            # bn1 is post-conv, foldable
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
    add_tensor("enc.feature_proj.weight",
               weights["encoder.feature_proj.0.weight"], flatten_conv=True)
    add_tensor("enc.feature_proj.bias",
               weights["encoder.feature_proj.0.bias"])
    print(f"  feature_proj: {list(weights['encoder.feature_proj.0.weight'].shape)}")

    # Encoder LayerNorm (after feature projection)
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

    # Positional encoding (pre-computed sinusoidal, stored as buffer)
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

        # LayerNorms (post-LN: norm1=after self-attn, norm2=after cross-attn, norm3=after FFN)
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
