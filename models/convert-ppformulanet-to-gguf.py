#!/usr/bin/env python3
"""Convert PPFormulaNet-S (HGNetv2 + MBart decoder) to GGUF.

Loads the PyTorch state dict (originally converted from PaddleOCR's
PPFormulaNet-S via Texo's mapping), folds all BatchNorm into preceding
Conv2d, and packs everything into a single GGUF file for CrispEmbed.

Architecture:
  Encoder: HGNetv2 (4 stages of HG_Blocks, Conv-BN-ReLU)
           384x384 RGB input -> (B, 144, 2048) sequence output
  Projection: Linear(2048, 384) enc_to_dec_proj
  Decoder: MBart (2 layers, 16 heads, d=384, FFN=1536)
           Post-LN, scale_embedding=True

Usage:
    python models/convert-ppformulanet-to-gguf.py \
        --checkpoint /mnt/storage/models/ppformulanet-s/checkpoints/formulanet.pt \
        --tokenizer /mnt/storage/models/ppformulanet-s/unimernet_tokenizer/tokenizer.json \
        --output /mnt/storage/models/ppformulanet-s-f32.gguf

    python models/convert-ppformulanet-to-gguf.py \
        --checkpoint /mnt/storage/models/ppformulanet-s/checkpoints/formulanet.pt \
        --tokenizer /mnt/storage/models/ppformulanet-s/unimernet_tokenizer/tokenizer.json \
        --output /mnt/storage/models/ppformulanet-s-f16.gguf \
        --fp16
"""

import argparse
import json
import sys
from pathlib import Path

import gguf
import numpy as np


# ---------------------------------------------------------------------------
# BatchNorm folding
# ---------------------------------------------------------------------------

def fold_bn_into_conv(conv_w, bn_weight, bn_bias, bn_mean, bn_var, eps=1e-5):
    """Fold BN into conv: fused_w = conv_w * scale, fused_b = bn_bias - bn_mean * scale."""
    out_ch = conv_w.shape[0]
    scale = bn_weight / np.sqrt(bn_var + eps)
    # Reshape scale for broadcasting: (out_ch, 1, 1, 1) for 4D, (out_ch, 1) for 2D
    if conv_w.ndim == 4:
        fused_w = conv_w * scale.reshape(out_ch, 1, 1, 1)
    elif conv_w.ndim == 2:
        fused_w = conv_w * scale.reshape(out_ch, 1)
    else:
        fused_w = conv_w * scale.reshape(out_ch, 1)
    fused_b = bn_bias - bn_mean * scale
    return fused_w, fused_b


def get_bn(weights, prefix):
    """Extract BN parameters from state dict."""
    return (
        weights[f"{prefix}.weight"],
        weights[f"{prefix}.bias"],
        weights[f"{prefix}.running_mean"],
        weights[f"{prefix}.running_var"],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Convert PPFormulaNet-S to GGUF")
    p.add_argument("--checkpoint", required=True, help="Path to formulanet.pt")
    p.add_argument("--tokenizer", required=True, help="Path to tokenizer.json")
    p.add_argument("--output", required=True, help="Output GGUF path")
    p.add_argument("--fp16", action="store_true", help="Store in FP16")
    args = p.parse_args()

    import torch
    print(f"Loading checkpoint: {args.checkpoint}")
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=True)

    # Convert all tensors to numpy
    weights = {}
    for k, v in sd.items():
        if hasattr(v, 'numpy'):
            weights[k] = v.float().numpy()
        else:
            weights[k] = np.array(v, dtype=np.float32)

    print(f"Loaded {len(weights)} tensors")

    # Load tokenizer
    with open(args.tokenizer) as f:
        tok_data = json.load(f)
    vocab = tok_data.get("model", {}).get("vocab", {})
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    token_list = [t[0] for t in sorted_vocab]
    print(f"Tokenizer: {len(token_list)} tokens")

    # Detect architecture params from weights
    dec_embed = weights["decoder.model.decoder.embed_tokens.weight"]
    vocab_size = dec_embed.shape[0]
    d_model = dec_embed.shape[1]
    # Count decoder layers
    n_dec_layers = 0
    while f"decoder.model.decoder.layers.{n_dec_layers}.self_attn.q_proj.weight" in weights:
        n_dec_layers += 1
    dec_ffn_dim = weights["decoder.model.decoder.layers.0.fc1.weight"].shape[0]
    dec_heads = 16  # From config
    max_pos = weights["decoder.model.decoder.embed_positions.weight"].shape[0]
    enc_hidden = weights["enc_to_dec_proj.weight"].shape[1]

    print(f"Encoder: hidden={enc_hidden}")
    print(f"Decoder: {n_dec_layers}L, {dec_heads}H, d={d_model}, FFN={dec_ffn_dim}, vocab={vocab_size}, max_pos={max_pos}")

    # Stage config from the model structure
    # stage_config[i] = (in_ch, mid_ch, out_ch, n_blocks, n_layers, kernel, light_block, residual)
    stage_config = [
        (48, 48, 128, 1, 6, 3, False, False),    # stage 0
        (128, 96, 512, 1, 6, 3, False, False),    # stage 1 (Texo says light=True but weights have single conv)
        (512, 192, 1024, 3, 6, 5, True, True),    # stage 2
        (1024, 384, 2048, 1, 6, 5, True, True),   # stage 3
    ]
    # Detect light vs regular from weight keys
    for si in range(4):
        key = f"encoder.stages.{si}.blocks.0.layers.0.conv1.conv.weight"
        if key in weights:
            stage_config[si] = (*stage_config[si][:6], True, stage_config[si][7])
        else:
            stage_config[si] = (*stage_config[si][:6], False, stage_config[si][7])

    # Write GGUF
    writer = gguf.GGUFWriter(str(args.output), arch="ppformulanet")

    writer.add_string("general.architecture", "ppformulanet")
    writer.add_string("general.name", "ppformulanet-s-ocr")
    writer.add_string("general.license", "Apache-2.0")
    writer.add_string("general.source", "PaddlePaddle/PaddleOCR PP-FormulaNet-S")

    # Encoder hparams
    writer.add_uint32("ppfn.encoder.image_size", 384)
    writer.add_uint32("ppfn.encoder.hidden_size", enc_hidden)
    writer.add_uint32("ppfn.encoder.n_stages", 4)
    for si, (in_ch, mid_ch, out_ch, n_blk, n_lay, ks, light, resid) in enumerate(stage_config):
        pfx = f"ppfn.encoder.stage{si}"
        writer.add_uint32(f"{pfx}.in_channels", in_ch)
        writer.add_uint32(f"{pfx}.mid_channels", mid_ch)
        writer.add_uint32(f"{pfx}.out_channels", out_ch)
        writer.add_uint32(f"{pfx}.n_blocks", n_blk)
        writer.add_uint32(f"{pfx}.n_layers", n_lay)
        writer.add_uint32(f"{pfx}.kernel_size", ks)
        writer.add_uint32(f"{pfx}.light_block", 1 if light else 0)

    # Decoder hparams
    writer.add_uint32("ppfn.decoder.d_model", d_model)
    writer.add_uint32("ppfn.decoder.decoder_layers", n_dec_layers)
    writer.add_uint32("ppfn.decoder.decoder_attention_heads", dec_heads)
    writer.add_uint32("ppfn.decoder.decoder_ffn_dim", dec_ffn_dim)
    writer.add_uint32("ppfn.decoder.vocab_size", vocab_size)
    writer.add_uint32("ppfn.decoder.max_position_embeddings", max_pos)
    writer.add_uint32("ppfn.decoder.bos_token_id", 0)
    writer.add_uint32("ppfn.decoder.eos_token_id", 2)
    writer.add_uint32("ppfn.decoder.pad_token_id", 1)
    writer.add_uint32("ppfn.decoder.decoder_start_token_id", 0)

    # Tokenizer
    writer.add_array("tokenizer.tokens", token_list)

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

    def fold_and_add(name, conv_key, bn_prefix, flatten_conv=True):
        """Fold BN into conv, emit weight + bias."""
        conv_w = weights[conv_key]
        bn_w, bn_b, bn_m, bn_v = get_bn(weights, bn_prefix)
        fused_w, fused_b = fold_bn_into_conv(conv_w, bn_w, bn_b, bn_m, bn_v)
        add_tensor(f"{name}.weight", fused_w, flatten_conv=flatten_conv)
        add_tensor(f"{name}.bias", fused_b)

    # -------------------------------------------------------------------
    # Encoder: StemBlock
    # -------------------------------------------------------------------
    print("\n--- Encoder: StemBlock ---")
    for stem_name in ["stem1", "stem2a", "stem2b", "stem3", "stem4"]:
        src = f"encoder.stem.{stem_name}"
        dst = f"enc.stem.{stem_name}"
        fold_and_add(dst, f"{src}.conv.weight", f"{src}.bn")
        print(f"  {stem_name}: {list(weights[f'{src}.conv.weight'].shape)} (BN folded)")

    # -------------------------------------------------------------------
    # Encoder: Stages
    # -------------------------------------------------------------------
    for si, (in_ch, mid_ch, out_ch, n_blk, n_lay, ks, light, resid) in enumerate(stage_config):
        print(f"\n--- Encoder: Stage {si} ---")

        # Downsample (stages 1-3)
        ds_key = f"encoder.stages.{si}.downsample.conv.weight"
        if ds_key in weights:
            fold_and_add(
                f"enc.stage{si}.downsample",
                ds_key,
                f"encoder.stages.{si}.downsample.bn",
            )
            print(f"  downsample: {list(weights[ds_key].shape)} (depthwise, BN folded)")

        # HG_Blocks
        for bi in range(n_blk):
            blk_src = f"encoder.stages.{si}.blocks.{bi}"
            blk_dst = f"enc.stage{si}.block{bi}"

            # Conv layers (6 per block)
            for li in range(n_lay):
                if light:
                    # LightConvBNAct: conv1 (1x1 pointwise) + conv2 (depthwise)
                    fold_and_add(
                        f"{blk_dst}.layer{li}.conv1",
                        f"{blk_src}.layers.{li}.conv1.conv.weight",
                        f"{blk_src}.layers.{li}.conv1.bn",
                    )
                    fold_and_add(
                        f"{blk_dst}.layer{li}.conv2",
                        f"{blk_src}.layers.{li}.conv2.conv.weight",
                        f"{blk_src}.layers.{li}.conv2.bn",
                    )
                else:
                    # ConvBNAct: single conv
                    fold_and_add(
                        f"{blk_dst}.layer{li}",
                        f"{blk_src}.layers.{li}.conv.weight",
                        f"{blk_src}.layers.{li}.bn",
                    )

            # Aggregation: squeeze (1x1) + excitation (1x1)
            fold_and_add(
                f"{blk_dst}.agg_squeeze",
                f"{blk_src}.aggregation_squeeze_conv.conv.weight",
                f"{blk_src}.aggregation_squeeze_conv.bn",
            )
            fold_and_add(
                f"{blk_dst}.agg_excite",
                f"{blk_src}.aggregation_excitation_conv.conv.weight",
                f"{blk_src}.aggregation_excitation_conv.bn",
            )

            print(f"  block{bi}: {n_lay} layers ({'light' if light else 'regular'}), BN folded")

    # -------------------------------------------------------------------
    # Encoder-to-decoder projection
    # -------------------------------------------------------------------
    print("\n--- enc_to_dec_proj ---")
    add_tensor("enc.proj.weight", weights["enc_to_dec_proj.weight"])
    add_tensor("enc.proj.bias", weights["enc_to_dec_proj.bias"])
    print(f"  proj: {list(weights['enc_to_dec_proj.weight'].shape)}")

    # -------------------------------------------------------------------
    # Decoder
    # -------------------------------------------------------------------
    print("\n--- Decoder ---")

    # Token and position embeddings
    add_tensor("dec.embed_tokens.weight", weights["decoder.model.decoder.embed_tokens.weight"])
    add_tensor("dec.embed_positions.weight", weights["decoder.model.decoder.embed_positions.weight"])
    print(f"  embed_tokens: {list(weights['decoder.model.decoder.embed_tokens.weight'].shape)}")
    print(f"  embed_positions: {list(weights['decoder.model.decoder.embed_positions.weight'].shape)}")

    # Embedding LayerNorm
    add_tensor("dec.embed_ln.weight", weights["decoder.model.decoder.layernorm_embedding.weight"])
    add_tensor("dec.embed_ln.bias", weights["decoder.model.decoder.layernorm_embedding.bias"])

    # Decoder layers
    for li in range(n_dec_layers):
        src = f"decoder.model.decoder.layers.{li}"
        dst = f"dec.layers.{li}"

        # Self-attention (separate Q/K/V projections)
        add_tensor(f"{dst}.self_attn.q.weight", weights[f"{src}.self_attn.q_proj.weight"])
        add_tensor(f"{dst}.self_attn.q.bias", weights[f"{src}.self_attn.q_proj.bias"])
        add_tensor(f"{dst}.self_attn.k.weight", weights[f"{src}.self_attn.k_proj.weight"])
        add_tensor(f"{dst}.self_attn.k.bias", weights[f"{src}.self_attn.k_proj.bias"])
        add_tensor(f"{dst}.self_attn.v.weight", weights[f"{src}.self_attn.v_proj.weight"])
        add_tensor(f"{dst}.self_attn.v.bias", weights[f"{src}.self_attn.v_proj.bias"])
        add_tensor(f"{dst}.self_attn.out.weight", weights[f"{src}.self_attn.out_proj.weight"])
        add_tensor(f"{dst}.self_attn.out.bias", weights[f"{src}.self_attn.out_proj.bias"])
        add_tensor(f"{dst}.self_attn_ln.weight", weights[f"{src}.self_attn_layer_norm.weight"])
        add_tensor(f"{dst}.self_attn_ln.bias", weights[f"{src}.self_attn_layer_norm.bias"])

        # Cross-attention (encoder_attn)
        add_tensor(f"{dst}.cross_attn.q.weight", weights[f"{src}.encoder_attn.q_proj.weight"])
        add_tensor(f"{dst}.cross_attn.q.bias", weights[f"{src}.encoder_attn.q_proj.bias"])
        add_tensor(f"{dst}.cross_attn.k.weight", weights[f"{src}.encoder_attn.k_proj.weight"])
        add_tensor(f"{dst}.cross_attn.k.bias", weights[f"{src}.encoder_attn.k_proj.bias"])
        add_tensor(f"{dst}.cross_attn.v.weight", weights[f"{src}.encoder_attn.v_proj.weight"])
        add_tensor(f"{dst}.cross_attn.v.bias", weights[f"{src}.encoder_attn.v_proj.bias"])
        add_tensor(f"{dst}.cross_attn.out.weight", weights[f"{src}.encoder_attn.out_proj.weight"])
        add_tensor(f"{dst}.cross_attn.out.bias", weights[f"{src}.encoder_attn.out_proj.bias"])
        add_tensor(f"{dst}.cross_attn_ln.weight", weights[f"{src}.encoder_attn_layer_norm.weight"])
        add_tensor(f"{dst}.cross_attn_ln.bias", weights[f"{src}.encoder_attn_layer_norm.bias"])

        # FFN
        add_tensor(f"{dst}.ffn.up.weight", weights[f"{src}.fc1.weight"])
        add_tensor(f"{dst}.ffn.up.bias", weights[f"{src}.fc1.bias"])
        add_tensor(f"{dst}.ffn.down.weight", weights[f"{src}.fc2.weight"])
        add_tensor(f"{dst}.ffn.down.bias", weights[f"{src}.fc2.bias"])
        add_tensor(f"{dst}.ffn_ln.weight", weights[f"{src}.final_layer_norm.weight"])
        add_tensor(f"{dst}.ffn_ln.bias", weights[f"{src}.final_layer_norm.bias"])

        print(f"  layer {li}: self_attn + cross_attn + FFN")

    # Final LayerNorm
    add_tensor("dec.final_ln.weight", weights["decoder.model.decoder.layer_norm.weight"])
    add_tensor("dec.final_ln.bias", weights["decoder.model.decoder.layer_norm.bias"])

    # LM head
    add_tensor("dec.lm_head.weight", weights["decoder.lm_head.weight"])
    # LM head may or may not have bias
    lm_bias_key = "decoder.lm_head.bias"
    if lm_bias_key in weights:
        add_tensor("dec.lm_head.bias", weights[lm_bias_key])
    print(f"  lm_head: {list(weights['decoder.lm_head.weight'].shape)}")

    # -------------------------------------------------------------------
    # Finalize
    # -------------------------------------------------------------------
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = Path(args.output).stat().st_size / 1024 / 1024
    print(f"\n{'='*60}")
    print(f"Output: {args.output}")
    print(f"Size: {size_mb:.1f} MB")
    print(f"Tensors: {tensor_count}")
    print(f"Parameters: {total_params:,}")
    print(f"Dtype: {'FP16' if args.fp16 else 'FP32'}")


if __name__ == "__main__":
    main()
