#!/usr/bin/env python3
"""Convert PP-FormulaNet-L (SAM-ViT + MBart decoder) to GGUF.

Loads the HuggingFace safetensors checkpoint and packs into a single GGUF file.

Architecture:
  Encoder: SAM-style ViT (12 layers, 768d, windowed + global attention)
           768x768 RGB input -> patch embed (16x16) -> 48x48 patches
           Neck: Conv1x1 + LN2d + Conv3x3 + LN2d (768 -> 256)
           Projector: Conv3x3(s2) + Conv3x3(s2) + Linear + Linear (256 -> 512)
           Output: (B, 144, 512) sequence for decoder
  Decoder: MBart PRE-LN (8 layers, 16 heads, d=512, FFN=2048)

Usage:
    python models/convert-ppformulanet-l-to-gguf.py \
        --model-dir /mnt/volume1/models/PP-FormulaNet-L \
        --output /mnt/volume1/models/ppformulanet-l-f32.gguf

    python models/convert-ppformulanet-l-to-gguf.py \
        --model-dir /mnt/volume1/models/PP-FormulaNet-L \
        --output /mnt/volume1/models/ppformulanet-l-f16.gguf \
        --fp16
"""

import argparse
import json
import sys
from pathlib import Path

import gguf
import numpy as np


def main():
    p = argparse.ArgumentParser(description="Convert PP-FormulaNet-L to GGUF")
    p.add_argument("--model-dir", required=True, help="Path to HF model directory")
    p.add_argument("--output", required=True, help="Output GGUF path")
    p.add_argument("--fp16", action="store_true", help="Store in FP16")
    p.add_argument("--q8_0", action="store_true", help="Quantize large tensors to Q8_0, keep critical in F16")
    args = p.parse_args()

    if args.q8_0 and args.fp16:
        print("Error: --q8_0 and --fp16 are mutually exclusive")
        sys.exit(1)

    model_dir = Path(args.model_dir)

    # Load safetensors
    from safetensors import safe_open
    print(f"Loading safetensors: {model_dir / 'model.safetensors'}")
    sf = safe_open(str(model_dir / "model.safetensors"), framework="numpy")

    weights = {}
    for k in sf.keys():
        weights[k] = sf.get_tensor(k)
    print(f"Loaded {len(weights)} tensors")

    # Load config
    with open(model_dir / "config.json") as f:
        config = json.load(f)
    vc = config["vision_config"]
    tc = config["text_config"]

    # Load tokenizer
    with open(model_dir / "tokenizer.json") as f:
        tok_data = json.load(f)
    vocab = tok_data.get("model", {}).get("vocab", {})
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    token_list = [t[0] for t in sorted_vocab]
    print(f"Tokenizer: {len(token_list)} tokens")

    # Architecture params from config
    image_size = vc["image_size"]          # 768
    patch_size = vc["patch_size"]          # 16
    hidden_size = vc["hidden_size"]        # 768
    n_enc_layers = vc["num_hidden_layers"] # 12
    n_enc_heads = vc["num_attention_heads"]# 12
    mlp_dim = vc["mlp_dim"]               # 3072
    window_size = vc["window_size"]        # 14
    global_attn_indexes = vc["global_attn_indexes"]  # [2, 5, 8, 11]
    output_channels = vc["output_channels"]           # 256
    post_conv_mid = vc["post_conv_mid_channels"]      # 512
    post_conv_out = vc["post_conv_out_channels"]      # 1024
    dec_hidden = vc["decoder_hidden_size"]             # 512

    d_model = tc["d_model"]               # 512
    n_dec_layers = tc["decoder_layers"]    # 8
    dec_heads = tc["decoder_attention_heads"]  # 16
    dec_ffn_dim = tc["decoder_ffn_dim"]    # 2048
    vocab_size = tc["vocab_size"]          # 50000
    max_pos = tc["max_position_embeddings"]# 1024

    n_patches = image_size // patch_size   # 48
    head_dim = hidden_size // n_enc_heads  # 64

    print(f"\nEncoder: {n_enc_layers}L, {n_enc_heads}H, d={hidden_size}, MLP={mlp_dim}")
    print(f"  image={image_size}x{image_size}, patch={patch_size}, patches={n_patches}x{n_patches}")
    print(f"  window_size={window_size}, global_layers={global_attn_indexes}")
    print(f"  neck: {hidden_size}->{output_channels}, projector: {output_channels}->{dec_hidden}")
    print(f"Decoder: {n_dec_layers}L, {dec_heads}H, d={d_model}, FFN={dec_ffn_dim}")
    print(f"  vocab={vocab_size}, max_pos={max_pos}")

    # Write GGUF
    writer = gguf.GGUFWriter(str(args.output), arch="ppformulanet_l")

    writer.add_string("general.architecture", "ppformulanet_l")
    writer.add_string("general.name", "ppformulanet-l-ocr")
    writer.add_string("general.license", "Apache-2.0")
    writer.add_string("general.source", "PaddlePaddle/PP-FormulaNet-L_safetensors")

    # Encoder hparams
    writer.add_uint32("ppfnl.encoder.image_size", image_size)
    writer.add_uint32("ppfnl.encoder.patch_size", patch_size)
    writer.add_uint32("ppfnl.encoder.hidden_size", hidden_size)
    writer.add_uint32("ppfnl.encoder.n_layers", n_enc_layers)
    writer.add_uint32("ppfnl.encoder.n_heads", n_enc_heads)
    writer.add_uint32("ppfnl.encoder.mlp_dim", mlp_dim)
    writer.add_uint32("ppfnl.encoder.window_size", window_size)
    writer.add_array("ppfnl.encoder.global_attn_indexes",
                     [int(x) for x in global_attn_indexes])
    writer.add_uint32("ppfnl.encoder.output_channels", output_channels)
    writer.add_uint32("ppfnl.encoder.post_conv_mid_channels", post_conv_mid)
    writer.add_uint32("ppfnl.encoder.post_conv_out_channels", post_conv_out)

    # Decoder hparams
    writer.add_uint32("ppfnl.decoder.d_model", d_model)
    writer.add_uint32("ppfnl.decoder.decoder_layers", n_dec_layers)
    writer.add_uint32("ppfnl.decoder.decoder_attention_heads", dec_heads)
    writer.add_uint32("ppfnl.decoder.decoder_ffn_dim", dec_ffn_dim)
    writer.add_uint32("ppfnl.decoder.vocab_size", vocab_size)
    writer.add_uint32("ppfnl.decoder.max_position_embeddings", max_pos)
    writer.add_uint32("ppfnl.decoder.bos_token_id", tc["bos_token_id"])
    writer.add_uint32("ppfnl.decoder.eos_token_id", tc["eos_token_id"])
    writer.add_uint32("ppfnl.decoder.pad_token_id", tc["pad_token_id"])
    writer.add_uint32("ppfnl.decoder.decoder_start_token_id", tc["bos_token_id"])

    # Tokenizer
    writer.add_array("tokenizer.tokens", token_list)

    if args.fp16:
        default_np = np.float16
        default_gguf = gguf.GGMLQuantizationType.F16
    elif args.q8_0:
        default_np = np.float16   # critical tensors in F16
        default_gguf = gguf.GGMLQuantizationType.F16
    else:
        default_np = np.float32
        default_gguf = gguf.GGMLQuantizationType.F32

    # Tensors to keep in F16 even under Q8_0 (small or critical):
    # - Embeddings, positional encodings, patch embed
    # - LayerNorm weights/biases
    # - Relative position bias tables (tiny, critical for attention geometry)
    # - LM head (directly determines output tokens)
    # - Neck + projector weights (bottleneck tensors)
    _f16_prefixes = {
        "enc.patch_embed.", "enc.pos_embed",
        "enc.neck.", "enc.proj.",
        "dec.embed_tokens.", "dec.embed_positions.",
        "dec.embed_ln.", "dec.final_ln.", "dec.lm_head.",
    }
    _f16_suffixes = {
        ".ln1.weight", ".ln1.bias", ".ln2.weight", ".ln2.bias",
        ".rel_pos_h", ".rel_pos_w",
        "self_attn_ln.weight", "self_attn_ln.bias",
        "cross_attn_ln.weight", "cross_attn_ln.bias",
        "ffn_ln.weight", "ffn_ln.bias",
    }

    def is_critical(name):
        """Tensors that should stay in F16 under quantization."""
        for pfx in _f16_prefixes:
            if name.startswith(pfx):
                return True
        for sfx in _f16_suffixes:
            if name.endswith(sfx):
                return True
        return False

    total_params = 0
    tensor_count = 0
    q8_count = 0

    def add_tensor(name, data):
        nonlocal total_params, tensor_count, q8_count

        if args.q8_0 and not is_critical(name) and data.ndim == 2 and data.shape[-1] % 32 == 0:
            # Quantize to Q8_0
            d = data.astype(np.float32)
            total_params += d.size
            qdata = gguf.quantize(d, gguf.GGMLQuantizationType.Q8_0)
            writer.add_tensor(name, qdata, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
            q8_count += 1
        else:
            d = data.astype(default_np)
            total_params += d.size
            writer.add_tensor(name, d, raw_dtype=default_gguf)
        tensor_count += 1

    # -------------------------------------------------------------------
    # Encoder: Patch Embedding
    # -------------------------------------------------------------------
    print("\n--- Encoder: Patch Embed ---")
    add_tensor("enc.patch_embed.weight", weights["model.encoder.patch_embed.projection.weight"])
    add_tensor("enc.patch_embed.bias", weights["model.encoder.patch_embed.projection.bias"])
    print(f"  projection: {list(weights['model.encoder.patch_embed.projection.weight'].shape)}")

    # Positional embedding (1, 48, 48, 768) -> store as-is
    add_tensor("enc.pos_embed", weights["model.encoder.pos_embed"])
    print(f"  pos_embed: {list(weights['model.encoder.pos_embed'].shape)}")

    # -------------------------------------------------------------------
    # Encoder: ViT Layers
    # -------------------------------------------------------------------
    for li in range(n_enc_layers):
        src = f"model.encoder.layers.{li}"
        dst = f"enc.layers.{li}"
        is_global = li in global_attn_indexes

        # Fused QKV
        add_tensor(f"{dst}.attn.qkv.weight", weights[f"{src}.attn.qkv.weight"])
        add_tensor(f"{dst}.attn.qkv.bias", weights[f"{src}.attn.qkv.bias"])

        # Output projection
        add_tensor(f"{dst}.attn.proj.weight", weights[f"{src}.attn.proj.weight"])
        add_tensor(f"{dst}.attn.proj.bias", weights[f"{src}.attn.proj.bias"])

        # Relative position bias
        rel_h = weights[f"{src}.attn.rel_pos_h"]
        rel_w = weights[f"{src}.attn.rel_pos_w"]
        add_tensor(f"{dst}.attn.rel_pos_h", rel_h)
        add_tensor(f"{dst}.attn.rel_pos_w", rel_w)

        # LayerNorms (pre-LN)
        add_tensor(f"{dst}.ln1.weight", weights[f"{src}.layer_norm1.weight"])
        add_tensor(f"{dst}.ln1.bias", weights[f"{src}.layer_norm1.bias"])
        add_tensor(f"{dst}.ln2.weight", weights[f"{src}.layer_norm2.weight"])
        add_tensor(f"{dst}.ln2.bias", weights[f"{src}.layer_norm2.bias"])

        # MLP
        add_tensor(f"{dst}.mlp.lin1.weight", weights[f"{src}.mlp.lin1.weight"])
        add_tensor(f"{dst}.mlp.lin1.bias", weights[f"{src}.mlp.lin1.bias"])
        add_tensor(f"{dst}.mlp.lin2.weight", weights[f"{src}.mlp.lin2.weight"])
        add_tensor(f"{dst}.mlp.lin2.bias", weights[f"{src}.mlp.lin2.bias"])

        attn_type = "GLOBAL" if is_global else f"WINDOW(ws={window_size})"
        print(f"  layer {li:2d}: {attn_type}, rel_pos_h={list(rel_h.shape)}")

    # -------------------------------------------------------------------
    # Encoder: Neck (Conv1x1 + LN2d + Conv3x3 + LN2d)
    # -------------------------------------------------------------------
    print("\n--- Encoder: Neck ---")
    add_tensor("enc.neck.conv1.weight", weights["model.encoder.neck.conv1.weight"])
    add_tensor("enc.neck.conv2.weight", weights["model.encoder.neck.conv2.weight"])
    add_tensor("enc.neck.ln1.weight", weights["model.encoder.neck.layer_norm1.weight"])
    add_tensor("enc.neck.ln1.bias", weights["model.encoder.neck.layer_norm1.bias"])
    add_tensor("enc.neck.ln2.weight", weights["model.encoder.neck.layer_norm2.weight"])
    add_tensor("enc.neck.ln2.bias", weights["model.encoder.neck.layer_norm2.bias"])
    print(f"  conv1: {list(weights['model.encoder.neck.conv1.weight'].shape)}")
    print(f"  conv2: {list(weights['model.encoder.neck.conv2.weight'].shape)}")

    # -------------------------------------------------------------------
    # Encoder: Multi-modal Projector
    # -------------------------------------------------------------------
    print("\n--- Encoder: Projector ---")
    add_tensor("enc.proj.conv1.weight", weights["model.encoder.multi_modal_projector.conv1.weight"])
    add_tensor("enc.proj.conv2.weight", weights["model.encoder.multi_modal_projector.conv2.weight"])
    add_tensor("enc.proj.linear1.weight", weights["model.encoder.multi_modal_projector.linear_1.weight"])
    add_tensor("enc.proj.linear1.bias", weights["model.encoder.multi_modal_projector.linear_1.bias"])
    add_tensor("enc.proj.linear2.weight", weights["model.encoder.multi_modal_projector.linear_2.weight"])
    add_tensor("enc.proj.linear2.bias", weights["model.encoder.multi_modal_projector.linear_2.bias"])
    print(f"  conv1: {list(weights['model.encoder.multi_modal_projector.conv1.weight'].shape)}")
    print(f"  conv2: {list(weights['model.encoder.multi_modal_projector.conv2.weight'].shape)}")
    print(f"  linear1: {list(weights['model.encoder.multi_modal_projector.linear_1.weight'].shape)}")
    print(f"  linear2: {list(weights['model.encoder.multi_modal_projector.linear_2.weight'].shape)}")

    # -------------------------------------------------------------------
    # Decoder
    # -------------------------------------------------------------------
    print("\n--- Decoder ---")

    # Token and position embeddings
    add_tensor("dec.embed_tokens.weight", weights["model.decoder.embed_tokens.weight"])
    add_tensor("dec.embed_positions.weight", weights["model.decoder.embed_positions.weight"])
    print(f"  embed_tokens: {list(weights['model.decoder.embed_tokens.weight'].shape)}")
    print(f"  embed_positions: {list(weights['model.decoder.embed_positions.weight'].shape)}")

    # Embedding LayerNorm
    add_tensor("dec.embed_ln.weight", weights["model.decoder.layernorm_embedding.weight"])
    add_tensor("dec.embed_ln.bias", weights["model.decoder.layernorm_embedding.bias"])

    # Decoder layers
    for li in range(n_dec_layers):
        src = f"model.decoder.layers.{li}"
        dst = f"dec.layers.{li}"

        # Self-attention
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

        # Cross-attention
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
    add_tensor("dec.final_ln.weight", weights["model.decoder.layer_norm.weight"])
    add_tensor("dec.final_ln.bias", weights["model.decoder.layer_norm.bias"])

    # LM head
    add_tensor("dec.lm_head.weight", weights["lm_head.weight"])
    print(f"  lm_head: {list(weights['lm_head.weight'].shape)}")

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
    dtype_str = 'Q8_0 (critical in F16)' if args.q8_0 else ('FP16' if args.fp16 else 'FP32')
    print(f"Dtype: {dtype_str}")
    if args.q8_0:
        print(f"Q8_0 tensors: {q8_count}, F16 tensors: {tensor_count - q8_count}")


if __name__ == "__main__":
    main()
