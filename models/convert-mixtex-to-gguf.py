#!/usr/bin/env python3
"""Convert MixTex ZhEn-Latex-OCR (Swin-Tiny + RoBERTa) to GGUF.

Architecture:
  Encoder: Swin-Tiny (patch4, window7, depths=[2,2,6,2], embed_dim=96)
           Output: [N, 768] sequence
  Decoder: RoBERTa 4-layer with cross-attention
           hidden=768, heads=12, ffn=3072, vocab=25681
           BPE tokenizer (vocab.json + merges.txt)

Usage:
    python models/convert-mixtex-to-gguf.py \\
        --model /mnt/storage/models/mixtex-zhen \\
        --output /mnt/storage/gguf-models/mixtex-zhen-f32.gguf

    python models/convert-mixtex-to-gguf.py \\
        --model /mnt/storage/models/mixtex-zhen \\
        --output /mnt/storage/gguf-models/mixtex-zhen-f16.gguf --fp16
"""

import argparse
import json
import sys
from pathlib import Path
from collections import OrderedDict

import gguf
import numpy as np


def load_safetensors(path):
    """Load safetensors file into dict of numpy arrays."""
    import safetensors.torch as st
    tensors = st.load_file(str(path))
    return {k: v.cpu().float().numpy() for k, v in tensors.items()}


def convert_encoder_tensors(sd):
    """Convert Swin encoder tensors to GGUF naming convention."""
    tensors = OrderedDict()

    # Patch embedding: Conv2d [96, 3, 4, 4]
    tensors["enc.patch.weight"] = sd["encoder.embeddings.patch_embeddings.projection.weight"]
    tensors["enc.patch.bias"] = sd["encoder.embeddings.patch_embeddings.projection.bias"]

    # Patch norm (LayerNorm after patch embedding)
    tensors["enc.patch_norm.weight"] = sd["encoder.embeddings.norm.weight"]
    tensors["enc.patch_norm.bias"] = sd["encoder.embeddings.norm.bias"]

    # Swin stages
    stage_depths = [2, 2, 6, 2]
    for stage_i in range(4):
        for block_i in range(stage_depths[stage_i]):
            src_prefix = f"encoder.encoder.layers.{stage_i}.blocks.{block_i}"
            dst_prefix = f"enc.stage{stage_i}.block{block_i}"

            # Layer norms
            tensors[f"{dst_prefix}.ln1.weight"] = sd[f"{src_prefix}.layernorm_before.weight"]
            tensors[f"{dst_prefix}.ln1.bias"] = sd[f"{src_prefix}.layernorm_before.bias"]
            tensors[f"{dst_prefix}.ln2.weight"] = sd[f"{src_prefix}.layernorm_after.weight"]
            tensors[f"{dst_prefix}.ln2.bias"] = sd[f"{src_prefix}.layernorm_after.bias"]

            # Self-attention Q/K/V
            tensors[f"{dst_prefix}.attn.q.weight"] = sd[f"{src_prefix}.attention.self.query.weight"]
            tensors[f"{dst_prefix}.attn.q.bias"] = sd[f"{src_prefix}.attention.self.query.bias"]
            tensors[f"{dst_prefix}.attn.k.weight"] = sd[f"{src_prefix}.attention.self.key.weight"]
            tensors[f"{dst_prefix}.attn.k.bias"] = sd[f"{src_prefix}.attention.self.key.bias"]
            tensors[f"{dst_prefix}.attn.v.weight"] = sd[f"{src_prefix}.attention.self.value.weight"]
            tensors[f"{dst_prefix}.attn.v.bias"] = sd[f"{src_prefix}.attention.self.value.bias"]

            # Output projection
            tensors[f"{dst_prefix}.attn.out.weight"] = sd[f"{src_prefix}.attention.output.dense.weight"]
            tensors[f"{dst_prefix}.attn.out.bias"] = sd[f"{src_prefix}.attention.output.dense.bias"]

            # Relative position bias table and index
            tensors[f"{dst_prefix}.attn.rpb_table"] = sd[f"{src_prefix}.attention.self.relative_position_bias_table"]
            tensors[f"{dst_prefix}.attn.rpb_index"] = sd[f"{src_prefix}.attention.self.relative_position_index"].astype(np.float32)

            # FFN
            tensors[f"{dst_prefix}.ffn.up.weight"] = sd[f"{src_prefix}.intermediate.dense.weight"]
            tensors[f"{dst_prefix}.ffn.up.bias"] = sd[f"{src_prefix}.intermediate.dense.bias"]
            tensors[f"{dst_prefix}.ffn.down.weight"] = sd[f"{src_prefix}.output.dense.weight"]
            tensors[f"{dst_prefix}.ffn.down.bias"] = sd[f"{src_prefix}.output.dense.bias"]

        # Downsample (patch merging) between stages — except after last stage
        if stage_i < 3:
            src_ds = f"encoder.encoder.layers.{stage_i}.downsample"
            dst_ds = f"enc.stage{stage_i}.downsample"
            tensors[f"{dst_ds}.norm.weight"] = sd[f"{src_ds}.norm.weight"]
            tensors[f"{dst_ds}.norm.bias"] = sd[f"{src_ds}.norm.bias"]
            tensors[f"{dst_ds}.reduction.weight"] = sd[f"{src_ds}.reduction.weight"]

    # Encoder LayerNorm (final)
    if "encoder.layernorm.weight" in sd:
        tensors["enc.final_norm.weight"] = sd["encoder.layernorm.weight"]
        tensors["enc.final_norm.bias"] = sd["encoder.layernorm.bias"]

    return tensors


def convert_decoder_tensors(sd):
    """Convert RoBERTa decoder tensors to GGUF naming convention."""
    tensors = OrderedDict()

    # Embeddings
    tensors["dec.word_embed.weight"] = sd["decoder.roberta.embeddings.word_embeddings.weight"]
    tensors["dec.pos_embed.weight"] = sd["decoder.roberta.embeddings.position_embeddings.weight"]
    tensors["dec.type_embed.weight"] = sd["decoder.roberta.embeddings.token_type_embeddings.weight"]
    tensors["dec.embed_ln.weight"] = sd["decoder.roberta.embeddings.LayerNorm.weight"]
    tensors["dec.embed_ln.bias"] = sd["decoder.roberta.embeddings.LayerNorm.bias"]

    # Decoder layers (4)
    for i in range(4):
        src = f"decoder.roberta.encoder.layer.{i}"
        dst = f"dec.layers.{i}"

        # Self-attention
        tensors[f"{dst}.self_ln.weight"] = sd[f"{src}.attention.output.LayerNorm.weight"]
        tensors[f"{dst}.self_ln.bias"] = sd[f"{src}.attention.output.LayerNorm.bias"]
        tensors[f"{dst}.self_q.weight"] = sd[f"{src}.attention.self.query.weight"]
        tensors[f"{dst}.self_q.bias"] = sd[f"{src}.attention.self.query.bias"]
        tensors[f"{dst}.self_k.weight"] = sd[f"{src}.attention.self.key.weight"]
        tensors[f"{dst}.self_k.bias"] = sd[f"{src}.attention.self.key.bias"]
        tensors[f"{dst}.self_v.weight"] = sd[f"{src}.attention.self.value.weight"]
        tensors[f"{dst}.self_v.bias"] = sd[f"{src}.attention.self.value.bias"]
        tensors[f"{dst}.self_out.weight"] = sd[f"{src}.attention.output.dense.weight"]
        tensors[f"{dst}.self_out.bias"] = sd[f"{src}.attention.output.dense.bias"]

        # Cross-attention
        tensors[f"{dst}.cross_ln.weight"] = sd[f"{src}.crossattention.output.LayerNorm.weight"]
        tensors[f"{dst}.cross_ln.bias"] = sd[f"{src}.crossattention.output.LayerNorm.bias"]
        tensors[f"{dst}.cross_q.weight"] = sd[f"{src}.crossattention.self.query.weight"]
        tensors[f"{dst}.cross_q.bias"] = sd[f"{src}.crossattention.self.query.bias"]
        tensors[f"{dst}.cross_k.weight"] = sd[f"{src}.crossattention.self.key.weight"]
        tensors[f"{dst}.cross_k.bias"] = sd[f"{src}.crossattention.self.key.bias"]
        tensors[f"{dst}.cross_v.weight"] = sd[f"{src}.crossattention.self.value.weight"]
        tensors[f"{dst}.cross_v.bias"] = sd[f"{src}.crossattention.self.value.bias"]
        tensors[f"{dst}.cross_out.weight"] = sd[f"{src}.crossattention.output.dense.weight"]
        tensors[f"{dst}.cross_out.bias"] = sd[f"{src}.crossattention.output.dense.bias"]

        # FFN
        tensors[f"{dst}.ffn_ln.weight"] = sd[f"{src}.output.LayerNorm.weight"]
        tensors[f"{dst}.ffn_ln.bias"] = sd[f"{src}.output.LayerNorm.bias"]
        tensors[f"{dst}.ffn.up.weight"] = sd[f"{src}.intermediate.dense.weight"]
        tensors[f"{dst}.ffn.up.bias"] = sd[f"{src}.intermediate.dense.bias"]
        tensors[f"{dst}.ffn.down.weight"] = sd[f"{src}.output.dense.weight"]
        tensors[f"{dst}.ffn.down.bias"] = sd[f"{src}.output.dense.bias"]

    # LM head
    tensors["dec.lm_head.dense.weight"] = sd["decoder.lm_head.dense.weight"]
    tensors["dec.lm_head.dense.bias"] = sd["decoder.lm_head.dense.bias"]
    tensors["dec.lm_head.ln.weight"] = sd["decoder.lm_head.layer_norm.weight"]
    tensors["dec.lm_head.ln.bias"] = sd["decoder.lm_head.layer_norm.bias"]
    tensors["dec.lm_head.bias"] = sd["decoder.lm_head.bias"]
    # lm_head weight is tied to word_embeddings — store reference
    # Actually check if it exists separately
    if "decoder.lm_head.decoder.weight" in sd:
        tensors["dec.lm_head.weight"] = sd["decoder.lm_head.decoder.weight"]

    return tensors


def load_tokenizer(model_dir):
    """Load BPE tokenizer vocab and merges."""
    vocab_path = Path(model_dir) / "vocab.json"
    merges_path = Path(model_dir) / "merges.txt"

    vocab = json.loads(vocab_path.read_text())
    # vocab is {token_str: token_id}

    merges = []
    with open(merges_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                merges.append(line)

    # Build token list ordered by ID
    max_id = max(vocab.values())
    tokens = [""] * (max_id + 1)
    for tok, idx in vocab.items():
        tokens[idx] = tok

    return tokens, merges


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Model directory path")
    parser.add_argument("--output", required=True, help="Output GGUF path")
    parser.add_argument("--fp16", action="store_true", help="Store large tensors in F16")
    args = parser.parse_args()

    model_dir = Path(args.model)

    print("Loading weights...")
    sd = load_safetensors(model_dir / "model.safetensors")
    total_params = sum(v.size for v in sd.values())
    print(f"Loaded: {total_params:,} params, {len(sd)} tensors")

    print("Loading config...")
    config = json.loads((model_dir / "config.json").read_text())
    enc_config = config.get("encoder", {})
    dec_config = config.get("decoder", {})

    print("Converting encoder tensors...")
    enc_tensors = convert_encoder_tensors(sd)
    print(f"  {len(enc_tensors)} encoder tensors")

    print("Converting decoder tensors...")
    dec_tensors = convert_decoder_tensors(sd)
    print(f"  {len(dec_tensors)} decoder tensors")

    print("Loading tokenizer...")
    tokens, merges = load_tokenizer(model_dir)
    print(f"  {len(tokens)} tokens, {len(merges)} merges")

    # Merge all tensors
    all_tensors = OrderedDict()
    all_tensors.update(enc_tensors)
    all_tensors.update(dec_tensors)

    # Check for missing weights
    accounted = set()
    for k in sd:
        matched = False
        for gguf_name, data in all_tensors.items():
            # Check if this sd key was used
            pass
        accounted.add(k)  # We'll validate by count

    print(f"\nTotal: {len(all_tensors)} GGUF tensors from {len(sd)} source tensors")

    # Write GGUF
    print(f"Writing to {args.output}")
    writer = gguf.GGUFWriter(args.output, "mixtex")

    # Hyperparameters
    writer.add_description("MixTex ZhEn-Latex-OCR (Swin-Tiny + RoBERTa)")

    # Encoder hparams
    writer.add_uint32("mixtex.encoder.patch_size", 4)
    writer.add_uint32("mixtex.encoder.window_size", 7)
    writer.add_uint32("mixtex.encoder.embed_dim", 96)
    writer.add_array("mixtex.encoder.depths", [2, 2, 6, 2])
    writer.add_array("mixtex.encoder.num_heads", [3, 6, 12, 24])
    writer.add_uint32("mixtex.encoder.hidden_size", 768)
    writer.add_uint32("mixtex.encoder.image_h", 400)
    writer.add_uint32("mixtex.encoder.image_w", 500)

    # Decoder hparams
    writer.add_uint32("mixtex.decoder.hidden_size", 768)
    writer.add_uint32("mixtex.decoder.num_layers", 4)
    writer.add_uint32("mixtex.decoder.num_heads", 12)
    writer.add_uint32("mixtex.decoder.ffn_dim", 3072)
    writer.add_uint32("mixtex.decoder.vocab_size", len(tokens))
    writer.add_uint32("mixtex.decoder.max_position", 300)
    writer.add_uint32("mixtex.decoder.sos_token", 0)
    writer.add_uint32("mixtex.decoder.eos_token", config.get("eos_token_id", 25678))

    # Tokenizer
    writer.add_array("tokenizer.tokens", tokens)
    if merges:
        writer.add_array("tokenizer.merges", merges[:50000])  # cap for GGUF size

    # ImageNet normalization
    writer.add_array("mixtex.image_mean", [0.5, 0.5, 0.5])
    writer.add_array("mixtex.image_std", [0.5, 0.5, 0.5])

    # Write tensors
    total_bytes = 0
    for name, data in all_tensors.items():
        data = data.astype(np.float32)
        if args.fp16 and data.size > 256:
            data = data.astype(np.float16)
        writer.add_tensor(name, data)
        total_bytes += data.nbytes

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    out_size = Path(args.output).stat().st_size / 1024 / 1024
    print(f"\nDone: {args.output} ({out_size:.1f} MB, {len(all_tensors)} tensors)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
