#!/usr/bin/env python3
"""Convert CLIP/SigLIP text encoder to GGUF format for CrispEmbed.

Extracts the text tower from a CLIP or SigLIP model and stores it as a
standalone GGUF file with embedded tokenizer (BPE or SentencePiece).

    python models/convert-clip-text-to-gguf.py \
        --model openai/clip-vit-base-patch16 \
        --output clip-text-base.gguf

    python models/convert-clip-text-to-gguf.py \
        --model google/siglip-base-patch16-224 \
        --output siglip-text-base.gguf

CLIP: causal attention, EOS pooling, BPE tokenizer
SigLIP: bidirectional attention, pooler head, SentencePiece tokenizer
"""

import argparse
import json
import sys
from pathlib import Path

import gguf
import numpy as np
import torch


def f32(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().numpy().astype(np.float32)


def main():
    p = argparse.ArgumentParser(description="Convert CLIP text encoder to GGUF")
    p.add_argument("--model", required=True, help="HuggingFace CLIP model ID")
    p.add_argument("--output", required=True, help="Output GGUF path")
    args = p.parse_args()

    from transformers import AutoModel, AutoTokenizer, AutoConfig

    print(f"Loading model: {args.model}")
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.float32,
                                       trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.eval()
    sd = model.state_dict()

    is_siglip = "siglip" in config.model_type.lower()

    tc = config.text_config
    hidden = tc.hidden_size
    layers = tc.num_hidden_layers
    heads = tc.num_attention_heads
    inter = tc.intermediate_size
    vocab_size = tc.vocab_size
    max_pos = tc.max_position_embeddings
    proj_dim = config.projection_dim if hasattr(config, 'projection_dim') else hidden

    print(f"  Type: {'SigLIP' if is_siglip else 'CLIP'}")
    print(f"  Text: hidden={hidden} layers={layers} heads={heads} inter={inter}")
    print(f"  Vocab: {vocab_size} max_pos={max_pos} proj_dim={proj_dim}")

    writer = gguf.GGUFWriter(str(args.output), arch="clip_text")

    # ── Metadata ──
    writer.add_uint32("clip_text.hidden_size", hidden)
    writer.add_uint32("clip_text.num_hidden_layers", layers)
    writer.add_uint32("clip_text.num_attention_heads", heads)
    writer.add_uint32("clip_text.intermediate_size", inter)
    writer.add_uint32("clip_text.vocab_size", vocab_size)
    writer.add_uint32("clip_text.max_position_embeddings", max_pos)
    writer.add_uint32("clip_text.projection_dim", proj_dim)
    writer.add_float32("clip_text.layer_norm_eps", getattr(tc, "layer_norm_eps", 1e-5))
    writer.add_string("clip_text.hidden_act", getattr(tc, "hidden_act", "quick_gelu"))
    writer.add_bool("clip_text.causal", not is_siglip)  # CLIP=causal, SigLIP=bidirectional
    writer.add_bool("clip_text.has_head", is_siglip)     # SigLIP has head projection
    bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 0
    eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 1
    writer.add_uint32("clip_text.bos_token_id", bos)
    writer.add_uint32("clip_text.eos_token_id", eos)

    # ── Tokenizer ──
    vocab = tokenizer.get_vocab()
    tokens = [""] * vocab_size
    scores = [0.0] * vocab_size
    for token_str, token_id in vocab.items():
        if token_id < vocab_size:
            tokens[token_id] = token_str
            scores[token_id] = -float(token_id)

    # Try to get SentencePiece scores if available
    if hasattr(tokenizer, 'sp_model') and tokenizer.sp_model is not None:
        sp = tokenizer.sp_model
        for i in range(min(sp.get_piece_size(), vocab_size)):
            scores[i] = sp.get_score(i)

    writer.add_array("tokenizer.ggml.tokens", tokens)
    writer.add_array("tokenizer.ggml.scores", scores)

    if is_siglip:
        # SentencePiece tokenizer
        writer.add_uint32("tokenizer.ggml.type", 2)  # 2 = SentencePiece
        print(f"  Tokenizer: SentencePiece, {len(tokens)} tokens")
    else:
        # BPE tokenizer with merges
        writer.add_uint32("tokenizer.ggml.type", 1)  # 1 = BPE
        merges = tokenizer.bpe_ranks
        merge_list = sorted(merges.keys(), key=lambda x: merges[x])
        merge_strings = [f"{a} {b}" for a, b in merge_list]
        writer.add_array("tokenizer.ggml.merges", merge_strings)
        print(f"  Tokenizer: BPE, {len(tokens)} tokens, {len(merge_strings)} merges")

    # ── Embeddings ──
    tpfx = "text_model."

    # Token embeddings
    tok_embd = sd[f"{tpfx}embeddings.token_embedding.weight"]
    writer.add_tensor("token_embd.weight", f32(tok_embd))
    print(f"  token_embd: {list(tok_embd.shape)}")

    # Position embeddings
    pos_embd = sd[f"{tpfx}embeddings.position_embedding.weight"]
    writer.add_tensor("position_embd.weight", f32(pos_embd))
    print(f"  position_embd: {list(pos_embd.shape)}")

    # ── Encoder layers ──
    for i in range(layers):
        lpfx = f"{tpfx}encoder.layers.{i}."

        # Layer norm 1
        writer.add_tensor(f"enc.{i}.ln1.weight", f32(sd[f"{lpfx}layer_norm1.weight"]))
        writer.add_tensor(f"enc.{i}.ln1.bias", f32(sd[f"{lpfx}layer_norm1.bias"]))

        # Q/K/V (separate projections for CLIP text)
        for proj in ['q_proj', 'k_proj', 'v_proj']:
            p_char = proj[0]
            writer.add_tensor(f"enc.{i}.attn.{p_char}.weight",
                             f32(sd[f"{lpfx}self_attn.{proj}.weight"]))
            writer.add_tensor(f"enc.{i}.attn.{p_char}.bias",
                             f32(sd[f"{lpfx}self_attn.{proj}.bias"]))

        # Output projection
        writer.add_tensor(f"enc.{i}.attn.o.weight",
                         f32(sd[f"{lpfx}self_attn.out_proj.weight"]))
        writer.add_tensor(f"enc.{i}.attn.o.bias",
                         f32(sd[f"{lpfx}self_attn.out_proj.bias"]))

        # Layer norm 2
        writer.add_tensor(f"enc.{i}.ln2.weight", f32(sd[f"{lpfx}layer_norm2.weight"]))
        writer.add_tensor(f"enc.{i}.ln2.bias", f32(sd[f"{lpfx}layer_norm2.bias"]))

        # FFN
        writer.add_tensor(f"enc.{i}.ffn.fc1.weight", f32(sd[f"{lpfx}mlp.fc1.weight"]))
        writer.add_tensor(f"enc.{i}.ffn.fc1.bias", f32(sd[f"{lpfx}mlp.fc1.bias"]))
        writer.add_tensor(f"enc.{i}.ffn.fc2.weight", f32(sd[f"{lpfx}mlp.fc2.weight"]))
        writer.add_tensor(f"enc.{i}.ffn.fc2.bias", f32(sd[f"{lpfx}mlp.fc2.bias"]))

        print(f"  enc.{i}: ok")

    # Final layer norm
    writer.add_tensor("final_ln.weight", f32(sd[f"{tpfx}final_layer_norm.weight"]))
    writer.add_tensor("final_ln.bias", f32(sd[f"{tpfx}final_layer_norm.bias"]))
    print("  final_ln: ok")

    # Text projection (CLIP) or head projection (SigLIP)
    tp_key = "text_projection.weight"
    head_w_key = f"{tpfx}head.weight"
    if tp_key in sd:
        writer.add_tensor("text_proj.weight", f32(sd[tp_key]))
        print(f"  text_projection: {list(sd[tp_key].shape)}")
    if head_w_key in sd:
        writer.add_tensor("head.weight", f32(sd[head_w_key]))
        head_b_key = f"{tpfx}head.bias"
        if head_b_key in sd:
            writer.add_tensor("head.bias", f32(sd[head_b_key]))
        print(f"  head: {list(sd[head_w_key].shape)}")

    # Write
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = Path(args.output).stat().st_size / 1024 / 1024
    print(f"\nWrote {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
