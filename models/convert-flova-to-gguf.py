#!/usr/bin/env python3
"""Convert Flova/omr_transformer (Apache-2.0) to GGUF for CrispEmbed.

Optical Music Recognition for handwritten / whiteboard "simple notes" → LilyPond.
A Donut VisionEncoderDecoder:
  Encoder : DonutSwin (Swin-Base scale) — patch 4, window 10, embed_dim 128,
            depths [2,2,14,2], heads [4,8,16,32], hidden 1024, image 583×409.
  Decoder : mBART 4-layer (PRE-norm) — d_model 1024, 16 heads, ffn 4096,
            vocab 75, learned positions (offset 2), scale_embedding (×√1024),
            GELU. decoder_start/bos 56 (top-level config), eos 54 (</s>),
            pad 55. NOTE: generation_config's eos_token_id=2 is a stale mBART
            default — the real eos is the tokenizer's </s>=54 (verified: with
            eos=54 the greedy decode stops cleanly and matches the model card).

Swin encoder tensor mapping mirrors convert-mixtex-to-gguf.py (identical HF
DonutSwin naming). Blueprint: github.com/UHHRobotics22-23/robot_project
(marimbabot_vision).

Usage:
    python models/convert-flova-to-gguf.py --repo <hf id or local dir> \
        --output flova-f32.gguf [--fp16]
"""

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import gguf
import numpy as np
import torch

ARCH = "flova_ocr"
DEPTHS = [2, 2, 14, 2]
NUM_HEADS = [4, 8, 16, 32]


def convert_encoder(sd):
    t = OrderedDict()
    t["enc.patch.weight"] = sd["encoder.embeddings.patch_embeddings.projection.weight"]
    t["enc.patch.bias"] = sd["encoder.embeddings.patch_embeddings.projection.bias"]
    t["enc.patch_norm.weight"] = sd["encoder.embeddings.norm.weight"]
    t["enc.patch_norm.bias"] = sd["encoder.embeddings.norm.bias"]
    for s in range(4):
        for b in range(DEPTHS[s]):
            src = f"encoder.encoder.layers.{s}.blocks.{b}"
            dst = f"enc.stage{s}.block{b}"
            t[f"{dst}.ln1.weight"] = sd[f"{src}.layernorm_before.weight"]
            t[f"{dst}.ln1.bias"] = sd[f"{src}.layernorm_before.bias"]
            t[f"{dst}.ln2.weight"] = sd[f"{src}.layernorm_after.weight"]
            t[f"{dst}.ln2.bias"] = sd[f"{src}.layernorm_after.bias"]
            t[f"{dst}.attn.q.weight"] = sd[f"{src}.attention.self.query.weight"]
            t[f"{dst}.attn.q.bias"] = sd[f"{src}.attention.self.query.bias"]
            t[f"{dst}.attn.k.weight"] = sd[f"{src}.attention.self.key.weight"]
            t[f"{dst}.attn.k.bias"] = sd[f"{src}.attention.self.key.bias"]
            t[f"{dst}.attn.v.weight"] = sd[f"{src}.attention.self.value.weight"]
            t[f"{dst}.attn.v.bias"] = sd[f"{src}.attention.self.value.bias"]
            t[f"{dst}.attn.out.weight"] = sd[f"{src}.attention.output.dense.weight"]
            t[f"{dst}.attn.out.bias"] = sd[f"{src}.attention.output.dense.bias"]
            t[f"{dst}.attn.rpb_table"] = sd[f"{src}.attention.self.relative_position_bias_table"]
            t[f"{dst}.attn.rpb_index"] = sd[f"{src}.attention.self.relative_position_index"].astype(np.float32)
            t[f"{dst}.ffn.up.weight"] = sd[f"{src}.intermediate.dense.weight"]
            t[f"{dst}.ffn.up.bias"] = sd[f"{src}.intermediate.dense.bias"]
            t[f"{dst}.ffn.down.weight"] = sd[f"{src}.output.dense.weight"]
            t[f"{dst}.ffn.down.bias"] = sd[f"{src}.output.dense.bias"]
        if s < 3:
            src = f"encoder.encoder.layers.{s}.downsample"
            dst = f"enc.stage{s}.downsample"
            t[f"{dst}.norm.weight"] = sd[f"{src}.norm.weight"]
            t[f"{dst}.norm.bias"] = sd[f"{src}.norm.bias"]
            t[f"{dst}.reduction.weight"] = sd[f"{src}.reduction.weight"]
    # DonutSwin final layernorm (applied to last_hidden_state)
    if "encoder.layernorm.weight" in sd:
        t["enc.final_norm.weight"] = sd["encoder.layernorm.weight"]
        t["enc.final_norm.bias"] = sd["encoder.layernorm.bias"]
    return t


def convert_decoder(sd):
    t = OrderedDict()
    D = "decoder.model.decoder"
    t["dec.embed_tokens.weight"] = sd[f"{D}.embed_tokens.weight"]
    t["dec.embed_positions.weight"] = sd[f"{D}.embed_positions.weight"]
    t["dec.embed_ln.weight"] = sd[f"{D}.layernorm_embedding.weight"]
    t["dec.embed_ln.bias"] = sd[f"{D}.layernorm_embedding.bias"]
    for i in range(4):
        src = f"{D}.layers.{i}"
        dst = f"dec.layers.{i}"
        for a, pfx in (("self_attn", "self"), ("encoder_attn", "cross")):
            for p in ("q", "k", "v", "out"):
                t[f"{dst}.{pfx}_{p}.weight"] = sd[f"{src}.{a}.{p}_proj.weight"]
                t[f"{dst}.{pfx}_{p}.bias"] = sd[f"{src}.{a}.{p}_proj.bias"]
        t[f"{dst}.self_ln.weight"] = sd[f"{src}.self_attn_layer_norm.weight"]
        t[f"{dst}.self_ln.bias"] = sd[f"{src}.self_attn_layer_norm.bias"]
        t[f"{dst}.cross_ln.weight"] = sd[f"{src}.encoder_attn_layer_norm.weight"]
        t[f"{dst}.cross_ln.bias"] = sd[f"{src}.encoder_attn_layer_norm.bias"]
        t[f"{dst}.ffn_ln.weight"] = sd[f"{src}.final_layer_norm.weight"]
        t[f"{dst}.ffn_ln.bias"] = sd[f"{src}.final_layer_norm.bias"]
        t[f"{dst}.ffn.up.weight"] = sd[f"{src}.fc1.weight"]
        t[f"{dst}.ffn.up.bias"] = sd[f"{src}.fc1.bias"]
        t[f"{dst}.ffn.down.weight"] = sd[f"{src}.fc2.weight"]
        t[f"{dst}.ffn.down.bias"] = sd[f"{src}.fc2.bias"]
    t["dec.final_norm.weight"] = sd[f"{D}.layer_norm.weight"]
    t["dec.final_norm.bias"] = sd[f"{D}.layer_norm.bias"]
    t["dec.lm_head.weight"] = sd["decoder.lm_head.weight"]
    return t


def load_vocab(repo):
    """75-token id→string list, via the HF tokenizer (contiguous 0..74)."""
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(repo)
    v = tk.get_vocab()  # token -> id
    n = max(v.values()) + 1
    out = [""] * n
    for tok, i in v.items():
        out[i] = tok
    return out, tk


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, help="HF id (Flova/omr_transformer) or local dir")
    p.add_argument("--output", required=True)
    p.add_argument("--fp16", action="store_true")
    args = p.parse_args()

    from huggingface_hub import hf_hub_download
    src = args.repo
    binp = Path(src) / "pytorch_model.bin" if Path(src).is_dir() else Path(hf_hub_download(src, "pytorch_model.bin"))
    sd = torch.load(binp, map_location="cpu", weights_only=True)
    sd = {k: v.detach().cpu().float().numpy() for k, v in sd.items()}
    print(f"Loaded {len(sd)} tensors")

    enc = convert_encoder(sd)
    dec = convert_decoder(sd)
    tokens, _ = load_vocab(src)
    print(f"  encoder {len(enc)}, decoder {len(dec)} tensors; vocab {len(tokens)}")

    w = gguf.GGUFWriter(args.output, ARCH)
    w.add_name("Flova OMR Transformer")
    w.add_description("Flova/omr_transformer: handwritten/whiteboard OMR -> LilyPond")
    w.add_string("general.license", "apache-2.0")

    w.add_uint32("flova.encoder.patch_size", 4)
    w.add_uint32("flova.encoder.window_size", 10)
    w.add_uint32("flova.encoder.embed_dim", 128)
    w.add_uint32("flova.encoder.hidden_size", 1024)
    w.add_uint32("flova.encoder.image_h", 583)
    w.add_uint32("flova.encoder.image_w", 409)
    w.add_array("flova.encoder.depths", DEPTHS)
    w.add_array("flova.encoder.num_heads", NUM_HEADS)

    w.add_uint32("flova.decoder.hidden_size", 1024)
    w.add_uint32("flova.decoder.num_layers", 4)
    w.add_uint32("flova.decoder.num_heads", 16)
    w.add_uint32("flova.decoder.ffn_dim", 4096)
    w.add_uint32("flova.decoder.vocab_size", int(dec["dec.embed_tokens.weight"].shape[0]))
    w.add_uint32("flova.decoder.max_position", 1536)
    w.add_uint32("flova.decoder.scale_embedding", 1)
    w.add_uint32("flova.decoder_start_token", 56)  # <s> (top-level config)
    w.add_uint32("flova.eos_token", 54)             # </s> (NOT gen_config's stale 2)
    w.add_uint32("flova.pad_token", 55)
    w.add_uint32("flova.unk_token", 0)
    w.add_array("flova.image_mean", [0.5, 0.5, 0.5])
    w.add_array("flova.image_std", [0.5, 0.5, 0.5])
    w.add_array("tokenizer.tokens", tokens)

    allt = OrderedDict()
    allt.update(enc)
    allt.update(dec)
    n = 0
    for name, d in allt.items():
        d = np.ascontiguousarray(d.astype(np.float32))
        if args.fp16 and d.ndim == 2 and not name.endswith(".bias") and d.shape[0] > 1:
            d = d.astype(np.float16)
        w.add_tensor(name, d)
        n += 1

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    import os
    print(f"Done: {args.output} ({os.path.getsize(args.output)/1e6:.1f} MB, {n} tensors)")


if __name__ == "__main__":
    sys.exit(main() or 0)
