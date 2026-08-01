#!/usr/bin/env python3
"""Convert Transcoda-59M (zero-shot OMR) safetensors → GGUF — NO PyTorch.

Optical Music Recognition: full-page score image → Humdrum ``**kern`` tokens.
Reads model.safetensors + config.json directly (torch-free) and packs into a
single GGUF for CrispEmbed's transcoda_ocr inference engine.

Model: btrkeks/transcoda-59M-zeroshot-v1 (weights cc-by-4.0).  Engine written
CLEAN-ROOM from the paper (arXiv 2605.10835) + these config/data files + an
oracle activation dump — the AGPL reference *code* is never read/transcribed.

Architecture facts (from config.json + safetensors weight map, both data files):
  Encoder : HF ConvNextV2Model (facebook/convnextv2-tiny-22k-224) run fully-
            convolutionally.  dims [96,192,384,768], depths [3,3,9,3], /32.
            V2 block: dwconv7x7 -> LN -> pwconv1(->4x) -> GELU -> GRN -> pwconv2,
            residual, NO LayerScale.  Final layernorm[768].
  Bridge  : 2-layer projector 768 -> 2048 -> 512 + 2D sinusoidal PE (host-side).
  Decoder : 8-layer pre-LN cross-attn Transformer.  d_model 512, 8 heads, ffn
            1024 (GELU).  Self-attn RoPE (theta 1e4) causal, fused qkv_proj.
            Cross-attn to the encoder memory.  Untied LM head (vocab_projection).

Tensor names are shortened from the verbatim HF names (the double
``frontend.encoder.encoder`` prefix would blow past GGML_MAX_NAME=64); the
engine's map_tensors() mirrors the short scheme below.

Dependencies: safetensors, gguf, numpy  (no torch / transformers)

Usage:
    python convert-transcoda-to-gguf.py --model-dir /path/to/transcoda-src \
        --output transcoda-f32.gguf [--fp16]
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import gguf
import numpy as np
from safetensors import safe_open

ARCH = "transcoda_ocr"

# ConvNeXt-V2-Tiny fixed shape (not in config; set by the backbone class).
ENC_DIMS = [96, 192, 384, 768]
ENC_DEPTHS = [3, 3, 9, 3]
ENC_NUM_STAGES = 4
ENC_STEM_KERNEL = 4
ENC_REDUCTION = 32
ENC_NUM_CHANNELS = 3
# Transcoda normalizes pixel_values to [-1,1] via (x/255 - 0.5)/0.5 — NOT the
# ImageNet stats of the ConvNeXt-V2 base (documented in the HF model card's
# "Preprocessing" section). mean=std=0.5 per channel.
NORM_MEAN = [0.5, 0.5, 0.5]
NORM_STD = [0.5, 0.5, 0.5]
# Fixed input geometry: RGB, height 1485 x width 1050 (portrait).
FIXED_H = 1485
FIXED_W = 1050


def rename(hf: str) -> str:
    """Map a verbatim HF Transcoda tensor name -> short GGUF name.

    Raises if a name is not recognised (no silent drops)."""
    n = hf
    # --- encoder embeddings ---
    n = n.replace("frontend.encoder.embeddings.patch_embeddings", "enc.embed.patch")
    n = n.replace("frontend.encoder.embeddings.layernorm", "enc.embed.ln")
    # --- encoder final layernorm (applied to pooled output in HF; kept here) ---
    if n == "frontend.encoder.layernorm.weight":
        return "enc.ln.weight"
    if n == "frontend.encoder.layernorm.bias":
        return "enc.ln.bias"
    # --- encoder stages ---
    m = re.match(r"frontend\.encoder\.encoder\.stages\.(\d+)\.downsampling_layer\.(\d+)\.(weight|bias)$", n)
    if m:
        s, idx, wb = m.group(1), m.group(2), m.group(3)
        sub = "ln" if idx == "0" else "conv"
        return f"enc.st{s}.ds.{sub}.{wb}"
    m = re.match(r"frontend\.encoder\.encoder\.stages\.(\d+)\.layers\.(\d+)\.(\w+)\.(weight|bias)$", n)
    if m:
        s, i, part, wb = m.groups()
        pmap = {"dwconv": "dw", "layernorm": "ln", "pwconv1": "pw1", "grn": "grn", "pwconv2": "pw2"}
        if part not in pmap:
            raise ValueError(f"unknown encoder layer part: {hf}")
        return f"enc.st{s}.l{i}.{pmap[part]}.{wb}"
    # --- projector ---
    n = n.replace("frontend.projector.fc1", "proj.fc1")
    n = n.replace("frontend.projector.fc2", "proj.fc2")
    if n.startswith("proj."):
        return n
    if n.startswith("enc."):
        return n
    # --- decoder ---
    if n == "decoder.embedding.weight":
        return "dec.tok_embed.weight"
    if n == "decoder.vocab_projection.weight":
        return "dec.lm_head.weight"
    if n == "decoder.vocab_projection.bias":
        return "dec.lm_head.bias"
    if n == "decoder.decoder.final_norm.weight":
        return "dec.final_norm.weight"
    if n == "decoder.decoder.final_norm.bias":
        return "dec.final_norm.bias"
    m = re.match(r"decoder\.decoder\.layers\.(\d+)\.(.+)\.(weight|bias)$", n)
    if m:
        i, part, wb = m.groups()
        pmap = {
            "self_attn.qkv_proj": "qkv",
            "self_attn.out_proj": "sa_out",
            "cross_attn.q_proj": "ca_q",
            "cross_attn.k_proj": "ca_k",
            "cross_attn.v_proj": "ca_v",
            "cross_attn.out_proj": "ca_out",
            "norm_layers.0": "n0",
            "norm_layers.1": "n1",
            "norm_layers.2": "n2",
            "ffn.0": "ff0",
            "ffn.3": "ff3",
        }
        if part not in pmap:
            raise ValueError(f"unknown decoder layer part: {hf}")
        return f"dec.l{i}.{pmap[part]}.{wb}"
    raise ValueError(f"unmapped tensor: {hf}")


def main():
    p = argparse.ArgumentParser(description="Convert Transcoda-59M safetensors to GGUF")
    p.add_argument("--model-dir", required=True, help="Dir with config.json + model.safetensors")
    p.add_argument("--output", required=True, help="Output GGUF path")
    p.add_argument("--fp16", action="store_true", help="Store 2-D matmul weights FP16 (norms/bias/conv/grn stay F32)")
    p.add_argument("--name", default=None, help="general.name override")
    args = p.parse_args()

    model_dir = Path(args.model_dir)
    with open(model_dir / "config.json") as f:
        cfg = json.load(f)

    d_model = int(cfg["d_model"])
    dim_ff = int(cfg["dim_ff"])
    n_layers = int(cfg["num_hidden_layers"])
    n_heads = int(cfg["num_attn_heads"])
    vocab_size = int(cfg["vocab_size"])
    rope_theta = float(cfg.get("rope_theta", 10000.0))
    bos = int(cfg.get("bos_token_id", 1))
    eos = int(cfg.get("eos_token_id", 2))
    pad = int(cfg.get("pad_token_id", 0))
    max_seq = int(cfg.get("max_length", 2048))

    # vocab: i2w is index-keyed; build tokens[idx] = word
    i2w = cfg["i2w"]
    tokens = [""] * vocab_size
    for idx, word in i2w.items():
        idx = int(idx)
        if 0 <= idx < vocab_size:
            tokens[idx] = word

    print(f"Decoder: {n_layers}L / {n_heads}H / d_model={d_model} / ff={dim_ff}, vocab={vocab_size}")
    print(f"Encoder: ConvNeXt-V2 dims={ENC_DIMS} depths={ENC_DEPTHS}, {ENC_REDUCTION}x, {ENC_NUM_CHANNELS}ch")
    print(f"Tokens : bos={bos} eos={eos} pad={pad}  rope_theta={rope_theta}")

    # ---- weights ----
    st_path = model_dir / "model.safetensors"
    print(f"Loading {st_path} ...")
    src = {}
    with safe_open(str(st_path), framework="numpy") as f:
        for k in f.keys():
            src[k] = f.get_tensor(k)
    print(f"Loaded {len(src)} tensors")

    # rename + squeeze GRN; check name lengths + uniqueness
    out = {}
    for hf in sorted(src.keys()):
        short = rename(hf)
        if len(short) >= 64:
            raise ValueError(f"name too long ({len(short)}): {short}")
        data = src[hf]
        if ".grn." in short and data.ndim == 4:  # [1,1,1,C] -> [C]
            data = data.reshape(-1)
        if short in out:
            raise ValueError(f"duplicate short name {short} (from {hf})")
        out[short] = data
    print(f"Renamed {len(out)} tensors, max name len = {max(len(k) for k in out)}")

    # ---- write GGUF ----
    print(f"Writing GGUF -> {args.output}")
    w = gguf.GGUFWriter(args.output, ARCH)
    w.add_name(args.name or "Transcoda-59M zero-shot OMR")
    w.add_description("Transcoda: full-page score image -> Humdrum **kern tokens")
    # weights are cc-by-4.0; attribution obligation.
    w.add_string("general.license", "cc-by-4.0")
    w.add_string("general.source.url", "https://huggingface.co/btrkeks/transcoda-59M-zeroshot-v1")

    # decoder hparams
    w.add_uint32("transcoda.d_model", d_model)
    w.add_uint32("transcoda.n_layers", n_layers)
    w.add_uint32("transcoda.n_heads", n_heads)
    w.add_uint32("transcoda.dim_ff", dim_ff)
    w.add_uint32("transcoda.vocab_size", vocab_size)
    w.add_float32("transcoda.rope_theta", rope_theta)
    w.add_uint32("transcoda.max_seq_len", max_seq)
    # encoder hparams
    w.add_uint32("transcoda.enc_num_stages", ENC_NUM_STAGES)
    w.add_array("transcoda.enc_hidden_sizes", ENC_DIMS)
    w.add_array("transcoda.enc_depths", ENC_DEPTHS)
    w.add_uint32("transcoda.enc_num_channels", ENC_NUM_CHANNELS)
    w.add_uint32("transcoda.enc_stem_kernel", ENC_STEM_KERNEL)
    w.add_uint32("transcoda.enc_reduction", ENC_REDUCTION)
    w.add_array("transcoda.image_mean", [float(x) for x in NORM_MEAN])
    w.add_array("transcoda.image_std", [float(x) for x in NORM_STD])
    w.add_uint32("transcoda.fixed_height", FIXED_H)
    w.add_uint32("transcoda.fixed_width", FIXED_W)
    # special tokens
    w.add_uint32("transcoda.bos_token_id", bos)
    w.add_uint32("transcoda.eos_token_id", eos)
    w.add_uint32("transcoda.pad_token_id", pad)
    # tokenizer
    w.add_array("tokenizer.tokens", tokens)

    n_written = 0
    for name in sorted(out.keys()):
        data = out[name]
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        is_norm_or_bias = name.endswith(".bias") or ".ln." in name or "norm" in name or ".grn." in name
        is_conv = data.ndim == 4
        if args.fp16 and data.ndim == 2 and not is_norm_or_bias:
            data = data.astype(np.float16)
        w.add_tensor(name, data)
        n_written += 1

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"Done: {args.output} ({size_mb:.1f} MB, {n_written} weight tensors)")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
