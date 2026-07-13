#!/usr/bin/env python3
"""Convert Polyphonic-TrOMR (NetEase, Apache-2.0) to GGUF for CrispEmbed.

Optical Music Recognition: staff image → 3 parallel token streams
(rhythm / pitch / lift), later merged into semantic music notation.

Architecture (github.com/NetEase/Polyphonic-TrOMR, read line-by-line):
  Encoder : timm hybrid ViT — ResNetV2 backbone (StdConv2dSame + GroupNorm,
            layers [2,3,7], 1→64→256→512→1024) → HybridEmbed 1×1 proj (1024→256)
            → ViT (depth 4, 8 heads, dim 256, cls token, custom 2D pos-index).
  Decoder : x_transformers Decoder (depth 4, 8 heads, dim 256; per depth:
            self-attn → cross-attn → GLU-FF, with attn-on-attn gating). Input =
            rhythm_emb + pitch_emb + lift_emb + abs pos_emb. 4 heads
            (rhythm 260 / pitch 71 / lift 7 / note 2).

The StdConv2dSame backbone convs are weight-standardized HERE (exactly as timm:
F.batch_norm over each output channel's flattened weights, eps=1e-6) so the C++
engine runs plain convs. Everything else is packed verbatim.

Dependencies: torch, gguf, numpy, pyyaml
Usage:
    python convert-tromr-to-gguf.py --repo <Polyphonic-TrOMR/tromr dir> \
        --output tromr.gguf [--fp16]
"""

import argparse
import json
import sys
from pathlib import Path

import gguf
import numpy as np
import torch
import torch.nn.functional as F
import yaml

ARCH = "tromr_ocr"


def std_conv(w):
    """timm StdConv2dSame weight standardization (per output channel, eps=1e-6)."""
    oc = w.shape[0]
    return F.batch_norm(w.reshape(1, oc, -1), None, None, None, None, training=True, momentum=0.0,
                        eps=1e-6).reshape_as(w)


def load_vocab(path):
    """PreTrainedTokenizerFast JSON → id-indexed token list."""
    tok = json.load(open(path))
    vocab = tok["model"]["vocab"]  # token -> id
    for at in tok.get("added_tokens", []):
        vocab[at["content"]] = at["id"]
    n = max(vocab.values()) + 1
    out = [""] * n
    for t, i in vocab.items():
        if 0 <= i < n:
            out[i] = t
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, help="Polyphonic-TrOMR/tromr directory")
    p.add_argument("--output", required=True)
    p.add_argument("--fp16", action="store_true")
    args = p.parse_args()

    repo = Path(args.repo)
    ws = repo / "workspace"
    cfg = yaml.safe_load(open(ws / "config.yaml"))

    ckpt = ws / cfg["filepaths"]["checkpoint"]
    sd = torch.load(ckpt, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    print(f"Loaded {len(sd)} tensors from {ckpt}")

    # weight-standardize the StdConv backbone convs. Every 4D tensor under
    # patch_embed.backbone is a StdConv2dSame kernel (stem/conv1/2/3/downsample);
    # the HybridEmbed proj (patch_embed.proj) is a plain Conv2d, excluded by name.
    n_std = 0
    for k in list(sd.keys()):
        if "patch_embed.backbone" in k and sd[k].ndim == 4:
            sd[k] = std_conv(sd[k])
            n_std += 1
    print(f"Weight-standardized {n_std} StdConv backbone conv weights")

    tokv = {
        "rhythm": load_vocab(ws / cfg["filepaths"]["rhythmtokenizer"]),
        "pitch": load_vocab(ws / cfg["filepaths"]["pitchtokenizer"]),
        "lift": load_vocab(ws / cfg["filepaths"]["lifttokenizer"]),
    }
    for name, v in tokv.items():
        print(f"  {name} vocab: {len(v)} tokens")

    w = gguf.GGUFWriter(args.output, ARCH)
    w.add_name("Polyphonic-TrOMR OMR")
    w.add_description("TrOMR: staff image -> rhythm/pitch/lift token streams")
    w.add_string("general.license", "Apache-2.0")

    # hparams
    w.add_uint32("tromr.channels", int(cfg["channels"]))
    w.add_uint32("tromr.patch_size", int(cfg["patch_size"]))
    w.add_uint32("tromr.max_height", int(cfg["max_height"]))
    w.add_uint32("tromr.max_width", int(cfg["max_width"]))
    w.add_uint32("tromr.max_seq_len", int(cfg["max_seq_len"]))
    w.add_uint32("tromr.pad_token", int(cfg["pad_token"]))
    w.add_uint32("tromr.bos_token", int(cfg["bos_token"]))
    w.add_uint32("tromr.eos_token", int(cfg["eos_token"]))
    w.add_uint32("tromr.nonote_token", int(cfg["nonote_token"]))
    w.add_uint32("tromr.encoder_dim", int(cfg["encoder_dim"]))
    w.add_uint32("tromr.encoder_depth", int(cfg["encoder_depth"]))
    w.add_uint32("tromr.encoder_heads", int(cfg["encoder_heads"]))
    w.add_array("tromr.backbone_layers", [int(x) for x in cfg["backbone_layers"]])
    w.add_uint32("tromr.decoder_dim", int(cfg["decoder_dim"]))
    w.add_uint32("tromr.decoder_depth", int(cfg["decoder_depth"]))
    w.add_uint32("tromr.decoder_heads", int(cfg["decoder_heads"]))
    w.add_uint32("tromr.num_rhythm_tokens", int(cfg["num_rhythm_tokens"]))
    w.add_uint32("tromr.num_pitch_tokens", int(cfg["num_pitch_tokens"]))
    w.add_uint32("tromr.num_lift_tokens", int(cfg["num_lift_tokens"]))
    w.add_uint32("tromr.num_note_tokens", int(cfg["num_note_tokens"]))
    # normalization stats (staff2score transform)
    w.add_float32("tromr.norm_mean", 0.7931)
    w.add_float32("tromr.norm_std", 0.1738)
    # tokenizers
    w.add_array("tromr.rhythm_tokens", tokv["rhythm"])
    w.add_array("tromr.pitch_tokens", tokv["pitch"])
    w.add_array("tromr.lift_tokens", tokv["lift"])

    # ggml enforces GGML_MAX_NAME=64; the backbone downsample names reach 69 chars
    # (encoder.patch_embed.backbone.stages.0.blocks.0.downsample.conv.weight). Shorten
    # the backbone prefix so the ggml C loader accepts every tensor. The engine's
    # map_tensors() uses the same "enc.bb" prefix.
    def short(name):
        return name.replace("encoder.patch_embed.backbone", "enc.bb")

    n = 0
    for name in sorted(sd.keys()):
        d = sd[name].detach().cpu().float().numpy()
        if args.fp16 and d.ndim == 2 and not name.endswith(".bias"):
            d = d.astype(np.float16)
        w.add_tensor(short(name), np.ascontiguousarray(d))
        n += 1

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    import os
    print(f"Done: {args.output} ({os.path.getsize(args.output)/1e6:.1f} MB, {n} tensors)")


if __name__ == "__main__":
    sys.exit(main() or 0)
