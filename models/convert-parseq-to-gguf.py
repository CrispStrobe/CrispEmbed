#!/usr/bin/env python3
"""Convert PARSeq scene text recognition model to GGUF.

Loads PyTorch checkpoint from baudm/parseq (Apache-2.0), packs into GGUF
for CrispEmbed inference.

Architecture:
  Encoder: ViT (12 layers, pre-LN, GELU FFN, learned pos embed)
    - Base:  384d, 6 heads, 1536 FFN, patch [4,8], img [32,128] → 128 tokens
    - Tiny:  192d, 3 heads,  768 FFN, patch [4,8], img [32,128] → 128 tokens
  Decoder: 1-layer transformer with two-stream attention
    - Self-attention: position queries attend to context tokens
    - Cross-attention: attends to encoder output (image memory)
    - Pre-LN with norm_q, norm_c, norm1, norm2
  Head: Linear(embed_dim, 95) → 94 printable chars + EOS
  Charset: string.printable[:94] = digits + lower + upper + punctuation

Usage:
    python models/convert-parseq-to-gguf.py \
        --checkpoint /mnt/storage/models/parseq-bb5792a6.pt \
        --output /mnt/storage/gguf-models/parseq-f32.gguf

    python models/convert-parseq-to-gguf.py \
        --checkpoint /mnt/storage/models/parseq_tiny-e7a21b54.pt \
        --output /mnt/storage/gguf-models/parseq-tiny-f32.gguf --fp16
"""

import argparse
import string
import sys
from pathlib import Path

import gguf
import numpy as np


# PARSeq charset: 94 printable ASCII chars (same order as string.printable[:94])
PARSEQ_CHARSET = string.printable[:94]  # 0-9a-zA-Z + 32 punctuation
# Token ordering (from BaseTokenizer):
#   specials_first = (EOS,) → index 0
#   charset → indices 1..94
#   specials_last = (BOS, PAD) → indices 95, 96
# Head output: 95 classes = EOS(0) + 94 chars(1..94), excludes BOS and PAD
PARSEQ_TOKENS = ["[E]"] + list(PARSEQ_CHARSET) + ["[B]", "[P]"]


def main():
    p = argparse.ArgumentParser(description="Convert PARSeq to GGUF")
    p.add_argument("--checkpoint", required=True, help="Path to .pt file")
    p.add_argument("--output", required=True, help="Output GGUF path")
    p.add_argument("--fp16", action="store_true", help="Store in FP16")
    args = p.parse_args()

    # Load checkpoint
    sys.path = [pp for pp in sys.path if '.local' not in pp]
    for mod in list(sys.modules.keys()):
        if 'torch' in mod:
            del sys.modules[mod]
    import torch

    print(f"Loading checkpoint: {args.checkpoint}")
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    # It's a raw state dict (not wrapped in {"state_dict": ...})
    if "state_dict" in sd:
        sd = sd["state_dict"]

    print(f"Total tensors: {len(sd)}")

    # Infer architecture from weight shapes
    embed_dim = sd["encoder.norm.weight"].shape[0]
    n_enc_layers = max(int(k.split(".")[2]) for k in sd
                       if k.startswith("encoder.blocks.")) + 1
    n_heads_enc = embed_dim // 64  # head_dim=64 for base, 64 for tiny
    # Actually: base=384/6=64, tiny=192/3=64. Both use head_dim=64
    ffn_dim = sd["encoder.blocks.0.mlp.fc1.weight"].shape[0]
    patch_h, patch_w = sd["encoder.patch_embed.proj.weight"].shape[2:]
    n_patches = sd["encoder.pos_embed"].shape[1]  # 128
    max_label_len = sd["pos_queries"].shape[1]  # 26 (T+1 = 25+1)
    vocab_size = sd["head.weight"].shape[0]  # 95 (94 chars + EOS)
    n_tokens = sd["text_embed.embedding.weight"].shape[0]  # 97

    # Decoder dims
    dec_ffn = sd["decoder.layers.0.linear1.weight"].shape[0]
    dec_heads = embed_dim // 32  # decoder uses head_dim=32

    # Image size: n_patches = (H/patch_h) * (W/patch_w)
    # For base/tiny: 128 = 8*16 = (32/4)*(128/8) → H=32, W=128
    img_h = 32
    img_w = 128

    variant = "tiny" if embed_dim == 192 else "base"
    print(f"\nPARSeq-{variant}:")
    print(f"  embed_dim={embed_dim}, enc_layers={n_enc_layers}, enc_heads={n_heads_enc}")
    print(f"  ffn_dim={ffn_dim}, patch=[{patch_h},{patch_w}], img=[{img_h},{img_w}]")
    print(f"  n_patches={n_patches}, max_label_len={max_label_len}")
    print(f"  dec_heads={dec_heads}, dec_ffn={dec_ffn}")
    print(f"  vocab_size={vocab_size} (head output), n_tokens={n_tokens} (embedding)")
    print(f"  charset: {len(PARSEQ_CHARSET)} chars")

    # Write GGUF
    writer = gguf.GGUFWriter(str(args.output), arch="parseq")

    writer.add_string("general.name", f"parseq-{variant}-scene-text-ocr")
    writer.add_string("general.license", "Apache-2.0")
    writer.add_string("general.source", "baudm/parseq")

    # Encoder hparams
    writer.add_uint32("parseq.encoder.embed_dim", embed_dim)
    writer.add_uint32("parseq.encoder.num_layers", n_enc_layers)
    writer.add_uint32("parseq.encoder.num_heads", n_heads_enc)
    writer.add_uint32("parseq.encoder.ffn_dim", ffn_dim)
    writer.add_uint32("parseq.encoder.patch_h", patch_h)
    writer.add_uint32("parseq.encoder.patch_w", patch_w)
    writer.add_uint32("parseq.encoder.img_h", img_h)
    writer.add_uint32("parseq.encoder.img_w", img_w)
    writer.add_uint32("parseq.encoder.n_patches", n_patches)

    # Decoder hparams
    writer.add_uint32("parseq.decoder.num_heads", dec_heads)
    writer.add_uint32("parseq.decoder.ffn_dim", dec_ffn)
    writer.add_uint32("parseq.decoder.max_label_len", max_label_len)

    # Token IDs (PARSeq ordering: EOS=0, chars=1..94, BOS=95, PAD=96)
    # Head output has 95 classes: EOS=0, chars=1..94 (excludes BOS and PAD)
    writer.add_uint32("parseq.bos_token", 95)  # BOS in embedding space
    writer.add_uint32("parseq.eos_token", 0)   # EOS in both head and embedding space
    writer.add_uint32("parseq.pad_token", 96)
    writer.add_uint32("parseq.vocab_size", vocab_size)
    writer.add_uint32("parseq.n_tokens", n_tokens)

    # Tokenizer (character set)
    writer.add_array("tokenizer.tokens", PARSEQ_TOKENS)

    # Write tensors
    n_written = 0
    for name, tensor in sorted(sd.items()):
        data = tensor.numpy().astype(np.float32)

        # Determine GGUF type
        if args.fp16 and data.ndim >= 2 and data.size >= 256:
            data = data.astype(np.float16)
            dtype = gguf.GGMLQuantizationType.F16
        else:
            dtype = gguf.GGMLQuantizationType.F32

        # Flatten conv2d weights to 2D for quantization compatibility
        # patch_embed.proj.weight: [embed_dim, 3, patch_h, patch_w]
        orig_shape = data.shape
        if data.ndim == 4:
            data = data.reshape(data.shape[0], -1)  # [OC, IC*KH*KW]

        writer.add_tensor(name, data, raw_dtype=dtype)
        n_written += 1

        if n_written <= 5 or n_written % 50 == 0:
            print(f"  [{n_written:3d}] {name:55s} "
                  f"{str(list(orig_shape)):20s} → {str(list(data.shape)):20s} "
                  f"{'F16' if dtype == gguf.GGMLQuantizationType.F16 else 'F32'}")

    print(f"\nWrote {n_written} tensors")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    out_size = Path(args.output).stat().st_size
    print(f"Output: {args.output} ({out_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
