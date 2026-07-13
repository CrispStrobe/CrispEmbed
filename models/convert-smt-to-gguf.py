#!/usr/bin/env python3
"""Convert Sheet Music Transformer (SMT / SMT++) safetensors to GGUF — NO PyTorch.

Optical Music Recognition: staff-notation image → bekern token sequence.
Reads model.safetensors + config.json directly (torch-free) and packs into a
single GGUF for CrispEmbed's smt_ocr inference engine.

The published PRAIG weights (antoniorv6/smt-grandstaff, -camera-grandstaff)
match the **SMT-plusplus** module naming
(`input_attention`/`cross_attention`/`ffNet`/`out_layer`), NOT the SMT-main
rewrite. This converter targets those verbatim names.

Architecture (from SMT-plusplus/smt_model/modeling_smt.py, read line-by-line):
  Encoder : stock HF ConvNextModel(num_channels=1, num_stages=3,
            hidden_sizes=[64,128,256], depths=[3,3,9]). 16x H/W reduction.
            last_hidden_state (pre-pooler-LN) feeds the decoder at 256 = d_model.
  Decoder : 8 layers, d_model=256, 4 heads, dim_ff=256 (1x), ReLU FFN,
            post-norm (self-attn -> norm1 -> cross-attn -> norm2 -> FFN -> norm3),
            sinusoidal 1D pos-enc on tokens, LM head = Conv1d(256->V,k1).
  Cross-attn: memory_key = encoder features + 2D sinusoidal PE (keys only);
              memory_value = raw encoder features. QK^T is UNSCALED (the model's
              scale_factor is defined but never applied in MHA.forward).

Baked here (not in the checkpoint): the 1-D decoder positional-encoding table.
The 2-D encoder PE is image-size dependent (full table would be ~800 MB) so the
C++ engine generates it at runtime from the same formula.

Dependencies: safetensors, gguf, numpy  (no torch / transformers)

Usage:
    python convert-smt-to-gguf.py --model-dir /path/to/smt-grandstaff \
        --output smt-grandstaff-f32.gguf [--fp16]
"""

import argparse
import json
import sys
from pathlib import Path

import gguf
import numpy as np
from safetensors import safe_open

ARCH = "smt_ocr"


# The published weights come in two naming schemes for the SAME network:
#   • "smt-plusplus" (antoniorv6/smt-grandstaff): decoder layers use
#     input_attention.l{q,k,v}/out_proj, cross_attention.*, ffNet.0/3,
#     norm1/2/3; LM head = decoder.out_layer (Conv1d k1). Attention is UNSCALED
#     and a ReLU precedes the head. This is what src/smt_ocr.cpp expects.
#   • "smt-main" (antoniorv6/SMT rewrite, incl. PRAIG/smt-fp-grandstaff full
#     page): decoder layers use self_attn.{q,k,v,out}_proj,
#     cross_attn.{q,k,v,out}_proj, ffn.0/3, norm_layers.0/1/2; LM head =
#     decoder.vocab_projection (plain Linear). Attention IS scaled by
#     d_head**-0.5 and there is NO ReLU before the head (read line-by-line from
#     SMT/smt_model/modeling_smt.py).
# We normalise smt-main tensor names to the smt-plusplus names the engine loads,
# and emit behaviour flags so the C++ engine applies the right forward.
_DEC = "decoder.decoder.layers."


def detect_scheme(tensor_names):
    keys = set(tensor_names)
    if any(".self_attn.q_proj." in k for k in keys) or "decoder.vocab_projection.weight" in keys:
        return "smt-main"
    return "smt-plusplus"


def map_name_smt_main(name):
    """Rename an smt-main tensor to the smt-plusplus name the engine loads."""
    if name == "decoder.vocab_projection.weight":
        return "decoder.out_layer.weight"
    if name == "decoder.vocab_projection.bias":
        return "decoder.out_layer.bias"
    if name.startswith(_DEC):
        rest = name[len(_DEC):]
        idx, suf = rest.split(".", 1)
        sub = {
            "self_attn.q_proj": "input_attention.lq",
            "self_attn.k_proj": "input_attention.lk",
            "self_attn.v_proj": "input_attention.lv",
            "self_attn.out_proj": "input_attention.out_proj",
            "cross_attn.q_proj": "cross_attention.lq",
            "cross_attn.k_proj": "cross_attention.lk",
            "cross_attn.v_proj": "cross_attention.lv",
            "cross_attn.out_proj": "cross_attention.out_proj",
            "ffn.0": "ffNet.0",
            "ffn.3": "ffNet.3",
            "norm_layers.0": "norm1",
            "norm_layers.1": "norm2",
            "norm_layers.2": "norm3",
        }
        for src, dst in sub.items():
            if suf.startswith(src + "."):
                return _DEC + idx + "." + dst + suf[len(src):]
        return name  # encoder-style / already-mapped
    return name  # encoder.* names are identical across schemes


def build_positional_1d(dim, len_max):
    """Mirror PositionalEncoding1D (modeling_smt.py:38-49).

    pe stored (1, dim, len_max): pe[::2]=sin(l*div), pe[1::2]=cos(l*div),
    div = exp(-arange(0,dim,2)/dim * ln(10000)).  Baked here as (len_max, dim)
    row-major so C++ adds pe[l] to token position l directly.
    """
    pe = np.zeros((len_max, dim), dtype=np.float32)
    div = np.exp(-np.arange(0.0, dim, 2) / dim * np.log(10000.0))  # (dim/2,)
    l_pos = np.arange(0.0, len_max)[:, None]  # (len_max, 1)
    pe[:, 0::2] = np.sin(l_pos * div[None, :])
    pe[:, 1::2] = np.cos(l_pos * div[None, :])
    return pe


def main():
    p = argparse.ArgumentParser(description="Convert SMT/SMT++ safetensors to GGUF")
    p.add_argument("--model-dir", required=True, help="Model dir (config.json + model.safetensors)")
    p.add_argument("--output", required=True, help="Output GGUF path")
    p.add_argument("--fp16", action="store_true", help="Store 2-D weights in FP16 (norms/bias/PE stay F32)")
    p.add_argument("--name", default=None, help="general.name metadata override")
    args = p.parse_args()

    model_dir = Path(args.model_dir)

    # ---- config (raw dict; embeds w2i/i2w) ----
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        print(f"Error: {cfg_path} not found", file=sys.stderr)
        return 1
    with open(cfg_path) as f:
        cfg = json.load(f)

    d_model = int(cfg.get("d_model", 256))
    dim_ff = int(cfg.get("dim_ff", 256))
    num_dec_layers = int(cfg.get("num_dec_layers", 8))
    num_heads = int(cfg.get("num_attn_heads", 4))
    maxlen = int(cfg.get("maxlen", 1281))
    maxh = int(cfg.get("maxh", 256))
    maxw = int(cfg.get("maxw", 3056))
    in_channels = int(cfg.get("in_channels", 1))
    out_categories = int(cfg.get("out_categories", 20578))
    pad_token = int(cfg.get("padding_token", 0))

    # ConvNext encoder shape is fixed by the model class (not config)
    enc_hidden_sizes = [64, 128, 256]
    enc_depths = [3, 3, 9]
    enc_num_stages = 3
    enc_stem_kernel = 4  # stem conv kernel/stride
    enc_reduction = 16   # total H/W downsample

    # ---- vocab (w2i -> tokens[idx] = word) ----
    w2i = cfg.get("w2i", {})
    if not w2i:
        print("Error: config.json has no w2i vocab", file=sys.stderr)
        return 1
    vocab_n = max(int(v) for v in w2i.values()) + 1
    tokens = [""] * vocab_n
    for word, idx in w2i.items():
        idx = int(idx)
        if 0 <= idx < vocab_n:
            tokens[idx] = word
    bos = int(w2i.get("<bos>", 0))
    eos = int(w2i.get("<eos>", 0))
    pad = int(w2i.get("<pad>", pad_token))
    if vocab_n != out_categories:
        print(f"WARNING: vocab_n={vocab_n} != out_categories={out_categories}")

    print(f"Encoder: ConvNext {enc_num_stages} stages, sizes={enc_hidden_sizes}, "
          f"depths={enc_depths}, in_ch={in_channels}, {enc_reduction}x reduction")
    print(f"Decoder: {num_dec_layers}L / {num_heads}H / d_model={d_model} / ff={dim_ff}, "
          f"vocab={vocab_n}, maxlen={maxlen}")
    print(f"Tokens : bos={bos} eos={eos} pad={pad}")

    # ---- weights ----
    st_path = model_dir / "model.safetensors"
    if not st_path.exists():
        print(f"Error: {st_path} not found", file=sys.stderr)
        return 1
    print(f"Loading {st_path} ...")
    raw = {}
    with safe_open(str(st_path), framework="numpy") as f:
        for k in f.keys():
            raw[k] = f.get_tensor(k)
    print(f"Loaded {len(raw)} tensors")

    scheme = detect_scheme(raw.keys())
    print(f"Naming scheme: {scheme}")
    if scheme == "smt-main":
        tensors = {map_name_smt_main(k): v for k, v in raw.items()}
        if len(tensors) != len(raw):
            print("Error: name collision during smt-main remap", file=sys.stderr)
            return 1
    else:
        tensors = raw

    # out_layer is a 1x1 Conv1d [V, d_model, 1] -> squeeze to Linear [V, d_model].
    # (smt-main's vocab_projection is already Linear [V, d_model]; no squeeze.)
    ol = "decoder.out_layer.weight"
    if ol in tensors and tensors[ol].ndim == 3 and tensors[ol].shape[-1] == 1:
        tensors[ol] = tensors[ol][:, :, 0]
        print(f"  Squeezed {ol} -> {tensors[ol].shape}")

    # Behaviour flags — smt-main scales attention (d_head**-0.5) and applies the
    # LM head WITHOUT a preceding ReLU; smt-plusplus is unscaled with a pre-head
    # ReLU. Preprocessing: smt-main full-page finetune uses reduce_ratio=1.0 and
    # RandomInvert(p=1.0)+Grayscale (SMT/data_augmentation convert_img_to_tensor);
    # smt-plusplus base uses the legacy clamp policy (reduce_ratio<=0 sentinel).
    if scheme == "smt-main":
        scale_attention = True
        head_pre_relu = False
        preproc_reduce_ratio = 1.0
        preproc_invert = True
    else:
        scale_attention = False
        head_pre_relu = True
        preproc_reduce_ratio = 0.0  # sentinel: engine uses legacy clamp policy
        preproc_invert = False

    # ---- write GGUF ----
    print(f"Writing GGUF -> {args.output}")
    w = gguf.GGUFWriter(args.output, ARCH)
    w.add_name(args.name or "Sheet Music Transformer (SMT) OMR")
    w.add_description("Sheet Music Transformer: staff-notation image -> bekern tokens")
    w.add_string("general.license", "MIT")

    # encoder hparams
    w.add_uint32("smt.encoder.num_stages", enc_num_stages)
    w.add_array("smt.encoder.hidden_sizes", enc_hidden_sizes)
    w.add_array("smt.encoder.depths", enc_depths)
    w.add_uint32("smt.encoder.num_channels", in_channels)
    w.add_uint32("smt.encoder.stem_kernel", enc_stem_kernel)
    w.add_uint32("smt.encoder.reduction", enc_reduction)
    # decoder hparams
    w.add_uint32("smt.decoder.num_layers", num_dec_layers)
    w.add_uint32("smt.decoder.d_model", d_model)
    w.add_uint32("smt.decoder.num_heads", num_heads)
    w.add_uint32("smt.decoder.dim_ff", dim_ff)
    w.add_uint32("smt.decoder.vocab_size", vocab_n)
    w.add_uint32("smt.decoder.maxlen", maxlen)
    w.add_uint32("smt.decoder.maxh", maxh)
    w.add_uint32("smt.decoder.maxw", maxw)
    # special tokens
    w.add_uint32("smt.bos_token_id", bos)
    w.add_uint32("smt.eos_token_id", eos)
    w.add_uint32("smt.pad_token_id", pad)
    # Forward-behaviour flags (see detect_scheme / map_name_smt_main above)
    w.add_bool("smt.scale_attention", scale_attention)
    w.add_bool("smt.head_pre_relu", head_pre_relu)
    w.add_float32("smt.preproc.reduce_ratio", preproc_reduce_ratio)
    w.add_bool("smt.preproc.invert", preproc_invert)
    # tokenizer
    w.add_array("tokenizer.tokens", tokens)

    # baked 1-D decoder positional encoding (len_max, d_model)
    pe1d = build_positional_1d(d_model, maxlen)
    w.add_tensor("smt.positional_1d", pe1d, raw_dtype=gguf.GGMLQuantizationType.F32)
    print(f"  Baked smt.positional_1d {pe1d.shape}")

    # model weights (verbatim HF names). Keep norms/bias/conv/small tensors F32;
    # optionally fp16 the big 2-D matmul weights.
    n_written = 0
    for name in sorted(tensors.keys()):
        data = tensors[name]
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        is_norm_or_bias = name.endswith(".bias") or "layernorm" in name or "norm" in name \
            or name.endswith("layer_scale_parameter")
        if args.fp16 and data.ndim == 2 and not is_norm_or_bias:
            data = data.astype(np.float16)
        w.add_tensor(name, data)
        n_written += 1

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    import os
    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"Done: {args.output} ({size_mb:.1f} MB, {n_written} weight tensors + 1 baked PE)")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
