#!/usr/bin/env python3
"""Convert an official PP-OCRv6 safetensors model to CrispEmbed GGUF.

The official PP-OCRv6 Hugging Face repositories contain the inference graph
already translated to Transformers naming.  This converter deliberately keeps
that graph's tensor names stable, folds inference BatchNorm into its preceding
convolution, and writes one GGUF per detector/recognizer size.

Supported model names:
  PP-OCRv6_{tiny,small,medium}_{det,rec}

Large source and output files should live on the external model volume, e.g.:
  python models/convert-ppocrv6-to-gguf.py \
    --model-dir /Volumes/backups/ai/crispembed-gguf/source/PP-OCRv6_small_rec_safetensors \
    --output /Volumes/backups/ai/crispembed-gguf/PP-OCRv6_small_rec-f16.gguf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gguf
import numpy as np
from safetensors import safe_open


def _fuse(w: np.ndarray, b: np.ndarray | None, gamma: np.ndarray,
          beta: np.ndarray, mean: np.ndarray, var: np.ndarray,
          eps: float, channel_axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    scale = gamma / np.sqrt(var + eps)
    shape = [1] * w.ndim
    shape[channel_axis] = scale.shape[0]
    centered_bias = (np.zeros_like(mean) if b is None else b) - mean
    return w * scale.reshape(shape), centered_bias * scale + beta


def _tokens(model_dir: Path, kind: str) -> list[str]:
    # inference.yml is shipped by the official repositories and is the
    # driving post-process configuration, including the exact dictionary.
    try:
        import yaml
        doc = yaml.safe_load((model_dir / "inference.yml").read_text())
        chars = doc.get("PostProcess", {}).get("character_dict", [])
        if chars:
            tokens = [str(x) for x in chars]
            # The PP-OCRv6 training configs set use_space_char, so PaddleOCR's
            # BaseRecLabelDecode appends ' ' to the dictionary before
            # CTCLabelDecode prepends 'blank'.  The label list is therefore
            # blank + dict + ' ', which is exactly the head_out_channels the
            # checkpoint carries (18710 against an 18708-entry dict).  Emitting
            # only the dict leaves the last class decoding to nothing, so every
            # space is silently dropped and the page comes out run-on.
            if " " not in tokens:
                tokens.append(" ")
            return tokens
    except (ImportError, OSError, ValueError):
        pass
    return []


def _short_name(name: str) -> str:
    """Keep GGUF tensor names below ggml's 64-byte name limit."""
    replacements = (
        ("backbone.encoder.convolution", "bb.stem"),
        ("backbone.encoder.blocks", "bb.blk"),
        ("neck.intraclass_blocks", "nk.ic"),
        ("horizontal_small_to_long_conv_longratio", "hsl"),
        ("horizontal_small_to_long_conv_midratio", "hsm"),
        ("horizontal_small_to_long_conv_shortratio", "hss"),
        ("vertical_long_to_small_conv_longratio", "vll"),
        ("vertical_long_to_small_conv_midratio", "vlm"),
        ("vertical_long_to_small_conv_shortratio", "vls"),
        ("symmetric_conv_long_longratio", "sll"),
        ("symmetric_conv_long_midratio", "slm"),
        ("symmetric_conv_long_shortratio", "sls"),
        (".blocks.", ".b."),
        (".channel_conv1.convolution", ".cm1"),
        (".channel_conv2.convolution", ".cm2"),
        (".token_squeeze_excitation.convolutions.0", ".se1"),
        (".token_squeeze_excitation.convolutions.2", ".se2"),
        (".token_conv.convolution", ".dw"),
        (".token_conv", ".dw"),
        (".convolution", ".conv"),
        (".normalization", ".norm"),
    )
    for old, new in replacements:
        name = name.replace(old, new)
    return name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--model-name", help="Override the model name from config.json")
    ap.add_argument("--fp32", action="store_true", help="Keep floating tensors as F32")
    args = ap.parse_args()

    cfg = json.loads((args.model_dir / "config.json").read_text())
    model_name = args.model_name or cfg["model_type"].replace("pp_ocrv6_", "PP-OCRv6_")
    if not model_name.startswith("PP-OCRv6_") or not model_name.endswith(("_det", "_rec")):
        raise SystemExit(f"unsupported PP-OCRv6 model name: {model_name}")
    kind = "det" if model_name.endswith("_det") else "rec"
    variant = model_name.split("_")[1]
    st_path = args.model_dir / "model.safetensors"
    if not st_path.exists():
        raise SystemExit(f"missing {st_path}")

    with safe_open(str(st_path), framework="numpy") as src:
        names = list(src.keys())
        data = {name: src.get_tensor(name).astype(np.float32, copy=False) for name in names}

    # Fuse all Conv/BN pairs.  The official inference graph exposes running
    # statistics, so leaving BN as a runtime op would needlessly complicate the
    # C++ graph and makes quantization more fragile.
    fused: dict[str, np.ndarray] = {}
    consumed: set[str] = set()
    eps = 1e-5
    for name, value in data.items():
        if not name.endswith(".convolution.weight"):
            continue
        stem = name[:-len("convolution.weight")]
        # Backbone layers use `.normalization`; the detector neck/head uses
        # `.norm`.  Both are inference BatchNorm and must be folded before
        # writing GGUF so detector and recognizer share the same runtime path.
        norm = stem + ("normalization." if stem + "normalization.weight" in data else "norm.")
        if all(norm + x in data for x in ("weight", "bias", "running_mean", "running_var")):
            bias_name = stem + "convolution.bias"
            # Paddle Conv2DTranspose kernels are [in, out, kh, kw], while
            # ordinary convolutions are [out, in, kh, kw].  Its BN therefore
            # scales axis 1, not axis 0.
            transpose_bn = "head.conv_up.convolution.weight" in name
            w, b = _fuse(value, data.get(bias_name), data[norm + "weight"],
                         data[norm + "bias"], data[norm + "running_mean"],
                         data[norm + "running_var"], eps, 1 if transpose_bn else 0)
            fused[name] = w
            fused[bias_name] = b
            consumed.update({name, bias_name, norm + "weight", norm + "bias",
                             norm + "running_mean", norm + "running_var"})

    for name, value in data.items():
        if name in consumed:
            continue
        # A few original Paddle layers use `token_conv.weight` + BN naming
        # rather than the Transformers `convolution` spelling.  Their BN is
        # retained until the C++ loader sees the exact pair.
        fused[name] = value

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(args.output), arch="ppocrv6")
    writer.add_string("general.name", model_name)
    writer.add_string("general.license", "Apache-2.0")
    writer.add_string("general.source", f"PaddlePaddle/{model_name}_safetensors")
    writer.add_string("ppocrv6.kind", kind)
    writer.add_string("ppocrv6.variant", variant)
    writer.add_uint32("ppocrv6.fused_batch_norm", 1)
    writer.add_uint32("ppocrv6.input_height", 48 if kind == "rec" else 0)
    writer.add_uint32("ppocrv6.input_width", 320 if kind == "rec" else 0)
    tokens = _tokens(args.model_dir, kind)
    # Use the standard GGUF tokenizer key consumed by CrispEmbed.  Keep the
    # short legacy alias for tools that used the first prototype converter.
    writer.add_array("tokenizer.ggml.tokens", tokens)
    writer.add_array("tokenizer.tokens", tokens)
    if kind == "rec":
        writer.add_uint32("ppocrv6.vocab_size", int(cfg.get("head_out_channels", 0)))
        writer.add_uint32("ppocrv6.hidden_size", int(cfg.get("hidden_size", 0)))

    dtype = gguf.GGMLQuantizationType.F32 if args.fp32 else gguf.GGMLQuantizationType.F16
    count = 0
    for name in sorted(fused):
        value = fused[name]
        # GGUF tensors are named without the HF wrapper's leading `model.`.
        out_name = name[6:] if name.startswith("model.") else name
        out_name = _short_name(out_name)
        # Preserve row-major flat storage for convolution kernels.  The C++
        # runtime interprets [out, in, kh, kw] in this exact order.
        if value.ndim == 4:
            value = value.reshape(value.shape[0], -1)
        # The DB/SVTR output head is the most sensitive part of this small
        # CNN/CTC family.  Keep it in F32 even for the compact F16 artifact;
        # this is the same critical-weight policy used by the other GGUF
        # converters and prevents threshold/logit drift from compounding at
        # the final output.
        critical = out_name.startswith("head.")
        raw = value if args.fp32 or critical or value.ndim < 2 else value.astype(np.float16)
        raw_dtype = gguf.GGMLQuantizationType.F32 if args.fp32 or critical or raw.ndim < 2 else dtype
        writer.add_tensor(f"{kind}.{out_name}", raw, raw_dtype=raw_dtype)
        count += 1
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {args.output} ({count} tensors, {args.output.stat().st_size / 1048576:.1f} MiB)")


if __name__ == "__main__":
    main()
