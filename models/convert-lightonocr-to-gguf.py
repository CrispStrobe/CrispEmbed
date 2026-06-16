#!/usr/bin/env python3
"""Convert LightOnOCR-2-1B to GGUF format.

Architecture: Pixtral ViT (24L, 1024d) + Qwen3 decoder (28L, 1024d).

    pip install torch safetensors gguf huggingface_hub
    python convert-lightonocr-to-gguf.py \
        --model lightonai/LightOnOCR-2-1B \
        --output lightonocr-1b-f16.gguf --dtype f16
"""

import argparse
import json
import os
import sys

import gguf
import numpy as np

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

ARCH = "lightonocr"


def f32(t) -> np.ndarray:
    if hasattr(t, 'detach'):
        return t.detach().float().cpu().numpy().astype(np.float32)
    return np.array(t, dtype=np.float32)


def f16(t) -> np.ndarray:
    if hasattr(t, 'detach'):
        return t.detach().float().cpu().numpy().astype(np.float16)
    return np.array(t, dtype=np.float16)


def main():
    parser = argparse.ArgumentParser(description="Convert LightOnOCR to GGUF")
    parser.add_argument("--model", required=True, help="HF model ID or local path")
    parser.add_argument("--output", required=True, help="Output .gguf path")
    parser.add_argument("--dtype", choices=["f16", "f32"], default="f16",
                        help="Weight dtype for large matrices (default: f16)")
    args = parser.parse_args()

    wt = f16 if args.dtype == "f16" else f32

    print(f"Loading model: {args.model}")

    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from pathlib import Path

    # Load config
    model_path = Path(args.model)
    if model_path.is_dir():
        with open(model_path / "config.json") as f:
            config = json.load(f)
        sf_path = str(model_path / "model.safetensors")
    else:
        cfg_path = hf_hub_download(args.model, "config.json",
            cache_dir='/mnt/akademie_storage/huggingface/hub/')
        with open(cfg_path) as f:
            config = json.load(f)
        sf_path = hf_hub_download(args.model, "model.safetensors",
            cache_dir='/mnt/akademie_storage/huggingface/hub/')

    vc = config["vision_config"]
    tc = config["text_config"]

    print(f"  Vision: {vc['num_hidden_layers']}L, {vc['hidden_size']}d, {vc['num_attention_heads']}H")
    print(f"  Text:   {tc['num_hidden_layers']}L, {tc['hidden_size']}d, {tc['num_attention_heads']}H/{tc['num_key_value_heads']}KV")
    print(f"  Vocab:  {tc['vocab_size']}")

    writer = gguf.GGUFWriter(str(args.output), arch=ARCH)

    # ── Metadata ──
    # Vision
    writer.add_uint32("lightonocr.vision.num_hidden_layers", vc["num_hidden_layers"])
    writer.add_uint32("lightonocr.vision.hidden_size", vc["hidden_size"])
    writer.add_uint32("lightonocr.vision.num_attention_heads", vc["num_attention_heads"])
    writer.add_uint32("lightonocr.vision.intermediate_size", vc["intermediate_size"])
    writer.add_uint32("lightonocr.vision.head_dim", vc.get("head_dim", 64))
    writer.add_uint32("lightonocr.vision.patch_size", vc["patch_size"])
    writer.add_uint32("lightonocr.vision.image_size", vc["image_size"])
    writer.add_float32("lightonocr.vision.rope_theta", vc.get("rope_theta", 10000.0))
    # Text (Qwen3)
    writer.add_uint32("lightonocr.text.vocab_size", tc["vocab_size"])
    writer.add_uint32("lightonocr.text.num_hidden_layers", tc["num_hidden_layers"])
    writer.add_uint32("lightonocr.text.hidden_size", tc["hidden_size"])
    writer.add_uint32("lightonocr.text.num_attention_heads", tc["num_attention_heads"])
    writer.add_uint32("lightonocr.text.num_key_value_heads", tc["num_key_value_heads"])
    writer.add_uint32("lightonocr.text.intermediate_size", tc["intermediate_size"])
    writer.add_uint32("lightonocr.text.head_dim", tc.get("head_dim", 128))
    writer.add_float32("lightonocr.text.rms_norm_eps", tc.get("rms_norm_eps", 1e-6))
    writer.add_float32("lightonocr.text.rope_theta", tc.get("rope_theta", 1000000.0))
    writer.add_bool("lightonocr.text.use_qk_norm", tc.get("use_qk_norm", True))
    # General
    writer.add_uint32("lightonocr.spatial_merge_size", config.get("spatial_merge_size", 2))
    writer.add_uint32("lightonocr.image_token_id", config.get("image_token_id", 151655))
    writer.add_uint32("lightonocr.eos_token_id", config.get("eos_token_id", 151645))
    writer.add_uint32("lightonocr.pad_token_id", config.get("pad_token_id", 151643))

    # ── Tensors (lazy loading) ──
    n_vis = vc["num_hidden_layers"]
    n_txt = tc["num_hidden_layers"]

    with safe_open(sf_path, framework="pt") as sf:
        tensor_names = sorted(sf.keys())
        total = 0

        for name in tensor_names:
            t = sf.get_tensor(name)

            # Map HF names to GGUF names
            gguf_name = name
            # Strip model. prefix
            if gguf_name.startswith("model."):
                gguf_name = gguf_name[6:]

            # Vision encoder
            gguf_name = gguf_name.replace("vision_encoder.ln_pre.", "vis.ln_pre.")
            gguf_name = gguf_name.replace("vision_encoder.patch_conv.", "vis.patch_conv.")
            gguf_name = gguf_name.replace("vision_encoder.transformer.layers.", "vis.blk.")
            gguf_name = gguf_name.replace(".attention.", ".attn.")
            gguf_name = gguf_name.replace(".attention_norm.", ".attn_norm.")
            gguf_name = gguf_name.replace(".feed_forward.", ".ffn.")
            gguf_name = gguf_name.replace(".ffn_norm.", ".ffn_norm.")

            # Projection
            gguf_name = gguf_name.replace("vision_projection.", "proj.")

            # LM decoder
            gguf_name = gguf_name.replace("language_model.embed_tokens.", "lm.embed.")
            gguf_name = gguf_name.replace("language_model.layers.", "lm.blk.")
            gguf_name = gguf_name.replace("language_model.norm.", "lm.norm.")
            gguf_name = gguf_name.replace(".self_attn.", ".attn.")
            gguf_name = gguf_name.replace(".input_layernorm.", ".attn_norm.")
            gguf_name = gguf_name.replace(".post_attention_layernorm.", ".ffn_norm.")
            gguf_name = gguf_name.replace(".mlp.", ".ffn.")

            # Decide dtype: keep norms/embeds at f32, large matrices at wt
            is_small = (t.numel() < 1024 * 1024) or "norm" in gguf_name or "embed" in gguf_name
            data = f32(t) if is_small else wt(t)

            writer.add_tensor(gguf_name, data)
            total += 1

        print(f"  Wrote {total} tensors")

    # ── Tokenizer ──
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        vocab = [tokenizer.convert_ids_to_tokens(i) or f"<tok_{i}>" for i in range(tc["vocab_size"])]
        writer.add_token_list(vocab)
        writer.add_uint32("tokenizer.ggml.bos_token_id", tokenizer.bos_token_id or 0)
        writer.add_uint32("tokenizer.ggml.eos_token_id", config.get("eos_token_id", 151645))
        writer.add_uint32("tokenizer.ggml.padding_token_id", config.get("pad_token_id", 151643))
        print(f"  Tokenizer: {len(vocab)} tokens")
    except Exception as e:
        print(f"  WARNING: tokenizer failed: {e}")

    # Write
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\nWrote {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
