#!/usr/bin/env python3
"""Convert GLiNER DeBERTa-v3 (urchade/gliner_medium-v2.1) → GGUF.

Architecture:
  DeBERTa-v3-base encoder (12L, 768 hidden, disentangled attention)
  + Linear projection (768 → 512)
  + BiLSTM (1-layer bidirectional, hidden=256, output=512)
  + GLiNER head: span_rep (markerV0, start+end only) + prompt_rep + dot-product scorer

Usage:
  python models/convert-gliner-deberta-to-gguf.py \
      --model /mnt/volume1/hf-cache/gliner-medium-v2.1 \
      --tokenizer microsoft/deberta-v3-base \
      --output gliner-deberta-f32.gguf
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "ggml" / "scripts"))
try:
    import gguf
except ImportError:
    print("ERROR: gguf package not found. pip install gguf", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# GGUF tensor naming
# ---------------------------------------------------------------------------

# Maps HF GLiNER DeBERTa state_dict keys to GGUF tensor names.
# DeBERTa encoder uses CrispEmbed-compatible naming (enc.{i}.attn.q.weight etc.)
# GLiNER head uses same naming as the LFM variant.

def remap_tensor_name(hf_name: str) -> str | None:
    """Map HuggingFace state_dict key → GGUF tensor name. Return None to skip."""
    n = hf_name

    # --- DeBERTa-v3 encoder ---
    prefix = "token_rep_layer.bert_layer.model."
    if n.startswith(prefix):
        rest = n[len(prefix):]

        # Embeddings
        if rest == "embeddings.word_embeddings.weight":
            return "token_embd.weight"
        if rest == "embeddings.LayerNorm.weight":
            return "embd_ln.weight"
        if rest == "embeddings.LayerNorm.bias":
            return "embd_ln.bias"

        # Encoder-level LayerNorm (applied to relative embeddings)
        if rest == "encoder.LayerNorm.weight":
            return "encoder_ln.weight"
        if rest == "encoder.LayerNorm.bias":
            return "encoder_ln.bias"

        # Relative position embeddings
        if rest == "encoder.rel_embeddings.weight":
            return "rel_embd.weight"

        # Per-layer weights
        if rest.startswith("encoder.layer."):
            parts = rest.split(".", 3)  # encoder, layer, idx, remainder
            idx = parts[2]
            remainder = parts[3]

            # Attention Q/K/V (DeBERTa uses query_proj/key_proj/value_proj)
            attn_map = {
                "attention.self.query_proj.weight": f"enc.{idx}.attn.q.weight",
                "attention.self.query_proj.bias":   f"enc.{idx}.attn.q.bias",
                "attention.self.key_proj.weight":   f"enc.{idx}.attn.k.weight",
                "attention.self.key_proj.bias":     f"enc.{idx}.attn.k.bias",
                "attention.self.value_proj.weight":  f"enc.{idx}.attn.v.weight",
                "attention.self.value_proj.bias":    f"enc.{idx}.attn.v.bias",
                "attention.output.dense.weight":     f"enc.{idx}.attn.o.weight",
                "attention.output.dense.bias":       f"enc.{idx}.attn.o.bias",
                "attention.output.LayerNorm.weight":  f"enc.{idx}.ln1.weight",
                "attention.output.LayerNorm.bias":    f"enc.{idx}.ln1.bias",
                "intermediate.dense.weight":          f"enc.{idx}.ffn.fc1.weight",
                "intermediate.dense.bias":            f"enc.{idx}.ffn.fc1.bias",
                "output.dense.weight":                f"enc.{idx}.ffn.fc2.weight",
                "output.dense.bias":                  f"enc.{idx}.ffn.fc2.bias",
                "output.LayerNorm.weight":            f"enc.{idx}.ln2.weight",
                "output.LayerNorm.bias":              f"enc.{idx}.ln2.bias",
            }
            if remainder in attn_map:
                return attn_map[remainder]

        # Skip position_embeddings / token_type_embeddings (DeBERTa-v3 doesn't use)
        if "position_embeddings" in rest or "token_type_embeddings" in rest:
            return None

    # --- Projection (768 → 512) ---
    if n == "token_rep_layer.projection.weight":
        return "projection.weight"
    if n == "token_rep_layer.projection.bias":
        return "projection.bias"

    # --- BiLSTM (rnn.lstm) ---
    if n.startswith("rnn.lstm."):
        rest = n[len("rnn.lstm."):]
        return f"lstm.{rest}"

    # --- GLiNER span representation (markerV0: start+end only, no first) ---
    span_prefix = "span_rep_layer.span_rep_layer."
    if n.startswith(span_prefix):
        rest = n[len(span_prefix):]
        return f"span.{rest}"

    # --- Prompt/entity representation ---
    if n.startswith("prompt_rep_layer."):
        rest = n[len("prompt_rep_layer."):]
        return f"prompt_rep.{rest}"

    # --- Scorer temperature (not present in this model) ---
    if n == "log_score_temperature":
        return "scorer.log_temperature"

    print(f"  WARN: unmapped tensor: {n}", file=sys.stderr)
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert GLiNER DeBERTa-v3 to GGUF")
    parser.add_argument("--model", required=True, help="Local path to GLiNER model dir")
    parser.add_argument("--tokenizer", default="microsoft/deberta-v3-base",
                        help="DeBERTa tokenizer (HF model ID or local path)")
    parser.add_argument("--output", required=True, help="Output GGUF path")
    parser.add_argument("--dtype", default="f32", choices=["f32", "f16"],
                        help="Storage dtype (default: f32)")
    args = parser.parse_args()

    model_dir = args.model
    if not os.path.isdir(model_dir):
        print(f"ERROR: model dir not found: {model_dir}", file=sys.stderr)
        sys.exit(1)

    # Load GLiNER config
    config_path = os.path.join(model_dir, "gliner_config.json")
    with open(config_path) as f:
        config = json.load(f)

    # DeBERTa-v3-base hyperparameters
    hidden_size = 768
    n_layers = 12
    n_heads = 12
    intermediate_size = 3072
    max_position = 512
    position_buckets = 256
    layer_norm_eps = 1e-7  # DeBERTa default

    # GLiNER head params
    gliner_hidden = config.get("hidden_size", 512)
    max_width = config.get("max_width", 12)
    span_mode = config.get("span_mode", "markerV0")

    print(f"Model: {args.model}")
    print(f"  DeBERTa: hidden={hidden_size}, layers={n_layers}, heads={n_heads}")
    print(f"  GLiNER: hidden={gliner_hidden}, max_width={max_width}, span_mode={span_mode}")

    # Determine dtype
    if args.dtype == "f16":
        np_dtype = np.float16
        gguf_dtype = gguf.GGMLQuantizationType.F16
    else:
        np_dtype = np.float32
        gguf_dtype = gguf.GGMLQuantizationType.F32

    # --- Load SentencePiece tokenizer from DeBERTa-v3-base ---
    import sentencepiece as spm

    tok_source = args.tokenizer
    if os.path.isdir(tok_source):
        spm_path = os.path.join(tok_source, "spm.model")
    else:
        from huggingface_hub import hf_hub_download
        cache_dir = os.environ.get("HF_HUB_CACHE",
                                   "/mnt/akademie_storage/huggingface/hub")
        spm_path = hf_hub_download(tok_source, "spm.model", cache_dir=cache_dir)

    sp = spm.SentencePieceProcessor()
    sp.Load(spm_path)
    spm_vocab_size = sp.GetPieceSize()  # 128000

    # GLiNER adds 3 special tokens: [FLERT], <<ENT>>, <<SEP>>
    # After resize: vocab = spm_vocab_size + 3 + 1 (padding for alignment)
    ent_token_id = spm_vocab_size + 1   # 128001
    sep_token_id = spm_vocab_size + 2   # 128002

    print(f"  SPM vocab: {spm_vocab_size}, <<ENT>>={ent_token_id}, <<SEP>>={sep_token_id}")

    # --- Initialize GGUF writer ---
    writer = gguf.GGUFWriter(args.output, arch="gliner")

    # General metadata
    writer.add_string("general.architecture", "gliner")
    writer.add_string("general.name", "GLiNER-medium-v2.1-DeBERTa-v3")
    writer.add_string("general.license", "apache-2.0")
    writer.add_string("general.source", "urchade/gliner_medium-v2.1")

    # Backbone type
    writer.add_string("gliner.backbone", "deberta")

    # DeBERTa encoder hyperparameters (using bert.* prefix for crispembed compat)
    writer.add_uint32("bert.hidden_size", hidden_size)
    writer.add_uint32("bert.num_hidden_layers", n_layers)
    writer.add_uint32("bert.num_attention_heads", n_heads)
    writer.add_uint32("bert.intermediate_size", intermediate_size)
    writer.add_uint32("bert.max_position_embeddings", max_position)
    writer.add_uint32("bert.position_buckets", position_buckets)
    writer.add_float32("bert.layer_norm_eps", layer_norm_eps)

    # GLiNER head params
    writer.add_uint32("gliner.hidden_size", gliner_hidden)
    writer.add_uint32("gliner.max_width", max_width)
    writer.add_string("gliner.span_mode", span_mode)
    writer.add_uint32("gliner.ent_token_id", ent_token_id)
    writer.add_uint32("gliner.sep_token_id", sep_token_id)

    # --- Tokenizer ---
    # Build vocab: SPM tokens + GLiNER special tokens
    total_vocab = spm_vocab_size + 4  # match embedding size [128004, 768]
    tokens = []
    scores = []
    for i in range(spm_vocab_size):
        tokens.append(sp.IdToPiece(i))
        scores.append(sp.GetScore(i))

    # Added tokens
    tokens.append("[FLERT]");   scores.append(0.0)  # 128000
    tokens.append("<<ENT>>");  scores.append(0.0)   # 128001
    tokens.append("<<SEP>>");  scores.append(0.0)   # 128002
    tokens.append("[PAD_3]");  scores.append(0.0)    # 128003 (padding slot)

    writer.add_array("tokenizer.ggml.tokens", tokens)
    writer.add_array("tokenizer.ggml.scores", scores)
    writer.add_uint32("tokenizer.ggml.type", 2)  # SentencePiece
    writer.add_uint32("tokenizer.ggml.bos_token_id", sp.bos_id())    # 1 = [CLS]
    writer.add_uint32("tokenizer.ggml.eos_token_id", sp.eos_id())    # 2 = [SEP]
    writer.add_uint32("tokenizer.ggml.unknown_token_id", sp.unk_id()) # 3
    writer.add_uint32("tokenizer.ggml.padding_token_id", sp.pad_id()) # 0
    writer.add_uint32("bert.vocab_size", total_vocab)

    print(f"  tokenizer: {total_vocab} tokens ({spm_vocab_size} SPM + 4 special)")

    # --- Load and write tensors ---
    from safetensors import safe_open

    safetensors_path = os.path.join(model_dir, "model.safetensors")
    print(f"Loading weights from: {safetensors_path}")

    f = safe_open(safetensors_path, framework="pt")
    hf_keys = sorted(f.keys())

    n_written = 0
    n_skipped = 0
    for hf_name in hf_keys:
        gguf_name = remap_tensor_name(hf_name)
        if gguf_name is None:
            n_skipped += 1
            continue

        tensor = f.get_tensor(hf_name)
        data = tensor.float().numpy().astype(np_dtype)

        writer.add_tensor(gguf_name, data, raw_dtype=gguf_dtype)
        n_written += 1
        if n_written <= 10 or n_written % 20 == 0:
            print(f"  [{n_written}] {gguf_name:50s} {list(data.shape)}")

    print(f"\nWritten: {n_written} tensors, skipped: {n_skipped}")

    # Finalize
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    out_size = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Output: {args.output} ({out_size:.1f} MB)")


if __name__ == "__main__":
    main()
