#!/usr/bin/env python3
"""Dump GLiNER per-layer reference activations for crispembed_diff parity testing.

Captures intermediates at each architectural boundary:
  - post_embed:    after embedding lookup (before any layers)
  - layer_N:       output of LFM2 layer N (0..15)
  - final_norm:    after final RMSNorm
  - fused:         after layer fusion
  - lstm_out:      after BiLSTM
  - ent_reps:      entity type representations after prompt_rep MLP
  - span_scores:   raw dot-product scores (before sigmoid)

Usage:
  python tools/dump_gliner_reference.py \
      --model VAGOsolutions/SauerkrautLM-LFM2.5-GLiNER \
      --text "Barack Obama was born in Hawaii" \
      --labels person organization location email \
      --output /mnt/volume1/gliner-ref.gguf
"""

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "ggml" / "scripts"))
try:
    import gguf
except ImportError:
    print("pip install gguf", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="VAGOsolutions/SauerkrautLM-LFM2.5-GLiNER")
    parser.add_argument("--text", default="Barack Obama was born in Hawaii")
    parser.add_argument("--labels", nargs="+", default=["person", "organization", "location", "email"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", default="/mnt/akademie_storage/huggingface/hub")
    args = parser.parse_args()

    # Load model — work around tokenizer class issue by patching config
    import json, os, shutil, tempfile
    model_dir = args.model
    if not os.path.isdir(model_dir):
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download(args.model, cache_dir=args.cache_dir)

    # Patch tokenizer_config.json to use a standard class
    tok_cfg_path = os.path.join(model_dir, "tokenizer_config.json")
    with open(tok_cfg_path) as f:
        tok_cfg = json.load(f)
    if tok_cfg.get("tokenizer_class") == "TokenizersBackend":
        # Create a temp copy with patched config
        tmp_dir = tempfile.mkdtemp(dir="/mnt/volume1")
        for fn in os.listdir(model_dir):
            src = os.path.join(model_dir, fn)
            dst = os.path.join(tmp_dir, fn)
            if os.path.isfile(src):
                os.symlink(src, dst)
        # Overwrite the tokenizer config
        tok_cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
        # Fix extra_special_tokens: list → dict
        if isinstance(tok_cfg.get("extra_special_tokens"), list):
            tok_cfg["extra_special_tokens"] = {}
        patched_path = os.path.join(tmp_dir, "tokenizer_config.json")
        os.unlink(patched_path)
        with open(patched_path, "w") as f:
            json.dump(tok_cfg, f)
        model_dir = tmp_dir
        print(f"Patched tokenizer_config in {tmp_dir}")

    # Register Lfm2BiModel from code_modification before loading
    code_mod_dir = os.path.join(args.model if os.path.isdir(args.model) else model_dir, "code_modification")
    if os.path.isdir(code_mod_dir):
        sys.path.insert(0, code_mod_dir)
        print(f"Registered code_modification from {code_mod_dir}")

    from gliner import GLiNER
    model = GLiNER.from_pretrained(model_dir, cache_dir=args.cache_dir)
    model.eval()

    # Replace backbone with bidirectional variant if available
    try:
        from lfm2_bi import Lfm2BiModel
        bert = model.model.token_rep_layer.bert_layer
        old_model = bert.model
        if type(old_model).__name__ != "Lfm2BiModel":
            bi_model = Lfm2BiModel(old_model.config)
            bi_model.load_state_dict(old_model.state_dict())
            bi_model.eval()
            bert.model = bi_model
            print(f"Replaced {type(old_model).__name__} → Lfm2BiModel (bidirectional)")
    except ImportError:
        print("WARNING: Lfm2BiModel not available, using causal model")

    # Collect intermediates
    intermediates = {}

    # Hook into the LFM2 backbone layers
    # GLiNER structure: model.model.token_rep_layer.bert_layer.model.layers[i]
    inner = model.model  # UniEncoderSpanModel
    bert_layer = inner.token_rep_layer.bert_layer

    # Hook: post-embedding
    def hook_embed(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        intermediates["post_embed"] = output.detach().float().cpu().numpy()

    # Register hooks on each transformer layer
    layer_hooks = []
    backbone = bert_layer.model
    embed_hook = backbone.embed_tokens.register_forward_hook(hook_embed)

    for i, layer in enumerate(backbone.layers):
        def make_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                intermediates[f"layer_{layer_idx}"] = output.detach().float().cpu().numpy()
            return hook
        h = layer.register_forward_hook(make_hook(i))
        layer_hooks.append(h)

    # Hook: embedding norm (final norm)
    if hasattr(backbone, 'embedding_norm'):
        def hook_final_norm(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            intermediates["final_norm"] = output.detach().float().cpu().numpy()
        norm_hook = backbone.embedding_norm.register_forward_hook(hook_final_norm)

    # Hook: layer fuser output
    if hasattr(bert_layer, 'layers_fuser'):
        def hook_fuser(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            intermediates["fused"] = output.detach().float().cpu().numpy()
        fuser_hook = bert_layer.layers_fuser.register_forward_hook(hook_fuser)

    # Hook: BiLSTM output (rnn module)
    if hasattr(inner, 'rnn'):
        def hook_lstm(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            intermediates["lstm_out"] = output.detach().float().cpu().numpy()
        lstm_hook = inner.rnn.register_forward_hook(hook_lstm)

    # Run inference with hooks
    print(f"Text: {args.text}")
    print(f"Labels: {args.labels}")

    with torch.no_grad():
        entities = model.predict_entities(args.text, args.labels, threshold=0.3)

    print(f"\nPython GLiNER result ({len(entities)} entities):")
    for e in entities:
        print(f"  [{e['start']}:{e['end']}] \"{e['text']}\" => {e['label']} ({e['score']:.3f})")

    # Remove hooks
    embed_hook.remove()
    for h in layer_hooks:
        h.remove()

    # ---- Also capture span/entity representations manually ----
    # Re-run the tokenization to get the same input_ids
    print(f"\nCaptured intermediates: {list(intermediates.keys())}")

    # Save to GGUF
    writer = gguf.GGUFWriter(args.output, arch="gliner_ref")

    # Metadata
    writer.add_string("general.architecture", "gliner_ref")
    writer.add_string("ref.text", args.text)
    writer.add_array("ref.labels", args.labels)
    writer.add_uint32("ref.n_entities", len(entities))

    # Write intermediates as tensors
    for name, data in intermediates.items():
        arr = np.ascontiguousarray(data, dtype=np.float32)
        # Squeeze batch dim if present
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        writer.add_tensor(name, arr, raw_dtype=gguf.GGMLQuantizationType.F32)
        print(f"  {name}: shape={arr.shape}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    import os
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\nWrote: {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
