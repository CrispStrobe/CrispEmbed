#!/usr/bin/env python3
"""Dump LiLT per-layer activations to GGUF for parity testing.

Usage:
    python tools/dump_lilt_reference.py \
        --model SCUT-DLVCLab/lilt-roberta-en-base \
        --output /tmp/lilt-ref.gguf \
        --max-layers 2

Captures: text embeddings, layout embeddings, per-layer text + layout outputs,
final hidden state, and (if present) classifier logits.
"""

import argparse
import gc
import struct
import numpy as np
import os

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'


def write_ref_gguf(path: str, tensors: dict):
    """Write a reference GGUF with named float32 tensors."""
    import gguf
    writer = gguf.GGUFWriter(path, arch="lilt-ref")
    for name, arr in tensors.items():
        writer.add_tensor(name, arr.astype(np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model ID or path")
    parser.add_argument("--output", required=True, help="Output reference .gguf")
    parser.add_argument("--max-layers", type=int, default=2,
                        help="Capture first N layers (default 2)")
    parser.add_argument("--text", default="Date: 2026-06-15 Total: $48.60",
                        help="Input text")
    args = parser.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoConfig

    print(f"Loading: {args.model}")
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Detect if it's a token classification model
    has_classifier = hasattr(config, 'num_labels') and config.num_labels > 1

    if has_classifier:
        from transformers import LiltForTokenClassification as ModelCls
    else:
        from transformers import LiltModel as ModelCls

    model = ModelCls.from_pretrained(args.model, trust_remote_code=True)
    model.eval()

    # Tokenize — LiLT uses LayoutLMv3 tokenizer which expects pre-tokenized
    # words + per-word bounding boxes.
    words = args.text.split()
    # Create dummy bboxes: spread words across the page
    word_bboxes = []
    x = 10
    for w in words:
        w_width = len(w) * 20
        word_bboxes.append([x, 50, x + w_width, 80])
        x += w_width + 15

    encoding = tokenizer(words, boxes=word_bboxes, return_tensors="pt",
                         padding=False, truncation=True)
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    bbox = encoding["bbox"]
    seq_len = input_ids.shape[1]

    print(f"  text: {args.text}")
    print(f"  tokens: {seq_len}")
    print(f"  input_ids: {input_ids[0].tolist()}")

    tensors = {}

    # Hook into layers to capture intermediates
    captured = {}

    def make_hook(name):
        def hook(module, input, output):
            # Dig through nested tuples to find the first tensor
            o = output
            while isinstance(o, tuple):
                o = o[0]
            captured[name] = o.detach().float().cpu().numpy()
        return hook

    def make_layer_hook(i):
        """Capture both text and layout outputs from a LiLT layer."""
        def hook(module, input, output):
            # LiLT layer returns ((text_hidden, layout_hidden),)
            inner = output[0]
            text_out = inner[0].detach().float().cpu().numpy()
            layout_out = inner[1].detach().float().cpu().numpy()
            captured[f"layer_{i}_text"] = text_out
            captured[f"layer_{i}_layout"] = layout_out
        return hook

    # Get the base model
    base = model.lilt if has_classifier else model

    # Register hooks on text embeddings
    base.embeddings.register_forward_hook(make_hook("text_embed"))
    base.layout_embeddings.register_forward_hook(make_hook("layout_embed"))

    # Register hooks on encoder layers
    max_l = min(args.max_layers, config.num_hidden_layers)
    for i in range(max_l):
        layer = base.encoder.layer[i]
        layer.register_forward_hook(make_layer_hook(i))

    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids=input_ids, bbox=bbox,
                       attention_mask=attention_mask,
                       output_hidden_states=True)

    # Save captured activations
    for name, arr in captured.items():
        tensors[name] = arr.squeeze(0)  # remove batch dim
        print(f"  {name}: {tensors[name].shape}")

    # Save final hidden state
    if has_classifier:
        # LiltForTokenClassification returns logits
        if hasattr(outputs, 'logits') and outputs.logits is not None:
            logits = outputs.logits.detach().float().cpu().numpy().squeeze(0)
            tensors["logits"] = logits
            print(f"  logits: {logits.shape}")
            # Print predicted labels
            preds = logits.argmax(axis=-1)
            tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
            id2label = config.id2label
            print("\n  Predictions:")
            for t, p in zip(tokens, preds):
                print(f"    {t:15s} → {id2label.get(int(p), str(p))}")
        # Also get hidden states
        hs = outputs.hidden_states[-1].detach().float().cpu().numpy().squeeze(0)
        tensors["final_hidden"] = hs
        print(f"  final_hidden: {hs.shape}")
    else:
        hs = outputs.last_hidden_state.detach().float().cpu().numpy().squeeze(0)
        tensors["final_hidden"] = hs
        print(f"  final_hidden: {hs.shape}")

    # Save input metadata
    tensors["input_ids"] = input_ids[0].numpy().astype(np.float32)
    tensors["bbox"] = bbox[0].numpy().astype(np.float32)

    # Write reference GGUF
    write_ref_gguf(args.output, tensors)
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\nWrote {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
