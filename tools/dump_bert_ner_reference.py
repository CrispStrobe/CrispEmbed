#!/usr/bin/env python3
"""Dump BERT-NER reference activations for crispembed_diff parity testing.

Writes the two tensors test-bert-ner-diff expects:
  - input_ids:    the tokenized input (f32, [n_tok])
  - final_hidden: the BERT encoder's final hidden state (f32, [n_tok, hidden])

The C++ engine reads input_ids from the ref and recomputes final_hidden, so the
inputs stay aligned. bert_ner is BERT encoder + a linear BIO classifier; the
encoder hidden state is the meaningful parity signal (mirrors dump_lilt_reference.py).

Usage:
  python tools/dump_bert_ner_reference.py \
      --model dslim/bert-base-NER \
      --text "George Washington went to Washington" \
      --output bert-ner-ref.gguf
"""
import argparse
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dslim/bert-base-NER")
    # MUST match the text hardcoded in tests/test_bert_ner_diff.cpp (the engine encodes
    # this string; the ref is compared against it).
    ap.add_argument("--text", default="Barack Obama was born in Hawaii")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModel
    print(f"Loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    # AutoModel -> the BERT encoder (no classifier head); we want the encoder's
    # last_hidden_state, which is what the C++ bert_ner encoder produces.
    # use_safetensors=False: dslim/bert-base-NER's model.safetensors SIGBUSes on mmap on
    # some systems (macOS/M1); the pytorch_model.bin loads cleanly.
    try:
        model = AutoModel.from_pretrained(args.model, use_safetensors=False, dtype=torch.float32)
    except Exception:
        model = AutoModel.from_pretrained(args.model, dtype=torch.float32)
    model.eval()

    enc = tok(args.text, return_tensors="pt")
    input_ids = enc["input_ids"]
    n_tok = input_ids.shape[1]
    print(f"text: {args.text}")
    print(f"tokens: {n_tok}  input_ids: {input_ids[0].tolist()}")

    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    final_hidden = out.last_hidden_state[0].float().cpu().numpy()   # [n_tok, hidden]
    print(f"final_hidden: {final_hidden.shape}")

    writer = gguf.GGUFWriter(args.output, arch="bert_ner_ref")
    writer.add_string("ref.text", args.text)
    # f32 tensors — test-bert-ner-diff reads input_ids via get_f32 and compares final_hidden.
    writer.add_tensor("input_ids", input_ids[0].to(torch.float32).cpu().numpy())
    writer.add_tensor("final_hidden", final_hidden.astype(np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
