#!/usr/bin/env python3
"""Dump ModernBERT reference activations for crispembed_diff parity testing.

Writes the two tensors test-modernbert-diff expects:
  - input_ids:    the tokenized input (f32, [n_tok])
  - final_hidden: the encoder's final hidden state (f32, [n_tok, hidden])

The text is intentionally >128 tokens so ModernBERT's local sliding-window layers
(radius local_attention/2) actually restrict attention — this guards the SWA mask
(src/crispembed.cpp swa_mask), not just the backbone. The C++ engine re-tokenizes
the same text and recomputes final_hidden.

Usage:
  python tools/dump_modernbert_reference.py \
      --model Alibaba-NLP/gte-modernbert-base \
      --output modernbert-ref.gguf
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

# MUST match the text hardcoded in tests/test_modernbert_diff.cpp. ~113 tokens; the
# local window radius is local_attention/2 = 64, so token pairs >64 apart ARE
# restricted on the local layers — this exercises the SWA mask. (Verified HF and the
# CrispEmbed BPE tokenize this text identically.)
TEXT = (
    "Machine learning is a subset of artificial intelligence that enables systems "
    "to learn from data. Transformers use self attention to model long range "
    "dependencies across an entire sequence. ModernBERT alternates global and local "
    "attention layers to process long documents efficiently while keeping quality "
    "high. The sliding window restricts most layers to a local neighborhood, and "
    "every third layer attends globally to mix information across the whole passage. "
    "Berlin is the capital of Germany and the Eiffel Tower stands in Paris while "
    "water boils at one hundred degrees Celsius at sea level near the open ocean."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Alibaba-NLP/gte-modernbert-base")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModel
    print(f"Loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, dtype=torch.float32).eval()

    enc = tok(TEXT, return_tensors="pt")
    input_ids = enc["input_ids"]
    n_tok = input_ids.shape[1]
    print(f"tokens: {n_tok}  (need >128 for the local window to restrict attention)")

    with torch.no_grad():
        out = model(**enc)
    final_hidden = out.last_hidden_state[0].float().cpu().numpy()  # [n_tok, hidden]
    print(f"final_hidden: {final_hidden.shape}")

    writer = gguf.GGUFWriter(args.output, arch="modernbert_ref")
    writer.add_string("ref.text", TEXT)
    writer.add_tensor("input_ids", input_ids[0].to(torch.float32).cpu().numpy())
    writer.add_tensor("final_hidden", final_hidden.astype(np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
