#!/usr/bin/env python3
"""ModernBERT (gte-modernbert-base) end-to-end parity vs HuggingFace (item C8).

Guards three fixes that brought ModernBERT from broken to full parity:
  1. BPE tokenizer  — the converter's BPE detection + pooling detection used
     hf_hub_download(repo_id=<path>), which fails silently on a LOCAL path and
     fell back to WordPiece / mean. Fixed via models/convert-bert-to-gguf.py
     _resolve_file(). (Convert with --crisp for the CrispEmbed-native tokenizer.)
  2. CLS pooling    — same local-path bug defaulted pooling to mean; this model
     uses CLS (1_Pooling/config.json pooling_mode_cls_token=true).
  3. Sliding-window local attention — ModernBERT alternates global (every Nth)
     and local layers; local layers must be masked to a ±local_attention/2 window
     (src/crispembed.cpp swa_mask). Without it, local layers attend globally and
     long documents diverge. A/B: CRISPEMBED_ENCODER_NO_SWA=1 disables the mask.

Verified numbers (gte-modernbert-base, M1, f16): short-text cos(HF CLS) = 0.999999
on Metal + CPU; 140-token doc cos SWA-on 0.999998 vs SWA-off 0.9615; q8_0 0.99976.

Environment:
    CRISPEMBED_LIB              libcrispembed.{so,dylib}
    CRISPEMBED_MODERNBERT_MODEL gte-modernbert-base GGUF (convert with --crisp)
    CRISPEMBED_MODERNBERT_HF    HF model dir/id (default Alibaba-NLP/gte-modernbert-base)
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

GGUF = os.environ.get("CRISPEMBED_MODERNBERT_MODEL")
HF = os.environ.get("CRISPEMBED_MODERNBERT_HF", "Alibaba-NLP/gte-modernbert-base")
HAVE = bool(GGUF)

SHORT = "Berlin is the capital of Germany."
# >128 tokens so the local sliding window (radius 64) actually restricts attention.
LONG = (
    "Machine learning is a subset of artificial intelligence that enables systems "
    "to learn from data. Transformers use self attention to model long range "
    "dependencies across an entire sequence. ModernBERT alternates global and local "
    "attention layers to process long documents efficiently while keeping quality "
    "high. The sliding window restricts most layers to a local neighborhood, and "
    "every third layer attends globally to mix information across the whole passage. "
    "Berlin is the capital of Germany and the Eiffel Tower stands in Paris while "
    "water boils at one hundred degrees Celsius at sea level near the open ocean."
)


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _hf_cls(text):
    import torch
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(HF)
    enc = tok(text, return_tensors="pt", truncation=True, max_length=512)
    mdl = AutoModel.from_pretrained(HF, dtype=torch.float32).eval()
    with torch.no_grad():
        cls = mdl(**enc).last_hidden_state[0, 0].numpy()  # CLS, post final-norm
    return cls, enc["input_ids"].shape[1]


@unittest.skipUnless(HAVE, "set CRISPEMBED_MODERNBERT_MODEL")
class TestModernBertParity(unittest.TestCase):
    def _crisp(self, text, no_swa=False):
        from crispembed import CrispEmbed
        if no_swa:
            os.environ["CRISPEMBED_ENCODER_NO_SWA"] = "1"
        else:
            os.environ.pop("CRISPEMBED_ENCODER_NO_SWA", None)
        return np.asarray(CrispEmbed(GGUF).encode([text])[0])

    def test_short_parity(self):
        ref, _ = _hf_cls(SHORT)
        c = cos(ref, self._crisp(SHORT))
        print(f"  short cos(HF CLS) = {c:.6f}")
        self.assertGreaterEqual(c, 0.999)

    def test_long_parity_and_swa_effect(self):
        ref, ntok = _hf_cls(LONG)
        on = cos(ref, self._crisp(LONG, no_swa=False))
        off = cos(ref, self._crisp(LONG, no_swa=True))
        print(f"  long ({ntok} tok) cos SWA-on={on:.6f}  SWA-off={off:.6f}")
        self.assertGreaterEqual(on, 0.999)            # with the fix: parity
        self.assertGreater(on, off + 0.005)           # the window measurably helps


if __name__ == "__main__":
    if not HAVE:
        print("SKIP: set CRISPEMBED_MODERNBERT_MODEL")
        sys.exit(0)
    unittest.main(verbosity=2)
