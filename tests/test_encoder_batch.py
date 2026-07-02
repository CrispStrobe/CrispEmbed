#!/usr/bin/env python3
"""Packed encoder-batch parity + throughput A/B (convergence item C3).

Verifies that the packed block-diagonal encoder batch path
(CRISPEMBED_ENCODER_PACKED=1) produces embeddings identical to the trusted
per-sequence path (CRISPEMBED_ENCODER_PACKED=0) for bidirectional encoders
(BERT / XLM-R / MiniLM / BGE / E5 — absolute-position, no MPNet rel-bias,
no DeBERTa rel-embd, no RoPE).

The packed path collapses B sequences into one graph with a block-diagonal
F16 mask so each sequence only attends to its own tokens; positions restart
per segment. Output must match single-sequence encoding to cos >= 0.9999.

A/B summary (all-MiniLM-L6-v2 q8_0, M1, medians): packing is a CPU throughput
win (up to ~2x at moderate batch) but a Metal wash/loss on tiny models because
the block-diagonal mask makes attention O(T_total^2). It is therefore OPT-IN
(default off). Greedy token-budget grouping (CRISPEMBED_ENCODER_PACK_MAXTOK,
default 384) caps T_total so attention stays bounded.

Environment:
    CRISPEMBED_LIB            Path to libcrispembed.{so,dylib,dll}
    CRISPEMBED_ENCODER_MODEL  Path to an encoder GGUF (BERT/XLM-R family)
    CRISPEMBED_FORCE_CPU=1    (optional) also exercises the CPU backend

Usage:
    CRISPEMBED_LIB=build/libcrispembed.dylib \
    CRISPEMBED_ENCODER_MODEL=/path/all-MiniLM-L6-v2-q8_0.gguf \
    python tests/test_encoder_batch.py
"""

import os
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

MODEL = os.environ.get("CRISPEMBED_ENCODER_MODEL")
HAVE_MODEL = bool(MODEL)

# Deliberately mixed lengths (incl. a very short and a long text) to exercise
# per-segment positions and the block-diagonal mask boundaries.
TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a subset of artificial intelligence.",
    "Berlin is the capital of Germany.",
    "Quantum computing uses qubits instead of classical bits.",
    "Hi",
    "The Eiffel Tower is located in Paris, France. " * 8,
    "Water boils at 100 degrees Celsius at sea level.",
    "DNA carries genetic information across generations of living organisms.",
]


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _load():
    from crispembed import CrispEmbed
    return CrispEmbed(MODEL)


@unittest.skipUnless(HAVE_MODEL, "set CRISPEMBED_ENCODER_MODEL")
class TestEncoderBatchParity(unittest.TestCase):
    def _parity(self, cap):
        m = _load()
        os.environ["CRISPEMBED_ENCODER_PACKED"] = "0"
        ref = np.asarray(m.encode(TEXTS))
        os.environ["CRISPEMBED_ENCODER_PACKED"] = "1"
        os.environ["CRISPEMBED_ENCODER_PACK_MAXTOK"] = str(cap)
        got = np.asarray(m.encode(TEXTS))
        self.assertEqual(ref.shape, got.shape)
        worst = min(cosine(ref[i], got[i]) for i in range(len(TEXTS)))
        print(f"  cap={cap}: worst cos(packed, sequential) = {worst:.7f}")
        self.assertGreaterEqual(worst, 0.9999)

    def test_parity_single_group(self):
        # Large cap -> all texts in one packed group.
        self._parity(cap=100000)

    def test_parity_multi_group(self):
        # Small cap -> forces multiple packed groups + single-item groups.
        self._parity(cap=64)


@unittest.skipUnless(HAVE_MODEL, "set CRISPEMBED_ENCODER_MODEL")
class TestEncoder4DBatchParity(unittest.TestCase):
    """Rectangular 4D per-item-mask batch (CRISPEMBED_ENCODER_4D=1): sequences kept
    as separate 4D batch items padded to T_max, per-item padding mask. Attention is
    O(B·T²) (vs packing's O((B·T)²)) — the recommended throughput path. Must be
    bit-parity with per-sequence encoding (padded keys masked to −inf; padded query
    rows discarded in pooling)."""

    def _parity(self, group):
        m = _load()
        os.environ.pop("CRISPEMBED_ENCODER_PACKED", None)
        os.environ.pop("CRISPEMBED_ENCODER_4D", None)
        ref = np.asarray(m.encode(TEXTS))
        os.environ["CRISPEMBED_ENCODER_4D"] = "1"
        os.environ["CRISPEMBED_ENCODER_4D_GROUP"] = str(group)
        got = np.asarray(m.encode(TEXTS))
        self.assertEqual(ref.shape, got.shape)
        worst = min(cosine(ref[i], got[i]) for i in range(len(TEXTS)))
        print(f"  4D group={group}: worst cos(4D, sequential) = {worst:.7f}")
        self.assertGreaterEqual(worst, 0.9999)

    def test_parity_single_group(self):
        # Large group -> all texts padded together in one 4D graph.
        self._parity(group=100)

    def test_parity_multi_group(self):
        # Small group -> length-sorted, multiple padded chunks + single-item groups.
        self._parity(group=2)


@unittest.skipUnless(HAVE_MODEL, "set CRISPEMBED_ENCODER_MODEL")
class TestEncoderBatchThroughput(unittest.TestCase):
    """Informational only (no assertion) — packing is backend/size dependent."""

    def test_throughput(self):
        import statistics
        m = _load()
        sent = "Machine learning is a subset of artificial intelligence used today."

        def bench(texts, reps=7):
            for _ in range(2):
                m.encode(texts)
            ts = []
            for _ in range(reps):
                t = time.perf_counter()
                m.encode(texts)
                ts.append(time.perf_counter() - t)
            return statistics.median(ts)

        for n in (8, 32, 128):
            texts = [sent] * n
            os.environ["CRISPEMBED_ENCODER_PACKED"] = "0"
            seq = bench(texts)
            os.environ["CRISPEMBED_ENCODER_PACKED"] = "1"
            os.environ["CRISPEMBED_ENCODER_PACK_MAXTOK"] = "384"
            pk = bench(texts)
            print(f"  N={n:4d}: seq {seq*1000:7.1f}ms ({n/seq:6.1f}/s)  "
                  f"packed {pk*1000:7.1f}ms ({n/pk:6.1f}/s)  {seq/pk:.2f}x")


if __name__ == "__main__":
    if not HAVE_MODEL:
        print("SKIP: set CRISPEMBED_ENCODER_MODEL to an encoder GGUF")
        sys.exit(0)
    unittest.main(verbosity=2)
