#!/usr/bin/env python3
"""Batched decoder parity tests.

Verifies that batched decoder encoding (single graph for N texts) produces
the same embeddings as sequential encoding (N separate graph computes).

Environment variables:

    CRISPEMBED_LIB          Path to libcrispembed.{so,dylib,dll}
    CRISPEMBED_DECODER_MODEL Path to a decoder GGUF (e.g. octen-0.6b-q8_0.gguf)

Usage:
    CRISPEMBED_LIB=build/libcrispembed.so \
    CRISPEMBED_DECODER_MODEL=octen-0.6b-q8_0.gguf \
    python tests/test_decoder_batch.py
"""

import os
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

LIB = os.environ.get("CRISPEMBED_LIB")
DECODER_MODEL = os.environ.get("CRISPEMBED_DECODER_MODEL")
HAVE_MODEL = bool(DECODER_MODEL)

TEST_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a subset of artificial intelligence.",
    "Berlin is the capital of Germany.",
    "Quantum computing uses qubits instead of classical bits.",
    "The Eiffel Tower is located in Paris, France.",
    "Water boils at 100 degrees Celsius at sea level.",
    "DNA carries genetic information.",
    "The speed of light is approximately 300,000 km/s.",
]

# Short and long texts for variable-length padding test
VARIABLE_TEXTS = [
    "Hello",
    "The quick brown fox jumps over the lazy dog " * 10,
    "Hi there",
    "Artificial intelligence and machine learning are reshaping industries worldwide " * 5,
]


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


class TestDecoderBatchParity(unittest.TestCase):
    """Batch encoding should match sequential encoding."""

    @classmethod
    def setUpClass(cls):
        if not HAVE_MODEL:
            return
        from crispembed import CrispEmbed
        cls.model = CrispEmbed(DECODER_MODEL, n_threads=2, lib_path=LIB)

    @unittest.skipUnless(HAVE_MODEL, "Set CRISPEMBED_DECODER_MODEL")
    def test_batch_matches_sequential(self):
        """Batch of 4 texts should match individual encoding."""
        texts = TEST_TEXTS[:4]

        # Sequential
        singles = []
        for text in texts:
            singles.append(self.model.encode(text))

        # Batch
        batch = self.model.encode(texts)

        for i, (s, text) in enumerate(zip(singles, texts)):
            cos = cosine(s, batch[i])
            self.assertGreaterEqual(cos, 0.999,
                f"Text {i} batch vs sequential cos={cos:.6f}: {text[:40]}...")

    @unittest.skipUnless(HAVE_MODEL, "Set CRISPEMBED_DECODER_MODEL")
    def test_batch_8_texts(self):
        """Larger batch (8 texts)."""
        singles = [self.model.encode(t) for t in TEST_TEXTS]
        batch = self.model.encode(TEST_TEXTS)

        for i in range(len(TEST_TEXTS)):
            cos = cosine(singles[i], batch[i])
            self.assertGreaterEqual(cos, 0.999,
                f"Text {i} cos={cos:.6f}")

    @unittest.skipUnless(HAVE_MODEL, "Set CRISPEMBED_DECODER_MODEL")
    def test_variable_length(self):
        """Variable-length texts (forces padding) should match sequential."""
        singles = [self.model.encode(t) for t in VARIABLE_TEXTS]
        batch = self.model.encode(VARIABLE_TEXTS)

        for i in range(len(VARIABLE_TEXTS)):
            cos = cosine(singles[i], batch[i])
            self.assertGreaterEqual(cos, 0.999,
                f"Variable text {i} cos={cos:.6f}")

    @unittest.skipUnless(HAVE_MODEL, "Set CRISPEMBED_DECODER_MODEL")
    def test_batch_1_matches_single(self):
        """B=1 batch should match single encode."""
        single = self.model.encode(TEST_TEXTS[0])
        batch = self.model.encode([TEST_TEXTS[0]])

        cos = cosine(single, batch[0])
        self.assertGreaterEqual(cos, 0.999, f"B=1 cos={cos:.6f}")


class TestDecoderBatchPerformance(unittest.TestCase):
    """Performance comparison (informational, not assertions)."""

    @classmethod
    def setUpClass(cls):
        if not HAVE_MODEL:
            return
        from crispembed import CrispEmbed
        cls.model = CrispEmbed(DECODER_MODEL, n_threads=4, lib_path=LIB)

    @unittest.skipUnless(HAVE_MODEL, "Set CRISPEMBED_DECODER_MODEL")
    def test_performance(self):
        """Time sequential vs batch."""
        texts = TEST_TEXTS[:4]
        n_runs = 3

        # Warmup
        self.model.encode(texts[0])

        # Sequential
        t0 = time.time()
        for _ in range(n_runs):
            for t in texts:
                self.model.encode(t)
        t_seq = (time.time() - t0) / n_runs

        # Batch
        t0 = time.time()
        for _ in range(n_runs):
            self.model.encode(texts)
        t_batch = (time.time() - t0) / n_runs

        speedup = t_seq / max(t_batch, 1e-6)
        print(f"\n  Sequential: {t_seq*1000:.1f}ms, Batch: {t_batch*1000:.1f}ms, "
              f"Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    unittest.main()
