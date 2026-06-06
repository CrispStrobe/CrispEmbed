#!/usr/bin/env python3
"""LoRA adapter hot-swap parity tests.

Tests that runtime LoRA switching produces the same embeddings as baked
(merge-at-convert-time) GGUFs, and that switching adapters is reversible.

Environment variables:

    CRISPEMBED_LIB          Path to libcrispembed.{so,dylib,dll}
    CRISPEMBED_LORA_GGUF    GGUF with --lora-mode=separate (all adapters)
    CRISPEMBED_BAKED_GGUF   GGUF with --lora-mode=merge (single adapter baked)
    CRISPEMBED_BAKED_ADAPTER Name of the adapter baked in BAKED_GGUF (default: retrieval)

Usage:
    python tests/test_lora_parity.py

    # With explicit paths
    CRISPEMBED_LIB=build/libcrispembed.so \
    CRISPEMBED_LORA_GGUF=jina-v5-nano-lora.gguf \
    CRISPEMBED_BAKED_GGUF=jina-v5-nano.gguf \
    python tests/test_lora_parity.py
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

LIB = os.environ.get("CRISPEMBED_LIB")
LORA_GGUF = os.environ.get("CRISPEMBED_LORA_GGUF")
BAKED_GGUF = os.environ.get("CRISPEMBED_BAKED_GGUF")
BAKED_ADAPTER = os.environ.get("CRISPEMBED_BAKED_ADAPTER", "retrieval")
# Non-LoRA decoder model for graceful-degradation test
PLAIN_DECODER = os.environ.get("CRISPEMBED_PLAIN_DECODER")

HAVE_LORA = LORA_GGUF and BAKED_GGUF
HAVE_PLAIN = bool(PLAIN_DECODER)

TEST_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a subset of artificial intelligence.",
    "Berlin is the capital of Germany.",
]


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


class TestLoraParity(unittest.TestCase):
    """Compare LoRA hot-swap vs baked adapter."""

    @classmethod
    def setUpClass(cls):
        if not HAVE_LORA:
            return
        from crispembed import CrispEmbed
        cls.lora_model = CrispEmbed(LORA_GGUF, n_threads=2, lib_path=LIB)
        cls.baked_model = CrispEmbed(BAKED_GGUF, n_threads=2, lib_path=LIB)

    @unittest.skipUnless(HAVE_LORA, "Set CRISPEMBED_LORA_GGUF and CRISPEMBED_BAKED_GGUF")
    def test_list_lora(self):
        adapters = self.lora_model.list_lora()
        self.assertGreater(len(adapters), 0, "Expected at least one adapter")
        self.assertIn(BAKED_ADAPTER, adapters)

    @unittest.skipUnless(HAVE_LORA, "Set CRISPEMBED_LORA_GGUF and CRISPEMBED_BAKED_GGUF")
    def test_parity_vs_baked(self):
        """LoRA hot-swap should match baked merge to cos >= 0.9999."""
        ok = self.lora_model.set_lora(BAKED_ADAPTER)
        self.assertTrue(ok, f"Failed to set LoRA adapter '{BAKED_ADAPTER}'")

        for text in TEST_TEXTS:
            lora_emb = self.lora_model.encode(text)
            baked_emb = self.baked_model.encode(text)
            cos = cosine(lora_emb, baked_emb)
            self.assertGreaterEqual(cos, 0.9999,
                f"LoRA vs baked cos={cos:.6f} for: {text[:40]}...")

    @unittest.skipUnless(HAVE_LORA, "Set CRISPEMBED_LORA_GGUF and CRISPEMBED_BAKED_GGUF")
    def test_round_trip(self):
        """Switch adapter → switch back should produce identical embeddings."""
        adapters = self.lora_model.list_lora()
        if len(adapters) < 2:
            self.skipTest("Need at least 2 adapters for round-trip test")

        self.lora_model.set_lora(adapters[0])
        emb_first = self.lora_model.encode(TEST_TEXTS[0]).copy()

        self.lora_model.set_lora(adapters[1])
        emb_other = self.lora_model.encode(TEST_TEXTS[0]).copy()

        self.lora_model.set_lora(adapters[0])
        emb_back = self.lora_model.encode(TEST_TEXTS[0]).copy()

        # First and third should be identical
        np.testing.assert_array_equal(emb_first, emb_back,
            "Round-trip should produce bit-identical embeddings")

        # Different adapters should give different embeddings
        cos_diff = cosine(emb_first, emb_other)
        self.assertLess(cos_diff, 0.9999,
            f"Different adapters should give different embeddings (cos={cos_diff})")

    @unittest.skipUnless(HAVE_LORA, "Set CRISPEMBED_LORA_GGUF and CRISPEMBED_BAKED_GGUF")
    def test_get_lora(self):
        self.lora_model.set_lora(BAKED_ADAPTER)
        self.assertEqual(self.lora_model.lora, BAKED_ADAPTER)

    @unittest.skipUnless(HAVE_LORA, "Set CRISPEMBED_LORA_GGUF and CRISPEMBED_BAKED_GGUF")
    def test_unmerge(self):
        """Unmerge (empty adapter name) should restore base weights."""
        self.lora_model.set_lora(BAKED_ADAPTER)
        self.assertTrue(len(self.lora_model.lora) > 0)
        self.lora_model.set_lora("")
        self.assertEqual(self.lora_model.lora, "")


class TestLoraGracefulDegradation(unittest.TestCase):
    """set_lora on a non-LoRA model should fail gracefully."""

    @unittest.skipUnless(HAVE_PLAIN, "Set CRISPEMBED_PLAIN_DECODER")
    def test_set_lora_on_plain_model(self):
        from crispembed import CrispEmbed
        model = CrispEmbed(PLAIN_DECODER, n_threads=2, lib_path=LIB)
        ok = model.set_lora("anything")
        self.assertFalse(ok)
        # Encoding should still work
        emb = model.encode(TEST_TEXTS[0])
        self.assertGreater(emb.shape[0], 0)

    @unittest.skipUnless(HAVE_PLAIN, "Set CRISPEMBED_PLAIN_DECODER")
    def test_list_lora_empty(self):
        from crispembed import CrispEmbed
        model = CrispEmbed(PLAIN_DECODER, n_threads=2, lib_path=LIB)
        adapters = model.list_lora()
        self.assertEqual(len(adapters), 0)


if __name__ == "__main__":
    unittest.main()
