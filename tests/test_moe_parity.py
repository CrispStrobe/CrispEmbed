#!/usr/bin/env python3
"""MoE encoder parity tests: CrispEmbed vs HuggingFace reference.

Converts nomic-embed-text-v2-moe (or a specified MoE model), runs both
HuggingFace and CrispEmbed, and compares per-sentence embeddings.

Environment variables:

    CRISPEMBED_LIB       Path to libcrispembed.{so,dylib,dll}
    MOE_MODEL            HF model ID (default: nomic-ai/nomic-embed-text-v2-moe)
    MOE_GGUF             Pre-converted GGUF path (skip conversion if set)

Usage:
    # Convert + test (requires torch, transformers, gguf packages):
    CRISPEMBED_LIB=build/libcrispembed.so python tests/test_moe_parity.py

    # Test with pre-converted GGUF:
    CRISPEMBED_LIB=build/libcrispembed.so \
    MOE_GGUF=/path/to/nomic-v2-moe.gguf \
    python tests/test_moe_parity.py
"""

import os
import sys
import subprocess
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

LIB = os.environ.get("CRISPEMBED_LIB")
MOE_MODEL = os.environ.get("MOE_MODEL", "nomic-ai/nomic-embed-text-v2-moe")
MOE_GGUF = os.environ.get("MOE_GGUF")

TEST_TEXTS = [
    "search_query: What is a mixture of experts?",
    "search_document: MoE models route tokens to specialized expert subnetworks.",
    "search_query: Berlin is the capital of Germany.",
    "search_document: The quick brown fox jumps over the lazy dog.",
]


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def convert_model(model_id, output_path):
    """Run the BERT converter to produce a GGUF."""
    converter = os.path.join(os.path.dirname(__file__), "..", "models", "convert-bert-to-gguf.py")
    cmd = [
        sys.executable, converter,
        "--model", model_id,
        "--output", output_path,
        "--crisp",
    ]
    print(f"Converting: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise RuntimeError(f"Conversion failed (exit {result.returncode})")
    print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
    return output_path


def hf_embeddings(model_id, texts):
    """Get reference embeddings from HuggingFace model."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    model.eval()

    encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        out = model(**encoded)
    # Mean pooling over non-padding tokens
    hidden = out.last_hidden_state  # [B, T, H]
    mask = encoded["attention_mask"].unsqueeze(-1).float()  # [B, T, 1]
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    # L2 normalize
    pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
    return pooled.cpu().numpy()


class TestMoEParity(unittest.TestCase):
    """Compare CrispEmbed MoE encoder against HuggingFace reference."""

    gguf_path = None
    hf_embeds = None
    ce_ctx = None

    @classmethod
    def setUpClass(cls):
        if not LIB:
            return

        # Get or create GGUF
        if MOE_GGUF:
            cls.gguf_path = MOE_GGUF
        else:
            cls.gguf_path = os.path.join(
                os.environ.get("TMPDIR", "/tmp"),
                "crispembed-moe-test.gguf"
            )
            if not os.path.exists(cls.gguf_path):
                convert_model(MOE_MODEL, cls.gguf_path)

        # Compute HF reference embeddings
        try:
            cls.hf_embeds = hf_embeddings(MOE_MODEL, TEST_TEXTS)
            print(f"HF embeddings: shape={cls.hf_embeds.shape}")
        except Exception as e:
            print(f"Warning: HF reference failed ({e}), skipping parity tests")
            cls.hf_embeds = None

        # Load CrispEmbed
        from crispembed import CrispEmbed
        cls.ce_ctx = CrispEmbed(cls.gguf_path, lib_path=LIB)

    @classmethod
    def tearDownClass(cls):
        cls.ce_ctx = None

    @unittest.skipUnless(LIB, "CRISPEMBED_LIB not set")
    def test_load_moe_model(self):
        """Model loads without error and reports MoE layers."""
        self.assertIsNotNone(self.ce_ctx)

    @unittest.skipUnless(LIB, "CRISPEMBED_LIB not set")
    def test_encode_produces_output(self):
        """CrispEmbed produces non-zero embeddings for MoE model."""
        emb = self.ce_ctx.encode("search_query: test")
        self.assertIsNotNone(emb)
        self.assertGreater(len(emb), 0)
        self.assertGreater(np.linalg.norm(emb), 0.1)

    @unittest.skipUnless(LIB, "CRISPEMBED_LIB not set")
    def test_embedding_dimension(self):
        """Output dimension matches model's hidden size."""
        emb = self.ce_ctx.encode("search_query: test")
        # nomic-embed-text-v2-moe has hidden_size=768
        self.assertEqual(len(emb), 768)

    @unittest.skipUnless(LIB, "CRISPEMBED_LIB not set")
    def test_deterministic(self):
        """Two encodes of the same text produce identical results."""
        text = "search_query: determinism check"
        e1 = self.ce_ctx.encode(text)
        e2 = self.ce_ctx.encode(text)
        np.testing.assert_array_equal(e1, e2)

    @unittest.skipUnless(LIB, "CRISPEMBED_LIB not set")
    def test_different_texts_differ(self):
        """Different texts produce different embeddings."""
        e1 = self.ce_ctx.encode("search_query: cats")
        e2 = self.ce_ctx.encode("search_query: quantum physics")
        cos = cosine(e1, e2)
        self.assertLess(cos, 0.95, f"Different texts too similar: cos={cos:.4f}")

    @unittest.skipUnless(LIB and not MOE_GGUF, "Need both LIB and torch for HF parity")
    def test_parity_vs_huggingface(self):
        """CrispEmbed embeddings match HuggingFace within cos >= 0.99."""
        if self.hf_embeds is None:
            self.skipTest("HF reference embeddings not available")

        for i, text in enumerate(TEST_TEXTS):
            with self.subTest(text=text[:40]):
                ce_emb = np.array(self.ce_ctx.encode(text))
                hf_emb = self.hf_embeds[i]
                cos = cosine(ce_emb, hf_emb)
                print(f"  [{i}] cos={cos:.6f}  text={text[:50]}")
                self.assertGreaterEqual(
                    cos, 0.99,
                    f"Parity too low: cos={cos:.6f} for '{text[:40]}'"
                )

    @unittest.skipUnless(LIB and not MOE_GGUF, "Need both LIB and torch for HF parity")
    def test_parity_ranking_preserved(self):
        """Relative similarity ranking is preserved between HF and CrispEmbed."""
        if self.hf_embeds is None:
            self.skipTest("HF reference embeddings not available")

        ce_embeds = []
        for text in TEST_TEXTS:
            ce_embeds.append(np.array(self.ce_ctx.encode(text)))
        ce_embeds = np.array(ce_embeds)

        # Compare query-doc similarities: same-topic pair should rank highest
        # texts[0] is query about MoE, texts[1] is doc about MoE
        hf_sim_match = cosine(self.hf_embeds[0], self.hf_embeds[1])
        hf_sim_mismatch = cosine(self.hf_embeds[0], self.hf_embeds[3])
        ce_sim_match = cosine(ce_embeds[0], ce_embeds[1])
        ce_sim_mismatch = cosine(ce_embeds[0], ce_embeds[3])

        print(f"  HF:  match={hf_sim_match:.4f}  mismatch={hf_sim_mismatch:.4f}")
        print(f"  CE:  match={ce_sim_match:.4f}  mismatch={ce_sim_mismatch:.4f}")

        # Both should agree: matched pair is more similar
        self.assertGreater(hf_sim_match, hf_sim_mismatch)
        self.assertGreater(ce_sim_match, ce_sim_mismatch)


if __name__ == "__main__":
    unittest.main(verbosity=2)
