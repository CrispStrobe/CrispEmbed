"""Tier 0 unit test for the community-GGUF import matrix driver (A3).

No models, no network, no binary. Validates:
  - community_gguf_matrix.json schema/shape (every entry has the required keys)
  - parse_load_banner()  — pulls n_layer/dim out of the CLI banner
  - cosine()             — incl. degenerate inputs
  - evaluate()           — the pass/fail logic, especially that it CATCHES the
                           issue-#33 silent-default shape (384-dim / 6-layer) and
                           a garbage embedding that still has the right dim

Runs in well under a second, so PR CI guards the driver without models.

Usage:
  python tests/test_community_gguf_smoke.py
  # or: python -m unittest tests.test_community_gguf_smoke
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_community_gguf as drv  # noqa: E402

REQUIRED = ("name", "repo", "file", "arch", "dim", "n_layer", "query", "related", "unrelated")


class ManifestSchema(unittest.TestCase):
    def test_manifest_entries_well_formed(self) -> None:
        models = drv.load_manifest()["models"]
        self.assertGreater(len(models), 0, "matrix must not be empty")
        names = set()
        for e in models:
            for k in REQUIRED:
                self.assertIn(k, e, f"{e.get('name', '?')} missing '{k}'")
            self.assertIsInstance(e["dim"], int)
            self.assertIsInstance(e["n_layer"], int)
            self.assertNotIn(e["name"], names, "duplicate entry name")
            names.add(e["name"])
            self.assertTrue(e["file"].endswith(".gguf"))
            self.assertNotIn("/", e["file"], "file must be a bare filename")

    def test_nomic_entry_pins_the_issue33_shape(self) -> None:
        # The regression this matrix exists for: 768/12, not the 384/6 defaults.
        e = next(m for m in drv.load_manifest()["models"] if m["name"] == "nomic-embed-text-v2-moe")
        self.assertEqual(e["dim"], 768)
        self.assertEqual(e["n_layer"], 12)
        self.assertEqual(e["arch"], "nomic-bert-moe")


class ParseBanner(unittest.TestCase):
    def test_parses_real_banner(self) -> None:
        s = ("crispembed: using SentencePiece tokenizer (250048 tokens)\n"
             "crispembed: MoE encoder (8 experts, top-2, 6/12 MoE layers)\n"
             "crispembed: loaded 12 layers, 768 dims, 250048 vocab\n")
        self.assertEqual(drv.parse_load_banner(s), {"n_layer": 12, "dim": 768})

    def test_missing_banner_is_empty(self) -> None:
        self.assertEqual(drv.parse_load_banner("crispembed: missing required tensor attn.q.weight"), {})
        self.assertEqual(drv.parse_load_banner(""), {})
        self.assertEqual(drv.parse_load_banner(None), {})


class Cosine(unittest.TestCase):
    def test_identical(self) -> None:
        self.assertAlmostEqual(drv.cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 1.0, places=6)

    def test_orthogonal(self) -> None:
        self.assertAlmostEqual(drv.cosine([1.0, 0.0], [0.0, 1.0]), 0.0, places=6)

    def test_degenerate(self) -> None:
        self.assertTrue(math.isnan(drv.cosine([], [])))
        self.assertTrue(math.isnan(drv.cosine([1.0], [1.0, 2.0])))  # length mismatch
        self.assertTrue(math.isnan(drv.cosine([0.0, 0.0], [1.0, 1.0])))  # zero vector


class Evaluate(unittest.TestCase):
    ENTRY = {"name": "x", "dim": 768, "n_layer": 12, "min_margin": 0.05}

    def _named(self, results):
        return {name: ok for name, ok, _ in results}

    def test_healthy_model_passes(self) -> None:
        r = self._named(drv.evaluate(self.ENTRY, {"n_layer": 12, "dim": 768}, 768, 0.72, 0.31))
        self.assertTrue(all(r.values()), r)

    def test_catches_issue33_silent_default_shape(self) -> None:
        # The #33 trap: hparams missing -> loads at 384-dim/6-layer.
        r = self._named(drv.evaluate(self.ENTRY, {"n_layer": 6, "dim": 384}, 384, 0.72, 0.31))
        self.assertFalse(r["n_layer"])
        self.assertFalse(r["banner_dim"])
        self.assertFalse(r["vector_dim"])

    def test_catches_garbage_with_correct_dim(self) -> None:
        # Right shape, meaningless vectors: related no better than unrelated.
        r = self._named(drv.evaluate(self.ENTRY, {"n_layer": 12, "dim": 768}, 768, 0.50, 0.49))
        self.assertTrue(r["n_layer"])
        self.assertFalse(r["garbage_guard"], "margin 0.01 < 0.05 must fail")

    def test_catches_inverted_similarity(self) -> None:
        r = self._named(drv.evaluate(self.ENTRY, {"n_layer": 12, "dim": 768}, 768, 0.20, 0.80))
        self.assertFalse(r["garbage_guard"])

    def test_nan_cosines_fail_not_crash(self) -> None:
        r = self._named(drv.evaluate(self.ENTRY, {"n_layer": 12, "dim": 768}, 768, float("nan"), 0.3))
        self.assertFalse(r["garbage_guard"])

    def test_missing_banner_fails_shape_checks(self) -> None:
        r = self._named(drv.evaluate(self.ENTRY, {}, 768, 0.72, 0.31))
        self.assertFalse(r["n_layer"])
        self.assertFalse(r["banner_dim"])

    def test_cross_conversion_gate(self) -> None:
        e = dict(self.ENTRY, min_ref_cos=0.90)
        ok = self._named(drv.evaluate(e, {"n_layer": 12, "dim": 768}, 768, 0.72, 0.31, ref_cos=0.97))
        self.assertTrue(ok["cross_conversion"])
        bad = self._named(drv.evaluate(e, {"n_layer": 12, "dim": 768}, 768, 0.72, 0.31, ref_cos=0.40))
        self.assertFalse(bad["cross_conversion"], "a divergent conversion must fail")

    def test_cross_conversion_absent_when_no_ref(self) -> None:
        r = self._named(drv.evaluate(self.ENTRY, {"n_layer": 12, "dim": 768}, 768, 0.72, 0.31))
        self.assertNotIn("cross_conversion", r)


if __name__ == "__main__":
    unittest.main()
