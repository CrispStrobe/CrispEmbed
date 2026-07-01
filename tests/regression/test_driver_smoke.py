"""Tier 0: dev-box smoke test for the OCR regression driver.

No model downloads, no binary invocations, no network. Validates:

  - manifest.json schema/shape (every model has required keys, valid types)
  - parse_diff_stdout()          — pulls cos_min per stage from canned output
  - evaluate_stage_thresholds()  — per-stage / global PASS-FAIL logic
  - normalize_text() + char_error_rate() — lenient match is punctuation/
                                   case/whitespace-insensitive
  - detect_garbage()             — the colorcolor… degeneration guard

Runs in well under a second. Catches manifest typos and parser/guard
regressions in PR CI without a built binary or HF auth.

Usage:
  python -m unittest tests/regression/test_driver_smoke.py
  # or: python tests/regression/test_driver_smoke.py
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_one  # noqa: E402

MANIFEST = HERE / "manifest.json"

# A canned test-*-diff stdout blob (got-ocr shape).
CANNED_DIFF = """\
  vis_layer_0: cos_min=0.999966 max_abs=0.102619 PASS
  llm_layer_0: cos_min=0.999960 max_abs=0.173531 PASS
  llm_layer_4: cos_min=0.936000 max_abs=177.0 FAIL
"""


class TestManifest(unittest.TestCase):
    def setUp(self):
        with open(MANIFEST) as f:
            self.m = json.load(f)

    def test_top_level(self):
        self.assertIn("version", self.m)
        self.assertIn("models", self.m)
        self.assertIsInstance(self.m["models"], list)
        self.assertGreater(len(self.m["models"]), 0)

    def test_model_entries(self):
        names = set()
        for e in self.m["models"]:
            for k in ("name", "engine", "gguf", "sample"):
                self.assertIn(k, e, f"{e.get('name','?')} missing {k}")
            self.assertNotIn(e["name"], names, f"duplicate name {e['name']}")
            names.add(e["name"])
            g = e["gguf"]
            for k in ("repo", "file"):
                self.assertIn(k, g, f"{e['name']}.gguf missing {k}")
            # expected_text may be null (not captured yet) but the key
            # should be present so gaps are visible.
            self.assertIn("expected_text", e, f"{e['name']} missing expected_text key")
            if "diff" in e:
                d = e["diff"]
                self.assertIn("binary", d)
                self.assertIn("ref", d)
                self.assertIn("repo", d["ref"])
                self.assertIn("file", d["ref"])

    def test_sample_paths_exist_or_hf(self):
        # in-tree sample paths must exist; HF-hosted samples need a fixtures repo
        for e in self.m["models"]:
            p = run_one.REPO_ROOT / e["sample"]
            if not p.exists():
                self.assertIn("fixtures", self.m,
                              f"{e['name']} sample not in tree and no fixtures repo")


class TestDiffParser(unittest.TestCase):
    def test_parse(self):
        stages = run_one.parse_diff_stdout(CANNED_DIFF)
        self.assertEqual(stages["vis_layer_0"], 0.999966)
        self.assertEqual(stages["llm_layer_0"], 0.999960)
        self.assertEqual(stages["llm_layer_4"], 0.936000)

    def test_global_threshold_fail(self):
        stages = run_one.parse_diff_stdout(CANNED_DIFF)
        fails, missing = run_one.evaluate_stage_thresholds(stages, {"*": 0.999})
        self.assertEqual(len(fails), 1)
        self.assertEqual(fails[0][0], "llm_layer_4")
        self.assertEqual(missing, [])

    def test_per_stage_and_missing(self):
        stages = run_one.parse_diff_stdout(CANNED_DIFF)
        fails, missing = run_one.evaluate_stage_thresholds(
            stages, {"llm_layer_0": 0.9999, "nonexistent": 0.99})
        self.assertEqual(fails, [])
        self.assertIn("nonexistent", missing)


class TestTextMatch(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(run_one.normalize_text("The QUICK, brown  fox!"),
                         "the quick brown fox")

    def test_cer_lenient_punctuation(self):
        # punctuation/case/spacing differences → CER 0
        cer = run_one.char_error_rate("The quick brown fox jumps over the lazy dog. 12345",
                                      "the quick  brown fox jumps over the lazy dog 12345")
        self.assertLess(cer, 0.01)

    def test_cer_detects_wrong_text(self):
        cer = run_one.char_error_rate("The quick brown fox",
                                      "completely different output")
        self.assertGreater(cer, 0.3)


class TestGarbageGuard(unittest.TestCase):
    def test_flags_color_tiling(self):
        blob = "color" * 60
        self.assertIsNotNone(run_one.detect_garbage(blob))

    def test_flags_word_repetition(self):
        self.assertIsNotNone(run_one.detect_garbage(("spam " * 40).strip()))

    def test_flags_real_signature_with_junk_prefix(self):
        # the actual 3fb1f8e output: junk chars, then colorcolor…
        real = "��� 雍" + "color" * 80
        self.assertIsNotNone(run_one.detect_garbage(real))

    def test_passes_real_text(self):
        self.assertIsNone(run_one.detect_garbage(
            "The quick brown fox jumps over the lazy dog. 12345"))

    def test_passes_short_text(self):
        self.assertIsNone(run_one.detect_garbage("sin(x)"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
