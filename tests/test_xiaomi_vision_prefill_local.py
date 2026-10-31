"""Consistency checks for the retained local composed-vision run.

The run is a measurement, not a pass: 19 of 21 stages miss the strict absolute
gate. These assertions pin what was actually observed so a later edit cannot
quietly turn a failing measurement into a claimed parity result.
"""
import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "tests/results/xiaomi-ocr-0/vision-prefill-local-v1"
SUMMARY, LOG = RESULTS / "summary.json", RESULTS / "vision-prefill.log"

EXPECTED = (["prefill.vision.patch_embed", "prefill.vision.position", "prefill.vision.embedded",
             "prefill.vision.rotary_cos", "prefill.vision.rotary_sin"] +
            [f"prefill.vision.layer.{i}" for i in range(12)] +
            ["prefill.vision.merger.norm", "prefill.vision.merger.up",
             "prefill.vision.merger.activation", "prefill.vision.merger"])


class RetainedVisionPrefillTest(unittest.TestCase):
    def setUp(self):
        self.summary = json.loads(SUMMARY.read_text())
        self.rows = {row["name"]: row for row in self.summary["stages"]}

    def test_stage_set_and_order_are_the_full_composed_tower(self):
        self.assertEqual([row["name"] for row in self.summary["stages"]], EXPECTED)
        self.assertEqual(self.summary["stages_total"], 21)
        self.assertEqual(self.summary["stages_passed"] + self.summary["stages_failed"], 21)

    def test_verdicts_recompute_from_the_literal_gate(self):
        for row in self.summary["stages"]:
            strict = row["max_abs"] <= 0.002 and row["cos_min"] >= 0.99999
            self.assertEqual(strict, row["pass"], row["name"])
        passed = [row["name"] for row in self.summary["stages"] if row["pass"]]
        self.assertEqual(passed, ["prefill.vision.rotary_cos", "prefill.vision.rotary_sin"])
        self.assertEqual(self.summary["first_failing_stage"], "prefill.vision.patch_embed")
        self.assertEqual(self.summary["native_exit"], 1)

    def test_rotary_matches_and_patch_embed_keeps_its_bf16_gap(self):
        self.assertLess(self.rows["prefill.vision.rotary_cos"]["max_abs"], 1e-6)
        self.assertLess(self.rows["prefill.vision.rotary_sin"]["max_abs"], 1e-6)
        patch = self.rows["prefill.vision.patch_embed"]
        self.assertAlmostEqual(patch["max_abs"], 0.011676, places=5)
        self.assertFalse(patch["pass"])

    def test_large_errors_remain_visible(self):
        for name in ("prefill.vision.layer.11", "prefill.vision.merger.norm"):
            row = self.rows[name]
            # A max element error divided by the whole-tensor L2 norm is not
            # an elementwise relative error or a BF16 ULP measurement.
            self.assertGreater(row["max_abs"], 0.002, name)
            self.assertLess(row["cos_global"], 1.0)
            self.assertFalse(row["pass"])

    def test_scope_and_link_caveat_are_recorded(self):
        self.assertFalse(self.summary["gate"]["relaxed"])
        self.assertTrue(self.summary["negative_control_rejected"])
        self.assertIn("not the pinned ggml", self.summary["link_caveat"])
        self.assertIn("decoded OCR text", self.summary["not_tested"])
        log = LOG.read_bytes()
        self.assertEqual(hashlib.sha256(log).hexdigest(), self.summary["log_sha256"])
        self.assertIn(b"NEGATIVE_CONTROL rejected=1", log)


if __name__ == "__main__":
    unittest.main()
