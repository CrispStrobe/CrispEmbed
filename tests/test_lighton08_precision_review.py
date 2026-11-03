#!/usr/bin/env python3
"""Controls for complete retained-logit coverage and strict owner bars."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("lighton_review", ROOT / "tools/lighton08_precision_review.py")
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


class ReviewControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frozen = json.loads((review.RESULTS / "quality.json").read_text())["cases"][0]

    def setUp(self):
        self.case = copy.deepcopy(self.frozen)

    def run_case(self, profile="f16"):
        return review.case_review(self.case["id"], self.case, self.case["reference"], profile)

    def test_actual_full_vocabulary_receipt(self):
        result = self.run_case()
        self.assertEqual(result["teacher_forced_positions"], 473)
        self.assertTrue(result["exceeds_bar_on_recorded_build"])

    def test_missing_late_logit_rejected(self):
        self.case["decode"]["metrics"].pop()
        with self.assertRaisesRegex(ValueError, "full-vocabulary"):
            self.run_case()

    def test_topk_is_not_full_vocabulary(self):
        self.case["decode"]["metrics"][-1]["elements"] = 20
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            self.run_case()

    def test_cap_cannot_be_changed(self):
        self.case["reference"]["max_new_tokens"] = 2048
        with self.assertRaisesRegex(ValueError, "cap"):
            self.run_case()

    def test_owner_bar_is_strict(self):
        self.case["decode"]["metrics"][-1]["cos_min"] = 0.94
        self.assertFalse(self.run_case("q4")["exceeds_bar_on_recorded_build"])

    def test_q6_mix_has_no_invented_policy(self):
        self.assertIsNone(self.run_case("q8-q6")["exceeds_bar_on_recorded_build"])


if __name__ == "__main__":
    unittest.main()
