"""Verify the retained hosted position replay without loading model weights."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "tests/results/xiaomi-ocr-0/position-replay-v1"


class PositionReplayEvidence(unittest.TestCase):
    def setUp(self):
        self.report = json.loads((EVIDENCE / "report.json").read_text())
        self.closure = json.loads((EVIDENCE / "source-closure.json").read_text())
        self.comparison_bytes = (EVIDENCE / "comparisons.json").read_bytes()
        self.comparisons = json.loads(self.comparison_bytes)

    def test_source_closure_and_sanitized_report(self):
        report = self.report
        self.assertEqual(report["phase"], "complete")
        self.assertEqual(report["exit_code"], 0)
        self.assertEqual(report["device"], "CPU")
        self.assertEqual(report["source_closure"], self.closure)
        self.assertEqual(self.closure["runner_sha256"],
                         "74a247c57b1bc2632d76c3c62b154534c1cef9b76cc4fde4a7989872242e0b57")
        self.assertEqual(report["comparisons_sha256"], hashlib.sha256(self.comparison_bytes).hexdigest())
        self.assertEqual(report["raw_report_sha256"],
                         "055dac556aa45ef322aabb9ba7d8357a0a41e4bbab89b10d994cfd3b484c796a")
        self.assertTrue(report["checkpoint"]["anonymous"])
        self.assertFalse(report["checkpoint"]["token"])
        self.assertTrue(report["weights_widening"]["weight_bitwise_equal"])
        self.assertTrue(report["native_trace"]["fnv1a_matches_report"])
        self.assertEqual(report["native_trace"]["sha256"],
                         "7e257d92b4acfd424684fe57c716e8e077661962cf68c9f91ddb52656af9ce1c")
        self.assertTrue(all(report["conditions"].values()))
        self.assertTrue(report["report_reproduction"]["ok"])
        text = (EVIDENCE / "report.json").read_text()
        self.assertNotIn("/kaggle/", text)
        self.assertNotIn("chr1s4", text)

    def test_measured_comparisons_and_negative_control(self):
        cases = self.comparisons
        for row in cases.values():
            strict = row["max_abs"] <= 0.002 and row["row_cos_min"] >= 0.99999
            self.assertEqual(row["strict_pass"], strict)
        bf16 = cases["bf16_vs_frozen_reference"]
        self.assertEqual(bf16["elements"], 1523712)
        self.assertEqual(bf16["mismatch_count"], 0)
        self.assertEqual(bf16["max_abs"], 0)
        self.assertTrue(cases["f32_vs_native_trace"]["strict_pass"])
        self.assertLess(cases["f32_vs_native_trace"]["max_abs"], 0.000002)
        self.assertFalse(cases["native_vs_frozen_reference"]["strict_pass"])
        self.assertFalse(cases["negative_shifted_rows_vs_frozen_reference"]["strict_pass"])
        self.assertFalse(self.report["claims"]["decoded_quality_claimed"])
        self.assertFalse(self.report["claims"]["speed_claimed"])


if __name__ == "__main__":
    unittest.main()
