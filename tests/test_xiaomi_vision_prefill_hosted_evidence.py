"""Verify retained hosted CPU vision evidence without requiring model weights."""
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "tests/results/xiaomi-ocr-0/vision-prefill-hosted-v3"
PREFIX = "prefill.vision."
EXPECTED = ([PREFIX + name for name in
             ("patch_embed", "position", "embedded", "rotary_cos", "rotary_sin")] +
            [PREFIX + f"layer.{i}" for i in range(12)] +
            [PREFIX + name for name in
             ("merger.norm", "merger.up", "merger.activation", "merger")])


class HostedVisionPrefillEvidence(unittest.TestCase):
    def setUp(self):
        self.report = json.loads((EVIDENCE / "report.json").read_text())
        self.closure = json.loads((EVIDENCE / "source-closure.json").read_text())

    def test_inputs_and_pinned_source(self):
        report = self.report
        self.assertEqual(report["phase"], "complete")
        self.assertEqual(report["source_closure"], self.closure)
        self.assertEqual(self.closure["base_commit"], "15adb672f71f8470a23db1284dafa102515adae9")
        self.assertEqual(self.closure["ggml_commit"], "a06fc0bb46b73cc10011fc90ae4054aee8438aeb")
        self.assertEqual(self.closure["staged_runner_sha256"],
                         "18bfc3e880032e00e7ceadd6f7eea5e2cc2ef3cce28d6723561e2abd155e15d2")
        self.assertEqual(report["input_sha256"], {
            "model": "1759b8da2a6fa2651de9cacdb5e90fe0551d7ab8b1fe783198f7339389451235",
            "reference": "2a58fc3fc2a4ae2d603ff600371cf71fa1f54e40ebd260d7680bfa4c1f868334"})
        self.assertEqual(report["device"], "CPU")
        self.assertIn("AUDIT_BACKEND CPU", report["backend_evidence"])
        self.assertTrue(report["negative_control_rejected"])

    def test_all_stages_recompute_the_unchanged_verdict(self):
        report = self.report
        self.assertEqual(report["thresholds"],
                         {"max_abs": 0.002, "cos_min": 0.99999, "relaxed_for_dtype": False})
        stages = report["stages"]
        self.assertEqual([row["name"] for row in stages], EXPECTED)
        passed = []
        for row in stages:
            strict = row["max_abs"] <= 0.002 and row["cos_min"] >= 0.99999
            self.assertEqual(row["pass"], strict, row["name"])
            self.assertEqual(row["strict_absolute_pass"], strict, row["name"])
            if strict:
                passed.append(row["name"])
        self.assertEqual(passed, [PREFIX + "rotary_cos", PREFIX + "rotary_sin"])
        self.assertEqual(report["stages_total"], 21)
        self.assertEqual(report["stages_passed"], 2)
        self.assertEqual(report["first_failing_stage"], PREFIX + "patch_embed")
        self.assertEqual(report["native_exit"], 1)
        self.assertEqual(report["strict_failures"], [r["name"] for r in stages if not r["pass"]])

    def test_trace_manifest_and_sanitized_log(self):
        trace = self.report["native_trace_sha256"]
        self.assertEqual(len(trace), 21)
        for row in self.report["stages"]:
            record = trace[row["name"] + ".f32"]
            self.assertEqual(record["bytes"], 4 * row["n"])
            self.assertEqual(len(bytes.fromhex(record["sha256"])), 32)
        log = (EVIDENCE / "vision-prefill.log").read_text()
        self.assertNotIn("/kaggle/input/", log)
        self.assertIn("NEGATIVE_CONTROL rejected=1", log)
        logged = [json.loads(line[len("STAGE "):]) for line in log.splitlines()
                  if line.startswith("STAGE ")]
        reported = [{key: value for key, value in row.items()
                     if key not in ("strict_absolute_pass", "max_abs_budget", "cos_min_floor")}
                    for row in self.report["stages"]]
        self.assertEqual(logged, reported)


if __name__ == "__main__":
    unittest.main()
