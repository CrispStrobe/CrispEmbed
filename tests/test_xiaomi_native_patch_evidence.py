"""Retained-evidence checks for the native patch projection and its attribution."""
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "tests/results/xiaomi-ocr-0"


def load(name):
    return json.loads((RESULTS / name).read_text())


class NativePatchEvidenceTest(unittest.TestCase):
    def test_cpu_and_cuda_agree_on_the_same_f32_inputs(self):
        report = load("native-patch-v1-report.json")
        self.assertEqual(report["phase"], "complete")
        self.assertIs(report["native_cuda_engine"], False)
        self.assertIn("decoded and full vision parity not tested", report["status"])
        self.assertEqual(report["device"], "Tesla T4")
        self.assertEqual(report["cuda_capability"], [7, 5])
        cross = report["cpu_vs_cuda_f32"]
        self.assertTrue(cross["pass"])
        self.assertLessEqual(cross["max_abs"], 0.002)
        self.assertGreaterEqual(cross["cos_min"], 0.99999)
        # Both arms must still fail the strict gate against the BF16 reference.
        self.assertEqual(report["cpu_vs_original_strict_exit"], 1)
        self.assertEqual(report["cuda_vs_original_strict_exit"], 1)
        self.assertNotEqual(report["output_sha256"]["cpu"], report["output_sha256"]["cuda"])

    def test_attribution_separates_operand_precision_from_output_precision(self):
        record = load("native-patch-v1-attribution.json")
        controls = record["controls"]
        native, operands, rounded = (controls["f32_operands_and_output"],
                                     controls["bf16_operands_f32_output"],
                                     controls["bf16_operands_and_output"])
        hosted = record["hosted_native_f32_vs_original"]
        # The recomputed F32 control must reproduce the hosted native measurement.
        self.assertAlmostEqual(hosted["max_abs"], native["max_abs"], delta=1e-6)
        self.assertAlmostEqual(hosted["rms"], native["rms"], delta=1e-9)
        self.assertAlmostEqual(hosted["cos_min"], native["cos_min"], delta=1e-7)
        self.assertFalse(hosted["strict_pass"])
        # These reconstructed controls jointly change precision. They suggest
        # a reduced-precision effect but do not isolate the original Conv3D.
        self.assertGreater(rounded["exact_fraction"], 0.999)
        self.assertLess(rounded["max_abs"], operands["max_abs"])
        self.assertLess(rounded["rms"], operands["rms"])
        self.assertLess(operands["max_abs"], native["max_abs"])
        self.assertAlmostEqual(record["hosted"]["cpu_vs_cuda"]["max_abs"], 2.384185791015625e-06)
        self.assertEqual(record["reference_sha256"],
                         "2a58fc3fc2a4ae2d603ff600371cf71fa1f54e40ebd260d7680bfa4c1f868334")
        self.assertEqual(record["model_sha256"],
                         "1759b8da2a6fa2651de9cacdb5e90fe0551d7ab8b1fe783198f7339389451235")


if __name__ == "__main__":
    unittest.main()
