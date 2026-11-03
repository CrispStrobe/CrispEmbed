"""Bounded runner controls; no build, model execution, or network."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load("xiaomi0_runner_test", ROOT / "tools/ci-heavy/xiaomi0_reference.py")
layout = load("xiaomi0_layout_test", ROOT / "tools/xiaomi0_layout_contract.py")


class XiaomiRunnerTest(unittest.TestCase):
    def test_frozen_budget_and_no_unrelated_gates(self):
        protocol = runner.preflight()
        self.assertEqual([c["task"] for c in protocol["cases"]], ["text", "table"])
        self.assertEqual(sum(c["max_new_tokens"] for c in protocol["cases"]), 32)
        self.assertEqual(protocol["dtype_arms"], ["f32", "f16"])
        with patch.dict(runner.os.environ, {"CRISPEMBED_QWEN35_QUANT": "1", "CRISPEMBED_OVISOCR3_FRONTEND": "1"}):
            env = runner.native_env()
            self.assertNotIn("CRISPEMBED_QWEN35_QUANT", env)
            self.assertNotIn("CRISPEMBED_OVISOCR3_FRONTEND", env)
            self.assertEqual(env["CRISPEMBED_BUILD_MODE"], "portable")
            self.assertEqual(env["CRISPEMBED_XIAOMI_OCR0_INFERENCE"], "1")

    def detector(self):
        evidence = layout.policy.evidence()
        return {"model": evidence["layout"]["model"], "revision": evidence["layout"]["revision"],
                "checkpoint_files": evidence["layout"]["checkpoint_files"], "threshold": 0.3,
                "forward_executed": True, "source_revision": evidence["upstream"]["revision"]}

    def test_actual_layout_identity_and_output_contract(self):
        regions = [{"index": 0, "label": "text", "task_type": "text", "bbox_2d": [10, 20, 400, 250]},
                   {"index": 1, "label": "table", "task_type": "table", "bbox_2d": [20, 300, 950, 900]},
                   {"index": 2, "label": "chart", "task_type": "skip", "bbox_2d": [700, 10, 980, 280]}]
        result = layout.region_requests(regions, self.detector())
        self.assertEqual(result[0]["bbox"], [.01, .02, .4, .25])
        self.assertEqual(result[1]["prompt"], "Parse the table in the image into OTSL.")
        self.assertFalse(result[2]["ocr_required"])
        self.assertIsNone(result[2]["prompt"])
        self.assertNotEqual(result[0]["native_prompt_order"], result[0]["original_vllm_prompt_order"])
        for key, value in (("model", "generic-layout"), ("revision", "main"), ("forward_executed", False),
                           ("threshold", .25), ("checkpoint_files", {})):
            detector = self.detector(); detector[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                layout.region_requests(regions, detector)
        with self.assertRaises(ValueError):
            layout.region_requests(regions, self.detector(), maximum=2)
        for key, value in (("index", 1), ("label", "unknown"), ("task_type", "formula"),
                           ("bbox_2d", [0, 0, 1001, 800]), ("bbox_2d", [0, 0, float("nan"), 5])):
            changed = copy.deepcopy(regions); changed[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                layout.region_requests(changed, self.detector())

    def test_layout_does_not_run_or_accept_generic_weights_locally(self):
        with patch.dict(layout.os.environ, {"GITHUB_ACTIONS": "false"}):
            with self.assertRaisesRegex(ValueError, "hosted CI"):
                layout.detect("image.png", "layout", "source")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                layout.verify_layout_assets(directory)

    def test_numeric_metrics_do_not_accept_high_cosine_alone_as_strict_pass(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("optional numpy comparison controls")
        original = np.asarray([[1, 2, 3], [3, 2, 1]], dtype=np.float32)
        scaled = original * 2
        result = runner.metrics(scaled, original)
        self.assertAlmostEqual(result["cosine_min"], 1)
        self.assertFalse(result["strict_pass"])
        self.assertEqual(result["max_abs"], 3)
        self.assertTrue(runner.metrics(original, original)["strict_pass"])
        with self.assertRaises(ValueError):
            runner.metrics(np.asarray([[float("nan")]]), np.asarray([[1.0]]))

    def test_fixture_and_original_format_are_not_benchmark_receipts(self):
        protocol = runner.preflight()
        self.assertFalse(protocol["full_document_qualification"])
        self.assertFalse(protocol["vllm_qualification"])
        self.assertEqual(protocol["layout_required_for_document_benchmark"], "PP-DocLayoutV3")
        source = (ROOT / "tools/ci-heavy/xiaomi0_reference.py").read_text()
        self.assertIn('"-DGGML_NATIVE=OFF"', source)
        self.assertIn('"-DCRISPEMBED_NATIVE=OFF"', source)
        self.assertIn('output_logits=True', source)
        self.assertIn('"precision_qualified": False', source)


if __name__ == "__main__":
    unittest.main()
