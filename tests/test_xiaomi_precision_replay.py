"""Weight-free tests for the hosted Xiaomi first-stage precision replay.

Covers the staged bytes, the comparator, and the documented gate separation.
The retained hosted report is checked by test_xiaomi_precision_replay_evidence
style assertions in the same module once the run's artifacts exist.
"""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "tools/kaggle/xiaomi-ocr-0-precision-replay"
STAGE = HERE / "stage.py"
RUNNER = HERE / "runner.py"
METADATA = HERE / "kernel-metadata.json"
RESULTS = ROOT / "tests/results/xiaomi-ocr-0/precision-replay"


def load(path, name="xiaomi_precision_runner"):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stage(directory):
    target = Path(directory) / "staged"
    result = subprocess.run([sys.executable, str(STAGE), str(target)], text=True,
                            capture_output=True)
    return target, result


class XiaomiPrecisionReplayStageTest(unittest.TestCase):
    def test_stage_freezes_runner_bytes_and_refuses_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            target, result = stage(directory)
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads(result.stdout)
            staged = (target / "runner.py").read_bytes()
            self.assertEqual(hashlib.sha256(staged).hexdigest(), manifest["staged_runner_sha256"])
            self.assertEqual(manifest["files_sha256"], {"runner.py": manifest["staged_runner_sha256"]})
            self.assertEqual(staged, RUNNER.read_bytes())
            self.assertNotIn(b"__CRISPEMBED_SOURCE_BUNDLE__", staged)
            self.assertEqual(json.loads((target / "source-closure.json").read_text()), manifest)
            metadata = json.loads((target / "kernel-metadata.json").read_text())
            self.assertEqual(metadata["code_file"], "runner.py")
            self.assertEqual(metadata["is_private"], "true")
            self.assertEqual(metadata["enable_gpu"], "true")
            self.assertEqual(metadata["dataset_sources"], [])
            self.assertEqual(len(metadata["kernel_sources"]), 3)
            repeated, second = stage(directory)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("already exists", second.stderr)

    def test_runner_imports_without_torch_and_declares_the_pinned_module(self):
        module = load(RUNNER)
        self.assertEqual(module.SCRIPT_VERSION, "xiaomi-precision-replay-v1")
        self.assertEqual(module.STRICT_MAX_ABS, 0.002)
        self.assertEqual(module.STRICT_COS_MIN, 0.99999)
        self.assertEqual(module.WIDTH, 768)
        self.assertEqual(module.PATCHES, 1984)
        self.assertEqual(module.CHANNELS, 1536)
        source = RUNNER.read_text()
        self.assertIn("Qwen3_5VisionPatchEmbed", source)
        self.assertIn("module.proj.register_forward_pre_hook", source)
        self.assertIn("torch.backends.cudnn.allow_tf32 = False", source)
        self.assertIn("torch.backends.cuda.matmul.allow_tf32 = False", source)
        self.assertIn("set_float32_matmul_precision(\"highest\")", source)
        self.assertNotIn("allow_tf32 = True", source)
        self.assertNotIn("read_hf_token", source)
        self.assertIn("token=False", source)
        tree = ast.parse(source)
        self.assertTrue(any(isinstance(node, ast.FunctionDef) and node.name == "bf16_budget_pass"
                            for node in tree.body))

    def test_no_tf32_reassignment_appears_after_setup(self):
        lines = [line.strip() for line in RUNNER.read_text().splitlines()]
        setup = [index for index, line in enumerate(lines) if "cudnn.allow_tf32 = False" in line][0]
        self.assertFalse(any("allow_tf32 = True" in line for line in lines[setup:]))
        self.assertFalse(any("float32_matmul_precision(\"high" in line and "highest" not in line
                             for line in lines))


class ComparatorTest(unittest.TestCase):
    def setUp(self):
        self.module = load(RUNNER)

    def test_identical_arrays_are_exact_and_pass_both_gates(self):
        values = np.array([[1.0, -2.0], [0.5, 3.0]], dtype=np.float32)
        metric = self.module.metrics(values, values)
        self.assertEqual(metric["max_abs"], 0.0)
        self.assertEqual(metric["mismatch_count"], 0)
        self.assertEqual(metric["exact_fraction"], 1.0)
        self.assertAlmostEqual(metric["cos_global"], 1.0, places=12)
        self.assertTrue(self.module.strict_pass(metric))
        self.assertTrue(self.module.bf16_budget_pass(metric))

    def test_reference_reproduction_requires_exact_match_not_bf16_budget(self):
        reference = np.ones((20, 100), dtype=np.float32)
        drift = reference.copy()
        drift[0, 0] += np.float32(1e-5)
        metric = self.module.metrics(drift, reference)
        self.assertTrue(self.module.bf16_budget_pass(metric))
        self.assertFalse(self.module.reference_reproduced_exactly(metric))
        self.assertTrue(self.module.reference_reproduced_exactly(self.module.metrics(reference, reference)))

    def test_strict_and_bf16_gates_are_distinct(self):
        reference = np.full((2, 4), 1.0, dtype=np.float32)
        reference[0, 0] = 2.0
        # A 0.0117 offset is what the hosted native F32 stage measured against
        # the BF16-valued reference: it fails the strict gate but is inside a
        # plausible BF16 regime only if it is exact/rms-clean, which this is not.
        drift = reference.copy()
        drift[0, 0] = 1.0 + 0.0117
        metric = self.module.metrics(drift, reference)
        self.assertFalse(self.module.strict_pass(metric))
        self.assertFalse(self.module.bf16_budget_pass(metric))

    def test_worst_element_is_reported_with_location_and_values(self):
        reference = np.full((3, 5), 0.5, dtype=np.float32)
        actual = reference.copy()
        actual[1, 4] = 0.75
        metric = self.module.metrics(actual, reference)
        self.assertEqual(metric["worst_index"], [1, 4])
        self.assertEqual(metric["worst_actual"], 0.75)
        self.assertEqual(metric["worst_reference"], 0.5)
        self.assertEqual(metric["mismatch_count"], 1)
        self.assertAlmostEqual(metric["exact_fraction"], 14 / 15, places=6)
        self.assertAlmostEqual(metric["rms"], float(np.sqrt(0.25 ** 2 / 15)), places=9)

    def test_shape_mismatch_and_zero_rows_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "shapes differ"):
            self.module.metrics(np.zeros((2, 3), dtype=np.float32), np.zeros((3, 2), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "zero-magnitude"):
            self.module.metrics(np.zeros((1, 3), dtype=np.float32), np.zeros((1, 3), dtype=np.float32))

    def test_bf16_round_matches_round_to_nearest_even(self):
        values = np.array([1.0, 1.0 + 2 ** -8, 1.0 + 2 ** -8 + 2 ** -20, 3.0 + 2 ** -6],
                          dtype=np.float32)
        rounded = self.module.bf16_round(values)
        self.assertEqual(float(rounded[0]), 1.0)
        self.assertEqual(float(rounded[1]), 1.0)          # exact midpoint, ties-to-even downward
        self.assertEqual(float(rounded[2]), 1.0 + 2 ** -7)  # just above the midpoint
        self.assertEqual(float(rounded[3]), 3.0 + 2 ** -6)
        self.assertEqual(self.module.bf16_representable_fraction(rounded), 1.0)
        self.assertLess(self.module.bf16_representable_fraction(values), 1.0)

    def test_saved_activation_fixture_is_bf16_storable(self):
        # Values taken from the retained reference activation.
        sample = np.array([0.03125, 0.302734375, -0.1923828125, -0.000949859619140625],
                          dtype=np.float32)
        self.assertEqual(self.module.bf16_representable_fraction(sample), 1.0)
        perturbed = sample.copy()
        perturbed[0] = 0.03125 + 2 ** -20
        self.assertLess(self.module.bf16_representable_fraction(perturbed), 1.0)


class RetainedEvidenceTest(unittest.TestCase):
    """The hosted run must be retained before its verdict can be cited."""

    def test_hosted_report_and_comparisons_are_retained_consistent(self):
        report = json.loads((RESULTS / "report.json").read_text())
        closure = json.loads((RESULTS / "source-closure.json").read_text())
        comparisons = json.loads((RESULTS / "comparisons.json").read_text())
        self.assertEqual(report["phase"], "complete")
        self.assertEqual(report["script_version"], "xiaomi-precision-replay-v1")
        self.assertEqual(report["source_closure"], closure)
        self.assertEqual(report["strict_gate"]["max_abs"], 0.002)
        self.assertEqual(report["strict_gate"]["row_cos_min"], 0.99999)
        self.assertTrue(report["strict_gate"]["unchanged"])
        self.assertFalse(report["strict_verdict"]["decoded_parity_claimed"])
        self.assertFalse(report["strict_verdict"]["f32_conv_vs_saved_strict_pass"])
        self.assertIs(report["attribution"]["confirmed"], True)
        self.assertEqual(report["attribution"]["reference_compute_regime"],
                         "BF16 Conv3D output stored as F32")
        self.assertEqual(report["inputs"]["saved_bf16_representable_fraction"], 1.0)
        self.assertTrue(report["weights_widening"]["weight_bitwise_equal"])
        self.assertTrue(report["weights_widening"]["bias_bitwise_equal"])
        self.assertTrue(report["cast_input"]["equals_explicit_bf16_cast_of_frozen_pixels"])
        self.assertTrue(all(report["conditions"].values()))
        self.assertFalse(report["torch_flags_after_setup"]["cudnn_allow_tf32"])
        self.assertFalse(report["torch_flags_after_setup"]["matmul_allow_tf32"])
        self.assertEqual(report["torch_flags_after_setup"]["float32_matmul_precision"], "highest")
        self.assertEqual(report["exit_code"], 0)
        key = comparisons[report["key_comparison"]]
        self.assertEqual(key["mismatch_count"], 0)
        self.assertEqual(key["elements"], 1523712)
        self.assertEqual(comparisons["module_bf16_repeat_vs_first"]["mismatch_count"], 0)
        self.assertTrue(comparisons["native_cpu_vs_f32_conv"]["strict_pass"])
        self.assertFalse(comparisons["native_cpu_vs_saved"]["strict_pass"])
        self.assertEqual(comparisons["module_bf16_vs_rounded_f32"]["mismatch_count"], 269)
        self.assertFalse(comparisons["module_bf16_vs_rounded_f32"]["strict_pass"])
        self.assertTrue(key["bf16_budget_pass"])
        self.assertGreaterEqual(key["exact_fraction"], 0.99)
        self.assertLess(key["rms"], 1e-4)
        self.assertFalse(comparisons["f32_conv_vs_saved"]["strict_pass"])
        self.assertFalse(comparisons["negative_shifted_input_vs_saved"]["strict_pass"])
        self.assertFalse(comparisons["negative_shifted_input_vs_saved"]["bf16_budget_pass"])
        self.assertEqual(len(comparisons), 13 if report["native_cpu_dump"]["present"] else 10)
        self.assertTrue(report["original_module"]["module_file_sha256"])
        self.assertTrue(report["original_module"]["casts_input_to_weight_dtype"])


if __name__ == "__main__":
    unittest.main()
