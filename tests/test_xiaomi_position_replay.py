"""Weight-free tests for the hosted Xiaomi-OCR-0 position attribution replay.

Scope: the staged bytes of tools/kaggle/xiaomi-ocr-0-position-replay, the pure
comparator/geometry helpers, the pinned-source declarations, and the
fail-closed verdict assembly. No torch, no checkpoint, no hosted run.
"""
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
HERE = ROOT / "tools/kaggle/xiaomi-ocr-0-position-replay"
STAGE = HERE / "stage.py"
RUNNER = HERE / "runner.py"
METADATA = HERE / "kernel-metadata.json"

SOURCE_MODULE_SHA256 = "641da34484f6b9769e28615c680cc4e3202afbf5bdbf1da7d56d5a3d678918c7"
NATIVE_POSITION_SHA256 = "7e257d92b4acfd424684fe57c716e8e077661962cf68c9f91ddb52656af9ce1c"
CHECKPOINT_SHA256 = "d4551e9f6127ac9e35cff7b4f7a72a90410e03e6c8676a36f0e2b7eaeb5d5348"
EXPORT_SHA256 = "1759b8da2a6fa2651de9cacdb5e90fe0551d7ab8b1fe783198f7339389451235"
REFERENCE_SHA256 = "2a58fc3fc2a4ae2d603ff600371cf71fa1f54e40ebd260d7680bfa4c1f868334"
V3_REPORT = ROOT / "tests/results/xiaomi-ocr-0/vision-prefill-hosted-v3/report.json"


def load(path, name="xiaomi_position_runner"):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stage(directory):
    target = Path(directory) / "staged"
    result = subprocess.run([sys.executable, str(STAGE), str(target)], text=True, capture_output=True)
    return target, result


def metric(max_abs=0.0, rms=0.0, cosine=1.0, exact=1.0, elements=8):
    return {"max_abs": max_abs, "rms": rms, "row_cos_min": cosine,
            "exact_fraction": exact, "elements": elements}


class XiaomiPositionReplayStageTest(unittest.TestCase):
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
            self.assertEqual(manifest["pinned"]["source_module_sha256"], SOURCE_MODULE_SHA256)
            self.assertEqual(manifest["pinned"]["native_position_sha256"], NATIVE_POSITION_SHA256)
            repeated, second = stage(directory)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("already exists", second.stderr)

    def test_metadata_is_cpu_only_with_pinned_kernel_sources(self):
        metadata = json.loads(METADATA.read_text())
        self.assertEqual(metadata["code_file"], "runner.py")
        self.assertEqual(metadata["kernel_type"], "script")
        self.assertEqual(metadata["enable_gpu"], "false")
        self.assertEqual(metadata["enable_internet"], "true")
        self.assertEqual(metadata["is_private"], "true")
        self.assertEqual(metadata["competition_sources"], [])
        self.assertEqual(metadata["dataset_sources"], [])
        self.assertEqual(metadata["model_sources"], [])
        self.assertIn("${KAGGLE_ACCOUNT}/crispembed-xiaomi-ocr-0-position-replay", metadata["id"])
        self.assertEqual(len(metadata["kernel_sources"]), 3)
        self.assertEqual(metadata["kernel_sources"], [
            "${KAGGLE_ACCOUNT}/crispembed-xiaomi-ocr-0-blueprint",
            "${KAGGLE_ACCOUNT}/crispembed-xiaomi-ocr-0-export-current-v2",
            "${KAGGLE_ACCOUNT}/crispembed-xiaomi-ocr-0-vision-prefill",
        ])

    def test_runner_imports_without_torch_and_pins_the_real_module(self):
        module = load(RUNNER)
        self.assertEqual(module.SCRIPT_VERSION, "xiaomi-position-replay-v1")
        self.assertEqual(module.STRICT_MAX_ABS, 0.002)
        self.assertEqual(module.STRICT_COS_MIN, 0.99999)
        self.assertEqual(module.SOURCE_MODULE_SHA256, SOURCE_MODULE_SHA256)
        self.assertEqual(module.NATIVE_POSITION_SHA256, NATIVE_POSITION_SHA256)
        self.assertEqual(module.CHECKPOINT_SHA256, CHECKPOINT_SHA256)
        self.assertEqual(module.EXPORT_SHA256, EXPORT_SHA256)
        self.assertEqual(module.REFERENCE_SHA256, REFERENCE_SHA256)
        self.assertEqual(module.GRID, [1, 62, 32])
        self.assertEqual((module.TOKENS, module.WIDTH), (1984, 768))
        source = RUNNER.read_text()
        self.assertIn("Qwen3_5VisionModel", source)
        self.assertIn("fast_pos_embed_interpolate", source)
        self.assertIn("token=False", source)
        self.assertNotIn("read_hf_token", source)
        self.assertNotIn("allow_tf32 = True", source)
        self.assertNotIn("torch.cuda.is_available() and", source)
        # No source-level short circuit of the verdict.
        self.assertNotIn('"confirmed": True', source)

    def test_runner_declares_no_parity_claim_scope(self):
        source = RUNNER.read_text()
        self.assertIn("no decoded text", source)
        self.assertIn("no speed", source)
        self.assertIn("not a native parity", source)


class ComparatorTest(unittest.TestCase):
    def setUp(self):
        self.module = load(RUNNER, "xiaomi_position_runner_cmp")

    def test_identical_arrays_pass_both_gates(self):
        values = np.array([[1.0, -2.0], [0.5, 3.0]], dtype=np.float32)
        metric_ = self.module.metrics(values, values)
        self.assertEqual(metric_["max_abs"], 0.0)
        self.assertEqual(metric_["mismatch_count"], 0)
        self.assertTrue(self.module.strict_pass(metric_))
        self.assertTrue(self.module.bf16_budget_pass(metric_))

    def test_bf16_gate_rejects_drift_that_strict_also_rejects(self):
        reference = np.full((2, 4), 1.0, dtype=np.float32)
        drift = reference.copy()
        drift[0, 0] = 1.0 + 0.0117
        metric_ = self.module.metrics(drift, reference)
        self.assertFalse(self.module.strict_pass(metric_))
        self.assertFalse(self.module.bf16_budget_pass(metric_))

    def test_worst_element_and_rms_are_located(self):
        reference = np.full((3, 5), 0.5, dtype=np.float32)
        actual = reference.copy()
        actual[1, 4] = 0.75
        metric_ = self.module.metrics(actual, reference)
        self.assertEqual(metric_["worst_index"], [1, 4])
        self.assertEqual(metric_["worst_actual"], 0.75)
        self.assertEqual(metric_["mismatch_count"], 1)

    def test_shape_mismatch_and_zero_rows_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "shapes differ"):
            self.module.metrics(np.zeros((2, 3), dtype=np.float32), np.zeros((3, 2), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "zero-magnitude"):
            self.module.metrics(np.zeros((2, 3), dtype=np.float32), np.zeros((2, 3), dtype=np.float32))

    def test_bf16_round_uses_round_to_nearest_even(self):
        values = np.array([1.0, 1.0 + 2 ** -8, 3.0 + 2 ** -6], dtype=np.float32)
        rounded = self.module.bf16_round(values)
        self.assertEqual(float(rounded[1]), 1.0)
        self.assertEqual(self.module.bf16_representable_fraction(rounded), 1.0)
        self.assertLess(self.module.bf16_representable_fraction(values + np.float32(2 ** -20)), 0.67)


class LayoutTest(unittest.TestCase):
    def setUp(self):
        self.module = load(RUNNER, "xiaomi_position_runner_layout")

    def test_position_tokens_follow_grid_thw(self):
        self.assertEqual(self.module.position_tokens([[1, 62, 32]]), 1984)
        self.assertEqual(self.module.position_tokens([[2, 4, 4], [1, 2, 2]]), 32 + 4)
        with self.assertRaisesRegex(ValueError, "grid"):
            self.module.position_tokens([[1, 62]])

    def test_num_grid_per_side_is_exact(self):
        self.assertEqual(self.module.num_grid_per_side(2304), 48)
        with self.assertRaisesRegex(ValueError, "not a perfect square"):
            self.module.num_grid_per_side(2303)

    def test_native_trace_has_contiguous_features_per_token(self):
        width, tokens = 2, 3
        native = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype="<f4")
        matrix = self.module.native_trace_matrix(native.tobytes(), tokens, width)
        self.assertEqual(matrix.shape, (tokens, width))
        np.testing.assert_array_equal(matrix, native)
        with self.assertRaisesRegex(ValueError, "native trace bytes"):
            self.module.native_trace_matrix(native.tobytes()[:-4], tokens, width)
        with self.assertRaisesRegex(ValueError, "native trace bytes"):
            self.module.native_trace_matrix(native.tobytes() + b"\x00" * 4, tokens, width)


class VerdictTest(unittest.TestCase):
    def setUp(self):
        self.module = load(RUNNER, "xiaomi_position_runner_verdict")

    def good(self, **overrides):
        values = dict(
            source_module_pinned=True, method_is_pinned=True, geometry_ok=True,
            weights_widening_equal=True, reference_bf16_representable=True,
            bf16_repeat_identical=True,
            bf16_vs_reference=metric(), f32_vs_native=metric(),
            native_vs_reference=metric(max_abs=0.0188269615, rms=0.000310289237,
                                       cosine=0.999989152),
            report_reproduction_ok=True, negative_control_rejected=True,
            report_strict_failure_reproduced=True,
        )
        values.update(overrides)
        return values

    def test_all_conditions_confirmed_together(self):
        verdict = self.module.attribution_conditions(**self.good())
        self.assertTrue(verdict["confirmed"])
        self.assertFalse(verdict["cpu_gpu_bf16_ambiguity"])
        self.assertFalse(verdict["native_deviation_beyond_precision"])
        self.assertTrue(all(verdict["conditions"].values()))

    def test_bf16_cpu_gpu_mismatch_reports_ambiguity_and_fails(self):
        verdict = self.module.attribution_conditions(
            **self.good(bf16_vs_reference=metric(max_abs=0.02, rms=5e-3, cosine=0.9998, exact=0.5)))
        self.assertFalse(verdict["confirmed"])
        self.assertTrue(verdict["cpu_gpu_bf16_ambiguity"])
        self.assertFalse(verdict["conditions"]["bf16_cpu_reproduces_frozen_reference"])

    def test_even_one_bf16_mismatch_prevents_bit_exact_reproduction_claim(self):
        verdict = self.module.attribution_conditions(
            **self.good(bf16_vs_reference=metric(max_abs=0.00001, rms=1e-6,
                                                  cosine=1.0, exact=0.999999)))
        self.assertFalse(verdict["confirmed"])
        self.assertTrue(verdict["cpu_gpu_bf16_ambiguity"])

    def test_f32_module_native_mismatch_is_reported_as_native_deviation(self):
        verdict = self.module.attribution_conditions(
            **self.good(f32_vs_native=metric(max_abs=0.02, rms=1e-3, cosine=0.9999, exact=0.0)))
        self.assertFalse(verdict["confirmed"])
        self.assertTrue(verdict["native_deviation_beyond_precision"])
        self.assertFalse(verdict["cpu_gpu_bf16_ambiguity"])

    def test_unreproduced_report_blocks_confirmation(self):
        verdict = self.module.attribution_conditions(**self.good(report_reproduction_ok=False))
        self.assertFalse(verdict["confirmed"])

    def test_missing_measurement_fails_closed(self):
        broken = self.good()
        del broken["f32_vs_native"]
        with self.assertRaisesRegex(ValueError, "missing measurement"):
            self.module.attribution_conditions(**broken)

    def test_non_finite_measurement_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.module.attribution_conditions(
                **self.good(f32_vs_native=metric(max_abs=float("nan"))))


class RetainedEvidenceTest(unittest.TestCase):
    """The mounted v3 artefact set is authenticated against the retained report."""

    def setUp(self):
        self.module = load(RUNNER, "xiaomi_position_runner_retained")
        self.report = json.loads(V3_REPORT.read_text())

    def native_path(self, directory):
        path = Path(directory) / "native-trace" / self.module.NATIVE_TRACE_NAME
        path.parent.mkdir()
        path.write_bytes(b"")
        return path

    def test_retained_report_passes_every_pin(self):
        with tempfile.TemporaryDirectory() as directory:
            stage_ = self.module.verify_report(self.report, self.native_path(directory))
        self.assertEqual(stage_["stage"]["name"], "prefill.vision.position")
        self.assertEqual(stage_["stage"]["shape"], [768, 1984])
        self.assertFalse(stage_["stage"]["pass"])

    def test_retained_report_with_a_flipped_verdict_is_rejected(self):
        report = json.loads(json.dumps(self.report))
        stage_ = next(row for row in report["stages"] if row["name"] == "prefill.vision.position")
        stage_["pass"] = True
        stage_["strict_absolute_pass"] = True
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "pass"):
                self.module.verify_report(report, self.native_path(directory))

    def test_report_without_the_pinned_native_trace_entry_is_rejected(self):
        report = json.loads(json.dumps(self.report))
        del report["native_trace_sha256"][self.module.NATIVE_TRACE_NAME]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "native trace"):
                self.module.verify_report(report, self.native_path(directory))

    def test_stage_missing_a_pinned_metric_is_rejected(self):
        report = json.loads(json.dumps(self.report))
        stage_ = next(row for row in report["stages"] if row["name"] == "prefill.vision.position")
        del stage_["rms"]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "rms"):
                self.module.verify_report(report, self.native_path(directory))

    def test_retained_report_source_pin_is_enforced(self):
        report = json.loads(json.dumps(self.report))
        report["source_closure"]["staged_runner_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "vision-prefill runner"):
                self.module.verify_report(report, self.native_path(directory))

    def test_trace_outside_the_native_trace_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / self.module.NATIVE_TRACE_NAME
            path.write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "native-trace"):
                self.module.verify_report(self.report, path)

    def test_metrics_track_float32_report_rounding(self):
        rng = np.random.default_rng(7)
        reference = (rng.standard_normal((9, 5)).astype(np.float32) * 0.1)
        actual = reference + (rng.standard_normal(reference.shape).astype(np.float32) * 1e-4)
        delta = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
        reported_max = float(np.max(delta))
        reported_mean = float(np.float32(np.mean(delta)))
        reported_rms = float(np.float32(np.sqrt(np.mean(delta * delta))))
        metric_ = self.module.metrics(actual, reference)
        self.assertTrue(self.module.close(metric_["max_abs"], reported_max))
        self.assertTrue(self.module.close(metric_["mean_abs"], reported_mean))
        self.assertTrue(self.module.close(metric_["rms"], reported_rms))
        self.assertFalse(self.module.close(metric_["max_abs"] * 1.01, reported_max))


class CompareOutputsTest(unittest.TestCase):
    """The comparison/reproduction assembly is exercised with synthetic arrays."""

    def setUp(self):
        self.module = load(RUNNER, "xiaomi_position_runner_compare")
        rng = np.random.default_rng(3)
        self.saved = (rng.standard_normal((1984, 768)).astype(np.float32) * 0.05)
        self.native = self.saved + (rng.standard_normal(self.saved.shape).astype(np.float32) * 1e-4)
        self.negative = np.ascontiguousarray(np.roll(self.saved, 993, axis=0))

    def compare(self, bf16=None, bf16_repeat=None, f32=None, negative=None):
        return self.module.compare_outputs(
            self.saved if bf16 is None else bf16,
            self.saved if bf16_repeat is None else bf16_repeat,
            self.native if f32 is None else f32,
            self.negative if negative is None else negative,
            self.saved, self.native)

    def test_comparison_set_is_complete_and_gate_annotated(self):
        result = self.compare()
        self.assertEqual(sorted(result["comparisons"]), sorted([
            "bf16_vs_frozen_reference", "bf16_repeat_vs_first", "f32_vs_native_trace",
            "f32_vs_frozen_reference", "native_vs_frozen_reference",
            "negative_shifted_rows_vs_frozen_reference"]))
        for metric_ in result["comparisons"].values():
            self.assertIn("strict_pass", metric_)
            self.assertIn("bf16_budget_pass", metric_)
        self.assertEqual(result["comparisons"]["native_vs_frozen_reference"]["shape"], [1984, 768])

    def test_identical_oracles_pass_their_gates(self):
        result = self.compare()
        self.assertTrue(result["comparisons"]["bf16_vs_frozen_reference"]["strict_pass"])
        self.assertTrue(result["comparisons"]["bf16_vs_frozen_reference"]["bf16_budget_pass"])
        self.assertTrue(result["comparisons"]["f32_vs_native_trace"]["strict_pass"])
        self.assertTrue(result["comparisons"]["bf16_repeat_vs_first"]["strict_pass"])

    def test_synthetic_data_cannot_reproduce_the_pinned_report(self):
        result = self.compare()
        self.assertFalse(result["reproduction"]["ok"])
        self.assertFalse(any(result["reproduction"]["checks"].values()))
        self.assertEqual(sorted(result["reproduction"]["checks"]), sorted([
            "max_abs", "mean_abs", "rms", "cos_min", "cos_mean", "cos_global",
            "native_norm", "reference_norm"]))
        self.assertEqual(result["reproduction"]["pinned"]["max_abs"], 0.0188269615)

    def test_negative_control_gates_both_fail(self):
        negative = self.compare()["comparisons"]["negative_shifted_rows_vs_frozen_reference"]
        self.assertFalse(negative["strict_pass"])
        self.assertFalse(negative["bf16_budget_pass"])

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "shapes differ"):
            self.compare(f32=np.zeros((1983, 768), dtype=np.float32))


class StatementTest(unittest.TestCase):
    def setUp(self):
        self.module = load(RUNNER, "xiaomi_position_runner_statement")
        self.metrics = {"max_abs": 0.001, "rms": 1e-5, "row_cos_min": 0.999999,
                        "exact_fraction": 1.0}
        self.verdict = {"confirmed": True, "cpu_gpu_bf16_ambiguity": False,
                        "native_deviation_beyond_precision": False}

    def test_confirmed_statement_names_each_measurement(self):
        text = self.module.attribution_statement(self.metrics, self.metrics, self.metrics,
                                                 self.verdict, True)
        self.assertIn("BF16 pinned-module output", text)
        self.assertIn("native F32 trace", text)
        self.assertIn("reproduces the retained report", text)
        self.assertNotIn("disagree", text)
        self.assertNotIn("does not reproduce", text)

    def test_unreproduced_report_is_stated_as_not_reproduced(self):
        text = self.module.attribution_statement(self.metrics, self.metrics, self.metrics,
                                                 self.verdict, False)
        self.assertIn("does not reproduce the retained report", text)

    def test_ambiguity_and_native_deviation_are_named(self):
        verdict = {"confirmed": False, "cpu_gpu_bf16_ambiguity": True,
                   "native_deviation_beyond_precision": True}
        text = self.module.attribution_statement(self.metrics, self.metrics, self.metrics,
                                                 verdict, True)
        self.assertIn("not bit-identical", text)
        self.assertIn("does not match the native F32 trace within the unchanged strict gate", text)


if __name__ == "__main__":
    unittest.main()
