"""Weight-free tests for the hosted Xiaomi-OCR-0 layer-0 attribution replay.

Scope: the staged bytes of tools/kaggle/xiaomi-ocr-0-layer0-replay, the pure
comparator/decomposition/geometry helpers, the pinned-source declarations, the
twelve-tensor widening check and the fail-closed verdict assembly. No torch, no
checkpoint, no hosted run, no CUDA.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import ast
from enum import IntEnum
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "tools/kaggle/xiaomi-ocr-0-layer0-replay"
STAGE = HERE / "stage.py"
RUNNER = HERE / "runner.py"
METADATA = HERE / "kernel-metadata.json"

SOURCE_MODULE_SHA256 = "641da34484f6b9769e28615c680cc4e3202afbf5bdbf1da7d56d5a3d678918c7"
EMBEDDED_SHA256 = "0fa3aa5a49166ae8467e582a9919dc36d4b26bda14775df9984fb95cd43f0d3b"
LAYER0_SHA256 = "dd1d3fa45703dfd92f790c7be8dd947b8c525bcb5bfc6544a911f00a567ddc37"
CHECKPOINT_SHA256 = "d4551e9f6127ac9e35cff7b4f7a72a90410e03e6c8676a36f0e2b7eaeb5d5348"
CONFIG_SHA256 = "2eb95982069a24336fe66b3a0869ea5a711aba7566787d4a599654907ecc070c"
EXPORT_SHA256 = "1759b8da2a6fa2651de9cacdb5e90fe0551d7ab8b1fe783198f7339389451235"
REFERENCE_SHA256 = "2a58fc3fc2a4ae2d603ff600371cf71fa1f54e40ebd260d7680bfa4c1f868334"
V3_REPORT = ROOT / "tests/results/xiaomi-ocr-0/vision-prefill-hosted-v3/report.json"

_LOADED = {}


def load(name="xiaomi_layer0_runner"):
    if name not in _LOADED:
        spec = importlib.util.spec_from_file_location(name, RUNNER)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _LOADED[name] = module
    return _LOADED[name]


def stage(directory):
    target = Path(directory) / "staged"
    result = subprocess.run([sys.executable, str(STAGE), str(target)], text=True, capture_output=True)
    return target, result


def metric(max_abs=0.0, rms=0.0, cosine=1.0, exact=1.0, elements=8):
    return {"max_abs": max_abs, "rms": rms, "row_cos_min": cosine,
            "exact_fraction": exact, "elements": elements}


class AttentionResolutionTest(unittest.TestCase):
    def test_eager_uses_the_model_specific_fallback_not_a_registry_entry(self):
        def eager(*args):
            pass

        def sdpa(*args):
            pass

        class Interface:
            def get_interface(self, name, default):
                return {"sdpa": sdpa}.get(name, default)

        class Modeling:
            ALL_ATTENTION_FUNCTIONS = Interface()
            eager_attention_forward = staticmethod(eager)

        resolved = load().attention_interfaces(Modeling)
        self.assertEqual(resolved["eager"], f"{eager.__module__}.{eager.__name__}")
        self.assertEqual(resolved["sdpa"], f"{sdpa.__module__}.{sdpa.__name__}")

    def test_negative_control_requires_both_thresholds_to_fail(self):
        rejected = load().negative_control_rejected
        self.assertFalse(rejected({"max_abs": 0.003, "row_cos_min": 1.0}))
        self.assertFalse(rejected({"max_abs": 0.001, "row_cos_min": 0.99}))
        self.assertTrue(rejected({"max_abs": 0.003, "row_cos_min": 0.99}))

    def test_f32_enum_is_checked_by_value_not_its_numeric_string(self):
        class QuantType(IntEnum):
            F32 = 0
            F16 = 1

        self.assertEqual(str(QuantType.F32), "0")
        check = load().require_gguf_f32_type
        self.assertEqual(check(QuantType.F32, QuantType.F32), "F32")
        with self.assertRaises(ValueError):
            check(QuantType.F16, QuantType.F32)
        with self.assertRaises(ValueError):
            check(None, QuantType.F32)

    def test_cuda_cache_is_released_through_the_module_not_a_device(self):
        module = load()

        class FakeCuda:
            calls = 0

            def empty_cache(self):
                FakeCuda.calls += 1

        class FakeTorch:
            cuda = FakeCuda()

        self.assertTrue(module.release_cuda_cache(FakeTorch))
        self.assertEqual(FakeCuda.calls, 1)
        # A bare device object has no empty_cache; that must not be what is called.
        self.assertFalse(module.release_cuda_cache(object()))


class StageTest(unittest.TestCase):
    def test_stage_freezes_exact_runner_bytes_and_refuses_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            target, result = stage(directory)
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads(result.stdout)
            staged = (target / "runner.py").read_bytes()
            self.assertEqual(hashlib.sha256(staged).hexdigest(), manifest["staged_runner_sha256"])
            self.assertEqual(staged, RUNNER.read_bytes())
            self.assertEqual(manifest["files_sha256"], {"runner.py": manifest["staged_runner_sha256"]})
            self.assertNotIn(b"__CRISPEMBED_SOURCE_BUNDLE__", staged)
            self.assertEqual(json.loads((target / "source-closure.json").read_text()), manifest)
            self.assertEqual(manifest["pinned"]["source_module_sha256"], SOURCE_MODULE_SHA256)
            self.assertEqual(manifest["pinned"]["native_embedded_sha256"], EMBEDDED_SHA256)
            self.assertEqual(manifest["pinned"]["native_layer0_sha256"], LAYER0_SHA256)
            self.assertEqual(manifest["strict_gate"], {"max_abs": 0.002, "row_cos_min": 0.99999,
                                                       "unchanged": True})
            self.assertEqual(manifest["block_tensors"]["count"], 12)
            self.assertEqual(manifest["geometry"]["cu_seqlens"], [0, 1984])
            self.assertEqual(manifest["decomposition"]["identity"], "D = B + I + G")
            repeated, second = stage(directory)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("already exists", second.stderr)

    def test_staged_bytes_import_without_torch_or_numpy(self):
        with tempfile.TemporaryDirectory() as directory:
            target, result = stage(directory)
            self.assertEqual(result.returncode, 0, result.stderr)
            blocker = (
                "import sys\n"
                "class Block:\n"
                "    def find_spec(self, name, path=None, target=None):\n"
                "        if name.split('.')[0] in {'torch', 'numpy', 'transformers', 'gguf',\n"
                "                                  'safetensors', 'huggingface_hub'}:\n"
                "            raise ImportError('blocked: ' + name)\n"
                "sys.meta_path.insert(0, Block())\n"
                "import importlib.util\n"
                f"spec = importlib.util.spec_from_file_location('staged', {str(target / 'runner.py')!r})\n"
                "module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)\n"
                "print(module.SCRIPT_VERSION)\n")
            run = subprocess.run([sys.executable, "-c", blocker], text=True, capture_output=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(run.stdout.strip(), "xiaomi-layer0-replay-v1")


class MetadataTest(unittest.TestCase):
    def test_metadata_is_gpu_internet_enabled_with_pinned_sources_and_no_datasets(self):
        metadata = json.loads(METADATA.read_text())
        self.assertEqual(metadata["id"], "${KAGGLE_ACCOUNT}/crispembed-xiaomi-ocr-0-layer0-replay")
        self.assertEqual(metadata["code_file"], "runner.py")
        self.assertEqual(metadata["kernel_type"], "script")
        self.assertEqual(metadata["language"], "python")
        self.assertEqual(metadata["enable_gpu"], "true")
        self.assertEqual(metadata["enable_internet"], "true")
        self.assertEqual(metadata["is_private"], "true")
        self.assertEqual(metadata["competition_sources"], [])
        self.assertEqual(metadata["dataset_sources"], [])
        self.assertEqual(metadata["model_sources"], [])
        self.assertEqual(metadata["kernel_sources"], [
            "${KAGGLE_ACCOUNT}/crispembed-xiaomi-ocr-0-blueprint",
            "${KAGGLE_ACCOUNT}/crispembed-xiaomi-ocr-0-export-current-v2",
            "${KAGGLE_ACCOUNT}/crispembed-xiaomi-ocr-0-vision-prefill",
        ])


class PinTest(unittest.TestCase):
    def setUp(self):
        self.module = load()

    def test_pins_and_unchanged_gate(self):
        module = self.module
        self.assertEqual(module.SCRIPT_VERSION, "xiaomi-layer0-replay-v1")
        self.assertEqual(module.STRICT_MAX_ABS, 0.002)
        self.assertEqual(module.STRICT_COS_MIN, 0.99999)
        self.assertEqual(module.SOURCE_MODULE_SHA256, SOURCE_MODULE_SHA256)
        self.assertEqual(module.CHECKPOINT_SHA256, CHECKPOINT_SHA256)
        self.assertEqual(module.CONFIG_SHA256, CONFIG_SHA256)
        self.assertEqual(module.EXPORT_SHA256, EXPORT_SHA256)
        self.assertEqual(module.REFERENCE_SHA256, REFERENCE_SHA256)
        self.assertEqual(module.NATIVE_EMBEDDED_SHA256, EMBEDDED_SHA256)
        self.assertEqual(module.NATIVE_LAYER0_SHA256, LAYER0_SHA256)
        self.assertEqual((module.TOKENS, module.WIDTH, module.HEADS, module.HEAD_DIM,
                          module.ROTARY, module.LAYERS), (1984, 768, 12, 64, 64, 12))
        self.assertEqual(module.CU_SEQLENS, [0, 1984])
        self.assertEqual(module.GRID, [1, 62, 32])

    def test_twelve_named_block_tensors_and_prefixes(self):
        layout = self.module.block_tensor_layout()
        self.assertEqual(layout["count"], 12)
        self.assertEqual(layout["names"], [
            "norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias",
            "attn.qkv.weight", "attn.qkv.bias", "attn.proj.weight", "attn.proj.bias",
            "mlp.linear_fc1.weight", "mlp.linear_fc1.bias",
            "mlp.linear_fc2.weight", "mlp.linear_fc2.bias"])
        self.assertEqual(layout["shapes"]["attn.qkv.weight"], [2304, 768])
        self.assertEqual(layout["shapes"]["mlp.linear_fc2.weight"], [768, 3072])
        self.assertEqual(layout["checkpoint"]["norm1.bias"], "model.visual.blocks.0.norm1.bias")
        self.assertEqual(layout["export"]["norm1.bias"], "v.blocks.0.norm1.bias")

    def test_runner_declares_comparison_consistency_scope(self):
        source = RUNNER.read_text()
        for phrase in ("no decoded text", "no speed", "comparison consistency",
                       "B passing does not establish that the block has no defect",
                       "does not prove all error is inherited", "token=False"):
            self.assertIn(phrase, source)
        self.assertNotIn("read_hf_token", source)
        self.assertNotIn("allow_tf32 = True", source)
        self.assertNotIn('"confirmed": True', source)
        self.assertNotIn("block_defect_ruled_out\": True", source)
        # No module-scope third-party import: every heavy import sits inside main().
        tree = ast.parse(source)
        allowed = {"hashlib", "json", "math", "os", "pathlib", "shutil",
                   "subprocess", "sys", "time", "traceback", "__future__"}
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertIn(alias.name.split(".")[0], allowed,
                                  f"module-scope third-party import: {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                self.assertIn((node.module or "").split(".")[0], allowed,
                              f"module-scope third-party import: {node.module}")

    def test_retained_v3_report_still_carries_the_pinned_layer0_failure(self):
        report = json.loads(V3_REPORT.read_text())
        stages = {row["name"]: row for row in report["stages"]}
        layer0 = stages["prefill.vision.layer.0"]
        self.assertFalse(layer0["pass"])
        self.assertFalse(layer0["strict_absolute_pass"])
        for key, pinned in self.module.LAYER0_STAGE_PINS.items():
            if key == "shape":
                self.assertEqual(layer0[key], pinned)
            elif isinstance(pinned, int):
                self.assertEqual(layer0[key], pinned)
            else:
                self.assertAlmostEqual(layer0[key], pinned, places=12)
        self.assertEqual(report["native_trace_sha256"]["prefill.vision.embedded.f32"]["sha256"],
                         EMBEDDED_SHA256)
        self.assertEqual(report["native_trace_sha256"]["prefill.vision.layer.0.f32"]["sha256"],
                         LAYER0_SHA256)

    def test_main_verdict_call_uses_exactly_every_required_measurement(self):
        tree = ast.parse(RUNNER.read_text())
        calls = [node for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "verdict"]
        self.assertEqual(len(calls), 1)
        keywords = {keyword.arg for keyword in calls[0].keywords}
        self.assertEqual(keywords, set(self.module.VERDICT_MEASUREMENTS))


class MetricsTest(unittest.TestCase):
    def setUp(self):
        self.module = load()

    def test_identical_arrays_pass_the_unchanged_gate(self):
        values = np.array([[1.0, -2.0], [0.5, 3.0]], dtype=np.float32)
        result = self.module.metrics(values, values)
        self.assertEqual(result["max_abs"], 0.0)
        self.assertEqual(result["mismatch_count"], 0)
        self.assertAlmostEqual(result["row_cos_min"], 1.0, places=15)
        self.assertTrue(result["cosines_reported"])
        self.assertTrue(self.module.strict_pass(result))

    def test_gate_requires_both_magnitude_and_row_cosine(self):
        reference = np.full((2, 4), 1.0, dtype=np.float32)
        drift = reference.copy()
        drift[0, 0] = 1.0 + 0.0021
        result = self.module.metrics(drift, reference)
        self.assertFalse(self.module.strict_pass(result))
        within = reference.copy()
        within[0, 0] = 1.0 + 0.0001
        self.assertTrue(self.module.strict_pass(self.module.metrics(within, reference)))
        rotated = reference.copy()
        rotated[0] = [1.0, 1.0, 1.0, 0.9]
        self.assertLess(self.module.metrics(rotated, reference)["row_cos_min"], 0.99999)

    def test_worst_element_is_located_and_exact_fraction_reported(self):
        reference = np.full((3, 5), 0.5, dtype=np.float32)
        actual = reference.copy()
        actual[1, 4] = 0.75
        result = self.module.metrics(actual, reference)
        self.assertEqual(result["worst_index"], [1, 4])
        self.assertEqual(result["worst_actual"], 0.75)
        self.assertEqual(result["mismatch_count"], 1)
        self.assertLess(result["exact_fraction"], 1.0)

    def test_error_terms_have_no_row_cosine_and_cannot_be_gated(self):
        reference = np.ones((2, 3), dtype=np.float32)
        error = reference.copy()
        error[0, 1] = 0.25
        result = self.module.metrics(error, reference, cosines=False)
        self.assertFalse(result["cosines_reported"])
        for key in ("row_cos_min", "row_cos_mean", "row_cos_max", "cos_global"):
            self.assertNotIn(key, result)
        with self.assertRaisesRegex(ValueError, "error terms are not gated"):
            self.module.strict_pass(result)

    def test_shape_mismatch_zero_rows_and_non_finite_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "shapes differ"):
            self.module.metrics(np.zeros((2, 3), np.float32), np.zeros((3, 2), np.float32))
        with self.assertRaisesRegex(ValueError, "zero-magnitude"):
            self.module.metrics(np.zeros((2, 3), np.float32), np.zeros((2, 3), np.float32))
        bad = np.ones((2, 3), np.float32)
        bad[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.module.metrics(bad, np.ones((2, 3), np.float32))

    def test_bf16_round_is_round_to_nearest_even(self):
        values = np.array([1.0, 1.0 + 2 ** -8, 3.0 + 2 ** -6], dtype=np.float32)
        rounded = self.module.bf16_round(values)
        self.assertEqual(float(rounded[1]), 1.0)
        self.assertEqual(self.module.bf16_representable_fraction(rounded), 1.0)
        self.assertLess(self.module.bf16_representable_fraction(values + np.float32(2 ** -20)), 0.67)


class DecomposeTest(unittest.TestCase):
    def setUp(self):
        self.module = load()

    def test_terms_satisfy_the_vector_identity_exactly(self):
        archive = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        r2 = archive + np.float32(0.5)
        r3 = r2 - np.float32(0.25)
        native = r3 + np.float32(0.125)
        result = self.module.decompose(native, archive, r2, r3)
        terms = result["terms"]
        np.testing.assert_array_equal(terms["D"], terms["B"] + terms["I"] + terms["G"])
        np.testing.assert_array_equal(terms["D"], native.astype(np.float64) - archive.astype(np.float64))
        self.assertTrue(result["report"]["identity_exact"])
        self.assertEqual(result["report"]["identity_max_abs"], 0.0)
        self.assertFalse(result["report"]["row_cosines_reported"])
        for section in ("magnitudes", "energy", "cross_terms", "cancellation"):
            self.assertIn(section, result["report"])
        self.assertNotIn("row_cos_min", json.dumps(result["report"]))

    def test_cross_terms_cancellation_and_energy_are_reported(self):
        archive = np.zeros((2, 4), dtype=np.float32)
        r2 = np.full((2, 4), 2.0, dtype=np.float32)
        r3 = np.full((2, 4), 2.5, dtype=np.float32)
        native = np.full((2, 4), 2.4, dtype=np.float32)
        result = self.module.decompose(native, archive, r2, r3)
        report = result["report"]
        self.assertEqual(set(report["cross_terms"]), {"B_I", "B_G", "I_G"})
        self.assertAlmostEqual(report["cross_terms"]["B_I"]["dot"], -0.4, delta=1e-5)
        self.assertAlmostEqual(report["cross_terms"]["B_G"]["dot"], -1.6, delta=1e-5)
        self.assertAlmostEqual(report["cross_terms"]["I_G"]["dot"], 8.0, delta=1e-5)
        self.assertGreater(report["energy"]["D"], 0)
        self.assertGreater(report["cancellation"]["ratio"], 0)
        self.assertLess(report["cancellation"]["ratio"], 1.0)  # terms are near-aligned, not cancelling
        reconstruction = (report["energy"]["B"] + report["energy"]["I"] + report["energy"]["G"]
                          + report["interaction_energy"])
        self.assertAlmostEqual(reconstruction, report["energy_reconstructed"])
        self.assertAlmostEqual(report["energy_reconstructed"],
                               float(np.square(result["terms"]["D"]).sum()), places=9)

    def test_strong_cancellation_is_quantified_not_hidden(self):
        base = np.full((2, 3), 10.0, dtype=np.float32)
        archive = base
        r2 = base + np.float32(5.0)
        r3 = r2 + np.float32(5.0)
        native = archive + np.float32(0.01)
        report = self.module.decompose(native, archive, r2, r3)["report"]
        self.assertGreater(report["cancellation"]["amplification"], 100)
        self.assertLess(report["cancellation"]["ratio"], 0.01)
        self.assertLess(report["interaction_fraction_of_D"], 0)

    def test_shape_and_finiteness_are_enforced(self):
        good = np.zeros((2, 3), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "shape"):
            self.module.decompose(good, np.zeros((3, 2), np.float32), good, good)
        bad = good.copy()
        bad[0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.module.decompose(bad, good, good, good)


class TraceTest(unittest.TestCase):
    def setUp(self):
        self.module = load()

    def test_trace_layout_is_contiguous_features_per_token(self):
        width, tokens = 2, 3
        native = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype="<f4")
        matrix = self.module.read_trace(native.tobytes(), tokens, width, "layer0")
        self.assertEqual(matrix.shape, (tokens, width))
        np.testing.assert_array_equal(matrix, native)
        with self.assertRaisesRegex(ValueError, "bytes"):
            self.module.read_trace(native.tobytes()[:-4], tokens, width, "layer0")
        with self.assertRaisesRegex(ValueError, "bytes"):
            self.module.read_trace(native.tobytes() + b"\x00" * 4, tokens, width, "layer0")

    def test_int32_cu_seqlens_and_rotary_geometry(self):
        module = self.module
        self.assertTrue(module.cu_seqlens_ok(np.array([0, 1984], dtype=np.int32)))
        self.assertFalse(module.cu_seqlens_ok(np.array([0, 1984], dtype=np.int64)))
        self.assertFalse(module.cu_seqlens_ok(np.array([1984], dtype=np.int32)))
        self.assertTrue(module.rotary_ok(np.zeros((1984, 64), np.float32),
                                         np.ones((1984, 64), np.float32)))
        with self.assertRaisesRegex(ValueError, "rotary cos shape"):
            module.rotary_ok(np.zeros((1984, 32), np.float32), np.ones((1984, 64), np.float32))


class WideningTest(unittest.TestCase):
    def setUp(self):
        self.module = load()
        self.layout = self.module.block_tensor_layout()

    def block(self):
        return {name: (np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 7)
                for name, shape in self.module.BLOCK_TENSORS}

    def widened(self):
        """A checkpoint already widened: every value is BF16-representable."""
        return {name: self.module.bf16_round(value).astype(np.float32)
                for name, value in self.block().items()}

    def test_exact_bf16_widening_passes_for_all_twelve(self):
        checkpoint = self.widened()
        values = {name: value.copy() for name, value in checkpoint.items()}
        report = self.module.widening_report(checkpoint, values)
        self.assertTrue(report["all_byte_equal"])
        self.assertEqual(report["count"], 12)
        self.assertEqual(report["mismatched"], [])
        self.assertTrue(all(row["byte_equal"] for row in report["per_name"].values()))

    def test_one_unaligned_export_tensor_fails_closed(self):
        checkpoint = self.widened()
        values = {name: value.copy() for name, value in checkpoint.items()}
        values["attn.proj.bias"] = values["attn.proj.bias"] + np.float32(2 ** -20)
        report = self.module.widening_report(checkpoint, values)
        self.assertFalse(report["all_byte_equal"])
        self.assertEqual(report["mismatched"], ["attn.proj.bias"])

    def test_missing_or_misshaped_tensor_fails_closed(self):
        checkpoint = self.widened()
        values = {name: value.copy() for name, value in checkpoint.items()}
        del checkpoint["norm1.bias"]
        with self.assertRaisesRegex(ValueError, "missing"):
            self.module.widening_report(checkpoint, values)
        checkpoint["norm1.bias"] = np.zeros((768,), np.float32)
        values["norm1.bias"] = np.zeros((1, 768), np.float32)
        with self.assertRaisesRegex(ValueError, "shape"):
            self.module.widening_report(checkpoint, values)


class ControlTest(unittest.TestCase):
    def setUp(self):
        self.module = load()

    def test_permuted_rows_change_data_and_fail_the_strict_gate(self):
        reference = np.random.default_rng(7).normal(size=(1984, 64)).astype(np.float32)
        shifted = self.module.negative_control(reference)
        self.assertEqual(shifted.shape, reference.shape)
        self.assertFalse(np.array_equal(shifted, reference))
        self.assertFalse(self.module.strict_pass(self.module.metrics(shifted, reference)))
        with self.assertRaisesRegex(ValueError, "did not change"):
            self.module.negative_control(np.zeros((1984, 64), np.float32))

    def test_retained_stage_pins_can_be_reproduced_or_not(self):
        module = self.module
        reproduced = dict(module.LAYER0_STAGE_PINS)
        reproduced.update({"max_abs": module.LAYER0_STAGE_PINS["max_abs"],
                           "row_cos_min": module.LAYER0_STAGE_PINS["cos_min"],
                           "row_cos_mean": module.LAYER0_STAGE_PINS["cos_mean"],
                           "actual_norm": module.LAYER0_STAGE_PINS["native_norm"]})
        result = module.reproduction_checks(reproduced)
        self.assertTrue(result["ok"])
        reproduced["max_abs"] = 0.04
        self.assertFalse(module.reproduction_checks(reproduced)["ok"])


class VerifyReportTest(unittest.TestCase):
    """The mounted v3 artefact set must fail closed before any file digest is read."""

    def setUp(self):
        self.module = load()
        self.stage = dict(self.module.LAYER0_STAGE_PINS)
        self.stage.update({"name": "prefill.vision.layer.0", "pass": False,
                           "strict_absolute_pass": False})
        self.embedded = {"name": "prefill.vision.embedded"}
        self.embedded.update(self.module.EMBEDDED_STAGE_PINS)
        self.report = {
            "thresholds": {"max_abs": 0.002, "cos_min": 0.99999, "relaxed_for_dtype": False},
            "input_sha256": {"model": EXPORT_SHA256, "reference": REFERENCE_SHA256},
            "stages": [dict(self.stage), dict(self.embedded)],
            "source_closure": {"staged_runner_sha256": self.module.VISION_PREFILL_RUNNER_SHA256},
            "backend_evidence": "AUDIT_BACKEND CPU grid=1,62,32 tokens=1984",
            "native_trace_sha256": {
                self.module.NATIVE_EMBEDDED_TRACE: {"sha256": EMBEDDED_SHA256,
                                                    "bytes": self.module.TRACE_BYTES},
                self.module.NATIVE_LAYER0_TRACE: {"sha256": LAYER0_SHA256,
                                                  "bytes": self.module.TRACE_BYTES},
            },
        }
        self.traces = {self.module.NATIVE_EMBEDDED_TRACE: Path("missing-embedded"),
                       self.module.NATIVE_LAYER0_TRACE: Path("missing-layer0")}

    def test_relaxed_or_changed_thresholds_are_rejected(self):
        relaxed = json.loads(json.dumps(self.report))
        relaxed["thresholds"]["relaxed_for_dtype"] = True
        with self.assertRaisesRegex(ValueError, "unchanged strict gate"):
            self.module.verify_report(relaxed, self.traces)
        changed = json.loads(json.dumps(self.report))
        changed["thresholds"]["max_abs"] = 0.003
        with self.assertRaisesRegex(ValueError, "unchanged strict gate"):
            self.module.verify_report(changed, self.traces)

    def test_different_input_pins_are_rejected(self):
        report = json.loads(json.dumps(self.report))
        report["input_sha256"]["reference"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "input pins"):
            self.module.verify_report(report, self.traces)

    def test_missing_duplicate_or_altered_layer0_stage_is_rejected(self):
        report = json.loads(json.dumps(self.report))
        report["stages"] = [self.embedded]
        with self.assertRaisesRegex(ValueError, "prefill.vision.layer.0"):
            self.module.verify_report(report, self.traces)
        report = json.loads(json.dumps(self.report))
        report["stages"].append(dict(self.stage))
        with self.assertRaisesRegex(ValueError, "prefill.vision.layer.0"):
            self.module.verify_report(report, self.traces)
        report = json.loads(json.dumps(self.report))
        report["stages"][0]["max_abs"] = 0.02
        with self.assertRaisesRegex(ValueError, "max_abs"):
            self.module.verify_report(report, self.traces)
        report = json.loads(json.dumps(self.report))
        report["stages"][0]["pass"] = True
        with self.assertRaisesRegex(ValueError, "different vision-prefill runner|strict pass|missing"):
            self.module.verify_report(report, self.traces)

    def test_compromised_runner_or_backend_evidence_is_rejected(self):
        report = json.loads(json.dumps(self.report))
        report["source_closure"]["staged_runner_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "different vision-prefill runner"):
            self.module.verify_report(report, self.traces)
        report = json.loads(json.dumps(self.report))
        report["backend_evidence"] = "AUDIT_BACKEND CUDA grid=1,62,32 tokens=1984"
        with self.assertRaisesRegex(ValueError, "backend evidence"):
            self.module.verify_report(report, self.traces)


class VerdictTest(unittest.TestCase):
    def setUp(self):
        self.module = load()

    def decomposition(self, residual=0.0):
        return {"identity_max_abs": residual, "identity_tolerance": 1e-9,
                "energy": {"D": 1.0, "B": 0.5, "I": 0.25, "G": 0.25},
                "cross_terms": {"B_I": {"dot": 0.1}, "B_G": {"dot": 0.1},
                                "I_G": {"dot": 0.1}},
                "cancellation": {"ratio": 0.5, "amplification": 2.0, "norm_D": 1.0},
                "magnitudes": {"D": {"max_abs": 0.03}},
                "interaction_fraction_of_D": 0.6}

    def good(self, **overrides):
        values = dict(
            source_pins_ok=True, weights_widening_ok=True, geometry_ok=True,
            reference_bf16_representable=True, archive_embedded_bf16_representable=True,
            repeat_identical=True,
            device_matches_archive=True,
            decomposition=self.decomposition(),
            r1_vs_archive=metric(), r2_vs_archive=metric(),
            r3_vs_native=metric(), r4_vs_native=metric(),
            negative_control_rejected=True, report_reproduction_ok=True,
            strict_failure_reproduced=True,
        )
        values.update(overrides)
        return values

    def test_all_conditions_confirmed_together(self):
        result = self.module.verdict(**self.good())
        self.assertTrue(result["confirmed"])
        self.assertFalse(result["cpu_gpu_backend_ambiguity"])
        self.assertTrue(all(result["conditions"].values()))
        self.assertTrue(result["comparison_consistency_only"])
        self.assertFalse(result["block_defect_ruled_out"])
        self.assertIsNone(result["sole_causal_label"])
        self.assertTrue(result["b_pass_does_not_prove_no_block_defect"])
        self.assertIn("does not establish that the block has no defect", result["statement"])

    def test_non_bitwise_archive_closure_reports_explicit_ambiguity(self):
        result = self.module.verdict(**self.good(
            r1_vs_archive=metric(max_abs=1e-5, rms=2e-7, cosine=1.0, exact=0.999)))
        self.assertFalse(result["confirmed"])
        self.assertTrue(result["cpu_gpu_backend_ambiguity"])
        self.assertFalse(result["conditions"]["archive_closure_is_bitwise_exact"])
        self.assertFalse(result["block_defect_ruled_out"])
        self.assertIn("CPU/GPU/backend", result["statement"])

    def test_non_representable_inputs_leave_ambiguity(self):
        result = self.module.verdict(**self.good(archive_embedded_bf16_representable=False))
        self.assertFalse(result["confirmed"])
        self.assertTrue(result["cpu_gpu_backend_ambiguity"])

    def test_non_t4_host_leaves_ambiguity(self):
        result = self.module.verdict(**self.good(device_matches_archive=False))
        self.assertFalse(result["confirmed"])
        self.assertTrue(result["cpu_gpu_backend_ambiguity"])
        self.assertFalse(result["conditions"]["hosted_device_matches_retained_t4"])

    def test_unrejected_negative_control_fails_the_verdict(self):
        result = self.module.verdict(**self.good(negative_control_rejected=False))
        self.assertFalse(result["confirmed"])
        self.assertFalse(result["conditions"]["negative_control_fails_both_gates"])

    def test_identity_residual_beyond_tolerance_fails_the_verdict(self):
        result = self.module.verdict(**self.good(decomposition=self.decomposition(1e-6)))
        self.assertFalse(result["confirmed"])
        self.assertFalse(result["conditions"]["decomposition_identity_within_tolerance"])

    def test_failed_retained_failure_reproduction_blocks_confirmation(self):
        result = self.module.verdict(**self.good(strict_failure_reproduced=False))
        self.assertFalse(result["confirmed"])
        result = self.module.verdict(**self.good(report_reproduction_ok=False))
        self.assertFalse(result["confirmed"])

    def test_widening_or_pin_failure_blocks_confirmation(self):
        self.assertFalse(self.module.verdict(**self.good(weights_widening_ok=False))["confirmed"])
        self.assertFalse(self.module.verdict(**self.good(source_pins_ok=False))["confirmed"])

    def test_r4_may_be_an_explicit_not_run_record(self):
        result = self.module.verdict(**self.good(r4_vs_native={"status": "not_run"}))
        self.assertTrue(result["confirmed"])
        self.assertIn("R4 BF16 CPU control was not run", result["statement"])

    def test_missing_measurement_fails_closed(self):
        broken = self.good()
        del broken["r3_vs_native"]
        with self.assertRaisesRegex(ValueError, "missing measurement"):
            self.module.verdict(**broken)

    def test_unexpected_measurement_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "unexpected measurement"):
            self.module.verdict(**self.good(extra=True))

    def test_non_finite_measurement_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.module.verdict(**self.good(r3_vs_native=metric(max_abs=float("nan"))))

    def test_negative_measurement_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.module.verdict(**self.good(r3_vs_native=metric(rms=-1.0)))

    def test_incomplete_decomposition_fails_closed(self):
        broken = self.decomposition()
        del broken["cross_terms"]
        with self.assertRaisesRegex(ValueError, "decomposition.cross_terms"):
            self.module.verdict(**self.good(decomposition=broken))

    def test_error_term_cosine_is_never_a_gate(self):
        ungated = self.module.metrics(np.eye(2, dtype=np.float32),
                                      np.eye(2, dtype=np.float32), cosines=False)
        with self.assertRaisesRegex(ValueError, "r3_vs_native.row_cos_min"):
            self.module.verdict(**self.good(r3_vs_native=ungated))
        with self.assertRaisesRegex(ValueError, "error terms are not gated"):
            self.module.strict_pass(ungated)

    def test_end_to_end_pure_assembly_is_json_serializable(self):
        module = self.module
        rng = np.random.default_rng(11)
        archive = rng.normal(size=(16, 768)).astype(np.float32)
        native = (archive + rng.normal(scale=1e-3, size=archive.shape)).astype(np.float32)
        r2 = (archive + np.float32(2e-4)).astype(np.float32)
        r3 = (native - np.float32(5e-5)).astype(np.float32)
        decomposition = module.decompose(native, archive, r2, r3)["report"]
        comparisons = {
            "r1_vs_archive": module.metrics(archive, archive),
            "r2_vs_archive": module.metrics(r2, archive),
            "r3_vs_native": module.metrics(r3, native),
            "r4_vs_native": module.metrics(r3, native),
            "negative": module.metrics(module.negative_control(r3, shift=5), native),
        }
        result = module.verdict(
            source_pins_ok=True, weights_widening_ok=True, geometry_ok=True,
            reference_bf16_representable=True, archive_embedded_bf16_representable=True,
            repeat_identical=True, device_matches_archive=True,
            decomposition=decomposition,
            r1_vs_archive=comparisons["r1_vs_archive"],
            r2_vs_archive=comparisons["r2_vs_archive"],
            r3_vs_native=comparisons["r3_vs_native"],
            r4_vs_native=comparisons["r4_vs_native"],
            negative_control_rejected=not module.strict_pass(comparisons["negative"]),
            report_reproduction_ok=True, strict_failure_reproduced=True,
        )
        self.assertTrue(result["confirmed"])
        self.assertTrue(result["conditions"]["decomposition_identity_within_tolerance"])
        payload = json.dumps({"verdict": result, "decomposition": decomposition}, allow_nan=False)
        self.assertIn("comparison consistency only", payload)


if __name__ == "__main__":
    unittest.main()
