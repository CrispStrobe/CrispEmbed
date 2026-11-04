"""Bounded runner controls; no build, model execution, or network."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
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
    def test_direct_dispatch_paths_are_disjoint_and_use_workflow_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(runner.os.environ, {"HEAVY_OUT": str(root / "out"), "HEAVY_SCRATCH": str(root / "scratch")}):
                self.assertEqual(runner.directories(), (root / "out/xiaomi0", root / "scratch/xiaomi0"))
            for out, scratch in ((root, root), (root, root / "nested"), (root / "nested", root)):
                with self.assertRaises(ValueError):
                    runner.directories(out, scratch)

    def test_resources_stop_before_dependency_installation(self):
        good = {"memory_bytes": {"MemAvailable": 12 * 2**30}, "scratch_free_bytes": 20 * 2**30,
                "output_free_bytes": 4 * 2**30, "same_filesystem": False}
        runner.require_resources(good, True)
        for key in ("memory_bytes", "scratch_free_bytes", "output_free_bytes"):
            bad = copy.deepcopy(good)
            bad[key] = {"MemAvailable": 1} if key == "memory_bytes" else 1
            with self.assertRaises(ValueError):
                runner.require_resources(bad, True)
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, "resource_snapshot", return_value=bad), \
                patch.object(runner.venv, "EnvBuilder") as builder:
            with self.assertRaises(ValueError):
                runner.bootstrap(SimpleNamespace(inference=True, model_dir=None),
                                 Path(directory) / "out", Path(directory) / "scratch")
            builder.assert_not_called()

    def test_shared_volume_reserves_scratch_and_retained_outputs_together(self):
        resources = {"memory_bytes": {"MemAvailable": 12 * 2**30},
                     "scratch_free_bytes": 13 * 2**30, "output_free_bytes": 13 * 2**30,
                     "same_filesystem": True}
        with self.assertRaises(ValueError):
            runner.require_resources(resources, True)
        resources.update(scratch_free_bytes=14 * 2**30, output_free_bytes=14 * 2**30)
        runner.require_resources(resources, True)

    def test_bootstrap_pins_dependencies_before_metadata_worker(self):
        good = {"memory_bytes": {"MemAvailable": 12 * 2**30}, "scratch_free_bytes": 20 * 2**30,
                "output_free_bytes": 4 * 2**30, "same_filesystem": False}
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, "resource_snapshot", return_value=good), \
                patch.object(runner.venv, "EnvBuilder"), \
                patch.object(runner, "run", return_value=SimpleNamespace(returncode=0)) as execute:
            root = Path(directory)
            runner.bootstrap(SimpleNamespace(inference=False, model_dir=None), root / "out", root / "scratch")
            commands = [list(map(str, call.args[0])) for call in execute.call_args_list]
            self.assertEqual(len(commands), 5)
            self.assertIn("torch==2.14.0", commands[0])
            self.assertIn("https://download.pytorch.org/whl/cpu", commands[0])
            self.assertIn("transformers==5.17.0", commands[1])
            self.assertIn("--prepared-environment", commands[-1])
            self.assertNotIn("--inference", commands[-1])
            self.assertEqual(commands[-1][0], str(root / "scratch/venv/bin/python"))

    def test_build_receipt_reads_actual_portable_flags_and_binary_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "test-xiaomi0-run").write_bytes(b"authored binary identity control, never executed")
            cache = root / "CMakeCache.txt"
            cache.write_text("GGML_NATIVE:BOOL=OFF\nCRISPEMBED_NATIVE:BOOL=OFF\n")
            receipt = runner.build_receipt(root, ["test-xiaomi0-run"], "1" * 40)
            self.assertEqual(receipt["binaries"]["test-xiaomi0-run"]["sha256"],
                             runner.policy.file_sha256(root / "test-xiaomi0-run"))
            cache.write_text("GGML_NATIVE:BOOL=ON\nCRISPEMBED_NATIVE:BOOL=OFF\n")
            with self.assertRaises(ValueError):
                runner.build_receipt(root, ["test-xiaomi0-run"], "1" * 40)

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
