"""Weight-free tests for the hosted Surya OCR 2.5 convert/quantize kernel.

Scope: the staged bytes of tools/kaggle/surya-ocr-25-quant, its pure helpers
(case table, type census, arm comparison, commit rendering) and the pinned
declarations. No torch, no gguf, no model bytes, no hosted run.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "tools/kaggle/surya-ocr-25-quant"
RUNNER = HERE / "run.py"
STAGE = HERE / "stage.py"
METADATA = HERE / "kernel-metadata.json"

REVISION = "00c843e9cae45611dca0a6dc2c9bbbf2dadaaed3"
CHECKPOINT_SHA256 = "6940baca2d16a20b15b7302cac7253246c78c95469920fcdb064963f48b5cb17"

_LOADED = {}


def load(name="surya25_kaggle_quant_run"):
    if name not in _LOADED:
        spec = importlib.util.spec_from_file_location(name, RUNNER)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _LOADED[name] = module
    return _LOADED[name]


class CaseTableTest(unittest.TestCase):
    def test_every_case_is_grid_aligned_inside_the_scale_to_fit_bounds(self):
        module = load()
        cases = module.CASE_TABLE
        self.assertGreaterEqual(len(cases), 6)
        self.assertEqual(len({name for name, _, _, _ in cases}), len(cases))
        for name, width, height, lines in cases:
            self.assertEqual(width % 28, 0, f"{name}: width {width} is off the 28-px grid")
            self.assertEqual(height % 28, 0, f"{name}: height {height} is off the 28-px grid")
            area = width * height
            self.assertGreaterEqual(area, module.MIN_AREA, f"{name}: {area} below the package floor")
            self.assertLessEqual(area, module.MAX_AREA, f"{name}: {area} above the package ceiling")
            self.assertTrue(lines, f"{name}: needs at least one text line")

    def test_cases_stay_under_the_unwired_patch_bound(self):
        module = load()
        for name, width, height, _ in module.CASE_TABLE:
            # patch 16, spatial merge 2 -> 8x8 per merged token
            patches = (width // 16) * (height // 16)
            self.assertLessEqual(patches, module.MAX_PATCHES,
                                 f"{name}: {patches} patches exceeds the native bound")


class TypeCensusTest(unittest.TestCase):
    def census(self):
        return {
            "v.blk.0.attn_qkv.weight": "Q8_0",
            "v.blk.0.attn_qkv.bias": "F32",
            "v.blk.0.norm1.weight": "F32",
            "l.0.ffn_up.weight": "Q4_K",
        }

    def test_protected_norms_and_biases_kept_at_f32_pass(self):
        module = load()
        self.assertEqual(module.census_violations(self.census()), [])

    def test_a_quantized_norm_is_reported(self):
        module = load()
        broken = dict(self.census(), **{"v.blk.0.norm1.weight": "Q8_0"})
        self.assertEqual(module.census_violations(broken), ["v.blk.0.norm1.weight"])

    def test_a_quantized_bias_is_reported(self):
        module = load()
        broken = dict(self.census(), **{"v.blk.0.attn_qkv.bias": "Q4_K"})
        self.assertEqual(module.census_violations(broken), ["v.blk.0.attn_qkv.bias"])

    def test_summary_counts_types_and_reports_the_matrix_count(self):
        module = load()
        summary = module.census_summary(self.census())
        self.assertEqual(summary["tensors"], 4)
        self.assertEqual(summary["by_type"], {"F32": 2, "Q8_0": 1, "Q4_K": 1})
        self.assertEqual(summary["quantized_matrices"], 2)


class ArmComparisonTest(unittest.TestCase):
    def reference(self):
        return {"token_ids": [5, 6, 7, 2], "text": "<p>x</p>", "stop": "EOS"}

    def test_identical_arm_reports_exact(self):
        module = load()
        row = module.compare_arm(self.reference(), dict(self.reference()))
        self.assertTrue(row["ids_exact"])
        self.assertTrue(row["text_exact"])
        self.assertIsNone(row["first_divergent_step"])

    def test_first_divergent_step_is_reported(self):
        module = load()
        arm = {"token_ids": [5, 9, 7, 2], "text": "<p>y</p>", "stop": "EOS"}
        row = module.compare_arm(self.reference(), arm)
        self.assertFalse(row["ids_exact"])
        self.assertEqual(row["first_divergent_step"], 1)
        self.assertFalse(row["text_exact"])

    def test_truncated_arm_diverges_at_the_shorter_length(self):
        module = load()
        arm = {"token_ids": [5, 6, 7], "text": "<p>x</p>", "stop": "cap"}
        row = module.compare_arm(self.reference(), arm)
        self.assertFalse(row["ids_exact"])
        self.assertEqual(row["first_divergent_step"], 3)
        self.assertEqual(row["stop"], "cap")


class StageTest(unittest.TestCase):
    def test_stage_renders_the_commit_and_refuses_holders(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "staged"
            result = subprocess.run([sys.executable, str(STAGE), str(target),
                                     "--commit", "0123456789abcdef0123456789abcdef01234567"],
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads(result.stdout)
            kernel = (target / "kernel.py").read_text()
            self.assertNotIn("@COMMIT@", kernel)
            self.assertIn("0123456789abcdef0123456789abcdef01234567", kernel)
            self.assertEqual(manifest["commit"], "0123456789abcdef0123456789abcdef01234567")
            self.assertEqual(manifest["pinned"]["revision"], REVISION)
            self.assertEqual(manifest["pinned"]["checkpoint_sha256"], CHECKPOINT_SHA256)
            self.assertEqual(manifest["arms"], ["f16", "q8_0", "q4_k"])
            staged = (target / "run.py").read_bytes()
            self.assertEqual(hashlib.sha256(staged).hexdigest(), manifest["files_sha256"]["run.py"])
            self.assertEqual(staged, RUNNER.read_bytes())
            again = subprocess.run([sys.executable, str(STAGE), str(target),
                                    "--commit", "0123456789abcdef0123456789abcdef01234567"],
                                   text=True, capture_output=True)
            self.assertNotEqual(again.returncode, 0)


class MetadataTest(unittest.TestCase):
    def test_cpu_kernel_with_internet_and_no_datasets(self):
        metadata = json.loads(METADATA.read_text())
        # Kaggle derives the slug from the title; keep both identical so a later
        # push updates this kernel instead of creating a second one.
        self.assertEqual(metadata["id"], "${KAGGLE_ACCOUNT}/crispembed-surya-ocr-25-convert-and-quantize")
        self.assertEqual(metadata["title"], "crispembed-surya-ocr-25-convert-and-quantize")
        self.assertEqual(metadata["code_file"], "kernel.py")
        self.assertEqual(metadata["kernel_type"], "script")
        self.assertEqual(metadata["enable_gpu"], "false")
        self.assertEqual(metadata["enable_internet"], "true")
        self.assertEqual(metadata["is_private"], "true")
        self.assertEqual(metadata["dataset_sources"], [])
        self.assertEqual(metadata["model_sources"], [])


if __name__ == "__main__":
    unittest.main()
