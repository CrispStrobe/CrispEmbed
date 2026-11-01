#!/usr/bin/env python3
"""Stdlib controls for explicit original-student import routing and provenance."""

import hashlib
import importlib
from pathlib import Path
import sys
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import latent_inpaint_original_import as adapter


class OriginalStudentImportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.source = Path(self.temporary.name)
        self.previous_path = sys.path[:]
        sys.path.insert(0, str(self.source))
        self.paths = {}
        self.write("model_lib/__init__.py", "raise AssertionError('unused teacher initializer executed')\n")
        self.write("model_lib/nets/unet_lambda_prune_lite.py",
                   "class " + adapter.STUDENT_CLASS + ":\n    def numerical_identity(self):\n        return 71\n")
        self.write("model_lib/nets/unet_lambda_dwconv_blocks.py",
                   "def custom_get_down_block():\n    return 'original-block-object'\n")
        self.write("removal/v1_2/__init__.py", "raise AssertionError('training dataset initializer executed')\n")
        self.write("removal/v1_2/removal_model.py",
                   "from model_lib import *\nRemovalModel = " + adapter.STUDENT_CLASS + "\n")
        self.write("removal/v1_2/pipeline.py", "from .removal_model import RemovalModel\nimport utils_infer\n")
        self.write("utils_infer.py", "def predict_noise(value):\n    return value\n")

    def tearDown(self):
        for name in tuple(sys.modules):
            if adapter._is_source_package(name):
                del sys.modules[name]
        sys.path[:] = self.previous_path
        importlib.invalidate_caches()
        self.temporary.cleanup()

    def write(self, name, contents):
        path = self.source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = contents.encode("utf-8")
        path.write_bytes(raw)
        self.paths[name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}

    def contract(self):
        return {"source": {"moebius": {"files": self.paths}},
                "models": {"moebius": {"source_model": "moebius"}}}

    def test_imports_original_objects_without_unrelated_initializers(self):
        constructor = adapter.prepare_moebius_cpu_imports(self.source, self.contract())
        original = importlib.import_module(adapter.STUDENT_MODULE)
        blocks = importlib.import_module(adapter.BLOCK_MODULE)
        pipeline = importlib.import_module("removal.v1_2.pipeline")
        self.assertIs(constructor, getattr(original, adapter.STUDENT_CLASS))
        self.assertIs(pipeline.RemovalModel, constructor)
        self.assertEqual(constructor().numerical_identity(), 71)
        self.assertIs(sys.modules["model_lib"].custom_get_down_block, blocks.custom_get_down_block)
        self.assertEqual(Path(original.__file__), self.source / "model_lib/nets/unet_lambda_prune_lite.py")
        self.assertNotIn("model_lib.nets.unet_gla", sys.modules)
        self.assertNotIn("removal.v1_2.dataset", sys.modules)
        verified = adapter.verify_imported_moebius_modules(self.source, self.contract())
        self.assertIn("utils_infer", {row["module"] for row in verified})
        self.assertEqual(len(verified), 5)

    def test_source_drift_rejected_before_package_changes(self):
        (self.source / "model_lib/nets/unet_lambda_prune_lite.py").write_text("raise RuntimeError('tampered')\n")
        with self.assertRaisesRegex(ValueError, "original source identity differs"):
            adapter.prepare_moebius_cpu_imports(self.source, self.contract())
        self.assertFalse(any(adapter._is_source_package(name) for name in sys.modules))

    def test_existing_packages_cannot_be_replaced(self):
        sentinel = types.ModuleType("model_lib")
        sys.modules["model_lib"] = sentinel
        with self.assertRaisesRegex(RuntimeError, "already imported"):
            adapter.prepare_moebius_cpu_imports(self.source, self.contract())
        self.assertIs(sys.modules["model_lib"], sentinel)

    def test_cached_top_level_numerical_helper_is_rejected(self):
        sys.modules["utils_infer"] = types.ModuleType("utils_infer")
        with self.assertRaisesRegex(RuntimeError, "already imported"):
            adapter.prepare_moebius_cpu_imports(self.source, self.contract())

    def test_imported_helper_must_resolve_to_the_verified_tree(self):
        adapter.prepare_moebius_cpu_imports(self.source, self.contract())
        importlib.import_module("removal.v1_2.pipeline")
        sys.modules["utils_infer"].__file__ = str(self.source.parent / "foreign-utils-infer.py")
        with self.assertRaisesRegex(ValueError, "outside the verified source"):
            adapter.verify_imported_moebius_modules(self.source, self.contract())

    def test_namespace_cannot_include_a_second_source_tree(self):
        adapter.prepare_moebius_cpu_imports(self.source, self.contract())
        importlib.import_module("removal.v1_2.pipeline")
        sys.modules["model_lib"].__path__.append(str(self.source.parent))
        with self.assertRaisesRegex(ValueError, "namespace search path differs"):
            adapter.verify_imported_moebius_modules(self.source, self.contract())

    def test_numerical_import_closure_must_be_complete(self):
        adapter.prepare_moebius_cpu_imports(self.source, self.contract())
        with self.assertRaisesRegex(ValueError, "closure is incomplete"):
            adapter.verify_imported_moebius_modules(self.source, self.contract())

    def test_failed_import_cleans_partial_packages(self):
        self.write("model_lib/nets/unet_lambda_prune_lite.py", "raise RuntimeError('original import failed')\n")
        with self.assertRaisesRegex(RuntimeError, "original import failed"):
            adapter.prepare_moebius_cpu_imports(self.source, self.contract())
        self.assertFalse(any(adapter._is_source_package(name) for name in sys.modules))

    def test_second_import_requires_a_fresh_process(self):
        adapter.prepare_moebius_cpu_imports(self.source, self.contract())
        with self.assertRaisesRegex(RuntimeError, "fresh process"):
            adapter.prepare_moebius_cpu_imports(self.source, self.contract())

    def test_explicit_profiles_report_the_exact_scope(self):
        standard = adapter.profile_metadata(adapter.STANDARD_PROFILE, "pixelhacker", "cuda:0")
        self.assertEqual(standard["package_initializers_skipped"], [])
        self.assertFalse(standard["numerical_sources_modified"])
        student = adapter.profile_metadata(adapter.CPU_STUDENT_PROFILE, "moebius", "cpu")
        self.assertEqual(student["name"], "moebius-student-only-cpu-v1")
        self.assertEqual(student["package_initializers_skipped"],
                         ["model_lib/__init__.py", "removal/v1_2/__init__.py"])
        self.assertFalse(student["numerical_sources_modified"])

    def test_cpu_profile_cannot_select_pixel_or_a_gpu(self):
        for model, device in (("pixelhacker", "cpu"), ("moebius", "cuda:0"), ("moebius", "mps")):
            with self.subTest(model=model, device=device):
                with self.assertRaises(ValueError):
                    adapter.profile_metadata(adapter.CPU_STUDENT_PROFILE, model, device)
        with self.assertRaisesRegex(ValueError, "unknown"):
            adapter.profile_metadata("unrecorded-import-modification", "moebius", "cpu")


if __name__ == "__main__":
    unittest.main()
