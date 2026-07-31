#!/usr/bin/env python3
"""Unit tests for preprocessing benchmark/provenance helpers."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PreprocessingToolsTest(unittest.TestCase):
    def test_effect_gate_has_conservative_margins(self):
        bench = load("preproc_bench", ROOT / "tests/ocr_preprocessor_benchmark.py")
        baseline = {"ocr": {"status": "ok", "cer": 0.10}}
        self.assertEqual(bench.effect({"ocr": {"status": "ok", "cer": 0.08}}, baseline), "helped")
        self.assertEqual(bench.effect({"ocr": {"status": "ok", "cer": 0.105}}, baseline), "neutral")
        self.assertEqual(bench.effect({"ocr": {"status": "ok", "cer": 0.12}}, baseline), "harmed")
        self.assertEqual(bench.effect({}, baseline), "unavailable")

    def test_generator_records_parent_and_derived_hashes(self):
        generator = ROOT / "tests/regression/generate_problematic_variants.py"
        source = ROOT / "tests/regression/images/cc0/receipt_example.png"
        with tempfile.TemporaryDirectory(prefix="crispembed-preproc-test-") as tmp:
            subprocess.check_call([
                sys.executable, str(generator), "--source", source.name,
                "--output-dir", tmp, "--variants", "skew_m4", "rotate_180",
            ], cwd=ROOT)
            manifest = json.loads((Path(tmp) / "MANIFEST.json").read_text())
            self.assertEqual(len(manifest["rows"]), 2)
            parent_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            self.assertTrue(all(r["parent_sha256"] == parent_hash for r in manifest["rows"]))
            for row in manifest["rows"]:
                derived = Path(row["file"])
                self.assertTrue(derived.exists())
                self.assertEqual(row["sha256"], hashlib.sha256(derived.read_bytes()).hexdigest())
                self.assertIn(row["recipe"]["operation"], {"skew_m4", "rotate_180"})


if __name__ == "__main__":
    unittest.main()
