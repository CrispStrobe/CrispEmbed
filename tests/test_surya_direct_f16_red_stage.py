"""Weight-free tests of direct-F16 RED kernel source closure."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "tools/kaggle/surya-direct-f16-red"


class RedKernelStage(unittest.TestCase):
    def test_staged_kernel_embeds_exact_test_and_is_cpu_anonymous(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "stage"
            result = subprocess.run([sys.executable, str(HERE / "stage.py"), str(target), "--commit", "a" * 40],
                                    check=True, capture_output=True, text=True)
            manifest = json.loads(result.stdout)
            raw = (ROOT / "tests/test_qwen35_direct_f16.cpp").read_bytes()
            tree = ast.parse((target / "kernel.py").read_text())
            constants = {node.targets[0].id: ast.literal_eval(node.value)
                         for node in tree.body if isinstance(node, ast.Assign) and len(node.targets) == 1
                         and isinstance(node.targets[0], ast.Name)
                         and node.targets[0].id in ("COMMIT", "TEST_B64", "TEST_SHA256")}
            self.assertEqual(base64.b64decode(constants["TEST_B64"], validate=True), raw)
            self.assertEqual(constants["TEST_SHA256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(manifest["kernel_sha256"], hashlib.sha256((target / "kernel.py").read_bytes()).hexdigest())
            metadata = json.loads((target / "kernel-metadata.json").read_text())
            self.assertEqual(metadata["enable_gpu"], "false")
            self.assertEqual(metadata["dataset_sources"], [])
            self.assertEqual(metadata["model_sources"], [])
            self.assertNotEqual(subprocess.run([sys.executable, str(HERE / "stage.py"), str(target), "--commit", "a" * 40],
                                              capture_output=True).returncode, 0)

    def test_bad_commit_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "stage"
            result = subprocess.run([sys.executable, str(HERE / "stage.py"), str(target), "--commit", "main"],
                                    capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(target.exists())

    def test_green_embeds_exact_implementation(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "stage"
            result = subprocess.run([sys.executable, str(HERE / "stage.py"), str(target),
                                     "--commit", "a" * 40, "--green"],
                                    check=True, capture_output=True, text=True)
            manifest = json.loads(result.stdout)
            tree = ast.parse((target / "kernel.py").read_text())
            values = {node.targets[0].id: ast.literal_eval(node.value)
                      for node in tree.body if isinstance(node, ast.Assign) and len(node.targets) == 1
                      and isinstance(node.targets[0], ast.Name)
                      and node.targets[0].id in ("SOURCE_B64", "SOURCE_SHA256", "MODE")}
            source = (ROOT / "src/qwen35_linear.cpp").read_bytes()
            self.assertEqual(base64.b64decode(values["SOURCE_B64"], validate=True), source)
            self.assertEqual(values["SOURCE_SHA256"], hashlib.sha256(source).hexdigest())
            self.assertEqual(manifest["source_sha256"], values["SOURCE_SHA256"])
            self.assertEqual(values["MODE"], "green")


if __name__ == "__main__":
    unittest.main()
