"""Weight-free tests for the frozen Kaggle export/GPU control submission."""
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zlib

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "tools/kaggle/xiaomi-ocr-0-export/stage.py"
RUNNER = ROOT / "tools/kaggle/xiaomi-ocr-0-export/runner.py"


def load(path):
    spec = importlib.util.spec_from_file_location("xiaomi_export_runner_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Array:
    shape = (1, 2)
    dtype = "int64"

    def tobytes(self):
        return b"frozen bytes"


class Fields(dict):
    @property
    def files(self):
        return list(self)


class XiaomiHostedExportTest(unittest.TestCase):
    def test_frozen_input_shape_type_bytes_and_keyset(self):
        runner = load(RUNNER)
        case = {"name": "one", "fields": {"input_ids": [[1, 2], "int64",
                hashlib.sha256(b"frozen bytes").hexdigest()]}}
        fields = Fields(input_ids=Array())
        runner.validate_frozen_fields(case, fields)
        for replacement in ([[1, 3], "int64", case["fields"]["input_ids"][2]],
                            [[1, 2], "float32", case["fields"]["input_ids"][2]],
                            [[1, 2], "int64", "0" * 64]):
            with self.subTest(replacement=replacement), self.assertRaisesRegex(ValueError, "field differs"):
                runner.validate_frozen_fields({"name": "one", "fields": {"input_ids": replacement}}, fields)
        with self.assertRaisesRegex(ValueError, "field names"):
            runner.validate_frozen_fields(case, Fields(other=Array()))

    def test_stage_freezes_bytes_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            stage = base / "staged"
            result = subprocess.run([sys.executable, str(STAGE), str(stage)], text=True,
                                    capture_output=True, check=True)
            manifest = json.loads(result.stdout)
            self.assertEqual(len(manifest["files_sha256"]), 3)
            self.assertEqual(hashlib.sha256((stage / "runner.py").read_bytes()).hexdigest(),
                             manifest["staged_runner_sha256"])
            staged = load(stage / "runner.py")
            work, root = base / "work", base / "source"
            work.mkdir()
            with patch.object(staged, "WORK", work), patch.object(staged, "ROOT", root):
                staged.unpack()
                for name, expected in manifest["files_sha256"].items():
                    self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), expected)
                bundle = json.loads(zlib.decompress(base64.b64decode(staged.BUNDLE)))
                bundle["files"][next(iter(bundle["files"]))]["content"] = base64.b64encode(b"changed").decode()
                tampered = base64.b64encode(zlib.compress(json.dumps(bundle).encode())).decode()
                with patch.object(staged, "BUNDLE", tampered), self.assertRaisesRegex(ValueError, "source hash differs"):
                    staged.unpack()
            repeated = subprocess.run([sys.executable, str(STAGE), str(stage)], text=True, capture_output=True)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn("already exists", repeated.stderr)


if __name__ == "__main__":
    unittest.main()
