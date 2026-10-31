"""Weight-free tests for the hosted native patch stage's source gate."""
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zlib

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools/kaggle/xiaomi-ocr-0-native-stage/runner.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("xiaomi_native_patch_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner


def bundle(runner, files):
    package = {"base_commit": runner.BASE, "ggml_commit": runner.GGML,
               "runner_template_sha256": "0" * 64, "files": files}
    return base64.b64encode(zlib.compress(json.dumps(package).encode())).decode()


class NativeStageTest(unittest.TestCase):
    def test_unstaged_template_fails_closed(self):
        runner = load_runner()
        with tempfile.TemporaryDirectory() as d, self.assertRaisesRegex(ValueError, "not staged"):
            runner.unpack(Path(d))

    def test_unpacked_sources_require_expected_names_and_content(self):
        runner = load_runner()
        content = b"source bytes"
        files = {"src/test.cpp": {"content": base64.b64encode(content).decode(),
                                  "sha256": hashlib.sha256(content).hexdigest()}}
        with tempfile.TemporaryDirectory() as d, patch.object(runner, "ALLOWED", set(files)):
            root = Path(d)
            with patch.object(runner, "BUNDLE", bundle(runner, files)):
                closure = runner.unpack(root)
            self.assertEqual((root / "src/test.cpp").read_bytes(), content)
            self.assertEqual(closure["files_sha256"]["src/test.cpp"], files["src/test.cpp"]["sha256"])
            files["src/test.cpp"]["content"] = base64.b64encode(b"tampered").decode()
            with patch.object(runner, "BUNDLE", bundle(runner, files)), self.assertRaisesRegex(ValueError, "digest differs"):
                runner.unpack(root)
            files["extra"] = files["src/test.cpp"]
            with patch.object(runner, "BUNDLE", bundle(runner, files)), self.assertRaisesRegex(ValueError, "closure differs"):
                runner.unpack(root)


if __name__ == "__main__":
    unittest.main()
