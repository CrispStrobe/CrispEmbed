"""Check tracked GPU bench submission metadata without contacting Kaggle."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("kpush", ROOT / "tools/kaggle/kpush.py")
kpush = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(kpush)


class OvisBenchSubmission(unittest.TestCase):
    def test_existing_kernel_identity_and_seed_survive_rendering(self):
        rendered = {}
        sha = "a" * 40

        def fake_push(cmd, **kwargs):
            destination = Path(cmd[cmd.index("-p") + 1])
            rendered.update(json.loads((destination / "kernel-metadata.json").read_text()))
            code = (destination / rendered["code_file"]).read_text()
            self.assertNotIn("__SHA__", code)
            self.assertIn(sha, code)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with patch.dict(kpush.os.environ, {"KAGGLE_ACCOUNT": "audit-placeholder",
                                          "KAGGLE_TOKEN": "noncredential-placeholder"}), \
             patch.object(kpush.subprocess, "run", side_effect=fake_push) as push, \
             contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            result = kpush.main(["kpush", str(ROOT / "tools/kaggle/ovisocr2-gpu-bench"),
                                 "--sha", sha])
        self.assertEqual(0, result)
        push.assert_called_once()
        self.assertEqual("audit-placeholder/crispembed-ovisocr2-gpu-bench", rendered["id"])
        self.assertEqual("crispembed-ovisocr2-gpu-bench", rendered["title"])
        self.assertEqual("kernel.py", rendered["code_file"])
        self.assertEqual(["audit-placeholder/crispembed-ccache"], rendered["dataset_sources"])
        self.assertEqual("latest", rendered["docker_image_pinning_type"])
        for key in ("is_private", "enable_gpu", "enable_internet"):
            self.assertEqual("true", rendered[key])


if __name__ == "__main__":
    unittest.main()
