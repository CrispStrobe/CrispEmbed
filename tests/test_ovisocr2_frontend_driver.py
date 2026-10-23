"""Model-free controls for the hosted Ovis original-artifact gate."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ovisocr2_frontend_driver", ROOT / "tools/ci-heavy/ovisocr2_frontend.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class FrontendDriverControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence = json.loads((ROOT / "tests/results/qwen35-ocr/ovisocr2-metadata.json").read_text())

    def test_exact_original_service_and_artifact_required(self):
        evidence = self.evidence
        policy = mock.Mock()
        policy._verified_evidence.return_value = evidence["producer"]
        run = {"id": driver.METADATA_RUN, "head_sha": driver.METADATA_SOURCE,
               "status": "completed", "conclusion": "success",
               "path": ".github/workflows/ovisocr2-metadata.yml"}
        artifact = {"id": evidence["artifact"]["id"],
                    "name": f"ovisocr2-metadata-{driver.METADATA_RUN}",
                    "expired": False, "workflow_run": {"id": driver.METADATA_RUN},
                    "expires_at": evidence["artifact"]["expires_at"],
                    "archive_download_url": "https://api.github.com/example/zip"}
        with mock.patch.object(driver, "run_info", return_value=run), mock.patch.object(
                driver.subprocess, "check_output", return_value=json.dumps(artifact).encode()):
            self.assertEqual(driver.preflight(policy, evidence), artifact)
            for field, bad in (("head_sha", "0" * 40), ("conclusion", "failure"),
                               ("path", "other.yml")):
                altered = dict(run, **{field: bad})
                with mock.patch.object(driver, "run_info", return_value=altered):
                    with self.assertRaises(ValueError):
                        driver.preflight(policy, evidence)
            altered = dict(artifact, expired=True)
            with mock.patch.object(driver.subprocess, "check_output", return_value=json.dumps(altered).encode()):
                with self.assertRaises(ValueError):
                    driver.preflight(policy, evidence)

    def test_original_zip_digest_required_before_extraction(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "original.zip"
            artifact = {"archive_download_url": "https://api.github.com/repos/example/actions/artifacts/1/zip"}

            def fake_download(command, stdout, check, timeout):
                self.assertEqual(command[-1], "repos/example/actions/artifacts/1/zip")
                stdout.write(b"not the pinned artifact")

            with mock.patch.object(driver.subprocess, "run", side_effect=fake_download):
                with self.assertRaisesRegex(ValueError, "ZIP SHA256 differs"):
                    driver.download_artifact(artifact, self.evidence["artifact"]["zip_sha256"], destination)
            self.assertFalse((Path(temporary) / "original").exists())

    def test_import_does_not_load_model_frameworks(self):
        import sys
        for name in ("torch", "transformers", "gguf", "numpy", "huggingface_hub"):
            self.assertNotIn(name, sys.modules)


if __name__ == "__main__":
    unittest.main()
