"""OvisOCR3 F16 publication stays closed without reviewed receipts."""

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ovisocr3_publish_test", ROOT / "tools/ci-heavy/ovisocr3_publish.py")
publisher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publisher)
SHA = "a" * 64
HEAD = "b" * 40


class PublicationControls(unittest.TestCase):
    def test_pinned_source_license_chain_and_dormant_receipts(self):
        source = publisher.source_license_contract()
        self.assertEqual(source["model"], "ATH-MaaS/OvisOCR3")
        self.assertEqual(source["base_base_declared"], "Qwen/Qwen3.5-0.8B-Base")
        with mock.patch.object(publisher, "SOURCE_CONTRACT_SHA256", "0" * 64):
            with self.assertRaisesRegex(ValueError, "source contract"):
                publisher.require_receipts()
        with self.assertRaisesRegex(ValueError, "destination"):
            publisher.require_receipts("cstr/other")
        with mock.patch.object(publisher, "receipt", side_effect=ValueError("receipt absent")):
            with self.assertRaisesRegex(ValueError, "receipt absent"):
                publisher.require_receipts()
        self.assertNotIn("huggingface_hub", sys.modules)
        self.assertNotIn("torch", sys.modules)

    def test_live_artifact_run_digest_and_workflow_gate(self):
        pins = {"native_run": 11, "native_source_commit": HEAD,
                "native_f16_artifact_id": 1, "native_f16_artifact_zip_sha256": SHA,
                "native_f16_artifact_bytes": 123, "native_report_artifact_id": 2,
                "native_report_artifact_zip_sha256": SHA, "original_run": 12,
                "original_source_commit": HEAD, "original_artifact_id": 3,
                "original_artifact_zip_sha256": SHA, "quality_run": 13,
                "quality_source_commit": HEAD, "quality_artifact_id": 4,
                "quality_artifact_zip_sha256": SHA, "public_run": 14,
                "public_source_commit": HEAD, "public_artifact_id": 5,
                "public_artifact_zip_sha256": SHA}
        def artifact_call(args):
            artifact_id = int(args[-1].split("/")[-1])
            names = ("", "ovisocr3-native14-f16-11", "ovisocr3-native14-reports-11",
                     "ovisocr3-original-quality-12", "ovisocr3-f16-quality-13",
                     "ovisocr3-public-surfaces-14")
            runs = (0, 11, 11, 12, 13, 14)
            return json.dumps({"id": artifact_id, "name": names[artifact_id], "expired": False,
                               "digest": "sha256:" + SHA, "size_in_bytes": 123,
                               "workflow_run": {"id": runs[artifact_id]}}).encode()
        self.assertEqual(len(publisher.artifact_service(pins, artifact_call)), 5)
        bad = copy.deepcopy(pins)
        bad["native_f16_artifact_zip_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "service proof"):
            publisher.artifact_service(bad, artifact_call)
        workflows = {11: ".github/workflows/ovisocr3-native14.yml",
                     12: ".github/workflows/ovisocr3-quality.yml",
                     13: ".github/workflows/ovisocr3-quality-f16.yml",
                     14: ".github/workflows/ovisocr3-public-surfaces.yml"}
        def run_call(args):
            run = int(args[-1].split("/")[-1])
            return json.dumps({"id": run, "head_sha": HEAD, "status": "completed",
                               "conclusion": "success", "path": workflows[run]}).encode()
        self.assertEqual(len(publisher.run_service(pins, run_call)), 4)
        workflows[14] = ".github/workflows/other.yml"
        with self.assertRaisesRegex(ValueError, "run source/workflow"):
            publisher.run_service(pins, run_call)

    def test_card_complete_and_model_mismatch_stops_before_hub(self):
        pins = {"candidate_sha256": SHA, "candidate_bytes": 2,
                "public_source_commit": HEAD, "owned_png_sha256": SHA}
        card = publisher.card(pins, evidence_source_commit=HEAD)
        self.assertIn(b"CPU only", card)
        self.assertNotIn(b"${", card)
        with self.assertRaisesRegex(ValueError, "source commit"):
            publisher.card(pins, evidence_source_commit="")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / publisher.FILENAME).write_bytes(b"unqualified")
            with self.assertRaisesRegex(ValueError, "GGUF missing or SHA/bytes differ"):
                publisher.publish(pins, root, root, root / "scratch", source_commit=HEAD)
        self.assertNotIn("huggingface_hub", sys.modules)


if __name__ == "__main__":
    unittest.main()
