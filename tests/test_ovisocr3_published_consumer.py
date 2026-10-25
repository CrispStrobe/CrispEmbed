"""Pure registry/publication controls for future OvisOCR3 alias consumer."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ovisocr3_consumer_test",
    ROOT / "tools/ci-heavy/ovisocr3_published_consumer.py")
consumer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(consumer)
SHA = "a" * 64
HEAD = "b" * 40


def sample():
    pins = {"destination": "cstr/OvisOCR3-GGUF", "candidate_sha256": SHA,
            "candidate_bytes": 1700000000, "precision": "f16", "recipe": "original-f16"}
    url = f'https://huggingface.co/cstr/OvisOCR3-GGUF/resolve/{HEAD}/ovisocr3-f16.gguf'
    producer = {"status": "published and immutable remote file reverified",
                "source_commit": HEAD, "destination": pins["destination"],
                "qualification": pins, "published_commit": HEAD,
                "remote": {"repo": pins["destination"], "commit": HEAD,
                           "filename": "ovisocr3-f16.gguf", "sha256": SHA,
                           "bytes": pins["candidate_bytes"],
                           "small_files_sha256": {name: SHA for name in
                               ("LICENSE", "UPSTREAM_README.md", "README.md", "manifest.json")}}}
    canonical = hashlib.sha256(json.dumps(producer, sort_keys=True, separators=(",", ":"),
                                        ensure_ascii=True, allow_nan=False).encode()).hexdigest()
    receipt = {"run": 123, "head_sha": HEAD, "artifact_id": 456,
               "artifact_zip_sha256": SHA, "producer": producer,
               "independent_review": {"publication_positive": True,
                                      "producer_canonical_sha256": canonical}}
    registry = '{ "ovisocr3-f16", "ovisocr3-f16.gguf", "' + url + '", "description", "1.7 GB", "apache-2.0" },'
    hashes = '{ "' + url + '", "' + SHA + '" },'
    return pins, receipt, registry, hashes


class ConsumerControls(unittest.TestCase):
    def test_exact_immutable_alias_and_publish_receipt(self):
        pins, receipt, registry, hashes = sample()
        result = consumer.require_publication(receipt, pins, registry, hashes)
        self.assertEqual(result["registry"]["alias"], "ovisocr3-f16")
        self.assertEqual(result["publication_artifact_id"], 456)
        self.assertNotIn("huggingface_hub", sys.modules)
        self.assertNotIn("torch", sys.modules)

    def test_wrong_or_unreviewed_candidate_and_registry_rejected(self):
        pins, receipt, registry, hashes = sample()
        for change in ("review", "model", "artifact", "small"):
            bad = copy.deepcopy(receipt)
            if change == "review":
                bad["independent_review"]["publication_positive"] = False
            elif change == "model":
                bad["producer"]["remote"]["sha256"] = "0" * 64
            elif change == "artifact":
                bad["artifact_zip_sha256"] = "not-a-sha"
            else:
                del bad["producer"]["remote"]["small_files_sha256"]["LICENSE"]
            with self.subTest(change=change), self.assertRaises(ValueError):
                consumer.require_publication(bad, pins, registry, hashes)
        with self.assertRaisesRegex(ValueError, "registry SHA pin"):
            consumer.require_publication(receipt, pins, registry, hashes.replace(SHA, "0" * 64))
        with self.assertRaisesRegex(ValueError, "alias/filename"):
            consumer.require_publication(receipt, pins, registry.replace("ovisocr3-f16", "ovisocr2-f16", 1), hashes)

    def test_publication_service_and_ovis3_only_environment(self):
        pins, receipt, registry, hashes = sample()
        value = consumer.require_publication(receipt, pins, registry, hashes)
        def call(args):
            return json.dumps({"id": 456, "workflow_run": {"id": 123},
                               "name": "ovisocr3-publication-report-123",
                               "digest": "sha256:" + SHA, "expired": False}).encode()
        self.assertEqual(consumer.service_artifact(value, call)["run"], 123)
        bad = copy.deepcopy(value)
        bad["publication_artifact_zip_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "artifact service"):
            consumer.service_artifact(bad, call)
        def run_call(args):
            return json.dumps({"id": 123, "head_sha": HEAD, "status": "completed",
                               "conclusion": "success",
                               "path": ".github/workflows/ovisocr3-publish.yml"}).encode()
        self.assertEqual(consumer.service_run(value, run_call)["run"], 123)
        bad_run = copy.deepcopy(value)
        bad_run["publication_head"] = "0" * 40
        with self.assertRaisesRegex(ValueError, "service run"):
            consumer.service_run(bad_run, run_call)
        preset = {"name": "both", "enabled_gates": ["CRISPEMBED_QWEN35_VISION_PRECISE_NORM",
                                                  "CRISPEMBED_QWEN35_VISION_PRECISE_ATTN"],
                  "values": {"CRISPEMBED_QWEN35_VISION_PRECISE_NORM": "1",
                             "CRISPEMBED_QWEN35_VISION_PRECISE_ATTN": "1",
                             "CRISPEMBED_QWEN35_VISION_CASCADE_NORM": "0"}}
        env = consumer.env_for(Path("/owned/cache"), preset)
        self.assertEqual([env[key] for key in consumer.GATES], ["1"] * 3)
        self.assertEqual(env["CRISPEMBED_OVISOCR2_FRONTEND"], "0")
        self.assertEqual(env["CRISPEMBED_QWEN35_QUANT"], "0")
        self.assertEqual(env["CRISPEMBED_CACHE_DIR"], "/owned/cache")

    def test_no_unreviewed_alias_allowed_locally(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "absent-publication.json"
            publisher = mock.Mock(require_receipts=mock.Mock(return_value={"destination": "cstr/OvisOCR3-GGUF"}))
            with mock.patch.object(consumer, "PUBLICATION", missing), \
                    mock.patch.object(consumer, "module", return_value=publisher):
                with self.assertRaisesRegex(ValueError, "publication is absent"):
                    consumer.preflight()


if __name__ == "__main__":
    unittest.main()
