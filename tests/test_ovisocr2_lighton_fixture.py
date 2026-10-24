"""Standard-library gates for the hosted exact LightOn wrong-profile fixture."""

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "ovisocr2_lighton_fixture", ROOT / "tools/ci-heavy/ovisocr2_lighton_fixture.py")
fixture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixture)


def hub_info(repo, revision, files):
    return {"id": repo, "sha": revision, "cardData": {"license": "apache-2.0"},
            "siblings": [{"rfilename": name, "size": size, "blobId": blob,
                          **({"lfs": {"sha256": lfs}} if lfs else {})}
                         for name, (size, blob, lfs) in files.items()]}


class LightOnFixtureControls(unittest.TestCase):
    def test_frozen_converter_and_receipt_exact_source(self):
        receipt = fixture.require_receipt()
        self.assertEqual(fixture.sha(fixture.CONVERTER), fixture.CONVERTER_SHA)
        self.assertEqual(fixture.sha(fixture.RECEIPT), fixture.RECEIPT_SHA)
        self.assertEqual(len(receipt["weights"]["tensor_audit"]), 473)
        with tempfile.TemporaryDirectory() as directory:
            tampered = Path(directory) / "converter.py"
            tampered.write_bytes(fixture.CONVERTER.read_bytes() + b"\n# changed")
            with self.assertRaisesRegex(ValueError, "source SHA"):
                fixture.require_receipt(converter=tampered)
            old = json.loads(fixture.RECEIPT.read_text())
            old["weights"]["sha256"] = "0" * 64
            different = Path(directory) / "receipt.json"
            different.write_text(json.dumps(old))
            with self.assertRaisesRegex(ValueError, "source SHA"):
                fixture.require_receipt(path=different)

    def test_original_failed_run_is_provenance_not_quality_pass(self):
        run = {"id": fixture.SOURCE_RUN, "head_sha": fixture.SOURCE_COMMIT,
               "path": fixture.SOURCE_WORKFLOW, "status": "completed", "conclusion": "failure"}
        call = lambda _args: json.dumps(run).encode()
        self.assertIn("failure", fixture.require_source_run(call)["conclusion"])
        for key, value in (("head_sha", "0" * 40), ("conclusion", "success"),
                           ("path", "other.yml")):
            changed = {**run, key: value}
            with self.assertRaisesRegex(ValueError, "run/source/workflow"):
                fixture.require_source_run(lambda _args: json.dumps(changed).encode())

    def test_hub_metadata_and_downloaded_file_identity(self):
        files = {"plain.json": (4, hashlib.sha1(b"blob 4\0data").hexdigest(), None),
                 "tokenizer.json": (4, "b" * 40, hashlib.sha256(b"text").hexdigest())}
        info = hub_info(fixture.MODEL, fixture.REVISION, files)
        self.assertEqual(fixture.require_hub_info(info, fixture.MODEL, fixture.REVISION, files)["license"],
                         "apache-2.0")
        for mutation in (lambda x: x["cardData"].update(license="other"),
                         lambda x: x["siblings"][0].update(size=5),
                         lambda x: x["siblings"][1]["lfs"].update(sha256="0" * 64)):
            changed = copy.deepcopy(info)
            mutation(changed)
            with self.assertRaises(ValueError):
                fixture.require_hub_info(changed, fixture.MODEL, fixture.REVISION, files)
        with tempfile.TemporaryDirectory() as directory:
            plain = Path(directory) / "plain.json"
            plain.write_bytes(b"data")
            self.assertEqual(fixture.verify_file(plain, files["plain.json"]),
                             hashlib.sha256(b"data").hexdigest())
            plain.write_bytes(b"fail")
            with self.assertRaisesRegex(ValueError, "payload differs"):
                fixture.verify_file(plain, files["plain.json"])
            plain.write_bytes(b"text")
            self.assertEqual(fixture.verify_file(plain, files["tokenizer.json"]),
                             hashlib.sha256(b"text").hexdigest())

    def test_whole_file_and_every_tensor_record_required(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / fixture.GGUF_NAME
            model.write_bytes(b"exact-model")
            record = {"name": "one", "shape": [1], "sha256": "a" * 64}
            manifest = {"source_model": fixture.MODEL, "source_revision": fixture.REVISION,
                        "dtype": "f16", "keep_f32": [], "checkpoint_tensor_count": 473,
                        "export_tensor_count": 473, "records": [record], "bytes": len(b"exact-model")}
            receipt = {"weights": {"tensor_audit": [record]}}
            with mock.patch.object(fixture, "GGUF_BYTES", len(b"exact-model")), \
                 mock.patch.object(fixture, "GGUF_SHA", hashlib.sha256(b"exact-model").hexdigest()):
                self.assertEqual(fixture.verify_export(model, manifest, receipt)["format_version"], 1)
                changed = {**manifest, "records": [{**record, "sha256": "0" * 64}]}
                with self.assertRaisesRegex(ValueError, "byte/tensor identity"):
                    fixture.verify_export(model, changed, receipt)
                model.write_bytes(b"wrong-model")
                with self.assertRaisesRegex(ValueError, "byte/tensor identity"):
                    fixture.verify_export(model, manifest, receipt)

    def test_preflight_stops_before_packages_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            source = hub_info(fixture.MODEL, fixture.REVISION, fixture.SOURCE_FILES)
            base = hub_info(fixture.BASE, fixture.BASE_REVISION,
                            {"LICENSE": fixture.BASE_LICENSE_FILE})
            def api(repo, revision):
                return source if repo == fixture.MODEL else base
            with mock.patch.object(fixture, "model_api", side_effect=api), \
                 mock.patch.object(fixture, "require_source_run", return_value={"id": fixture.SOURCE_RUN}), \
                 mock.patch.object(fixture, "check_versions", side_effect=AssertionError("packages touched")), \
                 mock.patch.dict(os.environ, {"HEAVY_OUT": directory, "HEAVY_SCRATCH": directory}):
                report = fixture.run_main(preflight=True)
            self.assertIn("before framework imports", report["status"])
            self.assertNotIn("versions", report)
            self.assertTrue((Path(directory) / "ovisocr2-lighton-fixture.json").is_file())
        for name in ("torch", "transformers", "gguf", "safetensors"):
            self.assertNotIn(name, sys.modules)


if __name__ == "__main__":
    unittest.main()
