"""No framework/model imports: original OvisOCR3 reference stays closed without review."""

import importlib.util
import hashlib
import json
from pathlib import Path
import re
import sys
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dumper = load(ROOT / "tools/dump_ovisocr3_reference.py", "ovisocr3_reference_dumper_test")
driver = load(ROOT / "tools/ci-heavy/ovisocr3_reference.py", "ovisocr3_reference_driver_test")


class ReferenceControls(unittest.TestCase):
    def test_native_plain_ocr_source_and_reviewed_rendered_prompt(self):
        source = (ROOT / "src/qwen35_tokenizer.cpp").read_text()
        receipt = json.loads((ROOT / "tests/results/qwen35-ocr/ovisocr3-metadata.json").read_text())
        prompt = re.search(r'R"OVIS3\((.*?)\)OVIS3"', source, re.S)
        self.assertIsNotNone(prompt)
        self.assertEqual(hashlib.sha256(prompt.group(1).encode()).hexdigest(),
                         "e037cf0013a757e19eae6d4d701f3b90848449fbebe44d1f8fa7d8ca4a5af9a7")
        rendered = re.search(r'ovis3_rendered_template = R"OVIS3\((.*?)\)OVIS3";', source, re.S)
        self.assertIsNotNone(rendered)
        self.assertEqual(rendered.group(1), receipt["rendered_prompt"])
        self.assertIn('ovis3_metadata_sha256 = "' +
                      receipt["independent_review"]["producer_canonical_sha256"] + '";', source)
        self.assertIn('core_env::on("CRISPEMBED_OVISOCR3_FRONTEND")', source)
        self.assertIn('!core_gguf::kv_bool(meta, "qwen35_ocr.effective_stops_qualified", false)', source)

    def test_review_and_missing_source_precede_checkpoint_and_framework(self):
        self.assertEqual(dumper.require_reviewed_metadata()["source_model"], "ATH-MaaS/OvisOCR3")
        with patch.object(dumper, "METADATA_RUN", 0):
            with self.assertRaisesRegex(ValueError, "reviewed metadata pins absent"):
                dumper.require_reviewed_metadata()
        with self.assertRaisesRegex(ValueError, "reviewed metadata pins absent"):
            with patch.object(dumper, "METADATA_RUN", 0):
                dumper.validate_request("/does-not-exist", "")
        with self.assertRaisesRegex(ValueError, "custom prompt override"):
            dumper.validate_request("/does-not-exist", "different task")
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("transformers", sys.modules)

    def test_metadata_service_exact_workflow_and_artifact(self):
        source = {"source_model": "ATH-MaaS/OvisOCR3"}
        run = {"status": "completed", "conclusion": "success", "head_sha": "a" * 40,
               "path": ".github/workflows/ovisocr2-metadata.yml"}
        artifact = {"workflow_run": {"id": 123}, "name": "ovisocr3-metadata-123", "expired": False,
                    "digest": "sha256:" + "b" * 64}
        def call(args):
            return json.dumps(artifact if args[-1].endswith("/456") else run).encode()
        with (patch.object(driver.dumper, "require_reviewed_metadata", return_value=source),
              patch.object(driver.dumper, "METADATA_RUN", 123),
              patch.object(driver.dumper, "METADATA_HEAD", "a" * 40),
              patch.object(driver.dumper, "METADATA_ARTIFACT_ID", 456),
              patch.object(driver.dumper, "METADATA_ARTIFACT_SHA256", "b" * 64)):
            producer, service = driver.metadata_service(call)
            self.assertIs(producer, source)
            self.assertEqual(service["artifact_id"], 456)
            artifact["expired"] = True
            with self.assertRaises(ValueError):
                driver.metadata_service(call)
            artifact["expired"] = False
            run["path"] = ".github/workflows/ovisocr3-metadata.yml"
            with self.assertRaises(ValueError):
                driver.metadata_service(call)

    def test_capture_stage_coverage_and_eos_at_cap(self):
        config = json.loads((ROOT / "tests/fixtures/ovisocr3-source/config.json").read_text())
        shaped = types.SimpleNamespace(
            vision_config=types.SimpleNamespace(depth=config["vision_config"]["depth"]),
            text_config=types.SimpleNamespace(num_hidden_layers=config["text_config"]["num_hidden_layers"],
                                              layer_types=config["text_config"]["layer_types"]))
        stages = dumper.required_stages(shaped, 12)
        self.assertEqual(len(stages), len(set(stages)))
        for stage in ("prefill.vision.layer.11", "cached.decoder.layer.23", "full_prefix.cache.23.keys",
                      "prefill.logits", "cached.logits", "full_prefix.logits", "generated.scores.11",
                      "prefill.delta.final_state", "cached.delta.initial_state"):
            if stage == "full_prefix.cache.23.keys":
                self.assertNotIn(stage, stages)
            else:
                self.assertIn(stage, stages)
        eos = dumper.generation_limit_report([100, 248046], 2, [248044, 248046])
        self.assertTrue(eos["reached_eos"])
        self.assertFalse(eos["length_limited"])
        limited = dumper.generation_limit_report([100, 101], 2, [248044, 248046])
        self.assertTrue(limited["length_limited"])
        with self.assertRaises(ValueError):
            dumper.generation_limit_report([1, 2, 3], 2, [248044, 248046])


if __name__ == "__main__":
    unittest.main()
