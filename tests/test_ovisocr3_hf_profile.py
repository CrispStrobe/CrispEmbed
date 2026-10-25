"""Pure OvisOCR3 HF-only promotion gates; no framework/checkpoint import."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hf = load(ROOT / "models/qwen35_ovis3_hf_profile.py", "ovisocr3_hf_profile_test")
converter = load(ROOT / "models/convert-qwen35-ocr-to-gguf.py", "ovisocr3_hf_converter_test")
FIXTURE = ROOT / "tests/fixtures/ovisocr3-source"


class HFPromotionControls(unittest.TestCase):
    def test_actual_raw_generation_and_resolved_version_bookkeeping(self):
        raw=json.loads((FIXTURE / "generation_config.json").read_text())
        metadata=json.loads(hf.METADATA_RECEIPT.read_text())["producer"]
        resolved=metadata["resolved_generation"]
        self.assertNotIn("transformers_version",raw)
        hf.validate_generation_resolution(raw,resolved)
        for key,value in (("eos_token_id",[248046]),("do_sample",True),
                          ("temperature",1.0),("transformers_version","5.16.2"),
                          ("max_new_tokens",12)):
            changed=dict(resolved,**{key:value})
            with self.subTest(field=key),self.assertRaises(ValueError):
                hf.validate_generation_resolution(raw,changed)

    def test_reviewed_small_and_large_reference_artifacts_are_distinct(self):
        receipt = {"run": 123, "artifact_id": 5, "artifact_zip_sha256": "a" * 64,
                   "independent_review": {"artifact_zip_sha256": "a" * 64,
                                          "reference_artifact_zip_sha256": "b" * 64},
                   "reference_artifact": {"id": 6, "name": "ovisocr3-reference-123",
                                          "zip_sha256": "b" * 64, "bytes": 100}}
        small, large = hf._reviewed_artifacts(receipt)
        self.assertEqual((small["id"], large["id"]), (5, 6))
        for path, bad in (("id", 5), ("name", "ovisocr3-reference-reports-123"),
                          ("zip_sha256", "a" * 64), ("bytes", 0)):
            changed = copy.deepcopy(receipt)
            changed["reference_artifact"][path] = bad
            with self.subTest(field=path), self.assertRaises(ValueError):
                hf._reviewed_artifacts(changed)
        changed = copy.deepcopy(receipt)
        changed["independent_review"]["artifact_zip_sha256"] = "b" * 64
        with self.assertRaises(ValueError):
            hf._reviewed_artifacts(changed)

    def test_absent_real_reference_rejects_before_any_source_reader(self):
        with patch.object(hf, "REFERENCE_RUN", 0):
            with self.assertRaisesRegex(ValueError, "Reviewed original OvisOCR3 HF reference proof absent"):
                hf._reviewed_receipts()
            with self.assertRaisesRegex(ValueError, "Reviewed original OvisOCR3 HF reference proof absent"):
                hf.load_profile("/missing", hf.MODEL, hf.REVISION)
        self.assertEqual(hf.source.SOURCE_EOS_IDS, [248044, 248046])

    def test_exact_scoped_profile_and_crossover_rejection(self):
        report, _, hashes = hf.source._evidence()
        generation = json.loads((FIXTURE / "generation_config.json").read_text())
        config = json.loads((FIXTURE / "config.json").read_text())
        prompt = (FIXTURE / "prompts/plain_ocr.txt").read_text()
        base = hf.source._expected_profile(report, hashes, generation, prompt)
        metadata = json.loads(hf.METADATA_RECEIPT.read_text())
        reference = {"producer": {}}
        qualified = hf._expected_profile(base, metadata, reference, "a" * 64)
        with patch.object(hf, "_reviewed_receipts", return_value=(metadata, reference)), \
             patch.object(hf, "_checked_stop_proof", return_value="a" * 64):
            hf.validate_config(config, qualified)
            with patch.object(converter, "_ovis3_hf_profile", return_value=hf):
                converter.validate_config(config, qualified)
            self.assertTrue(qualified["hf_stop_policy_qualified"])
            self.assertFalse(qualified["effective_stops_qualified"])
            self.assertFalse(qualified["native_alias_qualified"])
            self.assertEqual(qualified["stop_ids"], [248044, 248046])
            self.assertEqual(qualified["hf_stop_runtime_proof_sha256"], "a" * 64)
            for field, value in (("hf_stop_policy_qualified", False),
                                 ("effective_stops_qualified", True),
                                 ("name", "ovisocr2"),
                                 ("stop_qualification_scope", "vllm-runtime"),
                                 ("rendered_prompt", "changed")):
                changed = copy.deepcopy(qualified)
                changed[field] = value
                with self.subTest(field=field), self.assertRaises(ValueError):
                    hf.validate_config(config, changed)
            changed_config = copy.deepcopy(config)
            changed_config["text_config"]["mtp_num_hidden_layers"] = 1
            with self.assertRaises(ValueError):
                hf.validate_config(changed_config, qualified)

    def test_installed_hf_stop_proof_requires_both_eos_and_non_eos(self):
        proof = {"transformers_version": "5.16.1",
                 "class": "transformers.generation.stopping_criteria.EosTokenCriteria",
                 "source_sha256": hf.STOPPING_CRITERIA_SOURCE_SHA256,
                 "eos_ids": [248044, 248046], "input_last_ids": [248044, 248046, 42],
                 "actual_shape": [3], "actual_dtype": "torch.bool",
                 "actual_stops": [True, True, False], "expected_stops": [True, True, False]}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "proof.json"
            path.write_text(json.dumps(proof))
            with patch.dict(os.environ, {"CRISPEMBED_OVISOCR3_HF_STOP_PROOF": str(path)}):
                self.assertEqual(hf._checked_stop_proof(), hf.source._file_sha256(path))
                for field, wrong in (("actual_stops", [True, False, False]),
                                     ("source_sha256", "0" * 64), ("eos_ids", [248046]),
                                     ("actual_shape", [3, 1])):
                    changed = dict(proof, **{field: wrong})
                    path.write_text(json.dumps(changed))
                    with self.subTest(field=field), self.assertRaises(ValueError):
                        hf._checked_stop_proof()
            with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ValueError, "proof absent"):
                hf._checked_stop_proof()


if __name__ == "__main__":
    unittest.main()
