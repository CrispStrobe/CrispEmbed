"""Pure source-gate controls for the hosted weight-free OvisOCR3 probe."""

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ovisocr3_metadata_test", ROOT / "tools/ci-heavy/ovisocr3_metadata.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class MetadataControls(unittest.TestCase):
    def test_preflight_real_source_and_no_framework_import(self):
        source = worker.preflight()
        self.assertEqual(source["repository"]["revision"], worker.policy.REVISION)
        self.assertEqual(source["generation_config"]["eos_ids_declared"], [248044, 248046])
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("transformers", sys.modules)

    def test_official_schema_rejects_unknown_mtp_shape_and_missing_head(self):
        expected = {"model.weight": [2, 3], "lm_head.weight": [2, 3]}
        actual = {"model.weight": [2, 3]}
        self.assertTrue(worker.check_schema(expected, actual, True)["tied_head_absent"])
        for tensors, tied in ((actual, False), ({"model.weight": [2, 4]}, True),
                              ({**actual, "mtp.weight": [2, 3]}, True)):
            with self.assertRaises(ValueError):
                worker.check_schema(expected, tensors, tied)

    def test_pinned_stop_method_list_and_primary_semantics(self):
        source = '''class SamplingParams:
    def update_from_generation_config(self, generation_config: dict[str, Any], eos_token_id: int | None = None):
        if not self.ignore_eos: self._eos_token_id = eos_token_id
        if eos_token_id is not None: self._all_stop_token_ids.add(eos_token_id)
        if (eos_ids := generation_config.get("eos_token_id")) is not None:
            eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids)
            if eos_token_id is not None: eos_ids.discard(eos_token_id)
            if eos_ids:
                self._all_stop_token_ids.update(eos_ids)
                if not self.ignore_eos:
                    eos_ids.update(self.stop_token_ids)
                    self.stop_token_ids = list(eos_ids)
'''
        result = worker.original_stop_probe(source, {"eos_token_id": [248044, 248046]}, 248046)
        self.assertEqual(result, {"primary_eos": 248046, "additional_stop_ids": [248044],
                                  "all_stop_ids": [248044, 248046]})

    def test_failure_redaction(self):
        secret = "hf_" + "x" * 30
        result = worker.failure_detail(ValueError("https://example.org/a " + secret + " /private/operator/model"))
        self.assertNotIn(secret, result)
        self.assertNotIn("example.org", result)
        self.assertNotIn("/private", result)


if __name__ == "__main__":
    unittest.main()
