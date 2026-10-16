#!/usr/bin/env python3
"""Model-free negative controls for immutable UltraBERT source/weight coverage."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import ultrabert_contract as contract


class UltraBertContractTest(unittest.TestCase):
    def setUp(self):
        self.record = contract.schema_record()
        self.header = copy.deepcopy(self.record["tensors"])
        self.payload = self.record["size"] - self.record["header_bytes"] - 8

    def test_pinned_source(self):
        p = contract.profile()
        self.assertEqual(p["revision"], contract.REVISION)
        self.assertEqual(p["declared_license"], "apache-2.0")
        self.assertEqual(p["config"]["attention_windows"], [128, 128, None] * 8)
        self.assertEqual(p["config"]["rope_theta"], [10000., 10000., 2560000.] * 8)

    def test_complete_original_schema(self):
        result = contract.validate_schema(self.header, self.payload)
        self.assertEqual(len(result), 218)
        targets = [target for target, shape in contract.expected_tensors().values()]
        self.assertEqual(len(targets), len(set(targets)))
        self.assertEqual(self.record["parameters"], 151032576)

    def test_missing_tensor_rejected(self):
        del self.header["norm.weight"]
        with self.assertRaisesRegex(ValueError, "missing"):
            contract.validate_schema(self.header, self.payload)

    def test_unmapped_head_rejected(self):
        self.header["cls.weight"] = copy.deepcopy(self.header["embed_tokens.weight"])
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            contract.validate_schema(self.header, self.payload)

    def test_transposed_shape_rejected(self):
        self.header["layers.0.mlp.down_proj.weight"]["shape"].reverse()
        with self.assertRaisesRegex(ValueError, "shape/dtype"):
            contract.validate_schema(self.header, self.payload)

    def test_precision_change_rejected(self):
        self.header["embed_tokens.weight"]["dtype"] = "F16"
        with self.assertRaisesRegex(ValueError, "shape/dtype"):
            contract.validate_schema(self.header, self.payload)

    def test_overlap_rejected(self):
        item = self.header["norm.weight"]
        length = item["data_offsets"][1] - item["data_offsets"][0]
        item["data_offsets"] = [0, length]
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            contract.validate_schema(self.header, self.payload)

    def test_offset_type_and_size_rejected(self):
        for offsets in ([0, 1], [-1, 150994943], [False, 150994944]):
            with self.subTest(offsets=offsets):
                self.header["embed_tokens.weight"]["data_offsets"] = offsets
                with self.assertRaisesRegex(ValueError, "offsets"):
                    contract.validate_schema(self.header, self.payload)

    def test_trailing_payload_rejected(self):
        with self.assertRaisesRegex(ValueError, "payload length"):
            contract.validate_schema(self.header, self.payload + 4)

    def test_source_missing_or_changed_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "source hash mismatch"):
                contract.validate_sources(directory)
            Path(directory, "config.json").write_text('{}')
            with self.assertRaisesRegex(ValueError, "source hash mismatch"):
                contract.validate_sources(directory)

    def test_tiny_or_wrong_weights_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "model.safetensors")
            path.write_bytes(b"invalid")
            with self.assertRaisesRegex(ValueError, "file size"):
                contract.validate_weight_file(path)

    def test_query_and_boundary_cases_are_explicit(self):
        cases = {case["id"]: case for case in contract.reference_cases()}
        self.assertEqual(cases["query"]["prompt"], "query")
        self.assertNotIn("prompt", cases["document"])
        self.assertEqual(cases["empty"]["text"], "")
        self.assertIn("text_pair", cases["pair"])
        self.assertGreater(len(cases["window-boundary"]["text"].split()), 258)
        self.assertEqual(contract.MAX_REFERENCE_TOKENS, 512)
        self.assertTrue(contract.QUERY.endswith("\nQuery: "))

    def test_invalid_tokenizer_rejected(self):
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            contract.tokenizer_metadata({"model": {"type": "BPE", "vocab": {}}})

    def test_converter_refuses_existing_output_before_importing_dependencies(self):
        spec = importlib.util.spec_from_file_location("ultrabert_converter", ROOT / "models/convert-ultrabert-to-gguf.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory, "existing.gguf")
            output.write_bytes(b"keep")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                module.convert(directory, output, "f32")
            self.assertEqual(output.read_bytes(), b"keep")


if __name__ == "__main__":
    unittest.main()
