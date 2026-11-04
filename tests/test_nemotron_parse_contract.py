#!/usr/bin/env python3
"""Census controls catch silent tensor omission and architecture substitutions."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("nemotron", ROOT / "tools/nemotron_parse_contract.py")
nm = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(nm)


class Contract(unittest.TestCase):
    def setUp(self):
        self.header = json.loads((nm.FIXTURES / "tensor-header.json").read_text())
        self.config = json.loads((nm.FIXTURES / "config.json").read_text())
        self.processor = json.loads((nm.FIXTURES / "preprocessor_config.json").read_text())

    def check_header(self):
        return nm.audit_header(self.header, file_bytes=2523266640, header_bytes=64608)

    def test_real_census(self):
        result = nm.audit()
        self.assertEqual(result["tensor_count"], 510)
        self.assertEqual(result["parameter_elements"], 630800496)
        self.assertFalse(result["full_weight_hash_verified"])
        self.assertFalse(result["model_qualified"])

    def test_missing_optional_mtp_still_detected(self):
        del self.header["decoder.extra_heads.0.weight"]
        with self.assertRaisesRegex(ValueError, "name mismatch"):
            self.check_header()

    def test_cannot_substitute_cross_attention(self):
        self.header["decoder.layers.0.cross_attn.q_proj.weight"] = copy.deepcopy(self.header["decoder.layers.0.self_attn.q_proj.weight"])
        with self.assertRaisesRegex(ValueError, "name mismatch"):
            self.check_header()

    def test_gqa_shape_change_detected(self):
        self.header["decoder.layers.0.self_attn.k_proj.weight"]["shape"] = [1024, 1024]
        with self.assertRaisesRegex(ValueError, "shape/dtype"):
            self.check_header()

    def test_overlap_detected(self):
        self.header["decoder.embed_tokens.weight"]["data_offsets"][0] = 0
        with self.assertRaisesRegex(ValueError, "offset/size"):
            self.check_header()

    def test_noncausal_decoder_rejected(self):
        self.config["legacy_causal_vision_prefix"] = False
        with self.assertRaisesRegex(ValueError, "config drift"):
            nm.audit_config(self.config, self.processor)

    def test_center_padding_rejected(self):
        self.processor["navit_padding_mode"] = "center"
        with self.assertRaisesRegex(ValueError, "geometry drift"):
            nm.audit_config(self.config, self.processor)


if __name__ == "__main__":
    unittest.main()
