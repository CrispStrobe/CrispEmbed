#!/usr/bin/env python3
"""Small stdlib-only regressions; no framework/model imports or execution."""

import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import migan_reference_contract as contract


class MiganReferenceContractTest(unittest.TestCase):
    def test_fixture_hashes_and_polarity_are_frozen(self):
        frozen = contract.load_contract()
        self.assertEqual(frozen["fixtures"], [contract.fixture_identity(name) for name in contract.CASES])
        self.assertEqual(frozen["acceptance"]["every_float_stage"]["max_abs_max"], 0.002)
        self.assertEqual(frozen["acceptance"]["every_float_stage"]["cosine_min"], 0.99999)
        self.assertEqual(frozen["fixtures"][2]["fill_pixels"], 0)
        self.assertEqual(frozen["fixtures"][3]["fill_pixels"], 256 * 256)
        _, mask = contract.fixture("edge_touch")
        self.assertEqual(mask[48 * 256], 255)
        self.assertEqual(mask[48 * 256 + 35], 0)

    def test_input_rejects_ambiguous_or_wrong_shape(self):
        rgb, mask = contract.fixture("empty")
        for bad in (1, 127, 254):
            with self.assertRaisesRegex(ValueError, "binary"):
                contract.validate_input(rgb, bytes([bad]) + mask[1:])
        for args in ((rgb[:-1], mask), (rgb, mask[:-1]), (rgb, mask, 512, 256)):
            with self.assertRaises(ValueError):
                contract.validate_input(*args)
        with self.assertRaises(ValueError):
            contract.fixture("unfrozen-input")

    def test_state_schema_preserves_constants_and_scalars(self):
        state = contract.expected_state_shapes()
        self.assertEqual(len(state), 154)
        self.assertEqual(state["encoder.b256.fromrgb.weight"], (128, 4, 1, 1))
        self.assertEqual(state["synthesis.b256.conv1.noise_strength"], ())
        self.assertEqual(state["synthesis.b256.conv1.noise_const"], (256, 256))
        self.assertEqual(state["synthesis.b256.conv1.upsample.filter_const"], (1, 1, 256, 256))
        self.assertEqual(sum(name.endswith("noise_const") for name in state), 12)
        self.assertEqual(sum(name.endswith("filter_const") for name in state), 12)
        contract.validate_state_shapes(state)
        missing = dict(state)
        del missing["synthesis.b256.conv1.noise_const"]
        with self.assertRaisesRegex(ValueError, "missing"):
            contract.validate_state_shapes(missing)
        wrong = dict(state)
        wrong["encoder.b256.fromrgb.weight"] = (4, 128, 1, 1)
        with self.assertRaisesRegex(ValueError, "shape"):
            contract.validate_state_shapes(wrong)
        extra = dict(state, reference_probe=(4,))
        with self.assertRaisesRegex(ValueError, "extra"):
            contract.validate_state_shapes(extra)

    def test_gguf_axis_order_scalar_bytes_and_metadata(self):
        payload = struct.pack("<6f", 1, -2, 3, -4, 5, -6)
        scalar = struct.pack("<f", 0.125)
        tensors = [("conv.weight", (2, 3, 1, 1), payload), ("noise_strength", (), scalar)]
        metadata = {"general.architecture": "migan", "migan.image_size": 256}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.gguf"
            contract.write_f32_gguf(path, tensors, metadata)
            actual_meta, actual = contract.read_f32_gguf(path)
            self.assertEqual(actual_meta, metadata)
            self.assertEqual(actual, {"conv.weight": ((2, 3, 1, 1), payload), "noise_strength": ((1,), scalar)})
            raw = path.read_bytes()
            self.assertEqual(struct.unpack_from("<IIQQ", raw), (0x46554747, 3, 2, 2))
            # Independent raw tensor-info check catches a reader/writer sharing
            # the same erroneous dimension reversal convention.
            marker = struct.pack("<Q", len("conv.weight")) + b"conv.weight"
            start = raw.index(marker) + len(marker)
            self.assertEqual(struct.unpack_from("<I4QIQ", raw, start), (4, 1, 1, 3, 2, 0, 0))
            self.assertEqual(raw[-32:-28], scalar)
            with self.assertRaisesRegex(ValueError, "byte count"):
                contract.write_f32_gguf(path, [("bad", (3,), scalar)], {})
            with self.assertRaisesRegex(ValueError, "duplicate"):
                contract.write_f32_gguf(path, tensors + tensors, {})

    def test_no_local_model_execution_dependency(self):
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("numpy", sys.modules)

    def test_converter_requires_original_before_framework_import(self):
        source = Path(__file__).resolve().parents[1] / "models/convert-migan-to-gguf.py"
        spec = importlib.util.spec_from_file_location("migan_converter_contract_test", source)
        converter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(converter)
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            report.write_text(json.dumps({"schema": "crispembed-migan-original-v1", "complete": False}))
            with self.assertRaisesRegex(ValueError, "complete original"):
                converter.convert("unused", "unused", "0" * 64, report, "unused")
            report.write_text(json.dumps({"schema": "crispembed-migan-original-v1", "complete": True,
                                          "checkpoint_sha256": "1" * 64}))
            with self.assertRaisesRegex(ValueError, "checkpoint differs"):
                converter.convert("unused", "unused", "0" * 64, report, "unused")
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
