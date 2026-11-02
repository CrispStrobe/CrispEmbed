#!/usr/bin/env python3
"""Small stdlib tests for the independent512 source/shape/fixture boundary."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import migan512_reference_contract as m
import migan_reference_contract as old


class MIgan512ContractTests(unittest.TestCase):
    def test_independent_profile_preserves256_contract(self):
        contract = m.load_contract()
        self.assertEqual(contract["schema"], "crispembed-migan512-reference-contract-v1")
        self.assertEqual(contract["tensor_count"], 177)
        self.assertEqual(contract["input"]["width"], 512)
        self.assertEqual(contract["checkpoint"]["declared_bytes"], 29553256)
        self.assertIsNone(contract["checkpoint"]["upstream_sha256"])
        self.assertFalse(contract["native_qualified"])
        self.assertEqual(len(old.expected_state_shapes()), 154)
        self.assertEqual(old.load_contract()["input"]["width"], 256)
        self.assertEqual(contract["acceptance"], old.load_contract()["acceptance"])

    def test_schema_preserves_intermediate256_weights_and_adds512(self):
        shapes = m.expected_state_shapes()
        m.validate_state_shapes(shapes)
        self.assertEqual(len(shapes), 177)
        for name, shape in old.expected_state_shapes().items():
            if name.startswith("encoder.b256.fromrgb"):
                self.assertNotIn(name, shapes)
            else:
                self.assertEqual(shapes[name], shape)
        self.assertEqual(shapes["encoder.b512.fromrgb.weight"], (64, 4, 1, 1))
        self.assertEqual(shapes["synthesis.b512.conv1.upsample.filter_const"], (1, 1, 512, 512))
        self.assertEqual(shapes["synthesis.b512.conv1.noise_const"], (512, 512))
        self.assertEqual(shapes["synthesis.b512.conv1.noise_strength"], ())

    def test_schema_rejects256_or_missing_noise_or_wrong_shape(self):
        with self.assertRaises(ValueError):
            m.validate_state_shapes(old.expected_state_shapes())
        shapes = m.expected_state_shapes()
        del shapes["synthesis.b512.conv1.noise_const"]
        with self.assertRaises(ValueError):
            m.validate_state_shapes(shapes)
        shapes = m.expected_state_shapes()
        shapes["encoder.b512.fromrgb.weight"] = (128, 4, 1, 1)
        with self.assertRaises(ValueError):
            m.validate_state_shapes(shapes)

    def test_fixtures_match_frozen_bytes_and_real512_edge_masks(self):
        self.assertEqual(m.load_contract()["fixtures"], [m.fixture_identity(name) for name in m.CASES])
        rgb, mask = m.fixture("one_pixel")
        self.assertEqual(len(rgb), 512 * 512 * 3)
        self.assertEqual(mask.count(255), 1)
        self.assertEqual(mask[256 * 512 + 256], 255)
        m.validate_input(rgb, mask)
        with self.assertRaises(ValueError):
            m.validate_input(rgb, mask, 256, 512)
        with self.assertRaises(ValueError):
            m.validate_input(rgb[:-1], mask)
        with self.assertRaises(ValueError):
            m.validate_input(rgb, bytes([127]) + mask[1:])
        with self.assertRaises(ValueError):
            m.fixture("missing")

    def test_capture_cannot_accept256_or_incomplete_or_changed_source(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "report.json"
            for report in ({"schema": "crispembed-migan-original-v1", "complete": True},
                           {"schema": "crispembed-migan512-original-v1", "complete": False},
                           {"schema": "crispembed-migan512-original-v1", "complete": True,
                            "checkpoint_sha256": "a" * 64, "source_commit": "changed"}):
                path.write_text(json.dumps(report))
                with self.assertRaises(ValueError):
                    m.validate_capture(path, "a" * 64)

    def test_capture_rejects_missing512_stages_before_artifact_reads(self):
        report = {"schema": "crispembed-migan512-original-v1", "complete": True,
                  "checkpoint_sha256": "a" * 64, "source_commit": m.SOURCE_COMMIT,
                  "contract_sha256": m.sha256(m.CONTRACT),
                  "cases": [{**m.fixture_identity(name), "known_pixels_bit_exact": True,
                             "hook_neutrality_bit_exact": True, "stages": []} for name in m.CASES]}
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "mandatory/unique"):
                m.validate_capture(path, "a" * 64)

    def test_digest_syntax_rejects_unpinned_metadata(self):
        self.assertTrue(m.valid_digest("f" * 64))
        for value in (None, "", "F" * 64, "0" * 63, "g" * 64):
            self.assertFalse(m.valid_digest(value))


if __name__ == "__main__":
    unittest.main()
