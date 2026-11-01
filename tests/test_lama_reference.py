#!/usr/bin/env python3
"""Authored report/payload failure controls; no model or framework execution."""

import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import compare_lama_reference as compare
import lama_reference_contract as contract
import run_lama_native as driver


def payload(directory, filename, values=(0.5, -0.75)):
    path = directory / filename
    path.write_bytes(struct.pack("<" + "f" * len(values), *values))
    return {"file": filename, "shape": [1, 1, 1, len(values)], "sha256": contract.sha256(path)}


def write_case(directory, native=False):
    directory.mkdir()
    row = payload(directory, "vector.f32")
    records = [{"stage": name, **row} for name in contract.expected_stage_names()]
    identity = contract.fixture_identity("subtitle")
    _, _, rgb, mask = contract.fixture("subtitle")
    (directory / "output.rgb").write_bytes(rgb)
    manifest = {"schema": "crispembed.lama.native.v1" if native else "crispembed.lama.reference.v1",
                **identity, "records": list(reversed(records)) if native else records,
                "output_sha256": contract.sha256(directory / "output.rgb")}
    if native:
        manifest["model_sha256"] = "a" * 64
        manifest["controls"] = {key: True for key in compare.NATIVE_CONTROLS}
    else:
        manifest["hook_neutrality_bit_exact"] = True
        manifest["known_pixels_exact"] = True
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest, rgb, mask


class LamaReferenceTest(unittest.TestCase):
    def test_reference_framework_version_is_pinned(self):
        for version in ("2.8.0", "2.8.0+cpu"):
            contract.validate_reference_torch_version(version)
        for version in ("2.5.1", "2.8.1", "2.8.0.dev1", None):
            with self.assertRaisesRegex(ValueError, "requires PyTorch 2.8.0"):
                contract.validate_reference_torch_version(version)

    def test_exact_and_zero_vectors(self):
        for values in ([0.5, -1, 2], [0, 0]):
            result = compare.metrics(values, values)
            self.assertTrue(result["passed"])
            self.assertEqual(result["max_abs"], 0)
            self.assertEqual(result["cosine"], 1)

    def test_cosine_alone_cannot_pass_amplitude_error(self):
        result = compare.metrics([1.0, 2.0], [2.0, 4.0])
        self.assertEqual(result["cosine"], 1)
        self.assertFalse(result["passed"])
        self.assertEqual(result["max_abs"], 2)

    def test_relative_error_and_zero_reference_are_not_silently_skipped(self):
        result = compare.metrics([0.000001], [0.000002])
        self.assertLess(result["max_abs"], compare.MAX_ABSOLUTE_ERROR)
        self.assertFalse(result["passed"])
        result = compare.metrics([0], [0.000001])
        self.assertIsNone(result["relative_l2"])
        self.assertFalse(result["passed"])

    def test_nonfinite_empty_and_mismatched_vectors_rejected(self):
        for a, b in (([], []), ([1], [1, 2]), ([float("nan")], [0]), ([0], [float("inf")])):
            with self.assertRaises(ValueError):
                compare.metrics(a, b)

    def test_payload_size_hash_and_path_controls(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            row = payload(directory, "stage.f32")
            self.assertEqual(list(compare.read_stage(directory, row)), [0.5, -0.75])
            for changed in ({**row, "sha256": "0" * 64}, {**row, "shape": [1, 1, 1, 3]},
                            {**row, "file": "../stage.f32"}, {**row, "shape": [1, 1, 1, 2**31]}):
                with self.assertRaises(ValueError):
                    compare.read_stage(directory, changed)

    def test_stage_coverage_and_duplicate_names_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "case"
            manifest, _, _ = write_case(directory)
            for records in (manifest["records"][:-1], manifest["records"] + [manifest["records"][0]]):
                changed = {**manifest, "records": records}
                (directory / "manifest.json").write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    compare.read_manifest(directory, "crispembed.lama.reference.v1")

    def test_join_by_stage_name_with_different_native_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            original, native = Path(temporary) / "original", Path(temporary) / "native"
            write_case(original)
            write_case(native, True)
            report = compare.compare_case("subtitle", original, native, "a" * 64)
            self.assertTrue(report["passed"])
            self.assertEqual(len(report["stages"]), 713)
            self.assertIsNone(report["first_divergence"])

    def test_earliest_divergence_in_original_execution_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            original, native = Path(temporary) / "original", Path(temporary) / "native"
            write_case(original)
            manifest, _, _ = write_case(native, True)
            stages = contract.expected_stage_names()
            changed_payload = payload(native, "different.f32", (0.8, -0.75))
            for row in manifest["records"]:
                if row["stage"] in (stages[3], stages[12]):
                    row.update(changed_payload)
            (native / "manifest.json").write_text(json.dumps(manifest))
            report = compare.compare_case("subtitle", original, native, "a" * 64)
            self.assertFalse(report["passed"])
            self.assertEqual(report["first_divergence"]["stage"], stages[3])

    def test_known_pixel_change_fails_even_inside_rgb_error_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            original, native = Path(temporary) / "original", Path(temporary) / "native"
            write_case(original)
            manifest, rgb, mask = write_case(native, True)
            known = mask.index(0)
            changed = bytearray(rgb)
            changed[3 * known] += 1
            (native / "output.rgb").write_bytes(changed)
            manifest["output_sha256"] = contract.sha256(native / "output.rgb")
            (native / "manifest.json").write_text(json.dumps(manifest))
            report = compare.compare_case("subtitle", original, native, "a" * 64)
            self.assertEqual(report["image"]["max_abs"], 1)
            self.assertFalse(report["image"]["known_pixels_exact"])
            self.assertFalse(report["passed"])

    def test_native_output_hash_is_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            original, native = Path(temporary) / "original", Path(temporary) / "native"
            write_case(original)
            manifest, _, _ = write_case(native, True)
            manifest["output_sha256"] = "0" * 64
            (native / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "native decoded RGB identity"):
                compare.compare_case("subtitle", original, native, "a" * 64)

    def test_model_identity_and_missing_native_controls_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            original, native = Path(temporary) / "original", Path(temporary) / "native"
            write_case(original)
            manifest, _, _ = write_case(native, True)
            with self.assertRaisesRegex(ValueError, "model SHA-256"):
                compare.compare_case("subtitle", original, native, "b" * 64)
            del manifest["controls"]["fresh_context_exact"]
            (native / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "controls"):
                compare.compare_case("subtitle", original, native, "a" * 64)

    def test_original_first_barrier_prevents_native_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary, model = root / "binary", root / "model.gguf"
            binary.write_bytes(b"owned placeholder")
            model.write_bytes(b"owned placeholder")
            original = root / "original"
            original.mkdir()
            (original / "report.json").write_text(json.dumps({"schema": "crispembed-lama-original-v1", "complete": False}))
            with mock.patch.object(driver.subprocess, "run") as execute:
                with self.assertRaisesRegex(ValueError, "complete original-module"):
                    driver.run(binary, model, original, root / "native")
                execute.assert_not_called()
            self.assertFalse((root / "native").exists())

    def test_no_framework_imports(self):
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("numpy", sys.modules)


if __name__ == "__main__":
    unittest.main()
