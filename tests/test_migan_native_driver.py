#!/usr/bin/env python3
"""Dependency-free negative controls for native replay identity and reports."""

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import stat
import struct
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("migan_native_test", ROOT / "tools/ci-heavy/migan_native.py")
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


def good_args():
    return argparse.Namespace(phase="replay", reference_run="123", reference_artifact_id="456",
                              reference_head="1" * 40, reference_workflow=".github/workflows/migan-reference.yml",
                              reference_archive_sha256="2" * 64, reference_report_sha256="3" * 64,
                              conversion_report_sha256="4" * 64, checkpoint_sha256="5" * 64,
                              model_gguf_sha256="6" * 64)


def good_records(empty=True):
    stages = []
    for stage in ("generator.input", "known_mask", "generator.output"):
        stages.append({"stage": stage, "count": 4, "max_abs": 0.0, "mae": 0.0, "rms": 0.0,
                       "cos_min": 1.0, "cos_global": 1.0, "native_l2": 2.0, "original_l2": 2.0,
                       "finite": True, "shape_match": True, "passed": True})
    image = {"fill_channels": 0, "max_abs": None, "mae": None, "mse": None, "psnr_db": None,
             "known_errors": 0, "passed": True}
    if not empty:
        image.update(fill_channels=3, max_abs=0, mae=0, mse=0, exact_fill=True)
    return [("MIGAN_CONTRACT", {"passed": True}),
            ("MIGAN_GATES", {"unset_rejected": True, "empty_rejected": True, "literal_zero_rejected": True}),
            *[("MIGAN_DIFF", stage) for stage in stages], ("MIGAN_IMAGE", image),
            ("MIGAN_SUMMARY", {"compared": 3, "expected": 3, "failed_float_stages": 0, "first_failure": "",
                               "observer_neutrality_bit_exact": True, "observed_graph_buffer_bytes": 0,
                               "generator_seconds": 0.1, "image_passed": True, "passed": True})]


def log(records):
    return "backend diagnostics\n" + "\n".join(tag + " " + json.dumps(row) for tag, row in records)


class NativeDriverTest(unittest.TestCase):
    expected = ["generator.input", "known_mask", "generator.output"]

    def test_replay_requires_every_reviewed_identity(self):
        native.validate_inputs(good_args())
        for key in vars(good_args()):
            if key == "phase":
                continue
            args = good_args()
            setattr(args, key, "")
            with self.subTest(field=key), self.assertRaises(ValueError):
                native.validate_inputs(args)
        args = argparse.Namespace(phase="build-only")
        native.validate_inputs(args)
        args = good_args()
        args.reference_workflow = ".github/workflows/unrelated.yml"
        with self.assertRaises(ValueError):
            native.validate_inputs(args)

    def test_valid_empty_and_exact_image_records(self):
        for empty in (True, False):
            result = native.parse_log(log(good_records(empty)), self.expected)
            self.assertTrue(result["summary"]["passed"])

    def test_numerical_failure_is_preserved(self):
        records = good_records()
        records[2][1].update(max_abs=0.003, passed=False)
        records[-1][1].update(failed_float_stages=1, first_failure="generator.input", passed=False)
        result = native.parse_log(log(records), self.expected)
        self.assertFalse(result["summary"]["passed"])

    def test_rejects_missing_duplicate_order_or_contradictory_metrics(self):
        candidates = []
        candidates.append(good_records()[:-1])
        candidate = good_records()
        candidate.append(copy.deepcopy(candidate[-1]))
        candidates.append(candidate)
        candidate = good_records()
        candidate[2], candidate[3] = candidate[3], candidate[2]
        candidates.append(candidate)
        for key, value in (("max_abs", 0.1), ("cos_min", 2), ("mae", -1), ("count", 0), ("finite", False)):
            candidate = good_records()
            candidate[2][1][key] = value
            candidates.append(candidate)
        candidate = good_records()
        candidate[-2][1]["known_errors"] = 1
        candidates.append(candidate)
        candidate = good_records()
        candidate[-1][1]["compared"] = 2
        candidates.append(candidate)
        for records in candidates:
            with self.subTest(records=records), self.assertRaises(ValueError):
                native.parse_log(log(records), self.expected)

    def test_strict_json_rejects_duplicate_and_nonfinite(self):
        for payload in ('{"passed":true,"passed":false}', '{"max_abs":NaN}', '{"max_abs":Infinity}'):
            with self.assertRaises(ValueError):
                native.strict_json(payload)

    def test_zip_path_and_budget_guards(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for index, name in enumerate(("../outside", "/outside", "x\\outside", "C:outside")):
                archive = root / f"bad-{index}.zip"
                with zipfile.ZipFile(archive, "w") as stream:
                    stream.writestr(name, b"no")
                with self.assertRaises(ValueError):
                    native.safe_extract(archive, root / f"out-{index}")
            archive = root / "symlink.zip"
            member = zipfile.ZipInfo("link")
            member.external_attr = (stat.S_IFLNK | 0o777) << 16
            with zipfile.ZipFile(archive, "w") as stream:
                stream.writestr(member, "../outside")
            with self.assertRaises(ValueError):
                native.safe_extract(archive, root / "link-out")
            archive = root / "good.zip"
            with zipfile.ZipFile(archive, "w") as stream:
                stream.writestr("original/report.json", b"{}")
            with self.assertRaises(ValueError):
                native.safe_extract(archive, root / "oversize", max_bytes=1)
            native.safe_extract(archive, root / "good")
            self.assertEqual((root / "good/original/report.json").read_bytes(), b"{}")

    def test_no_framework_or_model_dependency(self):
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("numpy", sys.modules)

    def test_f16_negative_has_valid_scalar_payload_and_following_offset(self):
        tensors = {"before": ((2,), struct.pack("<2f", 1, 2)),
                   "noise_strength": ((1,), struct.pack("<f", 0.125)),
                   "after": ((2,), struct.pack("<2f", 3, 4))}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "f16.gguf"
            native.write_f32_gguf(path, [(key, shape, data) for key, (shape, data) in tensors.items()], {})
            original = path.read_bytes()
            native.retag_scalar_f16(path, tensors, "noise_strength")
            actual = path.read_bytes()
            self.assertEqual(len(actual), len(original))
            needle = struct.pack("<Q", len("noise_strength")) + b"noise_strength" + struct.pack("<IQ", 1, 1)
            start = actual.index(needle) + len(needle)
            self.assertEqual(struct.unpack_from("<IQ", actual, start), (1, 32))
            after = struct.pack("<Q", len("after")) + b"after" + struct.pack("<IQ", 1, 2)
            self.assertEqual(struct.unpack_from("<IQ", actual, actual.index(after) + len(after)), (0, 64))
            data_start = len(actual) - 96
            self.assertEqual(struct.unpack_from("<e", actual, data_start + 32)[0], 0.125)
            self.assertEqual(actual[data_start + 64:], original[data_start + 64:])


if __name__ == "__main__":
    unittest.main()
