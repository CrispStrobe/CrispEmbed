#!/usr/bin/env python3
"""Model-free controls: a malformed or failing native audit cannot look green."""
from pathlib import Path
import hashlib
import json
import tempfile
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/ci-heavy"))
from ultrabert_native import FROZEN_REFERENCE_SOURCE, parse_log, read_frozen_manifest, verify_frozen_reference


def log(failed=False):
    rows = [f"STAGE\tstage_{i}\t1\t0\t0\t0\t2\t2\t1" for i in range(292)]
    rows += [f"EXACT\texact_{i}\t1" for i in range(50)]
    if failed:
        rows[0] = "STAGE\tstage_0\t0.999\t0.003\t0.001\t0.001\t2\t2\t0"
    rows.append(f"TOTAL\t292\t50\t{int(failed)}")
    return "\n".join(rows)


class NativeLogTest(unittest.TestCase):
    def test_complete_passing_and_failing_records(self):
        self.assertEqual(parse_log(log())["failures"], 0)
        self.assertEqual(parse_log(log(True))["failures"], 1)
        self.assertFalse(parse_log(log(True))["stages"]["stage_0"]["pass"])
        baseline = {"source_commit": FROZEN_REFERENCE_SOURCE,
                    "status": "completed conversion/reference audit; native parity not evaluated",
                    "reference": {"frozen_arrays": "original SHA"}, "conversions": ["original bytes"]}
        verify_frozen_reference(baseline, baseline)
        for changed in ({**baseline, "reference": {}}, {**baseline, "conversions": ["changed"]}):
            with self.assertRaisesRegex(ValueError, "drifted"):
                verify_frozen_reference(baseline, changed)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            verify_frozen_reference({**baseline, "status": "started"}, baseline)

    def test_original_manifest_bytes_are_frozen_before_heavy_imports(self):
        raw_record = {"source_commit": FROZEN_REFERENCE_SOURCE,
                      "status": "completed conversion/reference audit; native parity not evaluated",
                      "reference": {"arrays": "original"}, "conversions": ["original"]}
        raw = json.dumps(raw_record).encode()
        baseline = {**raw_record, "audit_provenance": {"raw_report_sha256": hashlib.sha256(raw).hexdigest()}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ultrabert-reference-audit.json"
            path.write_bytes(raw)
            self.assertEqual(read_frozen_manifest(baseline, Path(directory)), raw_record)
            path.write_bytes(raw + b"\n")
            with self.assertRaisesRegex(ValueError, "report bytes"):
                read_frozen_manifest(baseline, Path(directory))
            path.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "report bytes"):
                read_frozen_manifest(raw_record, Path(directory))

    def test_missing_stage_rejected(self):
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            parse_log(log().split("\n", 1)[1])

    def test_duplicate_stage_and_total_rejected(self):
        for extra in ("STAGE\tstage_0\t1\t0\t0\t0\t2\t2\t1", "TOTAL\t292\t50\t0", "EXACT\texact_0\t1"):
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, "duplicate"):
                parse_log(log() + "\n" + extra)

    def test_forged_green_stage_rejected(self):
        with self.assertRaisesRegex(ValueError, "verdict"):
            parse_log(log(True).replace("0.003\t0.001\t0.001\t2\t2\t0", "0.003\t0.001\t0.001\t2\t2\t1"))

    def test_forged_zero_failure_total_rejected(self):
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            parse_log(log(True).replace("TOTAL\t292\t50\t1", "TOTAL\t292\t50\t0"))

    def test_exact_mismatch_is_retained(self):
        text = log().replace("EXACT\texact_0\t1", "EXACT\texact_0\t0").replace("TOTAL\t292\t50\t0", "TOTAL\t292\t50\t1")
        self.assertEqual(parse_log(text)["failures"], 1)
        self.assertFalse(parse_log(text)["exact"]["exact_0"])

    def test_nonfinite_failure_retained_but_green_rejected(self):
        bad = log(True).replace("0.999\t0.003", "nan\tinf")
        result = parse_log(bad)
        self.assertIsNone(result["stages"]["stage_0"]["cos_min"])
        self.assertEqual(result["failures"], 1)
        with self.assertRaisesRegex(ValueError, "verdict"):
            parse_log(log().replace("stage_0\t1\t0", "stage_0\tnan\tinf"))

    def test_negative_errors_and_malformed_fields_rejected(self):
        for text in (log().replace("stage_0\t1\t0", "stage_0\t1\t-1"),
                     log().replace("EXACT\texact_0\t1", "EXACT\texact_0\t2")):
            with self.subTest(text=text[:50]), self.assertRaises(ValueError):
                parse_log(text)


if __name__ == "__main__":
    unittest.main()
