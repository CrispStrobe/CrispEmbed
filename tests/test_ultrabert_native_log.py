#!/usr/bin/env python3
"""Model-free controls: a malformed or failing native audit cannot look green."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/ci-heavy"))
from ultrabert_native import parse_log


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
