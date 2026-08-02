#!/usr/bin/env python3
"""Unit tests for the model-free Tesseract page geometry comparator."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from compare_tesseract_page_geometry import compare, greedy_iou_matches, reading_order_is_monotonic, run  # noqa: E402
from compare_tesseract_page_metrics import acceptance_checks  # noqa: E402
from compare_tesseract_page_metrics import selected_pageseg_policy  # noqa: E402
from benchmark_tesseract_page import summarize  # noqa: E402


class TesseractPageGeometryTest(unittest.TestCase):
    def test_reading_order(self) -> None:
        ordered = [(0.0, 0.0, 10.0, 5.0), (2.0, 10.0, 10.0, 5.0)]
        reversed_boxes = [ordered[1], ordered[0]]
        self.assertTrue(reading_order_is_monotonic(ordered))
        self.assertFalse(reading_order_is_monotonic(reversed_boxes))

    def test_crop_and_spacing_deltas(self) -> None:
        reference = [(0.0, 0.0, 10.0, 5.0), (0.0, 10.0, 10.0, 5.0)]
        native = [(1.0, 0.0, 11.0, 5.0), (1.0, 11.0, 10.0, 5.0)]
        result = compare(reference, native)
        self.assertEqual(result["reference_lines"], 2)
        self.assertEqual(result["native_lines"], 2)
        self.assertEqual(result["count_delta"], 0)
        self.assertEqual(result["mean_abs_crop_delta"], 0.5)
        self.assertEqual(result["max_abs_crop_delta"], 1.0)
        self.assertEqual(result["mean_abs_interline_gap_delta"], 1.0)
        self.assertTrue(result["paired_reading_order_consistent"])

    def test_order_regression_is_reported_even_when_counts_match(self) -> None:
        reference = [(0.0, 0.0, 10.0, 5.0), (0.0, 10.0, 10.0, 5.0)]
        native = [reference[1], reference[0]]
        result = compare(reference, native)
        self.assertEqual(result["count_delta"], 0)
        self.assertFalse(result["native_reading_order_monotonic"])
        self.assertFalse(result["paired_reading_order_consistent"])

    def test_iou_matching_is_independent_of_index_order(self) -> None:
        reference = [(0.0, 0.0, 10.0, 5.0), (0.0, 10.0, 10.0, 5.0)]
        native = [reference[1], reference[0]]
        matches = greedy_iou_matches(reference, native)
        self.assertEqual([(r, n) for r, n, _ in matches], [(0, 1), (1, 0)])
        self.assertEqual(compare(reference, native)["matched_mean_iou"], 1.0)

    def test_matched_iou_is_distinct_from_indexed_iou(self) -> None:
        reference = [(0.0, 0.0, 10.0, 5.0), (0.0, 10.0, 10.0, 5.0)]
        native = [reference[1], reference[0]]
        result = compare(reference, native)
        self.assertLess(result["mean_indexed_iou"], result["matched_mean_iou"])
        self.assertEqual(result["matched_mean_abs_crop_delta"], 0.0)

    def test_matched_crop_delta_tracks_geometry_not_order(self) -> None:
        reference = [(0.0, 0.0, 10.0, 5.0), (0.0, 10.0, 10.0, 5.0)]
        native = [(1.0, 10.0, 10.0, 5.0), (1.0, 0.0, 10.0, 5.0)]
        result = compare(reference, native)
        self.assertEqual(result["matched_mean_abs_crop_delta"], 0.25)

    def test_matched_gap_delta_tracks_spacing_not_order(self) -> None:
        reference = [(0.0, 0.0, 10.0, 5.0), (0.0, 10.0, 10.0, 5.0)]
        native = [(0.0, 10.0, 10.0, 5.0), (0.0, 0.0, 10.0, 5.0)]
        result = compare(reference, native)
        self.assertEqual(result["matched_mean_abs_interline_gap_delta"], 0.0)

    def test_zero_iou_matches_are_reported(self) -> None:
        reference = [(0.0, 0.0, 10.0, 5.0)]
        native = [(100.0, 100.0, 10.0, 5.0)]
        result = compare(reference, native)
        self.assertEqual(result["matched_count"], 1)
        self.assertEqual(result["matched_positive_iou_count"], 0)
        self.assertEqual(result["matched_zero_iou_count"], 1)

    def test_positive_iou_coverage_is_distinguishable(self) -> None:
        reference = [(0.0, 0.0, 10.0, 5.0), (0.0, 10.0, 10.0, 5.0)]
        native = [(0.0, 0.0, 10.0, 5.0), (100.0, 100.0, 10.0, 5.0)]
        result = compare(reference, native)
        self.assertEqual(result["matched_positive_iou_count"], 1)
        self.assertEqual(result["matched_zero_iou_count"], 1)

    def test_geometry_harness_clears_runtime_experiment_gates(self) -> None:
        source = (ROOT / "tools/compare_tesseract_page_geometry.py").read_text()
        for key in (
            "CRISPEMBED_TESSERACT_PAGESEG_BOX_PAD",
            "CRISPEMBED_TESSERACT_CROP_PAD",
            "CRISPEMBED_TESSERACT_RECODE_BEAM_WIDTH",
            "CRISPEMBED_TESSERACT_DAWG_PREFIX_SCORE",
        ):
            self.assertIn(f'"{key}"', source)

    def test_geometry_harness_supports_explicit_tessdata(self) -> None:
        source = (ROOT / "tools/compare_tesseract_page_geometry.py").read_text()
        self.assertIn('"--tessdata-dir"', source)
        self.assertIn('env.pop("TESSDATA_PREFIX", None)', source)

    def test_geometry_harness_has_bounded_subprocess_timeout(self) -> None:
        source = (ROOT / "tools/compare_tesseract_page_geometry.py").read_text()
        self.assertIn('"--timeout"', source)
        self.assertIn("subprocess.TimeoutExpired", source)

    def test_timeout_becomes_explicit_runtime_error(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "command timed out"):
            run([sys.executable, "-c", "import time; time.sleep(0.2)"], timeout_seconds=0.01)

    def test_geometry_result_defines_timing_fields(self) -> None:
        source = (ROOT / "tools/compare_tesseract_page_geometry.py").read_text()
        self.assertIn('"official_geometry_subprocess"', source)
        self.assertIn('"native_geometry_subprocess"', source)
        self.assertIn('"native_stage_benchmark"', source)
        self.assertIn('"CRISPEMBED_OCR_ORCH_BENCH"', source)

    def test_page_quality_acceptance_gates(self) -> None:
        args = type("Args", (), {"min_native_regions": 12, "max_cer": 0.02, "max_wer": 0.09})()
        passing = acceptance_checks(args, {"regions": 12}, {"cer": 0.019, "wer": 0.089})
        failing = acceptance_checks(args, {"regions": 11}, {"cer": 0.021, "wer": 0.091})
        self.assertEqual(passing, {"min_native_regions": True, "max_cer": True, "max_wer": True})
        self.assertEqual(failing, {"min_native_regions": False, "max_cer": False, "max_wer": False})

    def test_page_quality_gates_are_opt_in(self) -> None:
        args = type("Args", (), {"min_native_regions": None, "max_cer": None, "max_wer": None})()
        self.assertEqual(acceptance_checks(args, {"regions": 0}, {"cer": 1.0, "wer": 1.0}), {})

    def test_all_pageseg_policies_are_explicit(self) -> None:
        for name in ("projection", "component", "baseline"):
            args = type("Args", (), {"projection": False, "component": False, "baseline": False})()
            setattr(args, name, True)
            self.assertEqual(selected_pageseg_policy(args), name)
        args = type("Args", (), {"projection": False, "component": False, "baseline": False})()
        self.assertEqual(selected_pageseg_policy(args), "legacy-fallback")

    def test_repeated_benchmark_summary_is_deterministic(self) -> None:
        self.assertEqual(summarize([1.0, 2.0, 3.0]), {"min": 1.0, "median": 2.0, "p90": 3.0, "max": 3.0})
        self.assertEqual(summarize([]), {"min": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0})


if __name__ == "__main__":
    unittest.main()
