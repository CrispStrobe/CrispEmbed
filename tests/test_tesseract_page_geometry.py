#!/usr/bin/env python3
"""Unit tests for the model-free Tesseract page geometry comparator."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from compare_tesseract_page_geometry import compare, reading_order_is_monotonic  # noqa: E402
from compare_tesseract_crop_geometry import compare as compare_crop_geometry  # noqa: E402
from compare_tesseract_crop_geometry import compare_geometry  # noqa: E402
from compare_tesseract_page_metrics import acceptance_checks  # noqa: E402
from compare_tesseract_page_metrics import observed_detector_route  # noqa: E402
from compare_tesseract_page_metrics import selected_detector_route  # noqa: E402
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

    def test_page_quality_acceptance_gates(self) -> None:
        args = type("Args", (), {"min_native_regions": 12, "max_cer": 0.02, "max_wer": 0.09})()
        passing = acceptance_checks(args, {"regions": 12}, {"cer": 0.019, "wer": 0.089})
        failing = acceptance_checks(args, {"regions": 11}, {"cer": 0.021, "wer": 0.091})
        self.assertEqual(passing, {"min_native_regions": True, "max_cer": True, "max_wer": True})
        self.assertEqual(failing, {"min_native_regions": False, "max_cer": False, "max_wer": False})

    def test_page_quality_gates_are_opt_in(self) -> None:
        args = type("Args", (), {"min_native_regions": None, "max_cer": None, "max_wer": None})()
        self.assertEqual(acceptance_checks(args, {"regions": 0}, {"cer": 1.0, "wer": 1.0}), {})

    def test_page_text_gate_rejects_approximate_only_match(self) -> None:
        args = type("Args", (), {
            "min_native_regions": None,
            "max_cer": None,
            "max_wer": None,
            "require_text_match": True,
        })()
        comparison = {
            "cer": 0.01,
            "wer": 0.02,
            "official_text": "Brighton",
            "native_text": "Drighton",
        }
        self.assertEqual(acceptance_checks(args, {"regions": 1}, comparison), {"text_match": False})

    def test_all_pageseg_policies_are_explicit(self) -> None:
        for name in ("projection", "component", "baseline"):
            args = type("Args", (), {"projection": False, "component": False, "baseline": False})()
            setattr(args, name, True)
            self.assertEqual(selected_pageseg_policy(args), name)
        args = type("Args", (), {"projection": False, "component": False, "baseline": False})()
        self.assertEqual(selected_pageseg_policy(args), "legacy-fallback")

    def test_native_pageseg_is_a_distinct_route(self) -> None:
        args = type("Args", (), {"native_pageseg": True})()
        self.assertEqual(selected_detector_route(args), "native-tesseract-pageseg")
        args.native_pageseg = False
        self.assertEqual(selected_detector_route(args), "dbnet")

    def test_observed_route_comes_from_the_run_not_the_request(self) -> None:
        # The law: a REQUEST is not an OBSERVATION. Both directions of the H9
        # router made the old single `detector_route` field lie, so the report
        # now carries the route the run itself printed.
        classical = ("noise\n[tesseract-seg-router] columns=1 ink_coverage=0.8877 "
                     "boxes=22 path=classical\nmore noise\n")
        self.assertEqual(observed_detector_route(classical),
                         {"path": "classical", "columns": 1, "ink_coverage": 0.8877, "boxes": 22})
        fallback = ("[tesseract-seg-router] columns=2 ink_coverage=1.0000 "
                    "boxes=289 path=dbnet(fallback)\n")
        self.assertEqual(observed_detector_route(fallback)["path"], "dbnet")
        self.assertEqual(observed_detector_route(fallback)["columns"], 2)

    def test_observed_route_is_none_when_unobserved(self) -> None:
        # Absent must read as absent. Defaulting to "dbnet" here is exactly the
        # guess that made the old label wrong.
        self.assertIsNone(observed_detector_route("no router line here\n"))

    def test_observed_route_takes_the_last_line(self) -> None:
        # A multi-stage chain prints one line per tesseract stage; the report is
        # about the stage that produced the metrics, i.e. the last one.
        two = ("[tesseract-seg-router] columns=1 ink_coverage=1.0000 boxes=3 path=classical\n"
               "[tesseract-seg-router] columns=2 ink_coverage=1.0000 boxes=289 path=dbnet(fallback)\n")
        self.assertEqual(observed_detector_route(two)["boxes"], 289)

    def test_benchmark_wrapper_exposes_native_route_flag(self) -> None:
        wrapper = (ROOT / "tools" / "benchmark_tesseract_page.py").read_text()
        self.assertIn('parser.add_argument("--native-pageseg"', wrapper)
        self.assertIn('command.append("--native-pageseg")', wrapper)

    def test_benchmark_wrapper_exposes_row_blob_bounds(self) -> None:
        wrapper = (ROOT / "tools" / "benchmark_tesseract_page.py").read_text()
        comparator = (ROOT / "tools" / "compare_tesseract_page_metrics.py").read_text()
        self.assertIn('parser.add_argument("--row-blob-bounds"', wrapper)
        self.assertIn('command.append("--row-blob-bounds")', wrapper)
        self.assertIn('parser.add_argument("--row-blob-bounds"', comparator)
        self.assertIn('CRISPEMBED_TESSERACT_PAGESEG_ROW_BLOB_BOUNDS', comparator)

    def test_direct_geometry_comparator_exposes_row_blob_bounds(self) -> None:
        comparator = (ROOT / "tools" / "compare_tesseract_page_geometry.py").read_text()
        self.assertIn('parser.add_argument("--row-blob-bounds"', comparator)
        self.assertIn('args.row_blob_bounds', comparator)

    def test_crop_geometry_rejects_index_alignment_on_count_mismatch(self) -> None:
        result = compare_crop_geometry(
            [{"box_x": 0.0, "box_y": 0.0, "box_w": 10.0, "box_h": 5.0, "crop_w": 10.0, "crop_h": 5.0}],
            [
                {"left": 0, "top": 0, "width": 10, "height": 5},
                {"left": 0, "top": 10, "width": 10, "height": 5},
            ],
        )
        self.assertFalse(result["alignment_valid"])
        self.assertEqual(result["paired_rows"], 1)

    def test_crop_geometry_can_report_unmatched_rows(self) -> None:
        native = [{"box_x": 0.0, "box_y": 0.0, "box_w": 10.0, "box_h": 20.0,
                   "crop_w": 10.0, "crop_h": 20.0}]
        official = [{"left": 0, "top": 0, "width": 10, "height": 5},
                    {"left": 0, "top": 10, "width": 10, "height": 5}]
        result = compare_geometry(native, official)
        self.assertEqual(result["alignment"], "monotonic-geometry")
        self.assertEqual(result["matched_rows"], 1)
        self.assertEqual(result["unmatched_official"], [0])
        self.assertEqual(result["merged_official_groups"],
                         [{"native_index": 0, "official_indices": [0, 1],
                           "primary_official_index": 0, "nested_official_indices": []}])

    def test_repeated_benchmark_summary_is_deterministic(self) -> None:
        self.assertEqual(summarize([1.0, 2.0, 3.0]), {"min": 1.0, "median": 2.0, "p90": 3.0, "max": 3.0})
        self.assertEqual(summarize([]), {"min": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0})


if __name__ == "__main__":
    unittest.main()
