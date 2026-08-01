#!/usr/bin/env python3
"""Unit tests for the model-free Tesseract page geometry comparator."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from compare_tesseract_page_geometry import compare, reading_order_is_monotonic  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
