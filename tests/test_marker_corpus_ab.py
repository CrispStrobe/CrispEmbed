#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Controls for physical coordinates and lossless retrieval/provenance accounting."""
import unittest

from marker_corpus_ab import image_dimensions, prediction_blocks, ranges_valid, utf8_windows, windows_cover


class CorpusAccountingTests(unittest.TestCase):
    def test_unboxed_dimension_discrepancy_is_recorded_without_rescaling_gold(self):
        page = {"image": {"width": 1600, "height": 2487}, "blocks": [{"bbox": None}]}
        result = image_dimensions(page, (1920, 2985))
        self.assertEqual(result["stored"], [1600, 2487])
        self.assertEqual(result["decoded"], [1920, 2985])
        self.assertFalse(result["match"])
        self.assertFalse(result["gold_rescaled"])
        self.assertFalse(result["spatial_qualification_available"])
        self.assertEqual(page["image"], {"width": 1600, "height": 2487})
        self.assertIsNone(page["blocks"][0]["bbox"])

    def test_boxed_dimension_discrepancy_still_rejects(self):
        page = {"image": {"width": 100, "height": 200}, "blocks": [{"bbox": [1, 2, 30, 40]}]}
        self.assertTrue(image_dimensions(page, (100, 200))["spatial_qualification_available"])
        with self.assertRaisesRegex(ValueError, "gold boxes"):
            image_dimensions(page, (200, 400))
        self.assertEqual(page["blocks"][0]["bbox"], [1, 2, 30, 40])

    def test_pdf_point_boxes_are_scaled_to_frozen_scan_pixels(self):
        page = {"bbox": [0, 0, 100, 200], "children": [
            {"id": "x", "block_type": "Text", "text": "A", "bbox": [10, 20, 40, 50]}]}
        prediction = prediction_blocks({"children": [page]}, 1000, 3000)
        self.assertEqual(prediction[0]["bbox"], [100, 300, 400, 750])

    def test_utf8_windows_preserve_every_byte_without_truncation(self):
        text = ("A page with German ß, French é, 日本語 and 📖. " * 25).strip()
        windows = utf8_windows(text)
        self.assertTrue(windows_cover(text, windows))
        self.assertTrue(all(len(row["text"].encode()) <= 240 for row in windows))
        self.assertFalse(windows_cover(text, windows[:-1]))
        changed = [dict(row) for row in windows]
        changed[-1]["text"] += "!"
        self.assertFalse(windows_cover(text, changed))

    def test_source_ranges_reject_fabricated_text_and_foreign_source(self):
        sources = {"s": {"text": "日本"}}
        spans = [{"source_id": "s", "source_start": 0, "source_end": 6,
                  "text_start": 0, "text_end": 6, "transform": "copy"}]
        self.assertTrue(ranges_valid("日本", spans, sources))
        self.assertFalse(ranges_valid("本日", spans, sources))
        spans[0]["source_id"] = "missing"
        self.assertFalse(ranges_valid("日本", spans, sources))

    def test_removed_hyphen_requires_actual_removed_byte(self):
        sources = {"s": {"text": "end-"}}
        spans = [{"source_id": "s", "source_start": 0, "source_end": 3,
                  "text_start": 0, "text_end": 3, "transform": "copy"},
                 {"source_id": "s", "source_start": 3, "source_end": 4,
                  "text_start": 3, "text_end": 3, "transform": "dehyphenation"}]
        self.assertTrue(ranges_valid("end", spans, sources, True))
        self.assertFalse(ranges_valid("end", spans, sources, False))
        sources["s"]["text"] = "end!"
        self.assertFalse(ranges_valid("end", spans, sources, True))

    def test_whitespace_matches_native_ascii_contract(self):
        for text, allowed in ((" \t\n\r", True), ("\f", False), ("\v", False),
                              ("\u00a0", False), ("\u3000", False), ("x", False)):
            with self.subTest(text=repr(text)):
                spans = [{"source_id": "s", "source_start": 0, "source_end": len(text.encode()),
                          "text_start": 0, "text_end": 1, "transform": "whitespace"}]
                self.assertEqual(ranges_valid(" ", spans, {"s": {"text": text}}), allowed)
                spans[0]["text_end"] = 0
                self.assertEqual(ranges_valid("", spans, {"s": {"text": text}}), allowed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
