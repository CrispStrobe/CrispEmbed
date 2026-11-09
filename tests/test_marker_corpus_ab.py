#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Controls for physical coordinates and lossless retrieval/provenance accounting."""
import unittest

from marker_corpus_ab import prediction_blocks, ranges_valid, utf8_windows, windows_cover


class CorpusAccountingTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
