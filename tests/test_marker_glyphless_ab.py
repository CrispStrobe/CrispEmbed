#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Model-free controls for the predeclared font follow-up's strict accounting."""
import copy
import json
import unittest

from marker_glyphless_ab import line_records, renderer_equal, style_warning_changes


class GlyphlessAccountingTests(unittest.TestCase):
    def test_renderer_comparison_excludes_only_elapsed_time(self):
        a = {"metadata": {"configuration": {"elapsed_seconds": "1", "source_revision": "old"}}, "text": "A"}
        b = copy.deepcopy(a)
        b["metadata"]["configuration"]["elapsed_seconds"] = "2"
        self.assertTrue(renderer_equal(json.dumps(a), json.dumps(b), "json"))
        b["metadata"]["configuration"]["source_revision"] = "different"
        self.assertFalse(renderer_equal(json.dumps(a), json.dumps(b), "json"))
        self.assertFalse(renderer_equal("A\n", "A", "markdown"))

    def test_line_multiset_retains_text_geometry_flags_and_duplicates(self):
        span = {"text": "Body text", "bbox": [1, 2, 3, 4], "font_size": 12,
                "monospace": True, "bold": False, "italic": False, "url": ""}
        block = {"block_type": "Code", "lines": [{"bbox": [1, 2, 3, 4], "spans": [span]}]}
        a = {"children": [{"id": "page0", "bbox": [0, 0, 100, 100], "children": [block]}]}
        b = copy.deepcopy(a)
        changed = b["children"][0]["children"][0]
        changed["block_type"] = "Text"
        changed["lines"][0]["spans"][0]["monospace"] = False
        self.assertEqual(line_records(a), line_records(b))
        self.assertNotEqual(line_records(a, True), line_records(b, True))
        for key, value in (("text", "Body typo"), ("bbox", [1, 2, 30, 4]), ("bold", True)):
            c = copy.deepcopy(b)
            c["children"][0]["children"][0]["lines"][0]["spans"][0][key] = value
            self.assertNotEqual(line_records(a), line_records(c))
        b["children"][0]["children"].append(copy.deepcopy(changed))
        self.assertNotEqual(line_records(a), line_records(b))

    def test_null_leaf_children_lines_and_spans_are_valid_empty_sequences(self):
        page = {"id": "page0", "bbox": [0, 0, 100, 100], "children": [
            {"block_type": "Picture", "children": None, "lines": None}]}
        self.assertFalse(line_records({"children": [page]}))
        page["children"][0]["lines"] = [{"bbox": [1, 2, 3, 4], "spans": None}]
        self.assertEqual(sum(line_records({"children": [page]}).values()), 1)

    def test_warning_accounting_does_not_waive_removal_or_unrelated_additions(self):
        a = {"children": [{"warnings": ["existing warning"]}]}
        b = {"children": [{"warnings": ["existing warning",
            "Glyphless OCR font style is unknown; artificial monospace styling ignored for 12 characters"]}]}
        self.assertTrue(style_warning_changes(a, b)["passed"])
        b["children"][0]["warnings"].append("unrelated new warning")
        self.assertFalse(style_warning_changes(a, b)["passed"])
        self.assertFalse(style_warning_changes(a, {"children": [{"warnings": []}]})["passed"])
        self.assertFalse(style_warning_changes(a, {"children": []})["passed"])
        self.assertFalse(style_warning_changes(a, {"children": [{"warnings": ["existing warning",
            "Another glyphless style is unknown diagnostic"]}]})["passed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
