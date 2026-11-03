#!/usr/bin/env python3
"""Negative controls for document structure, completion and provenance."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("document_eval", ROOT / "tools/ocr_document_eval.py")
ev = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ev)

TABLE = '<table><tr><th colspan="2">Stock</th></tr><tr><td>A</td><td>2</td></tr></table>'


def records():
    base = {"schema": ev.SCHEMA, "page_id": "owned-control", "protocol_sha256": "1" * 64,
            "input": {"source_sha256": "2" * 64, "rendered_sha256": "3" * 64,
                      "width": 300, "height": 400,
                      "rendering": {"tool": "owned-fixture", "version": "1", "settings": "RGB"}},
            "document": {"raw_text": "Stock A 2", "text": "Stock A 2", "html": TABLE,
                         "tables": [TABLE], "formulas": ["x^2"],
                         "blocks": [{"label": "table", "bbox": [0.1, 0.2, 0.9, 0.5],
                                     "text": "Stock A 2"}]}}
    gold = copy.deepcopy(base)
    gold["annotation"] = {"source": "owned synthetic control", "license": "MIT", "method": "authored"}
    prediction = copy.deepcopy(base)
    prediction["run"] = {"engine": "control", "precision": "F16", "backend": "cpu",
                         "task": "document", "build_mode": "portable", "runtime_revision": "4" * 40,
                         "expected_calls": ["page"],
                         "components": [{"name": "ocr", "repo": "owned/control", "revision": "5" * 40,
                                         "weights_sha256": "6" * 64, "processor_sha256": "7" * 64,
                                         "tokenizer_sha256": "8" * 64}]}
    prediction["generation"] = [{"id": "page", "status": "eos", "prompt": "", "max_new_tokens": 4,
                                  "generated_ids": [1, 2, 3], "eos_token_ids": [3]}]
    prediction["performance"] = {"elapsed_ms": None, "peak_rss_bytes": None, "span": "not measured"}
    return gold, prediction


class DocumentControls(unittest.TestCase):
    def test_complete_structured_document(self):
        result = ev.score(*records())
        self.assertTrue(result["exact_complete_document"])
        self.assertFalse(result["model_qualified"])
        self.assertIsNone(result["upstream_benchmark_score"])
        self.assertEqual(result["tables"]["items"][0]["prediction"]["cells"][0]["colspan"], 2)

    def test_raw_unicode_is_not_normalized(self):
        gold, pred = records()
        gold["document"]["raw_text"] = "é"
        pred["document"]["raw_text"] = "e\u0301"
        self.assertGreater(ev.score(gold, pred)["raw"]["cer"], 0)

    def test_empty_gold_penalizes_insertion(self):
        self.assertEqual(ev.text_errors("", "abc")["cer"], 3)

    def test_cap_never_counts_as_complete(self):
        gold, pred = records()
        pred["generation"][0].update(status="cap", generated_ids=[1, 2, 4, 5])
        result = ev.score(gold, pred)
        self.assertTrue(result["ordered_text"]["exact"])
        self.assertFalse(result["exact_complete_document"])

    def test_eos_requires_ids(self):
        _, pred = records()
        pred["generation"][0]["generated_ids"] = [1, 2]
        with self.assertRaisesRegex(ValueError, "EOS"):
            ev.validate_record(pred)

    def test_tokens_after_eos_rejected(self):
        _, pred = records()
        pred["generation"][0]["generated_ids"] = [3, 1, 3]
        with self.assertRaisesRegex(ValueError, "after EOS"):
            ev.validate_record(pred)

    def test_one_failed_region_makes_page_incomplete(self):
        gold, pred = records()
        pred["run"]["expected_calls"].append("crop")
        pred["generation"].append({"id": "crop", "status": "error", "prompt": "table", "max_new_tokens": 4,
                                   "generated_ids": None, "eos_token_ids": [3]})
        self.assertFalse(ev.score(gold, pred)["exact_complete_document"])

    def test_omitted_generation_call_rejected(self):
        _, pred = records()
        pred["run"]["expected_calls"].append("missing-crop")
        with self.assertRaisesRegex(ValueError, "coverage"):
            ev.validate_record(pred)

    def test_unknown_completion_is_not_success(self):
        gold, pred = records()
        pred["generation"][0].update(status="unknown", generated_ids=None)
        self.assertFalse(ev.score(gold, pred)["exact_complete_document"])

    def test_missing_gold_not_empty_gold(self):
        gold, pred = records()
        gold["document"]["tables"] = None
        result = ev.score(gold, pred)
        self.assertFalse(result["tables"]["available"])
        self.assertIsNone(result["exact_complete_document"])

    def test_missing_prediction_not_empty_prediction(self):
        gold, pred = records()
        pred["document"]["blocks"] = None
        self.assertFalse(ev.score(gold, pred)["blocks"]["prediction_available"])

    def test_missing_box_not_zero_iou(self):
        gold, pred = records()
        gold["document"]["blocks"][0]["bbox"] = None
        result = ev.score(gold, pred)
        self.assertIsNone(result["blocks"]["items"][0]["position_indexed_iou"])
        self.assertIsNone(result["exact_complete_document"])

    def test_reading_order_is_preserved(self):
        gold, pred = records()
        for record in (gold, pred):
            record["document"]["blocks"].append({"label": "text", "bbox": [0.1, 0.6, 0.9, 0.9], "text": "End"})
        pred["document"]["blocks"].reverse()
        self.assertFalse(ev.score(gold, pred)["blocks"]["exact"])

    def test_extra_block_is_failure(self):
        gold, pred = records()
        pred["document"]["blocks"] *= 2
        self.assertTrue(ev.score(gold, pred)["blocks"]["items"][1]["extra"])

    def test_table_span_error_does_not_hide_in_text(self):
        gold, pred = records()
        pred["document"]["tables"] = ['<table><tr><th>Stock</th><th></th></tr><tr><td>A</td><td>2</td></tr></table>']
        result = ev.score(gold, pred)
        self.assertTrue(result["ordered_text"]["exact"])
        self.assertFalse(result["tables"]["exact"])

    def test_rowspan_is_preserved(self):
        table = ev.Table().parse('<table><tr><td rowspan="2">A</td><td>B</td></tr><tr><td>C</td></tr></table>')
        self.assertEqual((table["rows"], table["columns"]), (2, 2))
        self.assertEqual(table["cells"][2]["column"], 1)

    def test_table_attributes_and_malformed_spans_fail_closed(self):
        for table in ('<table><tr><td rowspan="0">x</td></tr></table>',
                      '<table><tr><td rowspan="2">x</td></tr></table>',
                      '<table><tr><td colspan="2" colspan="2">x</td></tr></table>',
                      '<table><tr><td style="display:none">x</td></tr></table>',
                      '<table><tr><td>x</td><td>y</td></tr><tr><td>z</td></tr></table>',
                      '<table><tr><td><script>x</script></td></tr></table>',
                      '<table><tr><td>x</table>'):
            with self.subTest(table=table), self.assertRaises(ValueError):
                ev.Table().parse(table)

    def test_table_parse_error_is_retained(self):
        gold, pred = records()
        pred["document"]["tables"] = ["<table>"]
        result = ev.score(gold, pred)
        self.assertIn("prediction_error", result["tables"]["items"][0])
        self.assertFalse(result["exact_complete_document"])

    def test_formula_exactness_is_literal(self):
        gold, pred = records()
        pred["document"]["formulas"] = ["x^{2}"]
        self.assertFalse(ev.score(gold, pred)["formulas"]["exact"])

    def test_render_or_protocol_drift_rejected(self):
        for key in ("rendered_sha256", "source_sha256"):
            gold, pred = records()
            pred["input"][key] = "a" * 64
            with self.assertRaisesRegex(ValueError, "mismatch"):
                ev.score(gold, pred)
        gold, pred = records()
        pred["protocol_sha256"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "mismatch"):
            ev.score(gold, pred)

    def test_unpinned_checkpoint_rejected(self):
        _, pred = records()
        pred["run"]["components"][0]["revision"] = "main"
        with self.assertRaisesRegex(ValueError, "immutable"):
            ev.validate_record(pred)

    def test_bad_boxes_rejected_without_clamping(self):
        for box in ([0, 0, 2, 1], [0, 0, 0, 1], [1, 0, 0, 1], [False, 0, 1, 1], [float("nan"), 0, 1, 1]):
            _, pred = records()
            pred["document"]["blocks"][0]["bbox"] = box
            with self.subTest(box=box), self.assertRaisesRegex(ValueError, "bbox"):
                ev.validate_record(pred)

    def test_opb_export_keeps_structure(self):
        _, pred = records()
        output = ev.export_opb(pred)
        self.assertEqual(output["html"], TABLE)
        self.assertEqual(set(output["blocks"][0]), {"label", "bbox", "text"})
        self.assertEqual(output["blocks"][0]["bbox"], [0.1, 0.2, 0.9, 0.5])

    def test_opb_export_does_not_invent_layout(self):
        _, pred = records()
        pred["document"]["blocks"] = None
        self.assertNotIn("blocks", ev.export_opb(pred))
        pred["document"]["html"] = None
        with self.assertRaisesRegex(ValueError, "no observed"):
            ev.export_opb(pred)

    def test_json_rejects_duplicate_keys_and_nan(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            for value in ('{"x":1,"x":2}', '{"x":NaN}'):
                path.write_text(value, encoding="utf-8")
                with self.assertRaises(ValueError):
                    ev.load(path)


if __name__ == "__main__":
    unittest.main()
