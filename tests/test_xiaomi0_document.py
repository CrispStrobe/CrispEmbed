"""Authored document/OTSL controls. No OCR, detector, native build, or network."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("xiaomi0_document_tested", ROOT / "tools/xiaomi0_document.py")
doc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(doc)


def records():
    source = doc.policy.evidence()
    detector = {"model": source["layout"]["model"], "revision": source["layout"]["revision"],
                "checkpoint_files": source["layout"]["checkpoint_files"], "threshold": 0.3,
                "forward_executed": True, "source_revision": source["upstream"]["revision"],
                "image_sha256": "1" * 64}
    layout = {"schema": "xiaomi0-original-layout/v1", "detector": detector,
              "native_layout_implemented": False, "document_qualified": False,
              "regions": [
                  {"index": 0, "label": "doc_title", "task_type": "text", "bbox_2d": [10, 10, 900, 100]},
                  {"index": 1, "label": "table", "task_type": "table", "bbox_2d": [10, 150, 900, 500]},
                  {"index": 2, "label": "display_formula", "task_type": "formula", "bbox_2d": [10, 600, 900, 700]},
                  {"index": 3, "label": "chart", "task_type": "skip", "bbox_2d": [10, 800, 900, 990]},
              ]}
    protocol = {"schema": "xiaomi0-region-protocol/v1", "page_id": "authored-control",
                "layout_record_sha256": doc.digest(layout), "max_regions": 4, "max_total_new_tokens": 16,
                "caps": {"text": 4, "table": 4, "formula": 4},
                "input": {"source_sha256": "1" * 64, "rendered_sha256": "1" * 64, "width": 512, "height": 384,
                          "rendering": {"tool": "authored-control", "version": "1", "settings": "RGB"}},
                "recognition": {"precision": "f16", "build_mode": "portable", "weights_sha256": "2" * 64,
                                "runner_sha256": "3" * 64, "runtime_revision": "4" * 40}}
    texts = ["Stock", "<fcel>Heading<lcel><nl><fcel>A<fcel>2<nl>", "\\frac{x}{2} "]
    observations = [{"id": f"region-{i:03d}", "index": i,
                     "native": {"raw_text_utf8_hex": text.encode().hex(), "cap": 4,
                                "token_ids": [100, 248046], "confidences": [.7, .8],
                                "stop_kind": 1, "stop_id": 248046, "wall_seconds": .2, "init_seconds": .1}}
                    for i, text in enumerate(texts)]
    capture = {"schema": "xiaomi0-native-regions/v1", "protocol_sha256": "5" * 64,
               "recognition": copy.deepcopy(protocol["recognition"]), "observations": observations}
    return layout, protocol, capture


class OTSLControls(unittest.TestCase):
    def test_actual_upstream_valid_table_fixtures(self):
        path = ROOT / "tests/fixtures/xiaomi0-document/otsl-valid-reference.json"
        self.assertEqual(doc.policy.file_sha256(path), "e5aa3344ea74d06299ceb66212ba2662b9c1425552ca4fdb67c5d4f5a1d3bddc")
        reference = json.loads(path.read_text())
        self.assertEqual(reference["source"]["sha256"], doc.OTSL_SOURCE["sha256"])
        for case in reference["cases"]:
            with self.subTest(case=case["id"]):
                self.assertEqual(doc.otsl_to_html(case["otsl"]), case["html"])

    def test_literal_content_and_nonunit_spans(self):
        result = doc.otsl_table("<fcel><b>Raw & literal</b><lcel><nl><ucel><xcel><nl>")
        self.assertEqual(result["cells"][0]["rowspan"], 2)
        self.assertEqual(result["cells"][0]["colspan"], 2)
        self.assertIn("&lt;b&gt;Raw &amp; literal&lt;/b&gt;", result["html"])
        self.assertNotIn("<b>", result["html"])
        self.assertEqual(result["text"], "<b>Raw & literal</b>")

    def test_malformed_spans_rows_and_markup_are_errors(self):
        bad = ["", "|a|b|", "<table><tr><td>A</td></tr></table>", "<lcel><nl>", "<ucel><nl>", "<xcel><nl>",
               "<fcel>A<lcel><nl><ucel><fcel>B<nl>",
               "<fcel>A<fcel>B<nl><fcel>C<xcel><nl>",
               "<fcel>A<lcel><nl><ucel><lcel><nl>",
               "<fcel>A<lcel><nl><ucel><ucel><nl>",
               "<fcel>A<nl><fcel>B<fcel>C<nl>", "<ecel>oops<nl>",
               "<fcel>A<lcel>oops<nl>", "<fcel>A<nl><nl>", "<fcel>A<nl>suffix",
               "<fcel>A<nl/>", "<fcel>A<ucel", "<fcel>A<zcel>"]
        for source in bad:
            with self.subTest(source=source), self.assertRaises(ValueError):
                doc.otsl_to_html(source)

    def test_bounds_reject_without_truncation(self):
        for source in ("<ecel>" * 257 + "<nl>", ("<ecel><nl>") * 257,
                       "<fcel>" + "x" * (1024 * 1024)):
            with self.assertRaises(ValueError):
                doc.otsl_table(source)


class DocumentControls(unittest.TestCase):
    def test_region_order_tables_formulas_and_truthful_stops(self):
        layout, protocol, capture = records()
        result = doc.assemble(layout, protocol, capture, "5" * 64)
        self.assertEqual([b["label"] for b in result["document"]["blocks"]],
                         ["heading", "table", "equation", "figure"])
        self.assertEqual(result["document"]["blocks"][1]["bbox"], [.01, .15, .9, .5])
        self.assertIn('<td colspan="2">Heading</td>', result["document"]["tables"][0])
        self.assertEqual(result["document"]["formulas"], ["\\frac{x}{2} "])
        self.assertEqual(result["document"]["blocks"][3]["text"], "")
        self.assertEqual(len(result["generation"]), 3)
        self.assertTrue(all(c["status"] == "eos" for c in result["generation"]))
        self.assertFalse(result["adapter"]["native_layout_implemented"])
        self.assertFalse(result["full_document_qualified"])
        self.assertEqual(set(doc.LABELS), set(doc.policy.evidence()["layout"]["id2label"].values()))

    def test_cap_context_and_contradictory_stop_evidence(self):
        layout, protocol, capture = records()
        row = capture["observations"][0]["native"]
        row.update(token_ids=[10, 11, 12, 13], confidences=[.8] * 4, stop_kind=2, stop_id=None)
        result = doc.assemble(layout, protocol, capture, "5" * 64)
        self.assertEqual(result["generation"][0]["status"], "cap")
        row.update(token_ids=[10], confidences=[.8], stop_kind=3)
        result = doc.assemble(layout, protocol, capture, "5" * 64)
        self.assertEqual(result["generation"][0]["status"], "error")
        self.assertEqual(result["generation"][0]["native_stop_reason"], "context_limit")
        for change in ({"stop_kind": 1}, {"stop_kind": 2}, {"cap": True},
                       {"token_ids": [248046, 10], "confidences": [.8, .8]}, {"confidences": []}):
            changed = copy.deepcopy(capture)
            changed["observations"][0]["native"].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                doc.assemble(layout, protocol, changed, "5" * 64)

    def test_missing_reordered_or_extra_regions_and_changed_protocol_reject(self):
        layout, protocol, capture = records()
        variants = [capture["observations"][:-1], list(reversed(capture["observations"])),
                    capture["observations"] + capture["observations"][:1]]
        for items in variants:
            changed = copy.deepcopy(capture); changed["observations"] = items
            with self.assertRaises(ValueError):
                doc.assemble(layout, protocol, changed, "5" * 64)
        for key, value in (("max_regions", 2), ("max_total_new_tokens", 1), ("layout_record_sha256", "0" * 64)):
            changed = copy.deepcopy(protocol); changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                doc.crop_plan(layout, changed)
        changed = copy.deepcopy(layout); changed["detector"]["model"] = "substitute"
        protocol["layout_record_sha256"] = doc.digest(changed)
        with self.assertRaises(ValueError):
            doc.crop_plan(changed, protocol)

    def test_malformed_table_fails_without_losing_input_capture(self):
        layout, protocol, capture = records()
        capture["observations"][1]["native"]["raw_text_utf8_hex"] = "<ucel><nl>".encode().hex()
        before = copy.deepcopy(capture)
        with self.assertRaises(ValueError):
            doc.assemble(layout, protocol, capture, "5" * 64)
        self.assertEqual(capture, before)

    @unittest.skipUnless((ROOT / "tools/ocr_document_eval.py").is_file(), "document evaluator integrated on parent branch")
    def test_integrated_document_schema_and_html_parser(self):
        evaluator = doc.load_module("xiaomi_document_schema", ROOT / "tools/ocr_document_eval.py")
        result = doc.assemble(*records(), "5" * 64)
        evaluator.validate_record(result)
        self.assertTrue(evaluator.completion(result["generation"])["complete"])
        self.assertEqual(evaluator.Table().parse(result["document"]["tables"][0])["columns"], 2)


if __name__ == "__main__":
    unittest.main()
