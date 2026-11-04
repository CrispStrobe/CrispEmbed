"""Source/task/cleanup controls; no frameworks, model execution or old evidence edits."""
import copy
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import ovisocr3_document_contract as audit


class DocumentTaskContract(unittest.TestCase):
    def setUp(self):
        self.contract = audit.strict_json(audit.CONTRACT.read_text())

    def test_exact_source_and_old_failure_are_retained(self):
        result = audit.validate_contract(self.contract)
        self.assertEqual(result["task"]["prompt_bytes"], 512)
        self.assertEqual(result["generation"]["max_new_tokens"], 16384)
        self.assertEqual(result["preserved_plain_ocr"]["two_column_cap"], 1024)
        self.assertTrue(all(value is False for value in result["qualification"].values()))
        self.assertIsNone(result["evaluation"]["frozen_input_gold_manifest"])

    def test_plain_prompt_cannot_be_relabelled_as_markdown(self):
        self.contract["task"]["prompt_sha256"] = self.contract["preserved_plain_ocr"]["prompt_sha256"]
        with self.assertRaisesRegex(ValueError, "task/prompt"):
            audit.validate_contract(self.contract)

    def test_changed_budget_processor_or_checkpoint_fails_closed(self):
        for section, field, value in (("generation", "max_new_tokens", 1024),
                                      ("processor", "min_pixels", 65536),
                                      ("checkpoint", "sha256", "0" * 64)):
            with self.subTest(section=section):
                contract = copy.deepcopy(self.contract)
                contract[section][field] = value
                with self.assertRaises(ValueError):
                    audit.validate_contract(contract)

    def test_source_audit_cannot_grant_runtime_or_quality(self):
        for key in self.contract["qualification"]:
            with self.subTest(qualification=key):
                contract = copy.deepcopy(self.contract)
                contract["qualification"][key] = True
                with self.assertRaisesRegex(ValueError, "cannot grant"):
                    audit.validate_contract(contract)

    def test_old_cap_and_failure_cannot_be_superseded(self):
        for field, value in (("two_column_cap", 16384), ("source_quality_qualified", True), ("superseded", True)):
            with self.subTest(field=field):
                contract = copy.deepcopy(self.contract)
                contract["preserved_plain_ocr"][field] = value
                with self.assertRaisesRegex(ValueError, "preserve old"):
                    audit.validate_contract(contract)

    def test_evaluation_claims_and_guards_cannot_change(self):
        for field, value in (("benchmark_score_claim", True), ("outputs_observed", True),
                             ("completion_rule", "cleaned text is complete"),
                             ("required_capture", []), ("frozen_input_gold_manifest", {})):
            with self.subTest(field=field):
                contract = copy.deepcopy(self.contract)
                contract["evaluation"][field] = value
                with self.assertRaisesRegex(ValueError, "evaluation gates"):
                    audit.validate_contract(contract)

    def test_numeric_qualification_flags_are_not_booleans(self):
        for section, field, value in (("qualification", "publication", 0),
                                      ("task", "enable_thinking", 0),
                                      ("task", "images_per_request", True)):
            with self.subTest(section=section, field=field):
                contract = copy.deepcopy(self.contract)
                contract[section][field] = value
                with self.assertRaises(ValueError):
                    audit.validate_contract(contract)

    def test_ambiguous_and_nonfinite_json_rejected(self):
        for text in ('{"qualification": false, "qualification": true}',
                     '{"temperature": NaN}', '{"temperature": Infinity}',
                     '{"temperature": -Infinity}', '{"temperature": 1e999}'):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    audit.strict_json(text)

    def test_mutated_upstream_cleanup_cannot_execute(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            shutil.copytree(audit.FIXTURE, root)
            path = root / "README.md"
            path.write_bytes(path.read_bytes().replace(b"min_text_len: int = 8000", b"min_text_len: int = 1000"))
            with self.assertRaisesRegex(ValueError, "source bytes changed"):
                audit.reference_cleanup(root)

    def test_raw_and_image_preserving_outputs_survive_default_filter(self):
        raw = '  Heading\n\n<img src="images/bbox_1_2_90_99.jpg" />\n\nTail  '
        result = audit.observe_output(raw, [123, 248046], "stop", forced_eos=False)
        self.assertEqual(result["raw_markdown"], raw)
        self.assertEqual(result["upstream_default_markdown"], "Heading\n\nTail")
        self.assertIn("images/bbox_1_2_90_99.jpg", result["image_preserving_markdown"])
        self.assertTrue(result["generation_complete"])
        self.assertFalse(result["task_qualified"])

    def test_cleanup_cannot_make_a_length_stop_complete(self):
        raw = "Heading\n" + "abcd" * 2100
        result = audit.observe_output(raw, [123] * audit.MAX_NEW_TOKENS, "length", forced_eos=False)
        self.assertEqual(result["raw_markdown"], raw)
        self.assertEqual(result["upstream_default_markdown"], "Heading\nabcd")
        self.assertFalse(result["generation_complete"])
        self.assertFalse(result["task_qualified"])
        self.assertFalse(audit.observe_output(raw, [123], "stop", forced_eos=False)["generation_complete"])

    def test_forced_eos_at_cap_cannot_complete(self):
        ids = [123] * (audit.MAX_NEW_TOKENS - 1) + [248046]
        result = audit.observe_output("truncated document", ids, "stop", forced_eos=True)
        self.assertTrue(result["token_budget_reached"])
        self.assertTrue(result["forced_eos"])
        self.assertFalse(result["generation_complete"])
        for value in (None, 0, "false"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "explicit boolean"):
                    audit.observe_output("text", [123, 248046], "stop", forced_eos=value)

    def test_tokens_after_eos_rejected(self):
        for ids in ([248046, 123], [248044, 248046], [123, 248046, 123, 248046]):
            with self.subTest(ids=ids):
                with self.assertRaisesRegex(ValueError, "after EOS"):
                    audit.observe_output("text", ids, "stop", forced_eos=False)

    def test_malformed_ids_and_finish_reasons_rejected(self):
        for ids, reason in (([], "stop"), ([True], "stop"), ([248320], "stop"),
                            ([248046], "complete"), ([123] * (audit.MAX_NEW_TOKENS + 1), "length")):
            with self.subTest(reason=reason, count=len(ids)):
                with self.assertRaises(ValueError):
                    audit.observe_output("text", ids, reason, forced_eos=False)


if __name__ == "__main__":
    unittest.main()
