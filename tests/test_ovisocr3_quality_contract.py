"""Pure OvisOCR3 plain-text panel and completion checks; no model outputs."""

import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ovisocr3_quality_contract", ROOT / "tools/ci-heavy/ovisocr3_quality_contract.py")
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


class PlainOCRContract(unittest.TestCase):
    def test_frozen_eight_plus_three_case_source_and_license(self):
        panel = policy.load_panel()
        self.assertEqual([row["split"] for row in panel["cases"]],
                         ["development"] * 4 + ["heldout"] * 4 + ["realistic-evaluation"] * 3)
        self.assertEqual({row["license"] for row in panel["cases"]}, {"CC0-1.0", "MIT"})
        self.assertEqual(panel["task"], "plain_ocr")
        self.assertEqual(panel["policy"], "ovisocr3-plain-raw-v1")

    def test_plain_raw_edits_and_eos_at_cap(self):
        eos = {"raw_text": "Hello", "token_ids": [99, 248046], "stop_id": 248046, "stop_kind": "eos"}
        score = policy.score_case("Hello", eos, 2)
        self.assertTrue(score["complete"])
        self.assertTrue(score["cap_hit"])
        self.assertEqual(score["raw_cer"], 0)
        self.assertEqual(score["raw_wer"], 0)
        changed = dict(eos, raw_text="<table border=1>Hello</table>")
        self.assertGreater(policy.score_case("Hello", changed, 2)["raw_cer"], 1)
        limited = dict(eos, token_ids=[99, 100], stop_id=None, stop_kind="length")
        self.assertFalse(policy.score_case("Hello", limited, 2)["complete"])
        for invalid in (dict(eos, stop_id=None, stop_kind="length"),
                        dict(eos, token_ids=[99, 100, 248046]),
                        dict(eos, stop_id=248044),
                        dict(eos, stop_kind="length")):
            with self.assertRaises(ValueError):
                policy.score_case("Hello", invalid, 2)

    def test_complete_panel_and_paired_identity(self):
        panel = policy.load_panel()
        cases, rows = [], {}
        for spec in panel["cases"]:
            case = {"id": spec["id"], "split": spec["split"], "cap": spec["cap"],
                    "rgb_sha256": spec.get("rgb_sha256", "a" * 64),
                    "png_sha256": spec.get("png_sha256", "b" * 64)}
            gold = spec["gold"] if "gold" in spec else json.loads((ROOT / spec["gold_file"]).read_text())["text"]
            cases.append(case)
            rows[spec["id"]] = {"rgb_sha256": case["rgb_sha256"], "png_sha256": case["png_sha256"],
                                "cap": case["cap"], "raw_text": gold,
                                "token_ids": [1, 248046], "stop_id": 248046, "stop_kind": "eos"}
        scored = policy.score_panel(cases, rows)
        self.assertTrue(scored["all_complete"])
        self.assertEqual(scored["total_character_edits"], 0)
        self.assertTrue(policy.paired_screen(rows, rows)["all_exact"])
        changed = dict(rows)
        changed[panel["cases"][0]["id"]] = dict(changed[panel["cases"][0]["id"]], raw_text="wrong")
        self.assertFalse(policy.paired_screen(rows, changed)["all_exact"])
        with self.assertRaises(ValueError):
            policy.score_panel(cases, {key: value for key, value in rows.items() if key != panel["cases"][0]["id"]})
        with self.assertRaises(ValueError):
            policy.paired_screen(rows, {key: value for key, value in changed.items() if key != panel["cases"][0]["id"]})


if __name__ == "__main__":
    unittest.main()
