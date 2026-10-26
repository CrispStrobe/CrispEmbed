#!/usr/bin/env python3
"""Lightweight contract tests: no model/tokenizer/framework downloads or imports."""
import copy
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import josev2b_contract as contract


class ContractTests(unittest.TestCase):
    def test_frozen_panel_covers_types_and_order(self):
        panel = contract.panel()
        self.assertEqual(len(panel["cases"]), 4)
        request = panel["cases"][0]["request"]
        self.assertEqual(list(request["questions"]), ["bezahlt", "status", "sicherheit"])
        self.assertEqual(list(request["questions"]["status"]["criteria"]), ["z_offen", "a_bezahlt"])
        self.assertEqual({q["type"] for c in panel["cases"] for q in c["request"]["questions"].values()},
                         {"noul", "choice", "score"})

    def test_excluded_inputs_fail_without_normalization(self):
        original = contract.panel()["cases"][0]["request"]
        for change in (
            lambda r: r.update(images=[]),
            lambda r: r.update(state="literal <|im_start|> text"),
            lambda r: r.update(state=float("nan")),
            lambda r: r.update(state=float("inf")),
            lambda r: r.update(state="ä" * 65536),
            lambda r: r.update(questions={}),
            lambda r: r["questions"].update({"": {"type": "noul"}}),
            lambda r: r["questions"]["status"].update(criteria=["bad"]),
            lambda r: r["questions"]["sicherheit"].update(criteria={"0": "bad"}),
            lambda r: r["questions"]["bezahlt"].update(criteria={"other": "bad"}),
        ):
            with self.subTest(change=change):
                request = copy.deepcopy(original)
                change(request)
                with self.assertRaises(ValueError):
                    contract.validate_request(request)
        self.assertEqual(original, contract.panel()["cases"][0]["request"])

    def test_snapshot_payload_missing_or_altered_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "snapshot bytes differ"):
                contract.verify_snapshot(Path(tmp))
            (Path(tmp) / "LICENSE").write_text("Apache tag alone is insufficient", encoding="utf-8")
            with self.assertRaises(ValueError):
                contract.verify_snapshot(Path(tmp), source_only=True)

    def test_all_expected_assets_and_separate_head_are_pinned(self):
        cfg = contract.protocol()
        files = cfg["files"]
        self.assertEqual(len(files), 14)
        for filename in ("joint_schema_model.py", "joint_head.safetensors", "tokenizer.json",
                         "model.safetensors-00001-of-00001.safetensors", "LICENSE"):
            self.assertIn(filename, files)
        self.assertEqual(files["joint_schema_model.py"]["sha256"], cfg["upstream_inspection"]["cloudflare_code_sha256"])
        self.assertEqual(files["LICENSE"]["sha256"], cfg["upstream_inspection"]["upstream_license_sha256"])
        self.assertFalse(any(cfg["qualification"].values()))

    def test_help_does_not_import_frameworks(self):
        result = subprocess.run([sys.executable, str(ROOT / "tools/dump_josev2b_reference.py"), "--help"],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--reviewed-contract-sha256", result.stdout)

    def test_wrong_package_profile_rejects_without_framework_import(self):
        with patch.object(contract.importlib.metadata, "version", return_value="0.0.0"):
            with self.assertRaisesRegex(ValueError, "package profile differs"):
                contract.environment_versions()

    def test_unsupported_gpu_cannot_reach_payload(self):
        with self.assertRaisesRegex(ValueError, "no retry"):
            contract.require_cuda_capability((6, 0))
        with self.assertRaisesRegex(ValueError, "no retry"):
            contract.require_cuda_capability((7, 0))
        contract.require_cuda_capability((7, 5))

    def test_question_option_bounds_and_literal_state(self):
        request = {"model": contract.protocol()["repository"], "state": " \nä 😀\t ",
                   "questions": {f"q{i}": {"type": "score", "criteria": [0, 1, 2]} for i in range(8)}}
        original = copy.deepcopy(request)
        contract.validate_request(request)
        self.assertEqual(request, original)
        request["questions"]["q0"]["criteria"].append(3)
        with self.assertRaisesRegex(ValueError, "too many options"):
            contract.validate_request(request)
        request = copy.deepcopy(original)
        request["questions"]["q8"] = {"type": "noul"}
        with self.assertRaisesRegex(ValueError, "1..8 questions"):
            contract.validate_request(request)

    def test_unpinned_source_cannot_execute_in_inspector(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.py"
            source.write_text('raise RuntimeError("must not execute")\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact pinned"):
                contract.inspect_serialization(source)

    @unittest.skipUnless(os.environ.get("JOSEV_SOURCE_DIRECTORY"), "optional already-fetched pinned source")
    def test_real_pinned_serializer_with_byte_tokenizer(self):
        directory = Path(os.environ["JOSEV_SOURCE_DIRECTORY"])
        contract.verify_snapshot(directory, source_only=True)
        result = contract.inspect_serialization(directory / "joint_schema_model.py")
        self.assertEqual(result["serializer_controls"], "PASS")


if __name__ == "__main__":
    unittest.main()
