#!/usr/bin/env python3
"""Synthetic supplemental receipt controls; never accesses network or models."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / "tools/ci-heavy/validate_lighton3_eps_first.py"
SPEC = importlib.util.spec_from_file_location("validate_eps_first", PATH)
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


def receipt():
    rows = []
    for precision in ("f32", "f16"):
        for gate in ("0", "1"):
            rows.append({"case": "form-grounding", "precision": precision, "gate": gate,
                         "gate0_byte_controls": {arm: {"all_bytes_exact": True, "tensor_files": 0 if arm == "plain" else 1,
                                                                  "tensors": [] if arm == "plain" else [{"name": "stage.f32", "bytes": 4, "sha256": "a" * 64}]}
                                                 for arm in ("forced", "autonomous", "plain")},
                         "first_norm_improved": True,
                         "comparison": {"owner_bar_pass": True, "decoded": {"ids_exact": True, "text_exact": True}}})
    return {"source": validator.FIRST_SOURCE, "phase": "first", "completed": True, "pass": True,
            "native_request_processes": 12, "original_forward_calls": 0, "rows": rows}


class ReceiptControls(unittest.TestCase):
    def check_receipt(self, first):
        validator.validate(first, {"id": validator.FIRST_RUN, "head_sha": validator.FIRST_SOURCE}, validator.FIRST_RUN)

    def test_first_phase_is_noop(self):
        with patch("sys.argv", ["validator", "--phase", "first"]), \
                patch.object(validator.importlib.util, "spec_from_file_location", side_effect=AssertionError("worker imported")), \
                patch.object(validator.tempfile, "TemporaryDirectory", side_effect=AssertionError("scratch created")):
            self.assertEqual(validator.main(), 0)

    def test_complete_receipt(self):
        self.check_receipt(receipt())

    def test_source_and_run_binding(self):
        for run_id, producer_id, source, producer_source in (
                (1, validator.FIRST_RUN, validator.FIRST_SOURCE, validator.FIRST_SOURCE),
                (validator.FIRST_RUN, 1, validator.FIRST_SOURCE, validator.FIRST_SOURCE),
                (validator.FIRST_RUN, validator.FIRST_RUN, "a" * 40, "a" * 40),
                (validator.FIRST_RUN, validator.FIRST_RUN, validator.FIRST_SOURCE, "a" * 40)):
            with self.subTest(values=(run_id, producer_id, source, producer_source)):
                first = receipt(); first["source"] = source
                with self.assertRaises(ValueError):
                    validator.validate(first, {"id": producer_id, "head_sha": producer_source}, run_id)

    def test_flags_and_counters(self):
        for key, values in {"phase": ["full", None], "completed": [False, 1, None], "pass": [False, 1, None],
                            "native_request_processes": [11, 13, True, None],
                            "original_forward_calls": [1, False, None]}.items():
            for value in values:
                with self.subTest(key=key, value=value):
                    first = receipt(); first[key] = value
                    with self.assertRaises(ValueError): self.check_receipt(first)

    def test_coverage(self):
        variants = []
        first = receipt(); first["rows"].pop(); variants.append(first)
        first = receipt(); first["rows"][1] = copy.deepcopy(first["rows"][0]); variants.append(first)
        for key, value in (("case", "fox-grounding"), ("precision", "q8"), ("gate", 1), ("case", [])):
            first = receipt(); first["rows"][0][key] = value; variants.append(first)
        for first in variants:
            with self.subTest(first=first), self.assertRaises(ValueError): self.check_receipt(first)

    def test_gate0_controls(self):
        for arm in ("forced", "autonomous", "plain"):
            for field, value in (("all_bytes_exact", False), ("all_bytes_exact", 1),
                                 ("tensor_files", True), ("tensor_files", -1), ("tensor_files", None),
                                 ("tensor_files", 1 if arm == "plain" else 0)):
                with self.subTest(arm=arm, field=field, value=value):
                    first = receipt(); first["rows"][0]["gate0_byte_controls"][arm][field] = value
                    with self.assertRaises(ValueError): self.check_receipt(first)
            first = receipt(); del first["rows"][0]["gate0_byte_controls"][arm]
            with self.assertRaises(ValueError): self.check_receipt(first)

    def test_gate0_inventory(self):
        for field, value in (("name", "other"), ("bytes", 0), ("bytes", True), ("bytes", 3), ("sha256", "bad")):
            first = receipt(); first["rows"][0]["gate0_byte_controls"]["forced"]["tensors"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError): self.check_receipt(first)
        first = receipt(); first["rows"][0]["gate0_byte_controls"]["forced"]["tensors"] = []
        with self.assertRaises(ValueError): self.check_receipt(first)

    def test_gate1_requirements(self):
        for precision_index in (1, 3):
            for path in (("first_norm_improved",), ("comparison", "owner_bar_pass"),
                         ("comparison", "decoded", "ids_exact"), ("comparison", "decoded", "text_exact")):
                for value in (False, 1, None):
                    with self.subTest(index=precision_index, path=path, value=value):
                        first = receipt(); target = first["rows"][precision_index]
                        for key in path[:-1]: target = target[key]
                        target[path[-1]] = value
                        with self.assertRaises(ValueError): self.check_receipt(first)


if __name__ == "__main__":
    unittest.main()
