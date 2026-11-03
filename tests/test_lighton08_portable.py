"""Controls for a separate current-source replay, using retained real evidence."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("lighton08_portable_test", ROOT / "tools/ci-heavy/lighton08_portable.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class PortableReplayControls(unittest.TestCase):
    def test_three_unchanged_case_preflights_without_model_imports(self):
        for case_id in runner.review.CASES:
            protocol, _, old = runner.preflight(case_id)
            self.assertEqual(protocol["max_new_tokens"], old["specification"]["max_new_tokens"])
        with self.assertRaises(ValueError):
            runner.preflight("unreviewed-document")

    def test_both_portable_flags_and_cpu_backend_required(self):
        lines = {"GGML_NATIVE": "OFF", "CRISPEMBED_NATIVE": "OFF", "GGML_CUDA": "OFF", "GGML_METAL": "OFF"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "CMakeCache.txt"
            def write(values):
                path.write_text("".join(f"{key}:BOOL={value}\n" for key, value in values.items()))
            write(lines)
            self.assertEqual(runner.portable_cache(path), lines)
            for name in lines:
                changed = dict(lines); changed[name] = "ON"; write(changed)
                with self.subTest(name=name), self.assertRaises(ValueError):
                    runner.portable_cache(path)
                changed = dict(lines); changed.pop(name); write(changed)
                with self.subTest(missing=name), self.assertRaises(ValueError):
                    runner.portable_cache(path)

    def test_unrelated_profile_gates_cannot_leak_into_replay(self):
        _, prior, _ = runner.preflight("inventory-table")
        with patch.dict(runner.os.environ, {"CRISPEMBED_XIAOMI_OCR0_INFERENCE": "1", "CRISPEMBED_OVISOCR3_FRONTEND": "1"}):
            env = runner.native_environment(prior)
        self.assertNotIn("CRISPEMBED_XIAOMI_OCR0_INFERENCE", env)
        self.assertNotIn("CRISPEMBED_OVISOCR3_FRONTEND", env)
        self.assertEqual(env["CRISPEMBED_BUILD_MODE"], "portable")
        self.assertEqual(env["CRISPEMBED_QWEN35_QUANT"], "1")

    def arms(self):
        _, _, old = runner.preflight("inventory-table")
        prior = json.loads((ROOT / "tests/results/qwen35-ocr/quality-quant-q8-inventory-table.json").read_text())
        return [{**copy.deepcopy(old), "profile": "f16"},
                {**copy.deepcopy(prior["arms"][1]), "profile": "q8"}], old["reference"]

    def test_old_numerical_success_never_becomes_current_qualification(self):
        arms, reference = self.arms()
        result = runner.summarize("inventory-table", arms, reference)
        self.assertTrue(result["all_selected_precision_bars_met"])
        self.assertFalse(result["current_model_qualified"])
        self.assertFalse(result["publication_qualified"])
        self.assertFalse(result["historical_verdicts_changed"])
        self.assertFalse(result["all_three_case_panel_complete"])

    def test_incomplete_arms_or_topk_vectors_cannot_pass(self):
        arms, reference = self.arms()
        for changed in (arms[:1], arms[::-1], arms + arms[:1]):
            with self.assertRaises(ValueError):
                runner.summarize("inventory-table", changed, reference)
        arm = arms[1]
        row = next(row for row in arm["decode"]["metrics"] if row["stage"].startswith("generated.scores."))
        row["elements"] = 10
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            runner.summarize("inventory-table", arms, reference)

    def test_omitted_source_pin_and_changed_cap_fail_before_execution(self):
        original = json.loads(runner.PROTOCOL.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "protocol.json"
            for change in (lambda p: p["frozen_files_sha256"].pop("tools/ocr_quality.py"),
                           lambda p: p.update(max_new_tokens=2048),
                           lambda p: p.update(historical_verdicts_changed=True)):
                protocol = copy.deepcopy(original); change(protocol); path.write_text(json.dumps(protocol))
                with patch.object(runner, "PROTOCOL", path), self.assertRaises(ValueError):
                    runner.preflight("inventory-table")

    def test_tokens_after_eos_are_rejected_without_changing_frozen_helper(self):
        arms, reference = self.arms()
        arms[1]["decode"]["native_ids"][0] = reference["eos_ids"][0]
        with self.assertRaisesRegex(ValueError, "after EOS"):
            runner.summarize("inventory-table", arms, reference)
        with self.assertRaises(ValueError):
            runner.stop_boundary([], [248046])

    def test_shared_filesystem_requires_sum_of_scratch_and_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(runner.shutil, "disk_usage", return_value=SimpleNamespace(free=23 * 1024**3)):
                self.assertTrue(runner.storage_capacity(tmp, tmp)["same_filesystem"])
            with patch.object(runner.shutil, "disk_usage", return_value=SimpleNamespace(free=20 * 1024**3)):
                with self.assertRaisesRegex(ValueError, "shared filesystem"):
                    runner.storage_capacity(tmp, tmp)


if __name__ == "__main__":
    unittest.main()
