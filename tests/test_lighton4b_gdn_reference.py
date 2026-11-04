"""Stdlib controls for the hosted synthetic GDN evidence contract.

These controls do not import or execute a model framework, build native code,
or generate synthetic references. Framework/operator execution belongs to the
separate hosted component job.
"""

import builtins
import contextlib
import hashlib
import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/ci-heavy/lighton4b_gdn_reference.py"


def no_framework_import(name, *args, **kwargs):
    if name.split(".")[0] in {"torch", "numpy", "gguf", "transformers"}:
        raise AssertionError("Contract control imported a model framework: " + name)
    return ORIGINAL_IMPORT(name, *args, **kwargs)


ORIGINAL_IMPORT = builtins.__import__
with patch("builtins.__import__", side_effect=no_framework_import):
    SPEC = importlib.util.spec_from_file_location("lighton4b_gdn_controls", SCRIPT)
    reference = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(reference)


class LightOnFourBGDNReferenceControls(unittest.TestCase):
    def successful_records(self):
        return [{"lane": lane, "stage": name, "finite": True, "max_abs": 1e-7, "cos_min": 1.0}
                for lane, name, _ in reference.expected_stages()]

    def test_fixed_cases_cover_exact_head_groups_equal_path_and_cache_boundaries(self):
        cases = [reference.checked_case(case) for case in reference.CASES]
        self.assertEqual(len(cases), 4)
        self.assertEqual(len({case["name"] for case in cases}), 4)
        self.assertIn((16, 32, 65, 1), {(c["key_heads"], c["value_heads"], c["prefill"], c["batch"]) for c in cases})
        self.assertIn((16, 32, 1, 1), {(c["key_heads"], c["value_heads"], c["prefill"], c["batch"]) for c in cases})
        self.assertIn((16, 16, 65, 1), {(c["key_heads"], c["value_heads"], c["prefill"], c["batch"]) for c in cases})
        self.assertIn((2, 4, 3, 2), {(c["key_heads"], c["value_heads"], c["prefill"], c["batch"]) for c in cases})
        self.assertEqual((reference.HIDDEN, reference.WIDTH, reference.KERNEL), (16, 128, 4))
        self.assertEqual(reference.PHASES, ("prefill", "cached0", "cached1", "full_prefix"))

    def test_case_dimensions_fail_closed_before_execution(self):
        mutations = (("key_heads", 0), ("value_heads", 0), ("batch", True), ("prefill", 1.5),
                     ("key_heads", 3), ("key_heads", 64), ("value_heads", 64),
                     ("batch", 3), ("prefill", 66), ("name", "../escape"), ("seed", -1))
        for key, value in mutations:
            case = dict(reference.CASES[1], **{key: value})
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                reference.checked_case(case)
        with self.assertRaises(ValueError):
            reference.checked_case(dict(reference.CASES[1], hidden_size=2560))

    def test_native_stage_contract_covers_all_lanes_without_dump_collisions(self):
        rows = reference.expected_stages()
        self.assertEqual(len(rows), 52)
        self.assertEqual(len({path for _, _, path in rows}), 52)
        self.assertEqual({lane: sum(lane == value for value, _, _ in rows)
                          for lane in ("projected", "core", "delta")},
                         {"projected": 32, "core": 12, "delta": 8})
        for lane, name, path in rows:
            with self.subTest(lane=lane, stage=name):
                self.assertIn(name.split(".")[0], reference.PHASES)
                self.assertEqual(path, ("" if lane == "projected" else lane + "/") + name + ".f32")

    def test_missing_duplicate_reordered_or_unknown_native_stages_reject(self):
        rows = self.successful_records()
        self.assertTrue(reference.validate_stage_records(rows))
        for changed in (rows[:-1], rows + [rows[-1]], rows[::-1], [dict(rows[0], stage="unknown")] + rows[1:]):
            with self.subTest(count=len(changed)), self.assertRaisesRegex(ValueError, "native stage"):
                reference.validate_stage_records(changed)

    def test_one_bad_stage_cannot_be_hidden_by_other_successes(self):
        rows = self.successful_records()
        rows[15]["max_abs"] = 0.00201
        self.assertFalse(reference.validate_stage_records(rows))
        rows[15] = {"lane": rows[15]["lane"], "stage": rows[15]["stage"],
                    "finite": False, "error": "Missing native payload", "passed": False}
        self.assertFalse(reference.validate_stage_records(rows))

    def test_native_controls_require_complete_unique_gate_and_state_coverage(self):
        for equal in (False, True):
            names = sorted(reference.expected_controls(equal))
            text = "\n".join("CONTROL name=" + name + " pass=1" for name in names)
            parsed = reference.parse_controls(text, equal)
            self.assertEqual(len(parsed), 31 if equal else 25)
            self.assertTrue(all(item["passed"] for item in parsed))
            for invalid in ("\n".join(text.splitlines()[:-1]), text + "\n" + text.splitlines()[0],
                            text + "\nCONTROL name=extra pass=1", text.replace("pass=1", "pass=true", 1)):
                with self.subTest(equal=equal), self.assertRaises(ValueError):
                    reference.parse_controls(invalid, equal)

    def test_failed_native_control_remains_failed_even_with_complete_coverage(self):
        text = "\n".join("CONTROL name=" + name + " pass=" + ("0" if name == "gate_group_zero" else "1")
                         for name in sorted(reference.expected_controls(False)))
        parsed = reference.parse_controls(text, False)
        self.assertEqual([item["name"] for item in parsed if not item["passed"]], ["gate_group_zero"])

    def test_absolute_error_and_minimum_row_cosine_are_independent_gates(self):
        self.assertTrue(reference.metric_pass({"finite": True, "max_abs": 0.002, "cos_min": 0.99999}))
        self.assertFalse(reference.metric_pass({"finite": True, "max_abs": 0.00201, "cos_min": 1.0}))
        self.assertFalse(reference.metric_pass({"finite": True, "max_abs": 1e-9, "cos_min": 0.999989}))

    def test_nonfinite_negative_and_untyped_metrics_fail(self):
        base = {"finite": True, "max_abs": 0.0, "cos_min": 1.0}
        for key, value in (("finite", 1), ("finite", False), ("max_abs", float("nan")),
                           ("max_abs", float("inf")), ("max_abs", -1.0), ("max_abs", True),
                           ("cos_min", float("nan")), ("cos_min", float("inf")),
                           ("cos_min", 1.1), ("cos_min", "1.0")):
            with self.subTest(key=key, value=value):
                self.assertFalse(reference.metric_pass(dict(base, **{key: value})))
        self.assertFalse(reference.metric_pass({"finite": True}))

    def test_installed_source_bytes_are_checked_independently_of_version(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "module.py"
            source.write_bytes(b"# exact original source\n")
            pins = {"module.py": hashlib.sha256(source.read_bytes()).hexdigest()}
            accepted = reference.verify_source_files(root, pins)
            self.assertEqual(accepted["transformers/module.py"]["bytes"], source.stat().st_size)
            source.write_bytes(b"# edited original source\n")
            with self.assertRaisesRegex(ValueError, "original source differs"):
                reference.verify_source_files(root, pins)

    def test_missing_or_escaping_source_paths_reject(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "package"
            root.mkdir()
            outside = Path(directory) / "outside.py"
            outside.write_bytes(b"not a package module")
            digest = hashlib.sha256(outside.read_bytes()).hexdigest()
            for name in ("missing.py", "../outside.py", str(outside)):
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Missing or escaping"):
                    reference.verify_source_files(root, {name: digest})

    def test_wrong_package_version_rejects_before_framework_import(self):
        versions = dict(reference.PACKAGES, transformers="5.5.5")
        with patch.object(reference.importlib.metadata, "version", side_effect=versions.__getitem__), \
                patch("builtins.__import__", side_effect=no_framework_import), \
                self.assertRaisesRegex(ValueError, "package versions differ"):
            reference.load_original()

    def test_matching_package_version_does_not_bypass_module_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            module_path = root / "models/qwen3_5/modeling_qwen3_5.py"
            module_path.parent.mkdir(parents=True)
            module_path.write_bytes(b"# wrong source under the correct version\n")
            fake = types.ModuleType("transformers")
            fake.__file__, fake.__version__ = str(root / "__init__.py"), reference.TRANSFORMERS_VERSION
            with patch.object(reference.importlib.metadata, "version", side_effect=reference.PACKAGES.__getitem__), \
                    patch.dict(sys.modules, transformers=fake), \
                    self.assertRaisesRegex(ValueError, "original source differs"):
                reference.load_original()

    def test_checkpoint_and_product_qualification_remain_false(self):
        report = {name: False for name in reference.FALSE_QUALIFICATIONS}
        report["component_passed"] = True
        self.assertTrue(reference.report_qualification(report))
        for name in reference.FALSE_QUALIFICATIONS:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "cannot qualify"):
                reference.report_qualification(dict(report, **{name: True}))
        missing = dict(report)
        del missing["checkpoint_parity"]
        with self.assertRaises(ValueError):
            reference.report_qualification(missing)

    def test_requirements_match_runtime_package_pins(self):
        text = (ROOT / "tools/lighton4b-gdn-requirements.txt").read_text()
        pins = dict(line.strip().split("==", 1) for line in text.splitlines() if "==" in line)
        self.assertEqual(pins, reference.PACKAGES)
        self.assertIn("https://download.pytorch.org/whl/cpu", text)

    def test_non_hosted_cli_rejects_before_framework_or_filesystem_work(self):
        with tempfile.TemporaryDirectory() as directory:
            output, build = Path(directory) / "out", Path(directory) / "build"
            with patch.dict(os.environ, {"GITHUB_ACTIONS": "false"}), \
                    patch("builtins.__import__", side_effect=no_framework_import), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                reference.main(["--output", str(output), "--build-dir", str(build)])
            self.assertEqual(caught.exception.code, 2)
            self.assertFalse(output.exists())
            self.assertFalse(build.exists())

    def test_hosted_cli_requires_offline_execution_and_fresh_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            output, build = Path(directory) / "out", Path(directory) / "build"
            for offline in ("", "0"):
                with patch.dict(os.environ, {"GITHUB_ACTIONS": "true", "HF_HUB_OFFLINE": offline,
                                             "TRANSFORMERS_OFFLINE": "1"}), \
                        contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                    reference.main(["--output", str(output), "--build-dir", str(build)])
                self.assertEqual(caught.exception.code, 2)
                self.assertFalse(output.exists())
            output.mkdir()
            with patch.dict(os.environ, {"GITHUB_ACTIONS": "true", "HF_HUB_OFFLINE": "1",
                                         "TRANSFORMERS_OFFLINE": "1"}), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                reference.main(["--output", str(output), "--build-dir", str(build)])
            self.assertEqual(caught.exception.code, 2)
            self.assertFalse(build.exists())


if __name__ == "__main__":
    unittest.main()
