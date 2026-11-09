#!/usr/bin/env python3
"""Stdlib routing controls; git and event differences are mocked, never fetched."""

import ast
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import call, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import ci_selection
import lighton4b_ci as ci


class SelectionTests(unittest.TestCase):
    def test_each_reviewed_dependency_selects_reference(self):
        paths = (
            ".github/workflows/controls.yml", "CMakeLists.txt", ".gitmodules", "VERSION",
            "cmake/CrispEmbedC2pa.cmake", "cmake/crispembed-config.cmake.in",
            "third_party/c2pa/include/c2pa.h", "ggml", "ggml/src/ggml-cpu/ops.cpp",
            "src/qwen35_delta.cpp", "src/qwen35_weight_cache.h", "src/qwen35_future.cpp",
            "src/core/gguf_loader.cpp", "src/core/nested/input.h", "src/crispembed_diff.h",
            "include/crispembed.h", "tools/ci_selection.py", "tools/lighton4b_ci.py",
            "tests/test_lighton4b_ci.py", "tools/ci-heavy/lighton4b_gdn_reference.py",
            "tools/ci-heavy/lighton4b_new_reference.py", "tools/lighton4b-gdn-requirements.txt",
            "models/lightonocr3_4b_profile.py", "models/convert-qwen35-ocr-to-gguf.py",
            "tests/test_lighton4b_gdn_reference.py", "tests/test_lightonocr3_4b_export.py",
            "tests/test_lightonocr3_4b_export_controls.py", "tests/test_qwen35_gdn_heads.cpp",
            "tests/fixtures/lightonocr3-4b-source/README.md",
            "tests/fixtures/lightonocr3-4b-source/nested/new.json",
        )
        for path in paths:
            with self.subTest(path=path):
                report = ci.select_paths([path])
                self.assertTrue(report["run_reference"])
                self.assertEqual(report["relevant_paths"], [path])

    def test_proven_unrelated_and_empty_complete_diffs_skip(self):
        for paths in ([], ["README.md"], ["docs/lightonocr3-4b.md", "HISTORY.md"],
                      ["src/glm_ocr.cpp", "tools/ci-heavy/ultrabert_reference.py"],
                      ["tests/fixtures/lightonocr3-4b-source-other/config.json"],
                      ["src/core_extra/helper.h", "include-other/api.h"]):
            with self.subTest(paths=paths):
                self.assertFalse(ci.select_paths(paths)["run_reference"])

    def test_mixed_changes_use_the_union_and_preserve_old_rename_name(self):
        paths = ["README.md", "src/qwen35_linear.cpp", "archive/renamed.txt", "README.md"]
        report = ci.select_paths(paths)
        self.assertTrue(report["run_reference"])
        self.assertEqual(report["paths"], sorted(set(paths)))
        self.assertEqual(report["relevant_paths"], ["src/qwen35_linear.cpp"])

    def test_malformed_changed_path_evidence_fails_closed(self):
        for paths in (None, {}, "README.md", [None], [""], ["../README.md"], ["/README.md"],
                      ["src/../README.md"], ["./README.md"], ["src//x.h"], ["C:/README.md"],
                      ["src\\qwen35_linear.cpp"], ["README.md\0"]):
            with self.subTest(paths=paths):
                self.assertTrue(ci.select_paths(paths)["run_reference"])

    def test_every_declared_native_reference_source_is_selected(self):
        tree = ast.parse((ROOT / "tools/ci-heavy/lighton4b_gdn_reference.py").read_text(encoding="utf-8"))
        declarations = [node.value for node in tree.body if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == "NATIVE_SOURCES"
                                for target in node.targets)]
        self.assertEqual(len(declarations), 1)
        sources = ast.literal_eval(declarations[0])
        self.assertTrue(sources)
        for path in sources:
            with self.subTest(path=path):
                self.assertTrue(ci.select_paths([path])["run_reference"])


class EventTests(unittest.TestCase):
    def test_known_events_delegate_to_the_complete_diff_helper(self):
        repo, event = Path("repo"), {"payload": "unchanged"}
        for name in ("pull_request", "push"):
            with self.subTest(event=name), patch.object(ci, "changed_paths", return_value=["README.md"]) as diff:
                self.assertFalse(ci.select_event(name, event, repo)["run_reference"])
                diff.assert_called_once_with(name, event, repo)

    def test_manual_merge_queue_and_unknown_events_always_run_without_diff(self):
        for name in ("workflow_dispatch", "merge_group", "", "schedule", "pull_request_target", "new_event"):
            with self.subTest(event=name), patch.object(ci, "changed_paths") as diff:
                self.assertTrue(ci.select_event(name, {})["run_reference"])
                diff.assert_not_called()

    def test_every_diff_error_fails_closed(self):
        errors = (KeyError("sha"), TypeError("event"), ValueError("diff"), OSError("git"),
                  subprocess.CalledProcessError(1, ["git", "diff"]),
                  subprocess.TimeoutExpired(["git"], 120), UnicodeError("encoding"), RuntimeError("unexpected"))
        for error in errors:
            with self.subTest(error=type(error).__name__), patch.object(ci, "changed_paths", side_effect=error):
                report = ci.select_event("push", {})
                self.assertTrue(report["run_reference"])
                self.assertIn(type(error).__name__, report["reason"])

    def test_nonobject_events_and_invalid_diff_results_fail_closed(self):
        with patch.object(ci, "changed_paths") as diff:
            for event in (None, [], "payload"):
                self.assertTrue(ci.select_event("push", event)["run_reference"])
            diff.assert_not_called()
        with patch.object(ci, "changed_paths", return_value=None):
            self.assertTrue(ci.select_event("pull_request", {})["run_reference"])

    def test_pr_uses_merge_base_and_retains_both_rename_paths(self):
        repo, base, head = Path("repo"), "a" * 40, "b" * 40
        event = {"pull_request": {"base": {"sha": base}, "head": {"sha": head}}}
        renamed = b"R100\0src/qwen35_linear.cpp\0archive/renamed.txt\0"
        with patch.object(ci_selection, "commit", side_effect=lambda _, value: value), \
                patch.object(ci_selection, "git", side_effect=[b"merge-base\n", renamed]) as git:
            report = ci.select_event("pull_request", event, repo)
        self.assertTrue(report["run_reference"])
        self.assertIn("src/qwen35_linear.cpp", report["paths"])
        self.assertIn("archive/renamed.txt", report["paths"])
        self.assertEqual(git.call_args_list, [
            call(repo, "merge-base", base, head),
            call(repo, "diff", "--no-ext-diff", "--name-status", "-z", "--find-renames", base + "..." + head, "--"),
        ])

    def test_push_uses_entire_before_after_range(self):
        repo, before, after = Path("repo"), "a" * 40, "c" * 40
        event = {"before": before, "after": after}
        changed = b"M\0README.md\0D\0src/qwen35_delta.h\0"
        with patch.object(ci_selection, "commit", side_effect=lambda _, value: value), \
                patch.object(ci_selection, "git", return_value=changed) as git:
            report = ci.select_event("push", event, repo)
        self.assertTrue(report["run_reference"])
        git.assert_called_once_with(repo, "diff", "--no-ext-diff", "--name-status", "-z", "--find-renames",
                                    before + ".." + after, "--")

    def test_truncated_diff_cannot_produce_a_skip(self):
        with patch.object(ci_selection, "commit", side_effect=lambda _, value: value), \
                patch.object(ci_selection, "git", return_value=b"M\0README.md"):
            self.assertTrue(ci.select_event("push", {"before": "a" * 40, "after": "b" * 40})["run_reference"])


class CommandTests(unittest.TestCase):
    def invoke(self, arguments):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = ci.main(arguments)
        return result, stdout.getvalue(), stderr.getvalue()

    def test_explicit_output_is_lowercase_appended_and_ignores_output_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, decoy, event = root / "selected", root / "decoy", root / "event.json"
            output.write_text("another_output=keep\n", encoding="utf-8")
            event.write_text("{}", encoding="utf-8")
            with patch.dict("os.environ", {"GITHUB_OUTPUT": str(decoy)}), \
                    patch.object(ci, "changed_paths", return_value=["README.md"]):
                result, stdout, _ = self.invoke(["--event-name", "push", "--event-path", str(event),
                                                "--github-output", str(output)])
            self.assertEqual(result, 0)
            self.assertFalse(json.loads(stdout)["run_reference"])
            self.assertEqual(output.read_text(), "another_output=keep\nrun_reference=false\n")
            self.assertFalse(decoy.exists())

    def test_default_event_environment_and_explicit_repo_reach_helper(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, event = root / "selected", root / "event.json"
            event.write_text('{"before":"a","after":"b"}', encoding="utf-8")
            with patch.dict("os.environ", {"GITHUB_EVENT_NAME": "push", "GITHUB_EVENT_PATH": str(event)}), \
                    patch.object(ci, "changed_paths", return_value=["src/qwen35_delta.cpp"]) as diff:
                result, _, _ = self.invoke(["--repo", str(root), "--github-output", str(output)])
            self.assertEqual(result, 0)
            diff.assert_called_once_with("push", {"before": "a", "after": "b"}, root)
            self.assertEqual(output.read_text(), "run_reference=true\n")

    def test_missing_or_malformed_event_file_emits_true(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, contents in enumerate((None, "{broken", "[]")):
                with self.subTest(contents=contents):
                    event, output = root / f"event{index}", root / f"output{index}"
                    if contents is not None:
                        event.write_text(contents, encoding="utf-8")
                    result, _, _ = self.invoke(["--event-name", "push", "--event-path", str(event),
                                                "--github-output", str(output)])
                    self.assertEqual(result, 0)
                    self.assertEqual(output.read_text(), "run_reference=true\n")

    def test_missing_event_path_emits_true(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {}, clear=True):
            output = Path(directory) / "selected"
            result, _, _ = self.invoke(["--event-name", "push", "--github-output", str(output)])
            self.assertEqual(result, 0)
            self.assertEqual(output.read_text(), "run_reference=true\n")

    def test_manual_does_not_require_an_event_file(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(ci, "changed_paths") as diff:
            output = Path(directory) / "selected"
            result, _, _ = self.invoke(["--event-name", "workflow_dispatch", "--event-path", "missing.json",
                                        "--github-output", str(output)])
            self.assertEqual(result, 0)
            self.assertEqual(output.read_text(), "run_reference=true\n")
            diff.assert_not_called()

    def test_diff_exception_still_writes_true(self):
        with tempfile.TemporaryDirectory() as directory:
            output, event = Path(directory) / "selected", Path(directory) / "event.json"
            event.write_text("{}", encoding="utf-8")
            with patch.object(ci, "changed_paths", side_effect=RuntimeError("incomplete")):
                result, _, _ = self.invoke(["--event-name", "push", "--event-path", str(event),
                                            "--github-output", str(output)])
            self.assertEqual(result, 0)
            self.assertEqual(output.read_text(), "run_reference=true\n")

    def test_unwritable_explicit_output_fails_the_step(self):
        with tempfile.TemporaryDirectory() as directory:
            result, _, stderr = self.invoke(["--event-name", "workflow_dispatch", "--github-output", directory])
            self.assertEqual(result, 1)
            self.assertIn("::error::Cannot write explicit run_reference output", stderr)

    def test_output_argument_is_required_even_with_environment_output(self):
        with patch.dict("os.environ", {"GITHUB_OUTPUT": "unused"}), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                ci.main(["--event-name", "workflow_dispatch"])
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
