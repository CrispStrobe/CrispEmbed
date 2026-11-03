#!/usr/bin/env python3
"""Model-free regression tests for CI routing and its required result checks."""

import importlib.util
import contextlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ci_selection", ROOT / "tools/ci_selection.py")
ci = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ci)


class RoutingTests(unittest.TestCase):
    def lanes(self, *paths):
        return {key for key, selected in ci.selection(list(paths))["lanes"].items() if selected}

    def test_markdown_needs_no_compilers(self):
        self.assertEqual(self.lanes("README.md", "docs/inpainting-cli.md", "crispembed/README.md"), set())

    def test_inpaint_python_changes_do_not_compile_native_code(self):
        for path in (
            "tests/test_prepare_migan_inpaint.py", "tools/migan512_reference_contract.py",
            "tools/prepare_migan_inpaint.py", "models/convert-migan-to-gguf.py",
            "tests/test_inpaint_video_quality.py", "tests/test_inpaint_engine_live_controls.py",
        ):
            with self.subTest(path=path):
                self.assertEqual(self.lanes(path), {"inpaint_python"})

    def test_contracts_and_consumed_receipts_are_not_ignored(self):
        for path in (
            "docs/inpaint-publication-v1.json", "docs/latent-inpaint-source-contract.json",
            "docs/migan512-reference-contract.json", "tests/results/migan/original-reference.json",
        ):
            with self.subTest(path=path):
                self.assertEqual(self.lanes(path), {"inpaint_python"})

    def test_unknown_docs_json_and_fixtures_fail_closed(self):
        for path in ("docs/new-contract.json", "tests/fixtures/new.json", "tests/results/new/receipt.json"):
            with self.subTest(path=path):
                self.assertEqual(self.lanes(path), set(ci.LANES))

    def test_temporal_source_gap_is_covered(self):
        expected = {"native", "inpaint_python", "inpaint_native"}
        for path in ("src/inpaint_temporal.cpp", "src/inpaint_temporal.h", "examples/cli/inpaint_cli.cpp"):
            with self.subTest(path=path):
                self.assertEqual(self.lanes(path), expected)

    def test_native_contract_changes_use_all_three_inpaint_platforms(self):
        for path in ("tests/test_inpaint_temporal.cpp", "tests/inpaint_cli_driver.cpp", "tests/test_inpaint_cli.py", "tests/inpaint_video_quality.py"):
            with self.subTest(path=path):
                self.assertEqual(self.lanes(path), {"inpaint_python", "inpaint_native"})

    def test_shared_dependencies_and_unknown_code_run_full(self):
        for path in (
            "CMakeLists.txt", "cmake/FindThreads.cmake", "ggml", "ggml/src/ggml.c",
            ".gitmodules", "VERSION", "src/core/image_out.cpp", "src/crispembed.cpp",
            "src/new-engine.cpp", "tests/test_unknown.py", "unrecognized-file",
        ):
            with self.subTest(path=path):
                self.assertEqual(self.lanes(path), set(ci.LANES))

    def test_selector_or_cache_changes_exercise_every_lane(self):
        for path in ("tools/ci_selection.py", "tests/test_ci_selection.py", ".github/actions/ci-select/action.yml", ".github/actions/compiler-cache/action.yml"):
            with self.subTest(path=path):
                self.assertEqual(self.lanes(path), set(ci.LANES))

    def test_own_workflow_changes_trigger_their_complete_proof(self):
        self.assertEqual(self.lanes(".github/workflows/build.yml"), {"native", "native_tests", "mobile", "sibling", "bundle"})
        self.assertEqual(self.lanes(".github/workflows/inpaint-cli.yml"), {"inpaint_python", "inpaint_native"})
        self.assertEqual(self.lanes(".github/workflows/rust-crates.yml"), {"rust"})
        self.assertEqual(self.lanes(".github/workflows/build-wasm.yml"), {"wasm_ocr"})
        self.assertEqual(self.lanes(".github/workflows/build-wasm-embed.yml"), {"wasm_embed"})

    def test_wasm_and_rust_changes_do_not_start_unrelated_builds(self):
        self.assertEqual(self.lanes("wasm/crispembed-ocr.js"), {"wasm_ocr"})
        self.assertEqual(self.lanes("tests/wasm-browser/e2e.test.js"), {"wasm_ocr"})
        self.assertEqual(self.lanes("wasm/embed_wrapper.c"), {"wasm_embed"})
        self.assertEqual(self.lanes("build-embed-wasm.sh"), {"wasm_embed"})
        self.assertEqual(self.lanes("crispembed/src/lib.rs", "crispembed-sys/build.rs", "Cargo.lock"), {"rust"})

    def test_unreviewed_wasm_source_does_not_skip_embedding_build(self):
        for path in ("wasm/embed_helpers.h", "wasm/new-wrapper.js", "wasm/crispembed-ocr-wasm/src/lib.rs"):
            with self.subTest(path=path):
                self.assertEqual(self.lanes(path), set(ci.LANES))

    def test_mixed_changes_take_union(self):
        self.assertEqual(self.lanes("src/inpaint_temporal.cpp", "wasm/ocr_wrapper.c", "README.md"), {"native", "inpaint_python", "inpaint_native", "wasm_ocr"})

    def test_manual_and_missing_events_run_full(self):
        for event_name, event in (("workflow_dispatch", {}), ("merge_group", {}), ("push", {}), ("pull_request", {}), ("future_event", {})):
            with self.subTest(event=event_name):
                self.assertTrue(all(ci.select_event(event_name, event)["lanes"].values()))

    def test_incomplete_detection_cannot_pass_as_no_changes(self):
        for error in (ValueError("missing parent"), subprocess.TimeoutExpired("git", 120), OSError("unavailable")):
            with self.subTest(error=type(error).__name__), patch.object(ci, "changed_paths", side_effect=error):
                report = ci.select_event("pull_request", {})
                self.assertTrue(report["full"])
                self.assertIn("incomplete", report["reason"])
                self.assertTrue(all(report["lanes"].values()))

    def test_zero_sha_and_invalid_refs_cannot_be_git_options(self):
        for value in ("0" * 40, "--upload-pack=evil", "refs/heads/main", "", None):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    ci.commit(ROOT, value)

    def test_nul_delimited_renames_deletes_and_unusual_paths(self):
        data = b"R100\0src/inpaint_temporal.cpp\0notes renamed.md\0D\0docs/inpaint-publication-v1.json\0M\0a\tb\nc.cpp\0"
        paths = ci.parse_name_status(data)
        self.assertEqual(paths, ["src/inpaint_temporal.cpp", "notes renamed.md", "docs/inpaint-publication-v1.json", "a\tb\nc.cpp"])
        for bad in (b"M\0truncated", b"R100\0only-one\0", b"Q\0file\0", b"M\0\0", b"M\0\xff\0"):
            with self.subTest(data=bad), self.assertRaises(ValueError):
                ci.parse_name_status(bad)

    def test_outputs_are_explicit_lowercase_booleans(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            summary = Path(directory) / "summary"
            with patch.dict(os.environ, {"GITHUB_OUTPUT": str(output), "GITHUB_STEP_SUMMARY": str(summary)}), contextlib.redirect_stdout(io.StringIO()):
                ci.emit(ci.selection(["README.md"]))
            fields = dict(line.split("=", 1) for line in output.read_text().splitlines())
            self.assertEqual(set(fields), set(ci.LANES) | {"full"})
            self.assertEqual(set(fields.values()), {"false"})
            self.assertIn("CI selection", summary.read_text())


class GitDiffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        self.git("init", "-q")
        self.git("config", "user.email", "ci-test@example.invalid")
        self.git("config", "user.name", "CI routing test")
        self.base = self.commit_file("README.md", "base\n")

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.repo), *args], stderr=subprocess.PIPE).decode().strip()

    def commit_file(self, path, content):
        file = self.repo / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(content)
        self.git("add", "--", path)
        self.git("commit", "-qm", "fixture change")
        return self.git("rev-parse", "HEAD")

    def test_pr_uses_cumulative_diff_even_when_last_commit_is_docs(self):
        self.commit_file("src/inpaint_temporal.cpp", "int temporal;\n")
        head = self.commit_file("docs/note.md", "later docs\n")
        event = {"pull_request": {"base": {"sha": self.base}, "head": {"sha": head}}}
        report = ci.select_event("pull_request", event, self.repo)
        self.assertFalse(report["full"])
        self.assertTrue(report["lanes"]["inpaint_native"])
        self.assertEqual(set(report["paths"]), {"src/inpaint_temporal.cpp", "docs/note.md"})

    def test_pr_excludes_unrelated_changes_on_advanced_base_branch(self):
        head = self.commit_file("docs/note.md", "feature\n")
        self.git("checkout", "-q", "--detach", self.base)
        advanced = self.commit_file("src/unrelated.cpp", "int other;\n")
        event = {"pull_request": {"base": {"sha": advanced}, "head": {"sha": head}}}
        report = ci.select_event("pull_request", event, self.repo)
        self.assertEqual(report["paths"], ["docs/note.md"])
        self.assertFalse(any(report["lanes"].values()))

    def test_push_uses_before_and_after_for_all_commits(self):
        self.commit_file("src/inpaint_temporal.cpp", "int temporal;\n")
        head = self.commit_file("README.md", "docs-only tip\n")
        report = ci.select_event("push", {"before": self.base, "after": head}, self.repo)
        self.assertFalse(report["full"])
        self.assertTrue(report["lanes"]["inpaint_native"])

    def test_rename_away_from_native_and_deleted_contract_keep_old_dependency(self):
        self.commit_file("src/inpaint_temporal.cpp", "int temporal;\n")
        before = self.commit_file("docs/inpaint-publication-v1.json", "{}\n")
        self.git("mv", "src/inpaint_temporal.cpp", "archived.md")
        self.git("rm", "docs/inpaint-publication-v1.json")
        self.git("commit", "-qm", "rename and deletion")
        head = self.git("rev-parse", "HEAD")
        report = ci.select_event("push", {"before": before, "after": head}, self.repo)
        self.assertFalse(report["full"])
        self.assertTrue(report["lanes"]["inpaint_native"])
        self.assertIn("src/inpaint_temporal.cpp", report["paths"])
        self.assertIn("docs/inpaint-publication-v1.json", report["paths"])


class AggregateGateTests(unittest.TestCase):
    def needs(self, selected=True, result=None):
        outputs = {lane: str(selected).lower() for lane in ci.LANES}
        return {
            "changes": {"result": "success", "outputs": outputs},
            "build": {"result": result or ("success" if selected else "skipped")},
        }

    def test_selected_success_and_intentional_skip_pass(self):
        for selected in (True, False):
            self.assertEqual(ci.check_needs(self.needs(selected), "changes", {"build": "native"}), [])

    def test_selected_failure_cancel_and_unexpected_skip_fail(self):
        for result in ("failure", "cancelled", "skipped", "unknown"):
            with self.subTest(result=result):
                self.assertTrue(ci.check_needs(self.needs(True, result), "changes", {"build": "native"}))

    def test_failed_or_missing_selector_and_lost_outputs_fail(self):
        for state in ("failure", "cancelled", "skipped"):
            needs = self.needs(False)
            needs["changes"]["result"] = state
            self.assertTrue(ci.check_needs(needs, "changes", {"build": "native"}))
        needs = self.needs(False)
        del needs["changes"]["outputs"]["native"]
        self.assertTrue(ci.check_needs(needs, "changes", {"build": "native"}))
        self.assertTrue(ci.check_needs({}, "changes", {"build": "native"}))

    def test_unchecked_dependencies_and_failure_in_unselected_job_fail(self):
        needs = self.needs(False)
        needs["forgotten-job"] = {"result": "failure"}
        self.assertTrue(ci.check_needs(needs, "changes", {"build": "native"}))
        for state in ("failure", "cancelled"):
            self.assertTrue(ci.check_needs(self.needs(False, state), "changes", {"build": "native"}))


class NativeCacheTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("ci_cpu_signature", ROOT / "tools/ci_cpu_signature.py")
        self.cpu = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.cpu)
        self.record = "\n".join((
            "processor : 0", "vendor_id : GenuineIntel", "cpu family : 6",
            "model : 85", "stepping : 7", "cpu MHz : 2400.000",
            "flags : fpu sse sse2 avx avx2", "bogomips : 4800.00",
        ))

    def test_signature_ignores_frequency_processor_index_and_flag_order(self):
        changed = self.record.replace("processor : 0", "processor : 3").replace("2400.000", "3200.000").replace("4800.00", "6400.00").replace("fpu sse sse2 avx avx2", "avx2 fpu avx sse2 sse")
        expected = self.cpu.cpu_signature(self.record)
        self.assertIsNotNone(expected)
        self.assertEqual(expected, self.cpu.cpu_signature(changed))
        self.assertEqual(expected, self.cpu.cpu_signature(self.record + "\n\n" + changed))

    def test_model_and_instruction_set_changes_isolate_caches(self):
        signature = self.cpu.cpu_signature(self.record)
        for changed in (self.record.replace("model : 85", "model : 143"), self.record.replace("avx2", "avx2 avx512f"), self.record.replace("stepping : 7", "stepping : 8")):
            with self.subTest(cpu=changed):
                self.assertNotEqual(signature, self.cpu.cpu_signature(changed))

    def test_missing_unknown_or_heterogeneous_cpu_disables_cache(self):
        for invalid in ("", "processor : 0\nCPU part : 0xd03", self.record.replace("flags : fpu sse sse2 avx avx2", "flags :"), self.record + "\n\n" + self.record.replace("avx2", "avx512f")):
            with self.subTest(cpu=invalid):
                self.assertIsNone(self.cpu.cpu_signature(invalid))
        with patch.object(self.cpu.Path, "read_text", side_effect=OSError("unavailable")):
            self.assertEqual(self.cpu.main(), 1)


class WorkflowInvariantTests(unittest.TestCase):
    def test_targeted_desktop_build_covers_all_installed_binaries(self):
        # The later cmake --install step still installs every root target.
        # Omitting e.g. crispembed-server makes the filtered build fail late.
        cmake = (ROOT / "CMakeLists.txt").read_text()
        installed = set(re.findall(r"install\(TARGETS\s+(crispembed[^\s)]+)", cmake))
        workflow = (ROOT / ".github/workflows/build.yml").read_text()
        targeted = workflow.split("- name: Build delivered native targets for isolated CLI changes", 1)[1].split("- name:", 1)[0]
        for target in installed:
            with self.subTest(target=target):
                self.assertIn(target, targeted.split())

    def test_main_docs_push_cannot_cancel_an_earlier_native_build(self):
        for filename in ("build.yml", "build-wasm.yml", "build-wasm-embed.yml", "rust-crates.yml", "inpaint-cli.yml"):
            with self.subTest(workflow=filename):
                workflow = (ROOT / ".github/workflows" / filename).read_text()
                group = next(line for line in workflow.splitlines() if line.startswith("  group:"))
                self.assertIn("github.run_id", group)
                self.assertIn("refs/heads/main", group)
                self.assertIn("workflow_dispatch", group)
                self.assertNotIn("paths-ignore:", workflow)
                self.assertIn("    if: always()", workflow)

    def test_cli_only_targets_are_not_mobile_or_wasm_library_dependencies(self):
        # This protects the small routing exception from future CMake rewiring.
        # A source moved into a shared library needs a routing-map update even
        # though the CMake edit itself already takes the full fallback.
        cmake = (ROOT / "CMakeLists.txt").read_text()
        graph = {}
        for target, dependencies in re.findall(r"target_link_libraries\(\s*([^\s)]+)\s+([^)]*)\)", cmake):
            graph.setdefault(target, set()).update(dependencies.split())
        cli_targets = {"crispembed-inpaint", "crispembed-inpaint-temporal", "crispembed-migan", "crispembed-latent-inpaint", "crispembed-inpaint-graph", "crispembed-inpaint-weights"}
        for target in ("crispembed", "crispembed-shared", "crispembed-wasm", "crispembed-embed-wasm"):
            seen, queue = set(), [target]
            while queue:
                child = queue.pop()
                if child not in seen:
                    seen.add(child)
                    queue.extend(graph.get(child, ()))
            with self.subTest(target=target):
                self.assertFalse(seen & cli_targets, "update CLI-only dependency routing")
        library_sources = cmake.split("# CLI binaries (skip on iOS", 1)[0]
        self.assertNotRegex(library_sources, r"src/(?:inpaint|migan|latent_inpaint)[^\s)]*\.(?:cpp|c)")

    def test_portable_cpu_guard_sees_multiconfig_compiler_flags(self):
        spec = importlib.util.spec_from_file_location("cpu_baseline", ROOT / "scripts/check-cpu-baseline.py")
        baseline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline)
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory)
            (build / "CMakeFiles").mkdir()
            (build / "build.ninja").write_text("include CMakeFiles/impl-Release.ninja\n")
            flags = build / "CMakeFiles/impl-Release.ninja"
            flags.write_text("FLAGS = /O2 /arch:AVX512\n")
            hits = baseline.scan_generated_flags(build, baseline.BANNED_FLAGS_X86)
            self.assertEqual(hits, ["CMakeFiles/impl-Release.ninja: /arch:AVX512"])
            flags.write_text("FLAGS = /O2 /arch:AVX2\n")
            self.assertEqual(baseline.scan_generated_flags(build, baseline.BANNED_FLAGS_X86), [])


if __name__ == "__main__":
    unittest.main()
