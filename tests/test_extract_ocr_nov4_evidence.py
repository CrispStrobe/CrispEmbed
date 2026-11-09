"""Small offline provenance, archive-security and dispatch-routing controls."""
import copy
import hashlib
import io
import json
from pathlib import Path
import stat
import sys
import tempfile
import time
import unittest
from unittest import mock
import urllib.error
import warnings
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/ci-heavy"))
import extract_ocr_nov4_evidence as evidence


def archive(items, compression=zipfile.ZIP_STORED):
    buffer = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(buffer, "w", compression=compression) as output:
            for name, data in items:
                output.writestr(name, data)
    buffer.seek(0)
    return buffer


def metadata(target):
    run = {
        "id": target["run_id"], "head_sha": evidence.SOURCE, "head_branch": "main",
        "event": "workflow_dispatch", "path": ".github/workflows/heavy-cpu.yml",
        "status": "completed", "conclusion": "failure", "run_attempt": 1,
        "repository": {"full_name": evidence.REPOSITORY},
        "head_repository": {"full_name": evidence.REPOSITORY},
    }
    url = evidence.API + "/actions/artifacts/" + str(target["artifact_id"])
    artifact = {
        "id": target["artifact_id"], "name": "heavy-" + str(target["run_id"]),
        "size_in_bytes": target["bytes"], "digest": "sha256:" + target["sha256"],
        "expired": False, "url": url, "archive_download_url": url + "/zip",
        "workflow_run": {"id": target["run_id"], "head_sha": evidence.SOURCE,
                         "head_branch": "main", "repository_id": 1210669686,
                         "head_repository_id": 1210669686},
    }
    return run, artifact


class SourceControls(unittest.TestCase):
    def test_closed_two_original_archives(self):
        self.assertEqual([(row["run_id"], row["artifact_id"], row["bytes"])
                          for row in evidence.TARGETS],
                         [(41810161887, 13313715962, 745003580),
                          (41810175624, 13312356863, 3459070891)])
        self.assertEqual(evidence.SOURCE, "2f1bbdb11c3612acf7399e31b33ac715027f3184")
        for target in evidence.TARGETS:
            evidence.validate_metadata(target, *metadata(target))

    def test_run_identity_changes_rejected(self):
        target = evidence.TARGETS[0]
        for field, value in (("id", 1), ("head_sha", "0" * 40), ("head_branch", "other"),
                             ("path", ".github/workflows/build.yml"), ("event", "push"),
                             ("conclusion", "success"), ("status", "in_progress"),
                             ("run_attempt", 2), ("repository", {"full_name": "other/repo"}),
                             ("head_repository", {"full_name": "other/repo"})):
            run, artifact = metadata(target)
            run[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                evidence.validate_metadata(target, run, artifact)

    def test_artifact_identity_changes_rejected(self):
        target = evidence.TARGETS[1]
        for field, value in (("id", 1), ("name", "different"), ("size_in_bytes", 1),
                             ("digest", "sha256:" + "0" * 64), ("expired", True),
                             ("url", "https://example.invalid/file"),
                             ("archive_download_url", "https://example.invalid/file")):
            run, artifact = metadata(target)
            artifact[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                evidence.validate_metadata(target, run, artifact)
        for field in ("id", "head_sha", "head_branch", "repository_id", "head_repository_id"):
            run, artifact = metadata(target)
            artifact["workflow_run"][field] = "changed"
            with self.subTest(producer=field), self.assertRaises(ValueError):
                evidence.validate_metadata(target, run, artifact)
        changed = dict(target, artifact_id=1)
        with self.assertRaisesRegex(ValueError, "closed source"):
            evidence.validate_metadata(changed, *metadata(changed))

    def test_historical_guard_and_gold_remain_hash_pinned(self):
        for name, digest in evidence.FROZEN.items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest, name)
        protocol = json.loads((ROOT / "tests/fixtures/lighton08-portable/protocol.json").read_text())
        for name in ("tests/results/qwen35-ocr/quality.json",
                     "tools/ci-heavy/qwen35_quality_quant.py"):
            self.assertEqual(protocol["frozen_files_sha256"][name], evidence.FROZEN[name])

    def test_dispatch_rejects_execution_outside_closed_hosted_lane(self):
        valid = {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": evidence.REPOSITORY,
                 "RUNNER_OS": "Linux", "OCR_EVIDENCE_SCRIPT": evidence.SELECTOR,
                 "OCR_EVIDENCE_ARGS": "", "OCR_EVIDENCE_PIP": "",
                 "OCR_EVIDENCE_RUNNER": "ubuntu-latest",
                 "OCR_EVIDENCE_BUILD_MODE": "portable", "GH_TOKEN": "dummy"}
        evidence.validate_dispatch(valid)
        for key, value in (("GITHUB_ACTIONS", "false"), ("GITHUB_REPOSITORY", "other/repo"),
                           ("RUNNER_OS", "macOS"), ("OCR_EVIDENCE_SCRIPT", "other.py"),
                           ("OCR_EVIDENCE_ARGS", "--inference"), ("OCR_EVIDENCE_PIP", "torch"),
                           ("OCR_EVIDENCE_RUNNER", "macos-14"), ("GH_TOKEN", ""),
                           ("OCR_EVIDENCE_BUILD_MODE", "native")):
            with self.subTest(field=key), self.assertRaises(ValueError):
                evidence.validate_dispatch(dict(valid, **{key: value}))

    def test_current_workflow_routes_only_exact_selector_to_separate_permissions(self):
        workflow = (ROOT / ".github/workflows/heavy-cpu.yml").read_text()
        ordinary, selected = workflow.split("  ocr_evidence:\n", 1)
        # Preserve the complete historical stdlib-only job span; the newer
        # explicitly guarded native-attribution job has its own controls below.
        selected = selected.split("  retained-block21:\n", 1)[0]
        self.assertIn("if: inputs.script != '" + evidence.SELECTOR + "'", ordinary)
        self.assertIn("if: inputs.script == '" + evidence.SELECTOR + "'", selected)
        self.assertNotIn("actions: read", ordinary)
        self.assertNotIn("GH_TOKEN:", ordinary)
        self.assertIn("      contents: read\n      actions: read\n", selected)
        self.assertIn("persist-credentials: false", selected)
        self.assertIn("submodules: false", selected)
        self.assertIn("timeout-minutes: 25", selected)
        self.assertIn("python tests/test_extract_ocr_nov4_evidence.py -v", selected)
        self.assertIn("run: python " + evidence.SELECTOR, selected)
        self.assertIn("ocr-nov4-retained-text-" + "$" + "{{ github.run_id }}", selected)
        self.assertNotIn("secrets.", selected)
        self.assertNotIn("HF_TOKEN", selected)
        self.assertNotIn("KAGGLE", selected)
        self.assertNotIn("pip install", selected)
        self.assertIn("cancel-in-progress: false", workflow)

    def test_native_attribution_job_remains_separately_closed(self):
        workflow = (ROOT / ".github/workflows/heavy-cpu.yml").read_text()
        ordinary = workflow.split("  ocr_evidence:\n", 1)[0]
        retained = workflow.split("  retained-block21:\n", 1)[1].split("  lighton4b_reference_compare:\n", 1)[0]
        selectors = ("lighton4b_decoder_attribution.py", "lighton4b_decoder_upstream.py", "lighton4b_dense_trace.py")
        expected = "    if: " + " || ".join("inputs.script == 'tools/ci-heavy/" + name + "'" for name in selectors)
        self.assertIn(expected, retained)
        for name in selectors:
            self.assertIn("inputs.script != 'tools/ci-heavy/" + name + "'", ordinary)
        self.assertIn("      contents: read\n      actions: read\n", retained)
        self.assertIn("python -m pip install -r tools/lighton4b-gdn-requirements.txt", retained)
        self.assertIn("run-id: 41979168300", retained)
        self.assertIn("run-id: 41981848201", retained)
        for forbidden in ("secrets.", "HF_TOKEN", "KAGGLE", "inputs.pip", "inputs.args"):
            self.assertNotIn(forbidden, retained)

    def test_lighton4b_reference_comparison_is_separately_closed(self):
        workflow = (ROOT / ".github/workflows/heavy-cpu.yml").read_text()
        ordinary = workflow.split("  ocr_evidence:\n", 1)[0]
        selected = workflow.split("  lighton4b_reference_compare:\n", 1)[1]
        names = ("lighton4b_prefill_compare.py", "lighton4b_smoke_compare.py", "lighton4b_full_compare.py")
        self.assertIn("    if: " + " || ".join("inputs.script == 'tools/ci-heavy/" + name + "'" for name in names), selected)
        for name in names:
            self.assertIn("inputs.script != 'tools/ci-heavy/" + name + "'", ordinary)
        self.assertIn("      contents: read\n      actions: read\n", selected)
        self.assertIn("persist-credentials: false", selected)
        self.assertIn('test -z "$REF_COMPARE_PIP"', selected)
        self.assertIn('test "$REF_COMPARE_RUNNER" = ubuntu-latest', selected)
        self.assertIn('test "$REF_COMPARE_MODE" = portable', selected)
        for forbidden in ("HF_TOKEN", "contents: write", "actions: write", "eval "):
            self.assertNotIn(forbidden, selected)

    def test_api_authentication_is_not_forwarded_on_redirects(self):
        request = evidence.api_request("/actions/runs/41810161887", "dummy")
        self.assertEqual(request.host, "api.github.com")
        self.assertEqual(request.get_header("Authorization"), "Bearer dummy")
        self.assertIsNone(evidence.NoRedirect().redirect_request(
            request, None, 302, "Found", {}, "https://storage.example.invalid/archive"))
        with self.assertRaises(ValueError):
            evidence.api_request("/contents/file", "dummy")

    def test_non_https_or_credential_bearing_archive_redirects_rejected(self):
        for location in ("http://storage.example.invalid/a", "file:///tmp/a",
                         "https://user:password@storage.example.invalid/a",
                         "https://storage.example.invalid:444/a"):
            error = urllib.error.HTTPError("https://api.github.com/", 302, "Found",
                                           {"Location": location}, None)
            opener = mock.Mock()
            opener.open.side_effect = error
            with self.subTest(location=location), mock.patch.object(
                    evidence.urllib.request, "build_opener", return_value=opener):
                with self.assertRaisesRegex(ValueError, "unsafe archive redirect"):
                    evidence.archive_location(evidence.TARGETS[0], "dummy")


class ArchiveControls(unittest.TestCase):
    def test_only_small_text_members_are_read_and_hashed(self):
        payload = b'{"failure":"original mismatch"}\n'
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.zip"
            source.write_bytes(archive([
                ("nested/report.json", payload), ("original.log", b"failure\n"),
                ("states.gguf", b"not a model"), ("arrays/raw.npy", b"not an array"),
                ("image.png", b"not an image"), ("untrusted.py", b"raise SystemExit(42)"),
            ]).getvalue())
            original_open = zipfile.ZipFile.open
            opened = []

            def traced(instance, info, *args, **kwargs):
                opened.append(info.filename)
                return original_open(instance, info, *args, **kwargs)

            with mock.patch.object(zipfile.ZipFile, "open", autospec=True, side_effect=traced):
                directory, retained = evidence.extract_text(source, root / "text")
            self.assertEqual(opened, ["nested/report.json", "original.log"])
            self.assertEqual(len(directory), 6)
            self.assertEqual((root / "text/nested/report.json").read_bytes(), payload)
            self.assertEqual(retained[0]["sha256"], hashlib.sha256(payload).hexdigest())
            self.assertFalse((root / "text/states.gguf").exists())
            self.assertFalse((root / "text/untrusted.py").exists())

    def test_traversal_absolute_backslash_and_noncanonical_names_rejected(self):
        names = ("../escape.json", "/absolute.json", "a/../../escape.json",
                 "windows\\escape.json", "C:/escape.json", "a//b.json", "./a.json",
                 "a/\nlog.txt")
        for name in names:
            with self.subTest(name=name), zipfile.ZipFile(archive([(name, b"x")])) as source:
                with self.assertRaisesRegex(ValueError, "unsafe archive"):
                    evidence.safe_members(source)

    def test_duplicate_paths_rejected_before_destination_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.zip"
            source.write_bytes(archive([("report.json", b"{}"), ("report.json", b"{}")]).getvalue())
            with self.assertRaisesRegex(ValueError, "duplicate"):
                evidence.extract_text(source, root / "text")
            self.assertFalse((root / "text").exists())

    def test_nul_truncated_archive_name_is_rejected(self):
        info = zipfile.ZipInfo("report.json\x00hidden")
        source = mock.Mock()
        source.infolist.return_value = [info]
        with self.assertRaisesRegex(ValueError, "unsafe archive name"):
            evidence.safe_members(source)

    def test_symlinks_and_special_files_rejected(self):
        for mode in (stat.S_IFLNK, stat.S_IFIFO, stat.S_IFCHR):
            info = zipfile.ZipInfo("link.txt")
            info.create_system = 3
            info.external_attr = (mode | 0o777) << 16
            with self.subTest(mode=mode), zipfile.ZipFile(archive([(info, b"../target")])) as source:
                with self.assertRaisesRegex(ValueError, "special file"):
                    evidence.safe_members(source)

    def test_file_directory_collision_rejected(self):
        with zipfile.ZipFile(archive([("a", b"x"), ("a/report.json", b"{}")])) as source:
            with self.assertRaisesRegex(ValueError, "collision"):
                evidence.safe_members(source)

    def test_compression_bomb_rejected_without_expansion(self):
        with zipfile.ZipFile(archive([("bomb.json", b"x" * 100000)],
                                    zipfile.ZIP_DEFLATED)) as source:
            with self.assertRaisesRegex(ValueError, "compression ratio"):
                evidence.safe_members(source)

    def test_member_count_and_expanded_byte_caps(self):
        with zipfile.ZipFile(archive([("a.json", b"aaaa"), ("b.json", b"bbbb")])) as source:
            with mock.patch.object(evidence, "MAX_MEMBERS", 1):
                with self.assertRaisesRegex(ValueError, "member count"):
                    evidence.safe_members(source)
            with mock.patch.object(evidence, "MAX_MEMBER_BYTES", 3):
                with self.assertRaisesRegex(ValueError, "text member"):
                    evidence.safe_members(source)
            with mock.patch.object(evidence, "MAX_SELECTED_BYTES", 7):
                with self.assertRaisesRegex(ValueError, "selected expanded"):
                    evidence.safe_members(source)
            with mock.patch.object(evidence, "MAX_ARCHIVE_EXPANDED_BYTES", 7):
                with self.assertRaisesRegex(ValueError, "archive expanded"):
                    evidence.safe_members(source)

    def test_binary_disguised_as_text_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.zip"
            source.write_bytes(archive([("report.json", b"\xff\xfe")]).getvalue())
            with self.assertRaises(UnicodeDecodeError):
                evidence.extract_text(source, root / "text")
            self.assertFalse((root / "text/report.json").exists())

    def test_crc_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.zip"
            data = archive([("report.json", b"unique-original")]).getvalue()
            source.write_bytes(data.replace(b"unique-original", b"unique-mutated!"))
            with self.assertRaises(zipfile.BadZipFile):
                evidence.extract_text(source, root / "text")
            self.assertFalse((root / "text/report.json").exists())

    def test_signed_archive_size_and_sha_checked(self):
        data = b"small immutable ZIP stand-in"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "archive.zip"
            result = evidence.copy_verified(io.BytesIO(data), path, len(data),
                                            evidence.sha256(data), time.monotonic() + 10)
            self.assertEqual(result, {"bytes": len(data), "sha256": evidence.sha256(data)})
            for altered, size, digest in ((data[:-1], len(data), evidence.sha256(data)),
                                           (data + b"x", len(data), evidence.sha256(data)),
                                           (data, len(data), "0" * 64)):
                path.unlink()
                with self.assertRaises(ValueError):
                    evidence.copy_verified(io.BytesIO(altered), path, size, digest,
                                           time.monotonic() + 10)
                self.assertFalse(path.exists())
                path.write_bytes(data)

    def test_existing_download_destination_is_never_deleted(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "existing.zip"
            path.write_bytes(b"existing")
            with self.assertRaises(FileExistsError):
                evidence.copy_verified(io.BytesIO(b"new"), path, 3, evidence.sha256(b"new"),
                                       time.monotonic() + 10)
            self.assertEqual(path.read_bytes(), b"existing")

    def test_expired_transfer_deadline_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "archive.zip"
            with self.assertRaisesRegex(ValueError, "deadline"):
                evidence.copy_verified(io.BytesIO(b"x"), path, 1, evidence.sha256(b"x"),
                                       time.monotonic() - 1)
            self.assertFalse(path.exists())


class OriginalDiagnosticControls(unittest.TestCase):
    def original(self):
        reference = {"generated_ids": [1, 2], "cpu_controls": {"threads": 4},
                     "decoded": "frozen", "mean_confidence": 0.9, "generation_seconds": 1}
        prior = {"cases": [{"id": "inventory-table", "reference": reference}]}
        current = {"case_id": "inventory-table", "source_commit": evidence.SOURCE,
                   "reference": copy.deepcopy(reference),
                   "error": {"type": "ValueError", "message": "retained original failure"}}
        return prior, current

    def test_only_existing_timing_exclusion_is_ignored(self):
        prior, current = self.original()
        current["reference"]["generation_seconds"] = 42
        result = evidence.original_diagnostic(current, prior)
        self.assertTrue(result["guard_result_exact"])
        self.assertEqual(result["differences"], [])
        self.assertEqual(result["ignored_fields"], ["generation_seconds"])
        self.assertFalse(result["historical_verdicts_changed"])
        self.assertFalse(result["model_qualified"])

    def test_all_changed_original_fields_stay_negative_with_exact_values(self):
        prior, current = self.original()
        current["reference"]["cpu_controls"]["threads"] = 2
        current["reference"]["generated_ids"][1] = 3
        current["reference"]["mean_confidence"] += 1e-12
        result = evidence.original_diagnostic(current, prior)
        self.assertFalse(result["guard_result_exact"])
        self.assertEqual(result["difference_count"], 3)
        self.assertEqual([row["path"] for row in result["differences"]],
                         ["/cpu_controls/threads", "/generated_ids/1", "/mean_confidence"])
        self.assertEqual(result["differences"][0]["expected"], 4)
        self.assertEqual(result["differences"][0]["observed"], 2)
        self.assertEqual(result["observed_failure"],
                         {"type": "ValueError", "message": "retained original failure"})
        self.assertEqual(prior["cases"][0]["reference"]["generated_ids"], [1, 2])

    def test_missing_and_added_fields_and_ids_are_diagnosed(self):
        rows = evidence.exact_differences({"ids": [1, 2], "old": True}, {"ids": [1], "new": None})
        self.assertEqual({row["path"] for row in rows}, {"/ids/length", "/ids/1", "/old", "/new"})
        with mock.patch.object(evidence, "MAX_DIFFS", 1), self.assertRaises(ValueError):
            evidence.exact_differences([1, 2], [3, 4])

    def test_wrong_retained_report_identity_rejected(self):
        prior, current = self.original()
        current["source_commit"] = "0" * 40
        with self.assertRaisesRegex(ValueError, "identity"):
            evidence.original_diagnostic(current, prior)

    def test_duplicate_or_nonfinite_json_is_rejected(self):
        for data in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
            with self.subTest(data=data), self.assertRaises(ValueError):
                evidence.read_json(data)


if __name__ == "__main__":
    unittest.main()
