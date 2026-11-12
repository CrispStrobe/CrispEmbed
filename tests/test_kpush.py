"""Check temporary submission metadata without contacting the remote service."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("kpush", ROOT / "tools/kaggle/kpush.py")
kpush = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(kpush)


class OvisBenchSubmission(unittest.TestCase):
    def test_private_service_output_preserves_exit_and_soft_failure_status(self):
        marker = "private-owner-fixture/private-execution-fixture"
        for rc, stdout, stderr, expected, category in (
            (0, marker, marker, 0, "accepted"),
            (13, marker, "authentication failed " + marker, 13, "authentication or permission"),
            (0, "Kernel push error: Maximum batch GPU session count " + marker, marker, 1, "resource limit"),
            (0, marker, "error: " + marker, 1, "service error"),
        ):
            with self.subTest(rc=rc, expected=expected):
                out, err = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    result = kpush.service_result(subprocess.CompletedProcess([], rc, stdout, stderr + " private-token-fixture"), "submission")
                self.assertEqual(result, expected)
                self.assertIn(category, out.getvalue() + err.getvalue())
                self.assertNotIn(marker, out.getvalue() + err.getvalue())
                self.assertNotIn("private-token-fixture", out.getvalue() + err.getvalue())

    def test_refresh_output_is_captured_and_a_refused_push_never_refreshes(self):
        marker = "private-refresh-fixture"
        for refused in (False, True):
            with self.subTest(refused=refused), tempfile.TemporaryDirectory() as temp:
                source = Path(temp)
                (source / "kernel.py").write_text("pass\n")
                (source / "kernel-metadata.json").write_text(json.dumps({
                    "id": "${KAGGLE_ACCOUNT}/" + marker, "code_file": "kernel.py",
                }))
                responses = [subprocess.CompletedProcess([], 0,
                    "Kernel push error: Maximum batch GPU session count " + marker if refused else marker,
                    marker)]
                if not refused:
                    responses.append(subprocess.CompletedProcess([], 17, marker, marker))
                out, err = io.StringIO(), io.StringIO()
                with patch.dict(kpush.os.environ, {"KAGGLE_ACCOUNT": marker, "KAGGLE_TOKEN": marker}), \
                        patch.object(kpush.subprocess, "run", side_effect=responses) as runner, \
                        patch.object(kpush.time, "sleep") as sleep, \
                        contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    self.assertEqual(kpush.main(["kpush", str(source), "--refresh-ccache"]), 1 if refused else 17)
                self.assertEqual(runner.call_count, 1 if refused else 2)
                if refused:
                    sleep.assert_not_called()
                else:
                    self.assertTrue(runner.call_args.kwargs["capture_output"])
                self.assertNotIn(marker, out.getvalue() + err.getvalue())

    def test_clean_clone_example_and_local_override_are_read_without_mutation(self):
        for override in (False, True):
            with self.subTest(override=override), tempfile.TemporaryDirectory() as temp:
                source = Path(temp)
                (source / "kernel.py").write_text('SOURCE = "__SHA__"\n')
                example = source / "kernel-metadata.example.json"
                example.write_text(json.dumps({"id": "${KAGGLE_ACCOUNT}/example",
                                               "code_file": "kernel.py", "enable_gpu": "false"}))
                if override:
                    (source / "kernel-metadata.json").write_text(json.dumps({
                        "id": "${KAGGLE_ACCOUNT}/local-fixture", "code_file": "kernel.py",
                        "enable_gpu": "true", "docker_image_pinning_type": "original",
                    }))
                before = {p.name: p.read_bytes() for p in source.iterdir()}
                staged = []

                def push(cmd, **kwargs):
                    destination = Path(cmd[cmd.index("-p") + 1])
                    staged.append(destination)
                    metadata = json.loads((destination / "kernel-metadata.json").read_text())
                    self.assertEqual(metadata["id"], "audit-placeholder/" + ("local-fixture" if override else "example"))
                    self.assertEqual(metadata["enable_gpu"], "true" if override else "false")
                    self.assertEqual(metadata["docker_image_pinning_type"], "original" if override else "latest")
                    self.assertIn("a" * 40, (destination / "kernel.py").read_text())
                    return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

                with patch.dict(kpush.os.environ, {"KAGGLE_ACCOUNT": "audit-placeholder", "KAGGLE_TOKEN": "noncredential-placeholder"}), \
                        patch.object(kpush.subprocess, "run", side_effect=push) as mocked:
                    self.assertEqual(kpush.main(["kpush", str(source), "--sha", "a" * 40]), 0)
                mocked.assert_called_once()
                self.assertEqual({p.name: p.read_bytes() for p in source.iterdir()}, before)
                self.assertFalse(staged[0].exists())

    def test_invalid_local_override_never_falls_back_to_example(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)
            (source / "kernel-metadata.example.json").write_text("{}")
            (source / "kernel-metadata.json").mkdir()
            with self.assertRaisesRegex(SystemExit, "local metadata must be a regular file"):
                kpush.metadata_path(source)

    def test_existing_kernel_identity_and_seed_survive_rendering(self):
        rendered = {}
        sha = "a" * 40

        def fake_push(cmd, **kwargs):
            destination = Path(cmd[cmd.index("-p") + 1])
            rendered.update(json.loads((destination / "kernel-metadata.json").read_text()))
            code = (destination / rendered["code_file"]).read_text()
            self.assertNotIn("__SHA__", code)
            self.assertIn(sha, code)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)
            (source / "kernel.py").write_text('SOURCE = "__SHA__"\n')
            (source / "kernel-metadata.json").write_text(json.dumps({
                "id": "${KAGGLE_ACCOUNT}/crispembed-ovisocr2-gpu-bench",
                "title": "crispembed-ovisocr2-gpu-bench", "code_file": "kernel.py",
                "dataset_sources": ["${KAGGLE_ACCOUNT}/crispembed-ccache"],
                "is_private": "true", "enable_gpu": "true", "enable_internet": "true",
            }))
            with patch.dict(kpush.os.environ, {"KAGGLE_ACCOUNT": "audit-placeholder",
                                          "KAGGLE_TOKEN": "noncredential-placeholder"}), \
             patch.object(kpush.subprocess, "run", side_effect=fake_push) as push, \
             contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
                result = kpush.main(["kpush", str(source), "--sha", sha])
        self.assertEqual(0, result)
        push.assert_called_once()
        self.assertEqual("audit-placeholder/crispembed-ovisocr2-gpu-bench", rendered["id"])
        self.assertEqual("crispembed-ovisocr2-gpu-bench", rendered["title"])
        self.assertEqual("kernel.py", rendered["code_file"])
        self.assertEqual(["audit-placeholder/crispembed-ccache"], rendered["dataset_sources"])
        self.assertEqual("latest", rendered["docker_image_pinning_type"])
        for key in ("is_private", "enable_gpu", "enable_internet"):
            self.assertEqual("true", rendered[key])

    def test_kernel_configs_stay_ignored_and_untracked(self):
        for name in ("ovisocr2-gpu-bench", "crispsorter-pin-01719-consumer"):
            path = f"tools/kaggle/{name}/kernel-metadata.json"
            subprocess.run(["git", "check-ignore", "--no-index", "-q", path], cwd=ROOT, check=True)
            tracked = subprocess.check_output(["git", "ls-files", "--", path], cwd=ROOT, text=True)
            self.assertEqual("", tracked)


if __name__ == "__main__":
    # The existing lightweight CI entrypoint also covers refresh diagnostics.
    # Pytest discovers the separate module normally, without collecting it twice.
    from test_ccache_refresh import RefreshPrivacy
    unittest.main()
