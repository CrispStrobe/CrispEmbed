"""Cache refresh diagnostics must never expose private service identifiers."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools/kaggle" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


kpush = load("refresh_kpush", "kpush.py")
with patch.dict(sys.modules, {"kpush": kpush}):
    refresh = load("cache_refresh", "ccache_refresh.py")


class RefreshPrivacy(unittest.TestCase):
    marker = "private-owner-fixture/private-execution-fixture"

    def captured(self, function):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            result = function()
        output = out.getvalue() + err.getvalue()
        self.assertNotIn(self.marker, output)
        self.assertNotIn("private-owner-fixture", output)
        self.assertNotIn("private-execution-fixture", output)
        self.assertNotIn("private-token-fixture", output)
        return result, output

    def response(self, rc=0, stdout=None, stderr=None):
        return subprocess.CompletedProcess([], rc, stdout or self.marker, (stderr or self.marker) + " private-token-fixture")

    def test_seed_decision_and_summary_keys_and_values_stay_private(self):
        def fake(*args):
            out = Path(args[args.index("-p") + 1])
            (out / "ccache-export.json").write_text(json.dumps({"export": False, "reasons": [self.marker], "hits": self.marker}))
            (out / "summary.json").write_text(json.dumps({"builds": {self.marker: {"rc": self.marker}}}))
            return self.response()
        with patch.object(refresh, "kaggle", side_effect=fake) as service:
            result, output = self.captured(lambda: refresh.refresh(self.marker, "private-owner-fixture", "private-execution-fixture"))
        self.assertEqual(result, 0)
        service.assert_called_once()
        self.assertIn("seed is current", output)

    def test_missing_artifact_and_service_failure_never_echo_raw_errors(self):
        for rc, stdout, expected in ((0, self.marker, 1), (11, self.marker, 11),
                                     (0, "error: " + self.marker, 1)):
            with self.subTest(rc=rc, expected=expected), patch.object(refresh, "kaggle", return_value=self.response(rc, stdout)):
                result, _ = self.captured(lambda: refresh.refresh(self.marker, "private-owner-fixture", "private-execution-fixture"))
                self.assertEqual(result, expected)

    def test_dataset_update_preserves_hard_and_soft_failures_and_hides_identifiers(self):
        for rc in (0, 19):
            def fake(*args):
                if args[:2] == ("kernels", "output"):
                    out = Path(args[args.index("-p") + 1])
                    (out / "ccache.tar").write_bytes(b"tiny fixture")
                    (out / "ccache-export.json").write_text(json.dumps({"export": True, "reasons": [self.marker]}))
                    return self.response()
                if args[:2] == ("datasets", "status"):
                    return self.response(stdout=self.marker + "/404-not-found: ready")
                stage = Path(args[args.index("-p") + 1])
                self.assertEqual(json.loads((stage / "dataset-metadata.json").read_text())["id"], self.marker)
                self.assertEqual(args[:2], ("datasets", "version"))
                message = args[args.index("-m") + 1]
                self.assertEqual(message, "Refresh compiler cache seed")
                self.assertNotIn(self.marker, message)
                self.assertNotIn("private-owner-fixture", message)
                self.assertNotIn("private-execution-fixture", message)
                return self.response(rc, "error: " + self.marker)
            with self.subTest(rc=rc), patch.object(refresh, "kaggle", side_effect=fake), patch.object(refresh, "MIN_TAR_BYTES", 1):
                result, _ = self.captured(lambda: refresh.refresh(self.marker, "private-owner-fixture", "private-execution-fixture"))
                self.assertEqual(result, rc or 1)

    def test_status_completion_timeout_and_query_failure_hide_identifiers(self):
        with patch.object(refresh, "kaggle", return_value=self.response(stdout='status "complete" ' + self.marker)):
            result, output = self.captured(lambda: refresh.wait_for_kernel(self.marker))
        self.assertEqual(result, "complete")
        self.assertIn("complete", output)
        with patch.object(refresh, "kaggle", return_value=self.response()), patch.object(refresh.time, "time", side_effect=[0, 2]):
            def timed_out():
                with self.assertRaises(SystemExit) as error:
                    refresh.wait_for_kernel(self.marker, timeout=1)
                self.assertNotIn(self.marker, str(error.exception))
                return str(error.exception)
            _, output = self.captured(timed_out)
        with patch.object(refresh, "kaggle", return_value=self.response(7)):
            def failed():
                with self.assertRaises(SystemExit) as error:
                    refresh.wait_for_kernel(self.marker)
                return error.exception.code
            result, _ = self.captured(failed)
        self.assertEqual(result, 7)

    def test_unrecognized_status_and_invalid_metadata_do_not_echo_private_fields(self):
        responses = [self.response(stdout='status "private_execution_fixture" ' + self.marker),
                     self.response(stdout='status "complete" ' + self.marker)]
        with patch.object(refresh, "kaggle", side_effect=responses), \
                patch.object(refresh.time, "time", side_effect=[0, 1, 2]), patch.object(refresh.time, "sleep"):
            result, output = self.captured(lambda: refresh.wait_for_kernel(self.marker, timeout=10))
        self.assertEqual(result, "complete")
        self.assertIn("unrecognized", output)
        self.assertNotIn("private_execution_fixture", output)

        def invalid_metadata(*args):
            out = Path(args[args.index("-p") + 1])
            (out / "ccache-export.json").write_text(self.marker + " invalid JSON")
            return self.response()
        with patch.object(refresh, "kaggle", side_effect=invalid_metadata), \
                patch.dict(refresh.os.environ, {"KAGGLE_ACCOUNT": "private-owner-fixture"}):
            result, output = self.captured(lambda: refresh.main(["--from-kernel", self.marker]))
        self.assertEqual(result, 1)
        self.assertIn("invalid cache metadata", output)

    def test_only_confirmed_missing_dataset_can_trigger_creation(self):
        cases = (
            (False, 1, "authentication failed"),
            (False, 0, "authentication failed"),
            (False, 0, "authentication failed\n404 Not Found"),
            (False, 1, "network error\n404 Not Found"),
            (False, 1, self.marker + "/404-not-found was not found in a local cache"),
            (False, 0, "error: " + self.marker + " was not found in a local cache"),
            (True, 1, "404 Not Found"),
            (True, 0, "404 Not Found"),
            (True, 0, json.dumps({"status": 404, "message": "Not Found"})),
            (True, 0, json.dumps({"status": 404, "message": "Not Found"}, indent=2)),
        )
        for missing, status_rc, status_message in cases:
            calls = []
            def fake(*args):
                calls.append(args[:2])
                if args[:2] == ("kernels", "output"):
                    out = Path(args[args.index("-p") + 1])
                    (out / "ccache.tar").write_bytes(b"tiny fixture")
                    return self.response()
                if args[:2] == ("datasets", "status"):
                    return self.response(status_rc, status_message)
                self.assertEqual(args[:2], ("datasets", "create"))
                return self.response()
            with self.subTest(missing=missing, status_rc=status_rc), patch.object(refresh, "kaggle", side_effect=fake), patch.object(refresh, "MIN_TAR_BYTES", 1):
                result, _ = self.captured(lambda: refresh.refresh(self.marker, "private-owner-fixture", "private-execution-fixture"))
            self.assertEqual(result, 0 if missing else 1)
            self.assertEqual(len(calls), 3 if missing else 2)


if __name__ == "__main__":
    unittest.main()
