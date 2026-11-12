"""Model-free regressions for pinned-head CI polling; no GitHub calls or sleeps."""
import contextlib
import importlib.util
import io
import itertools
from pathlib import Path
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "tools/ci/wait_required_checks.py"
SPEC = importlib.util.spec_from_file_location("wait_required_checks", SCRIPT)
checks = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checks)
SHA = "a" * 40


def pr(head=SHA, state="open"):
    return {"head": head, "base": "main", "state": state, "merged": False}


class WorkflowPrecedence(unittest.TestCase):
    def states(self, newer_state):
        def api(path, paginate=False):
            if "/check-runs?" in path:
                return [{"check_runs": []}]
            if "/actions/runs?" in path:
                return [{"workflow_runs": [
                    {"id": 100, "name": "Build", "event": "pull_request",
                     "run_attempt": 2, "created_at": "2026-11-12T01:00:00Z",
                     "status": "completed", "conclusion": "success"},
                    {"id": 200, "name": "Build", "event": "pull_request",
                     "run_attempt": 1, "created_at": "2026-11-12T02:00:00Z",
                     "status": "queued" if newer_state == "queued" else "completed",
                     "conclusion": None if newer_state == "queued" else newer_state},
                ]}]
            return {"statuses": []}
        with patch.object(checks, "gh_api", api):
            return checks.checks_for("owner/repo", SHA)

    def test_older_rerun_cannot_hide_newer_queued_workflow(self):
        self.assertIn("queued", self.states("queued").values())

    def test_older_rerun_cannot_hide_newer_failed_workflow(self):
        self.assertIn("failure", self.states("failure").values())


class RequiredContextErrors(unittest.TestCase):
    def error_result(self, code, message):
        detail = f"gh: {message}" + (f" (HTTP {code})" if code else "")
        return checks.subprocess.CompletedProcess([], 1, stdout="", stderr=detail)

    def test_unprotected_or_invisible_branch_keeps_documented_fallback(self):
        for code, message in ((403, "Resource not accessible by integration"),
                              (404, "Branch not protected")):
            with self.subTest(code=code), \
                 patch.object(checks.subprocess, "run", return_value=self.error_result(code, message)):
                self.assertIsNone(checks.required_contexts("owner/repo", "main"))

    def test_transient_and_rate_limit_errors_cannot_report_green(self):
        for code, message in (
            (429, "Too Many Requests"), (500, "Internal Server Error"),
            (503, "Service Unavailable"), (None, "error connecting to api.github.com"),
            (403, "API rate limit exceeded"),
            (403, "You have exceeded a secondary rate limit"),
            (403, "You have triggered an abuse detection mechanism"),
        ):
            with self.subTest(code=code, message=message), \
                 patch.object(checks.subprocess, "run", return_value=self.error_result(code, message)), \
                 patch.object(checks, "pr_info", return_value=pr()), \
                 patch.object(checks, "checks_for", return_value={"scan": "success"}), \
                 patch.object(checks.time, "sleep"), \
                 patch.object(checks.time, "time", side_effect=itertools.count()), \
                 contextlib.redirect_stdout(io.StringIO()) as output, \
                 contextlib.redirect_stderr(io.StringIO()):
                result = checks.main(["--pr", "219", "--repo", "owner/repo",
                                      "--sha", SHA, "--settle", "0"])
                self.assertEqual(4, result)
                self.assertNotIn("GREEN:", output.getvalue())

    def test_other_endpoints_keep_runtime_error_contract(self):
        with patch.object(checks.subprocess, "run",
                          return_value=self.error_result(403, "Resource not accessible by integration")):
            with self.assertRaises(RuntimeError):
                checks.gh_api("repos/owner/repo/commits/" + SHA + "/check-runs")

    def test_successful_api_response_is_unchanged(self):
        result = checks.subprocess.CompletedProcess([], 0, stdout='{"contexts": ["build"]}', stderr="")
        with patch.object(checks.subprocess, "run", return_value=result):
            self.assertEqual(["build"], checks.required_contexts("owner/repo", "main"))


class FinalHeadAndStalledChecks(unittest.TestCase):
    def run_poll(self, states, info, *args):
        with patch.object(checks, "pr_info", info), \
             patch.object(checks, "checks_for", states), \
             patch.object(checks, "required_contexts", return_value=[]), \
             patch.object(checks.time, "sleep"), \
             patch.object(checks.time, "time", side_effect=itertools.count()), \
             contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            return checks.main(["--pr", "219", "--repo", "owner/repo",
                                "--sha", SHA, "--settle", "0", *args])

    def final_change(self, final_info):
        current = {"info": pr(), "polls": 0}
        def states(*args):
            current["polls"] += 1
            if current["polls"] == 2:
                current["info"] = final_info
            return {"build": "success"}
        def info(*args):
            if isinstance(current["info"], Exception):
                raise current["info"]
            return current["info"]
        return self.run_poll(states, info)

    def test_head_move_during_final_check_fetch_is_rejected(self):
        self.assertEqual(3, self.final_change(pr("b" * 40)))

    def test_closed_pr_during_final_check_fetch_is_rejected(self):
        self.assertEqual(1, self.final_change(pr(state="closed")))

    def test_final_head_api_error_fails_closed(self):
        self.assertEqual(4, self.final_change(RuntimeError("API unavailable")))

    def test_stalled_exception_cannot_ignore_running_job(self):
        result = self.run_poll(lambda *args: {"build": "in_progress"},
                               lambda *args: pr(), "--allow-stalled", "build", "--timeout", "10")
        self.assertEqual(2, result)

    def test_stalled_exception_keeps_queued_and_pending_support(self):
        for state in ("queued", "pending"):
            with self.subTest(state=state):
                result = self.run_poll(lambda *args: {"scan": "success", "build": state},
                                       lambda *args: pr(), "--allow-stalled", "build")
                self.assertEqual(0, result)

    def test_stalled_exceptions_alone_cannot_report_green(self):
        for state in ("queued", "pending"):
            with self.subTest(state=state):
                result = self.run_poll(lambda *args: {"build": state},
                                       lambda *args: pr(), "--allow-stalled", "build",
                                       "--timeout", "10")
                self.assertEqual(2, result)


if __name__ == "__main__":
    unittest.main()
