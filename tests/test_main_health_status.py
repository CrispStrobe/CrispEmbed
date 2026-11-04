#!/usr/bin/env python3
"""Offline regressions for truthful recorded-main CI status and its cheap workflow."""

import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit


ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


health = module("main_health_status", ROOT / "tools/main_health_status.py")
NOW = "2026-11-03T16:00:00Z"
CURRENT = "f" * 40
OLD = "a" * 40
OWNER = {"id": health.REPOSITORY_ID, "full_name": health.REPOSITORY}


def run(workflow, run_id=1, created="2026-11-03T12:00:00Z", conclusion="success", **changes):
    result = {"id": run_id, "run_attempt": 1, "workflow_id": workflow[0],
              "path": ".github/workflows/" + workflow[1], "event": "push", "head_branch": "main",
              "repository": dict(OWNER), "head_repository": dict(OWNER), "head_sha": CURRENT,
              "created_at": created, "run_started_at": created, "status": "completed", "conclusion": conclusion}
    result.update(changes)
    return result


class FakeAPI:
    def __init__(self):
        self.runs = {workflow[0]: [run(workflow)] for workflow in health.WATCHED}
        self.calls = []

    def __call__(self, path):
        self.calls.append(path)
        if path == health.PREFIX:
            return dict(OWNER)
        if path == health.PREFIX + "/branches/main":
            return {"name": "main", "commit": {"sha": CURRENT}}
        parsed = urlsplit(path)
        workflow_id = int(parsed.path.split("/")[-2])
        params = parse_qs(parsed.query)
        assert params["branch"] == ["main"]
        rows = copy.deepcopy(self.runs[workflow_id])
        created = params["created"][0]
        def bad_time(row):
            return not isinstance(row.get("created_at"), str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", row["created_at"])
        if created.startswith("<="):
            rows = [r for r in rows if bad_time(r) or r["created_at"] <= created[2:]]
        else:
            start, end = created.split("..")
            rows = [r for r in rows if bad_time(r) or start <= r["created_at"] <= end]
        if params.get("status") == ["success"]:
            rows = [r for r in rows if r.get("conclusion") == "success"]
        size, page = int(params["per_page"][0]), int(params["page"][0])
        return {"total_count": len(rows), "workflow_runs": rows[(page - 1) * size:page * size]}


class StatusTests(unittest.TestCase):
    def test_success_and_historical_success_are_distinct_without_demanding_every_head(self):
        api = FakeAPI()
        api.runs[health.WATCHED[1][0]][0]["head_sha"] = OLD
        result = health.report(api, NOW)
        self.assertEqual(result["state"], "RECORDED_SUCCESS")
        self.assertEqual(result["workflows"][0]["state"], "RECORDED_SUCCESS")
        self.assertEqual(result["workflows"][1]["state"], "HISTORICAL_SUCCESS")
        text = health.markdown(result)
        self.assertIn(CURRENT, text)
        self.assertIn(OLD, text)
        self.assertIn("not coverage of every change", text)
        self.assertNotIn("are green", text)

    def test_actual_five_cancelled_outcomes_never_report_success(self):
        api = FakeAPI()
        for rows in api.runs.values():
            rows[0].update(conclusion="cancelled", head_sha="7e5028e34972d1beb544d1cae87b381c6a995993")
        result = health.report(api, NOW)
        self.assertEqual(result["state"], "UNVERIFIED")
        self.assertTrue(all(row["state"] == "UNVERIFIED" for row in result["workflows"]))

    def test_unfinished_cancelled_and_empty_conclusions_do_not_fall_back_to_old_green(self):
        workflow = health.WATCHED[0]
        for status, conclusion in [(s, None) for s in ("queued", "pending", "waiting", "requested", "in_progress")]+[("completed", c) for c in (None, "cancelled", "skipped", "neutral", "stale")]:
            with self.subTest(status=status, conclusion=conclusion):
                api = FakeAPI()
                api.runs[workflow[0]] = [run(workflow, head_sha=OLD), run(workflow, 2, "2026-11-03T13:00:00Z", conclusion, status=status)]
                result = health.report(api, NOW)
                row = result["workflows"][0]
                self.assertEqual(result["state"], "UNVERIFIED")
                self.assertEqual(row["latest"]["id"], 2)
                self.assertEqual(row["last_success"]["sha"], OLD)

    def test_failed_latest_is_not_hidden_by_previous_success(self):
        workflow = health.WATCHED[0]
        for conclusion in health.FAILED:
            with self.subTest(conclusion=conclusion):
                api = FakeAPI()
                api.runs[workflow[0]].append(run(workflow, 2, "2026-11-03T13:00:00Z", conclusion))
                self.assertEqual(health.report(api, NOW)["state"], "FAILED")

    def test_known_failure_remains_failed_when_historical_lookup_errors(self):
        api = FakeAPI()
        api.runs[health.WATCHED[0][0]][0]["conclusion"] = "failure"
        def get(path):
            if "status=success" in path:
                raise health.Unverified("historical lookup unavailable")
            return api(path)
        result = health.report(get, NOW)
        self.assertEqual(result["state"], "FAILED")
        self.assertEqual(result["workflows"][0]["latest"]["conclusion"], "failure")
        self.assertIn("historical lookup unavailable", result["workflows"][0]["reason"])

    def test_newest_created_source_wins_even_when_old_run_is_rerun_later(self):
        workflow = health.WATCHED[0]
        api = FakeAPI()
        api.runs[workflow[0]] = [run(workflow, run_attempt=3, run_started_at="2026-11-03T15:00:00Z"), run(workflow, 2, "2026-11-03T13:00:00Z", "failure")]
        result = health.report(api, NOW)
        self.assertEqual(result["state"], "FAILED")
        self.assertEqual(result["workflows"][0]["latest"]["id"], 2)

    def test_run_id_breaks_equal_creation_time_ties(self):
        workflow = health.WATCHED[0]
        api = FakeAPI()
        api.runs[workflow[0]].append(run(workflow, 2, conclusion="cancelled"))
        self.assertEqual(health.report(api, NOW)["workflows"][0]["latest"]["id"], 2)

    def test_fork_pr_or_foreign_head_named_main_cannot_supply_health(self):
        workflow = health.WATCHED[0]
        for changes in ({"event": "pull_request"}, {"head_repository": {"id": 123, "full_name": "other/CrispEmbed"}}, {"head_branch": "feature"}):
            with self.subTest(changes=changes):
                api = FakeAPI()
                api.runs[workflow[0]] = [run(workflow, conclusion="failure"), run(workflow, 2, "2026-11-03T13:00:00Z", **changes)]
                self.assertEqual(health.report(api, NOW)["state"], "FAILED")

    def test_missing_workflow_fails(self):
        api = FakeAPI()
        api.runs[health.WATCHED[0][0]] = []
        self.assertEqual(health.report(api, NOW)["state"], "UNVERIFIED")

    def test_malformed_newest_record_cannot_fall_back_to_success(self):
        workflow = health.WATCHED[0]
        for changes in ({"head_branch": None}, {"head_sha": "bad"}, {"created_at": "2026-11-03Z"}, {"head_repository": None}, {"status": None}, {"conclusion": []}, {"run_attempt": 0}, {"workflow_id": 3}, {"path": "wrong.yml"}):
            with self.subTest(changes=changes):
                api = FakeAPI()
                api.runs[workflow[0]].append(run(workflow, 2, "2026-11-03T13:00:00Z", **changes))
                self.assertEqual(health.report(api, NOW)["state"], "UNVERIFIED")

    def test_full_pagination_finds_newer_run_beyond_first_page_without_order_assumption(self):
        workflow = health.WATCHED[0]
        api = FakeAPI()
        api.runs[workflow[0]] += [run(workflow, 2, "2026-11-03T13:00:00Z"), run(workflow, 3, "2026-11-03T14:00:00Z", "failure")]
        with patch.object(health, "PER_PAGE", 1):
            result = health.report(api, NOW)
        self.assertEqual(result["workflows"][0]["latest"]["id"], 3)
        self.assertEqual(result["state"], "FAILED")
        self.assertTrue(any("page=3" in call for call in api.calls))

    def test_pagination_limit_is_unverified(self):
        workflow = health.WATCHED[0]
        api = FakeAPI()
        api.runs[workflow[0]].append(run(workflow, 2, "2026-11-03T13:00:00Z"))
        with patch.object(health, "PER_PAGE", 1), patch.object(health, "MAX_PAGES", 1):
            self.assertEqual(health.report(api, NOW)["state"], "UNVERIFIED")

    def test_api_errors_and_malformed_pages_fail_instead_of_empty_green(self):
        for bad in (None, {}, {"total_count": "1", "workflow_runs": []}, {"total_count": 1, "workflow_runs": []}):
            with self.subTest(bad=bad):
                api = FakeAPI()
                get = lambda path: bad if "/runs?" in path else api(path)
                self.assertEqual(health.report(get, NOW)["state"], "UNVERIFIED")
        def fail(path):
            raise health.Unverified("API request failed")
        self.assertEqual(health.report(fail, NOW)["state"], "UNVERIFIED")

    def test_repository_is_pinned(self):
        api = FakeAPI()
        result = health.report(lambda p: {"id": 1, "full_name": "other/CrispEmbed"} if p == health.PREFIX else api(p), NOW)
        self.assertEqual(result["state"], "UNVERIFIED")
        with self.assertRaises(health.Unverified):
            health.GitHub("test-only")("repos/other/CrispEmbed/actions/runs")

    def test_client_reports_invalid_json_and_network_errors(self):
        for error in (json.JSONDecodeError("bad", "", 0), OSError("network unavailable")):
            with self.subTest(error=type(error).__name__), patch.object(health, "urlopen", side_effect=error):
                with self.assertRaises(health.Unverified):
                    health.GitHub("test-only")(health.PREFIX)

    def test_exit_code_and_summary_preserve_unverified_state(self):
        api = FakeAPI()
        api.runs[health.WATCHED[0][0]][0]["conclusion"] = "cancelled"
        original_report = health.report
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), patch.dict(os.environ, {"GITHUB_REPOSITORY": health.REPOSITORY}), patch.object(health, "report", side_effect=lambda api: original_report(api, NOW)):
            summary = Path(tmp) / "summary.md"
            self.assertEqual(health.main(["--summary", str(summary)], api=api), 1)
            self.assertIn("UNVERIFIED", summary.read_text())
            self.assertEqual(health.main(["--summary", str(summary)], api=FakeAPI()), 0)


class WorkflowTests(unittest.TestCase):
    def test_changes_run_only_offline_controls_and_live_jobs_remain_manual_or_scheduled(self):
        text = (ROOT / ".github/workflows/main-health.yml").read_text()
        paths = [".github/workflows/main-health.yml", "tools/main_health_status.py", "tests/test_main_health_status.py"]
        for event in ("push", "pull_request"):
            block = re.search(r"^  " + event + r":\n(.*?)(?=^  \S|^\S)", text, re.M | re.S).group(1)
            self.assertIn("branches: [main]", block)
            self.assertEqual(re.findall(r"^      - '([^']+)'", block, re.M), paths)
        for job in ("check", "licenses"):
            block = re.search(r"^  " + job + r":\n(.*?)(?=^  \S|\Z)", text, re.M | re.S).group(1)
            self.assertIn("github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'", block)
        self.assertIn("run: python tests/test_main_health_status.py -v", text)

    def test_reporter_paths_select_no_native_lanes(self):
        ci = module("ci_selection_for_main_health", ROOT / "tools/ci_selection.py")
        for event in ("push", "pull_request"):
            for path in ("tools/main_health_status.py", "tests/test_main_health_status.py", ".github/workflows/main-health.yml"):
                with self.subTest(event=event, path=path):
                    self.assertFalse(any(ci.selection([path], event_name=event)["lanes"].values()))


if __name__ == "__main__":
    unittest.main()
