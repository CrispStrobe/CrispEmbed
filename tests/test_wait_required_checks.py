"""Regression controls for workflow-run selection in the merge guard."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / 'tools/ci/wait_required_checks.py'
spec = importlib.util.spec_from_file_location('wait_required_checks', PATH)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


class WorkflowOrderingTest(unittest.TestCase):
    def checks(self, runs):
        def fake_api(path, paginate=False):
            if '/check-runs?' in path:
                return [{'check_runs': []}]
            if '/actions/runs?' in path:
                return [{'workflow_runs': runs}]
            if '/status?' in path:
                return {'statuses': []}
            raise AssertionError(path)
        with patch.object(guard, 'gh_api', side_effect=fake_api):
            return guard.checks_for('owner/repo', 'f' * 40)['workflow Build [pull_request]']

    def test_newer_queued_run_is_not_hidden_by_older_successful_rerun(self):
        older = dict(name='Build', event='pull_request', id=1, created_at='2026-11-01T00:00:00Z',
                     run_attempt=2, status='completed', conclusion='success')
        newer = dict(name='Build', event='pull_request', id=2, created_at='2026-11-02T00:00:00Z',
                     run_attempt=1, status='queued', conclusion=None)
        self.assertEqual(self.checks([older, newer]), 'queued')
        newer.update(status='completed', conclusion='failure')
        self.assertEqual(self.checks([newer, older]), 'failure')

    def test_later_attempt_of_same_run_supersedes_earlier_attempt(self):
        run = dict(name='Build', event='pull_request', id=3, created_at='2026-11-02T00:00:00Z')
        old = dict(run, run_attempt=1, status='completed', conclusion='failure')
        new = dict(run, run_attempt=2, status='completed', conclusion='success')
        self.assertEqual(self.checks([new, old]), 'success')


if __name__ == '__main__':
    unittest.main()
