#!/usr/bin/env python3
"""Follow-up producer provenance controls; archive security is shared and pinned."""
import copy
import hashlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/ci-heavy"))
import extract_ocr_followup_evidence as evidence


class Provenance(unittest.TestCase):
    def fixture(self):
        t = evidence.TARGETS[0]
        run = {"id": t["run_id"], "head_sha": t["source"], "head_branch": "fix/ocr-qualification-nov4",
               "event": "workflow_dispatch", "path": ".github/workflows/heavy-cpu.yml", "status": "completed",
               "conclusion": t["conclusion"], "run_attempt": 1,
               "repository": {"full_name": evidence.archive.REPOSITORY},
               "head_repository": {"full_name": evidence.archive.REPOSITORY}}
        url = evidence.archive.API + "/actions/artifacts/" + str(t["artifact_id"])
        a = {"id": t["artifact_id"], "name": "heavy-" + str(t["run_id"]), "size_in_bytes": t["bytes"],
             "digest": "sha256:" + t["sha256"], "expired": False, "url": url, "archive_download_url": url + "/zip",
             "workflow_run": {"id": t["run_id"], "head_sha": t["source"], "head_branch": "fix/ocr-qualification-nov4",
                              "repository_id": 1210669686, "head_repository_id": 1210669686}}
        return t, run, a

    def test_exact_producer_and_archive_identity(self):
        evidence.validate_metadata(*self.fixture())

    def test_changed_source_attempt_status_and_repository_rejected(self):
        t, run, a = self.fixture()
        for key, value in (("head_sha", "0" * 40), ("run_attempt", 2), ("status", "in_progress"),
                           ("event", "pull_request"), ("head_branch", "main"), ("conclusion", "failure"),
                           ("repository", {"full_name": "other/repo"})):
            with self.subTest(key=key), self.assertRaises(ValueError):
                evidence.validate_metadata(t, dict(run, **{key: value}), a)

    def test_changed_archive_and_workflow_owner_rejected(self):
        t, run, a = self.fixture()
        for key, value in (("digest", "sha256:" + "0" * 64), ("size_in_bytes", 1), ("expired", True),
                           ("id", 1), ("archive_download_url", "https://example.invalid/zip")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                evidence.validate_metadata(t, run, dict(a, **{key: value}))
        changed = copy.deepcopy(a); changed["workflow_run"]["repository_id"] = 1
        with self.assertRaises(ValueError):
            evidence.validate_metadata(t, run, changed)
        with self.assertRaises(ValueError):
            evidence.validate_metadata(dict(t, source="0" * 40), run, a)

    def test_original_security_helper_is_pinned(self):
        self.assertEqual(hashlib.sha256(Path(evidence.archive.__file__).read_bytes()).hexdigest(), evidence.HELPER_SHA256)


if __name__ == "__main__":
    unittest.main()
